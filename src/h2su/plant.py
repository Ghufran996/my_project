"""PEM plant engine (Brief 3 item 5; D32a-d). One-minute event-based state machine in numba.

States: 0 OFF (depressurised, inerted, cooling), 1 STANDBY (pressure and temperature kept), 2 START, 3 PRODUCTION, 4 SHUTDOWN.
START phases: 0 inerting + pre-start (no production), 1 purge (product vented), 2 pressurisation (into the inventory), 3 off-spec diversion
(until the N2 washout reaches the ISO limit), then PRODUCTION. Thermal: start-up current ramp 0 -> rated over t_su(T0) (D32a, L25 anchors),
stack temperature ramps T0 -> T_op over t_su; idle cool-down ODE with C, R per MW (D32a).
Fate balance (PL4): M_stack = M_del + M_purge + M_offspec + M_deoxo + M_dryer + M_blowdown + dM_inv  (dM_inv = inventory at the end - at the start).
Cycle window W_c (D32d) opens at stop entry (PRODUCTION -> SHUTDOWN/STANDBY; at t = 0 for the initial OFF state) and closes when START ends its off-spec
step. m_SU,c = sum over W_c of (purge + offspec + deoxo + dryer + blowdown) - sum over W_c of M_stack x s_SS(L).
All parameter values are passed in (built from the registry / result files in plantrun.py); nothing is fitted here.
"""
from __future__ import annotations

import numpy as np
from numba import njit

OFF, STANDBY, START, PRODUCTION, SHUTDOWN = 0, 1, 2, 3, 4

(P_RATED, P_MIN_F, T_DEB, SB_F, T_SB_MAX, POLICY, MRATED_MIN, I_INV, P_OP, P0, T_OP, TAU_COOL, T_SU_COLD, T_SU_HOT, T0_COLD, T0_HOT,
 T_INERT, T_PURGE, THETA_SPEC, T_BLOW, T_FINAL_INERT, PHI, B_PSA, XO2_C, XO2_N, XO2_OVERRIDE, M_INV0, NITRO0, PSA_LMIN, XO2_JMIN, INIT_STATE, DT, CURVE_MODE, C3_A1, C3_B1, C3_A2, C3_B2,
 BOP_MODE, BOP_R1, HOLD_MODE, T_HOLD, COOL_STANDBY) = range(42)
SW_RAMP, SW_INERT, SW_PURGE, SW_OFFSPEC, SW_BLOW, SW_DEOXO, SW_DRYER = range(7)
# tot indices
(T_STACK, T_DEL, T_PURGE_, T_OFFSPEC, T_DEOXO, T_DRYER, T_BLOWDOWN, T_DINV, T_ESTACK, T_EBOP, T_ESB, T_EAVAIL, T_EUNUSED, T_NHOT, T_NWARM, T_NCOLD,
 T_NDIURNAL, T_NINTRA, T_NABORT, T_NSTOPS, T_MIN_OFF, T_MIN_SB, T_MIN_START, T_MIN_PROD, T_MIN_SD, T_LOSS_OUT, T_REF_OUT, T_ERAMP, T_EBELOW, T_MINV_END,
 T_OPEN_END, T_RESV, T_NCOMPLETE, T_STBY_DT, T_AB_NET, T_CMP_NET, T_PAUSE_MIN) = range(37)


@njit(cache=True)
def _interp2(tab, lg, tg, l, T):
    """Bilinear interpolation of tab[iT, il] on the grids tg (temperature) and lg (load fraction)."""
    nl = lg.size
    nt = tg.size
    l = min(max(l, lg[0]), lg[nl - 1])
    T = min(max(T, tg[0]), tg[nt - 1])
    il = min(max(np.searchsorted(lg, l) - 1, 0), nl - 2)
    it = min(max(np.searchsorted(tg, T) - 1, 0), nt - 2)
    fl = (l - lg[il]) / (lg[il + 1] - lg[il])
    ft = (T - tg[it]) / (tg[it + 1] - tg[it])
    a = tab[it, il] * (1 - fl) + tab[it, il + 1] * fl
    b = tab[it + 1, il] * (1 - fl) + tab[it + 1, il + 1] * fl
    return a * (1 - ft) + b * ft


@njit(cache=True)
def _enter_shutdown(par, sw, m_inv, tot, w_loss, win_open):
    """Blow-down at shutdown entry (POL-A sequence): the inventory is vented unless the mechanism is switched off."""
    if sw[SW_BLOW] == 1:
        tot[T_BLOWDOWN] += m_inv
        if win_open == 1:
            w_loss += m_inv
        m_inv = 0.0
    return m_inv, w_loss


@njit(cache=True)
def simulate(pa, zen, ta, fill, par, sw, lg, tg, mtab, jtab, bop_l, bop_r, bop_g, bop_lg, trace):
    n = pa.size
    p_rated = par[P_RATED]
    p_min = par[P_MIN_F] * p_rated
    dt = par[DT]
    t_deb = int(round(par[T_DEB] / dt))
    mr = par[MRATED_MIN]
    m_full = par[I_INV] * mr * (par[P_OP] - par[P0])
    pol = int(par[POLICY])
    sd_len = par[T_BLOW] + par[T_FINAL_INERT]
    tau = par[TAU_COOL]
    state = OFF
    T = ta[0]
    m_inv = par[M_INV0]
    nitro = par[NITRO0]
    init_prod = 1 if par[INIT_STATE] > 2.5 else 0
    theta = 0.0
    purged = 0
    ph = 0
    ph_clock = 0.0
    t_ramp = 0.0
    t_su = 1.0
    T0 = T
    ramp_on = 0
    above = 0
    below = 0
    sb_clock = 0
    sd_clock = 0.0
    cls = 2
    diurnal = 0
    day_started = -1
    tot = np.zeros(40)
    cyc = np.zeros((12000, 13))
    ncyc = 0
    tr = np.zeros((n if trace else 1, 10))
    win_open = 1
    w_t0 = 0
    w_loss = 0.0
    w_ref = 0.0
    w_stack = 0.0
    w_abort = 0
    w_gap = 0
    bop_mode = int(par[BOP_MODE])
    hold_mode = int(par[HOLD_MODE])
    t_hold = int(round(par[T_HOLD] / dt))
    paused = 0
    in_att = 0
    a_loss = 0.0
    a_ref = 0.0
    w_ab = 0.0
    w_cmp = 0.0
    if init_prod == 1:
        state = PRODUCTION
        T = par[T_OP]
        ramp_on = 1
        t_ramp = 1e9
        t_su = 1.0
        T0 = T
        win_open = 0
        nitro = 0.0
    for k in range(n):
        p_bus = pa[k]
        if bop_mode == 0:
            p_av = p_bus
        elif bop_mode == 1:
            p_av = p_bus / (1.0 + par[BOP_R1])
        else:
            p_av = np.interp(p_bus / p_rated, bop_g, bop_lg) * p_rated
        ok = p_av >= p_min
        if ok:
            above += 1
            below = 0
        else:
            below += 1
            above = 0
        begin_start = 0
        go_shutdown = 0
        # ---------------- transitions (decided on the state at the start of the step)
        if state == OFF:
            if above >= t_deb:
                begin_start = 1
        elif state == STANDBY:
            sb_clock += dt
            if above >= t_deb:
                begin_start = 1
            elif (pol == 1 and sb_clock >= par[T_SB_MAX]) or (pol == 2 and zen[k] >= 90.0):
                go_shutdown = 1
        elif state == START:
            paused = 0
            abort_now = 0
            if hold_mode == 1:
                if below >= t_hold:
                    abort_now = 1
                elif below >= t_deb:
                    paused = 1
            elif below >= t_deb:
                abort_now = 1
            if abort_now == 1:
                tot[T_NABORT] += 1
                w_abort += 1
                paused = 0
                if pol == 0:
                    go_shutdown = 1
                else:
                    if win_open == 1:
                        w_ab += a_loss - a_ref
                    a_loss = 0.0
                    a_ref = 0.0
                    in_att = 0
                    state = STANDBY
                    sb_clock = 0
        elif state == PRODUCTION:
            if below >= t_deb:
                tot[T_NSTOPS] += 1
                win_open = 1
                w_t0 = k
                w_loss = 0.0
                w_ref = 0.0
                w_stack = 0.0
                w_abort = 0
                w_gap = 0
                w_ab = 0.0
                w_cmp = 0.0
                a_loss = 0.0
                a_ref = 0.0
                for q in range(max(0, k - t_deb), min(n, k + t_deb + 1)):
                    if fill[q] > 0:
                        w_gap = 1
                if pol == 0:
                    go_shutdown = 1
                else:
                    state = STANDBY
                    sb_clock = 0
        elif state == SHUTDOWN:
            sd_clock += dt
            if sd_clock >= sd_len - 1e-9:
                state = OFF
                if above >= t_deb:
                    begin_start = 1
        if go_shutdown == 1:
            w_loss_before = w_loss
            m_inv, w_loss = _enter_shutdown(par, sw, m_inv, tot, w_loss, win_open)
            if win_open == 1:
                if in_att == 1:                                       # abort of a START attempt: blow-down belongs to the aborted attempt
                    w_ab += a_loss + (w_loss - w_loss_before) - a_ref
                else:
                    w_cmp += w_loss - w_loss_before
            a_loss = 0.0
            a_ref = 0.0
            in_att = 0
            nitro = 1.0 if sw[SW_OFFSPEC] == 1 else 0.0
            theta = 0.0
            purged = 0
            sd_clock = 0.0
            state = SHUTDOWN
            if sd_len <= 0.0:
                state = OFF
                if above >= t_deb:
                    begin_start = 1
        if begin_start == 1:
            was_standby = 1 if state == STANDBY else 0
            T0 = T
            if was_standby == 1:
                cls = 0
            else:
                cls = 1 if (T - ta[k]) >= 0.5 * (par[T_OP] - ta[k]) else 2
            tot[T_NHOT + cls] += 1
            in_att = 1
            a_loss = 0.0
            a_ref = 0.0
            paused = 0
            th = (par[T_OP] - T0) / (par[T_OP] - par[T0_COLD])
            th_hot = (par[T_OP] - par[T0_HOT]) / (par[T_OP] - par[T0_COLD])
            if th <= th_hot:
                t_su = par[T_SU_HOT]
            elif th >= 1.0:
                t_su = par[T_SU_COLD]
            else:
                t_su = par[T_SU_HOT] + (par[T_SU_COLD] - par[T_SU_HOT]) * (th - th_hot) / (1.0 - th_hot)
            if t_su < 1e-6:
                t_su = 1e-6                          # bounding case t_su,hot = 0 (D40-5): instantaneous ramp, avoids a division by zero
            t_ramp = 0.0
            ramp_on = 1
            ph_clock = 0.0
            state = START
            if was_standby == 1:
                if m_inv < m_full - 1e-12:
                    ph = 2 if (purged == 1 or sw[SW_PURGE] == 0) else 1
                else:
                    ph = 3
            else:
                if sw[SW_INERT] == 1:
                    ph = 0
                    ramp_on = 0
                else:
                    ph = 1
                nitro = 1.0 if sw[SW_OFFSPEC] == 1 else 0.0
                theta = 0.0
                purged = 0
            day = int(k * dt) // 1440
            if day != day_started and zen[k] < 90.0:
                diurnal = 1
                day_started = day
                tot[T_NDIURNAL] += 1
            else:
                diurnal = 0
                tot[T_NINTRA] += 1
            for q in range(max(0, k - t_deb), min(n, k + t_deb + 1)):
                if fill[q] > 0:
                    w_gap = 1
        # ---------------- START phase normalisation and instant completion
        close_pre = 0
        if state == START and ph >= 1:
            if ph == 1 and sw[SW_PURGE] == 0:
                ph = 2
            if ph == 2 and m_inv >= m_full - 1e-12:
                ph = 3
            if ph == 3 and (nitro < 0.5 or sw[SW_OFFSPEC] == 0):
                close_pre = 1
        if close_pre == 1:
            state = PRODUCTION
            tot[T_NCOMPLETE] += 1
            if win_open == 1:
                w_cmp += a_loss - a_ref
            a_loss = 0.0
            a_ref = 0.0
            in_att = 0
            if win_open == 1:
                if ncyc < cyc.shape[0]:
                    cyc[ncyc, 0] = w_t0
                    cyc[ncyc, 1] = k
                    cyc[ncyc, 2] = w_loss
                    cyc[ncyc, 3] = w_ref
                    cyc[ncyc, 4] = w_stack
                    cyc[ncyc, 5] = cls
                    cyc[ncyc, 6] = diurnal
                    cyc[ncyc, 7] = w_abort
                    cyc[ncyc, 8] = w_gap
                    cyc[ncyc, 9] = t_su
                    cyc[ncyc, 10] = 1.0
                    cyc[ncyc, 11] = w_ab
                    cyc[ncyc, 12] = w_cmp
                    ncyc += 1
                win_open = 0
        # ---------------- thermal
        if state == OFF or state == SHUTDOWN:
            T = ta[k] + (T - ta[k]) * np.exp(-dt / tau)
        elif state == STANDBY:
            if par[COOL_STANDBY] > 0.5:                                  # POL-P bounding flag (Brief 5 item 2): the idle stack cools by the D32a ODE while pressure is retained
                T = ta[k] + (T - ta[k]) * np.exp(-dt / tau)
        else:
            if paused == 1:
                T = ta[k] + (T - ta[k]) * np.exp(-dt / tau)
            elif sw[SW_RAMP] == 0:
                T = par[T_OP]
            elif ramp_on == 1:
                T = T0 + (par[T_OP] - T0) * min(1.0, t_ramp / t_su)
            else:
                T = ta[k] + (T - ta[k]) * np.exp(-dt / tau)
        # ---------------- production
        p_st = 0.0
        ms = 0.0
        m_del = 0.0
        d_purge = 0.0
        d_off = 0.0
        d_deox = 0.0
        d_dry = 0.0
        d_inv = 0.0
        close_post = 0
        if state == START and paused == 1:
            tot[T_PAUSE_MIN] += dt
        elif state == START and ph == 0:
            ph_clock += dt
            if ph_clock >= par[T_INERT] - 1e-9:
                ph_clock = 0.0
                ramp_on = 1
                ph = 1
        elif state == START or state == PRODUCTION:
            cap = p_rated
            if ramp_on == 1 and sw[SW_RAMP] == 1:
                cap = p_rated * min(1.0, t_ramp / t_su)
            if ok:
                p_st = min(p_av, cap)
            l = p_st / p_rated
            if ramp_on == 1:
                t_ramp += 60.0 * dt
            if p_st > 0.0:
                if par[CURVE_MODE] > 0.5:
                    pm = p_st / 1000.0                                  # MW (C3 chord curve, P_cap = p_rated/1000 MW)
                    ms = (par[C3_A1] * pm + par[C3_B1] if l <= 0.3 else par[C3_A2] * pm + par[C3_B2]) / 60.0 * dt
                else:
                    ms = _interp2(mtab, lg, tg, l, T) * (p_rated / 1000.0) / 60.0 * dt
            if ms > 0.0:
                s_ss = par[PHI] + par[B_PSA] / max(l, par[PSA_LMIN])
                j = _interp2(jtab, lg, tg, l, T)
                if par[XO2_OVERRIDE] >= 0.0:
                    xo = par[XO2_OVERRIDE] * 1e-6
                else:
                    xo = par[XO2_C] * max(j, par[XO2_JMIN]) ** (-par[XO2_N]) * 1e-6
                if sw[SW_DEOXO] == 0:
                    xo = 0.0
                rem = ms
                if state == START:
                    if ph == 1:
                        frac = min(1.0, (par[T_PURGE] - ph_clock) / dt)
                        d_purge = rem * frac
                        theta += d_purge / mr
                        rem -= d_purge
                        ph_clock += dt
                        if ph_clock >= par[T_PURGE] - 1e-9:
                            purged = 1
                            ph = 2
                            ph_clock = 0.0
                    if ph == 2:
                        need = m_full - m_inv
                        to_inv = min(rem, max(need, 0.0))
                        d_inv = to_inv
                        m_inv += to_inv
                        rem -= to_inv
                        if m_inv >= m_full - 1e-12:
                            ph = 3
                    if ph == 3 and rem > 0.0:
                        room = (par[THETA_SPEC] - theta) * mr
                        dv = min(rem, max(room, 0.0))
                        d_off = dv
                        theta += dv / mr
                        rem -= dv
                        if theta >= par[THETA_SPEC] - 1e-12:
                            nitro = 0.0
                        if nitro < 0.5:
                            close_post = 1
                if rem > 0.0:
                    q_dry = min(s_ss * ms, rem) if sw[SW_DRYER] == 1 else 0.0
                    q_deox = min(2.0 * xo * rem, rem - q_dry)
                    d_dry = q_dry
                    d_deox = q_deox
                    m_del = rem - q_dry - q_deox
        # ---------------- accounting
        e_st = p_st / 60.0 * dt
        if bop_mode == 0:
            p_bop = 0.0
        elif bop_mode == 1:
            p_bop = p_st * par[BOP_R1]
        else:
            p_bop = p_st * np.interp(p_st / p_rated, bop_l, bop_r)
        tot[T_STACK] += ms
        tot[T_DEL] += m_del
        tot[T_PURGE_] += d_purge
        tot[T_OFFSPEC] += d_off
        tot[T_DEOXO] += d_deox
        tot[T_DRYER] += d_dry
        tot[T_ESTACK] += e_st
        tot[T_EAVAIL] += p_bus / 60.0 * dt
        tot[T_EUNUSED] += (p_bus - p_st - p_bop) / 60.0 * dt
        if ramp_on == 1 and sw[SW_RAMP] == 1 and (state == START or state == PRODUCTION) and ok and p_st < p_av - 1e-9:
            tot[T_ERAMP] += (p_bus - p_st - p_bop) / 60.0 * dt
        if (state == START or state == PRODUCTION) and (not ok):
            tot[T_EBELOW] += p_bus / 60.0 * dt
        if p_st > 0.0:
            if bop_mode == 1:
                tot[T_EBOP] += e_st * par[BOP_R1]
            else:
                tot[T_EBOP] += e_st * np.interp(p_st / p_rated, bop_l, bop_r)
        if state == STANDBY:
            tot[T_ESB] += par[SB_F] * p_rated / 60.0 * dt
            tot[T_STBY_DT] += (par[T_OP] - ta[k]) * dt / 60.0
        tot[T_MIN_OFF + (state if state < 4 else 4)] += dt
        if ms > 0.0 or d_inv > 0.0:
            lt = d_purge + d_off + d_deox + d_dry
            sref = ms * (par[PHI] + par[B_PSA] / max(p_st / p_rated, par[PSA_LMIN])) if ms > 0.0 else 0.0
            if win_open == 1:
                w_loss += lt
                w_ref += sref
                w_stack += ms
                if in_att == 1:
                    a_loss += lt
                    a_ref += sref
                else:
                    w_cmp += lt - sref
            else:
                tot[T_LOSS_OUT] += lt
                tot[T_REF_OUT] += sref
        if close_post == 1:
            state = PRODUCTION
            tot[T_NCOMPLETE] += 1
            if win_open == 1:
                w_cmp += a_loss - a_ref
            a_loss = 0.0
            a_ref = 0.0
            in_att = 0
            if win_open == 1:
                if ncyc < cyc.shape[0]:
                    cyc[ncyc, 0] = w_t0
                    cyc[ncyc, 1] = k
                    cyc[ncyc, 2] = w_loss
                    cyc[ncyc, 3] = w_ref
                    cyc[ncyc, 4] = w_stack
                    cyc[ncyc, 5] = cls
                    cyc[ncyc, 6] = diurnal
                    cyc[ncyc, 7] = w_abort
                    cyc[ncyc, 8] = w_gap
                    cyc[ncyc, 9] = t_su
                    cyc[ncyc, 10] = 1.0
                    cyc[ncyc, 11] = w_ab
                    cyc[ncyc, 12] = w_cmp
                    ncyc += 1
                win_open = 0
        if trace:
            tr[k, 0] = state
            tr[k, 1] = p_st
            tr[k, 2] = ms
            tr[k, 3] = m_del
            tr[k, 4] = d_purge
            tr[k, 5] = d_off
            tr[k, 6] = d_deox + d_dry
            tr[k, 7] = m_inv
            tr[k, 8] = T
            tr[k, 9] = tot[T_BLOWDOWN]
    if win_open == 1 and ncyc < cyc.shape[0]:           # unfinished window at the year end: recorded as not closed (excluded from tau_eq statistics)
        cyc[ncyc, 0] = w_t0
        cyc[ncyc, 1] = n
        cyc[ncyc, 2] = w_loss
        cyc[ncyc, 3] = w_ref
        cyc[ncyc, 4] = w_stack
        cyc[ncyc, 5] = cls
        cyc[ncyc, 7] = w_abort
        cyc[ncyc, 8] = w_gap
        cyc[ncyc, 10] = 0.0
        cyc[ncyc, 11] = w_ab
        cyc[ncyc, 12] = w_cmp + a_loss - a_ref
        ncyc += 1
    tot[T_DINV] = m_inv - par[M_INV0]
    tot[T_MINV_END] = m_inv
    tot[T_OPEN_END] = float(win_open)
    return tot, cyc, ncyc, tr

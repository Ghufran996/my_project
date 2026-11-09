"""B-S2: re-implementation of L4 (Virah-Sawmy, Beck, Sturmberg 2024, Int. J. Hydrogen Energy 72:49-59) Eqs. 1-16.

Objective Eq.(1): max sum_t [ M(t) LHV - beta_SU(t) P_cap (5 min) - P_SB beta_SB(t) dT ]  (energy units, MWh)
Constraints: Eq.(2)-(4) power flow and stand-by, Eq.(5)-(6) operating range [0.1, 1] P_cap, Eq.(7) start-up
beta_SU(t) = max(0, beta_On(t) - beta_On(t-1) - beta_SB(t-1)), Eq.(8)-(9) one state, Eq.(10)-(16) efficiency curve,
two-segment chord linearisation of the production curve (Table 1), M = mdot dT.
Interpretation flag (SN, stated): Eq.(8) is printed as beta_off(t)+beta_SB(t) <= 1, which is implied by Eq.(9); the text says
the electrolyser cannot move from off to stand-by and vice versa.  Primary reading = the text (transitions off<->SB forbidden);
`allow_sb_off=True` is the literal reading (sensitivity).  Initial state Off, final state free (not stated in L4).

Two solvers of the SAME problem: `solve_milp` (scipy.optimize.milp, HiGHS) and `solve_dp` (exact dynamic programme over the
three states, numba); their objectives are compared on one-week windows in verification (results/ver2r_milp_check.csv).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

LHV = 0.033            # MWh/kg (L4 Sec 2.1.3.3)
P_MIN, P_SB, E_SU_MIN = 0.1, 0.01, 5.0      # x P_cap, x P_cap, minutes x P_cap
SEG = (0.1, 0.3, 1.0)  # Table 1 breakpoints (x P_cap)
P_ETA_MAX = 0.3


@dataclass
class Curve:
    eta_max: float
    eta_min: float          # eta_min == eta_max with fixed=True means constant efficiency
    fixed: bool = False

    def eta(self, x):
        if self.fixed:
            return np.full_like(np.asarray(x, float), self.eta_max)
        d = self.eta_max - self.eta_min
        return -d / 0.49 * (np.asarray(x, float) - P_ETA_MAX) ** 2 + self.eta_max


def segments(curve: Curve, p_cap: float):
    """Chord slopes/intercepts of mdot(P) [kg/h] on the two segments, P in MW."""
    out = []
    for lo, hi in zip(SEG[:-1], SEG[1:]):
        p = np.array([lo, hi]) * p_cap
        m = curve.eta(np.array([lo, hi])) * p / LHV
        a = (m[1] - m[0]) / (p[1] - p[0])
        out.append((a, m[0] - a * p[0], lo * p_cap, hi * p_cap))
    return out


@njit(cache=True)
def _dp(pre, p_cap, p_min, p_sb, dt_h, e_su, a1, b1, a2, b2, allow_sb_off):
    n = len(pre)
    NEG = -1e300
    v = np.array([0.0, NEG, NEG])          # states: 0 off, 1 standby, 2 on ; start Off
    bp = np.zeros((n, 3), np.int8)
    gain_on = np.zeros(n)
    for t in range(n):
        nv = np.array([NEG, NEG, NEG])
        # stage values
        g_on = NEG
        if pre[t] >= p_min:
            p = min(pre[t], p_cap)
            if p <= 0.3 * p_cap:
                m = a1 * p + b1
            else:
                m = a2 * p + b2
            g_on = m * dt_h * 0.033
        g_sb = -p_sb * dt_h if pre[t] >= p_sb else NEG
        gain_on[t] = g_on
        # to off: from off, on (and from sb only if allowed)
        best, arg = v[0], 0
        if v[2] > best:
            best, arg = v[2], 2
        if allow_sb_off and v[1] > best:
            best, arg = v[1], 1
        nv[0], bp[t, 0] = best, arg
        # to standby: from sb, on (and from off only if allowed)
        if g_sb > NEG:
            best, arg = v[1], 1
            if v[2] > best:
                best, arg = v[2], 2
            if allow_sb_off and v[0] > best:
                best, arg = v[0], 0
            if best > NEG / 2:
                nv[1], bp[t, 1] = best + g_sb, arg
        # to on: from on, sb (free), off (start-up cost)
        if g_on > NEG:
            best, arg = v[2], 2
            if v[1] > best:
                best, arg = v[1], 1
            if v[0] - e_su > best:
                best, arg = v[0] - e_su, 0
            if best > NEG / 2:
                nv[2], bp[t, 2] = best + g_on, arg
        v = nv
    return v, bp, gain_on


def solve_dp(p_re_mw, dt_h: float, p_cap: float, curve: Curve, allow_sb_off: bool = False) -> dict:
    s1, s2 = segments(curve, p_cap)
    e_su = p_cap * E_SU_MIN / 60.0
    v, bp, _ = _dp(np.asarray(p_re_mw, float), p_cap, P_MIN * p_cap, P_SB * p_cap, dt_h, e_su, s1[0], s1[1], s2[0], s2[1], allow_sb_off)
    state = int(np.argmax(v))
    obj = float(v[state])
    n = len(p_re_mw)
    path = np.zeros(n, np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = int(bp[t, state])
    return _account(np.asarray(p_re_mw, float), path, dt_h, p_cap, curve, obj)


def _account(pre, path, dt_h, p_cap, curve, obj):
    from ..shared import account_c3
    s1, s2 = segments(curve, p_cap)
    h2, e_el, e_sb, n_su, hrs = account_c3(pre, path, dt_h, p_cap, P_SB, s1[0], s1[1], s2[0], s2[1])
    return dict(objective_MWh=obj, H2_kg=float(h2), n_startups=int(n_su), standby_MWh=float(e_sb), electrolyser_MWh=float(e_el),
                hours_on=float(hrs), path=path)


def solve_milp(p_re_mw, dt_h: float, p_cap: float, curve: Curve, allow_sb_off: bool = False, time_limit: float = 300.0) -> dict:
    """The same problem as a MILP (HiGHS via scipy.optimize.milp). Variables per step: on, sb, z1, z2, p1, p2, su."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    pre = np.asarray(p_re_mw, float)
    n = len(pre)
    s1, s2 = segments(curve, p_cap)
    e_su = p_cap * E_SU_MIN / 60.0
    nv = 7
    ix = lambda k, t: nv * t + k          # k: 0 on, 1 sb, 2 z1, 3 z2, 4 p1, 5 p2, 6 su
    c = np.zeros(nv * n)
    for t in range(n):
        c[ix(2, t)] = -s1[1] * dt_h * LHV
        c[ix(3, t)] = -s2[1] * dt_h * LHV
        c[ix(4, t)] = -s1[0] * dt_h * LHV
        c[ix(5, t)] = -s2[0] * dt_h * LHV
        c[ix(6, t)] = e_su
        c[ix(1, t)] = P_SB * p_cap * dt_h
    rows, lo, hi = [], [], []
    A = lil_matrix((12 * n + 2, nv * n))
    r = 0

    def add(entries, l, u):
        nonlocal r
        for j, val in entries:
            A[r, j] = val
        lo.append(l); hi.append(u); r += 1

    for t in range(n):
        add([(ix(2, t), 1), (ix(3, t), 1), (ix(0, t), -1)], 0, 0)                       # z1+z2 = on
        add([(ix(0, t), 1), (ix(1, t), 1)], 0, 1)                                       # on + sb <= 1
        add([(ix(4, t), 1), (ix(2, t), -s1[2])], 0, np.inf)                             # p1 >= lo z1
        add([(ix(4, t), 1), (ix(2, t), -s1[3])], -np.inf, 0)                            # p1 <= hi z1
        add([(ix(5, t), 1), (ix(3, t), -s2[2])], 0, np.inf)
        add([(ix(5, t), 1), (ix(3, t), -s2[3])], -np.inf, 0)
        add([(ix(4, t), 1), (ix(5, t), 1), (ix(1, t), P_SB * p_cap)], -np.inf, pre[t])  # load <= P_RE
        prev_on = [(ix(0, t - 1), 1)] if t > 0 else []
        prev_sb = [(ix(1, t - 1), 1)] if t > 0 else []
        # su(t) >= on(t) - on(t-1) - sb(t-1)
        ent = [(ix(6, t), 1), (ix(0, t), -1)] + [(j, v) for j, v in prev_on] + [(j, v) for j, v in prev_sb]
        add(ent, 0, np.inf)
        if not allow_sb_off:
            # sb(t) <= on(t-1) + sb(t-1)   (no off -> sb);  sb(t-1) <= on(t) + sb(t)   (no sb -> off)
            add([(ix(1, t), 1)] + [(j, -v) for j, v in prev_on] + [(j, -v) for j, v in prev_sb], -np.inf, 0)
            if t > 0:
                add([(ix(1, t - 1), 1), (ix(0, t), -1), (ix(1, t), -1)], -np.inf, 0)
    A = A[:r].tocsr()
    lb = np.zeros(nv * n)
    ub = np.ones(nv * n)
    for t in range(n):
        ub[ix(4, t)] = s1[3]
        ub[ix(5, t)] = s2[3]
        ub[ix(6, t)] = 1
    integrality = np.zeros(nv * n)
    for t in range(n):
        for k in (0, 1, 2, 3):
            integrality[ix(k, t)] = 1
    res = milp(c, constraints=LinearConstraint(A, lo, hi), bounds=Bounds(lb, ub), integrality=integrality,
               options=dict(time_limit=time_limit, mip_rel_gap=1e-9))
    if res.x is None:
        return dict(objective_MWh=float("nan"), status=res.message, success=False)
    x = res.x.reshape(n, nv)
    mdot = (x[:, 4] * s1[0] + x[:, 2] * s1[1] + x[:, 5] * s2[0] + x[:, 3] * s2[1])
    return dict(objective_MWh=float(-res.fun), H2_kg=float(np.sum(mdot * dt_h)), n_startups=int(round(x[:, 6].sum())),
                status=res.message, success=bool(res.success), mip_gap=getattr(res, "mip_gap", float("nan")))


from ..shared import block_mean  # noqa: E402,F401  (shared with the plant engine, VER-2L)

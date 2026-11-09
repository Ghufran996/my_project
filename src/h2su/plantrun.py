"""Parameter assembly, stack tables, run wrapper and metrics for the plant engine (Brief 3 items 5, 9).

Plant: 1 MW DC electrolyser (P_rated), 1.5 MWp PV (pv.py). Stack H2 rate and current density as functions of load fraction l = P/P_rated and
stack temperature from the D29 polarisation (L26 equations + L44 conductivity scaled once to the L26 operating point 1.80 V at 1.75 A/cm2, 75 C) and
the L13 Eq.(16) Faradaic efficiency (T = stack temperature, SN07). Rated point = 1.75 A/cm2 at 75 C (the calibration point).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import plant as pl
from .constants import F_PAPER, M_H2
from .electrolyzer import calibrate_sigma_scale, cell_voltage_d29, faradaic_efficiency_persson
from .io import RESULTS, load_registry, param

P_RATED_KW = 1000.0
J_RATED, U_RATED_NOM, T_OP_C = 1.75, 1.80, 75.0
TG = np.arange(20.0, 95.1, 5.0)
LG = np.concatenate([[0.0], np.geomspace(1e-3, 1.0, 140)])


@lru_cache(maxsize=1)
def stack_tables():
    """mtab[iT, il] [kg/h for 1 MW rated], jtab[iT, il] [A/cm2], plus rated flow [kg/h] and sigma_scale."""
    ss = calibrate_sigma_scale()
    p_rated_cm2 = J_RATED * float(cell_voltage_d29(J_RATED, T_OP_C + 273.15, 200.0, ss))      # W/cm2 at the rated point (= 3.15 for 1.80 V)
    area_cm2 = P_RATED_KW * 1e3 / p_rated_cm2                                                # total active area for 1 MW at the rated point
    jg = np.geomspace(1e-4, 4.0, 4000)
    mtab = np.zeros((TG.size, LG.size))
    jtab = np.zeros_like(mtab)
    for it, Tc in enumerate(TG):
        T = Tc + 273.15
        U = cell_voltage_d29(jg, T, 200.0, ss)
        p = jg * U / p_rated_cm2                                                                # load fraction
        eta = faradaic_efficiency_persson(U, T)
        m = eta * jg * area_cm2 / (2.0 * F_PAPER) * M_H2 * 3600.0                                # kg/h
        order = np.argsort(p)
        mtab[it] = np.interp(LG, p[order], m[order], left=0.0)
        jtab[it] = np.interp(LG, p[order], jg[order], left=0.0)
    rated = float(mtab[list(TG).index(T_OP_C), -1])
    return dict(mtab=mtab, jtab=jtab, m_rated_kg_h=rated, sigma_scale=ss, area_cm2=area_cm2, p_rated_cm2=p_rated_cm2)


def bop_ratio():
    """BoP energy per unit stack energy vs load from L1 Table 1 (energy_split_loadXX 'BoP/stack/pressure/purge/eff', first entry = BoP share of input)."""
    reg = load_registry().set_index("name")
    L, r = [], []
    for ld in range(10, 101, 10):
        v = [float(x) for x in str(reg.loc[f"energy_split_load{ld}", "value_or_range"]).split("/")]
        f = v[0] / 100.0
        L.append(ld / 100.0)
        r.append(f / (1.0 - f))
    return np.array(L), np.array(r)


def xo2_fit():
    """VT4 primary fit (CAL 600/900/1200 A of L24, 600 cm2): c [ppm at 1 A/cm2], n."""
    area = param("pem50_area")
    I = np.array([600.0, 900.0, 1200.0])
    x = np.array([param(f"pem50_OTH_{int(i)}A") for i in I])
    j = I / area
    A = np.vstack([np.ones(3), -np.log10(j)]).T
    (lc, n), *_ = np.linalg.lstsq(A, np.log10(x), rcond=None)
    return 10 ** lc, float(n)


def psa_fit():
    """VT5a: s(L) = phi + b/L (fractions), CAL 10 % and 100 % load of L1 Table 2."""
    s10, s100 = param("psa_purge_loss_load10") / 100.0, param("psa_purge_loss_load100") / 100.0
    b = (s10 - s100) / (1 / 0.10 - 1.0)
    return s100 - b, b


def bop_l24_ratio() -> float:
    """BoP / stack power with the L24 constant share (D41): P_BoP / (P_sys - P_BoP) = 10.2 / (69.7 - 10.2)."""
    pb, ps = param("pem50_BoP_power"), param("pem50_system_input_power")
    return pb / (ps - pb)


def bop_inverse(bl, br):
    """Grids for the stack power as a function of the PV-bus power when the BoP load-ratio r(l) of L1 Table 1 is drawn from the PV bus:
    p_bus/p_rated = l (1 + r(l)); returns (g, l) with g strictly increasing."""
    l = np.linspace(0.0, 1.0, 1001)
    g = l * (1.0 + np.interp(l, bl, br))
    # the piecewise-linear L1 ratio makes g(l) dip by 1.4 % between l = 0.188 and 0.2 (peak 0.449 P_rated); the running maximum selects the largest
    # feasible stack power for a bus power inside that band (a band 0.2 % P_rated wide); strictly increasing by a 1e-12 tilt
    g = np.maximum.accumulate(g) + 1e-12 * l
    assert np.all(np.diff(g) > 0)
    return g, l


def build_params(policy: str = "A", t_sb_min: float = 0.0, t_deb: float = 1.0, **over) -> np.ndarray:
    """Nominal parameter vector. policy: 'A', 'B' (t_sb_min), 'C', 'D' (= B with 1440 min)."""
    st = stack_tables()
    inv = pd.read_csv(RESULTS / "inventory_ranges.csv").set_index("id")
    th = pd.read_csv(RESULTS / "thermal_ranges.csv")
    vt3 = pd.read_csv(RESULTS / "vt3.csv").set_index("item")
    C_kJ = float(th[(th.quantity == "C_per_MW") & (th.end.str.startswith("L26 printed"))].value.iloc[0])
    R_KkW = float(th[(th.quantity == "R_per_MW") & (th.end.str.startswith("L26 printed"))].value.iloc[0])
    tau_min = R_KkW * C_kJ / 60.0                                      # K/kW x kJ/K = kJ/kW = s -> min
    c_o, n_o = xo2_fit()
    phi, b = psa_fit()
    pcode = {"A": 0, "B": 1, "C": 2, "D": 1}[policy]
    t_sb = 1440.0 if policy == "D" else t_sb_min
    p = np.zeros(42)
    p[pl.P_RATED] = P_RATED_KW
    p[pl.P_MIN_F] = param("pmin_pem_low")
    p[pl.T_DEB] = t_deb
    p[pl.SB_F] = 0.5 * (param("standby_power_low") + param("standby_power_high"))
    p[pl.T_SB_MAX] = t_sb
    p[pl.POLICY] = pcode
    p[pl.MRATED_MIN] = st["m_rated_kg_h"] / 60.0
    p[pl.I_INV] = float(inv.loc["PEM_L13_nominal", "i_min_per_bar"])           # min of rated production per bar
    p[pl.P_OP] = param("plant_operating_pressure_inferred")
    p[pl.P0] = 1.01325
    p[pl.T_OP] = T_OP_C
    p[pl.TAU_COOL] = tau_min
    p[pl.T_SU_COLD] = param("pem190_coldstart_time")
    p[pl.T_SU_HOT] = param("pem190_hotstart_time")
    p[pl.T0_COLD] = param("pem190_coldstart_T0")
    p[pl.T0_HOT] = param("pem190_hotstart_T0")
    p[pl.T_INERT] = param("awe_inerting_initial_total") + param("awe_prestart")
    p[pl.T_PURGE] = param("awe_purge_duration")
    p[pl.THETA_SPEC] = float(vt3.loc["theta_spec_L28 (rated-minutes of diverted H2 to spec)", "value"])
    p[pl.T_BLOW] = param("awe_depressurisation_duration_avg")
    p[pl.T_FINAL_INERT] = param("awe_final_inerting_total")
    p[pl.PHI] = phi
    p[pl.B_PSA] = b
    p[pl.XO2_C] = c_o
    p[pl.XO2_N] = n_o
    p[pl.XO2_OVERRIDE] = -1.0
    p[pl.M_INV0] = 0.0
    p[pl.NITRO0] = 1.0
    p[pl.PSA_LMIN] = 0.10
    p[pl.XO2_JMIN] = 0.01
    p[pl.INIT_STATE] = 0
    p[pl.DT] = 1.0
    p[pl.BOP_MODE] = 0.0
    p[pl.BOP_R1] = bop_l24_ratio()
    p[pl.HOLD_MODE] = 0.0
    p[pl.T_HOLD] = 15.0               # min, D43b: longest PL8 t_deb
    p[pl.COOL_STANDBY] = 0.0          # Brief 5 item 2: POL-P bounding flag, off by default (neutral)
    for k, v in over.items():
        p[getattr(pl, k)] = v
    return p


ALL_ON = np.ones(7, dtype=np.int64)
ALL_OFF = np.zeros(7, dtype=np.int64)


def switches(**off) -> np.ndarray:
    s = ALL_ON.copy()
    for k in off:
        s[getattr(pl, "SW_" + k.upper())] = 0
    return s


def s1_config(par: np.ndarray):
    """B-S1: every start-up mechanism off (instant start/stop, steady PSA loss only); inventory pre-filled so no pressurisation is needed."""
    p = par.copy()
    p[pl.T_BLOW] = 0.0
    p[pl.T_FINAL_INERT] = 0.0
    p[pl.NITRO0] = 0.0
    p[pl.M_INV0] = p[pl.I_INV] * p[pl.MRATED_MIN] * (p[pl.P_OP] - p[pl.P0])
    sw = ALL_OFF.copy()
    sw[pl.SW_DRYER] = 1
    return p, sw


def s3_config(par: np.ndarray):
    """B-S3: purification mechanisms off (no inerting, purge, off-spec, blow-down, deoxo, dryer) but the thermal start-up ramp kept."""
    p, sw = s1_config(par)
    sw[pl.SW_RAMP] = 1
    sw[pl.SW_DRYER] = 0            # no purification at all (C4/C2/C5 class)
    return p, sw


def s2rule_config(par: np.ndarray, curve):
    """B-S2-rule diagnostic: the full model's state machine/policy and PV input, C3 efficiency curve (chord, L4), C3 start-up penalty (P_cap x 5 min of no production
    after an off -> on start; a standby -> on restart is free), purification off, no ramp, no shutdown sequence, no inventory."""
    from .baselines import bs2_c3 as b
    p, sw = s1_config(par)
    s1, s2 = b.segments(curve, P_RATED_KW / 1000.0)
    p[pl.CURVE_MODE] = 1.0
    p[pl.C3_A1], p[pl.C3_B1], p[pl.C3_A2], p[pl.C3_B2] = s1[0], s1[1], s2[0], s2[1]
    p[pl.T_INERT] = b.E_SU_MIN
    p[pl.T_PURGE] = 0.0
    sw = ALL_OFF.copy()
    sw[pl.SW_INERT] = 1                # the 5-min start-up penalty is carried by the inerting phase slot (no production)
    sw[pl.SW_RAMP] = 0
    return p, sw


def fixed_eff_mtab(eta: float, lhv_kwh_kg: float) -> np.ndarray:
    """Stack H2 table of a fixed-LHV-efficiency stack (B-CL convention), kg/h per MW rated, same grid as stack_tables()['mtab']."""
    return np.tile(eta * LG * 1000.0 / lhv_kwh_kg, (TG.size, 1))


def run(df: pd.DataFrame, par: np.ndarray, sw: np.ndarray, trace: bool = False, p_rated_kw: float = P_RATED_KW, mtab: np.ndarray | None = None):
    st = stack_tables()
    bl, br = bop_ratio()
    bg, blg = bop_inverse(bl, br)
    mode = int(par[pl.BOP_MODE])
    clip = p_rated_kw * (1.0 if mode == 0 else (1.0 + par[pl.BOP_R1]) if mode == 1 else (1.0 + br[-1]))      # PV bus rating = stack rating (+ BoP when drawn from the bus)
    pa = np.minimum(df.p_dc_kw.to_numpy(float), clip)
    out = pl.simulate(pa, df.zenith.to_numpy(float), df.ta_c.to_numpy(float), (df.fill.to_numpy() > 0).astype(np.int64), par, sw, pl.np.asarray(LG), pl.np.asarray(TG),
                      st["mtab"] if mtab is None else mtab, st["jtab"], bl, br, bg, blg, trace)
    return out


def metrics(tot: np.ndarray, cyc: np.ndarray, ncyc: int, par: np.ndarray) -> dict:
    """PL4 metrics + D32d accounting. Cycles: closed windows only enter tau_eq statistics; the unfinished window at the year end is included in lambda_SU."""
    mr = par[pl.MRATED_MIN]
    c = cyc[:ncyc]
    stack = tot[pl.T_STACK]
    loss_tot = tot[pl.T_PURGE_] + tot[pl.T_OFFSPEC] + tot[pl.T_DEOXO] + tot[pl.T_DRYER] + tot[pl.T_BLOWDOWN]
    w_loss, w_ref = c[:, 2].sum(), c[:, 3].sum()
    su = w_loss - w_ref
    ss = w_ref + tot[pl.T_LOSS_OUT]
    closed = c[c[:, 10] > 0.5]
    tau_eq = (closed[:, 2] - closed[:, 3]) / mr if len(closed) else np.array([])
    m = dict(M_stack=stack, M_del=tot[pl.T_DEL], M_purge=tot[pl.T_PURGE_], M_offspec=tot[pl.T_OFFSPEC], M_deoxo=tot[pl.T_DEOXO], M_dryer=tot[pl.T_DRYER],
             M_blowdown=tot[pl.T_BLOWDOWN], dM_inv=tot[pl.T_DINV], lambda_SU=su / stack * 100 if stack > 0 else np.nan, lambda_SS=ss / stack * 100 if stack > 0 else np.nan,
             eta_del=tot[pl.T_DEL] / stack if stack > 0 else np.nan, N_r_hot=tot[pl.T_NHOT], N_r_warm=tot[pl.T_NWARM], N_r_cold=tot[pl.T_NCOLD],
             N_r_diurnal=tot[pl.T_NDIURNAL], N_r_intra=tot[pl.T_NINTRA], N_abort=tot[pl.T_NABORT], N_stops=tot[pl.T_NSTOPS],
             tau_eq_median=float(np.median(tau_eq)) if tau_eq.size else np.nan, tau_eq_P10=float(np.percentile(tau_eq, 10)) if tau_eq.size else np.nan,
             tau_eq_P90=float(np.percentile(tau_eq, 90)) if tau_eq.size else np.nan, n_cycles_closed=int(len(closed)),
             SEC_del=(tot[pl.T_ESTACK] + tot[pl.T_EBOP] + tot[pl.T_ESB]) / tot[pl.T_DEL] if tot[pl.T_DEL] > 0 else np.nan, E_stack=tot[pl.T_ESTACK], E_bop=tot[pl.T_EBOP],
             E_standby=tot[pl.T_ESB], E_avail=tot[pl.T_EAVAIL], E_ramp_limited=tot[pl.T_ERAMP], E_below_min=tot[pl.T_EBELOW],
             share_gap_affected=float(closed[:, 8].mean()) if len(closed) else np.nan)
    mode = int(par[pl.BOP_MODE])
    ebop_drawn = tot[pl.T_EBOP] if mode != 0 else 0.0
    m["closure_H2_rel"] = abs(stack - (tot[pl.T_DEL] + loss_tot + tot[pl.T_DINV])) / stack if stack > 0 else 0.0
    m["closure_E_rel"] = abs(tot[pl.T_EAVAIL] - tot[pl.T_ESTACK] - ebop_drawn - tot[pl.T_EUNUSED]) / tot[pl.T_EAVAIL] if tot[pl.T_EAVAIL] > 0 else 0.0
    # --- Brief 4 item 4: restart counting split, abort tag, SEC_del conventions, standby energy
    att = tot[pl.T_NHOT] + tot[pl.T_NWARM] + tot[pl.T_NCOLD]
    ab_net, cmp_net = c[:, 11].sum(), c[:, 12].sum()
    m.update(N_r_attempts=att, N_r_completed=tot[pl.T_NCOMPLETE], N_r_aborted=tot[pl.T_NABORT], N_r_unfinished=att - tot[pl.T_NCOMPLETE] - tot[pl.T_NABORT],
             lambda_SU_complete=cmp_net / stack * 100 if stack > 0 else np.nan, lambda_SU_abort=ab_net / stack * 100 if stack > 0 else np.nan,
             identity_abort_rel=abs(su - ab_net - cmp_net) / stack if stack > 0 else 0.0, pause_min=tot[pl.T_PAUSE_MIN], min_standby=tot[pl.T_MIN_SB])
    pr_ = par[pl.P_RATED]
    r24 = par[pl.BOP_R1]
    dl = tot[pl.T_DEL]
    for nm, f in (("sb001", 0.01), ("sb0095", par[pl.SB_F]), ("sb018", 0.18)):
        esb = f * pr_ * tot[pl.T_MIN_SB] / 60.0
        m[f"E_standby_{nm}"] = esb
        m[f"SEC_del_L24_{nm}"] = (tot[pl.T_ESTACK] * (1.0 + r24) + esb) / dl if dl > 0 and mode == 0 else np.nan
        m[f"SEC_del_L1_{nm}"] = (tot[pl.T_ESTACK] + tot[pl.T_EBOP] + esb) / dl if dl > 0 and mode == 0 else np.nan
    m["SEC_del_noBoP_sb0095"] = (tot[pl.T_ESTACK] + par[pl.SB_F] * pr_ * tot[pl.T_MIN_SB] / 60.0) / dl if dl > 0 else np.nan
    for nm, R in (("R2p3", 2.3), ("R10p5", 10.467910447761195)):
        m[f"E_standby_thermal_{nm}"] = tot[pl.T_STBY_DT] / R * (pr_ / 1000.0)          # (T_op - T_amb)/R per MW, kWh
    if mode != 0:
        m["SEC_del_BoP_from_PV"] = (tot[pl.T_ESTACK] + tot[pl.T_EBOP] + tot[pl.T_ESB]) / dl if dl > 0 else np.nan
    m["identity_SU_SS_rel"] = abs((stack - tot[pl.T_DEL] - tot[pl.T_DINV]) / stack - (su + ss) / stack) if stack > 0 else 0.0
    return m


def run_c3(pre_mw, path, dt_h: float, p_cap_mw: float, curve):
    """Plant engine in its C3 configuration (purification off): an imposed on/standby/off path is accounted with the SHARED power -> H2/energy function
    (h2su.shared.account_c3), the same one used by baselines/bs2_c3.py (VER-2L)."""
    from .baselines import bs2_c3 as b
    from .shared import account_c3
    s1, s2 = b.segments(curve, p_cap_mw)
    return account_c3(np.asarray(pre_mw, float), np.asarray(path), dt_h, p_cap_mw, b.P_SB, s1[0], s1[1], s2[0], s2[1])

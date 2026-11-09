"""Brief 2 item 9: VT1 (L26 part), VT2 (L28 Table 13), VT5a (L1 Table 2) with the pre-registered CAL/VAL roles.

    python code/vt_runs.py {vt1|vt2|vt5a|all}

CAL values are frozen and their file SHA-256 is printed/logged before any VAL comparison; VAL rows are read only through
registry_scalar(..., ('VAL',)) in the evaluation functions, never by a calibration routine (PermissionError otherwise).
A failed VT is reported, not tuned (PL9).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su import electrolyzer as el  # noqa: E402
from h2su import gas, purification, thermal  # noqa: E402
from h2su.io import RESULTS  # noqa: E402

TOL = 0.20


def vt1_pipeline(label: str, n_stacks: int = 2, galv2_step2: float = 1.0, lam: float | None = None, C_scale: float = 1.0):
    row = thermal.load_tables().loc[50]                         # test station (Table 2/4 row printed '50 kWel', SN04)
    C, G_tables = thermal.lumped_from_tables(row, n_stacks, thermal.Rauls())
    C *= C_scale
    s = el.calibrate_sigma_scale(lam=lam)
    m0 = thermal.Vt1(C=C, R=1.0, n_cells=27 * n_stacks, sigma_scale=s)
    strat = thermal.vt1_strategies(galv2_step2)
    cal_min = thermal.registry_scalar("exp_heatup_galv_2step", ("CAL",))
    t_lossfree = thermal.vt1_time_to_75C(strat["galvanostatic_2step"], thermal.Vt1(C, 1e3, m0.n_cells, m0.area_cm2, m0.d_um, s))
    feasible = bool(t_lossfree <= cal_min)                      # R only adds losses: the CAL case is reachable only if the loss-free time <= measured
    R = thermal.vt1_calibrate_R(C, cal_min, m0, strat["galvanostatic_2step"]) if feasible else float("nan")
    return dict(label=label, n_stacks=n_stacks, galv2_step2_A_cm2=galv2_step2, lam=lam, C_scale=C_scale, sigma_scale=s, C_J_per_K=C,
                R_K_per_W=R, R_K_per_kW=R * 1e3, R_tables_K_per_kW=1e3 / G_tables, cal_minutes=cal_min,
                cal_time_loss_free_min=t_lossfree, cal_feasible=feasible), strat


def printed_over_table_C_ratio() -> float:
    """Mean of printed total C_th (L26 Sec 3.3.2.2) over the Table 3 x Table 4 sum for the 400/750/1000 kW rows (diagnostic only)."""
    pub = pd.read_csv(Path(__file__).resolve().parent / "configs" / "tables" / "L26_rauls2023_published_lumped.csv", comment="#")
    tab = thermal.load_tables()
    r = []
    for kw in (400, 750, 1000):
        v = float(pub[(pub.quantity == "C_th_system_total") & (pub.P_kW == kw)].value.iloc[0]) * 1e3
        c, _ = thermal.lumped_from_tables(tab.loc[kw], 1, thermal.Rauls())
        r.append(v / c)
    return float(np.mean(r))


def run_vt1():
    out = RESULTS / "vt1_cal_frozen.json"
    rows = []
    ratio = printed_over_table_C_ratio()
    variants = [("primary", dict()), ("sens_one_stack", dict(n_stacks=1)), ("sens_galv2_text_1p5", dict(galv2_step2=1.5)),
                ("sens_lambda_14", dict(lam=14.0)), ("DIAG_C_x_printed_over_table_ratio", dict(C_scale=ratio))]
    frozen_all = {}
    for label, kw in variants:
        frozen_all[label], _ = vt1_pipeline(label, **kw)
    out.write_text(json.dumps(frozen_all, indent=1))
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print("CAL frozen file sha256:", h, "| printed/table C ratio (diagnostic):", round(ratio, 4))
    for label, kw in variants:
        fr = json.loads(out.read_text())[label]                       # VAL evaluation reads only the frozen file
        strat = thermal.vt1_strategies(fr["galv2_step2_A_cm2"])
        R_use = fr["R_K_per_W"] if fr["cal_feasible"] else 1e3        # CAL infeasible -> loss-free bound, flagged DIAGNOSTIC
        basis = "calibrated R" if fr["cal_feasible"] else "CAL INFEASIBLE: loss-free bound R = 1000 K/W (diagnostic, not a verdict)"
        m = thermal.Vt1(C=fr["C_J_per_K"], R=R_use, n_cells=27 * fr["n_stacks"], sigma_scale=fr["sigma_scale"])
        cases = [("galvanostatic_2step", "exp_heatup_galv_2step", ("CAL",), "CAL"),
                 ("galvanostatic_3step", "exp_heatup_galv_3step", ("VAL",), "VAL"),
                 ("potentiostatic_1p8V", "exp_heatup_potentiostatic_1p8V", ("VAL",), "VAL"),
                 ("potentiostatic_1p8V_plus_4kW", "exp_heatup_potentiostatic_1p8V_plus_4kW", ("VAL",), "VAL"),
                 ("heating_only", "exp_heatup_heating_only", ("AUX",), "AUX (supplementary held-out)")]
        for sname, reg, roles, role in cases:
            meas = thermal.registry_scalar(reg, roles)
            t = thermal.vt1_time_to_75C(strat[sname], m)
            re_ = (t - meas) / meas
            rows.append(dict(variant=label, basis=basis, cal_feasible=fr["cal_feasible"], case=sname, role=role, t_measured_min=meas,
                             t_model_min=t, RE=re_, abs_RE_le_20pct=bool(abs(re_) <= TOL) if np.isfinite(re_) else False,
                             sigma_scale=fr["sigma_scale"], C_kJ_per_K=fr["C_J_per_K"] / 1e3, R_cal_K_per_kW=fr["R_K_per_kW"],
                             R_tables_K_per_kW=fr["R_tables_K_per_kW"], cal_time_loss_free_min=fr["cal_time_loss_free_min"],
                             cal_file_sha256=h))
        print(label, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "vt1_l26.csv", index=False)
    print(df[["variant", "case", "role", "t_measured_min", "t_model_min", "RE", "abs_RE_le_20pct", "cal_feasible"]].round(4).to_string(index=False))
    return df


def run_vt2():
    val, cal = gas.vt2_awe()
    kp = gas.awe_gas_capacity_mol_per_bar(cal)
    pem_lo, pem_hi = gas.pem_gas_capacity_range_mol_per_bar()
    val.to_csv(RESULTS / "vt2_l28.csv", index=False)
    extra = pd.DataFrame([dict(item="AWE K_p (CAL, power set 75 %)", value_mol_per_bar=kp, note="n_dot(75 %)=0.75 x 2 Nm3/h / gamma; V_eff = K_p R T needs T_gas (SN25)"),
                          dict(item="PEM K_p lower (p0 = 0 bar)", value_mol_per_bar=pem_lo, note="L13: 1.7 kg H2 in 45 min to 30 bar (inferred); CAL only, no PEM VAL (SN24)"),
                          dict(item="PEM K_p upper (p0 = 1 bar)", value_mol_per_bar=pem_hi, note="same")])
    extra.to_csv(RESULTS / "vt2_gas_capacity.csv", index=False)
    print("CAL:", cal)
    print(val.round(4).to_string(index=False))
    print(extra.round(3).to_string(index=False))
    return val, extra


def run_vt5a():
    df, par = purification.vt5a()
    df.to_csv(RESULTS / "vt5a_l1.csv", index=False)
    print(par)
    print(df.round(4).to_string(index=False))
    return df


def run_vt3():
    """VT3 (Brief 3 item 3a): the L28 purge is a fixed procedure time (registry awe_purge_termination), not a purity-terminated step, so
    VT3 is NOT EXECUTABLE as pre-registered. Reported: UQ range of the N2 washout time constant tau (N = 1 tank), both ends labelled."""
    from h2su.io import param
    n2_lim_ppm = param("iso_N2_max")
    k = float(np.log(1e6 / n2_lim_ppm))                         # time constants from pure N2 to the ISO limit
    t_l28 = param("awe_purge_duration")
    t_l14 = 100.0                                               # min, L14 abstract (full text NOT FOUND, SN16): 100 min cold start to the ISO N2 level
    rows = [dict(item="VT3 verdict", value="NOT EXECUTABLE", note="L28 purge ends on a fixed procedure time (Sec 6.3, Table 12), no composition criterion; VAL (L14 100 min) not usable without the L14 conditions (SN16)"),
            dict(item="ln(1e6/300) time constants to ISO N2 limit", value=k, note=f"N2 limit {n2_lim_ppm} umol/mol (registry iso_N2_max)"),
            dict(item="tau_L28 (assumes purge ends at spec)", value=t_l28 / k, note="min per rated flow; 6.5 min / ln(1e6/300); LABEL: L28-derived assuming purge-to-spec"),
            dict(item="tau_L14 (abstract 100 min, rated current assumed, flagged)", value=t_l14 / k, note="min per rated flow; LABEL: L14-derived from abstract only; L14 is a 50 kW AWE, start current unknown"),
            dict(item="theta_spec_L28 (rated-minutes of diverted H2 to spec)", value=t_l28, note="= tau_L28 x k"),
            dict(item="theta_spec_L14", value=t_l14, note="= tau_L14 x k")]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "vt3.csv", index=False)
    print(df.to_string(index=False))
    return df


def run_vt4():
    """VT4 (Brief 3 item 3b): x_O2 = c (j/j_ref)^(-n), fitted in log10 on CAL 600/900/1200 A only (L24 text values; 600 cm2 -> j = 1.0/1.5/2.0 A/cm2);
    secondary Fickian x_O2 = c/j (1 parameter). VAL = L24 300 A (extrapolation; the printed value is a start-up transient peak, Fig. 3).
    L25 0.25 % / 0.040 % NOT EXECUTABLE (current density / instant current not printed). The L24 digitisation hash is checked first."""
    from h2su.io import param, load_registry
    h = hashlib.sha256((Path(__file__).resolve().parent / "configs" / "tables" / "L24_fig3.csv").read_bytes()).hexdigest()
    assert h in (RESULTS / "vt4_digitisation_hash.txt").read_text(), "L24 digitisation file hash changed"
    area = thermal.registry_scalar("pem50_area", ("INPUT",))
    I = np.array([600.0, 900.0, 1200.0])
    x = np.array([thermal.registry_scalar(f"pem50_OTH_{int(i)}A", ("CAL",)) for i in I])
    j = I / area
    cal_frozen = dict(j=j.tolist(), x_ppm=x.tolist(), registry_sha256=hashlib.sha256((Path(__file__).resolve().parent / "configs" / "parameters.csv").read_bytes()).hexdigest())
    (RESULTS / "vt4_cal_frozen.json").write_text(json.dumps(cal_frozen, indent=1))
    A = np.vstack([np.ones(3), -np.log10(j)]).T
    (lc, n), *_ = np.linalg.lstsq(A, np.log10(x), rcond=None)
    c_f = 10 ** np.mean(np.log10(x * j))
    j_val = 300.0 / area
    x_val = thermal.registry_scalar("pem50_OTH_peak_300A", ("VAL",))
    rows = []
    for name, pred, npar in (("primary power law", 10 ** lc * j_val ** (-n), 2), ("secondary Fickian c/j", c_f / j_val, 1)):
        d = float(np.log10(pred) - np.log10(x_val))
        rows.append(dict(model=name, n_params=npar, c_ppm=(10 ** lc if npar == 2 else c_f), n=(float(n) if npar == 2 else 1.0), point="L24 300 A (extrapolation; transient peak)", j_A_cm2=j_val,
                         x_pred_ppm=float(pred), x_meas_ppm=x_val, delta_log10=d, abs_dlog10_le_0p15=bool(abs(d) <= 0.15), verdict="PASS" if abs(d) <= 0.15 else "FAIL",
                         note="300 A value = start-up transient (Fig. 3, 0-130 s, about 1000 ppm); steady-state law cannot represent it; disclosed: the value was in the Stage 1 cache when the form was fixed"))
    for lab in ("L25 0.25 % (end of cold start, 3800 A)", "L25 0.040 % (hot start, average over ramp)"):
        rows.append(dict(model="both", point=lab, verdict="NOT EXECUTABLE", note="active area / current density not printed (L25); hot-start value is an average over the ramp"))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "vt4.csv", index=False)
    print(df[["model", "point", "x_pred_ppm", "x_meas_ppm", "delta_log10", "verdict"]].to_string(index=False)); print("c, n =", 10 ** lc, n, "; CAL fit residual (log10):", (A @ np.array([lc, n]) - np.log10(x)).round(4))
    return df


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("vt1", "all"):
        run_vt1()
    if which in ("vt2", "all"):
        run_vt2()
    if which in ("vt5a", "all"):
        run_vt5a()
    if which in ("vt3", "all"):
        run_vt3()
    if which in ("vt4", "all"):
        run_vt4()

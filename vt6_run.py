"""Brief 4 item 1: L15 consistency + transcription check, VT6 calibration (BT2), freeze, held-out validation, local-equilibrium bound.

    python code/vt6_run.py {consistency|val|regen|all}
    python code/vt6_run.py cal --freeze     write results/vt6_cal_frozen.json and .sha256 (refuses if the frozen file exists)
    python code/vt6_run.py cal --verify     re-run the BT2 fit, compare with the frozen value (rel_diff <= 1e-9) -> results/vt6_cal_verify.csv; never writes the frozen file; exit 1 on failure

Outputs: results/l15_consistency.csv, results/vt6_cal_frozen.json (+ .sha256), results/vt6_cal_verify.csv, results/vt6_l15.csv, results/vt6_bt6bt7_regen_check.csv
D46 rules: CAL = BT2 breakpoint t_b (5 umol/mol first crossing) reproduced exactly by one k_LDF; VAL = BT1, BT3, BT4, BT5, BT8 with abs RE <= 20 %;
the frozen file is written and hashed before any VAL run; VAL values are never read by the fitting routine.
Freeze rule (D81, Brief 11): the frozen file is written once by `cal --freeze`. The chain runs `cal --verify` only. The freeze is identified by a content hash (SHA-256 of the
canonical JSON without the field "created", h2su.freeze); the .sha256 file holds that hash; readers assert file hash == .sha256 value == the hard-coded constant and the
R39 value at 7 significant digits.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import constants as sc
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from h2su import dryer as dr  # noqa: E402
from h2su.freeze import content_hash, load_frozen_vt6  # noqa: E402
from h2su.io import CONFIGS, RESULTS, load_registry  # noqa: E402

R = sc.gas_constant
S4 = pd.read_csv(CONFIGS / "tables" / "L15_S4.csv").set_index("experiment")
FROZEN = RESULTS / "vt6_cal_frozen.json"
SHA = RESULTS / "vt6_cal_frozen.json.sha256"
REL_TOL_VERIFY = 1e-9
N_REF = 400          # reference grid (upwind finite volume) for CAL and VAL
RTOL = 1e-6
THR = 5.0            # umol/mol, L15 eq 9 breakpoint


def t_le(p, iso):
    """Local-equilibrium, dispersion-free breakthrough time of the leading shock (k_LDF -> infinity), method of characteristics:
    t = (L/u_s)(eps + rho_bed/(1e-6 c_T) min_y q*(y)/y), y in (0, y_in] (tangent from the origin of the isotherm = leading shock)."""
    T, pbar = p["T_K"], p["p_Pa"] / 1e5
    cT = p["p_Pa"] / (R * T)
    us = p["N_flux_mol_m2_s"] / cT
    y = np.geomspace(1e-3, p["yW_in_umol_mol"], 20000)
    ratio = dr.q_water(y * 1e-6 * pbar, T, iso) / y
    j = int(np.argmin(ratio))
    return p["L_m"] / us * (p["eps"] + p["rho_bed_kg_m3"] / (1e-6 * cT) * ratio[j]), float(y[j])


def consistency():
    df = load_registry()
    iso = dr.iso_params(df)
    rows = []
    add = lambda chk, qty, a, b, note="", unit="": rows.append(dict(check=chk, quantity=qty, value_a=a, value_b=b, unit=unit,
                                                                 rel_diff=(a - b) / b if isinstance(a, float) and isinstance(b, float) and b != 0 else np.nan,
                                                                 note=note))
    # main text vs SI
    add("main vs SI", "packing density: main text (p. N) vs SI Table S3", 790.0, 789.43, "main text vs SI", "kg/m3")
    add("main vs SI", "bed density from the BT bed: Table S1 mass/volume vs main text 790", 6.17e-3 / 7.85e-6, 790.0, "6.17 g / 7.85 cm3", "kg/m3")
    add("main vs SI", "bed density from the VPSA bed: Table S2 mass/volume vs main text 790", 6.11e-3 / 7.85e-6, 790.0, "6.11 g / 7.85 cm3", "kg/m3")
    add("main vs SI", "bed volume Table S1/S2 (7.85 cm3) vs pi/4 Di^2 H (Di 1.0 cm, H 10 cm)", 7.85, float(np.pi / 4 * 1.0 ** 2 * 10.0), "", "cm3")
    add("main vs SI", "purge-to-feed ratio: Table S2 / Table 7 (1 %) vs registry vpsa_purge_to_feed", 1.0, float(df.loc[df.name == "vpsa_purge_to_feed", "value_or_range"].iloc[0]), "equal", "%")
    add("main vs SI", "H2 recovery: Table 9 experimental (98.9803) vs abstract (98.98)", 98.9803, 98.98, "rounding", "%")
    add("main vs SI", "regeneration pressure: Table S2 P_EV (0.025 bar) vs text 'ca. 25 mbar' (p. H)", 0.025, 0.025, "equal", "bar")
    Vd = dr.design_volume(df)
    add("main vs SI", "design vessel volume pi/4 Di^2 Hv (0.346 m, 3.46 m) vs SI Table S3 V (0.326 m3)", float(Vd), 0.326, "", "m3")
    # internal source inconsistencies
    L, A = dr.lab_geometry(df)
    n_lab = 1.5e-3 / 60.0 / (R * 273.15 / 101325.0)
    m_lab_kg_h = n_lab * dr.M_H2 * 3600.0
    scale_m = Vd / (L * A) * m_lab_kg_h
    add("source-internal", "design feed mass flow implied by 'volume proportional to feed flow' from the lab bed (V_design/V_lab x lab flow) vs printed 300 kg/h", float(scale_m), 300.0, "scale rule vs printed design flow", "kg/h")
    cT78 = 7.8e5 / (R * 301.65)
    A_d = np.pi / 4 * 0.346 ** 2
    u_calc = 300.0 / 3600.0 / dr.M_H2 / cT78 / A_d
    add("source-internal", "superficial velocity of 300 kg/h in Di 0.346 m at 7.8 bar, 28.5 degC (computed) vs printed 0.354 m/s (p. M)", float(u_calc), 0.354, "ratio about 4: printed velocity not reproducible from the printed flow and geometry; NOT used", "m/s")
    add("source-internal", "Table S4 regeneration times of BT6/BT7: text (p. I) BT6 = 1.5 h, BT7 = 2.5 h vs Table S4 BT6 = 2.5 h, BT7 = 1.5 h", np.nan, np.nan, "UNRESOLVED conflict; the Table S4 row values (qs 6.78 / 4.70) are consistent with more regeneration -> higher capacity only for the 2.5 h label of Table S4", "")
    add("source-internal", "saturation pressure of water at the 5 umol/mol breakpoint (-65.5 degC) with eq 7 (water) in umol/mol at 1 bar vs the stated 5 umol/mol", float(dr.psat_bar(273.15 - 65.5, iso) / 1.0 * 1e6), 5.0, "eq 7 is over liquid water; a frost point at -65.5 degC corresponds to about 5 umol/mol (interpretation, not stated by the source)", "umol/mol")
    # absolute humidity check (p. I: 1.1 and 0.8 'mol m-3')
    for T_C, printed in ((28.5, 1.1), (23.0, 0.8)):
        T = T_C + 273.15
        w = 0.7 * dr.psat_bar(T, iso) * 1e5 / (R * T)
        add("source-internal", f"water concentration at RH 70 %, {T_C} degC: RH psat(eq 7)/(R T) vs printed (p. I, unit printed 'mol mN-3')", float(w), printed, "confirms hPa unit of eq 7 and the RH route", "mol/m3")
    # Tdp at 1 bar vs RH route
    for bt, r in S4.iterrows():
        T = r.T_C + 273.15
        y_rh = r.RH_pct / 100 * dr.psat_bar(T, iso) / r.P_bar * 1e6
        y_td = dr.psat_bar(273.15 + r.Tdp_at_1bar_C, iso) / 1.0 * 1e6
        add("transcription", f"{bt}: feed y from Tdp@1 bar (eq 7, water) vs from RH psat/P", float(y_td), float(y_rh), "Tdp < 0 degC probably reported as frost point (ice); BT4 (Tdp > 0) agrees", "umol/mol")
    # transcription check: Toth loading at each BT condition vs Table S4 q_s
    worst = 0.0
    for bt, r in S4.iterrows():
        if bt in ("BT6", "BT7"):
            continue
        T = r.T_C + 273.15
        pw = r.RH_pct / 100 * dr.psat_bar(T, iso)
        q = float(dr.q_water(pw, T, iso))
        add("transcription", f"{bt}: Toth-AD loading at feed vs Table S4 q_s (D46 stop rule: abs RE > 10 %)", q, float(r.q_s_mol_kg), f"P_w = {pw:.5f} bar", "mol/kg")
        worst = max(worst, abs(q - r.q_s_mol_kg) / r.q_s_mol_kg)
        n_w = r.F_feed_LN_min / 1e3 / 60.0 / (R * 273.15 / 101325.0) * r.RH_pct / 100 * dr.psat_bar(T, iso) / r.P_bar
        add("transcription", f"{bt}: mass-balance upper bound n_w t_s / m_ads vs q_s", float(n_w * r.t_s_h * 3600 / 6.17e-3), float(r.q_s_mol_kg), "area-above-curve <= t_s x feed rate", "mol/kg")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "l15_consistency.csv", index=False, float_format="%.6g")
    print(out[["check", "quantity", "value_a", "value_b", "rel_diff"]].to_string())
    print("worst transcription |RE|:", worst)
    if worst > 0.10:
        raise SystemExit("STOP: transcription check > 10 %")
    return worst


def lab_row(bt, kLDF, df, iso, t_end_s=8 * 3600.0):
    return dr.lab_case(df, iso, S4.loc[bt].to_dict(), kLDF, t_end_s=t_end_s)


def _fit_bt2():
    """BT2 fit (reads Table S4 row BT2 only): brentq on log10(k_LDF) in [4e-3, 8e-3] 1/s. Returns k, t_b, n_evaluations, t_LE, y_shock, target_s."""
    df = load_registry()
    iso = dr.iso_params(df)
    target_s = float(S4.loc["BT2", "t_b_h"]) * 3600.0       # the ONLY measured number the fit may read (BT2)

    def f(logk):
        tb, _ = dr.t_b(lab_row("BT2", 10 ** logk, df, iso, 6 * 3600.0), iso, N=N_REF, rtol=RTOL, thr=THR)
        return tb - target_s
    p_ref = lab_row("BT2", 1.0, df, iso)
    tle, ys = t_le(p_ref, iso)
    lo, hi = np.log10(4e-3), np.log10(8e-3)
    flo, fhi = f(lo), f(hi)
    print("bracket", flo, fhi, "t_LE h", tle / 3600, flush=True)
    if flo * fhi > 0:
        raise SystemExit(f"CAL infeasible in the bracket: f(lo)={flo:.1f}, f(hi)={fhi:.1f}; LE bound {tle/3600:.3f} h")
    n_eval = [0]

    def g(x):
        n_eval[0] += 1
        t0 = time.time()
        v = f(x)
        print(f"  eval {n_eval[0]}: k={10 ** x:.10g} f={v:.4f} s wall={time.time() - t0:.1f}s", flush=True)
        return v
    lk = brentq(g, lo, hi, xtol=1e-7, rtol=1e-9, maxiter=60)
    k = 10 ** lk
    tb, r = dr.t_b(lab_row("BT2", k, df, iso, 6 * 3600.0), iso, N=N_REF, rtol=RTOL, thr=THR)
    return k, tb, n_eval[0], tle, ys, target_s


def cal_freeze():
    """Write the frozen calibration once (refuses if the file exists)."""
    if FROZEN.exists():
        raise SystemExit(f"REFUSED: {FROZEN} exists; the frozen calibration is written once (D81). Use `cal --verify`.")
    k, tb, n_eval, tle, ys, target_s = _fit_bt2()
    obj = dict(parameter="kLDF_1_s", value=float(k), unit="1/s", calibrated_on="BT2 (L15 Table S4): t_b = 3.00 h at the 5 umol/mol breakpoint (first crossing)",
               fit_target_t_b_s=target_s, fit_result_t_b_s=float(tb), residual_s=float(tb - target_s), n_evaluations=n_eval,
               model=dict(grid="upwind finite volume, N=%d uniform, BDF rtol=%g, atol_y=1e-3 umol/mol, atol_q=1e-5 mol/kg" % (N_REF, RTOL),
                          equations="src/h2su/dryer.py docstring", axial_dispersion_m2_s=0.0, isothermal=True, porosity=0.4, packing_density_kg_m3=790.0,
                          lab_bed="L=0.10 m, Di=1.0 cm", feed_normal_state="0 degC, 101325 Pa (assumed)", optimiser="scipy.optimize.brentq on log10(k), bracket [4e-3, 8e-3] (from the scan t_b(3e-3) = 2.72 h, t_b(5e-3) = 2.95 h, t_b(1e-2) = 3.16 h at N = 400)",
                          local_equilibrium_bound_t_b_h=float(tle / 3600), local_equilibrium_shock_y_umol_mol=ys),
               created=time.strftime("%Y-%m-%d %H:%M:%S"))
    FROZEN.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    h = content_hash(FROZEN)
    SHA.write_text(h + "  vt6_cal_frozen.json content-sha256, created excluded (D81)\n", encoding="utf-8")
    print("CAL k_LDF =", k, "t_b h", tb / 3600, "content sha256", h, flush=True)
    return k, h


def cal_verify():
    """Re-run the BT2 fit exactly as the freeze run and compare with the frozen value; never writes the frozen file (exit code 1 is raised by the caller on failure)."""
    k_frozen, h = load_frozen_vt6(FROZEN, SHA)
    k, tb, n_eval, tle, ys, target_s = _fit_bt2()
    rel = abs(k - k_frozen) / k_frozen
    row = dict(k_frozen=k_frozen, k_rerun=float(k), rel_diff=float(rel), n_evaluations=int(n_eval), residual_s=float(tb - target_s), passed=bool(rel <= REL_TOL_VERIFY))
    pd.DataFrame([row]).to_csv(RESULTS / "vt6_cal_verify.csv", index=False, float_format="%.12g")
    print("CAL verify:", row, "frozen content sha256", h, flush=True)
    return row


def val():
    df = load_registry()
    iso = dr.iso_params(df)
    k, sha = load_frozen_vt6(FROZEN, SHA)
    rows = []
    thr_water = dr.psat_bar(273.15 - 65.5, iso) * 1e6              # eq 7 conversion of -65.5 degC (reported, not selected)
    for bt in ("BT2", "BT1", "BT3", "BT4", "BT5", "BT8"):
        r = S4.loc[bt]
        p = lab_row(bt, k, df, iso, 8 * 3600.0)
        t0 = time.time()
        tb, res = dr.t_b(p, iso, N=N_REF, rtol=RTOL, thr=THR)
        tb800, _ = dr.t_b(p, iso, N=800, rtol=RTOL, thr=THR) if bt in ("BT2", "BT1", "BT4") else (np.nan, None)
        tb_alt = dr.first_crossing(res["t"], res["y_out"], thr_water)
        tle, ys = t_le(p, iso)
        meas = float(r.t_b_h) * 3600.0
        re = (tb - meas) / meas
        rows.append(dict(test="VT6", role="CAL" if bt == "BT2" else "VAL", experiment=bt, P_bar=r.P_bar, RH_pct=r.RH_pct, T_C=r.T_C, y_in_umol_mol=p["yW_in_umol_mol"],
                         t_b_pred_h=tb / 3600, t_b_meas_h=meas / 3600, RE=re, verdict=("PASS" if abs(re) <= 0.20 else "FAIL") if bt != "BT2" else "CAL (exact by construction)",
                         tolerance="abs RE <= 20 %", t_b_LE_bound_h=tle / 3600, RE_LE=(tle - meas) / meas,
                         t_b_pred_N800_h=tb800 / 3600 if not np.isnan(tb800) else np.nan, t_b_alt_threshold_h=tb_alt / 3600, alt_threshold_umol_mol=thr_water,
                         t_s_meas_h=r.t_s_h, kLDF_1_s=k, frozen_sha256=sha, wall_s=time.time() - t0,
                         note="alt threshold = eq 7 (water) conversion of -65.5 degC, reported only; isothermal model vs measured bed temperature rises of 2.5-6.5 degC"))
        print(rows[-1]["experiment"], rows[-1]["t_b_pred_h"], rows[-1]["RE"], rows[-1]["verdict"], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "vt6_l15.csv", index=False, float_format="%.6g")
    return out


def regen_check():
    """BT6/BT7 executability: regeneration of the BT2-saturated lab bed at 25 mbar, 15 mL/min dry H2, 28.5 degC with the frozen k and the trace-species
    form (constant total molar flux). Reports the maximum outlet water mole fraction; the trace form requires y << 1."""
    df = load_registry()
    iso = dr.iso_params(df)
    k, _ = load_frozen_vt6(FROZEN, SHA)
    p2 = lab_row("BT2", k, df, iso)
    q_sat = float(dr.q_water(p2["yW_in_umol_mol"] * 1e-6 * p2["p_Pa"] / 1e5, p2["T_K"], iso))
    p = dict(p2)
    p.update(p_Pa=0.025e5, N_flux_mol_m2_s=0.01 * p2["N_flux_mol_m2_s"], yW_in_umol_mol=0.0, yW_init_umol_mol=0.0, q_init_mol_kg=q_sat, t_end_s=2.5 * 3600, dt_out_s=30.0)
    res = dr.run_case(p, iso, N=200, rtol=RTOL)
    ymax = float(res["y_out"].max())
    rows = []
    for tag, hrs in (("1.5 h", 1.5), ("2.5 h", 2.5)):
        i = int(round(hrs * 3600 / p["dt_out_s"]))
        mean_q = float(np.mean(res["sol"]["F"][1, :, i]))
        n_w_out = float(np.trapezoid(res["y_out"][:i + 1] * 1e-6, res["t"][:i + 1]) * p["N_flux_mol_m2_s"] * p["A_m2"])
        rows.append(dict(regen_duration=tag, q_saturated_mol_kg=q_sat, bed_mean_q_after_mol_kg=mean_q, fraction_removed=1 - mean_q / q_sat, y_out_max_umol_mol=ymax,
                         y_out_max_mol_mol=ymax * 1e-6, trace_assumption_ok=bool(ymax * 1e-6 < 0.05), water_removed_mol_trace_form=n_w_out,
                         purge_gas_mol=p["N_flux_mol_m2_s"] * p["A_m2"] * hrs * 3600))
    out = pd.DataFrame(rows)
    out["verdict_BT6_BT7"] = "NOT EXECUTABLE (trace-species constant-flux form invalid at the water mole fractions of the 25 mbar regeneration; a pressure-controlled total-flux form is a new model form for the analyst)" \
        if not rows[0]["trace_assumption_ok"] else "executable"
    out.to_csv(RESULTS / "vt6_bt6bt7_regen_check.csv", index=False, float_format="%.6g")
    print(out.to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    flag = sys.argv[2] if len(sys.argv) > 2 else None
    if what in ("consistency", "all"):
        consistency()
    if what == "cal":
        if flag == "--freeze":
            cal_freeze()
        elif flag == "--verify":
            if not cal_verify()["passed"]:
                raise SystemExit(1)
        else:
            raise SystemExit("usage: vt6_run.py cal --freeze | --verify")
    if what == "all" and not cal_verify()["passed"]:
        raise SystemExit(1)
    if what in ("val", "all"):
        val()
    if what in ("regen", "all"):
        regen_check()

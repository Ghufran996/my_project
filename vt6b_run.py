"""Brief 5 item 5: VT6b (L15 BT6/BT7 partial regeneration, pre-registered non-trace form D53) and the plant dryer cases dry_regen / dry_su_ii in both purge directions.

    python code/vt6b_run.py defcheck    -> results/vt6b_definitions_check.csv  (t_s, q_s definitions checked on the fully regenerated BT1-BT5, BT8 with the frozen k_LDF; informational)
    python code/vt6b_run.py vt6b        -> results/vt6b_l15.csv
    python code/vt6b_run.py note        -> adds/refreshes the column dlog10_note of the stored results/vt6b_l15.csv (no solver run; same function as the vt6b writer, D87)
    python code/vt6b_run.py plant       -> results/ver6_dryer_nontrace.csv, adds on-spec throughput and off-spec timing columns (Brief 6 item 6); the event-bound file is written by dryer_gridband.py bound

Protocol (D53, fixed before any run): start from q uniform at saturation with the BT2 feed (8 bar, RH 70 %, 28.5 degC); regeneration 25 mbar, 28.5 degC, dry H2 purge at 1 % of the feed
molar flow (= 15 mL/min of 1.5 LN/min), duration per Table S4 (BT6 2.5 h, BT7 1.5 h; swapped text reading = sensitivity); purge in the feed direction (primary, L15 p. I) or countercurrent
(sensitivity); then breakthrough at the BT2 feed conditions. The frozen k_LDF (results/vt6_cal_frozen.json) is used; nothing is fitted.
Definitions (L15 Sec. Adsorption equilibrium, eqs 8-9): q_s = F_feed W0 / (60 m_ads) integral_0^{t_s} (1 - W_s/W0) dt, t_s = saturation time. The source gives no numeric criterion for t_s;
IMPLEMENTER READING: t_s = first time the outlet water mole fraction reaches 0.99 y_in (0.95 y_in as sensitivity). m_ads = 6.17 g (Table S1).
Tdp_min (measured: lowest outlet dew point of the breakthrough run, Table S4) -> model: minimum outlet mole fraction for t >= t_hold = 5 L/u_s (IMPLEMENTER READING: the first 5 gas residence
times are the displacement of the initial dry gas; sensitivities 3 and 10), mole fraction -> dew point by L15 eq 7 scaled so that -65.5 degC = 5 umol/mol (primary) or unscaled with 1 bar (reported).
Targets/tolerances (D53): abs log10 ratio of the mole fraction <= 0.15 (Tdp_min); abs RE <= 20 % (t_s, q_s). VT6b PASS only if all targets pass at BT6 and BT7; the two dew-point readings
disagreeing on the verdict -> INCONCLUSIVE.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2su import dryer as dr  # noqa: E402
from h2su import dryer_regen as rg  # noqa: E402
from h2su.constants import M_H2  # noqa: E402
from h2su.io import CONFIGS, RESULTS, load_registry  # noqa: E402
from dryer_gridband import offspec_metrics  # noqa: E402

S4 = pd.read_csv(CONFIGS / "tables" / "L15_S4.csv").set_index("experiment")
FROZEN = RESULTS / "vt6_cal_frozen.json"
SHA = RESULTS / "vt6_cal_frozen.json.sha256"
M_ADS_KG = 6.17e-3
TS_FRAC = 0.99
THOLD_MULT = 5.0
N_GRID = 200


def frozen_k():
    """Frozen k_LDF with the D81 content-hash check (file == .sha256 == constant) and the R39 value at 7 significant digits."""
    from h2su.freeze import load_frozen_vt6
    return load_frozen_vt6(FROZEN, SHA)


def eq7_scaled_y(tdp_c: float, iso) -> float:
    """Mole fraction from a dew point: eq 7 scaled so that -65.5 degC = 5 umol/mol (primary)."""
    return 5e-6 * dr.psat_bar(273.15 + tdp_c, iso) / dr.psat_bar(273.15 - 65.5, iso)


def eq7_unscaled_y(tdp_c: float, iso) -> float:
    return dr.psat_bar(273.15 + tdp_c, iso) / 1.01325


def breakthrough_metrics(res, p, frac=TS_FRAC, m_ads=M_ADS_KG, thold_mult=THOLD_MULT):
    t, y = res["t"], res["y_out"] * 1e-6                               # mole fraction
    yin = p["yW_in_umol_mol"] * 1e-6
    us = res["us"]
    thold = thold_mult * p["L_m"] / us
    m = t >= thold
    ymin = float(y[m].min())
    idx = np.where(y >= frac * yin)[0]
    ts = float(t[idx[0]]) if len(idx) else float("nan")
    if len(idx):
        k = idx[0]
        mask = t <= t[k]
        n_feed = p["N_flux_mol_m2_s"] * p["A_m2"]
        qs = n_feed * yin / m_ads * np.trapezoid(1.0 - np.minimum(y[mask] / yin, 1.0), t[mask])
    else:
        qs = float("nan")
    tb = dr.first_crossing(t, res["y_out"], 5.0)
    return dict(y_min=ymin, t_s_s=ts, q_s=qs, t_b_s=tb, t_hold_s=thold)


def lab_params(df, iso, k, exp="BT2", t_end=7 * 3600.0):
    return dr.lab_case(df, iso, S4.loc[exp].to_dict(), k, t_end_s=t_end, dt_out_s=10.0)


def regen_then_bt(df, iso, k, regen_h, direction, N=N_GRID, rtol=1e-6, thold_mult=THOLD_MULT, frac=TS_FRAC, y_init_bt=0.0):
    pb = lab_params(df, iso, k)
    pr_ = dict(pb)
    pr_.update(p_Pa=0.025e5, N0_mol_m2_s=0.01 * pb["N_flux_mol_m2_s"], y_init=pb["yW_in_umol_mol"] * 1e-6)
    qsat = float(dr.q_water(pb["yW_in_umol_mol"] * 1e-6 * pb["p_Pa"] / 1e5, pb["T_K"], iso))
    pr_["q_init_mol_kg"] = qsat
    t = np.arange(0.0, regen_h * 3600.0 + 1e-9, 30.0)
    rs = rg.regen_solve(pr_, iso, N, t, rtol)
    q_end = rs["q"][:, -1].copy()
    if direction == "countercurrent":
        q_end = q_end[::-1]
    y0 = np.vstack([np.full(N, y_init_bt), q_end])
    bt = dr.run_case(pb, iso, N, rtol, "upwind", y0_profiles=y0)
    m = breakthrough_metrics(bt, pb, frac, thold_mult=thold_mult)
    m.update(q_sat=qsat, q_mean_end=float(q_end.mean()), frac_removed=1.0 - float(q_end.mean()) / qsat, y_out_max_regen=float(rs["y"].max()),
             water_balance_rel=float(rs["water_balance_rel"].max()), total_balance_rel=float(rs["total_balance_rel"].max()), bt_balance=float(bt["sol"]["balance_err"].max()))
    return m, rs, bt, pb


def defcheck():
    k, h = frozen_k()
    df, iso = load_registry(), dr.iso_params()
    rows = []
    for exp in ("BT1", "BT2", "BT3", "BT4", "BT5", "BT8"):
        p = lab_params(df, iso, k, exp, t_end=9 * 3600.0)
        r = dr.run_case(p, iso, 400, 1e-6, "upwind")
        for frac in (0.95, 0.99):
            m = breakthrough_metrics(r, p, frac)
            rows.append(dict(experiment=exp, t_s_threshold=frac, t_s_pred_h=m["t_s_s"] / 3600, t_s_meas_h=S4.loc[exp, "t_s_h"], t_s_RE=(m["t_s_s"] / 3600 - S4.loc[exp, "t_s_h"]) / S4.loc[exp, "t_s_h"],
                             q_s_pred=m["q_s"], q_s_meas=S4.loc[exp, "q_s_mol_kg"], q_s_RE=(m["q_s"] - S4.loc[exp, "q_s_mol_kg"]) / S4.loc[exp, "q_s_mol_kg"],
                             note="informational: definition check on fully regenerated beds with the frozen k_LDF (not part of VT6 VAL)"))
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "vt6b_definitions_check.csv", index=False, float_format="%.5g")
    print(d.round(3).to_string(index=False))


DLOG10_NOTE = "y_min_pred = 0; dlog10 evaluated at the 1e-30 floor, lower bound only"


def dlog10_note(y_min_pred_umol_mol):
    """D87: dlog10 of a row with y_min_pred = 0 is evaluated at the 1e-30 floor of verdict_row, so it is a lower bound, not a prediction error of that many decades."""
    return pd.Series([DLOG10_NOTE if float(v) == 0.0 else "" for v in y_min_pred_umol_mol], index=getattr(y_min_pred_umol_mol, "index", None), dtype=object)


def note_stored():
    """Brief 12 item 2: append/refresh the column dlog10_note of the stored vt6b_l15.csv as text; every other column is kept byte-for-byte (no solver run)."""
    import csv
    f = RESULTS / "vt6b_l15.csv"
    with open(f, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    if "dlog10_note" in head:
        j = head.index("dlog10_note")
        head, body = head[:j] + head[j + 1:], [r[:j] + r[j + 1:] for r in body]
    i = head.index("y_min_pred_umol_mol")
    notes = [DLOG10_NOTE if float(r[i]) == 0.0 else "" for r in body]
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\r\n")
        w.writerow(head + ["dlog10_note"])
        for r, n in zip(body, notes):
            w.writerow(r + [n])
    print("dlog10_note rows:", sum(1 for n in notes if n), "of", len(notes))


def verdict_row(m, meas, iso):
    out = {}
    for rd, fy in (("eq7_scaled", eq7_scaled_y), ("eq7_unscaled", eq7_unscaled_y)):
        ym = fy(meas["Tdp_min_C"], iso)
        d = abs(np.log10(max(m["y_min"], 1e-30) / ym))
        out[rd] = dict(y_meas=ym, dlog10=d, ok=d <= 0.15)
    ts_re = (m["t_s_s"] / 3600 - meas["t_s_h"]) / meas["t_s_h"]
    qs_re = (m["q_s"] - meas["q_s_mol_kg"]) / meas["q_s_mol_kg"]
    return out, ts_re, qs_re


def vt6b():
    k, h = frozen_k()
    df, iso = load_registry(), dr.iso_params()
    rows = []
    t0 = time.time()
    # primary: BT6 2.5 h, BT7 1.5 h (Table S4), feed-direction purge; sensitivities: swapped durations, countercurrent purge, thold 3/10, t_s threshold 0.95, grid N=400
    cfgs = [("primary", dict(swap=False, direction="feed", thold=5.0, frac=0.99, N=200)),
            ("grid N=400", dict(swap=False, direction="feed", thold=5.0, frac=0.99, N=400)),
            ("swapped durations (text reading)", dict(swap=True, direction="feed", thold=5.0, frac=0.99, N=200)),
            ("countercurrent purge", dict(swap=False, direction="countercurrent", thold=5.0, frac=0.99, N=200)),
            ("t_hold 3 L/u", dict(swap=False, direction="feed", thold=3.0, frac=0.99, N=200)),
            ("t_hold 10 L/u", dict(swap=False, direction="feed", thold=10.0, frac=0.99, N=200)),
            ("t_s threshold 0.95", dict(swap=False, direction="feed", thold=5.0, frac=0.95, N=200))]
    for label, c in cfgs:
        for bt in ("BT6", "BT7"):
            dur = float(S4.loc[bt, "regeneration"].split()[0])
            if c["swap"]:
                dur = 4.0 - dur                                      # 2.5 <-> 1.5 h
            m, rs, btres, pb = regen_then_bt(df, iso, k, dur, c["direction"], c["N"], 1e-6, c["thold"], c["frac"])
            meas = S4.loc[bt].to_dict()
            vr, ts_re, qs_re = verdict_row(m, meas, iso)
            pass_ts, pass_qs = abs(ts_re) <= 0.20, abs(qs_re) <= 0.20
            if "_full" not in locals() or _full[0] != (c["N"], c["thold"], c["frac"]):
                pbf = lab_params(df, iso, k)
                rf = dr.run_case(pbf, iso, c["N"], 1e-6, "upwind")
                mf = breakthrough_metrics(rf, pbf, c["frac"], thold_mult=c["thold"])
                _full = ((c["N"], c["thold"], c["frac"]), mf)
            mf = _full[1]
            ratio_ts, ratio_qs = m["t_s_s"] / mf["t_s_s"], m["q_s"] / mf["q_s"]
            meas_ts, meas_qs = meas["t_s_h"] / S4.loc["BT2", "t_s_h"], meas["q_s_mol_kg"] / S4.loc["BT2", "q_s_mol_kg"]
            rows.append(dict(reading=label, experiment=bt, regen_h=dur, purge_direction=c["direction"], N=c["N"], Tdp_min_meas_C=meas["Tdp_min_C"],
                             y_min_pred_umol_mol=m["y_min"] * 1e6, y_meas_scaled_umol_mol=vr["eq7_scaled"]["y_meas"] * 1e6, dlog10_scaled=vr["eq7_scaled"]["dlog10"], ok_Tdp_scaled=vr["eq7_scaled"]["ok"],
                             y_meas_unscaled_umol_mol=vr["eq7_unscaled"]["y_meas"] * 1e6, dlog10_unscaled=vr["eq7_unscaled"]["dlog10"], ok_Tdp_unscaled=vr["eq7_unscaled"]["ok"],
                             t_s_pred_h=m["t_s_s"] / 3600, t_s_meas_h=meas["t_s_h"], t_s_RE=ts_re, ok_t_s=pass_ts, q_s_pred=m["q_s"], q_s_meas=meas["q_s_mol_kg"], q_s_RE=qs_re, ok_q_s=pass_qs,
                             ratio_t_s_pred_over_fullregen_pred=ratio_ts, ratio_t_s_meas_over_BT2_meas=meas_ts, ratio_q_s_pred_over_fullregen_pred=ratio_qs, ratio_q_s_meas_over_BT2_meas=meas_qs,
                             t_b5_pred_h=m["t_b_s"] / 3600, frac_water_removed=m["frac_removed"], y_out_max_regen=m["y_out_max_regen"], water_balance_rel=m["water_balance_rel"],
                             total_balance_rel=m["total_balance_rel"], bt_balance_err=m["bt_balance"], t_hold_s=m["t_hold_s"], wall_s=time.time() - t0))
        print(label, f"{time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    # verdicts
    def v(g):
        s = bool(g.ok_Tdp_scaled.all() and g.ok_t_s.all() and g.ok_q_s.all())
        u = bool(g.ok_Tdp_unscaled.all() and g.ok_t_s.all() and g.ok_q_s.all())
        return "PASS" if s and u else ("INCONCLUSIVE (the two dew-point readings differ)" if s != u else "FAIL")
    d["verdict_reading"] = d.reading.map({r: v(d[d.reading == r]) for r in d.reading.unique()})
    d["dlog10_note"] = dlog10_note(d.y_min_pred_umol_mol)
    d.to_csv(RESULTS / "vt6b_l15.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print(d[["reading", "experiment", "y_min_pred_umol_mol", "y_meas_scaled_umol_mol", "dlog10_scaled", "dlog10_unscaled", "t_s_pred_h", "t_s_meas_h", "t_s_RE", "q_s_pred", "q_s_meas", "q_s_RE",
             "frac_water_removed", "y_out_max_regen", "water_balance_rel", "verdict_reading"]].round(4).to_string(index=False))
    print("frozen k", k, "sha ok", h)


def plant():
    from h2su import plantrun as pr
    k, h = frozen_k()
    df, iso = load_registry(), dr.iso_params()
    m_rated = float(pr.stack_tables()["m_rated_kg_h"])
    Lp, Ap, Vp = dr.plant_geometry(df, m_rated)
    reg = lambda n: float(df.loc[df.name == n, "value_or_range"].iloc[0])
    T = 28.5 + 273.15
    P_AD = reg("vpsa_P_AD_opt") * 1e5
    y_in = reg("l15_feed_RH") / 100.0 * dr.psat_bar(T, iso) / (P_AD / 1e5)
    n_h2 = m_rated / 3600.0 / M_H2
    N_tot = n_h2 / (1.0 - y_in)
    pf = reg("vpsa_purge_to_feed") / 100.0
    qsat = float(dr.q_water(y_in * P_AD / 1e5, T, iso))
    t_AD = reg("l15_opt_tAD") * 3600.0
    ads = dict(L_m=Lp, A_m2=Ap, eps=reg("l15_bed_porosity"), rho_bed_kg_m3=reg("l15_bed_packing_density"), Dax_m2_s=0.0, N_flux_mol_m2_s=N_tot / Ap, p_Pa=P_AD, T_K=T,
               yW_in_umol_mol=y_in * 1e6, yW_init_umol_mol=0.0, q_init_mol_kg=0.0, kLDF_1_s=k, t_end_s=t_AD, dt_out_s=30.0)
    rgp = dict(ads)
    rgp.update(p_Pa=reg("vpsa_P_regen") * 1e5, N0_mol_m2_s=pf * N_tot / Ap, y_init=y_in, q_init_mol_kg=qsat)
    rows = []
    for regen_h in (1.5, 2.5):
        for N in (200, 400, 800):
            t = np.arange(0.0, regen_h * 3600.0 + 1e-9, 30.0)
            t0 = time.time()
            rs = rg.regen_solve(rgp, iso, N, t, 1e-6)
            n_purge_mol = rgp["N0_mol_m2_s"] * Ap * regen_h * 3600.0
            purge_kg_mass_balance = n_purge_mol * M_H2
            h2_out = float(rs["T_out"][-1] - rs["W_out"][-1]) * Ap * M_H2          # exact ODE-integrated total minus water outflow (= inlet H2 + change of the gas H2 hold-up)
            water_out_mol = rs["W_out"][-1] * Ap
            q_end = rs["q"][:, -1]
            # time to 5 umol/mol of the regeneration outlet (it never drops below 5 umol/mol while desorbing: report the first crossing downward if any)
            yo = rs["y_out"] * 1e6
            below = np.where(yo <= 5.0)[0]
            t_to_spec_regen = float(rs["t"][below[0]]) if len(below) else float("nan")
            for direction in ("feed", "countercurrent"):
                qp = q_end if direction == "feed" else q_end[::-1]
                y0 = np.vstack([np.zeros(N), qp])
                pa = dict(ads)
                bt = dr.run_case(pa, iso, N, 1e-6, "upwind", y0_profiles=y0)
                tt, yy = bt["t"], bt["y_out"]
                m = tt >= 5.0 * Lp / bt["us"]
                off = yy > 5.0
                t_off = float(np.trapezoid(off[m].astype(float), tt[m]) + (0.0))     # time above 5 umol/mol inside the adsorption window (after the displacement of the initial dry gas)
                nm = offspec_metrics(tt, yy, bt["us"], Lp, t_AD, m_rated)          # Brief 6 item 6: on-spec throughput to breakthrough and timing of the off-spec period
                rows.append(dict(case="dry_su_ii" if True else "", regen_h=regen_h, purge_direction=direction, N=N, q_sat=qsat, frac_water_removed=1.0 - float(q_end.mean()) / qsat,
                                 y_out_max_regen=float(rs["y"].max()), regen_outlet_t_to_5umol_s=t_to_spec_regen, purge_H2_kg_mass_balance=purge_kg_mass_balance, purge_H2_kg_outlet_integrated=h2_out,
                                 water_removed_mol=water_out_mol, water_balance_rel=float(rs["water_balance_rel"].max()), total_balance_rel=float(rs["total_balance_rel"].max()),
                                 y_min_adsorption_umol_mol=float(yy[m].min()), t_b5_adsorption_s=dr.first_crossing(tt, yy, 5.0), t_offspec_in_AD_window_s=t_off, t_AD_window_s=t_AD,
                                 offspec_fraction_of_window=t_off / t_AD, H2_diverted_to_spec_at_rated_flow_kg=m_rated / 3600.0 * t_off, on_spec_throughput_to_breakthrough_kg=nm["on_spec_throughput_to_breakthrough_kg"],
                                 offspec_timing=nm["offspec_timing"], offspec_first_s=nm["offspec_first_s"], offspec_last_s=nm["offspec_last_s"], wall_s=time.time() - t0, bt_balance_err=float(bt["sol"]["balance_err"].max())))
            print(regen_h, N, f"{time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    d["label"] = "plant bed (L15 scale-up rule) at the engine rated H2 flow; non-trace regeneration form D53; bound, not a result"
    d.to_csv(RESULTS / "ver6_dryer_nontrace.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print(d.drop(columns=["label", "wall_s"]).round(5).to_string(index=False))
    # grid check N 200 -> 400 -> 800
    for c in ("purge_H2_kg_mass_balance", "water_removed_mol", "H2_diverted_to_spec_at_rated_flow_kg", "t_b5_adsorption_s", "frac_water_removed"):
        for n0, n1 in ((200, 400), (400, 800)):
            a = d[d.N == n0].set_index(["regen_h", "purge_direction"])[c]
            b = d[d.N == n1].set_index(["regen_h", "purge_direction"])[c]
            print(f"N{n0}->N{n1} max relative change", c, float(((b - a).abs() / a.abs().clip(lower=1e-30)).max()))
    # column (iii) of results/sens/dryer_event_bound.csv: written by `python code/dryer_gridband.py bound` (Brief 6 item 6; per-event kg first, annual pp for POL-A only)


if __name__ == "__main__":
    {"defcheck": defcheck, "vt6b": vt6b, "plant": plant, "note": note_stored}[sys.argv[1]]()

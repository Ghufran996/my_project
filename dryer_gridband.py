"""Brief 6 item 6 (D59): dryer time-to-spec grid band and the per-event / annual dryer bounds.

    python code/dryer_gridband.py one N [--regen 1.5] [--dir countercurrent]   one plant-bed case (regeneration + adsorption step) -> results/ver6/gridband/<dir>_<h>h_N<N>.json
    python code/dryer_gridband.py summary                                       observed order, Richardson value, GCI (Celik) from N = 800, 1600, 3200 -> results/ver6_dryer_gridband.csv, ..._gci.csv
    python code/dryer_gridband.py bound                                         results/sens/dryer_event_bound.csv (per-event kg first; annual pp for POL-A only, capped at M_stack)

Case (identical to vt6b_run.plant(), D53/D59): plant bed at the engine rated H2 flow (L15 scale-up rule), regeneration 25 mbar, dry H2 purge at 1 % of the feed molar flow, 1.5 h, then
adsorption at 7.8 bar for the 2.95 h window; non-trace regeneration form with the frozen k_LDF; upwind finite volumes, rtol 1e-6; countercurrent purge = the q profile of the regeneration
reversed before the adsorption step. Nothing is fitted. Quantities per case: time to the first crossing of 5 umol/mol (t_b5), off-spec time inside the window (existing definition: outlet above
5 umol/mol for t >= 5 L/u_s on the 30-s output grid), H2 diverted at the rated flow, on-spec throughput to breakthrough (rated H2 flow x t_b5), timing of the off-spec period.
The GCI uses the Celik et al. procedure with a constant refinement ratio r = 2 (N = 800, 1600, 3200); if the three values are not monotone the range is reported and no order is claimed.
These quantities are bounds on a model that did not pass VT6b; no value feeds lambda_SU.
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
from h2su.io import RESULTS, load_registry, param  # noqa: E402

GB = RESULTS / "ver6" / "gridband"
GRIDS = (800, 1600, 3200)
EPS_ORDER = 1e-12


def setup():
    from h2su import plantrun as pr
    from vt6b_run import frozen_k          # lazy: vt6b_run imports offspec_metrics from this module
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
    return dict(iso=iso, m_rated=m_rated, Lp=Lp, Ap=Ap, ads=ads, rgp=rgp, qsat=qsat, t_AD=t_AD, N_tot=N_tot, k=k)


def offspec_metrics(tt, yy, us, Lp, t_AD, m_rated):
    """Shared by vt6b_run.plant() and this script. yy in umol/mol. Returns the new Brief 6 columns."""
    t_hold = 5.0 * Lp / us
    m = tt >= t_hold
    off = yy > 5.0
    t_off = float(np.trapezoid(off[m].astype(float), tt[m]))
    tb5 = dr.first_crossing(tt, yy, 5.0)
    rate = m_rated / 3600.0
    if not np.any(off[m]):
        timing, t_first, t_last = "none (on-spec through the whole window)", float("nan"), float("nan")
        on_spec = rate * t_AD
    else:
        idx = np.where(off & m)[0]
        t_first, t_last = float(tt[idx[0]]), float(tt[idx[-1]])
        on_spec = rate * (tb5 if np.isfinite(tb5) else 0.0)
        if np.isfinite(tb5) and tb5 < t_hold:
            timing = "start of the adsorption step (outlet above 5 umol/mol from the first residence times to the end of the window)" if t_last >= tt[-1] - 60.0 else "start of the adsorption step"
        elif t_last >= tt[-1] - 60.0:
            timing = "end of the adsorption step (breakthrough, then above 5 umol/mol to the end of the window)"
        else:
            timing = "intermediate (off-spec period ends before the end of the window)"
    return dict(on_spec_throughput_to_breakthrough_kg=on_spec, offspec_timing=timing, offspec_first_s=t_first, offspec_last_s=t_last, t_hold_s=t_hold, t_offspec_in_AD_window_s=t_off,
                t_b5_adsorption_s=tb5, offspec_from_crossing_s=(t_AD - tb5) if np.isfinite(tb5) else float("nan"))


def one(N: int, regen_h: float = 1.5, direction: str = "countercurrent"):
    S = setup()
    iso, rgp, ads = S["iso"], S["rgp"], S["ads"]
    t = np.arange(0.0, regen_h * 3600.0 + 1e-9, 30.0)
    t0 = time.time()
    rs = rg.regen_solve(rgp, iso, N, t, 1e-6)
    w_reg = time.time() - t0
    q_end = rs["q"][:, -1]
    qp = q_end if direction == "feed" else q_end[::-1]
    t1 = time.time()
    bt = dr.run_case(dict(ads), iso, N, 1e-6, "upwind", y0_profiles=np.vstack([np.zeros(N), qp]))
    w_ads = time.time() - t1
    tt, yy = bt["t"], bt["y_out"]
    m = offspec_metrics(tt, yy, bt["us"], S["Lp"], S["t_AD"], S["m_rated"])
    n_purge_mol = rgp["N0_mol_m2_s"] * S["Ap"] * regen_h * 3600.0
    res = dict(case="dry_su_ii", regen_h=regen_h, purge_direction=direction, N=N, frac_water_removed=1.0 - float(q_end.mean()) / S["qsat"], purge_H2_kg_mass_balance=n_purge_mol * M_H2,
               y_min_adsorption_umol_mol=float(yy[tt >= m["t_hold_s"]].min()), t_AD_window_s=S["t_AD"], H2_diverted_to_spec_at_rated_flow_kg=S["m_rated"] / 3600.0 * m["t_offspec_in_AD_window_s"],
               H2_diverted_from_crossing_kg=S["m_rated"] / 3600.0 * m["offspec_from_crossing_s"] if np.isfinite(m["offspec_from_crossing_s"]) else float("nan"),
               water_balance_rel=float(rs["water_balance_rel"].max()), total_balance_rel=float(rs["total_balance_rel"].max()), bt_balance_err=float(bt["sol"]["balance_err"].max()),
               wall_regen_s=w_reg, wall_ads_s=w_ads, wall_s=w_reg + w_ads, **m)
    GB.mkdir(parents=True, exist_ok=True)
    (GB / f"{direction}_{regen_h}h_N{N}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


def gci(f800: float, f1600: float, f3200: float, r: float = 2.0):
    """Celik et al. (2008) with constant r: f1 = finest (3200), f2 = 1600, f3 = 800. Returns a dict; monotone only if eps32 / eps21 > 0 (same sign, nonzero)."""
    e21, e32 = f1600 - f3200, f800 - f1600
    out = dict(f_800=f800, f_1600=f1600, f_3200=f3200, eps_1600_3200=e21, eps_800_1600=e32, range_min=min(f800, f1600, f3200), range_max=max(f800, f1600, f3200))
    if not np.all(np.isfinite([f800, f1600, f3200])):
        out.update(monotone=False, note="non-finite value")
        return out
    if abs(e21) < EPS_ORDER and abs(e32) < EPS_ORDER:
        out.update(monotone=True, p_observed=float("nan"), f_richardson=f3200, gci_fine_pct=0.0, note="identical on all three grids")
        return out
    ratio = e32 / e21 if abs(e21) > EPS_ORDER else float("nan")
    if not np.isfinite(ratio) or ratio <= 0.0:
        out.update(monotone=False, p_observed=float("nan"), f_richardson=float("nan"), gci_fine_pct=float("nan"), note="not monotone: range reported, no order claimed")
        return out
    p = float(np.log(ratio) / np.log(r))
    f_ext = f3200 + (f3200 - f1600) / (r ** p - 1.0)
    gci_fine = 1.25 * abs(e21 / f3200) / (r ** p - 1.0) * 100.0 if f3200 != 0 else float("nan")
    out.update(monotone=True, p_observed=p, f_richardson=float(f_ext), gci_fine_pct=float(gci_fine), note="monotone; Celik GCI with Fs = 1.25")
    return out


def summary():
    rows = []
    for N in GRIDS:
        f = GB / f"countercurrent_1.5h_N{N}.json"
        if f.exists():
            rows.append(json.loads(f.read_text(encoding="utf-8")))
    d = pd.DataFrame(rows)
    d["label"] = "plant bed, countercurrent purge, 1.5 h regeneration; non-trace form (D53); bound, model not validated (VT6b FAIL); no value feeds lambda_SU"
    d.to_csv(RESULTS / "ver6_dryer_gridband.csv", index=False, float_format="%.8g")
    print(d[["N", "wall_regen_s", "wall_ads_s", "t_b5_adsorption_s", "t_offspec_in_AD_window_s", "offspec_from_crossing_s", "H2_diverted_to_spec_at_rated_flow_kg", "on_spec_throughput_to_breakthrough_kg",
             "offspec_timing"]].to_string(index=False))
    if set(GRIDS) <= set(d.N):
        g = d.set_index("N")
        out = []
        for q in ("t_b5_adsorption_s", "t_offspec_in_AD_window_s", "offspec_from_crossing_s", "H2_diverted_to_spec_at_rated_flow_kg", "H2_diverted_from_crossing_kg", "on_spec_throughput_to_breakthrough_kg",
                  "frac_water_removed", "y_min_adsorption_umol_mol"):
            r = gci(float(g.loc[800, q]), float(g.loc[1600, q]), float(g.loc[3200, q]))
            r["quantity"] = q
            out.append(r)
        # the earlier grids N = 200, 400, 800 of ver6_dryer_nontrace.csv (same case), for context: refinement ratio 2, order from the three coarser grids
        nt = pd.read_csv(RESULTS / "ver6_dryer_nontrace.csv")
        nt = nt[(nt.regen_h == 1.5) & (nt.purge_direction == "countercurrent")].set_index("N")
        ctx = []
        for q in ("t_b5_adsorption_s", "t_offspec_in_AD_window_s", "H2_diverted_to_spec_at_rated_flow_kg"):
            if q in nt.columns and {200, 400, 800} <= set(nt.index):
                r = gci(float(nt.loc[200, q]), float(nt.loc[400, q]), float(nt.loc[800, q]))
                r["quantity"] = q + " (coarse triple N = 200, 400, 800; columns f_800/f_1600/f_3200 hold N = 200/400/800)"
                ctx.append(r)
        o = pd.DataFrame(out + ctx)
        o = o[["quantity"] + [c for c in o.columns if c != "quantity"]]
        o.to_csv(RESULTS / "ver6_dryer_gridband_gci.csv", index=False, float_format="%.8g")
        print(o.to_string(index=False))
        return d, o
    return d, None


def bound():
    """Per-event kg first (rows with scope = per_event), then annual pp for POL-A only (scope = annual), capped at M_stack."""
    ev = pd.read_csv(RESULTS / "ver6" / "ver6_events.csv")
    rep = float(ev[(ev.case_id == "dry_su_i") & (ev.model == "python_mol") & (ev.grid_id == "N400") & (ev.rtol == 1e-6)].m_H2_repress_kg.iloc[0])
    S = setup()
    mr = S["m_rated"]
    pf = param("vpsa_purge_to_feed") / 100.0
    nt = pd.read_csv(RESULTS / "ver6_dryer_nontrace.csv")
    gb = pd.read_csv(RESULTS / "ver6_dryer_gridband.csv")
    gci_f = RESULTS / "ver6_dryer_gridband_gci.csv"
    gc = pd.read_csv(gci_f) if gci_f.exists() else pd.DataFrame()
    LAB_I = "bound (outside the engine, D53/D59); not in lambda_SU"
    LAB_II = "bound (outside the engine, D53/D59); not in lambda_SU"
    LAB_CC = "bound, not validated (VT6b FAIL), grid band in results/ver6_dryer_gridband_gci.csv; upper envelope, one interrupted regeneration per completed restart; not in lambda_SU"
    LAB_FEED = "WITHDRAWN as a bound: contradicted by VT6b; values > 100 pp violate the physical bound (D59); failure record only"
    per = [dict(scope="per_event", bound="i_repressurisation", purge_direction="", regen_h=np.nan, grid="N400 (dry_su_i)", kg_per_event=rep, label=LAB_I,
                description="re-pressurisation H2 per completed restart (dry_su_i, plant bed, results/ver6/ver6_events.csv)"),
           dict(scope="per_event", bound="ii_extra_regeneration_1.5h", purge_direction="", regen_h=1.5, grid="mass balance", kg_per_event=pf * mr * 1.5, label=LAB_II,
                description="one extra regeneration purge per completed restart: P/F x rated H2 flow x 1.5 h (L15 Table S4 BT7)"),
           dict(scope="per_event", bound="ii_extra_regeneration_2.5h", purge_direction="", regen_h=2.5, grid="mass balance", kg_per_event=pf * mr * 2.5, label=LAB_II,
                description="same, 2.5 h (L15 Table S4 BT6)")]
    for x in gb.sort_values("N").itertuples():
        per.append(dict(scope="per_event", bound="iii_offspec_countercurrent_purge_1.5h", purge_direction="countercurrent", regen_h=1.5, grid=f"N{x.N}", kg_per_event=x.H2_diverted_to_spec_at_rated_flow_kg, label=LAB_CC,
                        description=f"H2 diverted at the rated flow while the dryer outlet exceeds 5 umol/mol in the 2.95 h window after one 1.5 h countercurrent regeneration; off-spec timing: {x.offspec_timing}"))
    for qn, qlab in (("H2_diverted_to_spec_at_rated_flow_kg", "off-spec time on the 30-s output grid (existing definition; quantised to 15-30 s)"), ("H2_diverted_from_crossing_kg", "off-spec time = window minus interpolated first crossing of 5 umol/mol")):
        r = gc[gc.quantity == qn] if len(gc) else gc
        if len(r):
            r = r.iloc[0]
            if bool(r.monotone):
                per.append(dict(scope="per_event", bound="iii_offspec_countercurrent_purge_1.5h", purge_direction="countercurrent", regen_h=1.5, grid=f"Richardson (N = 800, 1600, 3200); {qlab}", kg_per_event=float(r.f_richardson), label=LAB_CC,
                                description=f"Richardson value; observed order {r.p_observed:.3f}; GCI of the fine grid {r.gci_fine_pct:.2f} %; range over the three grids {r.range_min:.4g} to {r.range_max:.4g} kg"))
            else:
                per.append(dict(scope="per_event", bound="iii_offspec_countercurrent_purge_1.5h", purge_direction="countercurrent", regen_h=1.5, grid=f"range over N = 800, 1600, 3200; {qlab}", kg_per_event=float("nan"), label=LAB_CC,
                                description=f"not monotone; range {r.range_min:.4g} to {r.range_max:.4g} kg per event"))
    for x in nt[(nt.N == 800)].sort_values(["purge_direction", "regen_h"]).itertuples():
        if x.purge_direction == "countercurrent" and x.regen_h == 1.5:
            continue                                                           # covered by the grid band above
        lab = LAB_FEED if x.purge_direction == "feed" else (LAB_CC if x.regen_h == 1.5 else "bound, not validated (VT6b FAIL); 2.5 h countercurrent: outlet stays below 5 umol/mol; not in lambda_SU")
        per.append(dict(scope="per_event", bound=f"iii_offspec_{x.purge_direction}_purge_{x.regen_h}h", purge_direction=x.purge_direction, regen_h=x.regen_h, grid="N800", kg_per_event=x.H2_diverted_to_spec_at_rated_flow_kg,
                        label=lab, description="H2 diverted at the rated flow while the dryer outlet exceeds 5 umol/mol inside the 2.95 h window, bed state after one regeneration of the stated duration"))
    P = pd.DataFrame(per)
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    f = ann[(ann.model == "full") & (ann.policy == "A")]
    rows = []
    for r in f.itertuples():
        for p in per:
            kge = p["kg_per_event"]
            if not np.isfinite(kge):
                continue
            kg_raw = kge * r.N_r_completed
            kg = min(kg_raw, r.M_stack)
            lab = p["label"]
            if p["bound"].startswith("ii_"):
                lab = "upper envelope, one interrupted regeneration per completed restart; " + lab
            rows.append(dict(scope="annual", site=r.site, year=r.year, policy="A", N_r_completed=r.N_r_completed, M_stack_kg=r.M_stack, lambda_SU_pct=r.lambda_SU, bound=p["bound"], purge_direction=p["purge_direction"],
                             regen_h=p["regen_h"], grid=p["grid"], kg_per_event=kge, kg_per_year_uncapped=kg_raw, kg_per_year=kg, capped_at_M_stack=bool(kg_raw > r.M_stack), pp_of_lambda_SU=kg / r.M_stack * 100.0,
                             label=lab, description=p["description"]))
    A = pd.DataFrame(rows)
    out = pd.concat([P, A], ignore_index=True)
    out.to_csv(RESULTS / "sens" / "dryer_event_bound.csv", index=False, float_format="%.8g")
    print(P[["bound", "purge_direction", "regen_h", "grid", "kg_per_event"]].to_string(index=False))
    print(A.groupby(["bound", "grid"]).pp_of_lambda_SU.mean().round(3).to_string())
    return out


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "one":
        one(int(a[2]), float(a[a.index("--regen") + 1]) if "--regen" in a else 1.5, a[a.index("--dir") + 1] if "--dir" in a else "countercurrent")
    elif a[1] == "summary":
        summary()
    elif a[1] == "bound":
        bound()

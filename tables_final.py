"""Brief 5 item 10: final tables T1-T6 as CSV in results/tables/ (status DRAFT until the analyst signs).

    python code/tables_final.py

T1 parameters with source, location and role (registry rows referenced by code); T2 validation (one row per test with verdict; raw rows in T2_supplementary.csv);
T3 scenarios; T4 lambda_SU, tau_eq, eta_del, beta with 90 % intervals and SC-M class (D60 RR10 gate: written only for a site x policy with 5 seeds, all site-years and closure <= 1e-6, otherwise the literal PENDING; B-S2 beta cells numeric from beta_intervals.csv, label "M_del,B-S2 at nominal; interval from the full-model denominator only", D67b);
T5 baselines; T6 comparison table exported from .work/literature.md (block "Comparison table, judge TASK B after Brief 4") with the Brief 5 values in the last row.
Verdict wording of T2 follows the analyst's gate table (D45/D56) and the result files named in the column `source_file`; counts are computed from those files where a file holds them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2su.io import CODE_DIR, ROOT, RESULTS, load_registry  # noqa: E402

T = RESULTS / "tables"
UQ = RESULTS / "uq"


def t1():
    reg = load_registry()
    names = set()
    for f in list(CODE_DIR.glob("*.py")) + list((CODE_DIR / "src" / "h2su").rglob("*.py")):
        names |= set(re.findall(r'param\(\s*"([A-Za-z0-9_]+)"\s*\)', f.read_text(encoding="utf-8", errors="ignore")))
        names |= set(re.findall(r'reg_value\(\s*df\s*,\s*"([A-Za-z0-9_]+)"\s*\)', f.read_text(encoding="utf-8", errors="ignore")))
        names |= set(re.findall(r'g\("([A-Za-z0-9_]+)"\)', f.read_text(encoding="utf-8", errors="ignore")))
        names |= set(re.findall(r'"(l15_[A-Za-z0-9_]+|l15si_[A-Za-z0-9_]+|vpsa_[A-Za-z0-9_]+|awe_[A-Za-z0-9_]+|pem[0-9]+_[A-Za-z0-9_]+)"', f.read_text(encoding="utf-8", errors="ignore")))
    d = reg[reg.name.isin(names)][["name", "symbol", "value_or_range", "unit", "technology", "source_id", "location", "role", "note"]].copy()
    inv = pd.read_csv(RESULTS / "inventory_ranges.csv")
    extra = pd.DataFrame([dict(name="gas inventory i (nominal / ends)", symbol="i", value_or_range="0.699 nominal; PEM ends 0.120-0.888; AWE pilot scenario 3.178", unit="min of rated production per bar", technology="PEM (AWE labelled scenario)",
                               source_id="L13 / L32 / L28", location="results/inventory_ranges.csv", role="INPUT/UQ", note=""),
                          dict(name="N2 washout theta_spec", symbol="theta_spec", value_or_range="6.5 nominal; scenarios 0-100 (falsification grid; < 6.5 'bounding, no source'; 100 'AWE-derived, L14 abstract')", unit="rated-minutes",
                               technology="PEM", source_id="L28 (procedural purge-to-spec assumption); L14 abstract", location="results/vt3.csv; results/sens/falsification_grid.csv", role="INPUT/SCENARIO", note="D47")])
    d = pd.concat([d, extra], ignore_index=True)
    d.to_csv(T / "T1_parameters_sources.csv", index=False)
    return len(d)


def read(n):
    return pd.read_csv(RESULTS / n)


def t2():
    rows, sup = [], []
    v1 = read("vt1_l26.csv")
    p1 = v1[v1.variant == "primary"]
    rows.append(dict(id="VT1", scope="PEM thermal start-up (L26 heat-up times)", n_points=len(p1), result=f"{int(p1.abs_RE_le_20pct.sum())}/{len(p1)} within 20 % but CAL infeasible (loss-free bound)", verdict="NOT PASSED (diagnostic only)", source_file="results/vt1_l26.csv", note="D31"))
    v2 = read("vt2_l28.csv")
    rows.append(dict(id="VT2", scope="AWE-pilot pressurisation time (L28)", n_points=len(v2), result=f"{int(v2.abs_RE_le_20pct.sum())}/{len(v2)} within 20 % (RE range {v2.RE.min():+.0%}...{v2.RE.max():+.0%})" if "RE" in v2 else "", verdict="FAIL", source_file="results/vt2_l28.csv", note="D45"))
    v2b = read("vt2b_l32.csv")
    pb = v2b[v2b.row_type == "primary"]
    rows.append(dict(id="VT2b", scope="PEM pressurisation (L32)", n_points=len(pb), result=f"{int(pb.abs_RE_le_20pct.sum())}/{len(pb)} within 20 % (RE {pb.RE.min():+.0%}...{pb.RE.max():+.0%})", verdict="FAIL (disclosed foreknowledge)", source_file="results/vt2b_l32.csv", note="D45"))
    rows.append(dict(id="VT3", scope="purge and N2 washout to spec", n_points=0, result="not executable (fixed procedural purge)", verdict="NOT EXECUTABLE", source_file="results/vt3.csv", note="D45"))
    v4 = read("vt4.csv")
    rows.append(dict(id="VT4", scope="O2 in H2 vs current (L24)", n_points=int(v4.delta_log10.notna().sum()), result="validation point is a start-up transient peak", verdict="FAIL (L25 points not executable)", source_file="results/vt4.csv", note="not relevant for lambda_SU (D40 #7)"))
    v5 = read("vt5a_l1.csv")
    rows.append(dict(id="VT5a", scope="steady PSA purge share vs load (L1 Table 2)", n_points=len(v5), result=f"{int(v5.abs_RE_le_20pct.sum())}/{len(v5)} within 20 %", verdict="PASS (consistency, provenance unconfirmed)", source_file="results/vt5a_l1.csv", note="D45"))
    rows.append(dict(id="VT5b", scope="dryer purge/recovery chain", n_points=0, result="not executed", verdict="BLOCKED (SN17)", source_file="-", note="D45"))
    v6 = read("vt6_l15.csv")
    v6v = v6[v6.role != "CAL"]
    rows.append(dict(id="VT6", scope="dryer breakthrough time, frozen k_LDF (L15)", n_points=len(v6v), result=f"{int(v6v.verdict.str.startswith('PASS').sum())}/{len(v6v)} within 20 %; max |RE| {v6v.RE.abs().max():.1%}", verdict="PASS", source_file="results/vt6_l15.csv", note="additional test, not a component of lambda_SU drivers"))
    b = read("vt6b_l15.csv")
    bp = b[b.reading == "primary"]
    rows.append(dict(id="VT6b", scope="dryer partial regeneration BT6/BT7, non-trace form (D53), no fitting", n_points=2, result=f"outlet mole fraction RE: dlog10 {bp.dlog10_scaled.round(2).tolist()}; t_s RE {bp.t_s_RE.round(2).tolist()}; q_s RE {bp.q_s_RE.round(2).tolist()}",
                     verdict="FAIL (all readings; scaled and unscaled dew-point readings agree)", source_file="results/vt6b_l15.csv", note="t_s threshold is an implementer reading; even fully regenerated beds miss t_s (results/vt6b_definitions_check.csv)"))
    sc = read("scb_plant.csv")
    rows.append(dict(id="SC-B / VER-4 / VER-5", scope="plant engine closure, limiting cases, step independence", n_points=int(len(sc)), result=f"{int(sc.passed.sum())}/{len(sc)} checks passed", verdict="PASS", source_file="results/scb_plant.csv", note=""))
    r2 = read("ver2r_c3.csv")
    rows.append(dict(id="VER-2R", scope="C3 (L4) Tables 3 and 5 re-run on AEMO 2022", n_points=len(r2), result=f"{int((r2.verdict == 'PASS').sum())}/{len(r2)} PASS", verdict="PASS", source_file="results/ver2r_c3.csv", note="SC-R scope: dispatch, aggregation, accounting"))
    t3log = (RESULTS / "smoke" / "run_all_log.txt").read_text(encoding="utf-8", errors="ignore")
    m3 = re.findall(r"(\d+) passed", t3log)
    n3 = int(m3[-1]) if m3 else 0
    rows.append(dict(id="VER-3", scope="analytic checks of the components (washout CSTR chain vs Erlang, molar volume, pressurisation, deoxo, ISO grade, closure detection)", n_points=n3, result=f"{n3}/10 pytest checks pass (R8)",
                     verdict="PASS" if n3 == 10 else f"CHECK: {n3}/10", source_file="results/smoke/run_all_log.txt", note="verification, not validation"))
    rows.append(dict(id="VER-1 / VER-1b", scope="L26 heat-up (Table 5) and L13 Eq.16 Faradaic efficiency", n_points=0, result="not reproduced from printed information (SN01-SN08)", verdict="FAILED (documented attempts)", source_file="results/ver1_rauls.csv; results/ver1b_persson.csv", note="D23"))
    dv = read("ver6_dryer_verdict.csv")
    rows.append(dict(id="VER-6 dryer", scope="1-D reference, grid/tolerance convergence, lumped forms", n_points=len(dv), result=f"{int(dv.verdict.astype(str).str.startswith('PASS').sum())}/{len(dv)} rows PASS", verdict="PARTIAL (dry_regen trace form FAIL; replaced by results/ver6_dryer_nontrace.csv)", source_file="results/ver6_dryer_verdict.csv", note="COMSOL dryer comparison non-gating, user run"))
    rows.append(dict(id="VER-6 deoxo", scope="deoxo start-up fidelity", n_points=0, result="not required (<= 0.001 pp)", verdict="NOT REQUIRED (D42)", source_file="results/nominal/ablation.csv", note=""))
    cc = read("ver6c_comsol_compare.csv")
    cv = cc[cc.case_id.str.startswith("ver_")]
    nok = int((cv.status == "OK").sum())
    rows.append(dict(id="VER-6c verification classes", scope="user-run COMSOL code-to-code: plug flow, dispersion Pe 10/100, breakthrough (D38)", n_points=len(cv), result=f"{nok}/{len(cv)} cases OK; outlet NRMSE falls with N for dispersion and breakthrough; plug-flow 0.7-2.3 % without grid trend (scheme-dependent numerical diffusion at a sharp front)",
                     verdict="CONSISTENT", source_file="results/ver6c_comsol_compare.csv", note="non-gating; D38"))
    dd = cc[cc.case_id.isin(["dry_bt2", "dry_su_i"]) & (cc.status == "OK")]
    h = dd.pivot_table(index="case_id", columns="grid_id", values="nrmse_y_H2O_out") * 100
    rows.append(dict(id="VER-6c dryer adsorption", scope="COMSOL vs Python dry_bt2 and dry_su_i (outlet H2O, breakthrough time, mass balance); COMSOL carries the same imposed numerical diffusion (R-C4), so this checks the implementation of isotherm, LDF and balances, not numerical diffusion",
                     n_points=len(dd), result=f"outlet H2O NRMSE (N = 25 / 50) dry_bt2 {h.loc['dry_bt2', 'N025']:.2f} / {h.loc['dry_bt2', 'N050']:.2f} %, dry_su_i {h.loc['dry_su_i', 'N025']:.2f} / {h.loc['dry_su_i', 'N050']:.2f} %; |t_break difference| {dd.diff_t_break_s.abs().min():.3f}-{dd.diff_t_break_s.abs().max():.3f} s; COMSOL mass-balance error <= {dd.comsol_mass_balance_err.max():.1e}",
                     verdict="CONSISTENT", source_file="results/ver6c_comsol_compare.csv", note="non-gating; implementation check only (D58)"))
    rows.append(dict(id="VER-6c dry_regen and dry_su_ii", scope="COMSOL regeneration and post-regeneration start-up cases", n_points=0, result="failed at every level: trace form invalid (gas water about 64.5 %), same class as R-B4 6e", verdict="NOT COMPARABLE",
                     source_file="results/ver6c_comsol_compare.csv", note="covered by the Python non-trace form (D53, R68, R69), which has internal checks only and no second code (D58)"))
    v8 = read("ver8_pv_pvgis.csv")
    rows.append(dict(id="VER-8", scope="pvlib PV chain vs PVGIS hourly output, 17 site-years (supporting)", n_points=len(v8), result=f"annual DC energy difference {v8.annual_diff_pct.min():+.1f}...{v8.annual_diff_pct.max():+.1f} %; hourly NRMSE up to {v8.NRMSE_pct_of_mean_daytime_pvgis.max():.1f} % of mean daytime PVGIS output",
                     verdict="SUPPORTING (no gate)", source_file="results/ver8_pv_pvgis.csv", note="D24; PV chain without IAM and converter loss (limitation)"))
    nt = pd.read_csv(RESULTS / "ver6_dryer_nontrace.csv")
    n8 = nt[nt.N == 800]
    gb = pd.read_csv(RESULTS / "ver6_dryer_gridband_gci.csv").set_index("quantity")
    h2 = gb.loc["H2_diverted_to_spec_at_rated_flow_kg"]
    h2c = gb.loc["H2_diverted_from_crossing_kg"]
    cc_txt = (f"per-event off-spec {h2.range_min:.2f}-{h2.range_max:.2f} kg over N = 800-3200 (grid band); Richardson {h2.f_richardson:.2f}-{h2c.f_richardson:.2f} kg (two definitions: window and crossing); "
              f"GCI {h2.gci_fine_pct:.1f} % and {h2c.gci_fine_pct:.1f} %")
    for dirn, verdict, note in (("countercurrent", "BOUND, not validated; outside the engine", "D53, D59; grid band and GCI in results/ver6_dryer_gridband_gci.csv (R85); no measurement of countercurrent regeneration"),
                                ("feed", "WITHDRAWN as a bound: contradicted by VT6b; values above 100 pp violate the physical bound (D59)", "D59; raw values kept in T2_supplementary.csv")):
        x = n8[n8.purge_direction == dirn]
        res = f"off-spec fraction of window (N = 800): {x.offspec_fraction_of_window.min():.2f}-{x.offspec_fraction_of_window.max():.2f}"
        if dirn == "countercurrent":
            res += "; " + cc_txt
        rows.append(dict(id=f"Dryer start-up (non-trace), {dirn} purge", scope="plant bed after one regeneration, off-spec time inside the 2.95 h window", n_points=len(x), result=res, verdict=verdict,
                         source_file="results/ver6_dryer_nontrace.csv" + ("; results/ver6_dryer_gridband_gci.csv" if dirn == "countercurrent" else ""), note=note))
    pd.DataFrame(rows).to_csv(T / "T2_validation.csv", index=False)
    # supplementary raw rows
    for nm, f in (("VT1", "vt1_l26.csv"), ("VT2", "vt2_l28.csv"), ("VT2b", "vt2b_l32.csv"), ("VT4", "vt4.csv"), ("VT5a", "vt5a_l1.csv"), ("VT6", "vt6_l15.csv"), ("VT6b", "vt6b_l15.csv"), ("VT6b definitions", "vt6b_definitions_check.csv"), ("Dryer non-trace raw (feed and countercurrent)", "ver6_dryer_nontrace.csv"), ("VER-8", "ver8_pv_pvgis.csv")):
        x = read(f)
        x.insert(0, "validation_test", nm)
        sup.append(x)
    pd.concat(sup, ignore_index=True).to_csv(T / "T2_supplementary.csv", index=False)
    return len(rows)


def t3():
    sy = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    sy = sy[(sy.model == "full") & (sy.policy == "A")]
    rows = [("Plant", "PEM 1 MW stack-level DC rating; auxiliaries from an auxiliary supply (nominal), BoP from the PV bus as sensitivity (L24 constant share; L1 Table 1 laboratory-scale bound)"),
            ("PV", "1.5 MWp (ratio 1.5; scenarios 1.0 and 2.0), BSRN 1-min GHI/DNI/DHI via the pvlib chain"),
            ("Sites and years", f"Cabauw {int((sy.site == 'Cabauw').sum())}, Carpentras {int((sy.site == 'Carpentras').sum())}, Gobabeb {int((sy.site == 'Gobabeb').sum())} site-years (2013-2018, 95 % screen); {len(sy)} in total"),
            ("Policies", "POL-A depressurise and inert at every stop; POL-B 15/30/60/120 min hot standby then A; POL-C standby through daylight stops; POL-D 24 h hot standby; POL-P bounding (retained inventory, cooling, no source)"),
            ("Input resolution", "1-min reference; 5/15/30/60-min block means held on the 1-min grid"),
            ("Controller", "t_deb 1/5/15 min; abort-hold variant"),
            ("Falsification grid", "theta_spec {0,1,2,3.25,6.5,13,25,50,100} rated-min x inventory {0.120, 0.699} min/bar x P_OP {12, 15.8, 30} bar, POL-A and POL-C"),
            ("Dryer steady term", "s(L) nominal; constant 1.02 % (L15); constant 1.6 % (L14)"),
            ("Dryer start-up", "outside the engine: re-pressurisation, extra regeneration purge and off-spec bounds (D53)"),
            ("UQ", "10 epistemic factors (PV ratio fixed 1.5; design factor in the Sobol rows), N = 1024 scrambled Sobol sequence x 5 seeds (11/22/33/44/55); Saltelli N = 1024 x 5 seeds for POL-A, POL-B60, POL-C at the reference year"),
            ("Technology", "PEM; AWE procedure values only as labelled sensitivities")]
    pd.DataFrame(rows, columns=["item", "value"]).to_csv(T / "T3_sites_scenarios.csv", index=False)


PENDING = "PENDING"
NOTE_T4 = ("tau_eq median = 0 means that more than half of the completed restarts of the sample are returns from hot standby with no excess loss; it does not mean zero start-up loss; use tau_mean. "
           "Nominal P_OP equals the upper bound of its range and nominal P_MIN_F the lower bound of its range.")
NOTE_T4_HIGH = " The nominal lambda_SU lies high in the interval."
NOTE_T4_NOTHIGH = " The nominal lambda_SU is not high in the interval; see nominal_percentile_in_ensemble."
NOTE_T4_CABAUW_D = " P95 at the 0.5 % threshold: pooled 0.5008 %, per seed 0.4997-0.5021 %. The first cycle of each simulated year contributes 0.004-0.005 pp, more than the margin to the threshold."
NOTE_T4_GOBABEB_D = " Nominal 0.00236 % is below P5 0.00244 %; the value is the first cycle of each simulated year."


def t4_note(site, pol):
    if pol == "D":
        extra = NOTE_T4_CABAUW_D if site == "Cabauw" else (NOTE_T4_GOBABEB_D if site == "Gobabeb" else "")
        return NOTE_T4 + NOTE_T4_NOTHIGH + extra
    return NOTE_T4 + NOTE_T4_HIGH


SEEDS_FINAL = (11, 22, 33, 44, 55)     # = sobol_pipeline.SEEDS (design.seeds)
CLOSURE_TOL = 1e-6


def expected_years():
    import yaml
    y = yaml.safe_load((CODE_DIR / "configs" / "sites.yaml").read_text())
    return {s: list(v["years_selected"]) for s, v in y["sites"].items()}


def set_complete(site, pol, years, uq=None, seeds=SEEDS_FINAL, tol=CLOSURE_TOL):
    """Integrity gate (D60 RR10): True only if every seed x site-year checkpoint of the site x policy exists, loads, and has closure <= tol."""
    uq = UQ if uq is None else Path(uq)
    for y in years:
        for sd in seeds:
            f = uq / f"ens_{site}_{y}_{pol}_seed{sd}.npz"
            if not f.exists():
                return False
            try:
                z = np.load(f, allow_pickle=True)
                cols = [str(c) for c in z["cols"]]
                Y = z["Y"]
                for c in ("closure_H2_rel", "closure_E_rel"):
                    v = Y[:, cols.index(c)]
                    if not np.all(np.isfinite(v)) or np.nanmax(np.abs(v)) > tol:
                        return False
            except Exception:
                return False
    return True


def falsification_cols():
    """lambda_SU at i = 0.120 min/bar, P_OP = 30 bar, theta = 6.5 rated-min (D60 RR5): POL-A and POL-C from falsification_grid.csv, POL-B60 and POL-D from sens/rr5_b60_d.csv (Brief 7 item 4b); site-year mean, min, max."""
    g = read(Path("sens") / "falsification_grid.csv")
    x = g[(g.i_req == 0.12) & (g.p_op_req == 30) & (g.theta_req == 6.5)][["site", "year", "policy", "lambda_SU"]]
    f2 = RESULTS / "sens" / "rr5_b60_d.csv"
    if f2.exists():
        x = pd.concat([x, read(Path("sens") / "rr5_b60_d.csv")[["site", "year", "policy", "lambda_SU"]]], ignore_index=True)
    r = x.groupby(["site", "policy"]).lambda_SU.agg(["mean", "min", "max", "count"]).reset_index()
    r.columns = ["site", "policy", "lambda_SU_i0.120_30bar_theta6.5_mean", "lambda_SU_i0.120_30bar_theta6.5_min_year", "lambda_SU_i0.120_30bar_theta6.5_max_year", "n_site_years_grid"]
    return r


BASES = ["B-CL", "B-EP", "B-S1", "B-S2 curve1", "B-S2 curve2", "B-S2 curve3", "B-S3"]


def t4_build(ann, uq=None, expected=None, fals=None, b60=None):
    """Return the T4 frame. Interval, class and beta cells hold numbers only for a site x policy whose checkpoint set is complete (5 seeds, all site-years, closure <= 1e-6); otherwise the literal PENDING."""
    uq = UQ if uq is None else Path(uq)
    expected = expected_years() if expected is None else expected
    f = ann[ann.model == "full"].copy()
    nom = f.groupby(["site", "policy"]).agg(lambda_SU_nominal_mean=("lambda_SU", "mean"), lambda_SU_nominal_min_year=("lambda_SU", "min"), lambda_SU_nominal_max_year=("lambda_SU", "max"),
                                          tau_eq_median_nominal=("tau_eq_median", "mean"), eta_del_nominal=("eta_del", "mean"), beta_B_CL=("beta_B_CL_1min", "mean"), beta_B_EP=("beta_B_EP", "mean"),
                                          beta_B_S1=("beta_B_S1", "mean"), beta_B_S3=("beta_B_S3", "mean"), beta_B_S2_curve1=("beta_B_S2_curve1", "mean"), beta_B_S2_curve2=("beta_B_S2_curve2", "mean"),
                                          beta_B_S2_curve3=("beta_B_S2_curve3", "mean"), n_site_years=("year", "count")).reset_index()
    nom["policy"] = nom.policy.replace({"B": "B60"})
    if b60 is not None:           # POL-B60 nominal cells from nominal/b60_nominal.csv (polB_pareto.csv variant POL-B_60 + B-S3 direct runs, Brief 7 item 4a)
        b60 = b60.groupby("site").agg(lambda_SU_nominal_mean=("lambda_SU", "mean"), lambda_SU_nominal_min_year=("lambda_SU", "min"), lambda_SU_nominal_max_year=("lambda_SU", "max"),
                                      tau_eq_median_nominal=("tau_eq_median", "mean"), eta_del_nominal=("eta_del", "mean"), beta_B_CL=("beta_B_CL", "mean"), beta_B_EP=("beta_B_EP", "mean"),
                                      beta_B_S1=("beta_B_S1", "mean"), beta_B_S3=("beta_B_S3", "mean"), beta_B_S2_curve1=("beta_B_S2_curve1", "mean"), beta_B_S2_curve2=("beta_B_S2_curve2", "mean"),
                                      beta_B_S2_curve3=("beta_B_S2_curve3", "mean"), n_site_years=("year", "count")).reset_index()
        b60["policy"] = "B60"
    else:
        b60 = pd.DataFrame([dict(site=s_, policy="B60", n_site_years=len(y_)) for s_, y_ in expected.items()])
    nom = pd.concat([nom, b60], ignore_index=True)
    nom["policy"] = pd.Categorical(nom.policy, ["A", "B60", "C", "D"], ordered=True)
    nom = nom.sort_values(["site", "policy"]).reset_index(drop=True)
    nom["policy"] = nom.policy.astype(str)
    if fals is not None:
        nom = nom.merge(fals, on=["site", "policy"], how="left")
    icols = [f"{o}_{q}" for o in ("lambda_SU", "tau_eq_median", "tau_mean", "eta_del", "M_del") for q in ("P5", "P50", "P95")] + ["nominal_percentile_in_ensemble"]
    ivp, bp, sm_p = uq / "intervals_pooled.csv", uq / "beta_intervals.csv", uq / "scm_verdict.csv"
    iv = pd.read_csv(ivp) if ivp.exists() else pd.DataFrame()
    if len(iv):
        iv = iv[iv.pv_ratio == 1.5]
    sm = pd.read_csv(sm_p) if sm_p.exists() else pd.DataFrame()
    if len(sm):
        sm = sm[sm.source.astype(str).str.startswith("interval") & (sm.pv_ratio.astype(str) == "1.5")]
    be = pd.read_csv(bp) if bp.exists() else pd.DataFrame()
    out_rows = []
    for _, r in nom.iterrows():
        s, pol = r.site, r.policy
        ok = set_complete(s, pol, expected.get(s, []), uq=uq)
        d = r.to_dict()
        ivr = iv[(iv.site == s) & (iv.policy == pol)] if len(iv) else iv
        final = (ok and len(ivr) > 0 and int(ivr.n_seeds.iloc[0]) == len(SEEDS_FINAL) and int(ivr.n_years.iloc[0]) == len(expected[s])
                 and max(ivr.closure_H2_max.iloc[0], ivr.closure_E_max.iloc[0]) <= CLOSURE_TOL)
        if final:
            for c in icols:
                d[c] = ivr[c].iloc[0] if c in ivr.columns else PENDING
            smr = sm[(sm.site == s) & (sm.policy == pol)] if len(sm) else sm
            d["SC_M_class"] = smr.SC_M_class.iloc[0] if len(smr) else PENDING
            d["interval_status"] = ("final: 5 of 5 seeds, all site-years, closure <= 1e-6; P5-P95 is the 90 % interval; conditions: dryer regeneration state excluded (D59), "
                                    "inventory 0.120-0.888 min/bar, P_OP 15.8-30 bar, washout 6.5 rated-min fixed (D60 RR4)")
            ber = be[(be.site == s) & (be.policy == pol)] if len(be) else be
        else:
            for c in icols + ["SC_M_class"]:
                d[c] = PENDING
            d["interval_status"] = "PENDING: checkpoint set incomplete or closure unchecked (D60 RR10)"
            ber = None
        for b in BASES:
            for q in ("P5", "P50", "P95"):
                c = f"beta_{q}_{b.replace(' ', '')}"
                if ber is None:
                    d[c] = PENDING
                else:
                    v = ber[ber.baseline == b][f"beta_{q}"] if len(ber) else []
                    d[c] = v.iloc[0] if len(v) else PENDING
        d["note"] = t4_note(s, pol)
        out_rows.append(d)
    return pd.DataFrame(out_rows)


def t4():
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    f60 = RESULTS / "nominal" / "b60_nominal.csv"
    out = t4_build(ann, fals=falsification_cols(), b60=pd.read_csv(f60) if f60.exists() else None)
    out.to_csv(T / "T4_lambda_beta_intervals.csv", index=False, float_format="%.5g")
    return len(out)


def beta_unit_text(be, baseline):
    """beta cell definition: estimator, n_seeds per baseline (from beta_intervals.csv) and the B-EP / B-S2 labels."""
    txt = ("% of delivered H2 of the full model; nominal columns = site-year means at the nominal parameters (policy A); P5/P50/P95 = 90 % interval, estimator: per seed all site-years of the site concatenated "
           "(equal year weight), beta per ensemble row, percentiles per seed, pooled value = mean of the per-seed percentiles (D49, RR1); numeric only for a complete, closure-checked set (D60 RR10)")
    if be is not None and len(be):
        k = be[be.baseline == baseline]
        if len(k):
            txt += f"; n_seeds = {int(k.n_seeds.min())}" + ("" if k.n_seeds.min() == k.n_seeds.max() else f"-{int(k.n_seeds.max())}") + f" for {baseline}, n_years = {int(k.n_years.min())}-{int(k.n_years.max())}"
            txt += "; " + str(k.label.iloc[0])
    return txt


def t5_build(ann, t4df, be=None):
    f = ann[ann.model == "full"]
    g = f[f.policy == "A"].groupby("site")[["beta_B_CL_1min", "beta_B_EP", "beta_B_S1", "beta_B_S3", "beta_B_S2_curve1", "beta_B_S2_curve2", "beta_B_S2_curve3"]].mean().round(1)
    desc = [("B-CL", "classical: fixed 75 % LHV stack efficiency, constant 98.4 % recovery, P_min cut-off", "beta_B_CL_1min", "B-CL"), ("B-EP", "engineering practice: PVGIS hourly, constant net SEC 67 kWh/kg, P_min cut-off", "beta_B_EP", "B-EP"),
            ("B-S1", "SOTA (C1 class): steady load-dependent purification loss, instant switching", "beta_B_S1", "B-S1"), ("B-S2 curve 1", "SOTA (C3): efficiency curve 1, DP = MILP, 5-min start-up penalty", "beta_B_S2_curve1", "B-S2curve1"),
            ("B-S2 curve 2", "as above, curve 2", "beta_B_S2_curve2", "B-S2curve2"), ("B-S2 curve 3", "as above, curve 3", "beta_B_S2_curve3", "B-S2curve3"),
            ("B-S3", "SOTA (C4/C2/C5 class): thermal ramp kept, purification off", "beta_B_S3", "B-S3")]
    rows = []
    for a, b, c, key in desc:
        sites = list(g.index)
        r = dict(baseline=a, description=b, reported_or_rerun="re-run on the same input")
        r.update({f"beta_POLA_{s}": g.loc[s, c] for s in sites})
        for s in sites:
            for q in ("P5", "P50", "P95"):
                v = t4df[(t4df.site == s) & (t4df.policy == "A")][f"beta_{q}_{key}"]
                r[f"beta_POLA_{s}_{q}"] = v.iloc[0] if len(v) else PENDING
        r["beta_unit"] = beta_unit_text(be, a)
        rows.append(r)
    return pd.DataFrame(rows)


def t5():
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    t4df = pd.read_csv(T / "T4_lambda_beta_intervals.csv", keep_default_na=False)
    bp = UQ / "beta_intervals.csv"
    t5_build(ann, t4df, pd.read_csv(bp) if bp.exists() else None).to_csv(T / "T5_baselines.csv", index=False)


def t6(lit: Path | None = None):
    lit = ROOT / ".work" / "literature.md" if lit is None else lit
    if not lit.exists():     # Brief 9 item 4: the working notes are not part of a repository copy; keep the stored export
        print(f"T6: {lit.name} absent, stored T6_comparison.csv kept unchanged")
        return pd.read_csv(T / "T6_comparison.csv", keep_default_na=False, dtype=str)
    txt = lit.read_text(encoding="utf-8")
    i = re.search(r"## Comparison table, Gate 2 (FINAL|DRAFT)", txt).start()
    blk = txt[i:].split("\n\n")[0].splitlines()
    tab = [l for l in blk if l.startswith("|")]
    hdr = [c.strip() for c in tab[0].strip("|").split("|")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in tab[2:]]
    df = pd.DataFrame(rows, columns=hdr)
    df.to_csv(T / "T6_comparison.csv", index=False)
    return df


if __name__ == "__main__":
    T.mkdir(parents=True, exist_ok=True)
    print("T1 rows", t1())
    print("T2 rows", t2())
    t3()
    print("T4 rows", t4())
    t5()
    print("T6 rows", len(t6()))
    for old in ("T2_validation_all_rows.csv", "T4_beta_summary.csv"):
        pass

"""Final paper figures (Figures 2-7 of the manuscript). PNG 1000 dpi, 7.48 in wide, one style (configs/paper.mplstyle), Okabe-Ito colours; the raw data of every panel is a CSV in results/figures/data/.
Two stages, kept apart (Brief 13 item 1):
  data  : data_f2 ... data_f9 compute the panel data from the results files and write the data CSVs (unchanged from Brief 5; data_f2 re-runs the engine for one site-year with the trace on).
  draw  : draw_fig2 ... draw_fig7 read ONLY the stored data CSVs (and results/resolution/factorial_2x2.csv for Figure 5C); they run no engine and compute no value beyond what the legend states.

    python code/figures_final.py            # data stage F2-F9, then draw stage (what run_all.py final calls)
    python code/figures_final.py draw       # draw stage only (stored CSVs)
    python code/figures_final.py draw fig3  # a subset of the draw stage
    python code/figures_final.py F3 F5      # a subset of the data stage

Manuscript figure -> file: 2 component_tests, 3 restart_fate_mechanisms, 4 daily_restart_statistics, 5 resolution_and_model_bias, 6 sobol_restart_loss, 7 standby_tradeoff.
No paper title on any figure; no coloured type; red and green never together (pass = blue, outside = vermillion open marker).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import figures_brief4 as fb  # noqa: E402
from h2su import plant as pl  # noqa: E402
from h2su import plantrun as pr  # noqa: E402
from h2su.figures import FULL_W_IN, apply_style, save_png  # noqa: E402
from h2su.io import RESULTS  # noqa: E402
from h2su.pv import pv_dc_series  # noqa: E402
from sweeps import site_years  # noqa: E402

CB, SITES, SCOL = fb.CB, fb.SITES, fb.SCOL
SMARK = fb.SMARK
csv = fb.csv
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#7f7f7f"
POLNAME = {"A": "Full shutdown", "B15": "15-min standby", "B30": "30-min standby", "B60": "60-min standby", "B120": "120-min standby", "C": "Daytime standby", "D": "24-h standby"}


def data_dir() -> Path:
    return fb.DATA


def csv_note(df, name, note):
    """Data CSV with a leading comment line ('# ...') carrying a note that is not drawn on the figure (read back with pandas comment='#')."""
    fb.DATA.mkdir(parents=True, exist_ok=True)
    with open(fb.DATA / f"{name}_data.csv", "w", newline="", encoding="utf-8") as fh:
        fh.write("# " + note.replace("\n", " ") + "\n")
        df.to_csv(fh, index=False, float_format="%.6g", lineterminator="\n")


def read(name, **kw):
    p = data_dir() / f"{name}_data.csv"
    if not p.exists():
        raise SystemExit(f"{p} missing: run 'python code/figures_final.py' (data stage) first")
    return pd.read_csv(p, **kw)


def plab(ax, lab, dx=-26.0, dy=3.0):
    """Capital panel label, 9 pt bold, upper left of the axes (offset in points from the axes corner)."""
    ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", fontweight="bold", fontsize=9, ha="left", va="bottom", annotation_clip=False)


# ================================================================================================================================ data stage
def data_f2():
    s, y = "Cabauw", 2013
    df = pv_dc_series(s, y)
    par = pr.build_params("A")
    tot, cyc, nc, tr = pr.run(df, par, pr.switches(), trace=True)
    st = tr[:, 0].astype(int)
    n = len(st)
    nd = n // 1440
    prev = np.concatenate([[st[0]], st[:-1]])
    complete = (prev == pl.START) & (st == pl.PRODUCTION)
    abort = (prev == pl.START) & (st != pl.START) & (st != pl.PRODUCTION)
    starts = (prev != pl.START) & (st == pl.START)
    day = np.arange(n) // 1440
    rows = []
    for d in range(nd):
        m = day == d
        rows.append(dict(day=d + 1, n_attempts=int(starts[m].sum()), n_completed=int(complete[m].sum()), n_aborts=int(abort[m].sum()), M_stack_kg=float(tr[m, 2].sum()),
                         M_del_kg=float(tr[m, 3].sum())))
    D = pd.DataFrame(rows)
    cand = D[(D.n_completed == 1) & (D.n_aborts >= 1)].sort_values("M_stack_kg").reset_index(drop=True)
    pick = cand.iloc[len(cand) // 2] if len(cand) % 2 == 1 else cand.iloc[len(cand) // 2 - 1]      # median M_stack (lower median for an even count)
    d0 = int(pick.day) - 1
    D["selected"] = D.day == pick.day
    D["max_attempts_day"] = D.n_attempts == D.n_attempts.max()
    csv(D, "F2_supplementary_day_list")
    i0, i1 = d0 * 1440, (d0 + 1) * 1440
    sl = slice(i0, i1)
    bd = np.diff(np.concatenate([[tr[max(i0 - 1, 0), 9]], tr[sl, 9]]))
    inv0 = tr[i0 - 1, 7] if i0 > 0 else par[pl.M_INV0]
    rows = pd.DataFrame(dict(minute=np.arange(1440), p_available_clipped_kW=np.minimum(df.p_dc_kw.to_numpy()[sl], 1000.0), p_stack_kW=tr[sl, 1], state=st[sl], M_stack_kg=tr[sl, 2],
                             delivered_kg=tr[sl, 3], purge_kg=tr[sl, 4], offspec_kg=tr[sl, 5], deoxo_dryer_kg=tr[sl, 6], blowdown_kg=bd, inventory_kg=tr[sl, 7]))
    closure = abs(rows.M_stack_kg.sum() - (rows.delivered_kg.sum() + rows.purge_kg.sum() + rows.offspec_kg.sum() + rows.deoxo_dryer_kg.sum() + rows.blowdown_kg.sum() + (tr[i1 - 1, 7] - inv0)))
    csv_note(rows, "F2_fate_day", f"{s} {y}, day {int(pick.day)} (median-M_stack day among {len(cand)} candidates with exactly one completed restart and at least one aborted attempt), POL-A: "
                                  f"{int(pick.n_attempts)} START entries, {int(pick.n_completed)} completed, {int(pick.n_aborts)} aborted; closure residual {closure:.1e} kg (content of the former figure header line)")


def data_f3():
    r = []
    v1 = pd.read_csv(RESULTS / "vt1_l26.csv")
    for rr in v1[v1.variant == "primary"].itertuples():
        r.append(dict(test="VT1 heat-up (L26)", label=rr.case, ratio=rr.t_model_min / rr.t_measured_min, kind="diagnostic", status="diagnostic, CAL infeasible (D31)"))
    v2 = pd.read_csv(RESULTS / "vt2_l28.csv")
    for rr in v2.itertuples():
        r.append(dict(test="VT2 pressurisation (L28)", label=f"PS {rr.power_set_pct} %", ratio=rr.t_predicted_min / rr.t_measured_min, kind="pass" if rr.abs_RE_le_20pct else "fail", status="FAIL (disclosed foreknowledge)"))
    v2b = pd.read_csv(RESULTS / "vt2b_l32.csv")
    for rr in v2b[v2b.row_type == "primary"].itertuples():
        r.append(dict(test="VT2b PEM pressurisation (L32)", label=f"{rr.p_set_bar} bar", ratio=rr.t_pred_min / rr.target_min, kind="pass" if rr.abs_RE_le_20pct else "fail", status="FAIL"))
    v4 = pd.read_csv(RESULTS / "vt4.csv")
    for rr in v4[v4.delta_log10.notna()].itertuples():
        r.append(dict(test="VT4 O$_2$ in H$_2$ (L24)", label=rr.model, ratio=10 ** rr.delta_log10, kind="fail", status="FAIL (transient peak vs steady law)"))
    v5 = pd.read_csv(RESULTS / "vt5a_l1.csv")
    for rr in v5.itertuples():
        r.append(dict(test="VT5a PSA purge (L1)", label=f"{rr.load_pct} % load", ratio=rr.purge_model_pct / rr.purge_measured_pct, kind="consistency", status="consistency, provenance unconfirmed"))
    v6 = pd.read_csv(RESULTS / "vt6_l15.csv")
    for rr in v6[v6.role != "CAL"].itertuples():
        r.append(dict(test="VT6 dryer breakthrough (L15)", label=rr.experiment, ratio=rr.t_b_pred_h / rr.t_b_meas_h, kind="pass" if rr.verdict.startswith("PASS") else "fail", status=rr.verdict))
    b = pd.read_csv(RESULTS / "vt6b_l15.csv")
    b = b[b.reading == "primary"]
    for rr in b.itertuples():
        r.append(dict(test="VT6b partial regeneration (L15)", label=f"{rr.experiment} t_s", ratio=1 + rr.t_s_RE, kind="pass" if rr.ok_t_s else "fail", status="FAIL"))
        r.append(dict(test="VT6b partial regeneration (L15)", label=f"{rr.experiment} q_s", ratio=1 + rr.q_s_RE, kind="pass" if rr.ok_q_s else "fail", status="FAIL"))
        r.append(dict(test="VT6b partial regeneration (L15)", label=f"{rr.experiment} outlet mole fraction", ratio=rr.y_min_pred_umol_mol / rr.y_meas_scaled_umol_mol, kind="pass" if rr.ok_Tdp_scaled else "fail", status="FAIL"))
    csv(pd.DataFrame(r), "F3_validation")


def data_f4():
    cls_name = {0: "Hot", 1: "Warm", 2: "Cold"}
    rows = []
    for s, yy in site_years():
        c = pd.read_parquet(RESULTS / "nominal" / f"cycles_{s}_{yy}_A.parquet")
        c = c[c.closed > 0.5]
        for k, g in c.groupby("class_0hot_1warm_2cold"):
            for v in g.tau_eq_min.to_numpy():
                rows.append(dict(site=s, year=yy, cls=cls_name[int(k)], tau_eq_min=v))
    tau = pd.DataFrame(rows)
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    f = ann[ann.model == "full"].copy()
    cats = ["Shutdown blow-down", "Off-spec diversion", "Start-up purge", "Deoxo", "Steady PSA purge"]
    for c_, nm in (("M_blowdown", cats[0]), ("M_offspec", cats[1]), ("M_purge", cats[2]), ("M_deoxo", cats[3]), ("M_dryer", cats[4])):
        f[nm] = f[c_] / f.M_stack * 100
    comp = f.groupby(["site", "policy"])[cats].mean().reset_index()
    pb = pd.read_csv(RESULTS / "policy" / "polB_pareto.csv")
    b60 = pb[(pb.variant == "POL-B_60") & np.isclose(pb.sb_f, 0.095)].copy()
    for c_, nm in (("M_blowdown", cats[0]), ("M_offspec", cats[1]), ("M_purge", cats[2]), ("M_deoxo", cats[3]), ("M_dryer", cats[4])):
        b60[nm] = b60[c_] / b60.M_stack * 100
    cb = b60.groupby("site")[cats].mean().reset_index()
    cb["policy"] = "B60"
    comp = pd.concat([comp, cb], ignore_index=True)
    ab = pd.read_csv(RESULTS / "nominal" / "ablation.csv")
    ab = ab[(ab.policy == "A") & (ab.variant == "off")].groupby(["site", "mechanism"]).d_lambda_SU_pp.mean().reset_index()
    csv_note(tau.groupby(["site", "cls"]).tau_eq_min.describe(percentiles=[0.1, 0.5, 0.9]).reset_index(), "F4_tau_eq_by_class",
             "Panel A of F4: the dot is the median (50 %), the vertical line (whisker) runs from the 10th to the 90th percentile of tau_eq (rated-min) over the completed cycles of the site-years of the site (POL-A); "
             "the number n of cycles is in the tick label; column 'count' = n; class labels Hot/Warm/Cold follow the cycle-file column class_0hot_1warm_2cold")
    csv(comp, "F4_loss_composition")
    csv(ab, "F4_ablation")


def data_f5():
    b = pd.read_csv(RESULTS / "nominal" / "binned_day_ratio.csv")
    b = b[(b.policy == "A") & (~b.bin_intra_day_attempts.str.startswith("no stack"))].copy()
    order = ["0", "1", "2", "3", "4-5", "6-8", ">=9"]
    b["x"] = b.bin_intra_day_attempts.map({k: i for i, k in enumerate(order)})
    csv(b, "F5_binned_ratio")
    csv(pd.read_csv(RESULTS / "nominal" / "intraday_slope.csv"), "F5_slope_descriptive")


def data_f6():
    sm = pd.read_csv(RESULTS / "resolution" / "resolution_summary.csv")
    fac = pd.read_csv(RESULTS / "resolution" / "factorial_2x2.csv")
    csv(sm, "F6_resolution_summary")
    csv(fac.groupby(["policy"])[["main_effect_resolution_pct", "main_effect_model_pct", "interaction_pct"]].agg(["mean", "std"]).reset_index(), "F6_factorial")


def data_f7():
    d = pd.read_csv(RESULTS / "nominal" / "beta_decomposition.csv")
    cl = d[d.chain == "B-CL"]
    terms = ["term_efficiency_convention_pp", "term_steady_purification_model_pp", "term_startup_delays_pp", "term_startup_purification_losses_pp"]
    ep = d[d.chain == "B-EP"]
    ep_terms = ["term_irradiance_source_pp", "term_resolution_pp", "term_sec_convention_and_steady_purification_pp", "term_startup_delays_and_losses_pp"]
    g = cl.groupby(["site", "policy"])[terms + ["beta_B_CL_pct", "rev_term_E_pp", "rev_term_L_pp", "fwd_E_pp", "fwd_L_pp"]].mean()
    g2 = ep.groupby(["site", "policy"])[ep_terms + ["beta_B_EP_vs_BoPPV_pct"]].mean()
    csv(g.reset_index(), "F7_beta_B_CL_chain")
    csv(g2.reset_index(), "F7_beta_B_EP_chain")


def data_f8():
    p = RESULTS / "uq" / "sobol_pooled.csv"
    if not p.exists():
        raise SystemExit("F8: results/uq/sobol_pooled.csv missing (run code/uq_final_summary.py sobol after the POL-B60 Sobol runs)")
    g = pd.read_csv(p)
    g = g[g.estimator.str.startswith("pooled")]
    g = g.assign(output="lambda_SU")
    sbp = RESULTS / "uq" / "sobol_beta.csv"
    gb = pd.read_csv(sbp) if sbp.exists() else pd.DataFrame()                      # beta indices (Brief 6 item 10): B-CL and B-S1, pooled 5-seed estimator, reference year
    keep = ["output", "site", "policy", "factor", "S1", "S1_conf", "ST", "ST_conf", "n_seeds"]
    allg = pd.concat([g[keep + ["ST_seed_sd"]], gb[keep] if len(gb) else None], ignore_index=True)
    allg["note"] = "pooled 5-seed Saltelli estimator; *_conf = 95 % bootstrap half-width of the pooled estimator; reference year per site; PV_RATIO is a design variable (D50b); figure shows lambda_SU only"
    csv(allg, "F8_sobol_pooled")


def data_f9():
    p = pd.read_csv(RESULTS / "policy" / "polB_pareto.csv")
    be = pd.read_csv(RESULTS / "policy" / "standby_breakeven.csv")
    pp = pd.read_csv(RESULTS / "policy" / "pol_p_decomposition.csv")
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    fa = ann[ann.model == "full"].set_index(["site", "year", "policy"])
    sb_nom = 0.095
    rows = []
    names = [("POL-A", "A"), ("POL-B_15", "B 15"), ("POL-B_30", "B 30"), ("POL-B_60", "B 60"), ("POL-B_120", "B 120"), ("POL-D", "D")]
    for s in SITES:
        for vn, tag in names:
            g = p[(p.site == s) & (p.variant == vn)]
            lo, nom, hi = [g[np.isclose(g.sb_f, f)].groupby("year").agg(E=("E_standby_kWh", "mean"), dM=("d_M_del_vs_POLA_pct", "mean")).mean() for f in (0.01, sb_nom, 0.18)]
            rows.append(dict(site=s, policy=tag, E_standby_MWh_at_0p01=lo.E / 1e3, E_standby_MWh_at_0p095=nom.E / 1e3, E_standby_MWh_at_0p18=hi.E / 1e3, dM_del_vs_POLA_pct=nom.dM))
        cc = [(fa.loc[(s, y, "C"), "M_del"] - fa.loc[(s, y, "A"), "M_del"]) / fa.loc[(s, y, "A"), "M_del"] * 100 for ss_, y in site_years() if ss_ == s]
        ec = [fa.loc[(s, y, "C"), "E_standby"] / 1e3 for ss_, y in site_years() if ss_ == s]
        rows.append(dict(site=s, policy="C", E_standby_MWh_at_0p095=np.mean(ec), dM_del_vs_POLA_pct=np.mean(cc)))
        q = pp[pp.site == s]
        dmP = float(((q.M_del_P - q.M_del_A) / q.M_del_A * 100).mean())
        rows.append(dict(site=s, policy="P (bounding: no pressure decay, no crossover handling, no source)", E_standby_MWh_at_0p095=0.0, dM_del_vs_POLA_pct=dmP))
        # break-even standby marker of POL-D: standby power at which the energy equals the stack energy value of the gained H2
        bd = be[(be.site == s) & (be.policy == "D") & np.isclose(be.standby_fraction, sb_nom)].iloc[0]
        gD = p[(p.site == s) & (p.variant == "POL-D") & np.isclose(p.sb_f, sb_nom)]
        eD = gD.groupby("year").E_standby_kWh.mean().mean() / 1e3
        dD = gD.groupby("year").d_M_del_vs_POLA_pct.mean().mean()
        e_be = eD * bd.breakeven_standby_fraction / sb_nom
        rows.append(dict(site=s, policy="D break-even marker", E_standby_MWh=e_be, dM_del_vs_POLA_pct=dD, breakeven_standby_fraction=bd.breakeven_standby_fraction))
    csv(pd.DataFrame(rows), "F9_policy_pareto")


DATA_STEPS = {"F2": data_f2, "F3": data_f3, "F4": data_f4, "F5": data_f5, "F6": data_f6, "F7": data_f7, "F8": data_f8, "F9": data_f9}


# ================================================================================================================================ draw stage
def draw_fig2():
    """Figure 2: ratio of predicted to measured values in the component tests (data: F3_validation_data.csv)."""
    apply_style()
    d = read("F3_validation")
    tests = list(dict.fromkeys(d.test))
    xlab = ["Heat-up time\nRauls 2023", "Pressurization,\nalkaline pilot\nBonaccorsi 2025", "Pressurization,\nPEM\nFlamm 2021", "O$_2$ in H$_2$\nLiu 2026",
            "Steady purge share\nRenau 2025", "Dryer breakthrough\nSantos 2026", "Partial regeneration\nSantos 2026"]
    assert len(xlab) == len(tests), tests
    fig, ax = plt.subplots(1, 1, figsize=(FULL_W_IN, 3.5))
    ax.axhspan(0.8, 1.2, color=GREY, alpha=0.22, lw=0, label="±20 % band")
    ax.axhline(1.0, color="black", lw=0.5)
    for i, tn in enumerate(tests):
        g = d[d.test == tn]
        x = i + np.linspace(-0.25, 0.25, len(g)) if len(g) > 1 else np.array([float(i)])
        for xx, rr in zip(x, g.itertuples()):
            if rr.kind == "pass":
                ax.plot(xx, rr.ratio, "o", mfc=BLUE, mec=BLUE, ms=4.5)
            elif rr.kind == "fail":
                ax.plot(xx, rr.ratio, "o", mfc="white", mec=VERM, mew=1.0, ms=4.5)
            elif rr.kind == "diagnostic":
                ax.plot(xx, rr.ratio, "o", mfc="white", mec=GREY, mew=1.0, ms=4.5)
            else:
                ax.plot(xx, rr.ratio, "D", mfc="white", mec=BLUE, mew=1.0, ms=4.0)
    ax.plot([], [], "o", mfc=BLUE, mec=BLUE, label="Within ±20 % (held-out)")
    ax.plot([], [], "o", mfc="white", mec=VERM, mew=1.0, label="Outside ±20 %")
    ax.plot([], [], "o", mfc="white", mec=GREY, mew=1.0, label="Diagnostic, calibration infeasible")
    ax.plot([], [], "D", mfc="white", mec=BLUE, mew=1.0, label="Consistency, provenance unconfirmed")
    ax.set_yscale("log")
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(xlab, fontsize=6)
    ax.set_ylabel("Predicted / measured (-)")
    ax.set_ylim(0.008, 120)
    ax.legend(loc="upper left", ncol=2, fontsize=6)
    ax.minorticks_off()
    tk = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    ax.set_yticks(tk)
    ax.set_yticklabels([f"{v:g}" for v in tk])
    return save_png(fig, "component_tests")


def draw_fig3():
    """Figure 3: hydrogen fate at restart (A-C one day; D ablation; E loss composition; F equivalent restart time by thermal class)."""
    apply_style()
    rows = read("F2_fate_day", comment="#")
    ab = read("F4_ablation")
    comp = read("F4_loss_composition")
    tq = read("F4_tau_eq_by_class", comment="#")
    fig = plt.figure(figsize=(FULL_W_IN, 8.4))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.05, 1.05, 1.75], width_ratios=[1.2, 1.3, 1.0])
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, :], sharex=axA)
    axC = fig.add_subplot(gs[2, :], sharex=axA)
    t = rows.minute.to_numpy() / 60.0
    axA.plot(t, rows.p_available_clipped_kW, color=BLUE, lw=0.7, label="Available DC power (clipped at 1 MW)")
    axA.plot(t, rows.p_stack_kW, color=VERM, lw=0.7, label="Stack power")
    axA.set_ylabel("Power (kW)")
    axA.legend(loc="upper right", fontsize=6)
    plab(axA, "A", dx=-36)
    h = lambda c: rows[c].to_numpy() * 60.0
    series = [("Delivered", h("delivered_kg"), CB[2]), ("Start-up purge (vented)", h("purge_kg"), CB[4]), ("Off-spec diversion", h("offspec_kg"), CB[3]), ("Deoxo + steady purge", h("deoxo_dryer_kg"), CB[5])]
    series = [(a, b, c) for a, b, c in series if np.nanmax(b) > 0]                      # legend only for series present
    polys = axB.stackplot(t, *[b for _, b, _ in series], colors=[c for _, _, c in series], linewidth=0, labels=[a for a, _, _ in series])
    for poly, (nm, _, _) in zip(polys, series):
        if nm == "Off-spec diversion":                                        # dotted fill separates this fate from its neighbours in grayscale
            poly.set_hatch("....")
            poly.set_edgecolor("white")
            poly.set_linewidth(0)
    axB.set_ylabel("H$_2$ flow (kg h$^{-1}$)")
    h1, l1 = axB.get_legend_handles_labels()
    bd = rows.blowdown_kg.to_numpy()
    ev = np.flatnonzero(bd > 0)
    if len(ev):
        axb = axB.twinx()
        axb.vlines(t[ev], 0, bd[ev], color="black", lw=0.9, label="Shutdown blow-down (kg per event)")
        axb.set_ylabel("Blow-down (kg)")
        axb.spines["right"].set_visible(True)
        h2, l2 = axb.get_legend_handles_labels()
        h1, l1 = h1 + h2, l1 + l2
    axB.legend(h1, l1, loc="upper left", fontsize=6)
    plab(axB, "B", dx=-36)
    axC.plot(t, rows.inventory_kg, color=BLUE, lw=0.8)
    axC.set_ylabel("Gas inventory (kg)")
    ax2 = axC.twinx()
    ax2.step(t, rows.state, where="post", color=CB[4], lw=0.6)
    ax2.set_yticks([0, 2, 3, 4])
    ax2.set_yticklabels(["OFF", "START", "PRODUCTION", "SHUTDOWN"], fontsize=6)
    ax2.set_ylim(0, 4.2)
    ax2.spines["right"].set_visible(True)
    axC.set_xlabel("Time of day (h, UTC)")
    axC.set_xlim(0, 24)
    axC.set_xticks(range(0, 25, 4))
    plt.setp(axA.get_xticklabels(), visible=False)
    plt.setp(axB.get_xticklabels(), visible=False)
    plab(axC, "C", dx=-36)
    # D: single-mechanism ablation
    ax = fig.add_subplot(gs[3, 0])
    mech = ["blowdown", "N2 washout (purge + off-spec)", "inert", "deoxo", "ramp (start-up delay)", "ALL K1 mechanisms off"]
    names = ["Blow-down", "N$_2$ washout", "Inerting", "Deoxo", "Start-up delay", "All off"]
    w = 0.26
    for i, s in enumerate(SITES):
        v = [float(ab[(ab.site == s) & (ab.mechanism == m)].d_lambda_SU_pp.iloc[0]) for m in mech]
        ax.barh(np.arange(len(mech)) + (i - 1) * w, v, height=w, color=SCOL[s], label=s)
    ax.set_yticks(np.arange(len(mech)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlim(-12.5, 1.0)
    ax.set_xlabel(r"Change of $\lambda_{SU}$ when switched off (pp)")
    ax.legend(loc="center left", fontsize=5.8)
    plab(ax, "D", dx=-62)
    # E: loss composition
    ax = fig.add_subplot(gs[3, 1])
    cats = ["Shutdown blow-down", "Off-spec diversion", "Start-up purge", "Deoxo", "Steady PSA purge"]
    catlab = ["Shutdown blow-down", "Off-spec diversion", "Start-up purge", "Deoxo", "Steady purge"]
    ccol = [CB[2], CB[3], CB[4], CB[5], "#bbbbbb"]      # blow-down: bluish green + hatch (not a site colour; side brief S item 4)
    chat = ["///", "....", "", "", ""]
    xt, xl, pos = [], [], 0
    pols = ["A", "B60", "C", "D"]
    centres = []
    for s in SITES:
        p0 = pos
        for pol in pols:
            r = comp[(comp.site == s) & (comp.policy == pol)].iloc[0]
            bot = 0
            for cname, cl, col, ht in zip(cats, catlab, ccol, chat):
                ax.bar(pos, r[cname], bottom=bot, color=col, width=0.8, hatch=ht, edgecolor="white" if ht else col, linewidth=0.0, label=cl if (s == "Cabauw" and pol == "A") else None)
                bot += r[cname]
            xt.append(pos)
            xl.append(POLNAME[pol])
            pos += 1
        centres.append((p0 + pos - 1) / 2)
        pos += 0.6
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=90, fontsize=5.8)
    ax.set_ylabel(r"H$_2$ not delivered (% of stack H$_2$)")
    ax.set_ylim(0, 20.5)
    ax.legend(loc="upper right", fontsize=5.4)
    for c_, s in zip(centres, SITES):
        ax.text(c_, -0.40, s, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color="black")
    plab(ax, "E", dx=-30)
    # F: equivalent restart time by thermal class
    ax = fig.add_subplot(gs[3, 2])
    pos, ticks, labs = 0, [], []
    for s in SITES:
        for cl in ("Hot", "Warm", "Cold"):
            g = tq[(tq.site == s) & (tq.cls == cl)]
            if len(g):
                g = g.iloc[0]
                ax.vlines(pos, g["10%"], g["90%"], color=SCOL[s], lw=1.6)
                ax.plot(pos, g["50%"], "o", color=SCOL[s], ms=4)
                ticks.append(pos)
                labs.append(f"{cl} (n = {int(g['count'])})")
            pos += 1
        pos += 0.7
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, rotation=90, fontsize=5.5)
    ax.set_ylabel(r"Equivalent restart time $\tau_{eq}$ (rated-minutes)")
    for s in SITES:
        ax.plot([], [], "o-", color=SCOL[s], label=s)
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.2)
    plab(ax, "F", dx=-30)
    return save_png(fig, "restart_fate_mechanisms")


def draw_fig4():
    """Figure 4: daily statistics by intra-day restart attempts (data: F5_binned_ratio_data.csv)."""
    apply_style()
    b = read("F5_binned_ratio")
    order = ["0", "1", "2", "3", "4-5", "6-8", ">=9"]
    fig, axs = plt.subplots(2, 2, figsize=(FULL_W_IN, 5.6), sharex=True)
    spec = ((axs[0, 0], "ratio_lambda_SU_pct", r"$\lambda_{SU}$, ratio of sums over the days of a bin (%)", "A"), (axs[0, 1], "eta_del", r"$\eta_{del}$, ratio of sums over the days of a bin (-)", "B"),
            (axs[1, 0], "m_SU_per_day_kg", r"$m_{SU}$ per day, mean over the days of a bin (kg d$^{-1}$)", "C"), (axs[1, 1], "M_stack_per_day_kg", r"$M_{stack}$ per day, mean over the days of a bin (kg d$^{-1}$)", "D"))
    for ax, col, ylab, lab in spec:
        for s in SITES:
            g = b[b.site == s].sort_values("x")
            g = g[g.n_days >= 3]                                 # bins with fewer than 3 days are not drawn (listed in the data CSV)
            ax.plot(g.x + (SITES.index(s) - 1) * 0.07, g[col], "-", color=SCOL[s], lw=0.8, alpha=0.7)
            ok = g.n_days >= 10
            ax.plot(g.x[ok] + (SITES.index(s) - 1) * 0.07, g[col][ok], SMARK[s], color=SCOL[s], ms=4.4, label=s)
            ax.plot(g.x[~ok] + (SITES.index(s) - 1) * 0.07, g[col][~ok], SMARK[s], mfc="white", mec=SCOL[s], ms=4.4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_ylabel(ylab, fontsize=6.6)
        plab(ax, lab, dx=-30)
    fig.supxlabel("Intra-day restart attempts per day (-)", fontsize=8)
    axs[0, 0].set_ylim(0, 50)
    axs[0, 1].set_ylim(0, 1.0)
    axs[1, 0].set_ylim(0, 14)
    axs[1, 1].set_ylim(0, 175)
    axs[0, 0].legend(loc="upper left", fontsize=6)
    axs[0, 1].plot([], [], "o", mfc="white", mec="black", label="Bin with 3-9 days")
    axs[0, 1].legend(loc="lower left", fontsize=6)
    return save_png(fig, "daily_restart_statistics")


def factorial_by_site():
    """Rows used for Figure 5C: mean and standard deviation over site-years of the 2 x 2 factorial effects, full shutdown, per site (results/resolution/factorial_2x2.csv)."""
    fac = pd.read_csv(RESULTS / "resolution" / "factorial_2x2.csv")
    eff = ["main_effect_resolution_pct", "main_effect_model_pct", "interaction_pct"]
    out = []
    for s in SITES:
        g = fac[(fac.site == s) & (fac.policy == "A")]
        for e in eff:
            out.append(dict(site=s, policy="A", effect=e, mean_pct=g[e].mean(), std_pct=g[e].std(), n_site_years=len(g)))
    df = pd.DataFrame(out)
    csv(df, "F6_factorial_by_site")
    return df


def draw_fig5():
    """Figure 5: input resolution (A-C) and bias of the conventional models (D, E)."""
    apply_style()
    sm = read("F6_resolution_summary")
    fcs = factorial_by_site()
    bcl = read("F7_beta_B_CL_chain")
    bep = read("F7_beta_B_EP_chain")
    fig = plt.figure(figsize=(FULL_W_IN, 6.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.15])
    axA, axB, axC = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])
    for pol, pn in (("A", "full shutdown"), ("C", "daytime standby")):
        for s in SITES:
            g = sm[(sm.site == s) & (sm.policy == pol)].sort_values("block_min")
            dash = "-" if pol == "A" else (0, (3.2, 1.6))
            axA.plot(g.block_min, g.N_att_ratio_mean, ls=dash, marker=SMARK[s], color=SCOL[s], ms=3.4, label=f"{s}, {pn}")
            axB.plot(g.block_min, g.d_lambda_SU_pp_mean, ls=dash, marker=SMARK[s], color=SCOL[s], ms=3.4)
    axA.set_ylabel("Restart attempts, ratio to 1-min input (-)")
    axB.set_ylabel(r"Change of $\lambda_{SU}$ against 1-min input (pp)")
    for ax, lab in ((axA, "A"), (axB, "B")):
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 15, 30, 60])
        ax.set_xticklabels(["1", "5", "15", "30", "60"])
        ax.set_xlabel(" ")                           # space reserved; one common label for A and B is placed below
        ax.minorticks_off()
        plab(ax, lab, dx=-30)
    axB.axhline(0, color="black", lw=0.4)
    axA.legend(fontsize=5.2, loc="lower left", ncol=1, handlelength=3.6)
    w = 0.26
    eff = ["main_effect_resolution_pct", "main_effect_model_pct", "interaction_pct"]
    for i, s in enumerate(SITES):
        g = fcs[fcs.site == s].set_index("effect").loc[eff]
        axC.bar(np.arange(3) + (i - 1) * w, g.mean_pct, w, yerr=g.std_pct, color=SCOL[s], error_kw=dict(lw=0.6, capsize=1.5), label=s)
    axC.set_xticks(range(3))
    axC.set_xticklabels(["Resolution\n(60 vs 1 min)", "Model (event vs\nsteady-purge)", "Interaction"], fontsize=5.8)
    axC.axhline(0, color="black", lw=0.5)
    axC.set_ylabel("Effect on $M_{del}$ (% of 1-min event model)")
    axC.legend(fontsize=6, loc="lower right")
    plab(axC, "C", dx=-30)

    def stacked(ax, gg, cols_terms, labels, tot, ylab, lab, reverse):
        cols = [CB[0], CB[4], CB[5], CB[3]]
        hatches = ["", "///", "...", "xxx"]       # grayscale separation of adjacent segments (side brief S item 4); white segment edge
        pos, xt, xl = 0, [], []
        first = True
        centres = []
        for s in SITES:
            p0 = pos
            for pol in "ACD":
                r = gg[(gg.site == s) & (gg.policy == pol)].iloc[0]
                bp, bn = 0.0, 0.0
                for c_, nm, col, ht in zip(cols_terms, labels, cols, hatches):
                    v = r[c_]
                    if v >= 0:
                        ax.bar(pos, v, bottom=bp, color=col, width=0.8, hatch=ht, edgecolor="white", linewidth=0.6, label=nm if first else None)
                        bp += v
                    else:
                        ax.bar(pos, v, bottom=bn, color=col, width=0.8, hatch=ht, edgecolor="white", linewidth=0.6, label=nm if first else None)
                        bn += v
                ax.plot(pos, r[tot], "kD", ms=3.2, mfc="white", label="Total" if first else None)
                if reverse:
                    ax.plot(pos + 0.3, r["rev_term_E_pp"], "^", color="black", ms=3.6, mfc="white", mew=0.8, label="Efficiency convention, reverse order" if first else None)
                    ax.plot(pos + 0.3, r[tot] - r["fwd_L_pp"] + r["rev_term_L_pp"], "v", color="black", ms=3.6, mfc="black", mew=0.8, label="Start-up purification losses, reverse order (segment top)" if first else None)
                first = False
                xt.append(pos)
                xl.append(POLNAME[pol])
                pos += 1
            centres.append((p0 + pos - 1) / 2)
            pos += 0.6
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, rotation=90, fontsize=6)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=5.4, loc="upper right", ncol=1)
        for c_, s in zip(centres, SITES):
            ax.text(c_, -0.30, s, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color="black")
        plab(ax, lab, dx=-30)

    axD, axE = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])
    stacked(axD, bcl, ["term_efficiency_convention_pp", "term_steady_purification_model_pp", "term_startup_delays_pp", "term_startup_purification_losses_pp"],
            ["Efficiency convention", "Steady purification model", "Start-up delays", "Start-up purification losses"], "beta_B_CL_pct", "Bias of the fixed-efficiency model (%)", "D", True)
    stacked(axE, bep, ["term_irradiance_source_pp", "term_resolution_pp", "term_sec_convention_and_steady_purification_pp", "term_startup_delays_and_losses_pp"],
            ["Irradiance source", "Time resolution", "Consumption convention + steady purification", "Start-up delays + losses"], "beta_B_EP_vs_BoPPV_pct",
            "Bias of the hourly feasibility model, BoP from PV (%)", "E", False)
    axD.set_ylim(-3, 62)
    fig.canvas.draw()                                # common x label of panels A and B (same quantity), centred under the pair
    bb = axA.xaxis.label.get_window_extent()
    xm = 0.5 * (0.5 * (axA.get_position().x0 + axA.get_position().x1) + 0.5 * (axB.get_position().x0 + axB.get_position().x1))
    ym = fig.transFigure.inverted().transform((0, bb.y0))[1]
    fig.text(xm, ym, "Input block length (min)", ha="center", va="bottom", fontsize=8)
    return save_png(fig, "resolution_and_model_bias")


SOBOL_LAB = dict(P_MIN_F="P$_{min}$", SB_F="Standby power", I_INV="Gas inventory", P_OP="Operating pressure", T_PURGE="Purge time", T_BLOW="Blow-down time", T_SU_COLD="Cold start-up time",
                 C_PER_MW="Thermal capacity", R_PER_MW="Thermal resistance", XO2_OVERRIDE="O$_2$ in H$_2$", PV_RATIO="PV sizing ratio (design)")


def draw_fig6():
    """Figure 6: Sobol indices of the restart loss (data: F8_sobol_pooled_data.csv, rows with output = lambda_SU)."""
    apply_style()
    g = read("F8_sobol_pooled")
    g = g[g.output == "lambda_SU"]
    pols = [p_ for p_ in ("A", "B60", "C") if p_ in set(g.policy)]
    order = g[(g.site == "Cabauw") & (g.policy == "A")].sort_values("ST", ascending=False).factor.tolist()
    fig, axs = plt.subplots(len(pols), 3, figsize=(FULL_W_IN, 1.9 * len(pols) + 0.6), sharex=True)
    axs = np.atleast_2d(axs)
    for i, pol in enumerate(pols):
        for j, s in enumerate(SITES):
            ax = axs[i, j]
            d = g[(g.site == s) & (g.policy == pol)].set_index("factor").loc[order]
            y = np.arange(len(order))
            ax.barh(y + 0.18, d.ST, 0.34, xerr=d.ST_conf, color=VERM, error_kw=dict(lw=0.5), label="Total order $S_T$")
            ax.barh(y - 0.18, d.S1, 0.34, xerr=d.S1_conf, color=BLUE, error_kw=dict(lw=0.5), label="First order $S_1$")
            ax.set_yticks(y)
            ax.set_yticklabels([SOBOL_LAB[f] for f in order] if j == 0 else [], fontsize=5.6)
            ax.invert_yaxis()
            ax.axvline(0, color="black", lw=0.4)
            ax.text(0.98, 0.04, f"{s}, {POLNAME[pol].lower()}", transform=ax.transAxes, ha="right", fontsize=6.3)
            ax.annotate("ABCDEFGHI"[i * 3 + j], xy=(0, 1), xycoords="axes fraction", xytext=(-92 if j == 0 else -6, 3), textcoords="offset points", fontweight="bold", fontsize=9, ha="left", va="bottom",
                        annotation_clip=False)
    fig.supxlabel("Sobol index (-)", fontsize=8)
    axs[0, 0].legend(fontsize=5.6, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    return save_png(fig, "sobol_restart_loss")


def draw_fig7():
    """Figure 7: delivered-hydrogen gain against standby energy (data: F9_policy_pareto_data.csv)."""
    apply_style()
    d = read("F9_policy_pareto")
    tags = [("A", "A", "o", 0), ("B 15", "B15", "s", 1), ("B 30", "B30", "^", 2), ("B 60", "B60", "v", 3), ("B 120", "B120", "D", 4), ("C", "C", "X", 6), ("D", "D", "P", 5)]
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W_IN, 3.7))
    handles = {}
    for ax, s, lab in zip(axs, SITES, "ABC"):
        g = d[d.site == s].set_index("policy")
        pts = []
        for tag, key, mk, k in tags:
            r = g.loc[tag]
            e_nom = float(r.E_standby_MWh_at_0p095)
            if tag not in ("C",):
                ax.plot([r.E_standby_MWh_at_0p01, r.E_standby_MWh_at_0p18], [r.dM_del_vs_POLA_pct] * 2, "-", color=CB[k % 7], lw=0.8, alpha=0.55)
            ms = 4.8 if tag == "C" else 4.5
            h, = ax.plot(e_nom, r.dM_del_vs_POLA_pct, mk, color=CB[k % 7], ms=ms)
            handles[key] = h
            pts.append((e_nom, float(r.dM_del_vs_POLA_pct)))
        pk = [i for i in g.index if str(i).startswith("P (")][0]
        rp = g.loc[pk]
        handles["P"], = ax.plot(float(rp.E_standby_MWh_at_0p095), rp.dM_del_vs_POLA_pct, "*", color="#F0E442", mec="black", mew=0.5, ms=7)
        pts = sorted(pts)                                                     # Pareto front over the stop policies; the pressure-hold bound is a separate, unconnected marker
        front, best = [], -1e9
        for e_, m_ in pts:
            if m_ > best:
                front.append((e_, m_))
                best = m_
        handles["front"], = ax.plot(*zip(*front), "-", color="black", lw=0.6)
        rb = g.loc["D break-even marker"]
        handles["be"], = ax.plot(rb.E_standby_MWh, rb.dM_del_vs_POLA_pct, "o", mfc="white", mec="black", ms=5.5, mew=1.0)
        ax.text(0.97, 0.05, s, transform=ax.transAxes, ha="right", fontsize=7)
        plab(ax, lab, dx=-30)
        if s == "Cabauw":
            ax.set_ylabel(r"$\Delta M_{del}$ vs full shutdown (%)")
    keys = ["A", "B15", "B30", "B60", "B120", "C", "D", "P", "front", "be"]
    labels = [POLNAME[k] for k in keys[:7]] + ["Pressure-hold bound (no pressure decay, no crossover handling)", "Front over the stop policies (0.095 $P_{rated}$)", "24-h standby break-even power"]
    fig.legend([handles[k] for k in keys], labels, loc="outside lower center", ncol=4, fontsize=5.6, columnspacing=1.2, handlelength=1.6)
    axs[1].set_xlabel("Standby energy (MWh yr$^{-1}$)")                   # one label for the three panels, under the middle one
    return save_png(fig, "standby_tradeoff")


DRAW_STEPS = {"fig2": draw_fig2, "fig3": draw_fig3, "fig4": draw_fig4, "fig5": draw_fig5, "fig6": draw_fig6, "fig7": draw_fig7}


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "draw":
        for w in (args[1:] or list(DRAW_STEPS)):
            print(w, DRAW_STEPS[w]())
    else:
        for w in (args or list(DATA_STEPS)):
            DATA_STEPS[w]()
            print(w, "data written")
        if not args:
            for w, fn in DRAW_STEPS.items():
                print(w, fn())

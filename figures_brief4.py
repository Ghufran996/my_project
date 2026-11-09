"""Brief 4 item 15: DRAFT paper figures F2-F7, F9 (F8 in figures_f8.py after the Sobol run) and draft tables T1-T5.

    python code/figures_brief4.py [F2 F3 F4 F5 F6 F7 F9 tables]

All figures: one style (configs/paper.mplstyle), PNG at 1000 dpi in results/figures/, raw data CSV in results/figures/data/<name>_data.csv, status DRAFT until the success gate.
Nothing is plotted that is not in a results file or computed here from the engine output (F2, F5 re-run the engine with the trace on, nominal inputs).
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
from scipy import stats  # noqa: E402

from h2su import plant as pl  # noqa: E402
from h2su import plantrun as pr  # noqa: E402
from h2su.figures import COL_W_IN, FULL_W_IN, apply_style, save_png  # noqa: E402
from h2su.io import FIGURES, RESULTS, load_registry  # noqa: E402
from h2su.pv import pv_dc_series  # noqa: E402
from sweeps import site_years  # noqa: E402

DATA = FIGURES / "data"
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
from h2su.figures import SITE_COLORS as SCOL, SITE_MARKERS as SMARK  # noqa: E402  (shared site palette, Brief 7 item 6)
SITES = ["Cabauw", "Carpentras", "Gobabeb"]


def panel(ax, lab):
    ax.text(-0.13, 1.03, lab, transform=ax.transAxes, fontweight="bold", fontsize=9)


def csv(df, name):
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA / f"{name}_data.csv", index=False, float_format="%.6g")


# ------------------------------------------------------------------------------------------------------------------------------------ F2
def f2():
    apply_style()
    s, y = "Cabauw", 2013
    df = pv_dc_series(s, y)
    par = pr.build_params("A")
    tot, cyc, nc, tr = pr.run(df, par, pr.switches(), trace=True)
    st = tr[:, 0]
    starts = np.flatnonzero(np.diff((st == pl.START).astype(int)) == 1)
    nday = np.bincount(starts // 1440, minlength=366)
    d = int(np.argmax(nday[:365]))
    i0, i1 = d * 1440, (d + 1) * 1440
    sl = slice(i0, i1)
    t = np.arange(1440) / 60.0
    bd = np.diff(np.concatenate([[tr[max(i0 - 1, 0), 9]], tr[sl, 9]]))
    inv0 = tr[i0 - 1, 7] if i0 > 0 else par[pl.M_INV0]
    rows = pd.DataFrame(dict(minute=np.arange(1440), p_available_clipped_kW=np.minimum(df.p_dc_kw.to_numpy()[sl], 1000.0), p_stack_kW=tr[sl, 1], state=st[sl], M_stack_kg=tr[sl, 2],
                             delivered_kg=tr[sl, 3], purge_kg=tr[sl, 4], offspec_kg=tr[sl, 5], deoxo_dryer_kg=tr[sl, 6], blowdown_kg=bd, inventory_kg=tr[sl, 7]))
    csv(rows, "F2_fate_day")
    closure = abs(rows.M_stack_kg.sum() - (rows.delivered_kg.sum() + rows.purge_kg.sum() + rows.offspec_kg.sum() + rows.deoxo_dryer_kg.sum() + rows.blowdown_kg.sum() + (tr[i1 - 1, 7] - inv0)))
    fig, ax = plt.subplots(3, 1, figsize=(FULL_W_IN, 5.6), sharex=True)
    ax[0].plot(t, rows.p_available_clipped_kW, color=CB[0], lw=0.7, label="Available DC power (clipped at 1 MW)")
    ax[0].plot(t, rows.p_stack_kW, color=CB[1], lw=0.7, label="Stack power")
    ax[0].set_ylabel("Power (kW)")
    ax[0].legend(loc="upper right", ncol=1, fontsize=6)
    panel(ax[0], "A")
    h = lambda c: rows[c].to_numpy() * 60.0
    ax[1].stackplot(t, h("delivered_kg"), h("purge_kg"), h("offspec_kg"), h("deoxo_dryer_kg"), colors=[CB[2], CB[4], CB[3], CB[5]], linewidth=0,
                    labels=["Delivered", "Start-up purge (vented)", "Off-spec diversion", "Deoxo + steady PSA purge"])
    ax[1].set_ylabel("H$_2$ flow (kg h$^{-1}$)")
    axb = ax[1].twinx()
    ev = np.flatnonzero(bd > 0)
    axb.vlines(t[ev], 0, bd[ev], color=CB[6], lw=0.9, label="Shutdown blow-down (kg per event)")
    axb.set_ylabel("Blow-down (kg)")
    axb.spines["right"].set_visible(True)
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="upper left", ncol=1, fontsize=6)
    panel(ax[1], "B")
    ax[2].plot(t, rows.inventory_kg, color=CB[0], lw=0.8, label="Gas inventory")
    ax[2].set_ylabel("Inventory (kg)")
    ax2 = ax[2].twinx()
    ax2.step(t, rows.state, where="post", color=CB[4], lw=0.6)
    ax2.set_yticks([0, 2, 3, 4])
    ax2.set_yticklabels(["OFF", "START", "PROD.", "SHUTDOWN"])
    ax2.set_ylim(0, 4.2)
    ax2.spines["right"].set_visible(True)
    ax[2].set_xlabel("Time of day (h, UTC)")
    panel(ax[2], "C")
    ax[0].text(0.01, 0.9, f"{s} {y}, day {d + 1}, POL-A, {int(nday[d])} START entries; day closure residual {closure:.1e} kg", transform=ax[0].transAxes, fontsize=6.5)
    return save_png(fig, "F2_h2_fate_restart_day")


# ------------------------------------------------------------------------------------------------------------------------------------ F3
def f3():
    apply_style()
    r = []
    v1 = pd.read_csv(RESULTS / "vt1_l26.csv")
    for rr in v1[v1.variant == "primary"].itertuples():
        r.append(dict(test="VT1 heat-up\n(L26)", label=rr.case, ratio=rr.t_model_min / rr.t_measured_min, role=rr.role.split(" ")[0], ok=bool(rr.abs_RE_le_20pct), unit="min"))
    v2 = pd.read_csv(RESULTS / "vt2_l28.csv")
    for rr in v2.itertuples():
        r.append(dict(test="VT2 pressurisation\n(L28)", label=f"PS {rr.power_set_pct} %", ratio=rr.t_predicted_min / rr.t_measured_min, role=rr.role, ok=bool(rr.abs_RE_le_20pct), unit="min"))
    v2b = pd.read_csv(RESULTS / "vt2b_l32.csv")
    for rr in v2b[v2b.row_type == "primary"].itertuples():
        r.append(dict(test="VT2b PEM\npressurisation (L32)", label=f"{rr.p_set_bar} bar", ratio=rr.t_pred_min / rr.target_min, role="VAL", ok=bool(rr.abs_RE_le_20pct), unit="min"))
    v4 = pd.read_csv(RESULTS / "vt4.csv")
    for rr in v4[v4.delta_log10.notna()].itertuples():
        r.append(dict(test="VT4 O$_2$ in H$_2$\n(L24)", label=rr.model, ratio=10 ** rr.delta_log10, role="VAL", ok=False, unit="ppm"))
    v5 = pd.read_csv(RESULTS / "vt5a_l1.csv")
    for rr in v5.itertuples():
        r.append(dict(test="VT5a PSA purge\n(L1)", label=f"{rr.load_pct} % load", ratio=rr.purge_model_pct / rr.purge_measured_pct, role=rr.role, ok=bool(rr.abs_RE_le_20pct), unit="%"))
    v6 = pd.read_csv(RESULTS / "vt6_l15.csv")
    for rr in v6.itertuples():
        r.append(dict(test="VT6 dryer breakthrough\n(L15)", label=rr.experiment, ratio=rr.t_b_pred_h / rr.t_b_meas_h, role=rr.role, ok=(rr.verdict.startswith("PASS") or rr.role == "CAL"), unit="h"))
    d = pd.DataFrame(r)
    csv(d, "F3_validation")
    tests = list(dict.fromkeys(d.test))
    fig, ax = plt.subplots(1, 1, figsize=(FULL_W_IN, 3.1))
    ax.axhspan(0.8, 1.2, color="#999999", alpha=0.25, lw=0, label="±20 % acceptance band (SC-V)")
    ax.axhline(1.0, color="black", lw=0.5)
    rng = np.random.default_rng(0)
    for i, tname in enumerate(tests):
        g = d[d.test == tname]
        x = i + np.linspace(-0.22, 0.22, len(g)) if len(g) > 1 else np.array([float(i)])
        for xx, rr in zip(x, g.itertuples()):
            mk = "s" if rr.role == "CAL" else "o"
            ax.plot(xx, rr.ratio, mk, mfc=(CB[2] if rr.ok else "white"), mec=(CB[2] if rr.ok else CB[1]), mew=1.0, ms=4.5)
    ax.plot([], [], "o", mfc=CB[2], mec=CB[2], label="within ±20 % (VAL)")
    ax.plot([], [], "o", mfc="white", mec=CB[1], label="outside ±20 % (failure shown)")
    ax.plot([], [], "s", mfc=CB[2], mec=CB[2], label="calibration point (CAL)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(tests, fontsize=6.3)
    ax.set_ylabel("Predicted / measured (-)")
    ax.legend(loc="upper left", ncol=2, fontsize=6.3)
    ax.minorticks_off()
    ax.set_yticks([0.1, 0.5, 1, 2, 5, 10])
    ax.set_yticklabels(["0.1", "0.5", "1", "2", "5", "10"])
    return save_png(fig, "F3_component_validation")


# ------------------------------------------------------------------------------------------------------------------------------------ F4
def f4():
    apply_style()
    cls_name = {0: "Hot", 1: "Warm", 2: "Cold"}
    rows = []
    for s, yy in site_years():
        for pol in "AC":
            c = pd.read_parquet(RESULTS / "nominal" / f"cycles_{s}_{yy}_{pol}.parquet")
            c = c[c.closed > 0.5]
            for k, g in c.groupby("class_0hot_1warm_2cold"):
                for v in g.tau_eq_min.to_numpy():
                    rows.append(dict(site=s, year=yy, policy=pol, cls=cls_name[int(k)], tau_eq_min=v))
    tau = pd.DataFrame(rows)
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    f = ann[ann.model == "full"].copy()
    for c_, nm in (("M_blowdown", "Shutdown blow-down"), ("M_offspec", "Off-spec diversion"), ("M_purge", "Start-up purge"), ("M_deoxo", "Deoxo"), ("M_dryer", "Steady PSA purge")):
        f[nm] = f[c_] / f.M_stack * 100
    comp = f.groupby(["site", "policy"])[["Shutdown blow-down", "Off-spec diversion", "Start-up purge", "Deoxo", "Steady PSA purge"]].mean().reset_index()
    ab = pd.read_csv(RESULTS / "nominal" / "ablation.csv")
    ab = ab[(ab.policy == "A") & (ab.variant == "off")].groupby(["site", "mechanism"]).d_lambda_SU_pp.mean().reset_index()
    csv(tau.groupby(["site", "policy", "cls"]).tau_eq_min.describe(percentiles=[0.1, 0.5, 0.9]).reset_index(), "F4_tau_eq_by_class")
    csv(comp, "F4_loss_composition")
    csv(ab, "F4_ablation")
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W_IN, 3.2), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.2]))
    ax = axs[0]
    pos = 0
    ticks, labs = [], []
    for s in SITES:
        for cl in ("Hot", "Warm", "Cold"):
            g = tau[(tau.site == s) & (tau.policy == "A") & (tau.cls == cl)].tau_eq_min
            if len(g) == 0:
                pos += 1
                continue
            p10, p50, p90 = np.percentile(g, [10, 50, 90])
            ax.vlines(pos, p10, p90, color=SCOL[s], lw=1.6)
            ax.plot(pos, p50, "o", color=SCOL[s], ms=4)
            ticks.append(pos)
            labs.append(cl)
            pos += 1
        pos += 0.7
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, rotation=90, fontsize=6)
    ax.set_ylabel(r"Equivalent restart time $\tau_{eq}$ (rated-min)")
    for i, s in enumerate(SITES):
        ax.plot([], [], "o-", color=SCOL[s], label=s)
    ax.legend(loc="upper right", fontsize=6)
    ax.set_title("POL-A; median and P10-P90 over cycles", fontsize=6.5)
    panel(ax, "A")
    ax = axs[1]
    cats = ["Shutdown blow-down", "Off-spec diversion", "Start-up purge", "Deoxo", "Steady PSA purge"]
    ccol = [CB[6], CB[3], CB[4], CB[5], "#999999"]
    xl, xt = [], []
    pos = 0
    for s in SITES:
        for pol in "ACD":
            r = comp[(comp.site == s) & (comp.policy == pol)].iloc[0]
            b = 0
            for cname, col in zip(cats, ccol):
                ax.bar(pos, r[cname], bottom=b, color=col, width=0.8, label=cname if (s == "Cabauw" and pol == "A") else None)
                b += r[cname]
            xt.append(pos)
            xl.append(f"POL-{pol}")
            pos += 1
        pos += 0.6
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=90, fontsize=6)
    ax.set_ylabel(r"H$_2$ lost (% of stack H$_2$)")
    ax.legend(loc="upper right", fontsize=5.8)
    for i, s in enumerate(SITES):
        ax.text(i * 3.6 + 1.0, -0.2, s, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color=SCOL[s])
    panel(ax, "B")
    ax = axs[2]
    mech = ["blowdown", "N2 washout (purge + off-spec)", "inert", "deoxo", "ramp (start-up delay)", "ALL K1 mechanisms off"]
    names = ["Blow-down", "N$_2$ washout", "Inerting", "Deoxo", "Ramp", "All off"]
    w = 0.26
    for i, s in enumerate(SITES):
        v = [float(ab[(ab.site == s) & (ab.mechanism == m)].d_lambda_SU_pp.iloc[0]) for m in mech]
        ax.barh(np.arange(len(mech)) + (i - 1) * w, v, height=w, color=SCOL[s], label=s)
    ax.set_yticks(np.arange(len(mech)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(r"Change of $\lambda_{SU}$ when switched off (pp, POL-A)")
    ax.legend(loc="center right", fontsize=6)
    panel(ax, "C")
    return save_png(fig, "F4_equivalent_restart_time_composition_ablation")


# ------------------------------------------------------------------------------------------------------------------------------------ F5
def f5():
    apply_style()
    rows = []
    for s, yy in site_years():
        df = pv_dc_series(s, yy)
        par = pr.build_params("A")
        tot, cyc, nc, tr = pr.run(df, par, pr.switches(), trace=True)
        st = tr[:, 0]
        k_start = np.flatnonzero(np.diff((st == pl.START).astype(int)) == 1) + 1
        day = lambda k: np.asarray(k) // 1440
        ndays = 365 if len(df) < 366 * 1440 else 366
        ms_d = np.bincount(day(np.arange(len(df))), weights=tr[:, 2], minlength=ndays)[:ndays]
        md_d = np.bincount(day(np.arange(len(df))), weights=tr[:, 3], minlength=ndays)[:ndays]
        zen = df.zenith.to_numpy()
        c = cyc[:nc]
        c = c[c[:, 10] > 0.5]
        msu = (c[:, 2] - c[:, 3])
        msu_d = np.bincount(day(c[:, 1].astype(int) - 1), weights=msu, minlength=ndays)[:ndays]
        ncl_d = np.bincount(day(c[:, 1].astype(int) - 1), minlength=ndays)[:ndays]
        n_all = np.bincount(day(k_start), minlength=ndays)[:ndays]
        first_dirunal = np.zeros(ndays)
        seen = set()
        for k in k_start:
            dd = int(k // 1440)
            if dd not in seen and zen[k] < 90.0:
                first_dirunal[dd] = 1
                seen.add(dd)
        for dd in range(ndays):
            if ms_d[dd] > 1.0 and n_all[dd] > 0:
                rows.append(dict(site=s, year=yy, day=dd + 1, N_r_attempts=n_all[dd], N_r_intra=max(n_all[dd] - first_dirunal[dd], 0), M_stack_kg=ms_d[dd], M_del_kg=md_d[dd],
                                 m_SU_kg=msu_d[dd], n_cycles_closed=ncl_d[dd], lambda_SU_day_pct=msu_d[dd] / ms_d[dd] * 100, eta_del_day=md_d[dd] / ms_d[dd]))
    d = pd.DataFrame(rows)
    csv(d, "F5_site_days")
    fit = []
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W_IN, 3.0))
    for ax, col, ylab, lab in ((axs[0], "lambda_SU_day_pct", r"$\lambda_{SU}$ of the day (%)", "A"), (axs[1], "eta_del_day", r"Deliverable fraction $\eta_{del}$ of the day (-)", "B")):
        for s in SITES:
            g = d[d.site == s]
            if col == "lambda_SU_day_pct":
                g = g[g.n_cycles_closed > 0]                  # lambda_SU is defined on days with at least one closed restart cycle
            x, y = g.N_r_intra.to_numpy(), g[col].to_numpy()
            ax.plot(x + (SITES.index(s) - 1) * 0.12, y, "o", color=SCOL[s], ms=1.6, alpha=0.35, mew=0, label=None)
            lr = stats.linregress(x, y)
            n = len(x)
            ci = stats.t.ppf(0.975, n - 2) * lr.stderr
            xs = np.array([0, x.max()])
            ax.plot(xs, lr.intercept + lr.slope * xs, color=SCOL[s], lw=1.2, label=f"{s}: slope {lr.slope:.3g} ± {ci:.2g}, R$^2$ = {lr.rvalue ** 2:.2f}, n = {n}")
            fit.append(dict(site=s, output=col, n_days=n, slope=lr.slope, slope_ci95=ci, intercept=lr.intercept, R2=lr.rvalue ** 2, p=lr.pvalue))
        ax.set_xlabel("Intra-day restart attempts per day (-)")
        ax.set_ylabel(ylab)
        ax.legend(loc="upper left" if col != "eta_del_day" else "lower left", fontsize=5.8)
        panel(ax, lab)
    axs[0].set_ylim(-5, 60)
    csv(pd.DataFrame(fit), "F5_regression")
    return save_png(fig, "F5_lambda_su_vs_intraday_restarts")


# ------------------------------------------------------------------------------------------------------------------------------------ F6
def f6():
    apply_style()
    sm = pd.read_csv(RESULTS / "resolution" / "resolution_summary.csv")
    fac = pd.read_csv(RESULTS / "resolution" / "factorial_2x2.csv")
    csv(sm, "F6_resolution_summary")
    csv(fac.groupby(["policy"])[["main_effect_resolution_pct", "main_effect_model_pct", "interaction_pct"]].agg(["mean", "std"]).reset_index(), "F6_factorial")
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W_IN, 2.9))
    for pol, ls in (("A", "-"), ("C", "--")):
        for s in SITES:
            g = sm[(sm.site == s) & (sm.policy == pol)].sort_values("block_min")
            dash = "-" if pol == "A" else (0, (3.2, 1.6))          # POL-C dashed; legend handles long enough to show the dashes
            axs[0].plot(g.block_min, g.N_att_ratio_mean, ls=dash, marker=SMARK[s], color=SCOL[s], ms=3.4, label=f"{s}, POL-{pol}")
            axs[1].plot(g.block_min, g.lambda_SU_mean, ls=dash, marker=SMARK[s], color=SCOL[s], ms=3.4)
    axs[0].set_ylabel(r"$N_r(\Delta t)/N_r$(1 min) (-)")
    axs[1].set_ylabel(r"$\lambda_{SU}$ (%)")
    for ax, lab in zip(axs[:2], "AB"):
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 15, 30, 60])
        ax.set_xticklabels(["1", "5", "15", "30", "60"])
        ax.set_xlabel(r"Input block length $\Delta t$ (min)")
        ax.minorticks_off()
        panel(ax, lab)
    axs[0].legend(fontsize=5.2, loc="lower left", ncol=1, handlelength=3.6)
    ax = axs[2]
    w = 0.26
    eff = ["main_effect_resolution_pct", "main_effect_model_pct", "interaction_pct"]
    for i, s in enumerate(SITES):
        g = fac[(fac.site == s) & (fac.policy == "A")]
        m = [g[e].mean() for e in eff]
        sd = [g[e].std() for e in eff]
        ax.bar(np.arange(3) + (i - 1) * w, m, w, yerr=sd, color=SCOL[s], error_kw=dict(lw=0.6, capsize=1.5), label=s)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Resolution\n(60 vs 1 min)", "Model\n(event vs B-S1)", "Interaction"], fontsize=6.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel(r"Effect on $M_{del}$ (% of 1-min event model)")
    ax.legend(fontsize=6)
    panel(ax, "C")
    return save_png(fig, "F6_input_resolution")


# ------------------------------------------------------------------------------------------------------------------------------------ F7
def f7():
    apply_style()
    d = pd.read_csv(RESULTS / "nominal" / "beta_decomposition.csv")
    cl = d[d.chain == "B-CL"]
    terms = [("term_efficiency_convention_pp", "Efficiency convention"), ("term_steady_purification_model_pp", "Steady purification model s(L)"),
             ("term_startup_delays_pp", "Start-up delays"), ("term_startup_purification_losses_pp", "Start-up purification losses")]
    ep = d[d.chain == "B-EP"]
    ep_terms = [("term_irradiance_source_pp", "Irradiance source"), ("term_resolution_pp", "Time resolution"), ("term_sec_convention_and_steady_purification_pp", "SEC convention + steady purification"),
                ("term_startup_delays_and_losses_pp", "Start-up delays + losses")]
    g = cl.groupby(["site", "policy"])[[t for t, _ in terms] + ["beta_B_CL_pct"]].agg(["mean", "std"])
    g2 = ep.groupby(["site", "policy"])[[t for t, _ in ep_terms] + ["beta_B_EP_vs_BoPPV_pct"]].agg(["mean", "std"])
    csv(g.reset_index(), "F7_beta_B_CL_chain")
    csv(g2.reset_index(), "F7_beta_B_EP_chain")
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W_IN, 3.1))
    for ax, gg, tt, tot, lab, ttl in ((axs[0], g, terms, "beta_B_CL_pct", "A", r"$\beta$ vs B-CL (%)"), (axs[1], g2, ep_terms, "beta_B_EP_vs_BoPPV_pct", "B", r"$\beta$ vs B-EP (BoP drawn from PV, %)")):
        pos, xt, xl = 0, [], []
        for s in SITES:
            for pol in "ACD":
                bp, bn = 0.0, 0.0
                for (c_, nm), col in zip(tt, CB):
                    v = gg.loc[(s, pol), (c_, "mean")]
                    if v >= 0:
                        ax.bar(pos, v, bottom=bp, color=col, width=0.8, label=nm if (s == "Cabauw" and pol == "A") else None)
                        bp += v
                    else:
                        ax.bar(pos, v, bottom=bn, color=col, width=0.8, label=nm if (s == "Cabauw" and pol == "A") else None)
                        bn += v
                ax.plot(pos, gg.loc[(s, pol), (tot, "mean")], "kD", ms=3.2, mfc="white", label="Total" if (s == "Cabauw" and pol == "A") else None)
                xt.append(pos)
                xl.append(f"{pol}")
                pos += 1
            ax.text(pos - 2, -0.2, s, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color=SCOL[s])
            pos += 0.6
        ax.set_xticks(xt)
        ax.set_xticklabels([f"POL-{x}" for x in xl], rotation=90, fontsize=6)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel(ttl)
        ax.legend(fontsize=5.6, loc="upper right")
        panel(ax, lab)
    return save_png(fig, "F7_beta_decomposition")


# ------------------------------------------------------------------------------------------------------------------------------------ F9
def f9():
    apply_style()
    p = pd.read_csv(RESULTS / "policy" / "polB_pareto.csv")
    sb_nom = 0.095
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    fa = ann[ann.model == "full"].set_index(["site", "year", "policy"])
    rows = []
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W_IN, 2.9), sharey=False)
    for ax, s, lab in zip(axs, SITES, "ABC"):
        names = [("POL-A", "A"), ("POL-B_15", "B 15"), ("POL-B_30", "B 30"), ("POL-B_60", "B 60"), ("POL-B_120", "B 120"), ("POL-D", "D")]
        for k, (vn, tag) in enumerate(names):
            g = p[(p.site == s) & (p.variant == vn)]
            lo, nom, hi = [g[np.isclose(g.sb_f, f)].groupby("year").agg(E=("E_standby_kWh", "mean"), dM=("d_M_del_vs_POLA_pct", "mean")).mean() for f in (0.01, sb_nom, 0.18)]
            ax.plot([lo.E / 1e3, hi.E / 1e3], [nom.dM, nom.dM], "-", color=CB[k % 7], lw=0.8, alpha=0.6)
            ax.plot(nom.E / 1e3, nom.dM, "o", color=CB[k % 7], ms=4.5, label=tag)
            rows.append(dict(site=s, policy=tag, E_standby_MWh_at_0p01=lo.E / 1e3, E_standby_MWh_at_0p095=nom.E / 1e3, E_standby_MWh_at_0p18=hi.E / 1e3, dM_del_vs_POLA_pct=nom.dM))
        # POL-C from the nominal runs (standby through daylight stops, nominal standby 0.095)
        cc = [(fa.loc[(s, y, "C"), "M_del"] - fa.loc[(s, y, "A"), "M_del"]) / fa.loc[(s, y, "A"), "M_del"] * 100 for _, y in site_years() if _ == s]
        ec = [fa.loc[(s, y, "C"), "E_standby"] / 1e3 for _, y in site_years() if _ == s]
        ax.plot(np.mean(ec), np.mean(cc), "s", color=CB[6], ms=4.5, label="C")
        rows.append(dict(site=s, policy="C", E_standby_MWh_at_0p095=np.mean(ec), dM_del_vs_POLA_pct=np.mean(cc)))
        ax.set_xlabel("Standby energy (MWh yr$^{-1}$)")
        ax.set_title(s, fontsize=7.5)
        panel(ax, lab)
        if s == "Cabauw":
            ax.set_ylabel(r"$\Delta M_{del}$ vs POL-A (%)")
            ax.legend(fontsize=6, loc="lower right", title="Policy (B: t_sb,max in min)", title_fontsize=6)
    csv(pd.DataFrame(rows), "F9_policy_pareto")
    return save_png(fig, "F9_policy_pareto")


# ------------------------------------------------------------------------------------------------------------------------------------ tables
def tables():
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    reg = load_registry()
    keys = ["pmin_pem_low", "pmin_pem_high", "standby_power_low", "standby_power_high", "plant_operating_pressure_inferred", "pem190_coldstart_time", "pem190_hotstart_time",
            "awe_inerting_initial_total", "awe_prestart", "awe_purge_duration", "awe_depressurisation_duration_avg", "awe_final_inerting_total", "psa_purge_loss_load10", "psa_purge_loss_load100",
            "pem50_BoP_power", "pem50_system_input_power", "bcl_recovery_primary", "c3_fixed_efficiency", "l15_isoW_qm", "l15_opt_tAD", "vpsa_P_AD_opt"]
    t1 = reg[reg.name.isin(keys)][["name", "value_or_range", "unit", "source_id", "location", "role"]].copy()
    inv = pd.read_csv(RESULTS / "inventory_ranges.csv")
    t1 = pd.concat([t1, pd.DataFrame([dict(name="gas inventory i (nominal / ends)", value_or_range="0.699 / 0.120-0.888 (PEM), 3.178 (AWE pilot scenario)", unit="min of rated production per bar",
                                           source_id="L13 / L32 / L28 (results/inventory_ranges.csv)", location="see file", role="INPUT/UQ")])], ignore_index=True)
    t1.to_csv(out / "T1_parameters_sources.csv", index=False)
    v = []
    for nm, f, cols in (("VT1", "vt1_l26.csv", None), ("VT2", "vt2_l28.csv", None), ("VT2b", "vt2b_l32.csv", None), ("VT4", "vt4.csv", None), ("VT5a", "vt5a_l1.csv", None), ("VT6", "vt6_l15.csv", None)):
        d = pd.read_csv(RESULTS / f)
        d.insert(0, "validation_test", nm)
        v.append(d)
    pd.concat(v, ignore_index=True).to_csv(out / "T2_validation_all_rows.csv", index=False)
    sy = pd.DataFrame([dict(site=s, year=y) for s, y in site_years()])
    pd.DataFrame(dict(item=["Plant", "PV", "Sites", "Years", "Policies", "Input resolution", "Technology"], value=["PEM 1 MW stack-level DC rating, auxiliaries from an auxiliary supply (D41)", "1.5 MWp, BSRN 1-min GHI/DNI/DHI via pvlib chain",
                                                                                                                  "Cabauw, Carpentras, Gobabeb (BSRN)", f"{len(sy)} site-years 2013-2018 after the 95 % year screen", "POL-A, B (15/30/60/120 min), C, D",
                                                                                                                  "1-min reference; 5/15/30/60-min block means", "PEM (AWE-procedure values as labelled sensitivities)"])).to_csv(out / "T3_sites_scenarios.csv", index=False)
    bd = pd.read_csv(RESULTS / "nominal" / "beta_decomposition.csv")
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    f = ann[ann.model == "full"]
    t4 = f.groupby(["site", "policy"]).agg(lambda_SU_mean=("lambda_SU", "mean"), lambda_SU_std=("lambda_SU", "std"), eta_del_mean=("eta_del", "mean"), N_attempts=("N_r_attempts", "mean"),
                                         N_completed=("N_r_completed", "mean"), beta_B_CL_1min=("beta_B_CL_1min", "mean"), beta_B_EP=("beta_B_EP", "mean"), beta_B_S1=("beta_B_S1", "mean"),
                                         beta_B_S3=("beta_B_S3", "mean"), beta_B_S2_curve1=("beta_B_S2_curve1", "mean"), n_years=("year", "count")).reset_index()
    t4.to_csv(out / "T4_beta_summary.csv", index=False, float_format="%.5g")
    pd.DataFrame(dict(baseline=["B-CL", "B-EP", "B-S1", "B-S2 (C3, L4)", "B-S3"], description=["Fixed 75 % LHV stack efficiency, constant 98.4 % purification recovery, P_min cut-off, no start-up loss (classical)",
                                                                                                    "PVGIS hourly power, constant net SEC 67 kWh/kg, P_min cut-off (engineering practice)",
                                                                                                    "Steady load-dependent purification loss s(L) (L1 Table 2 fit) on an instantly switching plant (SOTA C1 class)",
                                                                                                    "Dispatch-optimising C3 formulation with efficiency curve and start-up penalty, re-run on the same PV input",
                                                                                                    "Thermal start-up ramp kept, purification mechanisms off (C4/C2/C5 class)"],
                  reported_or_rerun=["re-run", "re-run", "re-run", "re-run", "re-run"])).to_csv(out / "T5_baselines.csv", index=False)
    print("tables written")


if __name__ == "__main__":
    print("figures_brief4.py provides helper functions only. Run python code/figures_final.py to regenerate the manuscript figures.")

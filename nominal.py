"""Brief 3 items 10 and 11: trial cost runs T1-T5 and the first NOMINAL annual runs (PRELIMINARY until the analyst reviews them).

    python code/nominal.py trial      -> results/trial_cost_chain.csv (re-measurement written by the chain; the Brief 3 decision record results/trial_cost.csv is a stored file and is never overwritten, D82-2)
    python code/nominal.py nominal    -> results/nominal/{annual.csv, ablation.csv, gap_exclusion.csv, plausibility.csv, cycles_*.parquet}, results/diagnostics/
Plant: PEM, 1 MW stack-level DC rating, 1.5 MWp PV, 1-min input, nominal registry values (plantrun.build_params), policies POL-A, POL-C, POL-D.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su import plant as pl  # noqa: E402
from h2su import plantrun as pr  # noqa: E402
from h2su.baselines import bs2_c3 as b2  # noqa: E402
from h2su.baselines.bcl import bcl, bep  # noqa: E402
from h2su.baselines.bs1 import bs1_annual  # noqa: E402
from h2su.io import CODE_DIR, CONFIGS, RESULTS, param  # noqa: E402
from h2su.pv import block_mean_hold, pv_dc_series  # noqa: E402

POLS = ("A", "C", "D")


def site_years():
    y = yaml.safe_load((CODE_DIR / "configs" / "sites.yaml").read_text())
    return [(s, yr) for s, v in y["sites"].items() for yr in v["years_selected"]]


def peak_rss_mb():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_all import _peak_rss_mb
    return _peak_rss_mb()


def trial():
    import numba
    rows = []
    cfg = [("T1", "PEM Cabauw 2014 POL-A, 1-min year", "Cabauw", 2014, "A", 1, 525600), ("T2", "PEM Gobabeb 2014 POL-A, 1-min year", "Gobabeb", 2014, "A", 1, 525600),
           ("T3", "PEM Carpentras 2016 POL-D, 1-min year (AWE not the second system, awe_sourcing_check.csv)", "Carpentras", 2016, "D", 1, 525600),
           ("T4", "PEM Cabauw 2014 POL-A, 60-min block-mean input on the 1-min grid", "Cabauw", 2014, "A", 60, 525600),
           ("T5", "PEM Cabauw 2014 POL-A, 10-s internal step, 30 days", "Cabauw", 2014, "A", 1, 30 * 1440)]
    for tid, desc, s, y, pol, block, n in cfg:
        df = pv_dc_series(s, y).iloc[:n].copy()
        par = pr.build_params(pol)
        if block > 1:
            df["p_dc_kw"] = block_mean_hold(df.p_dc_kw.to_numpy(), block)
        if tid == "T5":
            df = pd.DataFrame({c: np.repeat(df[c].to_numpy(), 6) for c in ("p_dc_kw", "zenith", "ta_c", "fill")})
            par = par.copy()
            par[pl.DT] = 1.0 / 6.0
        w = []
        for _ in range(5):
            t0 = time.perf_counter()
            pr.run(df, par, pr.switches())
            w.append(time.perf_counter() - t0)
        rows.append(dict(id=tid, case=desc, n_steps=len(df), wall_s_min=min(w), wall_s_median=float(np.median(w)), wall_s_max=max(w), wall_first_call_s=w[0], n_repeats=5,
                         peak_rss_mb=peak_rss_mb(), threads_used=1, logical_cpus=os.cpu_count(), numba_threads_available=numba.config.NUMBA_NUM_THREADS))
    df = pd.DataFrame(rows)
    ann = df[df.id.isin(["T1", "T2", "T3", "T4"])].wall_s_median.max()
    n_set = 1024 * (12 + 2)
    dec = (f"PL8 rule: trial annual run <= 2 s -> direct Saltelli/Sobol (N = 1024, k = 12 factors, N(k+2) = {n_set} runs per site x tech x policy). Slowest annual trial median "
           f"{ann:.3f} s -> estimated {n_set * ann / 60:.1f} min single-thread per set; 6 sets (3 sites x 2 policies) = {6 * n_set * ann / 3600:.2f} h single-thread, about "
           f"{6 * n_set * ann / 3600 / 14 * 60:.0f} min on 14 workers (RAM per worker about {df.peak_rss_mb.max():.0f} MiB). No Morris screening needed. No Sobol run in this brief.")
    df["sobol_design_decision"] = dec
    df.to_csv(RESULTS / "trial_cost_chain.csv", index=False)
    print(df.drop(columns=["sobol_design_decision"]).round(4).to_string(index=False))
    print(dec)


def b2_dp(p_dc_kw, k_curve):
    tab = pd.read_csv(CONFIGS / "tables" / "L4_virahsawmy2024.csv")
    e = tab[tab.table == "Fig2"]
    emax = float(e[e.item == "eta_max"].value.iloc[0])
    emin = {int(r.curve): float(r.value) for r in e[e.item == "eta_min"].itertuples()}
    cv = b2.Curve(emax, emin[k_curve])
    d = b2.solve_dp(np.minimum(p_dc_kw, pr.P_RATED_KW) / 1000.0, 1.0 / 60.0, 1.0, cv)
    return d["H2_kg"], cv


def run_model(df, pol, sw_off=None, over=None, trace=False):
    par = pr.build_params(pol, **(over or {}))
    sw = pr.switches(**(sw_off or {}))
    tot, cyc, nc, tr = pr.run(df, par, sw, trace=trace)
    return par, tot, cyc, nc, tr, pr.metrics(tot, cyc, nc, par)


def plaus(m, tot, df, par, nm):
    cf = float(df.p_dc_kw.sum() / 60.0 / (1500.0 * len(df) / 60.0))
    spec = (m["E_stack"] / m["M_stack"]) if m["M_stack"] > 0 else np.nan
    fates = [m[k] for k in ("M_del", "M_purge", "M_offspec", "M_deoxo", "M_dryer", "M_blowdown")]
    return dict(fates_nonneg=bool(min(fates) >= 0.0), eta_del_le_1=bool(m["eta_del"] <= 1.0), lambda_SU_ge_0=bool(m["lambda_SU"] >= -1e-9), closure_H2=m["closure_H2_rel"],
                closure_E=m["closure_E_rel"], closure_ok=bool(m["closure_H2_rel"] <= 1e-6 and m["closure_E_rel"] <= 1e-6), pv_capacity_factor=cf,
                pv_cf_in_bounds=bool(0.05 <= cf <= 0.35), stack_specific_energy_kWh_kg=spec, spec_energy_in_bounds=bool(45.0 <= spec <= 70.0),
                stack_efficiency_LHV=33.33 / spec if spec == spec else np.nan, eff_le_1=bool(33.33 / spec <= 1.0) if spec == spec else False)


def nominal():
    out = RESULTS / "nominal"
    out.mkdir(parents=True, exist_ok=True)
    (RESULTS / "diagnostics").mkdir(parents=True, exist_ok=True)
    ann, abl, gap, pla = [], [], [], []
    # D42: purge + off-spec diversion are two parts of one theta_spec requirement -> one mechanism "N2 washout"; the steady PSA term is a steady-term sensitivity (D40-8),
    # not a dryer start-up ablation (dryer start-up NOT modelled, SN17), excluded from the K1 list; the thermal ramp is a start-up delay, not a purification mechanism.
    mech = [("blowdown", dict(blow=0), True), ("N2 washout (purge + off-spec)", dict(purge=0, offspec=0), True), ("inert", dict(inert=0), True), ("deoxo", dict(deoxo=0), True),
            ("steady PSA loss (steady-term sensitivity)", dict(dryer=0), False), ("ramp (start-up delay)", dict(ramp=0), False)]
    interaction = []
    t0 = time.time()
    day_rows = {}
    for s, y in site_years():
        df = pv_dc_series(s, y)
        pdc = df.p_dc_kw.to_numpy()
        bcl1 = bcl(pdc, 1)
        bclh = bcl(pdc, 60)
        bclh2 = bcl(pdc, 1, recovery=param("vpsa_recovery_opt") / 100.0)
        bepv = bep(s, y)
        bs2 = {k: b2_dp(pdc, k) for k in (1, 2, 3)}
        for pol in POLS:
            par, tot, cyc, nc, tr, m = run_model(df, pol, trace=(pol == "A"))
            if pol == "A":
                nstart = np.bincount(np.minimum((np.flatnonzero(np.diff((tr[:, 0] == pl.START).astype(int)) == 1)) // 1440, 366), minlength=367)
                d = int(np.argmax(nstart[:366]))
                day_rows.setdefault(s, []).append((nstart[d], y, d, tr[d * 1440:(d + 1) * 1440].copy(), df.p_dc_kw.to_numpy()[d * 1440:(d + 1) * 1440].copy()))
            # baselines on the same dispatch policy
            p1, sw1 = pr.s1_config(par)
            t1 = pr.run(df, p1, sw1)
            p3, sw3 = pr.s3_config(par)
            t3 = pr.run(df, p3, sw3)
            md = m["M_del"]
            bet = lambda x: (x - md) / md * 100.0
            m1 = bs1_annual(pdc)
            row = dict(site=s, year=y, policy=pol, model="full", **{k: m[k] for k in m}, beta_B_CL_1min=bet(bcl1["M_del"]), beta_B_CL_60min=bet(bclh["M_del"]), beta_B_CL_98p98=bet(bclh2["M_del"]),
                       beta_B_EP=bet(bepv["M_del"]), beta_B_S1=bet(t1[0][pl.T_DEL]), beta_B_S3=bet(t3[0][pl.T_DEL]))
            for k, (h2, cv) in bs2.items():
                row[f"beta_B_S2_curve{k}"] = bet(h2)
            # diagnostic B-S2-rule (curves 1..3): full model's policy, C3 curve, purification off
            for k in (1, 2, 3):
                cv = bs2[k][1]
                pr_, sw_r = pr.s2rule_config(par, cv)
                tr_ = pr.run(df, pr_, sw_r)
                row[f"beta_B_S2rule_curve{k}"] = bet(tr_[0][pl.T_STACK])
                row[f"beta_B_S2_minus_rule_curve{k}"] = bet(bs2[k][0]) - bet(tr_[0][pl.T_STACK])           # dispatch-foresight part
                row[f"beta_B_S2rule_minus_full_curve{k}"] = bet(tr_[0][pl.T_STACK])                          # modelling part
            ann.append(row)
            for nm, M in (("B-CL_1min", bcl1["M_del"]), ("B-CL_60min", bclh["M_del"]), ("B-EP", bepv["M_del"]), ("B-S1", t1[0][pl.T_DEL]), ("B-S3", t3[0][pl.T_DEL]),
                          ("B-S2_curve1", bs2[1][0]), ("B-S2_curve2", bs2[2][0]), ("B-S2_curve3", bs2[3][0])):
                ann.append(dict(site=s, year=y, policy=pol, model=nm, M_del=M, M_stack=np.nan))
            pla.append(dict(site=s, year=y, policy=pol, **plaus(m, tot, df, par, "full")))
            # cycles
            c = cyc[:nc]
            ts = df.index
            cd = pd.DataFrame(c, columns=["i_open", "i_close", "loss_kg", "steady_ref_kg", "stack_kg", "class_0hot_1warm_2cold", "diurnal", "n_abort", "gap_affected", "t_su_s", "closed",
                                          "abort_net_kg", "complete_net_kg"])
            cd["t_open"] = [ts[int(i)] if i < len(ts) else pd.NaT for i in cd.i_open]
            cd["t_close"] = [ts[int(i)] if i < len(ts) else pd.NaT for i in cd.i_close]
            cd["m_SU_kg"] = cd.loss_kg - cd.steady_ref_kg
            cd["tau_eq_min"] = cd.m_SU_kg / par[pl.MRATED_MIN]
            cd.to_parquet(out / f"cycles_{s}_{y}_{pol}.parquet")
            # gap exclusion (D27)
            cl = cd[cd.closed > 0.5]
            ng = cl[cl.gap_affected < 0.5]
            gap.append(dict(site=s, year=y, policy=pol, n_cycles=len(cl), n_gap_affected=int((cl.gap_affected > 0.5).sum()), share_gap_affected=float((cl.gap_affected > 0.5).mean()) if len(cl) else np.nan,
                            lambda_SU_all_pct=m["lambda_SU"], lambda_SU_non_gap_cycles_pct=float(ng.m_SU_kg.sum() / m["M_stack"] * 100), contribution_gap_cycles_pp=float(cl[cl.gap_affected > 0.5].m_SU_kg.sum() / m["M_stack"] * 100),
                            tau_eq_median_all=float(cl.tau_eq_min.median()) if len(cl) else np.nan, tau_eq_median_non_gap=float(ng.tau_eq_min.median()) if len(ng) else np.nan))
            # ablation (mechanism off in turn; deoxo and dryer also at the upper end of the sourced range, D34)
            singles = {}
            for nm, off, in_k1 in mech:
                a = run_model(df, pol, sw_off=off)[5]
                singles[nm] = a["lambda_SU"] - m["lambda_SU"]
                abl.append(dict(site=s, year=y, policy=pol, mechanism=nm, variant="off", in_K1_list=in_k1, lambda_SU_nominal_pct=m["lambda_SU"], lambda_SU_variant_pct=a["lambda_SU"],
                                d_lambda_SU_pp=a["lambda_SU"] - m["lambda_SU"], d_M_del_pct=(a["M_del"] - md) / md * 100.0))
            # all start-up purification mechanisms off together (K1 set: blow-down, N2 washout, inerting, deoxo) and the interaction term (D42)
            a = run_model(df, pol, sw_off=dict(blow=0, purge=0, offspec=0, deoxo=0, inert=0))[5]
            d_all = a["lambda_SU"] - m["lambda_SU"]
            sum_single = sum(singles[k] for k in ("blowdown", "N2 washout (purge + off-spec)", "inert", "deoxo"))
            abl.append(dict(site=s, year=y, policy=pol, mechanism="ALL K1 mechanisms off", variant="off", in_K1_list=True, lambda_SU_nominal_pct=m["lambda_SU"], lambda_SU_variant_pct=a["lambda_SU"],
                            d_lambda_SU_pp=d_all, d_M_del_pct=(a["M_del"] - md) / md * 100.0))
            interaction.append(dict(site=s, year=y, policy=pol, d_lambda_all_off_pp=d_all, sum_single_deltas_pp=sum_single, interaction_pp=d_all - sum_single, lambda_SU_all_off_pct=a["lambda_SU"]))
            # deoxo at the upper end of its sourced range: L25 0.25 % O2-in-H2 (2500 ppm) at the deoxo inlet at all loads
            a = run_model(df, pol, over=dict(XO2_OVERRIDE=2500.0))[5]
            abl.append(dict(site=s, year=y, policy=pol, mechanism="deoxo", in_K1_list=True, variant="upper end 2500 ppm O2 (L25 0.25 %)", lambda_SU_nominal_pct=m["lambda_SU"], lambda_SU_variant_pct=a["lambda_SU"],
                            d_lambda_SU_pp=a["lambda_SU"] - m["lambda_SU"], d_M_del_pct=(a["M_del"] - md) / md * 100.0))
            # dryer: start-up mechanism not modelled (SN17 open); upper-end row records the PSA purge share at the sourced upper end (L1 Table 2, 26.7 % at 10 % load) as a steady-term check
            a = run_model(df, pol, over=dict(PHI=pr.psa_fit()[0] * 1.5, B_PSA=pr.psa_fit()[1] * 1.5))[5]
            abl.append(dict(site=s, year=y, policy=pol, mechanism="steady PSA loss (steady-term sensitivity)", in_K1_list=False,
                            variant="steady PSA purge share x1.5 (steady-term sensitivity; start-up dryer transient NOT modelled, SN17)", lambda_SU_nominal_pct=m["lambda_SU"],
                            lambda_SU_variant_pct=a["lambda_SU"], d_lambda_SU_pp=a["lambda_SU"] - m["lambda_SU"], d_M_del_pct=(a["M_del"] - md) / md * 100.0))
        print(s, y, f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(ann).to_csv(out / "annual.csv", index=False)
    pd.DataFrame(abl).to_csv(out / "ablation.csv", index=False)
    pd.DataFrame(interaction).to_csv(out / "ablation_interaction.csv", index=False)
    pd.DataFrame(gap).to_csv(out / "gap_exclusion.csv", index=False)
    pd.DataFrame(pla).to_csv(out / "plausibility.csv", index=False)
    diagnostics(day_rows)
    print("done", f"{time.time()-t0:.0f}s")


def diagnostics(day_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(str(CONFIGS / "paper.mplstyle"))
    for s, lst in day_rows.items():
        n, y, d, tr, pdc = max(lst, key=lambda t: t[0])
        t = np.arange(1440) / 60.0
        fig, ax = plt.subplots(3, 1, figsize=(6.0, 6.4), sharex=True)
        ax[0].plot(t, np.minimum(pdc, 1000.0), color="#0072B2", lw=0.8, label="available DC power (clipped)")
        ax[0].plot(t, tr[:, 1], color="#D55E00", lw=0.8, label="stack power")
        ax[0].set_ylabel("Power (kW)"); ax[0].legend(frameon=False)
        ax[1].plot(t, tr[:, 2] * 60, color="#000000", lw=0.8, label="stack H2")
        ax[1].plot(t, tr[:, 3] * 60, color="#009E73", lw=0.8, label="delivered")
        ax[1].plot(t, (tr[:, 4] + tr[:, 5] + tr[:, 6]) * 60, color="#CC79A7", lw=0.8, label="purge + off-spec + deoxo + dryer")
        ax[1].set_ylabel("H2 flow (kg/h)"); ax[1].legend(frameon=False)
        ax[2].plot(t, tr[:, 7], color="#0072B2", lw=0.8, label="gas inventory")
        ax2 = ax[2].twinx()
        ax2.plot(t, tr[:, 0], color="#E69F00", lw=0.6, drawstyle="steps-post")
        ax2.set_ylabel("State (0 OFF, 2 START, 3 PROD, 4 SHUTDOWN)"); ax[2].set_ylabel("Inventory (kg)"); ax[2].set_xlabel("Time of day (h, UTC)")
        ax[0].set_title(f"{s} {y} day {d + 1} (POL-A, {n} START events), diagnostic", fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / "diagnostics" / f"fate_day_{s}.png", dpi=1000)
        plt.close(fig)


if __name__ == "__main__":
    {"trial": trial, "nominal": nominal}[sys.argv[1]]()

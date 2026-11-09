"""Brief 4 item 8 (D43d, D43e): inventory OAT summary + pre-declared falsification verdict, and the closed-form estimator error.

    python code/inventory_analysis.py  (after `python code/sweeps.py inventory`)

Falsification rule (D43d, fixed before the run): if POL-A lambda_SU at i = 0.120 min/bar is < 1.0 % at ANY site-year, the POL-A materiality statement must carry the
inventory qualification in the abstract and conclusion.
Estimator (D43e, reported, no criterion): lambda_hat_SU = N_complete x m_dot_rated x (i (p_op - p0) + theta_spec) / M_stack, N_complete = completed restarts;
relative error vs the engine over all nominal runs (POL-A/C/D) and all OAT runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h2su import plant as pl  # noqa: E402
from h2su import plantrun as pr  # noqa: E402
from h2su.io import RESULTS  # noqa: E402


def main():
    raw = pd.read_csv(RESULTS / "sens" / "inventory_oat_raw.csv")
    out = RESULTS / "sens"
    rows = []
    for (site, pol, var), g in raw.groupby(["site", "policy", "variant"]):
        rows.append(dict(site=site, policy=pol, variant=var, i_min_per_bar=g.i_level.iloc[0], n_site_years=len(g), lambda_SU_mean=g.lambda_SU.mean(), lambda_SU_min=g.lambda_SU.min(),
                         lambda_SU_max=g.lambda_SU.max(), lambda_SU_complete_mean=g.lambda_SU_complete.mean(), lambda_SU_abort_mean=g.lambda_SU_abort.mean(),
                         N_r_completed_mean=g.N_r_completed.mean(), N_r_aborted_mean=g.N_r_aborted.mean(), M_del_mean=g.M_del.mean()))
    s = pd.DataFrame(rows)
    # slope per unit i per site (OLS over the six levels, and over the five PEM-technology levels without the 3.178 AWE-pilot level)
    sl = []
    for (site, pol), g in s.groupby(["site", "policy"]):
        for lab, gg in (("all six levels", g), ("without 3.178 (AWE pilot)", g[g.i_min_per_bar < 3.0])):
            b, a = np.polyfit(gg.i_min_per_bar, gg.lambda_SU_mean, 1)
            r2 = np.corrcoef(gg.i_min_per_bar, gg.lambda_SU_mean)[0, 1] ** 2
            sl.append(dict(site=site, policy=pol, levels=lab, slope_pp_per_min_per_bar=b, intercept_pp=a, R2=r2))
    sl = pd.DataFrame(sl)
    # falsification test
    lo = raw[(raw.policy == "A") & (raw.variant == "i_0.120_L32_end")]
    mn = lo.lambda_SU.min()
    verdict = "FALSIFICATION TRIGGERED: POL-A lambda_SU < 1.0 % at i = 0.120 at >= 1 site-year -> the POL-A materiality statement must carry the inventory qualification in the abstract and conclusion" \
        if (lo.lambda_SU < 1.0).any() else "NOT TRIGGERED: POL-A lambda_SU >= 1.0 % at i = 0.120 at every site-year"
    ver = pd.DataFrame([dict(rule="D43d: POL-A lambda_SU at i = 0.120 min/bar < 1.0 % at any site-year", n_site_years=len(lo), lambda_SU_min_pct=mn, lambda_SU_max_pct=lo.lambda_SU.max(),
                             n_below_1pct=int((lo.lambda_SU < 1.0).sum()), verdict=verdict)])
    for k, v in lo.groupby("site").lambda_SU.agg(["min", "max"]).iterrows():
        ver[f"{k}_min"] = v["min"]
        ver[f"{k}_max"] = v["max"]
    s.to_csv(out / "inventory_oat.csv", index=False, float_format="%.6g")
    sl.to_csv(out / "inventory_oat_slopes.csv", index=False, float_format="%.6g")
    ver.to_csv(out / "inventory_falsification.csv", index=False, float_format="%.6g")
    # ---- estimator
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    f = ann[ann.model == "full"].copy()
    rows = []
    for r in f.itertuples():
        par = pr.build_params(r.policy)
        rows.append(dict(source="nominal", site=r.site, year=r.year, policy=r.policy, variant="nominal", M_stack=r.M_stack, N_complete=r.N_r_completed, mr=par[pl.MRATED_MIN],
                         i=par[pl.I_INV], p_op=par[pl.P_OP], p0=par[pl.P0], theta=par[pl.THETA_SPEC], lambda_engine=r.lambda_SU))
    for r in raw.itertuples():
        rows.append(dict(source="OAT", site=r.site, year=r.year, policy=r.policy, variant=r.variant, M_stack=r.M_stack, N_complete=r.N_r_completed, mr=r.mr_kg_min, i=r.i_inv,
                         p_op=r.p_op_bar, p0=r.p0_bar, theta=r.theta_spec, lambda_engine=r.lambda_SU))
    e = pd.DataFrame(rows)
    e["lambda_hat"] = e.N_complete * e.mr * (e.i * (e.p_op - e.p0) + e.theta) / e.M_stack * 100.0
    e["rel_err"] = (e.lambda_hat - e.lambda_engine) / e.lambda_engine
    e.to_csv(out / "estimator_runs.csv", index=False, float_format="%.6g")
    summ = e.groupby(["source", "policy"]).rel_err.agg(["count", "mean", "median", "min", "max"]).reset_index()
    summ["mean_abs_rel_err"] = e.groupby(["source", "policy"]).rel_err.apply(lambda x: x.abs().mean()).values
    summ2 = e.groupby(["source", "policy", "variant"]).rel_err.agg(["mean", "min", "max"]).reset_index()
    summ.to_csv(out / "estimator.csv", index=False, float_format="%.6g")
    summ2.to_csv(out / "estimator_by_variant.csv", index=False, float_format="%.6g")
    print(ver.T.to_string())
    print(sl.round(3).to_string())
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()

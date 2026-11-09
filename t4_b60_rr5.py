"""Brief 7 item 4a, 4b (D67g): POL-B60 nominal cells and RR5 cells for POL-B60 and POL-D.

    python code/t4_b60_rr5.py [--workers N]

(a) results/nominal/b60_nominal.csv: per site-year, POL-B60 (variant POL-B_60, t_su_hot nominal, standby 0.095 P_rated) from results/policy/polB_pareto.csv (lambda_SU, tau_eq_median, eta_del, M_del)
    and B-S3 nominal by a direct engine run (s3_config) at the POL-B60 parameters (17 runs); beta against the annual baselines (B-CL 1 min, B-EP, B-S1, B-S2 curves: policy-independent in annual.csv) and B-S3.
(b) results/sens/rr5_b60_d.csv: 34 runs (POL-B60 and POL-D, 17 site-years) at i = 0.120 min/bar, P_OP = 30 bar, theta_spec = 6.5 rated-min (same override as falsification.py grid);
    control: the same override for POL-A on all 17 site-years must equal falsification_grid.csv (results/sens/rr5_control_polA.csv).
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2su import plant as pl  # noqa: E402
from h2su import plantrun as pr  # noqa: E402
from h2su.io import RESULTS  # noqa: E402
from h2su.pv import pv_dc_series  # noqa: E402
from sweeps import run_grid, site_years  # noqa: E402

OVER = dict(THETA_SPEC=6.5, I_INV=0.120, P_OP=30.0)


def _s3(args):
    s, y = args
    df = pv_dc_series(s, y)
    par = pr.build_params("B", t_sb_min=60.0)
    p3, sw3 = pr.s3_config(par)
    tot = pr.run(df, p3, sw3)[0]
    stack = tot[pl.T_STACK]
    loss = tot[pl.T_PURGE_] + tot[pl.T_OFFSPEC] + tot[pl.T_DEOXO] + tot[pl.T_DRYER] + tot[pl.T_BLOWDOWN]
    return s, y, float(tot[pl.T_DEL]), abs(stack - (tot[pl.T_DEL] + loss + tot[pl.T_DINV])) / stack


def nominal_b60(workers):
    pb = pd.read_csv(RESULTS / "policy" / "polB_pareto.csv")
    pb = pb[(pb.variant == "POL-B_60") & np.isclose(pb.sb_f, 0.095, atol=2e-3) & (pb.t_su_hot.astype(str) == "nominal")].copy()
    assert len(pb) == 17, len(pb)
    with mp.Pool(workers) as pool:
        r = pool.map(_s3, site_years())
    s3 = pd.DataFrame(r, columns=["site", "year", "M_del_B_S3", "closure_H2_B_S3"])
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    a = ann[ann.policy == "A"].pivot_table(index=["site", "year"], columns="model", values="M_del").reset_index()
    d = pb[["site", "year", "lambda_SU", "tau_eq_median", "eta_del", "M_del", "M_stack", "closure_H2_rel", "closure_E_rel"]].merge(s3, on=["site", "year"]).merge(
        a[["site", "year", "B-CL_1min", "B-EP", "B-S1", "B-S2_curve1", "B-S2_curve2", "B-S2_curve3"]], on=["site", "year"])
    for nm, c in (("B_CL", "B-CL_1min"), ("B_EP", "B-EP"), ("B_S1", "B-S1"), ("B_S2_curve1", "B-S2_curve1"), ("B_S2_curve2", "B-S2_curve2"), ("B_S2_curve3", "B-S2_curve3")):
        d[f"beta_{nm}"] = (d[c] - d.M_del) / d.M_del * 100.0
    d["beta_B_S3"] = (d.M_del_B_S3 - d.M_del) / d.M_del * 100.0
    d.insert(0, "policy", "B60")
    d.to_csv(RESULTS / "nominal" / "b60_nominal.csv", index=False, float_format="%.8g")
    print("b60 nominal: rows", len(d), "max closure H2", d[["closure_H2_rel", "closure_H2_B_S3"]].max().max())
    return d


def rr5(workers):
    specs = [dict(name="rr5_B60", pol="B", t_sb=60.0, over=OVER, tags=dict(theta_req=6.5, i_req=0.12, p_op_req=30.0, policy_label="B60")),
             dict(name="rr5_D", pol="D", over=OVER, tags=dict(theta_req=6.5, i_req=0.12, p_op_req=30.0, policy_label="D")),
             dict(name="rr5_control_A", pol="A", over=OVER, tags=dict(theta_req=6.5, i_req=0.12, p_op_req=30.0, policy_label="A"))]
    df = run_grid(specs, workers=workers)
    df["policy"] = df.policy_label
    ctl = df[df.variant == "rr5_control_A"]
    out = df[df.variant != "rr5_control_A"].drop(columns=["policy_label"])
    g = pd.read_csv(RESULTS / "sens" / "falsification_grid.csv")
    g = g[(g.i_req == 0.12) & (g.p_op_req == 30) & (g.theta_req == 6.5) & (g.policy == "A")].set_index(["site", "year"])
    c = ctl.set_index(["site", "year"])
    cmp_ = pd.DataFrame({"lambda_SU_rerun": c.lambda_SU, "lambda_SU_grid": g.lambda_SU.loc[c.index], "M_del_rerun": c.M_del, "M_del_grid": g.M_del.loc[c.index]})
    cmp_["rel_diff_lambda_SU"] = (cmp_.lambda_SU_rerun - cmp_.lambda_SU_grid).abs() / cmp_.lambda_SU_grid
    cmp_["rel_diff_M_del"] = (cmp_.M_del_rerun - cmp_.M_del_grid).abs() / cmp_.M_del_grid
    out.to_csv(RESULTS / "sens" / "rr5_b60_d.csv", index=False)
    cmp_.reset_index().to_csv(RESULTS / "sens" / "rr5_control_polA.csv", index=False, float_format="%.10g")
    print("rr5: rows", len(out), "max closure H2", out.closure_H2_rel.max(), "E", out.closure_E_rel.max(), "| control POL-A vs grid, max rel diff lambda_SU", cmp_.rel_diff_lambda_SU.max(), "M_del", cmp_.rel_diff_M_del.max())
    print(out.groupby(["site", "policy"]).lambda_SU.agg(["mean", "min", "max", "count"]).round(3).to_string())


if __name__ == "__main__":
    mp.freeze_support()
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    nominal_b60(w)
    rr5(w)

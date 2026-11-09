"""Side brief S item 2 (D72c): contribution of the first cycle of each simulated year (row i_open = 0) to lambda_SU.

    python code/first_cycle_term.py

Reads results/nominal/cycles_<site>_<year>_<policy>.parquet and results/nominal/annual.csv (model == full), policies A, C, D.
Per site-year: m_SU of the row i_open == 0, its class, pp of M_stack, share of lambda_SU; count, mean m_SU and shortest duration of the other cycles with m_SU > 0.
Per site x policy: the same, summed over the site-years (level = 'site_policy'). Output: results/nominal/first_cycle_term.csv. No engine or model change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su.io import RESULTS  # noqa: E402

NOM = RESULTS / "nominal"
POLS = ("A", "C", "D")


def site_year_rows(ann: pd.DataFrame) -> pd.DataFrame:
    f = ann[(ann.model == "full") & ann.policy.isin(POLS)]
    rows = []
    for _, r in f.sort_values(["site", "policy", "year"]).iterrows():
        c = pd.read_parquet(NOM / f"cycles_{r.site}_{int(r.year)}_{r.policy}.parquet")
        first = c[c.i_open == 0]
        assert len(first) == 1, (r.site, r.year, r.policy, len(first))
        first = first.iloc[0]
        oth = c[(c.i_open != 0) & (c.m_SU_kg > 0)]
        dur_h = (oth.t_close - oth.t_open).dt.total_seconds() / 3600.0
        tot = float(c.m_SU_kg.sum())
        rows.append(dict(level="site_year", site=r.site, policy=r.policy, year=int(r.year), M_stack_kg=float(r.M_stack),
                         lambda_SU_pct=float(r.lambda_SU), lambda_SU_from_cycles_pct=tot / float(r.M_stack) * 100.0,
                         first_m_SU_kg=float(first.m_SU_kg), first_class_0hot_1warm_2cold=int(first.class_0hot_1warm_2cold),
                         first_pp_of_M_stack=float(first.m_SU_kg) / float(r.M_stack) * 100.0,
                         first_share_of_lambda_SU_pct=(float(first.m_SU_kg) / tot * 100.0) if tot > 0 else np.nan,
                         other_positive_cycles=int(len(oth)), other_positive_m_SU_sum_kg=float(oth.m_SU_kg.sum()),
                         other_positive_m_SU_mean_kg=float(oth.m_SU_kg.mean()) if len(oth) else np.nan,
                         other_positive_shortest_h=float(dur_h.min()) if len(oth) else np.nan))
    return pd.DataFrame(rows)


def site_policy_rows(sy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, p), g in sy.groupby(["site", "policy"]):
        tot_cycle_kg = (g.lambda_SU_from_cycles_pct / 100.0 * g.M_stack_kg).sum()
        n_oth = int(g.other_positive_cycles.sum())
        rows.append(dict(level="site_policy", site=s, policy=p, year=-1, M_stack_kg=float(g.M_stack_kg.sum()),
                         lambda_SU_pct=float(g.lambda_SU_pct.mean()), lambda_SU_from_cycles_pct=float(g.lambda_SU_from_cycles_pct.mean()),
                         first_m_SU_kg=float(g.first_m_SU_kg.mean()), first_class_0hot_1warm_2cold=int(g.first_class_0hot_1warm_2cold.mode().iloc[0]),
                         first_pp_of_M_stack=float(g.first_pp_of_M_stack.mean()),
                         first_share_of_lambda_SU_pct=float(g.first_share_of_lambda_SU_pct.mean()),   # mean of the per-site-year shares
                         first_share_pooled_pct=float(g.first_m_SU_kg.sum() / tot_cycle_kg * 100.0) if tot_cycle_kg > 0 else np.nan,   # sum over sum
                         other_positive_cycles=n_oth, other_positive_m_SU_sum_kg=float(g.other_positive_m_SU_sum_kg.sum()),
                         other_positive_m_SU_mean_kg=float(g.other_positive_m_SU_sum_kg.sum() / n_oth) if n_oth else np.nan,
                         other_positive_shortest_h=float(g.other_positive_shortest_h.min()) if n_oth else np.nan))
    return pd.DataFrame(rows)


def main():
    ann = pd.read_csv(NOM / "annual.csv")
    sy = site_year_rows(ann)
    out = pd.concat([sy, site_policy_rows(sy)], ignore_index=True)
    out.to_csv(NOM / "first_cycle_term.csv", index=False, float_format="%.6g")
    d = out[out.policy == "D"]
    print("max |lambda_SU - lambda_SU_from_cycles| (pp):", float((sy.lambda_SU_pct - sy.lambda_SU_from_cycles_pct).abs().max()))
    for pol in POLS:
        g = sy[sy.policy == pol]
        print(pol, "first-cycle pp range by site:",
              {s: (round(x.first_pp_of_M_stack.min(), 4), round(x.first_pp_of_M_stack.max(), 4)) for s, x in g.groupby("site")})
    print(out[out.level == "site_policy"][["site", "policy", "first_pp_of_M_stack", "first_share_of_lambda_SU_pct", "other_positive_cycles",
                                           "other_positive_m_SU_mean_kg", "other_positive_shortest_h"]].to_string())
    return len(out)


if __name__ == "__main__":
    print("rows", main())

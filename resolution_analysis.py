"""Brief 4 item 10 summaries from results/resolution/resolution_raw.csv (python code/sweeps.py resolution).

Outputs: results/resolution/{resolution_summary,factorial_2x2,c4_comparable,cost}.csv
 * N_r(dt)/N_r(1 min) with N_r = START entries (attempts) and completed restarts; "switching events suppressed" = 1 - ratio (C4 metric, L5 Fig 8d)
 * lambda_SU(dt), beta_B-S1(dt) (B-S1 on the same input), H2 differences vs 1 min
 * 2x2 factorial {1, 60 min} x {B-S1, event model} on delivered H2: main effects and interaction, % of the 1-min event model
 * C4-comparable: hourly-vs-minute H2 difference (stack H2 = C4 'H2 yield' class, delivered H2) and switching-event suppression, full year and summer windows
 * cost: wall time per configuration and accuracy loss vs the 1-min run
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su.io import RESULTS  # noqa: E402

OUT = RESULTS / "resolution"


def main():
    r = pd.read_csv(OUT / "resolution_raw.csv")
    full = r[r.window == "full year"].copy()
    key = ["site", "year", "policy"]
    base = full[full.block_min == 1].set_index(key)
    full["N_att_ratio"] = full.apply(lambda x: x.N_r_attempts / base.loc[(x.site, x.year, x.policy), "N_r_attempts"], axis=1)
    full["N_comp_ratio"] = full.apply(lambda x: x.N_r_completed / base.loc[(x.site, x.year, x.policy), "N_r_completed"], axis=1)
    for c in ("M_stack", "M_del"):
        full[f"d_{c}_pct_vs_1min"] = full.apply(lambda x: (x[c] - base.loc[(x.site, x.year, x.policy), c]) / base.loc[(x.site, x.year, x.policy), c] * 100, axis=1)
    full["d_lambda_SU_pp_vs_1min"] = full.apply(lambda x: x.lambda_SU - base.loc[(x.site, x.year, x.policy), "lambda_SU"], axis=1)
    g = full.groupby(["site", "policy", "block_min"])
    summ = g.agg(lambda_SU_mean=("lambda_SU", "mean"), lambda_SU_std=("lambda_SU", "std"), N_att_ratio_mean=("N_att_ratio", "mean"), N_att_ratio_min=("N_att_ratio", "min"),
                 N_att_ratio_max=("N_att_ratio", "max"), N_comp_ratio_mean=("N_comp_ratio", "mean"), d_M_stack_pct_mean=("d_M_stack_pct_vs_1min", "mean"),
                 d_M_del_pct_mean=("d_M_del_pct_vs_1min", "mean"), d_M_del_pct_std=("d_M_del_pct_vs_1min", "std"), d_lambda_SU_pp_mean=("d_lambda_SU_pp_vs_1min", "mean"),
                 beta_B_S1_mean=("beta_B_S1", "mean"), beta_B_S1_std=("beta_B_S1", "std"), n_site_years=("year", "count")).reset_index()
    summ["switching_events_suppressed_pct"] = (1.0 - summ.N_att_ratio_mean) * 100.0
    summ.to_csv(OUT / "resolution_summary.csv", index=False, float_format="%.6g")
    # ---- 2x2 factorial on delivered H2
    rows = []
    for (s, y, p), b in full.groupby(key):
        b = b.set_index("block_min")
        y_ev1, y_ev60, y_s1_1, y_s1_60 = b.loc[1, "M_del"], b.loc[60, "M_del"], b.loc[1, "M_del_S1"], b.loc[60, "M_del_S1"]
        ref = y_ev1
        res_eff = ((y_ev60 - y_ev1) + (y_s1_60 - y_s1_1)) / 2
        mod_eff = ((y_ev1 - y_s1_1) + (y_ev60 - y_s1_60)) / 2
        inter = (y_ev60 - y_ev1) - (y_s1_60 - y_s1_1)
        rows.append(dict(site=s, year=y, policy=p, M_del_event_1min=y_ev1, M_del_event_60min=y_ev60, M_del_S1_1min=y_s1_1, M_del_S1_60min=y_s1_60,
                         main_effect_resolution_pct=res_eff / ref * 100, main_effect_model_pct=mod_eff / ref * 100, interaction_pct=inter / ref * 100,
                         resolution_effect_in_event_model_pct=(y_ev60 - y_ev1) / ref * 100, resolution_effect_in_S1_pct=(y_s1_60 - y_s1_1) / ref * 100,
                         model_effect_at_1min_pct=(y_ev1 - y_s1_1) / ref * 100, model_effect_at_60min_pct=(y_ev60 - y_s1_60) / ref * 100))
    fac = pd.DataFrame(rows)
    fac.to_csv(OUT / "factorial_2x2.csv", index=False, float_format="%.6g")
    # ---- C4-comparable quantities (full year + summer windows)
    rows = []
    for win, sub in (("full year", r[r.window == "full year"]), ("summer window", r[r.window.isin(["JJA", "DJF"])])):
        s1 = sub[sub.block_min == 1].set_index(key)
        s60 = sub[sub.block_min == 60].set_index(key)
        if win == "summer window":
            s1 = s1.copy()
        for k in s1.index.intersection(s60.index):
            a, b = s1.loc[k], s60.loc[k]
            rows.append(dict(window=win, site=k[0], year=k[1], policy=k[2], hourly_minus_minute_M_stack_pct=(b.M_stack - a.M_stack) / a.M_stack * 100,
                             hourly_minus_minute_M_del_pct=(b.M_del - a.M_del) / a.M_del * 100, N_att_1min=a.N_r_attempts, N_att_hourly=b.N_r_attempts,
                             switching_events_suppressed_pct=(1 - b.N_r_attempts / a.N_r_attempts) * 100 if a.N_r_attempts > 0 else np.nan))
    c4 = pd.DataFrame(rows)
    c4s = c4.groupby(["window", "site", "policy"]).agg(hourly_minus_minute_M_stack_pct=("hourly_minus_minute_M_stack_pct", "mean"), hourly_minus_minute_M_del_pct=("hourly_minus_minute_M_del_pct", "mean"),
                                                      switching_events_suppressed_pct=("switching_events_suppressed_pct", "mean"), n=("year", "count")).reset_index()
    c4s["C4_reference"] = "L5 (C4): hourly overestimates H2 by 4.2-6.8 % at Lyngby (Fig 8); hourly suppresses ~96 % (Lyngby) / 63 % (CNR) of switching events (Fig 8d); summer windows, 4 sites incl. Gobabeb (KNN-downscaled ERA5)"
    c4.to_csv(OUT / "c4_comparable_per_siteyear.csv", index=False, float_format="%.6g")
    c4s.to_csv(OUT / "c4_comparable.csv", index=False, float_format="%.6g")
    # ---- cost
    cost = full.groupby(["policy", "block_min"]).agg(wall_s_mean=("wall_s", "mean"), wall_s_min=("wall_s", "min"), wall_s_max=("wall_s", "max"), wall_s_median=("wall_s", "median")).reset_index()
    acc = full.groupby(["policy", "block_min"]).agg(abs_d_M_del_pct_mean=("d_M_del_pct_vs_1min", lambda x: x.abs().mean()), abs_d_lambda_SU_pp_mean=("d_lambda_SU_pp_vs_1min", lambda x: x.abs().mean())).reset_index()
    cost = cost.merge(acc, on=["policy", "block_min"])
    cost["note"] = "wall time of ONE annual engine run (1-min solver grid for every block size; the block mean only changes the input) per configuration; single process, 12 workers in parallel on the laptop, numba compiled"
    # Brief 12 item 1 (D89): the note of the A/1 cell is built from the data (single wall-time measurements vary with machine load)
    wa = full[(full.policy == "A") & (full.block_min == 1)].wall_s
    wh = wa[wa > 0.5]
    cost.loc[(cost.policy == "A") & (cost.block_min == 1), "note"] += (f"; {len(wh)} of these {len(wa)} rows lie above 0.5 s ({wh.min():.3f}-{wh.max():.3f} s); "
                                                                      "wall time is a single measurement that varies with machine load; this cell does not estimate run cost")
    cost.to_csv(OUT / "cost.csv", index=False, float_format="%.6g")
    # Brief 9 item 2 (D78): wall time over all full-year rows of the five block variants (dt1, dt5, dt15, dt30, dt60)
    w = full.wall_s
    hi = w > 0.5
    pd.DataFrame([dict(n=len(w), min_s=w.min(), median_s=w.median(), max_s=w.max(), n_above_0p5_s=int(hi.sum()), max_of_rest_s=w[~hi].max(),
                       variants="dt1, dt5, dt15, dt30, dt60", source="results/resolution/resolution_raw.csv, window = full year")]).to_csv(OUT / "cost_summary.csv", index=False, float_format="%.6g")
    print(summ[summ.block_min.isin([1, 60])].round(3).to_string())
    print(fac.groupby("policy")[["main_effect_resolution_pct", "main_effect_model_pct", "interaction_pct"]].mean().round(3))
    print(c4s.round(3).to_string())
    print(cost.round(4).to_string())


if __name__ == "__main__":
    main()

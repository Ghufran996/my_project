"""Brief 7 item 5b, 5c, 5d (D66, D67c, d, D68): capacity factors, timestep decomposition, delivered-H2 resolution effect in C3's basis.

    python code/c3_timestep.py

Outputs (results/resolution/):
 c3_site_cf.csv                   per site-year and ratio: DC capacity factor (mean 1-min DC power / PV peak rating, before any clipping), clipped energy share of the 5-min block means at P_cap;
                                  site means; C3 farm capacity factors (L4 Table 2) and model values (results/ver2r_stdout.txt).
 c3_timestep_decomposition.csv    fixed75 curve, ratios 0.667, 0.75, 1.0, per site-year and site mean: E5, E60 (energy of the unclipped block means, conserved by averaging), clipping-only, P_min-threshold-only
                                  and combined effects of 60-min against 5-min input on H2 at fixed efficiency (no start-up penalty, no standby, no dispatch optimisation), % of the 5-min value; remainder =
                                  file value (c3_comparable.csv, DP dispatch with start-up penalty) minus the combined effect.
 delivered_60v5.csv               M_del and M_stack of the plant engine, 60-min against 5-min input (variants dt60, dt5, full year), % of the 5-min value, POL-A, C, D; per site-year and site mean.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2su.baselines import bs2_c3 as b2  # noqa: E402
from h2su.io import CONFIGS, RESULTS  # noqa: E402
from h2su.pv import pv_dc_series  # noqa: E402
from sweeps import site_years  # noqa: E402

RATIOS = (1.0 / 1.5, 0.75, 1.0)
OUT = RESULTS / "resolution"


def h2_fixed(x, thr):
    """Relative H2 at fixed efficiency: sum of the powers at or above the threshold (dt constant, efficiency constant, so proportional to the energy used)."""
    return float(np.where(x >= thr - 1e-12, x, 0.0).sum())


def block_energy(x, n):
    return float(b2.block_mean(x, n).sum()) * n          # MWh x 60 (sum of block means x block length in minutes of 1-min samples)


def main():
    cf_rows, dec_rows = [], []
    c3 = pd.read_csv(OUT / "c3_comparable.csv")
    for s, y in site_years():
        dc = pv_dc_series(s, y).p_dc_kw.to_numpy() / 1000.0 * (1.0 / 1.5)              # MW at PV 1.0 MWp
        for ratio in RATIOS:
            p = dc / ratio                                                             # MW, PV rating 1/ratio MWp, electrolyser P_cap = 1 MW
            pv_peak = 1.0 / ratio
            x5u, x60u = b2.block_mean(p, 5), b2.block_mean(p, 60)
            x5, x60 = np.minimum(x5u, 1.0), np.minimum(x60u, 1.0)
            E5u, E60u = float(x5u.sum()) * 5.0, float(x60u.sum()) * 60.0               # MW-min
            cf_rows.append(dict(kind="plant site-year", site=s, year=y, electrolyser_over_pv_ratio=ratio, pv_peak_MWp=pv_peak, cf_dc_pct=float(p.mean()) / pv_peak * 100.0,
                                clipped_share_5min_pct=(E5u - float(x5.sum()) * 5.0) / E5u * 100.0, source="mean 1-min DC power / PV peak rating; clipping of the 5-min block means at P_cap = 1 MW"))
            thr = b2.P_MIN
            h5c, h60c = h2_fixed(x5, 0.0), h2_fixed(x60, 0.0)                          # clipping only
            h5t, h60t = h2_fixed(x5u, thr), h2_fixed(x60u, thr)                        # threshold only
            h5b, h60b = h2_fixed(x5, thr), h2_fixed(x60, thr)                          # both
            # block lengths differ (5 vs 60 min): energy = sum of block means x block length
            f = lambda a5, a60: (a60 * 60.0 - a5 * 5.0) / (a5 * 5.0) * 100.0
            file_v = c3[(c3.site == s) & (c3.year == y) & (c3.scenario == "fixed75") & np.isclose(c3.electrolyser_over_pv_ratio, ratio)].d_60_minus_5_pct_of_5min.iloc[0]
            both = f(h5b, h60b)
            dec_rows.append(dict(kind="site-year", site=s, year=y, electrolyser_over_pv_ratio=ratio, E5_unclipped_MWmin=E5u, E60_unclipped_MWmin=E60u, E_rel_diff=abs(E60u - E5u) / E5u,
                                 clipping_only_pct=f(h5c, h60c), pmin_threshold_only_pct=f(h5t, h60t), both_pct=both, file_value_pct=file_v, remainder_pct=file_v - both))
    cf = pd.DataFrame(cf_rows)
    dec = pd.DataFrame(dec_rows)
    # site means
    cm = cf.groupby(["site", "electrolyser_over_pv_ratio"], as_index=False).agg(pv_peak_MWp=("pv_peak_MWp", "first"), cf_dc_pct=("cf_dc_pct", "mean"), clipped_share_5min_pct=("clipped_share_5min_pct", "mean"))
    cm.insert(0, "kind", "plant site mean")
    cm["source"] = "mean over site-years (equal weight)"
    # C3 farms
    tab = pd.read_csv(CONFIGS / "tables" / "L4_virahsawmy2024.csv")
    t2 = tab[(tab.table == "Table2") & (tab.item == "capacity_factor_2022")].set_index("site").value.astype(float)
    stdout = (RESULTS / "ver2r_stdout.txt").read_text(encoding="utf-8").splitlines()[:5]
    farms = []
    for ln in stdout[1:]:
        m = re.match(r"\s*(Finley|Tailem Bend|Hamilton)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", ln)
        if m:
            farms.append(dict(kind="C3 farm (solar)", site=m.group(1), year=2022, cf_dc_pct=float(m.group(3)), cf_table2_pct=float(t2[m.group(1)]), source="cf_dc_pct = model value of VER-2R (AEMO 2022, results/ver2r_stdout.txt); cf_table2_pct = L4 Table 2"))
    assert len(farms) == 3, stdout
    cf_all = pd.concat([cf, cm, pd.DataFrame(farms)], ignore_index=True)
    cf_all.to_csv(OUT / "c3_site_cf.csv", index=False, float_format="%.8g")
    dm = dec.groupby(["site", "electrolyser_over_pv_ratio"], as_index=False)[["E5_unclipped_MWmin", "E60_unclipped_MWmin", "clipping_only_pct", "pmin_threshold_only_pct", "both_pct", "file_value_pct", "remainder_pct"]].mean()
    dm["E_rel_diff"] = dec.groupby(["site", "electrolyser_over_pv_ratio"]).E_rel_diff.max().to_numpy()
    dm.insert(0, "kind", "site mean (E_rel_diff = max over site-years)")
    pd.concat([dec, dm], ignore_index=True).to_csv(OUT / "c3_timestep_decomposition.csv", index=False, float_format="%.8g")
    print(cm.round(3).to_string(index=False))
    print(pd.DataFrame(farms)[["site", "cf_dc_pct", "cf_table2_pct"]].to_string(index=False))
    print(dm.round(3).to_string(index=False))
    print("max |E60 - E5|/E5 over all site-years and ratios:", dec.E_rel_diff.max())
    # (d) delivered H2, 60 vs 5 min
    r = pd.read_csv(OUT / "resolution_raw.csv")
    r = r[(r.window == "full year") & r.variant.isin(["dt5", "dt60"]) & r.policy.isin(["A", "C", "D"])]
    w = r.pivot_table(index=["site", "year", "policy"], columns="variant", values=["M_del", "M_stack"]).reset_index()
    w.columns = ["_".join([c for c in col if c]) for col in w.columns]
    w["delivered_60_minus_5_pct_of_5min"] = (w.M_del_dt60 - w.M_del_dt5) / w.M_del_dt5 * 100.0
    w["stack_60_minus_5_pct_of_5min"] = (w.M_stack_dt60 - w.M_stack_dt5) / w.M_stack_dt5 * 100.0
    w.insert(0, "kind", "site-year")
    m = w.groupby(["site", "policy"], as_index=False)[["delivered_60_minus_5_pct_of_5min", "stack_60_minus_5_pct_of_5min"]].mean()
    m.insert(0, "kind", "site mean")
    pd.concat([w, m], ignore_index=True).to_csv(OUT / "delivered_60v5.csv", index=False, float_format="%.8g")
    print(m.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

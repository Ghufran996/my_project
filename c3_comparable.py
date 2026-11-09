"""Brief 5 item 8: C3 (L4) comparison quantities on the plant DC power, 17 site-years, no new literature.

    python code/c3_comparable.py   -> results/resolution/c3_comparable.csv

B-S2 (C3 formulation: dispatch DP = MILP optimum, VER-2R, curves 1-3 with eta_max / eta_min from L4 Fig. 2, P_min 0.1 P_cap, standby 0.01 P_cap, 5-min start-up penalty, purification off) is run on
the plant DC power of each site-year (1.5 MWp, BSRN chain, electrolyser P_cap = 1 MW, i.e. electrolyser/PV ratio 0.667; C3 Table 5 uses 0.75, Table 3 uses 1.0: stated).
Inputs: 5-min and 60-min block means of the 1-min DC power (clipped at P_cap after averaging).
Metric A (C3 Table 5 type, percent form): (H2_60min - H2_5min)/H2_5min x 100 per curve (C3 prints tonnes; its percentages only in Fig. 7).
Metric B (C3 Table 3 type): (H2_curve - H2_fixed75)/H2_fixed75 x 100 at 5 min, reference = fixed 75 % LHV (the convention of C3 Table 3 and of D43a).
Both as 17 site-year rows and as site means.
Brief 7 item 5a (D67c, D68): ratio 0.75 rows added (C3 Sec. 4.2 / Table 5 timestep study ratio, as in VER-2R); the rows at 0.667 and 1.0 must reproduce the stored file (rtol 1e-6, matched on site, year, scenario, ratio).
Brief 6 item 7: the same curves and metrics are also run with the PV DC power scaled so that the electrolyser/PV ratio = 1.0 (PV 1.0 MWp = DC power x 1.0/1.5; C3 Table 3 convention), in addition to the
ratio 0.667 rows (unchanged; re-computed and compared with the stored file before it is overwritten). The column `electrolyser_over_pv_ratio` carries both values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2su import plantrun as pr  # noqa: E402
from h2su.baselines import bs2_c3 as b2  # noqa: E402
from h2su.io import CONFIGS, RESULTS  # noqa: E402
from h2su.pv import pv_dc_series  # noqa: E402
from sweeps import site_years  # noqa: E402


def main():
    tab = pd.read_csv(CONFIGS / "tables" / "L4_virahsawmy2024.csv")
    e = tab[tab.table == "Fig2"]
    emax = float(e[e.item == "eta_max"].value.iloc[0])
    emin = {int(r.curve): float(r.value) for r in e[e.item == "eta_min"].itertuples()}
    curves = {"fixed75": b2.Curve(emax, emax, True), **{f"curve{k}": b2.Curve(emax, emin[k]) for k in (1, 2, 3)}}
    rows = []
    old = RESULTS / "resolution" / "c3_comparable.csv"
    before = pd.read_csv(old) if old.exists() else None
    for s, y, ratio in [(s_, y_, r_) for r_ in (1.0 / 1.5, 0.75, 1.0) for s_, y_ in site_years()]:
        p = pv_dc_series(s, y).p_dc_kw.to_numpy() / 1000.0 * (1.0 / 1.5) / ratio     # MW; ratio 0.667 = PV 1.5 MWp as stored, ratio 1.0 = PV 1.0 MWp
        x5, x60 = (np.minimum(b2.block_mean(p, n), 1.0) for n in (5, 60))
        H = {}
        for nm, cv in curves.items():
            H[(nm, 5)] = b2.solve_dp(x5, 5 / 60.0, 1.0, cv)["H2_kg"]
            H[(nm, 60)] = b2.solve_dp(x60, 1.0, 1.0, cv)["H2_kg"]
        for nm in curves:
            rows.append(dict(site=s, year=y, scenario=nm, H2_5min_kg=H[(nm, 5)], H2_60min_kg=H[(nm, 60)], d_60_minus_5_pct_of_5min=(H[(nm, 60)] - H[(nm, 5)]) / H[(nm, 5)] * 100.0,
                             d_60_minus_5_kg=H[(nm, 60)] - H[(nm, 5)],
                             d_vs_fixed75_5min_pct=(H[(nm, 5)] - H[("fixed75", 5)]) / H[("fixed75", 5)] * 100.0 if nm != "fixed75" else 0.0,
                             d_vs_fixed75_60min_pct=(H[(nm, 60)] - H[("fixed75", 60)]) / H[("fixed75", 60)] * 100.0 if nm != "fixed75" else 0.0,
                             electrolyser_over_pv_ratio=ratio, convention="C3: reference = fixed 75 % LHV; 60-min = block mean of the 5-min series"))
        print(s, y, round(ratio, 3), flush=True)
    d = pd.DataFrame(rows)
    if before is not None:        # rows at ratio 0.667 and 1.0 must reproduce the stored rows (matched on keys)
        num = ["H2_5min_kg", "H2_60min_kg", "d_60_minus_5_pct_of_5min", "d_vs_fixed75_5min_pct", "d_vs_fixed75_60min_pct"]
        key = ["site", "year", "scenario"]
        for r_ in sorted(before.electrolyser_over_pv_ratio.unique()):
            a = d[np.isclose(d.electrolyser_over_pv_ratio, r_)].set_index(key).sort_index()
            b = before[np.isclose(before.electrolyser_over_pv_ratio, r_)].set_index(key).sort_index()
            assert a.index.equals(b.index) and np.allclose(a[num].to_numpy(float), b[num].to_numpy(float), rtol=1e-6, atol=1e-6), f"ratio {r_:.3f} rows changed"
            print(f"ratio {r_:.3f}: {len(a)} rows reproduce the stored file (rtol 1e-6, atol 1e-6)")
    out = RESULTS / "resolution"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "c3_comparable.csv", index=False, float_format="%.8g")
    m = d.groupby(["electrolyser_over_pv_ratio", "site", "scenario"])[["d_60_minus_5_pct_of_5min", "d_vs_fixed75_5min_pct", "d_vs_fixed75_60min_pct"]].mean().round(3)
    print(m.to_string())
    return d


if __name__ == "__main__":
    main()

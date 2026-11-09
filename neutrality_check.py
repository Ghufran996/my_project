"""Brief 5 rule: engine additions are neutral by default. Re-runs the 51 nominal full-model runs (17 site-years x POL-A/C/D) with the current engine and compares
every numeric column of metrics() that exists in results/nominal/annual.csv (R46) -> results/nominal/neutrality_check.csv; max relative difference must be <= 1e-12.

    python code/neutrality_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2su import plantrun as pr  # noqa: E402
from h2su.io import RESULTS  # noqa: E402
from h2su.pv import pv_dc_series  # noqa: E402
from sweeps import site_years  # noqa: E402


def main():
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    ann = ann[ann.model == "full"].set_index(["site", "year", "policy"])
    rows = []
    for s, y in site_years():
        df = pv_dc_series(s, y)
        for pol in "ACD":
            par = pr.build_params(pol)
            tot, cyc, nc, _ = pr.run(df, par, pr.switches())
            m = pr.metrics(tot, cyc, nc, par)
            ref = ann.loc[(s, y, pol)]
            worst = 0.0
            ncol = 0
            for k, v in m.items():
                if k in ref.index and np.isfinite(v) and np.isfinite(ref[k]):
                    d = abs(v - float(ref[k])) / max(abs(float(ref[k])), 1e-30) if abs(float(ref[k])) > 1e-12 else abs(v - float(ref[k]))
                    worst = max(worst, d)
                    ncol += 1
            rows.append(dict(site=s, year=y, policy=pol, n_columns=ncol, max_rel_diff=worst))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "nominal" / "neutrality_check.csv", index=False, float_format="%.3g")
    print("runs", len(out), "columns compared per run", int(out.n_columns.min()), "max relative difference", out.max_rel_diff.max())
    return out


if __name__ == "__main__":
    main()

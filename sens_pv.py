"""Brief 4 item 13: OAT PV-chain sensitivity for the SN26 items that obtained a source.

    python code/sens_pv.py  ->  results/sens/pv_chain.csv

Variants (each one change vs the nominal PV chain, same engine, POL-A and POL-C, all site-years):
  IAM            PVWatts V5 module-cover angular response, standard module (L56 Sec. 8 Eq. 3-6, n_glass = 1.526), applied to the direct beam only (L56 Sec. 8)
  DCDC_0.96      DC/DC efficiency 0.96 = the PVWatts V5 nominal INVERTER efficiency (L56 Sec. 'Inverter', default 0.96). ANALOGUE: an inverter value used for a DC/DC
                 converter, labelled as such; the nominal chain uses 1.0 (SN19/SN26 unsourced)
  IAM+DCDC_0.96  both
Albedo: only the PVWatts default 0.2 (L56 Sec. 7, fixed) is sourced; no site-specific albedo found -> no albedo variant (stated limitation, SN26 remains open for albedo).
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h2su.io import RESULTS  # noqa: E402
from h2su.pv import pv_dc_series  # noqa: E402
from sweeps import run_spec, site_years  # noqa: E402

VARS = {"nominal": dict(), "IAM": dict(iam=True, tag="_iam"), "DCDC_0.96": dict(dcdc=0.96, tag="_dcdc096"), "IAM+DCDC_0.96": dict(iam=True, dcdc=0.96, tag="_iam_dcdc096")}


def work(a):
    s, y = a
    out = []
    for nm, kw in VARS.items():
        df = pv_dc_series(s, y, **kw)
        e_pv = float(df.p_dc_kw.clip(upper=1000.0).sum() / 60.0)
        for pol in "AC":
            r = run_spec(df, dict(name=nm, pol=pol, s1=True))
            r.update(site=s, year=y, variant=nm, policy=pol, E_pv_clipped_kWh=e_pv)
            out.append(r)
    return out


def main():
    rows = []
    with mp.Pool(10) as pool:
        for r in pool.imap_unordered(work, site_years()):
            rows.extend(r)
    df = pd.DataFrame(rows).sort_values(["site", "year", "policy", "variant"]).reset_index(drop=True)
    nom = df[df.variant == "nominal"].set_index(["site", "year", "policy"])
    for c in ("M_del", "M_stack", "E_pv_clipped_kWh", "N_r_attempts", "lambda_SU"):
        df[f"d_{c}"] = df.apply(lambda r: r[c] - nom.loc[(r.site, r.year, r.policy), c], axis=1)
    df["d_M_del_pct"] = df.d_M_del / df.apply(lambda r: nom.loc[(r.site, r.year, r.policy), "M_del"], axis=1) * 100
    df["d_E_pv_pct"] = df.d_E_pv_clipped_kWh / df.apply(lambda r: nom.loc[(r.site, r.year, r.policy), "E_pv_clipped_kWh"], axis=1) * 100
    out = RESULTS / "sens"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pv_chain.csv", index=False)
    print(df[df.variant != "nominal"].groupby(["site", "policy", "variant"])[["d_E_pv_pct", "d_M_del_pct", "d_lambda_SU", "d_N_r_attempts"]].mean().round(3).to_string())


if __name__ == "__main__":
    mp.freeze_support()
    main()

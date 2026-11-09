"""Brief 4 item 2: VER-6c notes (D38) from existing files only (no COMSOL run).

 (a) first moment of the plug-flow outlet step response, m1 = integral_0^T (1 - c_out/c_ss) dt, c_ss = exp(-Da) (analytic steady outlet),
     Python (results/ver6/) and COMSOL (COMSOL_MATLAB/outputs/) vs the analytic L/v (v = u_s/eps);
 (b) Python mass balance recomputed with the COMSOL quadrature of verif_tracer_run.m (trapezoid on the exported output times,
     domain integral of c at those times) vs the COMSOL values of COMSOL_MATLAB/outputs/ver6_events.csv.
Output: results/ver6c_notes.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from h2su import ver6  # noqa: E402
from h2su.io import RESULTS, ROOT  # noqa: E402
from h2su.ver6 import params_for  # noqa: E402
from h2su.ver6_cases import build  # noqa: E402

COMSOL_OUT = ROOT / "COMSOL_MATLAB" / "outputs"
GRIDS = ["N025", "N050", "N100", "N200", "N400", "N400_r1e-04", "N400_r1e-08"]


def main():
    cases, rows = build()
    ev_c = pd.read_csv(COMSOL_OUT / "ver6_events.csv")
    out = []
    for case in [c for c in cases if c.group == "verif"]:
        p, g = params_for(case, rows)
        v = p["us_m_s"] / p["eps"]
        da = p["k1_s"] * p["L_m"] / p["us_m_s"]
        for n, rt, gid in ver6.levels():
            row, series, sol = ver6.run_verif(case, p, g, n, rt, gid, write=False)
            t = np.arange(0.0, p["t_end_s"] + 1e-9, p["dt_out_s"])
            h = p["L_m"] / n
            ct_int = h * sol["F"][0].sum(axis=0)
            c_out = sol["outlet"][0]
            flux_in = p["us_m_s"] * p["c_in_norm"] * (t[-1] - t[0])
            flux_out = np.trapezoid(p["us_m_s"] * c_out, t)
            consumed = p["k1_s"] * np.trapezoid(ct_int, t)
            accum = (p["eps"] + (1 - p["eps"]) * p["K_ads"]) * (ct_int[-1] - ct_int[0])
            e_quad = abs(flux_in - flux_out - consumed - accum) / max(flux_in, flux_out, consumed, abs(accum))
            e_comsol = ev_c[(ev_c.case_id == case.case_id) & (ev_c.grid_id == gid)]["mass_balance_err"]
            out.append(dict(quantity="mass_balance_err", case_id=case.case_id, grid_id=gid, python_in_code=row["mass_balance_err"], python_comsol_quadrature=e_quad,
                            comsol=float(e_comsol.iloc[0]) if len(e_comsol) else np.nan, analytic=np.nan,
                            note="python_in_code = exact discrete invariant (packedbed ext. states); python_comsol_quadrature = trapezoid on the exported output times as in verif_tracer_run.m"))
            if case.case_id == "ver_plugflow":
                css = np.exp(-da)
                m1_p = float(np.trapezoid(1 - series["y_O2_out"] / css, t))
                cs = pd.read_csv(COMSOL_OUT / f"{case.case_id}__comsol__{gid}.csv")
                m1_c = float(np.trapezoid(1 - cs["y_O2_out"].to_numpy() / css, cs["t_s"].to_numpy()))
                out.append(dict(quantity="first_moment_s", case_id=case.case_id, grid_id=gid, python_in_code=m1_p, python_comsol_quadrature=np.nan, comsol=m1_c,
                                analytic=p["L_m"] / v, note="m1 = int (1 - c_out/c_ss) dt, c_ss = exp(-Da); analytic L/v"))
    df = pd.DataFrame(out)
    df["rel_err_vs_analytic"] = np.where(df.analytic.notna(), (df.python_in_code - df.analytic) / df.analytic, np.nan)
    df["rel_err_comsol_vs_analytic"] = np.where(df.analytic.notna(), (df.comsol - df.analytic) / df.analytic, np.nan)
    df.to_csv(RESULTS / "ver6c_notes.csv", index=False, float_format="%.6g")
    print(df.drop(columns=["note"]).to_string())


if __name__ == "__main__":
    main()

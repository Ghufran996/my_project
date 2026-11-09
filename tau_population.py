"""Brief 7 item 3 (D66): nominal cycle populations behind tau_eq and the closed-form tau_mean check.

    python code/tau_population.py

Outputs: results/nominal/tau_population.csv (site x policy A, C, D: share of closed cycles with m_SU = 0, mean tau_eq of the cycles with m_SU > 0),
         results/nominal/tau_mean_closed_form_check.csv (site-year x policy: tau_mean = lambda_SU/100 x M_stack/(N_completed x mdot_rated) against the mean of tau_eq over the closed cycles).
tau_eq = (cycle loss - steady reference)/mdot_rated of a closed cycle [rated-min]; the closed form counts the aborted attempts and the unfinished window of lambda_SU in the numerator
and divides by completed restarts only, so the two differ by that part.
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
from sweeps import site_years  # noqa: E402

POLS = ("A", "C", "D")
MDOT = float(pr.build_params("A")[pr.pl.MRATED_MIN])


def main():
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    ann = ann[ann.model == "full"]
    pop, chk = [], []
    for site in ("Cabauw", "Carpentras", "Gobabeb"):
        for pol in POLS:
            frames = []
            for s, y in site_years():
                if s != site:
                    continue
                c = pd.read_parquet(RESULTS / "nominal" / f"cycles_{s}_{y}_{pol}.parquet")
                c = c[c.closed > 0.5]
                frames.append(c)
                a = ann[(ann.site == s) & (ann.year == y) & (ann.policy == pol)].iloc[0]
                tm = a.lambda_SU / 100.0 * a.M_stack / (a.N_r_completed * MDOT)
                te = float(c.tau_eq_min.mean())
                chk.append(dict(site=s, year=y, policy=pol, tau_mean_closed_form=tm, tau_eq_mean_over_closed_cycles=te, difference=tm - te, rel_difference_pct=(tm - te) / te * 100.0 if te > 0 else np.nan,
                                N_r_completed=a.N_r_completed, N_r_aborted=a.N_r_aborted, n_cycles_closed=len(c), lambda_SU_abort_pct=a.lambda_SU_abort))
            c = pd.concat(frames, ignore_index=True)
            pos = c[c.m_SU_kg > 0]
            pop.append(dict(site=site, policy=pol, n_site_years=len(frames), n_cycles_closed=len(c), share_closed_cycles_m_SU_zero_pct=float((c.m_SU_kg == 0).mean()) * 100.0,
                            n_cycles_m_SU_positive=len(pos), mean_tau_eq_cycles_m_SU_positive_rated_min=float(pos.tau_eq_min.mean()) if len(pos) else np.nan,
                            mean_tau_eq_all_closed_cycles_rated_min=float(c.tau_eq_min.mean()), median_tau_eq_all_closed_cycles_rated_min=float(c.tau_eq_min.median())))
    out = RESULTS / "nominal"
    p = pd.DataFrame(pop)
    k = pd.DataFrame(chk)
    p.to_csv(out / "tau_population.csv", index=False, float_format="%.6g")
    k.to_csv(out / "tau_mean_closed_form_check.csv", index=False, float_format="%.6g")
    print(p.round(3).to_string(index=False))
    print("closed form vs mean over closed cycles: n", len(k), "difference [rated-min] min/max", k.difference.min(), k.difference.max(), "rel % min/max", k.rel_difference_pct.min(), k.rel_difference_pct.max())


if __name__ == "__main__":
    main()

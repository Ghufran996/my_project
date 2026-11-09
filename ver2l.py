"""VER-2L link test (D31; Brief 3 item 7) -> results/ver2l_link.csv.

(i) [not executable since the Brief 6 cleanup deleted the pre-refactor file: value and passed empty, historical R32 result in the note] VER-2R re-run (code/ver2r.py after the refactor to the shared block-mean / accounting functions) reproduces every value of the pre-refactor
    results/ver2r_c3.csv to <= 1e-12 relative (the pre-refactor file is kept as results/_ver2r_c3_before_refactor.csv).
(ii) DP state/power paths (3 farms, 5 min, fixed / curve 1 / curve 3, full year) replayed through the plant engine in its C3 configuration
    (purification off; h2su.plantrun.run_c3, which calls h2su.shared.account_c3) give annual H2 and energy equal to the DP accounting to <= 1e-9.
    A third, independent numpy accounting (the pre-refactor formula) is compared as well.
Also asserted: bs2_c3.block_mean is h2su.shared.block_mean (one shared function).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2su import plantrun as pr  # noqa: E402
from h2su import shared  # noqa: E402
from h2su.baselines import bs2_c3 as b  # noqa: E402
from h2su.io import RESULTS  # noqa: E402
import ver2r as v  # noqa: E402


def numpy_account(pre, path, dt_h, p_cap, curve):
    """Independent numpy accounting (formula of the pre-refactor bs2_c3._account)."""
    on = path == 2
    p_el = np.where(on, np.minimum(pre, p_cap), 0.0)
    s1, s2 = b.segments(curve, p_cap)
    mdot = np.where(p_el <= 0.3 * p_cap, s1[0] * p_el + s1[1], s2[0] * p_el + s2[1]) * on
    prev = np.concatenate([[0], path[:-1]])
    return float(np.sum(mdot * dt_h)), float(np.sum(p_el * dt_h)), float(np.sum(path == 1) * b.P_SB * p_cap * dt_h), int((on & (prev == 0)).sum())


ROW_I_CHECK = "(i) VER-2R re-run vs pre-refactor ver2r_c3.csv (model, band, RE columns)"
ROW_I_NOTE = "not re-run: the pre-refactor file was deleted in the Brief 6 cleanup; historical result R32 (Brief 3): 0.0 relative difference, identical verdicts"


def row_i(pre=None):
    """Check (i). The pre-refactor file results/_ver2r_c3_before_refactor.csv was deleted in the Brief 6 cleanup, so the test cannot be executed: value and pass flag are empty
    (a pass flag for a test that did not run is not reported, D82-3); the note cites the historical result. If the file is present the comparison is run."""
    pre = RESULTS / "_ver2r_c3_before_refactor.csv" if pre is None else pre
    if not pre.exists():
        return dict(check=ROW_I_CHECK, value=float("nan"), tol=1e-12, passed=None, note=ROW_I_NOTE)
    old = pd.read_csv(pre)
    new = pd.read_csv(RESULTS / "ver2r_c3.csv")
    worst = 0.0
    for col in ("published", "model", "band_lo", "band_hi", "RE", "band_halfwidth_rel"):
        a, c = old[col].to_numpy(float), new[col].to_numpy(float)
        worst = max(worst, float(np.max(np.abs(a - c) / np.maximum(np.abs(a), 1e-300))))
    same_verdict = bool((old.verdict == new.verdict).all())
    return dict(check=ROW_I_CHECK, value=worst, tol=1e-12, passed=bool(worst <= 1e-12 and same_verdict), note=f"verdicts identical: {same_verdict}; block_mean is shared: True")


def main():
    rows = []
    assert b.block_mean is shared.block_mean
    rows.append(row_i())
    tab = pd.read_csv(v.TABLE)
    emax, emin, _ = v.curves(tab)
    curves = {"fixed": b.Curve(emax, emax, True), "curve1": b.Curve(emax, emin[1]), "curve3": b.Curve(emax, emin[3])}
    worst_h = worst_e = worst_np = 0.0
    n = 0
    for f, (duid, cap, _) in v.FARMS.items():
        s = v.load_farm(duid)
        x = s.fillna(0.0).clip(lower=0.0).to_numpy()
        for name, cv in curves.items():
            d = b.solve_dp(x, 5 / 60.0, cap, cv)
            h2, e_el, e_sb, nsu, hrs = pr.run_c3(x, d["path"], 5 / 60.0, cap, cv)
            rel_h = abs(h2 - d["H2_kg"]) / d["H2_kg"]
            rel_e = abs(e_el - d["electrolyser_MWh"]) / d["electrolyser_MWh"]
            rel_s = abs(e_sb - d["standby_MWh"]) / max(d["standby_MWh"], 1e-300) if d["standby_MWh"] > 0 else abs(e_sb)
            h2n, een, esn, nsn = numpy_account(x, d["path"], 5 / 60.0, cap, cv)
            worst_h, worst_e = max(worst_h, rel_h, rel_s), max(worst_e, rel_e)
            worst_np = max(worst_np, abs(h2n - d["H2_kg"]) / d["H2_kg"], abs(een - d["electrolyser_MWh"]) / d["electrolyser_MWh"])
            assert nsu == d["n_startups"] == nsn
            n += 1
            rows.append(dict(check=f"(ii) {f} {name}", value=max(rel_h, rel_e, rel_s), tol=1e-9, passed=bool(max(rel_h, rel_e, rel_s) <= 1e-9),
                             note=f"H2 DP {d['H2_kg']:.6f} kg, engine replay {h2:.6f} kg; E_el {e_el:.4f} MWh; startups {nsu}; standby MWh {e_sb:.4f}"))
    rows.append(dict(check="(ii) independent numpy accounting vs DP (all 9 paths)", value=worst_np, tol=1e-9, passed=bool(worst_np <= 1e-9), note="formula of the pre-refactor accounting"))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ver2l_link.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(out[["check", "value", "tol", "passed"]].to_string(index=False))
    print("VER-2L all pass (rows that ran):", bool(out.passed.dropna().all()), "; rows not run:", int(out.passed.isna().sum()))


if __name__ == "__main__":
    main()

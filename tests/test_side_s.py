"""Side brief S: T4 note text, first-cycle term, C3 load shape, T6 export, cost column (read result files; no engine run)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h2su.io import RESULTS, ROOT  # noqa: E402

_NEEDED = {"tables/T4_lambda_beta_intervals.csv": "step 47 tables", "tables/T6_comparison.csv": "step 47 tables", "nominal/first_cycle_term.csv": "step 43 first cycle",
           "resolution/c3_load_shape.csv": "step 44 C3 load shape", "resolution/cost.csv": "step 14 resolution analysis"}
_MISSING = [f"results/{k} ({v})" for k, v in _NEEDED.items() if not (RESULTS / k).exists()]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="result files absent: " + ", ".join(_MISSING) + "; run `python code/run_all.py final`")

BASE = ("tau_eq median = 0 means that more than half of the completed restarts of the sample are returns from hot standby with no excess loss; it does not mean zero start-up loss; use tau_mean. "
        "Nominal P_OP equals the upper bound of its range and nominal P_MIN_F the lower bound of its range.")


def test_t4_note_text_by_row():
    t = pd.read_csv(RESULTS / "tables" / "T4_lambda_beta_intervals.csv", keep_default_na=False)
    assert t.note.str.startswith(BASE).all()
    for _, r in t.iterrows():
        n = r.note
        if r.policy in ("A", "B60", "C"):
            assert n.endswith("The nominal lambda_SU lies high in the interval.") and "not high" not in n
        else:
            assert "The nominal lambda_SU is not high in the interval; see nominal_percentile_in_ensemble." in n and "lies high" not in n
        assert ("first cycle of each simulated year contributes 0.004-0.005 pp, more than the margin to the threshold." in n) == (r.site == "Cabauw" and r.policy == "D")
        assert ("Nominal 0.00236 % is below P5 0.00244 %; the value is the first cycle of each simulated year." in n) == (r.site == "Gobabeb" and r.policy == "D")
        assert ("P95 at the 0.5 % threshold: pooled 0.5008 %, per seed 0.4997-0.5021 %." in n) == (r.site == "Cabauw" and r.policy == "D")


def test_first_cycle_term_consistent_with_annual():
    f = pd.read_csv(RESULTS / "nominal" / "first_cycle_term.csv")
    sy = f[f.level == "site_year"]
    assert len(sy) == 51
    assert (sy.lambda_SU_pct - sy.lambda_SU_from_cycles_pct).abs().max() < 1e-9
    assert sy[sy.policy != "D"].first_pp_of_M_stack.max() < 0.03
    d = f[(f.level == "site_policy") & (f.policy == "D")].set_index("site")
    assert list(d.loc[["Cabauw", "Carpentras", "Gobabeb"], "other_positive_cycles"]) == [53, 11, 0]


def test_c3_load_shape_closed_form():
    f = pd.read_csv(RESULTS / "resolution" / "c3_load_shape.csv")
    # Brief 9 item 1 (D77): like-for-like pooled term; 0.04 pp is NOT met at Cabauw ratio 1.0 (0.0430 pp), recorded as a fact, tolerance not widened
    pooled = f[f.kind == "plant_site_pooled"]
    assert set(pooled.stored_basis) == {"pooled sums over site-years: 100 x (sum H2 curve k / sum H2 fixed75 - 1)"}
    assert abs(pooled[(pooled.site == "Cabauw") & np.isclose(pooled.ratio, 1.0)].diff_curve1_pp.iloc[0]) > 0.04
    assert pooled.diff_curve1_pp.abs().max() < 0.0431
    assert (f[f.kind == "farm"].diff_curve1_pp.abs() <= 0.17).all()
    sm = pd.read_csv(RESULTS / "resolution" / "c3_load_shape_summary.csv")
    assert (sm.n_closed_form_smaller_in_size == sm.n).all()
    assert sm.spearman_x_vs_curve1_term_12_rows.dropna().iloc[0] == -1.0
    assert np.isfinite(f[["x_share_above_0p3", "closed_form_curve2_pct", "closed_form_curve3_pct"]].to_numpy()).all()


def test_t6_equals_final_block_and_cost_median():
    txt = (ROOT / ".work" / "literature.md").read_text(encoding="utf-8")
    i = re.search(r"## Comparison table, Gate 2 (FINAL|DRAFT)", txt)
    assert i.group(1) == "FINAL"
    blk = [l for l in txt[i.start():].split("\n\n")[0].splitlines() if l.startswith("|")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in blk[2:]]
    t6 = pd.read_csv(RESULTS / "tables" / "T6_comparison.csv", keep_default_na=False, dtype=str)
    assert [list(r) for r in t6.values] == rows and t6.iloc[-1, 0] == "This work"
    c = pd.read_csv(RESULTS / "resolution" / "cost.csv")
    assert "wall_s_median" in c.columns and (c.wall_s_median > 0).all()


def test_c3_pooled_stored_term_two_year_synthetic():
    from c3_load_shape import pooled_stored_term
    sub = pd.DataFrame(dict(scenario=["fixed75", "curve1", "fixed75", "curve1"], H2_5min_kg=[100.0, 90.0, 300.0, 240.0]))
    # years: (100 -> 90) = -10 %, (300 -> 240) = -20 %; mean of yearly terms = -15 %, pooled sums: 330 / 400 - 1 = -17.5 %
    assert abs(pooled_stored_term(sub, 1) - (-17.5)) < 1e-12
    yearly = [(90 / 100 - 1) * 100, (240 / 300 - 1) * 100]
    assert abs(np.mean(yearly) - (-15.0)) < 1e-12


def test_cost_note_and_summary():
    c = pd.read_csv(RESULTS / "resolution" / "cost.csv")
    n = c[(c.policy == "A") & (c.block_min == 1)].note.iloc[0]
    # Brief 12 item 1 (D89): the note is data-derived; checked against resolution_raw.csv, not against a fixed string
    raw0 = pd.read_csv(RESULTS / "resolution" / "resolution_raw.csv")
    wa = raw0[(raw0.window == "full year") & (raw0.policy == "A") & (raw0.block_min == 1)].wall_s
    wh = wa[wa > 0.5]
    assert n.endswith(f"; {len(wh)} of these {len(wa)} rows lie above 0.5 s ({wh.min():.3f}-{wh.max():.3f} s); "
                      "wall time is a single measurement that varies with machine load; this cell does not estimate run cost")
    assert "first 12 in task order" not in n and "start-up cost" not in n
    s = pd.read_csv(RESULTS / "resolution" / "cost_summary.csv").iloc[0]
    # Brief 11 (R114): the wall times are machine-state measurements that every re-run of the resolution sweep re-measures (the Brief 9 partial chain gave a median of 0.299 s,
    # 106 rows above 0.5 s), so the summary is checked against its own source file, not against the values measured once (0.185 s, 12 rows, 0.246 s).
    raw = pd.read_csv(RESULTS / "resolution" / "resolution_raw.csv")
    w = raw[(raw.window == "full year") & raw.variant.isin(["dt1", "dt5", "dt15", "dt30", "dt60"])].wall_s
    assert s.n == len(w) == 255
    assert abs(s.min_s - w.min()) < 1e-5 and abs(s.median_s - w.median()) < 1e-5 and abs(s.max_s - w.max()) < 1e-5
    assert s.n_above_0p5_s == int((w > 0.5).sum()) and abs(s.max_of_rest_s - w[w <= 0.5].max()) < 1e-5


def test_t6_branch_without_literature_file(tmp_path, capsys):
    import tables_final as tf
    before = (tf.T / "T6_comparison.csv").read_bytes()
    df = tf.t6(tmp_path / "absent.md")
    assert "absent" in capsys.readouterr().out
    assert (tf.T / "T6_comparison.csv").read_bytes() == before and df.iloc[-1, 0] == "This work"

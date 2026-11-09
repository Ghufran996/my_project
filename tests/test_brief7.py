"""Brief 7 tests: beta pooling (concatenation before the percentile), site palette, tau_mean, and a value re-read of 3 cells per table (T4, T5) from their source files."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "src"))
_ROOT_RESULTS = CODE.parent / "results"
_NEEDED = {"inventory_ranges.csv": "step 1 brief3 chain", "tables/T4_lambda_beta_intervals.csv": "step 47 tables", "tables/T5_baselines.csv": "step 47 tables", "uq/beta_intervals.csv": "step 45 UQ summaries",
           "nominal/annual.csv": "step 6 nominal", "nominal/b60_nominal.csv": "step 40 POL-B60 nominal cells", "sens/rr5_b60_d.csv": "step 40 POL-B60 nominal cells"}
_MISSING = [f"results/{k} ({v})" for k, v in _NEEDED.items() if not (_ROOT_RESULTS / k).exists()]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="result files absent: " + ", ".join(_MISSING) + "; run `python code/run_all.py final`")
if _MISSING:   # uq_final_summary reads results/inventory_ranges.csv at import, so the module is skipped before the imports below
    pytest.skip("result files absent: " + ", ".join(_MISSING) + "; run `python code/run_all.py final`", allow_module_level=True)
import tables_final as tf  # noqa: E402
import uq_final_summary as ufs  # noqa: E402
from h2su.figures import SITE_COLORS, SITE_MARKERS, gray_level  # noqa: E402
from h2su.io import RESULTS  # noqa: E402

TABLES, UQ = RESULTS / "tables", RESULTS / "uq"


def test_beta_pool_concatenates_before_percentile():
    """Synthetic set: 2 site-years x 2 seeds. Year 1 low values, year 2 high values with different spread; the pooled percentile must come from the concatenation of both years per seed,
    then the mean over seeds. The old estimator (percentile per year, then mean over years) gives a different number."""
    rng = np.random.default_rng(1)
    y1 = {11: rng.normal(10.0, 1.0, 400), 22: rng.normal(10.0, 1.0, 400)}
    y2 = {11: rng.normal(30.0, 5.0, 100), 22: rng.normal(30.0, 5.0, 100)}
    by_seed = {sd: {"B": [y1[sd], y2[sd]]} for sd in (11, 22)}
    res = ufs.beta_pool(by_seed)["B"]
    expect = np.mean([np.percentile(np.concatenate([y1[sd], y2[sd]]), 95) for sd in (11, 22)])
    got = np.mean([res[sd][2] for sd in (11, 22)])
    old_estimator = np.mean([np.mean([np.percentile(y1[sd], 95), np.percentile(y2[sd], 95)]) for sd in (11, 22)])
    assert got == pytest.approx(expect, rel=1e-12)
    assert abs(got - old_estimator) > 1.0               # the two estimators differ materially on this set
    assert all(res[sd][3] == 500 and res[sd][4] == 2 for sd in (11, 22))     # 500 rows, 2 years per seed


def test_site_palette_gray_levels():
    cols = list(SITE_COLORS.values())
    assert SITE_COLORS == {"Cabauw": "#0072B2", "Carpentras": "#D55E00", "Gobabeb": "#000000"}
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(gray_level(cols[i]) - gray_level(cols[j])) >= 30.0
    assert len(set(SITE_MARKERS.values())) == 3


def test_tau_mean_from_stored_columns():
    f = UQ / "ens_Cabauw_2013_A_seed11.npz"
    if not f.exists():
        pytest.skip("checkpoint absent")
    z = np.load(f, allow_pickle=True)
    cols = [str(c) for c in z["cols"]]
    Y = z["Y"]
    lam, ms, nc = (Y[:, cols.index(c)] for c in ("lambda_SU", "M_stack", "N_r_completed"))
    tm = lam / 100.0 * ms / (nc * ufs.MDOT_RATED)
    assert np.all(np.isfinite(tm)) and tm.min() > 0
    # units: rated-min per completed restart; POL-A nominal is about 20-30 rated-min (D66), the ensemble median must be of that order
    assert 5.0 < np.median(tm) < 60.0


def _t4():
    return pd.read_csv(TABLES / "T4_lambda_beta_intervals.csv", keep_default_na=False)


def _t5():
    return pd.read_csv(TABLES / "T5_baselines.csv", keep_default_na=False)


def _close(a, b):
    return float(a) == pytest.approx(float(b), rel=2e-4, abs=1e-9)


def test_t4_no_empty_numeric_cell_and_no_not_applicable():
    t = _t4()
    assert "not applicable" not in " ".join(t.astype(str).values.ravel())
    num = [c for c in t.columns if c not in ("site", "policy", "SC_M_class", "interval_status", "note")]
    assert (t[num].astype(str).apply(lambda s: s.str.strip() != "").all()).all(), "empty cell in T4"
    assert not (t[num].astype(str) == "PENDING").any().any()
    assert t.note.str.startswith("tau_eq median = 0 means that more than half of the completed restarts").all()   # full text: test_side_s.py


def test_t4_three_cells_from_source_files():
    t = _t4().set_index(["site", "policy"])
    ip = pd.read_csv(UQ / "intervals_pooled.csv")
    ip = ip[ip.pv_ratio == 1.5].set_index(["site", "policy"])
    bi = pd.read_csv(UQ / "beta_intervals.csv").set_index(["site", "policy", "baseline"])
    b60 = pd.read_csv(RESULTS / "nominal" / "b60_nominal.csv")
    rr5 = pd.read_csv(RESULTS / "sens" / "rr5_b60_d.csv")
    # cell 1: pooled lambda_SU P5 and tau_mean P50 (intervals_pooled.csv)
    assert _close(t.loc[("Cabauw", "A"), "lambda_SU_P5"], ip.loc[("Cabauw", "A"), "lambda_SU_P5"])
    assert _close(t.loc[("Carpentras", "C"), "tau_mean_P50"], ip.loc[("Carpentras", "C"), "tau_mean_P50"])
    # cell 2: B-S2 interval and B-S3 interval (beta_intervals.csv)
    assert _close(t.loc[("Gobabeb", "D"), "beta_P95_B-S2curve3"], bi.loc[("Gobabeb", "D", "B-S2 curve3"), "beta_P95"])
    assert _close(t.loc[("Cabauw", "B60"), "beta_P5_B-S3"], bi.loc[("Cabauw", "B60", "B-S3"), "beta_P5"])
    # cell 3: POL-B60 nominal (polB_pareto-derived file) and RR5 cell for POL-D
    assert _close(t.loc[("Gobabeb", "B60"), "lambda_SU_nominal_mean"], b60[b60.site == "Gobabeb"].lambda_SU.mean())
    x = rr5[(rr5.site == "Carpentras") & (rr5.policy == "D")].lambda_SU
    assert _close(t.loc[("Carpentras", "D"), "lambda_SU_i0.120_30bar_theta6.5_mean"], x.mean())


def test_t5_no_not_applicable_and_three_cells_from_source_files():
    t = _t5()
    assert "not applicable" not in " ".join(t.astype(str).values.ravel())
    assert not (t.astype(str) == "PENDING").any().any()
    bi = pd.read_csv(UQ / "beta_intervals.csv").set_index(["site", "policy", "baseline"])
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    t = t.set_index("baseline")
    assert _close(t.loc["B-S2 curve 1", "beta_POLA_Cabauw_P5"], bi.loc[("Cabauw", "A", "B-S2 curve1"), "beta_P5"])
    assert _close(t.loc["B-S3", "beta_POLA_Gobabeb_P95"], bi.loc[("Gobabeb", "A", "B-S3"), "beta_P95"])
    nom = ann[(ann.site == "Carpentras") & (ann.policy == "A") & (ann.model == "full")].beta_B_S1.mean()
    assert float(t.loc["B-S1", "beta_POLA_Carpentras"]) == pytest.approx(nom, abs=0.051)       # table rounds the nominal to 0.1
    assert "n_seeds = 5" in t.loc["B-S3", "beta_unit"] and "auxiliaries from an auxiliary supply" in t.loc["B-EP", "beta_unit"]


def test_beta_table_check_extended():
    c = pd.read_csv(UQ / "beta_table_check.csv")
    assert c.rel_diff.max() <= 1e-3
    assert int((c.seed != 11).sum() // 3) >= 50          # 50 direct evaluations on seeds 22-55 (3 baselines each)

"""Tests of the L15 dryer model (D46): isotherm derivative, mass balance, frozen calibration reproduces BT2."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from h2su import dryer as dr  # noqa: E402
from h2su.io import CONFIGS, load_registry  # noqa: E402
from h2su.io import CONFIGS, RESULTS, load_registry  # noqa: E402

# Skipped when the frozen calibration file is absent (package without results/).
pytestmark = pytest.mark.skipif(not (RESULTS / "vt6_cal_frozen.json").exists(),
                                reason="results/vt6_cal_frozen.json is absent; run `python code/run_all.py final` (step 3, dryer VT6) to write it")


def test_isotherm_derivative_matches_finite_difference():
    iso = dr.iso_params()
    for p in (1e-6, 1e-4, 1e-2, 3e-2):
        h = p * 1e-6
        fd = (dr.q_water(p + h, 301.65, iso) - dr.q_water(p - h, 301.65, iso)) / (2 * h)
        assert abs(dr.dq_water_dp(p, 301.65, iso) - fd) / fd < 1e-5


def test_toth_loading_at_bt2_matches_table_s4_within_10pct():
    iso = dr.iso_params()
    s4 = pd.read_csv(CONFIGS / "tables" / "L15_S4.csv").set_index("experiment")
    for bt in ("BT1", "BT2", "BT3", "BT4", "BT5", "BT8"):
        r = s4.loc[bt]
        T = r.T_C + 273.15
        q = float(dr.q_water(r.RH_pct / 100 * dr.psat_bar(T, iso), T, iso))
        assert abs(q - r.q_s_mol_kg) / r.q_s_mol_kg < 0.10


def test_bt2_with_frozen_k_reproduces_calibration_target_and_balances_water():
    import json
    from h2su.io import RESULTS
    df = load_registry()
    iso = dr.iso_params(df)
    k = json.loads((RESULTS / "vt6_cal_frozen.json").read_text())["value"]
    s4 = pd.read_csv(CONFIGS / "tables" / "L15_S4.csv").set_index("experiment")
    p = dr.lab_case(df, iso, s4.loc["BT2"].to_dict(), k, t_end_s=4 * 3600.0, dt_out_s=30.0)
    tb, r = dr.t_b(p, iso, N=100)
    assert abs(tb - 3 * 3600) / (3 * 3600) < 0.05          # N = 100 grid error of the first-order scheme (CAL is on N = 400)
    assert r["sol"]["balance_err"].max() < 1e-9

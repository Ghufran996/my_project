"""Brief 6 item 6: GCI helper of dryer_gridband.py on a synthetic sequence with a known order, and the non-monotone branch."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dryer_gridband import gci  # noqa: E402


def test_gci_recovers_order_and_limit():
    f0, C, p = 4.0, 3.0, 1.3
    h = np.array([800.0, 1600.0, 3200.0]) ** -1.0
    f = f0 + C * h ** p
    r = gci(*f)
    assert r["monotone"]
    assert abs(r["p_observed"] - p) < 1e-9
    assert abs(r["f_richardson"] - f0) < 1e-9


def test_gci_non_monotone_reports_range():
    r = gci(6.1, 5.9, 6.0)
    assert not r["monotone"]
    assert r["range_min"] == 5.9 and r["range_max"] == 6.1

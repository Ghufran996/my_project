import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from h2su.baselines import bs2_c3 as b


def test_constant_power_one_startup_no_standby():
    cap = 10.0
    r = b.solve_dp(np.full(24, 6.0), 1.0, cap, b.Curve(0.75, 0.55))
    assert r["n_startups"] == 1 and r["standby_MWh"] == 0.0
    s1, s2 = b.segments(b.Curve(0.75, 0.55), cap)
    assert r["H2_kg"] == pytest.approx(24 * (s2[0] * 6.0 + s2[1]), rel=1e-12)     # 6 MW is in segment 2
    # chord value at the segment end points equals the exact curve
    assert s2[0] * cap + s2[1] == pytest.approx(0.55 * cap / b.LHV, rel=1e-12)


def test_fixed_efficiency_is_linear():
    r = b.solve_dp(np.full(10, 5.0), 1.0, 10.0, b.Curve(0.75, 0.75, fixed=True))
    assert r["H2_kg"] == pytest.approx(10 * 0.75 * 5.0 / b.LHV, rel=1e-12)


def test_power_below_pmin_gives_nothing():
    r = b.solve_dp(np.full(10, 0.5), 1.0, 10.0, b.Curve(0.75, 0.6))   # P_min = 1 MW
    assert r["H2_kg"] == 0.0


@pytest.mark.parametrize("sb_off", [False, True])
@pytest.mark.parametrize("curve", [b.Curve(0.75, 0.55), b.Curve(0.75, 0.75, fixed=True)])
def test_dp_equals_milp(curve, sb_off):
    rng = np.random.default_rng(7)
    x = np.clip(np.cumsum(rng.normal(0, 2.0, 60)) + 6, 0, 12)
    cap = 8.0
    d = b.solve_dp(x, 5 / 60, cap, curve, sb_off)
    m = b.solve_milp(x, 5 / 60, cap, curve, sb_off, time_limit=60)
    assert m["success"]
    assert d["objective_MWh"] == pytest.approx(m["objective_MWh"], rel=1e-6)

"""Unit tests of the first component models (Brief 2 item 9): thermal balance, polarisation, Faradaic efficiency limits."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from h2su import electrolyzer as el  # noqa: E402
from h2su import gas, purification, thermal  # noqa: E402


def _m(R, C=1.0e6, Q=4000.0):
    return thermal.Vt1(C=C, R=R, heater_W=Q)


def test_energy_balance_t0_and_analytic_heating():
    # hand calculation: T(t) = T_amb + R Q (1 - exp(-t/(R C))); initial slope Q/C (no loss at T = T_amb)
    R, C, Q = 0.02, 1.0e6, 4000.0                      # R Q = 80 K
    t_an = -R * C * np.log(1 - 50.0 / (R * Q)) / 60.0
    t_sim = thermal.vt1_time_to_75C(thermal.vt1_strategies()["heating_only"], _m(R, C, Q))
    assert t_sim == pytest.approx(t_an, rel=1e-4)
    slope = Q / C                                       # K/s at t = 0
    t_small = thermal.vt1_time_to_75C(thermal.vt1_strategies()["heating_only"], _m(R, C, Q), T_target_C=25.0 + slope * 10.0)
    assert t_small * 60.0 == pytest.approx(10.0, rel=2e-3)   # 10 s at the initial slope (curvature negligible)


def test_steady_state_T_equals_Tamb_plus_RQ():
    R, C, Q = 0.0075, 1.0e6, 4000.0                    # R Q = 30 K -> steady state 55 degC
    s = thermal.vt1_strategies()["heating_only"]
    assert np.isfinite(thermal.vt1_time_to_75C(s, _m(R, C, Q), T_target_C=54.9, t_max_s=1e5))
    assert np.isnan(thermal.vt1_time_to_75C(s, _m(R, C, Q), T_target_C=55.1, t_max_s=20 * R * C))


def test_zero_current_limit():
    assert el.j_at_voltage_d29(1.0, 298.15, 200.0, 1.0) == 0.0                         # below the reversible voltage
    s = thermal.vt1_strategies()
    m = _m(0.02)
    gal0 = dict(kind="galv", steps=[], heater=True)                                    # j = 0 everywhere
    assert thermal.vt1_time_to_75C(gal0, m) == pytest.approx(thermal.vt1_time_to_75C(s["heating_only"], m), rel=1e-12)


def test_polarisation_calibration_roundtrip_and_trends():
    sc = el.calibrate_sigma_scale()
    assert float(el.cell_voltage_d29(1.75, 348.15, 200.0, sc)) == pytest.approx(1.80, abs=1e-9)
    j = np.linspace(0.1, 2.0, 12)
    assert np.all(np.diff(el.cell_voltage_d29(j, 333.15, 200.0, sc)) > 0)               # monotone in j
    assert float(el.cell_voltage_d29(1.0, 348.15, 200.0, sc)) < float(el.cell_voltage_d29(1.0, 298.15, 200.0, sc))  # decreases with T
    assert el.sigma_springer(348.15) > el.sigma_springer(298.15) > 0


def test_faradaic_efficiency_limits():
    low = el.faradaic_efficiency_persson(np.array([1.23, 1.30]), 333.15)
    assert low[0] == 0.0 and low[1] == 0.0                                              # -> 0 as I -> 0 (below onset)
    U = np.linspace(1.4, 3.0, 40)
    eta = el.faradaic_efficiency_persson(U, 353.15)
    assert np.all(np.diff(eta) >= -1e-12)
    assert eta[-1] <= 0.97 and eta[-1] > 0.95                                           # asymptote factor 0.97 from below
    assert el.faradaic_efficiency_persson(50.0, 353.15) == pytest.approx(0.97, abs=3e-3)


def test_vt2_and_vt5a_use_roles_correctly():
    val, cal = gas.vt2_awe()
    assert cal["power_set_pct"] == 75 and set(val.power_set_pct) == {20, 30, 35, 50}
    df, par = purification.vt5a()
    assert set(df.load_pct) == {20, 30, 40, 50, 60, 70, 80, 90}
    with pytest.raises(PermissionError):
        thermal.registry_scalar("exp_heatup_galv_3step", ("CAL",))                      # VAL row must not be readable as CAL

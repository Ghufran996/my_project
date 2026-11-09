"""VER-3: analytic checks of the numerical building blocks (tolerance 0.1 %, closure 1e-9)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from h2su import accounting, gas, purification  # noqa: E402
from h2su.constants import M_H2, M_O2, R_GAS  # noqa: E402


def washout_analytic(N, tau, t):
    """Step response of N equal CSTRs in series (Erlang): c/c_in = 1 - exp(-x) sum_{k<N} x^k/k!, x = N t / tau."""
    x = N * t / tau
    return 1.0 - np.exp(-x) * sum(x**k / math.factorial(k) for k in range(N))


@pytest.mark.parametrize("N", [1, 2, 5, 10])
def test_tanks_in_series_washout(N):
    tau, c_in = 600.0, 1.0
    t, c = gas.tanks_in_series_step_response(N, tau, 3 * tau, 1.0, c_in)
    ref = washout_analytic(N, tau, t)
    mask = ref > 1e-3                      # relative error is meaningless where c is ~0
    rel = np.abs(c[mask] - ref[mask]) / ref[mask]
    assert rel.max() <= 1e-3
    assert np.abs(c - ref).max() <= 1e-6   # absolute check over the whole record


def test_washout_to_spec_time():
    """Time to reach a residual of 300/1e6 (N2 limit scale) in N = 5 tanks: numeric vs analytic root, <= 0.1 %."""
    from scipy.optimize import brentq
    N, tau = 5, 900.0
    target = 1.0 - 300e-6                  # outlet clean-gas fraction
    t, c = gas.tanks_in_series_step_response(N, tau, 6 * tau, 0.5, 1.0)
    t_num = t[np.argmax(c >= target)]
    t_ana = brentq(lambda s: washout_analytic(N, tau, s) - target, 1.0, 6 * tau)
    assert abs(t_num - t_ana) / t_ana <= 1e-3


def test_ideal_gas_molar_volume():
    """Independent reference: 1 mol at 273.15 K and 101325 Pa occupies 22.41396954 L (CODATA)."""
    V = R_GAS * 273.15 / 101325.0
    assert abs(V - 22.41396954e-3) / 22.41396954e-3 <= 1e-6
    assert abs(gas.inventory_mol(101325.0, 22.41396954e-3, 273.15) - 1.0) <= 1e-6


def test_pressurisation_constant_inflow():
    p0, V, T, ndot = 1.0e5, 2.0, 300.0, 0.05
    t = np.linspace(0.0, 3600.0, 3601)
    p = gas.pressurise_const_inflow(p0, V, T, ndot, t)
    p_ref = p0 + ndot * R_GAS * T / V * t
    assert np.max(np.abs(p - p_ref) / p_ref) <= 1e-3
    # inventory consistency
    assert abs(gas.inventory_mol(p[-1], V, T) - (gas.inventory_mol(p0, V, T) + ndot * t[-1])) / (ndot * t[-1]) <= 1e-3


def test_deoxo_two_to_one():
    assert purification.deoxo_h2_consumed_mol(1.0) == 2.0
    # element balance 2 H2 + O2 -> 2 H2O: mass conserved
    n_o2 = 0.37
    n_h2 = purification.deoxo_h2_consumed_mol(n_o2)
    m_in = n_h2 * M_H2 + n_o2 * M_O2
    m_h2o = 2 * n_o2 * (2 * 1.00794e-3 + 15.9994e-3)
    assert abs(m_in - m_h2o) / m_in <= 1e-3
    assert abs(purification.deoxo_h2_consumed_kg_per_kg_o2() - 2 * M_H2 / M_O2) <= 1e-12


def test_iso_grade_d_check():
    assert purification.iso14687_grade_d_ok({"O2": 5.0, "H2O": 5.0, "N2": 289.0})
    assert not purification.iso14687_grade_d_ok({"O2": 5.1})
    assert not purification.iso14687_grade_d_ok({"N2": 301.0})


def test_closure_synthetic_day():
    """Synthetic 1-day event sequence (1-min): stack H2 is split into fates and inventory change; closure <= 1e-9."""
    rng = np.random.default_rng(2026)
    n = 1440
    led = accounting.Ledger(n)
    led.stack[:] = np.clip(rng.normal(0.01, 0.003, n), 0, None) * (np.arange(n) > 360) * (np.arange(n) < 1080)
    names = list(accounting.FATES)
    for k in range(n):
        frac = rng.dirichlet(np.ones(len(names) + 1))   # last component = inventory change
        for j, nm in enumerate(names):
            led.fate[nm][k] = frac[j] * led.stack[k]
        led.d_inventory[k] = frac[-1] * led.stack[k]
    assert led.closure_residual() <= 1e-9
    # a bookkeeping error must be detected
    led.fate["purge"][100] += 1e-3
    assert led.closure_residual() > 1e-9

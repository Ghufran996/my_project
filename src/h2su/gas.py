"""Gas-handling primitives: ideal-gas inventory, pressurisation, tanks-in-series washout (numerical integrator).

These are the numerical building blocks of the purge / inerting / pressurisation models; they are verified against
analytic solutions in tests/test_analytic.py (VER-3).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .constants import R_GAS


def inventory_mol(p_Pa: float, V_m3: float, T_K: float) -> float:
    """Ideal-gas amount of substance n = p V / (R T)."""
    return p_Pa * V_m3 / (R_GAS * T_K)


def pressurise_const_inflow(p0_Pa: float, V_m3: float, T_K: float, n_dot_mol_s: float, t_s: np.ndarray) -> np.ndarray:
    """Pressure of a closed vessel filled at constant molar inflow, integrated numerically (RK4 on n, p = n R T / V)."""
    n = inventory_mol(p0_Pa, V_m3, T_K)
    out = np.empty_like(t_s, dtype=float)
    out[0] = p0_Pa
    for k in range(1, t_s.size):
        dt = t_s[k] - t_s[k - 1]
        # dn/dt = const: RK4 reduces to the exact step; written as RK4 so that a state-dependent inflow can replace n_dot later
        k1 = n_dot_mol_s
        k2 = n_dot_mol_s
        k3 = n_dot_mol_s
        k4 = n_dot_mol_s
        n += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k] = n * R_GAS * T_K / V_m3
    return out


@njit(cache=True)
def _tanks_rhs(c: np.ndarray, c_in: float, rate: float, out: np.ndarray) -> None:
    """dc_i/dt = rate (c_{i-1} - c_i), rate = N Q / V_total (= N / tau)."""
    out[0] = rate * (c_in - c[0])
    for i in range(1, c.size):
        out[i] = rate * (c[i - 1] - c[i])


@njit(cache=True)
def tanks_in_series_step_response(N: int, tau_s: float, t_end_s: float, dt_s: float, c_in: float) -> tuple:
    """Integrate N equal perfectly mixed tanks in series (total residence time tau) with a step in inlet concentration.

    Classical RK4. Returns (t, c_out) with c(0) = 0 in all tanks.
    """
    n_steps = int(round(t_end_s / dt_s))
    t = np.empty(n_steps + 1)
    cout = np.empty(n_steps + 1)
    c = np.zeros(N)
    k1 = np.empty(N)
    k2 = np.empty(N)
    k3 = np.empty(N)
    k4 = np.empty(N)
    rate = N / tau_s
    t[0] = 0.0
    cout[0] = 0.0
    for s in range(n_steps):
        _tanks_rhs(c, c_in, rate, k1)
        _tanks_rhs(c + 0.5 * dt_s * k1, c_in, rate, k2)
        _tanks_rhs(c + 0.5 * dt_s * k2, c_in, rate, k3)
        _tanks_rhs(c + dt_s * k3, c_in, rate, k4)
        c = c + dt_s / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t[s + 1] = (s + 1) * dt_s
        cout[s + 1] = c[N - 1]
    return t, cout


# ----------------------------------------------------------------------------------------------------------------------
# VT2 (Brief 2 item 9b, D25): pressurisation of a closed gas volume at constant H2 production.
#   dp/dt = n_dot R T / V_eff = n_dot / K_p ,  K_p = V_eff/(R T) [mol/bar] (the calibrated quantity; V_eff = K_p R T needs T_gas, SN25)
#   n_dot(PS) proportional to the power set PS (Faraday, approximately constant specific energy).
# CAL = power set 75 %; VAL = 20/30/35/50 %; tested scalar t_press,i = dp_i / gamma_model,i with dp_i = gamma_i t_i (both printed).
# ----------------------------------------------------------------------------------------------------------------------
def parse_ps_row(text: str) -> tuple:
    """'42 min ; 0.236 bar/min' -> (42.0, 0.236)."""
    import re
    a = re.findall(r"[-+]?\d+\.?\d*", str(text))
    return float(a[0]), float(a[1])


def vt2_awe(cal_ps: int = 75, val_ps: tuple = (20, 30, 35, 50)):
    """Returns (table of VAL results, calibration dict). The CAL row is read first; VAL rows never enter the calibration."""
    import pandas as pd
    from .io import load_registry
    from .thermal import registry_role
    reg = load_registry().set_index("name")
    cal_name = f"awe_pressurisation_PS{cal_ps}pct"
    assert registry_role(cal_name) == "CAL"
    t_cal, g_cal = parse_ps_row(reg.loc[cal_name, "value_or_range"])
    k_rel = cal_ps / g_cal          # gas capacity in units of (n_dot per % power set) per bar/min; the unit constant cancels
    rows = []
    for ps in val_ps:
        name = f"awe_pressurisation_PS{ps}pct"
        assert registry_role(name) == "VAL", name
        t_i, g_i = parse_ps_row(reg.loc[name, "value_or_range"])
        dp_i = g_i * t_i
        g_model = ps / k_rel
        t_pred = dp_i / g_model
        rows.append(dict(power_set_pct=ps, role="VAL", t_measured_min=t_i, gamma_measured_bar_min=g_i, dp_bar=dp_i,
                         gamma_model_bar_min=g_model, t_predicted_min=t_pred, RE=(t_pred - t_i) / t_i,
                         abs_RE_le_20pct=bool(abs((t_pred - t_i) / t_i) <= 0.20)))
    cal = dict(power_set_pct=cal_ps, t_min=t_cal, gamma_bar_min=g_cal, dp_bar=t_cal * g_cal)
    return pd.DataFrame(rows), cal


def awe_gas_capacity_mol_per_bar(cal: dict) -> float:
    """K_p [mol/bar] of the L28 pilot from the CAL point: n_dot(75 %) / gamma; n_dot = 0.75 x rated H2 flow (2 Nm3/h, registry
    awe_rated_H2_flow) / molar volume at normal conditions (R T_N / p_N, constants.py)."""
    from .constants import P_STD, R_GAS, T_STD
    from .io import param
    n_dot_rated = param("awe_rated_H2_flow") / 60.0 * P_STD / (R_GAS * T_STD) * 1.0      # mol/min at 100 % (Nm3/h -> m3/min -> mol)
    return cal["power_set_pct"] / 100.0 * n_dot_rated / cal["gamma_bar_min"]


def pem_gas_capacity_range_mol_per_bar() -> tuple:
    """L13 pressurisation: 1.7 kg H2-equivalent in 45 min to the inferred 30 bar; K_p = n/(p_op - p0), p0 in [0, 1] bar (SN24)."""
    from .constants import M_H2
    from .io import param
    n = param("pressurisation_H2_equiv_V1") * 1.0 / M_H2
    p = param("plant_operating_pressure_inferred")
    return n / p, n / (p - 1.0)

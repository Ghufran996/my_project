"""PEM electrolyzer heat-up model after L26 (Rauls et al. 2023, Appl. Energy 330:120350), Eq.(3)-(15), (31)-(37).

Implements the published structure (lumped thermal capacity, Joule heat I (U_cell - U_th), vapour-cooling term,
convective loss via R_th, potentiostatic or heating-only start-up). Items that the paper does not print are passed in
through `Rauls` fields and are listed in configs/source_needed.txt (SN01-SN06). Nothing here is tuned to Table 5.

Units: j [A/cm2], areas [cm2] for cells and [m2] for thermal surfaces, T [K], energies [J], masses [kg].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .constants import F_PAPER, M_H2, R_GAS
from .io import CONFIGS

# --- water properties: Antoine fit of Stull (1947) via NIST WebBook (SN06): log10(P/bar) = A - B/(T + C)
_A, _B, _C = 4.6543, 1435.264, -64.848   # valid 255.9-373 K


def p_sat_bar(T: float) -> float:
    return 10.0 ** (_A - _B / (T + _C))


def dh_vap(T: float) -> float:
    """J/mol, Clausius-Clapeyron on the Antoine fit (ideal vapour, negligible liquid volume)."""
    return R_GAS * T**2 * np.log(10.0) * _B / (T + _C) ** 2


# --- thermodynamics from printed constants (Sec 2.1): dH = 285.84, dG = 237.15 kJ/mol (constant dH, dS)
_DH, _DG = 285.84e3, 237.15e3
_DS = (_DH - _DG) / 298.15


def u_rev(T: float) -> float:
    return (_DG - _DS * (T - 298.15)) / (2.0 * F_PAPER)


def u_nernst(T: float, p_cat_bar: float, p_an_bar: float) -> float:
    """Eq.(4) with partial pressures from Eq.(36)-(37)."""
    ps = p_sat_bar(T)
    return u_rev(T) + R_GAS * T / (2.0 * F_PAPER) * np.log(np.sqrt(p_an_bar - ps) * (p_cat_bar - ps))


def sigma_membrane(T: float, a_w: float = 1.0) -> float:
    """Eq.(10) as printed: (0.6877 + a) exp{-10440 a^(-1/4) / (R T)}  [S/cm, unit assumed]. See SN01."""
    return (0.6877 + a_w) * np.exp(-10440.0 * a_w ** (-0.25) / (R_GAS * T))


def i0_of_T(T: float) -> float:
    """Eq.(16), reference 7.1e-7 A/cm2 at 25 degC, E_A = 40 kJ/mol."""
    return 7.1e-7 * np.exp(-40e3 / R_GAS * (1.0 / T - 1.0 / 298.15))


def alpha_of_T(T: float) -> float:
    """Eq.(17), reference 0.386 at 25 degC, E_A = 3.4 kJ/mol."""
    return 0.386 * np.exp(-3.4e3 / R_GAS * (1.0 / T - 1.0 / 298.15))


@dataclass
class Rauls:
    """Model switches. Defaults reproduce the equations as printed (attempt A)."""
    sigma_scale: float = 1.0          # multiplies Eq.(10); 1.0 = as printed
    R_omega_ohm_cm2: float = 0.007    # SN02: unit assumed Ohm cm2
    u_th: float = 1.48                # V, Sec 2.1 (T,p dependence from ref [36] not printed, SN03)
    p_cat_bar: float = 6.0
    p_an_bar: float = 5.0
    vapour_cooling: bool = True
    T_amb: float = 298.15
    cp: dict = field(default_factory=lambda: dict(steel=500.0, cu=385.0, ti=500.0, h2o=4185.0, poly=1250.0))
    h_steel: float = 21.0
    h_poly: float = 2.1


def load_tables() -> pd.DataFrame:
    t = pd.read_csv(CONFIGS / "tables" / "L26_rauls2023_tables.csv", comment="#")
    return t.set_index("row_label_kW")


def lumped_from_tables(row: pd.Series, n_stacks: int, m: Rauls) -> tuple[float, float]:
    """C_th [J/K] and thermal conductance G [W/K] from Table 3 x Table 4 (stack quantities x n_stacks)."""
    cp = m.cp
    C_stack = (row.t4_m_stack_steel_kg * cp["steel"] + row.t4_m_stack_copper_kg * cp["cu"]
               + row.t4_m_stack_titanium_kg * cp["ti"] + row.t4_m_stack_H2O_kg * cp["h2o"]) * n_stacks
    C_per = row.t4_m_periph_steel_kg * cp["steel"] + row.t4_m_periph_H2O_kg * cp["h2o"]
    G = m.h_steel * (row.t4_S_stack_steel_m2 * n_stacks + row.t4_S_periph_steel_m2) + m.h_poly * row.t4_S_stack_PEEK_m2 * n_stacks
    return C_stack + C_per, G


def cell_voltage(j: float, T: float, d_mem_um: float, m: Rauls) -> float:
    """Eq.(5): Nernst + Tafel activation (Eq.7, alpha in the denominator) + ohmic (Eq.8-9)."""
    eta_act = R_GAS * T / (2.0 * F_PAPER * alpha_of_T(T)) * np.log(j / i0_of_T(T))
    r_mem = d_mem_um * 1e-4 / (m.sigma_scale * sigma_membrane(T))      # cm / (S/cm) = Ohm cm2
    return u_nernst(T, m.p_cat_bar, m.p_an_bar) + eta_act + j * (r_mem + m.R_omega_ohm_cm2)


def j_at_voltage(U: float, T: float, d_mem_um: float, m: Rauls) -> float:
    f = lambda j: cell_voltage(j, T, d_mem_um, m) - U
    lo = 1e-9
    if f(lo) > 0:
        return 0.0
    return brentq(f, lo, 60.0, xtol=1e-12)


def heatup(row: pd.Series, n_stacks: int, m: Rauls, mode: str = "potentiostatic", U_cell: float = 1.80,
           Q_heat_W: float = 0.0, C_J_per_K: float | None = None, G_W_per_K: float | None = None,
           T_target_C: float = 75.0, t_max_s: float = 6 * 3600.0) -> dict:
    """Heat-up from 25 to 75 degC. mode 'potentiostatic' (U_cell fixed, + optional heater) or 'heating' (heater only)."""
    C, G = lumped_from_tables(row, n_stacks, m)
    if C_J_per_K is not None:
        C = C_J_per_K
    if G_W_per_K is not None:
        G = G_W_per_K
    n_cells = int(row.t2_n_cells) * n_stacks
    A = float(row.t2_cell_area_cm2)
    d = float(row.t2_membrane_um)
    T_t = T_target_C + 273.15

    def rhs(t, y):
        T = y[0]
        if mode == "potentiostatic":
            j = j_at_voltage(U_cell, T, d, m)
            I = j * A
            nH2 = I * n_cells / (2.0 * F_PAPER)
            nO2 = nH2 / 2.0
            Qov = I * n_cells * (U_cell - m.u_th)
            ps = p_sat_bar(T)
            Qvap = (nH2 * ps / (m.p_cat_bar - ps) + nO2 * ps / (m.p_an_bar - ps)) * dh_vap(T) if m.vapour_cooling else 0.0
        else:
            nH2 = Qov = Qvap = 0.0
        Qloss = G * (T - m.T_amb)
        return [(Qov + Q_heat_W - Qvap - Qloss) / C, nH2 * M_H2]

    ev = lambda t, y: y[0] - T_t
    ev.terminal = True
    ev.direction = 1
    sol = solve_ivp(rhs, [0, t_max_s], [298.15, 0.0], events=ev, rtol=1e-8, atol=1e-10, max_step=10.0, method="LSODA")
    reached = len(sol.t_events[0]) > 0
    return dict(t_min=(sol.t_events[0][0] / 60.0 if reached else np.nan), reached=reached,
                m_H2_kg=float(sol.y[1, -1]), T_end_C=float(sol.y[0, -1] - 273.15), C_kJ_per_K=C / 1e3, R_K_per_kW=1e3 / G)


# ----------------------------------------------------------------------------------------------------------------------
# VT1 (Brief 2 item 9a, D29): lumped heat-up of the L26 100 kWel test station,
#     C dT/dt = Q_el + Q_heat - (T - T_amb)/R ,   Q_el = N_cells I (U_cell - U_th)
# C from Table 3 x Table 4 (two 27-cell stacks, SN04 text configuration), R = the single parameter calibrated on the CAL
# case (two-step galvanostatic, 77 min); VAL cases are never read by the calibration (role check below).
# ----------------------------------------------------------------------------------------------------------------------
def registry_role(name: str) -> str:
    from .io import load_registry
    r = load_registry()
    return str(r.loc[r["name"] == name, "role"].iloc[0])


def registry_scalar(name: str, allowed_roles: tuple) -> float:
    """Numeric registry value; raises if the row's role is not allowed (guards the CAL/VAL separation)."""
    from .io import param
    role = registry_role(name)
    if role not in allowed_roles:
        raise PermissionError(f"{name} has role {role}; allowed here: {allowed_roles}")
    return param(name)


@dataclass
class Vt1:
    C: float                  # J/K
    R: float                  # K/W
    n_cells: int = 54
    area_cm2: float = 300.0
    d_um: float = 200.0
    sigma_scale: float = 1.0
    heater_W: float = 4000.0
    T_amb: float = 298.15


def vt1_strategies(galv2_step2: float = 1.0) -> dict:
    """Heat-up strategies of L26 Table 1 / Sec 3.1.1 (values registered in parameters.csv, rows l26_*)."""
    return {
        "heating_only": dict(kind="heating", heater=True),
        "potentiostatic_1p8V": dict(kind="pot", U=1.80, heater=False),
        "potentiostatic_1p8V_plus_4kW": dict(kind="pot", U=1.80, heater=True),
        "galvanostatic_2step": dict(kind="galv", steps=[(40.0 + 273.15, 0.5), (60.0 + 273.15, galv2_step2)], heater=True),
        "galvanostatic_3step": dict(kind="galv", steps=[(40.0 + 273.15, 0.5), (50.0 + 273.15, 1.0), (60.0 + 273.15, 1.5)], heater=True),
    }


def vt1_time_to_75C(strategy: dict, m: Vt1, t_max_s: float = 8 * 3600.0, T_target_C: float = 75.0) -> float:
    """Minutes to reach 75 degC from 25 degC (nan if not reached within t_max_s)."""
    from .electrolyzer import cell_voltage_d29, j_at_voltage_d29
    T_t = T_target_C + 273.15

    def rhs(t, y):
        T = y[0]
        Qh = m.heater_W if strategy["heater"] else 0.0
        Qel = 0.0
        if strategy["kind"] == "pot":
            j = j_at_voltage_d29(strategy["U"], T, m.d_um, m.sigma_scale)
            Qel = m.n_cells * j * m.area_cm2 * (strategy["U"] - 1.48)
        elif strategy["kind"] == "galv":
            j = 0.0
            for Ts, js in strategy["steps"]:
                if T >= Ts:
                    j = js
            if j > 0:
                Qel = m.n_cells * j * m.area_cm2 * (float(cell_voltage_d29(j, T, m.d_um, m.sigma_scale)) - 1.48)
        return [(Qel + Qh - (T - m.T_amb) / m.R) / m.C]

    ev = lambda t, y: y[0] - T_t
    ev.terminal = True
    ev.direction = 1
    sol = solve_ivp(rhs, [0, t_max_s], [298.15], events=ev, rtol=1e-8, atol=1e-10, max_step=10.0, method="LSODA")
    return float(sol.t_events[0][0] / 60.0) if len(sol.t_events[0]) else float("nan")


def vt1_calibrate_R(C: float, cal_minutes: float, m0: Vt1, strategy: dict) -> float:
    """The one thermal calibration: R [K/W] such that the CAL case reaches 75 degC at cal_minutes (brentq on log R)."""
    def f(logR):
        m = Vt1(C, float(np.exp(logR)), m0.n_cells, m0.area_cm2, m0.d_um, m0.sigma_scale, m0.heater_W, m0.T_amb)
        t = vt1_time_to_75C(strategy, m)
        return (t if np.isfinite(t) else 1e9) - cal_minutes
    return float(np.exp(brentq(f, np.log(1e-3), np.log(1e3), xtol=1e-10)))

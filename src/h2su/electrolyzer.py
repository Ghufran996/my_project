"""Stack-level relations taken from the literature (no plant model yet; skeleton)."""
from __future__ import annotations

import numpy as np

from .constants import F_PAPER, R_GAS


def faradaic_efficiency_persson(U_stack, T_K: float, A: float = 6.0, x2: float = 70.0, dU_act: float = 0.12,
                                U_rev: float = 1.23, factor: float = 0.97, z: int = 2):
    """L13 Persson 2020 Eq.(16) with Table 1 parameters.

    eta_F = factor * (exp(x1 (U - U_rev - dU_act)) - 1) / (exp(x1 (U - U_rev - dU_act)) + x2 U),
    x1 = (z F / (0.5 R_gas T)) / A.  Negative values (U below onset) are clipped to 0 (physical lower bound).
    T is not stated in the source (SN07).
    """
    U = np.asarray(U_stack, dtype=float)
    x1 = (z * F_PAPER / (0.5 * R_GAS * T_K)) / A
    a = x1 * (U - U_rev - dU_act)
    with np.errstate(over="ignore"):
        e = np.exp(np.minimum(a, 0.0))                  # a <= 0 branch
        em = np.exp(-np.maximum(a, 0.0))                # a > 0 branch, no overflow for large U
    eta = np.where(a > 0, factor * (1.0 - em) / (1.0 + x2 * U * em), factor * (e - 1.0) / (e + x2 * U))
    return np.clip(eta, 0.0, None)


def faradaic_current_h2_mass_rate(I_A: float, n_cells: int, eta_F: float) -> float:
    """Faraday's law with Faradaic efficiency, kg/s of H2 (cell current I_A through n_cells in series)."""
    from .constants import M_H2
    return eta_F * n_cells * I_A / (2.0 * F_PAPER) * M_H2


# ----------------------------------------------------------------------------------------------------------------------
# D29 polarisation model (Brief 2 item 9a): Nernst + Tafel (L26 Eq.4, 7, kinetics Eq.16-17) + ohmic term from the L44
# membrane conductivity (Eq.[24]-[25]) multiplied by ONE scale factor calibrated once to the L26 stated operating point
# (1.80 V at 1.75 A/cm2, 75 degC, 200 um, 6/5 bar).  R_Omega (SN02) is not used separately: the scaled membrane term carries
# the whole ohmic loss.  p_sat from the NIST/Stull Antoine fit (SN06).
# ----------------------------------------------------------------------------------------------------------------------
def sigma_springer(T_K, lam: float | None = None):
    """Nafion 117 conductivity [S/cm], L44 Eq.[24]-[25]: (0.005139 lam - 0.00326) exp{1268 (1/303 - 1/T)}."""
    from .io import param
    lam = param("l44_lambda_liquid_80C") if lam is None else lam
    s30 = param("l44_sigma30_slope") * lam - param("l44_sigma30_offset")
    return s30 * np.exp(param("l44_sigma_EaR") * (1.0 / param("l44_sigma_Tref") - 1.0 / np.asarray(T_K, float)))


def cell_voltage_d29(j, T_K: float, d_um: float, sigma_scale: float, p_cat_bar: float = 6.0, p_an_bar: float = 5.0,
                     lam: float | None = None):
    """Cell voltage [V] at current density j [A/cm2] (j > 0)."""
    from .constants import R_GAS as _R
    from .thermal import alpha_of_T, i0_of_T, u_nernst
    j = np.asarray(j, float)
    eta_act = _R * T_K / (2.0 * F_PAPER * alpha_of_T(T_K)) * np.log(j / i0_of_T(T_K))
    r_mem = d_um * 1e-4 / (sigma_scale * sigma_springer(T_K, lam))      # Ohm cm2
    return u_nernst(T_K, p_cat_bar, p_an_bar) + eta_act + j * r_mem


def j_at_voltage_d29(U: float, T_K: float, d_um: float, sigma_scale: float, **kw) -> float:
    from scipy.optimize import brentq
    f = lambda j: float(cell_voltage_d29(j, T_K, d_um, sigma_scale, **kw)) - U
    if f(1e-9) > 0:
        return 0.0
    return brentq(f, 1e-9, 60.0, xtol=1e-12)


def calibrate_sigma_scale(U_nom: float = 1.80, j_nom: float = 1.75, T_C: float = 75.0, d_um: float = 200.0,
                          lam: float | None = None) -> float:
    """The single polarisation calibration (CAL-polarisation, not a VT target): sigma_scale with U(j_nom, T) = U_nom.
    Nominal point = registry row nominal_station_operating_point (L26 Sec 3.2)."""
    from scipy.optimize import brentq
    T = T_C + 273.15
    f = lambda s: float(cell_voltage_d29(j_nom, T, d_um, s, lam=lam)) - U_nom
    return brentq(f, 1e-3, 1e3, xtol=1e-14)

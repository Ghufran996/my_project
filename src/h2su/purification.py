"""Purification-train primitives (skeleton): deoxo stoichiometry, ISO 14687 Grade D check, dryer recovery loss.

Rates and start-up transients of the deoxo reactor and dryer are NOT implemented here (later briefs).
"""
from __future__ import annotations

from .constants import M_H2, M_O2
from .io import param

# 2 H2 + O2 -> 2 H2O
STOICH_H2_PER_O2 = 2.0


def deoxo_h2_consumed_mol(n_O2_mol: float) -> float:
    """H2 consumed by catalytic recombination of n_O2 mol of O2 (2:1)."""
    return STOICH_H2_PER_O2 * n_O2_mol


def deoxo_h2_consumed_kg_per_kg_o2() -> float:
    return STOICH_H2_PER_O2 * M_H2 / M_O2


def iso14687_grade_d_ok(c_umol_mol: dict) -> bool:
    """Check a composition (umol/mol, impurities only) against the registry limits (D15). Missing species are taken as 0."""
    limits = {"H2O": param("iso_H2O_max"), "O2": param("iso_O2_max"), "N2": param("iso_N2_max"), "Ar": param("iso_Ar_max"),
              "He": param("iso_He_max"), "CH4": param("iso_CH4_max"), "CO2": param("iso_CO2_max"), "CO": param("iso_CO_max")}
    for sp, lim in limits.items():
        if c_umol_mol.get(sp, 0.0) > lim:
            return False
    return sum(c_umol_mol.values()) <= param("iso_total_nonH2_max")


def dryer_recovery_loss_kg(m_feed_kg: float, recovery_fraction: float) -> float:
    return m_feed_kg * (1.0 - recovery_fraction)


# ----------------------------------------------------------------------------------------------------------------------
# VT5a (Brief 2 item 9c, D29): fixed-cycle PSA purge share  s(L) = phi + b / L  (L = load fraction): a constant purge flow
# (phi) plus a loss inversely proportional to the throughput (b).  CAL = L1 Table 2 at 10 % and 100 %; VAL = intermediate loads.
# ----------------------------------------------------------------------------------------------------------------------
def vt5a():
    import pandas as pd
    from .thermal import registry_role, registry_scalar
    cal = {L: registry_scalar(f"psa_purge_loss_load{L}", ("CAL",)) for L in (10, 100)}
    b = (cal[10] - cal[100]) / (1.0 / 0.10 - 1.0 / 1.0)
    phi = cal[100] - b
    rows = []
    for L in (20, 30, 40, 50, 60, 70, 80, 90):
        assert registry_role(f"psa_purge_loss_load{L}") == "VAL"
        meas = registry_scalar(f"psa_purge_loss_load{L}", ("VAL",))
        pred = phi + b / (L / 100.0)
        rows.append(dict(load_pct=L, role="VAL", purge_measured_pct=meas, purge_model_pct=pred, RE=(pred - meas) / meas,
                         abs_RE_le_20pct=bool(abs((pred - meas) / meas) <= 0.20)))
    return pd.DataFrame(rows), dict(phi_pct=phi, b_pct=b, cal=cal)

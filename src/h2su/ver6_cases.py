"""VER-6 case definitions shared by the Python method-of-lines code and the user-run COMSOL package.

Single source of truth: `build()` returns (a) the case table, (b) the long-format parameter rows written to
`COMSOL_MATLAB/params_comsol.csv`, (c) the list of blocked cases with the SN item that blocks them.
No value is created here without a source: registry values are looked up by name; verification-design values
(test conditions of analytic cases, no physical claim) carry source_id = VERIF-DESIGN.

Conventions (fixed for both codes):
  * interstitial velocity v = u_s / eps ; Pe_L = v L / D_ax ; inv_Pe = 1 / Pe_L (0 = no dispersion)
  * first-order rate per BED volume r_v = k1 c  [mol m-3 bed s-1]  ->  Da = k1 L / u_s  (u_s superficial)
  * linear isotherm q_s = K c (solid-volume basis)  ->  retardation 1 + K (1 - eps) / eps
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .io import load_registry

VERIF = "VERIF-DESIGN"
VERIF_LOC = "Brief 2 item 6a: analytic verification case (test condition chosen by design; the analytic reference does not depend on it)"


@dataclass
class Case:
    case_id: str
    model_function: str
    group: str                      # verif | deoxo | dryer
    description: str
    params: dict = field(default_factory=dict)   # name -> (value, unit, source_id, location, extraction, note)
    status: str = "READY"
    blocked_by: str = ""


def _v(value, unit, note=""):
    return (value, unit, VERIF, VERIF_LOC, "design", note)


def verification_cases() -> list[Case]:
    base = dict(L_m=_v(1.0, "m"), eps=_v(0.4, "-"), c_in_norm=_v(1.0, "-", "normalised tracer inlet value"),
                tol_check=(0.01, "-", "Brief 2 item 6a", "acceptance criterion: abs error <= 1 % (brief item 6)", "criterion", ""))
    cases = []
    p = dict(base, us_m_s=_v(0.2, "m/s", "superficial velocity"), k1_s=_v(0.4, "1/s", "first-order rate per bed volume"),
             inv_Pe=_v(0.0, "-", "0 = no axial dispersion"), K_ads=_v(0.0, "-"), t_end_s=_v(20.0, "s"), dt_out_s=_v(0.5, "s"))
    cases.append(Case("ver_plugflow", "model_verif_plugflow", "verif",
                      "steady first-order plug flow, Da = k1 L / u_s = 2; X = 1 - exp(-Da)", p))
    for tag, inv in (("pe10", 0.1), ("pe100", 0.01)):
        p = dict(base, us_m_s=_v(0.2, "m/s"), k1_s=_v(0.4, "1/s"), inv_Pe=_v(inv, "-", f"Pe_L = {1/inv:g}"),
                 K_ads=_v(0.0, "-"), t_end_s=_v(40.0, "s"), dt_out_s=_v(0.5, "s"))
        cases.append(Case(f"ver_disp_{tag}", "model_verif_dispersion", "verif",
                          f"steady first-order axial dispersion, Danckwerts boundaries, Pe_L = {1/inv:g}, Da = 2", p))
    p = dict(base, us_m_s=_v(0.2, "m/s"), k1_s=_v(0.0, "1/s", "no reaction"), inv_Pe=_v(0.001, "-", "Pe_L = 1000; small dispersion kept in both codes for numerical stability, first moment is unaffected"),
             K_ads=_v(5.0, "-", "linear isotherm slope, solid-volume basis"), t_end_s=_v(40.0, "s"), dt_out_s=_v(0.5, "s"))
    cases.append(Case("ver_breakthrough", "model_verif_breakthrough", "verif",
                      "linear isotherm breakthrough, first moment t = (L/v)(1 + K(1-eps)/eps) = 17 s", p))
    return cases


def _reg(df: pd.DataFrame, name: str):
    row = df.loc[df["name"] == name]
    if row.empty:
        raise KeyError(name)
    r = row.iloc[0]
    return float(r["value_or_range"]), r["unit"], r["source_id"], r["location"], r["extraction"]


def global_rows(df: pd.DataFrame) -> list[dict]:
    """Registry-derived global parameters (name in COMSOL = key). Only scalar numeric registry rows."""
    from scipy import constants as sc
    from .constants import M_H2
    rows = []
    for out, reg in (("spec_O2_umol_mol", "iso_O2_max"), ("spec_N2_umol_mol", "iso_N2_max"), ("spec_H2O_umol_mol", "iso_H2O_max")):
        v, u, s, loc, ex = _reg(df, reg)
        rows.append(dict(scope="global", name=out, value=v, unit=u, source_id=s, location=loc, extraction=ex, role="INPUT",
                         note=f"registry row {reg}"))
    rows.append(dict(scope="global", name="R_gas_J_mol_K", value=sc.gas_constant, unit="J/(mol K)",
                     source_id="SI 2019 exact value N_A*k_B (CODATA via scipy.constants.gas_constant)",
                     location="scipy.constants.gas_constant", extraction="constant", role="INPUT", note=""))
    rows.append(dict(scope="global", name="event_break_frac", value=0.99, unit="-",
                     source_id="VER-6 schema definition (code/configs/ver6_schema.csv, t_break_s)",
                     location="ver6_schema.csv t_break_s", extraction="definition", role="INPUT",
                     note="breakthrough first moment is reported only if the outlet reached this fraction of the inlet value"))
    rows.append(dict(scope="global", name="M_H2_kg_mol", value=M_H2, unit="kg/mol",
                     source_id="IUPAC standard atomic weight of H x 2 (code/src/h2su/constants.py)",
                     location="constants.py M_H2", extraction="constant", role="INPUT", note=""))
    return rows


def blocked_models(df: pd.DataFrame) -> list[Case]:
    """Deoxo (SN18) stays BLOCKED; the dryer cases exist once the VT6 calibration is frozen (D46)."""
    from .io import RESULTS
    names = set(df["name"])
    out = []
    if not any(n.startswith("deoxo_") for n in names):
        out.append(Case("deox_cases", "model_deoxo_startup", "deoxo",
                        "cold/warm restart x O2 feed at the ends of the sourced range", {}, "BLOCKED", "SN18"))
    if not (RESULTS / "vt6_cal_frozen.json").exists():
        out.append(Case("dry_cases", "model_dryer_startup_regen", "dryer",
                        "dryer cases need the frozen VT6 calibration (results/vt6_cal_frozen.json)", {}, "BLOCKED", "VT6 CAL"))
    return out


def build():
    from .io import RESULTS
    df = load_registry()
    cases = verification_cases() + blocked_models(df)
    rows = global_rows(df)
    if (RESULTS / "vt6_cal_frozen.json").exists():
        from . import dryer_cases
        cases += dryer_cases.build(df, Case)
        rows += dryer_cases.global_rows(df)
    for c in cases:
        for name, (value, unit, src, loc, ex, note) in c.params.items():
            rows.append(dict(scope=c.case_id, name=name, value=value, unit=unit, source_id=src, location=loc,
                             extraction=ex, role="VERIFICATION", note=note))
    return cases, rows

"""Verification runs VER-1 (L26 Table 5) and VER-1b (L13 Eq.16 / Table 1). Outputs are CSV; nothing is tuned to targets."""
from __future__ import annotations

import io as _io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .electrolyzer import faradaic_efficiency_persson
from .io import CONFIGS, RESULTS, ROOT
from .thermal import Rauls, cell_voltage, heatup, load_tables, sigma_membrane

# Published lumped values (Sec 3.3.2.2 text), used only in the 'published-lumped' variants
PUB_C_KJ_PER_K = {5: 51.6, 400: 1434.3, 750: 2848.6, 1000: 4407.5}
PUB_R_K_PER_KW = {5: 39.7, 400: 5.0, 1000: 2.3}
TOL = 0.10


def sigma_scale_from_stated_operating_point() -> float:
    """Scale of Eq.(10) such that the paper's own nominal point (1.80 V, 1.75 A/cm2, 75 degC, 200 um) is met.
    This uses a published operating point, NOT Table 5 (diagnostic attempt B)."""
    m = Rauls()
    f = lambda s: cell_voltage(1.75, 348.15, 200.0, replace(m, sigma_scale=s)) - 1.80
    return brentq(f, 0.1, 50.0)


def run_ver1() -> pd.DataFrame:
    tab = load_tables()
    s_B = sigma_scale_from_stated_operating_point()
    mapping = [(5, 5, 1), (100, 50, 2), (100, 50, 1), (400, 400, 1), (750, 750, 1), (1000, 1000, 1)]
    rows = []
    for attempt, m in [("A_as_printed", Rauls()), ("B_sigma_scaled_to_stated_operating_point", Rauls(sigma_scale=s_B))]:
        for Pk, lab, ns in mapping:
            r = tab.loc[lab]
            ref = float(r.t5_mH2_max_kg)
            for cmode in ("tables", "published_lumped"):
                kw = {}
                if cmode == "published_lumped":
                    if Pk not in PUB_C_KJ_PER_K:
                        continue
                    kw["C_J_per_K"] = PUB_C_KJ_PER_K[Pk] * 1e3
                    if Pk in PUB_R_K_PER_KW:
                        kw["G_W_per_K"] = 1e3 / PUB_R_K_PER_KW[Pk]
                res = heatup(r, ns, m, mode="potentiostatic", U_cell=1.80, **kw)
                re = (res["m_H2_kg"] / ref - 1.0) * 100.0
                rows.append(dict(attempt=attempt, sigma_scale=m.sigma_scale, P_nom_kW=Pk, table_row_kW=lab, n_stacks=ns, thermal_basis=cmode,
                                 C_kJ_per_K=res["C_kJ_per_K"], R_K_per_kW=res["R_K_per_kW"], reached_75C=res["reached"], t_heatup_min=res["t_min"],
                                 T_end_C=res["T_end_C"], mH2_model_kg=res["m_H2_kg"], mH2_published_kg=ref, RE_pct=re, abs_RE_le_10pct=bool(abs(re) <= 100 * TOL)))
    df = pd.DataFrame(rows)
    # thermal-only consistency checks against published heating-only durations (Sec 4.1.2, 4.2.1)
    chk = []
    for Pk, lab, ns, QkW, pub_t in [(100, 50, 1, 4.0, 162.0), (100, 50, 2, 4.0, 162.0), (400, 400, 1, 12.0, 157.0), (400, 400, 1, 24.0, 59.0), (400, 400, 1, 80.0, 16.0)]:
        r = tab.loc[lab]
        for cmode in ("tables", "published_lumped"):
            kw = {}
            if cmode == "published_lumped":
                if Pk not in PUB_C_KJ_PER_K:
                    continue
                kw["C_J_per_K"] = PUB_C_KJ_PER_K[Pk] * 1e3
                if Pk in PUB_R_K_PER_KW:
                    kw["G_W_per_K"] = 1e3 / PUB_R_K_PER_KW[Pk]
            res = heatup(r, ns, Rauls(), mode="heating", Q_heat_W=QkW * 1e3, **kw)
            chk.append(dict(attempt="heating_only_check", sigma_scale=np.nan, P_nom_kW=Pk, table_row_kW=lab, n_stacks=ns, thermal_basis=cmode,
                            C_kJ_per_K=res["C_kJ_per_K"], R_K_per_kW=res["R_K_per_kW"], reached_75C=res["reached"], t_heatup_min=res["t_min"],
                            T_end_C=res["T_end_C"], mH2_model_kg=np.nan, mH2_published_kg=np.nan, RE_pct=(res["t_min"] / pub_t - 1) * 100 if res["reached"] else np.nan,
                            abs_RE_le_10pct=bool(res["reached"] and abs(res["t_min"] / pub_t - 1) <= TOL), heater_kW=QkW, t_published_min=pub_t))
    return pd.concat([df, pd.DataFrame(chk)], ignore_index=True)


def table2_consistency() -> pd.DataFrame:
    """Printed polarisation equations vs the paper's own Table 2 nominal points (50 um membranes, 75 degC)."""
    tab = load_tables()
    s_B = sigma_scale_from_stated_operating_point()
    out = []
    for lab, r in tab.iterrows():
        for name, m in [("A_as_printed", Rauls()), ("B_sigma_scaled", Rauls(sigma_scale=s_B))]:
            U = cell_voltage(float(r.t2_j_nom_A_cm2), 348.15, float(r.t2_membrane_um), m)
            out.append(dict(row_kW=lab, attempt=name, j_A_cm2=float(r.t2_j_nom_A_cm2), U_model_V=U, U_table2_V=float(r.t2_Ucell_nom_V), diff_V=U - float(r.t2_Ucell_nom_V)))
    return pd.DataFrame(out)


def run_ver1b() -> pd.DataFrame:
    """Persson Eq.(16) with Table 1 parameters against the manuscript's tabulated 'modelled Faraday's efficiency'."""
    tbl = pd.read_csv(CONFIGS / "tables" / "L13_persson2020_etaF_table.csv", comment="#")
    V = tbl["U_stack_V"].to_numpy()
    tab = tbl["eta_F_modelled_table"].to_numpy()
    rows = []
    for T in (293.15, 313.15, 333.15, 353.15):
        eta = faradaic_efficiency_persson(V, T)
        for v, a, b in zip(V, eta, tab):
            rows.append(dict(T_K=T, U_stack_V=v, eta_F_eq16=float(a), eta_F_table_modelled=float(b), abs_diff=float(a - b),
                             rel_diff_pct=float((a / b - 1) * 100) if b > 0 else np.nan, within_2pct=bool(abs(a - b) <= 0.02 * max(b, 1e-12) if b > 0 else abs(a - b) <= 1e-9)))
    return pd.DataFrame(rows)

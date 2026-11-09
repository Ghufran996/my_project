"""Brief 3 items 2, 14, 15: gas-inventory normalisation (D32b), VT2b (D37), range widening (D37), thermal ranges (D32a, D37).

Normalised inventory i = m_inv / (dp * mdot_H2,rated)  [kg per (kg/h) per bar = h/bar; reported in min/bar].
All inputs come from code/configs/parameters.csv (source + location per row). No parameter is fitted.
Usage: python code/inventories.py item2 | item14 | item15
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su.constants import M_H2, P_STD, R_GAS, T_STD  # noqa: E402
from h2su.gas import awe_gas_capacity_mol_per_bar, parse_ps_row, vt2_awe  # noqa: E402
from h2su.io import RESULTS, load_registry, param  # noqa: E402

P0_BAR = 1.01325            # standard atmosphere, constants.P_STD / 1e5


def l13_inventory(P_press_kW: float) -> dict:
    """L13: 1.7 kg H2-equivalent produced in 45 min while pressurising to the inferred 30 bar; plant rated 250 kW.
    Rated H2 flow of the L13 plant = P_rated / SEC_press, SEC_press = P_press / (1.7 kg / 0.75 h)  (power during the 45 min = P_press)."""
    m = param("pressurisation_H2_equiv_V1")
    t_h = param("pressurisation_duration_V1") / 60.0
    p = param("plant_operating_pressure_inferred")
    mdot_press = m / t_h                                      # kg/h
    sec = P_press_kW / mdot_press                             # kWh/kg (stack+BoP, effective)
    mdot_rated = param("plant_rated_power") / sec             # kg/h
    i = m / ((p - P0_BAR) * mdot_rated)                       # h/bar
    return dict(P_press_kW=P_press_kW, SEC_kWh_kg=sec, mdot_rated_kg_h=mdot_rated, dp_bar=p - P0_BAR, i_h_per_bar=i)


def l28_inventory(k_p_mol_bar: float) -> float:
    n_rated = param("awe_rated_H2_flow") * P_STD / (R_GAS * T_STD)      # mol/h
    mdot_rated = n_rated * M_H2                                       # kg/h
    return k_p_mol_bar * M_H2 / mdot_rated                            # h/bar


def l32_rate_kW_H2(P_kW: float, T_C: float) -> float:
    """L32 Eq.(1), branch 0 <= P <= P_nom (kW_H2)."""
    return param("l32_m_T") * T_C + param("l32_k_E") + param("l32_m_low") * (P_kW - param("l32_P_nom"))


def item2(extra_l32: bool = False) -> pd.DataFrame:
    rows = []
    lo, hi = [float(x) for x in str(load_registry().set_index("name").loc["pressurisation_power_range_V1", "value_or_range"]).split("-")]
    mid = 0.5 * (lo + hi)
    d_mid, d_lo, d_hi = l13_inventory(mid), l13_inventory(lo), l13_inventory(hi)
    rows.append(dict(id="PEM_L13_nominal", role="NOMINAL", source="L13 Persson 2020 (1.7 kg, 45 min, 30 bar inferred, 250 kW)", formula="i = m/((p_op-p0) mdot_rated); mdot_rated = P_rated x m/t / P_press; P_press = midpoint of 82-143 kW (implementer choice, flagged for analyst)", i_min_per_bar=d_mid["i_h_per_bar"] * 60, inputs=f"P_press={mid} kW; SEC={d_mid['SEC_kWh_kg']:.3f} kWh/kg; mdot_rated={d_mid['mdot_rated_kg_h']:.4f} kg/h; dp={d_mid['dp_bar']:.4f} bar"))
    rows.append(dict(id="PEM_L13_end_82kW", role="UQ_END", source="L13", formula="as above, P_press = 82 kW", i_min_per_bar=d_lo["i_h_per_bar"] * 60, inputs=f"SEC={d_lo['SEC_kWh_kg']:.3f}; mdot_rated={d_lo['mdot_rated_kg_h']:.4f}"))
    rows.append(dict(id="PEM_L13_end_143kW", role="UQ_END", source="L13", formula="as above, P_press = 143 kW", i_min_per_bar=d_hi["i_h_per_bar"] * 60, inputs=f"SEC={d_hi['SEC_kWh_kg']:.3f}; mdot_rated={d_hi['mdot_rated_kg_h']:.4f}"))
    tab, cal = vt2_awe()
    kp = awe_gas_capacity_mol_per_bar(cal)
    i28 = l28_inventory(kp)
    rows.append(dict(id="AWE_L28_CAL_PS75", role="NOMINAL_AWE", source="L28 Table 13 PS75 (CAL), D25", formula="K_p = PS/100 x n_dot_rated/gamma; i = K_p M_H2 / mdot_rated", i_min_per_bar=i28 * 60, inputs=f"K_p={kp:.4f} mol/bar; rated flow {param('awe_rated_H2_flow')} Nm3/h"))
    ks = []
    reg = load_registry().set_index("name")
    for ps in (20, 30, 35, 50, 75):
        t_i, g_i = parse_ps_row(reg.loc[f"awe_pressurisation_PS{ps}pct", "value_or_range"])
        k_i = ps / 100.0 * param("awe_rated_H2_flow") * P_STD / (R_GAS * T_STD) / 60.0 / g_i       # mol/bar (n_dot per min / gamma per min)
        ks.append((ps, k_i, l28_inventory(k_i)))
        rows.append(dict(id=f"AWE_L28_pointwise_PS{ps}", role="POST_VALIDATION_POINT", source="L28 Table 13 (contains VAL data; never a nominal)", formula="K_p,i = PS_i/100 n_dot_rated / gamma_i", i_min_per_bar=l28_inventory(k_i) * 60, inputs=f"K_p,i={k_i:.4f} mol/bar"))
    vals = [k[2] * 60 for k in ks]
    rows.append(dict(id="AWE_post_validation_min", role="AWE_RANGE_END", source="L28 point-wise PS_i/gamma_i, label: post-validation range", formula="min over five set points", i_min_per_bar=min(vals), inputs=""))
    rows.append(dict(id="AWE_post_validation_max", role="AWE_RANGE_END", source="L28 point-wise PS_i/gamma_i, label: post-validation range", formula="max over five set points", i_min_per_bar=max(vals), inputs=""))
    pem_ends = [d_lo["i_h_per_bar"] * 60, d_hi["i_h_per_bar"] * 60, i28 * 60]
    lab = ["L13 82 kW", "L13 143 kW", "L28 AWE CAL"]
    rows.append(dict(id="PEM_UQ_item2_min", role="PEM_RANGE_END", source=f"min of [L13 ends, L28 CAL]: {lab[int(np.argmin(pem_ends))]}", formula="PL8/D32b range [L13-derived, L28-derived]", i_min_per_bar=min(pem_ends), inputs=""))
    rows.append(dict(id="PEM_UQ_item2_max", role="PEM_RANGE_END", source=f"max of [L13 ends, L28 CAL]: {lab[int(np.argmax(pem_ends))]}", formula="same", i_min_per_bar=max(pem_ends), inputs=""))
    df = pd.DataFrame(rows)
    return df


def item15_rows(df2: pd.DataFrame) -> pd.DataFrame:
    """D37 range rule: add L32-derived i_L32(p_set) at 25 and 40 bar; PEM range becomes [min, max] of (L32, L13, L28)."""
    mdot_rated_32 = l32_rate_kW_H2(param("l32_P_nom"), 60.0) / param("l32_LHV_basis")          # Nm3/h, 100 kW, 60 degC
    mdot_100_25 = l32_rate_kW_H2(param("l32_P_nom"), 25.0) / param("l32_LHV_basis")           # Nm3/h at 25 degC
    rows = []
    ends = []
    for p_set in (25.0, 40.0):
        i_h_bar = (mdot_100_25 / mdot_rated_32) * (5.0 / 60.0) / (p_set - P0_BAR)
        rows.append(dict(id=f"PEM_L32_end_{int(p_set)}bar", role="UQ_END_L32", source="L32 Eq.(1) + 'roughly five minutes' (Sec 2.1); upper bound of the inventory (5 min also contains start-up)",
                         formula="i = [mdot(100 kW,25 C)/mdot_rated(100 kW,60 C)] x 5 min / (p_set - p0)", i_min_per_bar=i_h_bar * 60,
                         inputs=f"mdot(25C)={mdot_100_25:.4f} Nm3/h; mdot_rated(60C)={mdot_rated_32:.4f} Nm3/h; p_set={p_set} bar(a); p0={P0_BAR}"))
        ends.append(i_h_bar * 60)
    base = df2[df2.id.isin(["PEM_L13_end_82kW", "PEM_L13_end_143kW", "AWE_L28_CAL_PS75"])]
    allv = list(base.i_min_per_bar) + ends
    labs = list(base.id) + ["PEM_L32_end_25bar", "PEM_L32_end_40bar"]
    rows.append(dict(id="PEM_UQ_final_min", role="PEM_RANGE_END", source=f"min of [L32, L13, L28]: {labs[int(np.argmin(allv))]}", formula="D37 range rule", i_min_per_bar=min(allv), inputs=""))
    rows.append(dict(id="PEM_UQ_final_max", role="PEM_RANGE_END", source=f"max of [L32, L13, L28]: {labs[int(np.argmax(allv))]}", formula="D37 range rule", i_min_per_bar=max(allv), inputs=""))
    return pd.DataFrame(rows)


def item14() -> pd.DataFrame:
    """VT2b exactly per D37. Prediction t_p = m_inv(p_set) / mdot_H2(100 kW, 25 degC), m_inv = i_nominal x (p_set - p0) x mdot_rated(L32)."""
    df2 = pd.read_csv(RESULTS / "inventory_ranges_item2.csv")
    i_nom_h = float(df2.loc[df2.id == "PEM_L13_nominal", "i_min_per_bar"].iloc[0]) / 60.0       # h/bar
    kgh = lambda nm3h: nm3h * P_STD / (R_GAS * T_STD) * M_H2                                       # Nm3/h -> kg/h
    rate_kW = l32_rate_kW_H2(param("l32_P_nom"), 25.0)
    rate_kW_rated = l32_rate_kW_H2(param("l32_P_nom"), 60.0)
    mdot = kgh(rate_kW / param("l32_LHV_basis"))
    mdot_rated = kgh(rate_kW_rated / param("l32_LHV_basis"))
    rows = []
    for p_set in (25.0, 32.5, 40.0):
        m_inv = i_nom_h * (p_set - P0_BAR) * mdot_rated
        t_pred = m_inv / mdot * 60.0
        re = (t_pred - 5.0) / 5.0
        rows.append(dict(p_set_bar=p_set, m_inv_kg=m_inv, m_dot_H2_kg_h=mdot, m_dot_H2_rated_kg_h=mdot_rated, t_pred_min=t_pred, target_min=5.0, RE=re, abs_RE_le_20pct=bool(abs(re) <= 0.20), row_type="primary"))
    verdict = "PASS" if any(r["abs_RE_le_20pct"] for r in rows) else "FAIL"
    for r in rows:
        r["verdict_rule"] = f"PASS if abs RE <= 20 % at ANY p_set; overall = {verdict}"
    # sensitivity (no verdict): H2 rate from the plant engine polarisation (D29 + L13 Eq.16) scaled to 100 kW (rated 100 kW at 1.75 A/cm2/75 C; 100 kW operating point)
    from h2su.electrolyzer import calibrate_sigma_scale, cell_voltage_d29, faradaic_efficiency_persson
    from h2su.constants import F_PAPER
    ss = calibrate_sigma_scale()
    j = 1.75
    T = 75 + 273.15
    U = float(cell_voltage_d29(j, T, 200.0, ss))
    eta = float(faradaic_efficiency_persson(U, T))
    # kg H2 per kWh of stack energy = eta_F * M_H2 / (2 F U) [kg/J] -> x 3.6e6 J/kWh
    kg_per_kWh = eta * M_H2 / (2 * F_PAPER * U) * 3.6e6
    mdot_eng = 100.0 * kg_per_kWh
    for p_set in (25.0, 32.5, 40.0):
        m_inv = i_nom_h * (p_set - P0_BAR) * mdot_eng        # rated flow of the same engine at 100 kW
        t_pred = m_inv / mdot_eng * 60.0                     # rated normalisation cancels when rate = rated: sensitivity uses engine rate at 100 kW for both
        rows.append(dict(p_set_bar=p_set, m_inv_kg=m_inv, m_dot_H2_kg_h=mdot_eng, m_dot_H2_rated_kg_h=mdot_eng, t_pred_min=t_pred, target_min=5.0, RE=(t_pred - 5.0) / 5.0,
                         abs_RE_le_20pct=bool(abs((t_pred - 5.0) / 5.0) <= 0.2), row_type="sensitivity_no_verdict", verdict_rule="engine rate at 100 kW (L13 Eq.16 + D29 polarisation, stack-level at 1.75 A/cm2, 75 C); rated flow of the same engine"))
    return pd.DataFrame(rows), verdict


def thermal_ranges() -> pd.DataFrame:
    """C and R per MW (D32a/D37). L26 ends: printed lumped system totals at 1000 kW (4407.5 kJ/K, 2.3 K/kW) and Table 3 x Table 4 sums
    (lumped_from_tables, row 1000); L52-derived ends: C/P_stack and R x P_stack/1 MW (model-derived)."""
    from h2su.thermal import Rauls, load_tables, lumped_from_tables
    t = load_tables()
    row = t.loc[1000]
    C_tab, G_tab = lumped_from_tables(row, 1, Rauls())                         # J/K, W/K
    lum = pd.read_csv(Path(__file__).parent / "configs" / "tables" / "L26_rauls2023_published_lumped.csv", comment="#")
    C_pr = float(lum[(lum.quantity == "C_th_system_total") & (lum.P_kW == 1000)].value.iloc[0]) * 1e3      # J/K
    R_pr = float(lum[(lum.quantity == "R_th_system_total") & (lum.P_kW == 1000)].value.iloc[0])           # K/kW
    R_tab = 1.0 / G_tab * 1e3                                                                              # K/kW
    P_stack = param("l52_P_stack")
    C52 = [param("l52_C_text") / P_stack * 1e6, param("l52_C_table6") / P_stack * 1e6]                     # J/K per MW
    UA = param("l52_UA")
    R52_K_per_kW = (1.0 / UA) * P_stack / 1e6 * 1e3                                                        # K/kW at 1 MW
    rows = [
        dict(quantity="C_per_MW", end="L26 printed lumped system total (1000 kW)", value=C_pr / 1e3, unit="kJ/K per MW", label="measured-design source, printed text (L26)"),
        dict(quantity="C_per_MW", end="L26 Table 3 x Table 4 sum (row 1000 kW)", value=C_tab / 1e3, unit="kJ/K per MW", label="L26 tables, specific heats Table 3"),
        dict(quantity="C_per_MW", end="L52 text C/P_stack", value=C52[0] / 1e3, unit="kJ/K per MW", label="model-derived (L52), not measured"),
        dict(quantity="C_per_MW", end="L52 Table 6 C/P_stack", value=C52[1] / 1e3, unit="kJ/K per MW", label="model-derived (L52), not measured"),
        dict(quantity="R_per_MW", end="L26 printed lumped system total (1000 kW)", value=R_pr, unit="K/kW (1 MW plant)", label="L26 printed text"),
        dict(quantity="R_per_MW", end="L26 table-derived (1/G, row 1000 kW)", value=R_tab, unit="K/kW (1 MW plant)", label="L26 tables, h = 21 and 2.1 W/(m2 K)"),
        dict(quantity="R_per_MW", end="L52 1/UA scaled to 1 MW", value=R52_K_per_kW, unit="K/kW (1 MW plant)", label="model-derived (L52), not measured"),
    ]
    df = pd.DataFrame(rows)
    out = []
    for q in ("C_per_MW", "R_per_MW"):
        s = df[df.quantity == q]
        out.append(dict(quantity=q, end="HULL_MIN", value=s.value.min(), unit=s.unit.iloc[0], label=s.loc[s.value.idxmin(), "end"] + " | " + s.loc[s.value.idxmin(), "label"]))
        out.append(dict(quantity=q, end="HULL_MAX", value=s.value.max(), unit=s.unit.iloc[0], label=s.loc[s.value.idxmax(), "end"] + " | " + s.loc[s.value.idxmax(), "label"]))
    return pd.concat([df, pd.DataFrame(out)], ignore_index=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "item2":
        df = item2()
        df.to_csv(RESULTS / "inventory_ranges_item2.csv", index=False)
        df.to_csv(RESULTS / "inventory_ranges.csv", index=False)
        print(df[["id", "role", "i_min_per_bar"]].to_string(index=False))
    elif stage == "item14":
        df, v = item14()
        df.to_csv(RESULTS / "vt2b_l32.csv", index=False)
        print(df[["p_set_bar", "m_inv_kg", "m_dot_H2_kg_h", "t_pred_min", "RE", "abs_RE_le_20pct", "row_type"]].to_string(index=False)); print("VT2b verdict", v)
    elif stage == "item15":
        df2 = pd.read_csv(RESULTS / "inventory_ranges_item2.csv")
        df = pd.concat([df2, item15_rows(df2)], ignore_index=True)
        df.to_csv(RESULTS / "inventory_ranges.csv", index=False)
        print(df[["id", "role", "i_min_per_bar", "source"]].to_string(index=False))
        th = thermal_ranges()
        th.to_csv(RESULTS / "thermal_ranges.csv", index=False)
        print(th.to_string(index=False))

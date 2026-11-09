"""D46 dryer case definitions (L15) for the Python reference and the user-run COMSOL package.

Imported lazily by ver6_cases.build() when results/vt6_cal_frozen.json exists. Lab case = BT2 breakthrough; plant cases = L15 scale-up rule
(bed volume proportional to the feed H2 mass flow, H/D = 10, SI Table S3 design) at the 1 MW plant rated H2 flow, Table 9 operating point.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import dryer as dr
from .io import RESULTS

FROZEN = RESULTS / "vt6_cal_frozen.json"
L15M = "L15 Santos 2026 Ind Eng Chem Res 65:17315 (ACS ASAP version, user file 787929.pdf)"
L15S = "L15 Santos 2026 Ind Eng Chem Res 65:17315 Supporting Information (Europe PMC PMC13495766)"


def _reg_src(df, name):
    r = df.loc[df["name"] == name].iloc[0]
    return float(r["value_or_range"]), r["unit"], r["source_id"], r["location"]


def global_rows(df):
    rows = []
    spec = (("toth_qm_mol_kg", "l15_isoW_qm", 1, "mol/kg"), ("toth_f0", "l15_isoW_f0", 1, "-"), ("toth_alpha", "l15_isoW_alpha", 1, "-"),
            ("toth_b20_1_bar", "l15_isoW_b20", 1, "1/bar"), ("toth_Q2_J_mol", "l15_isoW_Q2", 1e3, "J/mol"), ("toth_b30_1_bar", "l15_isoW_b30", 1, "1/bar"),
            ("toth_Q3_J_mol", "l15_isoW_Q3", 1e3, "J/mol"), ("toth_n", "l15_isoW_n", 1, "-"), ("iso_T0_K", "l15_T0", 1, "K"),
            ("lang_qm_mol_kg", "l15_isoH2_qm", 1, "mol/kg"), ("lang_b10_1_bar", "l15_isoH2_b10", 1, "1/bar"), ("lang_Q1_J_mol", "l15_isoH2_Q1", 1e3, "J/mol"))
    for out, reg, mult, unit in spec:
        v, _, src, loc = _reg_src(df, reg)
        rows.append(dict(scope="global", name=out, value=v * mult, unit=unit, source_id=src, location=loc, extraction="table", role="INPUT",
                         note=f"registry row {reg}"))
    rows.append(dict(scope="global", name="smooth_dy_umol_mol", value=dr.DELTA_Y, unit="umol/mol", source_id="numerical regularisation of max(y, 0) in the isotherm argument (code/src/h2su/dryer.py DELTA_Y)",
                     location="Brief 4 item 1 (stalling of the BDF integrator at the kink); 2e-5 of the 5 umol/mol spec", extraction="numerical setting", role="INPUT", note="no physical meaning"))
    return rows


def _D(v, u, src, loc, note=""):
    return (v, u, src, loc, "derived/table", note)


def build(df: pd.DataFrame, Case):
    from .plantrun import stack_tables
    iso = dr.iso_params(df)
    fr = json.loads(FROZEN.read_text(encoding="utf-8"))
    k = fr["value"]
    s4 = pd.read_csv(Path(__file__).resolve().parents[2] / "configs" / "tables" / "L15_S4.csv").set_index("experiment")
    FR = f"VT6 CAL on BT2 (results/vt6_cal_frozen.json, k_LDF = {k:.6g} 1/s)"
    lab = dr.lab_case(df, iso, s4.loc["BT2"].to_dict(), k, t_end_s=6 * 3600.0, dt_out_s=30.0)

    def common(q):
        return dict(
            eps=_D(q["eps"], "-", L15M, "text after eq 20 (p. L): bed porosity (assumed 0.4); ASSUMED BY THE SOURCE", "flag: not measured"),
            rho_bed_kg_m3=_D(q["rho_bed_kg_m3"], "kg/m3", L15M, "p. N: packing density of 790 kg m-3 (SI Table S3: 789.43)"),
            Dax_m2_s=_D(0.0, "m2/s", "no source: axial-dispersion correlation (Wakao-Funazkri, L49) not available from a tool result", "Brief 4 item 1 log",
                        "set to zero; effective dispersion is absorbed by k_LDF; numerical dispersion of the upwind scheme only"),
            kLDF_1_s=_D(k, "1/s", FR, "VT6 CAL (D46)"),
            dt_out_s=_D(30.0, "s", "VER-6 case design", "output step"),
            tol_check=(0.01, "-", "n/a", "no analytic check (code-to-code case)", "criterion", ""),
        )

    cases = []
    t_AD = _reg_src(df, "l15_opt_tAD")[0] * 3600.0
    WIN = _D(t_AD, "s", L15M, "Table 9: t_AD = 2.95 h; spec events (time to spec, vented H2) are evaluated inside the adsorption window")
    pp = dict(common(lab),
              L_m=_D(lab["L_m"], "m", L15S, "Table S2 Hbed = 10 cm"), A_m2=_D(lab["A_m2"], "m2", L15S, "Table S2 Di = 1.0 cm: pi/4 Di^2"),
              N_flux_mol_m2_s=_D(lab["N_flux_mol_m2_s"], "mol/(m2 s)", L15S, "Table S1: 1.5 LN/min (normal state 0 degC, 101325 Pa assumed) / (60 s Vm_N A)"),
              p_Pa=_D(lab["p_Pa"], "Pa", L15S, "Table S4 BT2: 8 bar"), T_K=_D(lab["T_K"], "K", L15S, "Table S2: 28.5 degC"),
              yW_in_umol_mol=_D(float(lab["yW_in_umol_mol"]), "umol/mol", L15S + " + eq 7 (p. D)", "RH x psat(T)/P (Table S4 RH 70 %; eq 7)", "derived"),
              yW_init_umol_mol=_D(0.0, "umol/mol", "D46", "dry initial gas"),
              q_init_mol_kg=_D(0.0, "mol/kg", L15M, "p. H: bed regenerated overnight (q0 = 0, D46 start-up (i))"),
              t_end_s=_D(12 * 3600.0, "s", "VER-6 case design", "output horizon (the lumped forms reach 99 % of the feed value later than the 1-D bed)"),
              use_breakthrough=_D(1.0, "-", "VER-6 case design", "report first moment"), purge_all=_D(0.0, "-", "VER-6 case design", ""),
              p_init_Pa=_D(lab["p_Pa"], "Pa", "VER-6 case design", "no re-pressurisation in this case"), two_stage=_D(0.0, "-", "VER-6 case design", ""),
              t_spec_window_s=WIN)
    cases.append(Case("dry_bt2", "model_dryer_startup_regen", "dryer", "lab BT2 breakthrough (L15 Table S4), frozen k_LDF, q0 = 0", pp))

    m_rated = float(stack_tables()["m_rated_kg_h"])
    Lp, Ap, Vp = dr.plant_geometry(df, m_rated)
    P_AD = _reg_src(df, "vpsa_P_AD_opt")[0] * 1e5
    T = lab["T_K"]
    rh = _reg_src(df, "l15_feed_RH")[0] / 100.0
    y_in = rh * dr.psat_bar(T, iso) / (P_AD / 1e5) / 1e-6
    n_h2 = m_rated / 3600.0 / dr.M_H2
    N_tot = n_h2 / (1.0 - y_in * 1e-6)
    pf = _reg_src(df, "vpsa_purge_to_feed")[0] / 100.0
    P_regen = _reg_src(df, "vpsa_P_regen")[0] * 1e5
    qsat = float(dr.q_water(y_in * 1e-6 * P_AD / 1e5, T, iso))
    PLANT = f"plant engine rated H2 flow {m_rated:.6g} kg/h (h2su.plantrun.stack_tables, 1 MW at 1.75 A/cm2, 75 degC)"
    base = dict(common(lab))
    base.update(
        L_m=_D(Lp, "m", L15S + " Table S3 scale-up rule", f"H = 10 D, V = {Vp:.6g} m3 = V_design x m_rated/300 kg/h; {PLANT}"),
        A_m2=_D(Ap, "m2", L15S + " Table S3 scale-up rule", "pi/4 D^2 with D = (4 V/(10 pi))^(1/3)"),
        T_K=_D(T, "K", L15S, "Table S2"),
        yW_in_umol_mol=_D(y_in, "umol/mol", L15S + " + eq 7", "RH 80 % (Table S2) x psat(28.5 degC)/P_AD", "derived"),
        t_end_s=_D(12 * 3600.0, "s", "VER-6 case design", "output horizon (the plant-scale tail reaches 99 % of the feed value only after about 6-12 h)"),
        t_spec_window_s=WIN,
        use_breakthrough=_D(0.0, "-", "VER-6 case design", ""), purge_all=_D(0.0, "-", "VER-6 case design", ""), two_stage=_D(0.0, "-", "VER-6 case design", ""))
    adsorb = dict(N_flux_mol_m2_s=_D(N_tot / Ap, "mol/(m2 s)", PLANT + "; " + L15M + " Table 9", "total feed flux = n_H2/(1 - y_in)/A"),
                  p_Pa=_D(P_AD, "Pa", L15M, "Table 9: P_AD = 7.8 bar"), p_init_Pa=_D(P_regen, "Pa", L15M, "regeneration pressure 0.025 bar (Table 7 / S2)"))
    c1 = dict(base, **adsorb)
    c1.update(yW_init_umol_mol=_D(0.0, "umol/mol", "D46", "dry"), q_init_mol_kg=_D(0.0, "mol/kg", "D46", "start-up (i): q0 = 0"),
              use_breakthrough=_D(1.0, "-", "VER-6 case design", "report first moment"))
    cases.append(Case("dry_su_i", "model_dryer_startup_regen", "dryer", "plant-scale start-up (i): fully regenerated bed, adsorption at the Table 9 optimum", c1))
    regen = dict(N_flux_mol_m2_s=_D(pf * N_tot / Ap, "mol/(m2 s)", L15M, "P/F = 1 % of the feed molar flux (Table 7 / S2)"),
                 p_Pa=_D(P_regen, "Pa", L15M, "regeneration pressure 0.025 bar (Table 7 / S2)"),
                 yW_in_umol_mol=_D(0.0, "umol/mol", L15M, "dry H2 purge"),
                 yW_init_umol_mol=_D(0.0, "umol/mol", "D46", "void gas replaced by dry purge at the start of the step (declared)"),
                 q_init_mol_kg=_D(qsat, "mol/kg", "derived: L15 eq 2-4 at the feed water pressure", "saturated start state (BT regeneration starts from complete saturation, p. I)"),
                 t_end_s=_D(1.5 * 3600.0, "s", L15S, "Table S4: 1.5 h regeneration (text/table disagree on BT6/BT7, see l15_consistency.csv)"),
                 p_init_Pa=_D(P_AD, "Pa", L15M, "Table 9"), purge_all=_D(1.0, "-", "VER-6 case design", "all H2 leaving counts as purge H2"))
    c2 = dict(base, **regen)
    cases.append(Case("dry_regen", "model_dryer_startup_regen", "dryer",
                      "plant-scale one regeneration step (1.5 h, 25 mbar, P/F = 1 %) from the saturated bed (trace-species form; code-to-code only)", c2))
    c3 = dict(base, **adsorb)
    c3.update(yW_init_umol_mol=_D(0.0, "umol/mol", L15M, "gas phase re-pressurised with dry product H2 (backfill, Table 3 stage III)"), q_init_mol_kg=_D(0.0, "mol/kg", "D46", "overwritten by the reversed stage-1 profile"),
              use_breakthrough=_D(1.0, "-", "VER-6 case design", ""), two_stage=_D(1.0, "-", "D46", "stage 1 = regeneration (as dry_regen), stage 2 = adsorption on the reversed profile"))
    for n in ("N_flux_mol_m2_s", "p_Pa", "yW_in_umol_mol", "yW_init_umol_mol", "q_init_mol_kg", "t_end_s"):
        c3["s1_" + n] = regen[n]
    c3["s1_dt_out_s"] = _D(30.0, "s", "VER-6 case design", "")
    cases.append(Case("dry_su_ii", "model_dryer_startup_regen", "dryer",
                      "plant-scale start-up (ii): partially regenerated state computed by each model (1.5 h regeneration, bed reversed), then adsorption", c3))
    return cases

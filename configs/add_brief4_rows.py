"""Brief 4 item 1a: registry rows for L15 (ACS ASAP version, user file 787929.pdf; SI Europe PMC PMC13495766). Idempotent."""
import pandas as pd
from pathlib import Path

p = Path(__file__).parent / "parameters.csv"
df = pd.read_csv(p, dtype=str, keep_default_na=False)
cols = list(df.columns)
M = "L15 Santos 2026 Ind Eng Chem Res 65:17315 (ACS published ASAP version, CC-BY 4.0, user file 787929.pdf, rendered pages read 2026-10-30)"
S = "L15 Santos 2026 Ind Eng Chem Res 65:17315 Supporting Information (Europe PMC PMC13495766, CC-BY 4.0, ie6c00750_si_001.pdf, rendered page read 2026-10-30)"
rows = [
    ("l15_isoH2_qm", "q_m", "0.057", "mol/kg", "PEM", M, "Table 5 (p. G); eq 1 (p. C) q = q_m b1 P/(1 + b1 P)", "table", "none", "INPUT", "H2 Langmuir; used only for the adsorbed-H2 term of re-pressurisation"),
    ("l15_isoH2_b10", "b1_0", "0.1178", "1/bar", "PEM", M, "Table 5 (p. G); eq 3 (p. D)", "table", "none", "INPUT", ""),
    ("l15_isoH2_Q1", "Q1", "17.26", "kJ/mol", "PEM", M, "Table 5 (p. G)", "table", "none", "INPUT", ""),
    ("l15_isoW_qm", "q_m", "13.5", "mol/kg", "PEM", M, "Table 6 (p. H); eq 2 (p. C) q = [q_m b2 P/(1 + (b2 P)^f)^(1/f)] [1/(1 - b3 P)^n]", "table", "none", "INPUT", "Toth-Aranovich-Donohue; P = water partial pressure in bar (Fig. 5a axis P/bar up to 0.07)"),
    ("l15_isoW_f0", "f0", "0.286", "-", "PEM", M, "Table 6 (p. H); eq 4 (p. D) f = f0 + alpha [1 - T0/T]", "table", "none", "INPUT", ""),
    ("l15_isoW_alpha", "alpha", "0.0073", "-", "PEM", M, "Table 6 (p. H)", "table", "none", "INPUT", ""),
    ("l15_isoW_b20", "b2_0", "3262.0", "1/bar", "PEM", M, "Table 6 (p. H); eq 3 (p. D)", "table", "none", "INPUT", ""),
    ("l15_isoW_Q2", "Q2", "46.17", "kJ/mol", "PEM", M, "Table 6 (p. H)", "table", "none", "INPUT", ""),
    ("l15_isoW_b30", "b3_0", "7.1", "1/bar", "PEM", M, "Table 6 (p. H); eq 3 (p. D)", "table", "none", "INPUT", ""),
    ("l15_isoW_Q3", "Q3", "41.37", "kJ/mol", "PEM", M, "Table 6 (p. H)", "table", "none", "INPUT", ""),
    ("l15_isoW_n", "n", "23.6", "-", "PEM", M, "Table 6 (p. H)", "table", "none", "INPUT", ""),
    ("l15_isoW_RMSE", "", "0.21", "mol/kg", "PEM", M, "Table 6 (p. H)", "table", "none", "AUX", "fit RMSE of the source"),
    ("l15_T0", "T0", "273.15", "K", "PEM", M, "text after eq 3 (p. D): reference temperature set to 273.15 K", "text", "none", "INPUT", ""),
    ("l15_magnus_A", "", "6.1078", "hPa (unit not printed; hPa confirmed by the absolute-humidity check in results/l15_consistency.csv)", "PEM", M, "eq 7 (p. D) P_H2O = 6.1078 exp(17.269388 (Tdp - 273.15)/(Tdp - 35.86)), Tdp in K", "equation", "none", "INPUT", "Magnus-Tetens over liquid water, as printed"),
    ("l15_magnus_B", "", "17.269388", "-", "PEM", M, "eq 7 (p. D)", "equation", "none", "INPUT", ""),
    ("l15_magnus_C", "", "35.86", "K", "PEM", M, "eq 7 (p. D)", "equation", "none", "INPUT", ""),
    ("l15_bt_breakpoint", "", "5", "umol/mol", "PEM", M, "eq 9 text (p. D): breakpoint 5 umol/mol of H2; Table S4 header t_b at Tdp -65.5 degC", "text", "none", "INPUT", "t_b definition used by VT6 (first crossing of 5 umol/mol)"),
    ("l15_bt_table_S4", "", "file:configs/tables/L15_S4.csv", "-", "PEM", S, "Table S4 (SI page 8 of 13, rendered page read; text layer garbled)", "transcribed from rendered page", "none", "VAL", "BT1-BT8 conditions and results; CAL = BT2, VAL = BT1/BT3/BT4/BT5/BT8 (D46)"),
    ("l15_bed_bead_diameter", "d_p", "0.84", "mm", "PEM", M, "Technical Limitations of Scale-Up (p. M): laboratory-scale value of 0.84 mm; Table 10", "text", "none", "INPUT", "not used by the isothermal LDF model (no pressure drop)"),
    ("l15_bed_porosity", "eps", "0.4", "-", "PEM", M, "text after eq 20 (p. L): bed porosity (assumed 0.4)", "text", "none", "INPUT", "ASSUMED BY THE SOURCE (not measured); flagged"),
    ("l15_bed_packing_density", "rho_bed", "790", "kg/m3", "PEM", M, "Economic assessment (p. N): packing density of 790 kg m-3 determined from the laboratory setup", "text", "none", "INPUT", "SI Table S3 prints 789.43 (see results/l15_consistency.csv)"),
    ("l15_bed_lab_breakthrough_mass", "m_ads", "6.17", "g", "PEM", S, "Table S1 (breakthrough bed; Di 1.0 cm, H 10 cm, V 7.85 cm3)", "table", "none", "INPUT", "BT bed; the VPSA bed has 6.11 g (Table S2)"),
    ("l15_design_feed_mass_flow", "", "300", "kg/h H2", "PEM", M, "Technical Limitations of Scale-Up (p. M): 300 kg/h; SI Table S3 text", "text", "none", "INPUT", "L15 scale-up design point"),
    ("l15_design_vessel_Di", "", "0.346", "m", "PEM", S, "Table S3 and Economic-analysis text (SI p. 3)", "table", "none", "INPUT", "per vessel; Hv = 3.46 m, V = 0.326 m3 (Table S3); H/D = 10"),
    ("l15_design_vessel_Hv", "", "3.46", "m", "PEM", S, "Table S3", "table", "none", "INPUT", ""),
    ("l15_design_superficial_velocity", "", "0.354", "m/s", "PEM", M, "p. M: 300 kg/h corresponds to a superficial velocity of 0.354 m/s at 7.8 bar", "text", "none", "AUX", "consistency check and D29 constant-velocity sensitivity"),
    ("l15_design_depressurisation_time", "", "40", "s", "PEM", M, "p. M: minimum depressurisation 7.8 bar to 25 mbar rounds to 40 s (SI Table S5)", "text", "none", "AUX", ""),
    ("l15_feed_RH", "RH", "80", "%", "PEM", S, "Table S2 (RH ca. 80 %) and SI Laboratory VPSA setup text", "table", "ca.", "INPUT", "feed humidity of the VPSA experiments including the optimum cycle (Table 9 does not restate it)"),
    ("l15_feed_T", "T", "28.5", "degC", "PEM", S, "Table S2", "table", "none", "INPUT", ""),
    ("l15_feed_F_lab", "F_feed", "1.5", "LN/min", "PEM", S, "Tables S1, S2", "table", "none", "INPUT", "LN = normal litres; normal reference state (0 degC, 101325 Pa) NOT printed -> standard convention assumed, flagged"),
    ("l15_regen_conditions_bt", "", "25 mbar, 15 mL/min dry H2 purge, 28.5 degC, overnight", "-", "PEM", M, "Breakthrough Characterization (p. H) and Temperature Fronts (p. J): regenerated overnight at 28.5 degC under reduced pressure (ca. 25 mbar) and a 0.015 L/min purging flow of dry hydrogen", "text", "none", "INPUT", "regeneration of the BT beds"),
    ("l15_cycle_steps", "", "3 beds, 120 deg offset; stage I AD; stage II DPE 1 s, BD 1 s, EV 2 s, PGV t_opened, EV t_closed; stage III PPE 1 s, BF 1 s, idle 2 s, t_opened, t_closed; stage duration t_AD", "-", "PEM", M, "Table 3 (p. F), eq 15-16 (p. G): t_opened = t_AD - (4 s + t_closed)", "table", "none", "AUX", "regeneration step (PGV) lasts t_AD minus a few seconds; theta_closed of the optimum is not printed (term not significant, removed from the model)"),
    ("l15_opt_tAD", "t_AD", "2.95", "h", "PEM", M, "Table 9 (p. L)", "table", "none", "INPUT", "same as vpsa_t_AD_opt"),
    ("l15_bed_temp_rise", "", "approx. 2.5 / 3.5 / 4 (BT1/BT2/BT3); approx. 6.5 (BT4); ca. 2.5 (BT8)", "degC", "PEM", M, "Temperature Fronts (p. I-J)", "text", "none", "AUX", "isothermal-model assumption check"),
    ("l15_bt6_bt7_text_vs_table", "", "text: BT6 = 1.5 h, BT7 = 2.5 h regeneration; Table S4: BT6 = 2.5 h, BT7 = 1.5 h", "-", "PEM", M + " vs " + S, "p. I (Fig. 11 text) vs SI Table S4", "text/table", "conflict in the source", "AUX", "unresolved source inconsistency; BT6/BT7 not executable (see results/vt6_l15.csv)"),
]
have = set(df["name"])
new = [dict(zip(cols, r)) for r in rows if r[0] not in have]
df = pd.concat([df, pd.DataFrame(new, columns=cols)], ignore_index=True)
df.to_csv(p, index=False)
print("rows added", len(new), "total", len(df))

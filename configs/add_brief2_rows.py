"""Appends the Brief 2 registry rows (each with source + location) to code/configs/parameters.csv. Idempotent."""
import pandas as pd
from pathlib import Path
p = Path(__file__).parent / "parameters.csv"
df = pd.read_csv(p, dtype=str, keep_default_na=False)
cols = list(df.columns)
L26 = "L26 Rauls 2023 Appl Energy 330:120350"
L44 = "L44 Springer, Zawodzinski, Gottesfeld 1991 J Electrochem Soc 138:2334 (doi 10.1149/1.2085971; IOP open PDF read 2026-10-30)"
L19 = "L19 Kwon 2023 Energies 16:6685 (open access full text read 2026-10-30)"
L4 = "L4 Virah-Sawmy 2024 IJHE 72:49-59"
rows = [
 # L44 conductivity (VT1 polarisation, D29)
 ("l44_sigma30_slope", "", "0.005139", "S/cm per (H2O/SO3-)", "PEM", L44, "Eq.[24] sigma_30C = 0.005139 lambda - 0.00326 (lambda > 1)", "text", "none", "INPUT", "Nafion 117 fit"),
 ("l44_sigma30_offset", "", "0.00326", "S/cm", "PEM", L44, "Eq.[24]", "text", "none", "INPUT", ""),
 ("l44_sigma_EaR", "", "1268", "K", "PEM", L44, "Eq.[25] sigma(T) = sigma_30C exp{1268 (1/303 - 1/(273+T_cell))}", "text", "none", "INPUT", ""),
 ("l44_sigma_Tref", "", "303", "K", "PEM", L44, "Eq.[25]", "text", "none", "INPUT", ""),
 ("l44_lambda_liquid_80C", "", "16.8", "H2O/SO3-", "PEM", L44, "Sec. membrane water content: immersed in liquid water, lambda 16.8 at 80 degC (22 at boiling)", "text", "none", "INPUT", "D29/VT1 uses 16.8 (liquid water, closest to 75 degC); the scale factor sigma_scale is calibrated once, so lambda only sets the T-shape (1268 K) and is immaterial"),
 ("l44_lambda_vapour_30C", "", "14", "H2O/SO3-", "PEM", L44, "Sec. membrane water content: saturated vapour at 30 degC", "text", "none", "INPUT", "sensitivity value"),
 # L26 heat-up strategies (Table 1 / Sec 3.1.1)
 ("l26_stacks_test_station", "", "2 stacks x 27 cells x 300 cm2", "-", "PEM", L26, "Sec 3.1", "text", "none", "INPUT", "SN04 text configuration"),
 ("l26_heater_power", "", "4", "kW_th", "PEM", L26, "Sec 3.1 / Table 1", "text", "none", "INPUT", "cartridge heaters (up to 4 kWth)"),
 ("l26_galv2_j_step1", "", "0.5", "A/cm2", "PEM", L26, "Table 1 two-step galvanostatic: 0.5 A/cm2 from 40 degC", "table", "none", "CAL", "VT1 CAL case (77 min)"),
 ("l26_galv2_j_step2_table", "", "1.0", "A/cm2", "PEM", L26, "Table 1 two-step galvanostatic: 1.0 A/cm2 from 60 degC to 75 degC", "table", "none", "CAL", "Table 1 reading (primary, fixed before any VT1 run); conflicts with text, SN23"),
 ("l26_galv2_j_step2_text", "", "1.5", "A/cm2", "PEM", L26, "Sec 3.1.1 text: 'After reaching 60 degC, the current density was increased to 1.50 A cm-2'", "text", "none", "AUX", "sensitivity reading, SN23"),
 ("l26_galv2_T_switch1", "", "40", "degC", "PEM", L26, "Table 1 / Sec 3.1.1", "table", "none", "CAL", ""),
 ("l26_galv2_T_switch2", "", "60", "degC", "PEM", L26, "Table 1 / Sec 3.1.1", "table", "none", "CAL", ""),
 ("l26_galv3_j_steps", "", "0.5; 1.0; 1.5", "A/cm2", "PEM", L26, "Table 1 three-step galvanostatic", "table", "none", "VAL", "VT1 VAL 59 min"),
 ("l26_galv3_T_switch", "", "40; 50; 60", "degC", "PEM", L26, "Table 1 / Sec 3.1.1", "table", "none", "VAL", ""),
 ("l26_membrane_um_test_station", "", "200", "um", "PEM", L26, "Sec 3.1 (Nafion 117) / Table 2 footnote", "text", "none", "INPUT", ""),
 # C3 data pointers (VER-2R)
 ("c3_eta_curves_fig2", "eta_max; eta_min1-3", "0.75; 0.70; 0.60; 0.55", "-", "PEM", L4, "Fig. 2 legend 'Efficiency curve k: 0.75-0.xx' (text in the figure)", "text (figure legend) + automated colour trace", "u_d = 0.002 (trace max deviation 0.0014 + half line width 0.0006)", "VER2", "frozen in code/configs/tables/L4_virahsawmy2024.csv, sha256 in results/ver2r_digitisation_hash.txt"),
 ("c3_ratio_table5", "", "0.75", "electrolyser/farm capacity", "PEM", L4, "Sec 4.2 'An electrolyzer to renewable energy ratio of 0.75 was used for all the sites'", "text", "none", "VER2", ""),
 ("c3_aemo_duids", "", "FINLYSF1; TBSF1; HAMISF1", "-", "-", "AEMO MMSDM 2022-12 DUDETAILSUMMARY + STATION (nemweb.com.au)", "station names 'Finely Solar Farm' NSW1, 'Tailem Bend Solar Project 1' SA1, 'Hamilton Solar Farm' QLD1", "table", "none", "VER2", "DUIDs are not printed in L4; matched by name, state; capacity factors within 0.44 pp of L4 Table 2 (results/ver2r_identity.csv)"),
 # L19 deoxo chamber (SN18 stays OPEN: no rate constants, no absolute dimensions)
 ("l19_dH_rxn_per_H2O", "dH", "-242", "kJ/mol (0.5 O2 + H2 -> H2O)", "PEM", L19, "Eq.(9)", "text", "none", "INPUT", "= -484 kJ per mol O2; compare NIST dHf H2O(g) -241.826"),
 ("l19_LD_ratio", "", "38.4", "-", "PEM", L19, "Sec 2.2.4 / Table 1 (chamber length 38.4 D)", "text", "none", "INPUT", "absolute D not printed (scale bar only)"),
 ("l19_catalyst_bed_length", "", "33.2", "x D", "PEM", L19, "Sec 2.2.4 / Table 1", "table", "none", "INPUT", ""),
 ("l19_porosity_range", "", "0.397-0.476", "-", "PEM", L19, "Table 1 / Sec 3.2 (0.476 base case, 0.397 from 3 mm particles)", "table", "none", "INPUT", ""),
 ("l19_catalyst_mean_diameter", "", "3", "mm", "PEM", L19, "Table 1", "table", "none", "INPUT", ""),
 ("l19_feed_mole_fractions", "", "H2 0.97250; O2 0.00096; H2O 0.02654", "-", "PEM", L19, "Table 1 (CFD conditions)", "table", "none", "INPUT", "O2 960 ppm after water separator (Sec 3.1)"),
 ("l19_operating_ranges", "", "p 6.0-7.0 bar; T_in 50-70 degC; flow 1.00-1.50 kg/h", "-", "PEM", L19, "Table 1", "table", "none", "INPUT", ""),
 ("l19_catalyst_Pd_loading", "", "0.3", "wt%", "PEM", L19, "Sec 3.1.1 (experiment)", "text", "none", "INPUT", ""),
 ("nist_dHf_H2O_gas", "", "-241.826", "kJ/mol", "-", "NIST Chemistry WebBook C7732185 (Cox, Wagman et al. 1984 CODATA Key Values), accessed 2026-10-30", "Gas phase thermochemistry, standard enthalpy of formation (+-0.040)", "web page", "none", "INPUT", "2 H2 + O2 -> 2 H2O(g): -483.652 kJ per mol O2"),
]
have = set(df["name"])
new = [dict(zip(cols, r)) for r in rows if r[0] not in have]
if new:
    df = pd.concat([df, pd.DataFrame(new, columns=cols)], ignore_index=True)
    df.to_csv(p, index=False)
print(len(new), "rows added; total", len(df))
bad = df[(df.source_id.str.strip() == "") | (df.location.str.strip() == "")]
print("rows without source/location:", len(bad))

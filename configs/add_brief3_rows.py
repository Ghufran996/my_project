"""Appends the Brief 3 registry rows (each with source + location) to code/configs/parameters.csv. Idempotent (skips existing names)."""
import pandas as pd
from pathlib import Path
p = Path(__file__).parent / "parameters.csv"
df = pd.read_csv(p, dtype=str, keep_default_na=False)
cols = list(df.columns)
L25 = "L25 Chen (Aijun) 2025 Sci Rep 15:23230 (Europe PMC PMC12223099, CC-BY 4.0, XML tables read 2026-10-30)"
L24 = "L24 Liu 2026 Energies 19:3844 (MDPI PDF, read 2026-10-30)"
L28 = "L28 Bonaccorsi 2025 Energies 18:3949 (MDPI PDF, read 2026-10-30)"
L32 = "L32 Flamm 2021 Appl Energy 281:116031 (ETH Research Collection accepted manuscript, read 2026-10-30)"
L52 = "L52 Pfennig 2025 IJHE 104:567-583 (user PDF S0360319924016471.pdf, read 2026-10-30)"
L53 = "L53 Sakas 2022 IJHE 47:4328 (inside LUT dissertation; values from D37 analyst reading, NOT re-read by implementer)"
L56 = "L56 Dobos 2014 PVWatts Version 5 Manual NREL/TP-6A20-62641 (pvwatts.nlr.gov/downloads/pvwattsv5.pdf, read 2026-10-30)"
L21 = "L21 Hu 2024 eTransportation 19:100304 (abstract value from cache; full text not read)"
L4 = "L4 Virah-Sawmy 2024 IJHE 72:49-59"
L1 = "L1 Renau 2025 IJHE 189:152126"
L13 = "L13 Persson 2020 IJHE 45:31396 (accepted manuscript)"
rows = [
 # 1a L25 Tables 1 and 3 (SN13 resolved)
 ("pem190_cells", "", "120", "-", "PEM", L25, "Table 3 'Cell'", "table", "none", "INPUT", "SN13 resolved"),
 ("pem190_rated_H2", "", "190", "Nm3/h (printed m3/h)", "PEM", L25, "Table 3 'Rated hydrogen production'", "table", "none", "INPUT", "scale anchor of the plant model (D32a)"),
 ("pem190_coldstart_T0", "", "26", "degC", "PEM", L25, "Table 3 'Cold start-up temperature'", "table", "none", "INPUT", "after 3 h shutdown (text)"),
 ("pem190_coldstart_p0", "", "0.12", "MPa", "PEM", L25, "Table 3 'Cold start-up pressure'", "table", "none", "INPUT", "depressurised stop (supports POL-A as practice, D37/M5)"),
 ("pem190_hotstart_T0", "", "63.5", "degC", "PEM", L25, "Table 3 'Thermal start-up temperature'", "table", "none", "INPUT", "held in shutdown 1 min (text)"),
 ("pem190_hotstart_p0", "", "1.58", "MPa", "PEM", L25, "Table 3 'Thermal start-up pressure'", "table", "none", "INPUT", "pressure held for a hot restart"),
 ("pem190_T_operating_range", "", "70-80", "degC", "PEM", L25, "Table 1 'Temperature'", "table", "none", "INPUT", ""),
 ("pem190_SEC_range", "", "3.8-5", "kWh/Nm3", "PEM", L25, "Table 1 'Energy consumption'", "table", "none", "INPUT", "plant-scale rated power about 0.72-0.95 MW"),
 ("pem190_startup_time_range", "", "1200-1800", "s", "PEM", L25, "Table 1 'Start-up times'", "table", "none", "INPUT", "generic PEM statement of the paper (not a measurement)"),
 ("pem190_startup_definition", "", "time for the current to rise from 0 A to the rated current (3800 A)", "-", "PEM", L25, "Sec. test description before Fig. 3 / Table 3", "text", "none", "INPUT", "defines t_su in D32a"),
 ("pem190_OinH2_coldstart_current", "", "3800", "A", "PEM", L25, "Cold start-up section: After 6340 s, when the rated current is reached, the O2 in H2 concentration stabilizes at approximately 0.25%", "text", "none", "AUX", "current printed at the 0.25 % instant; active area / current density NOT printed -> j-based VT4 on L25 NOT EXECUTABLE"),
 ("pem190_hotstart_avg_current_end", "", "3780.3", "A", "PEM", L25, "Thermal start-up section: After approximately 1100 s, the average current reached 3780.3 A", "text", "none", "AUX", "0.040 % is an average over the 1100 s ramp, no instant current -> NOT EXECUTABLE for VT4"),
 # 1e L24 Fig. 3 trace
 ("pem50_fig3_trace", "", "file:configs/tables/L24_fig3.csv", "ppm", "PEM", L24, "Fig. 3 OTH vs time (page 8, image object extracted at 1000 ppi)", "digitised (automated trace, 4 variants)", "u_d = 4.6-6.4 ppm at the steady points, 6.0 ppm at 120 s (see file)", "VAL", "trace central values 1001.9/9.80/8.06/6.22 ppm vs text 1001.77/9.33/7.70/5.86; the 300 A stage is a start-up transient, not steady"),
 ("pem50_OTH_300A_stage_duration", "", "approx. 0-130", "s", "PEM", L24, "Fig. 3 (current 300 A until the Current step marker)", "digitised", "about +-5 s", "AUX", "the 300 A value is a transient peak; no steady 300 A value is printed or plotted"),
 # L28 purge criterion
 ("awe_purge_termination", "", "fixed procedure time (7/7/7/7/6 min per test), pressure 0.3 -> 0.1 bar; no gas-composition criterion printed", "-", "AWE", L28, "Sec 6.3 / Table 12; Sec 3 description of the purging phase (aimed at removing residual nitrogen, gases discarded)", "text/table", "none", "AUX", "-> VT3 NOT EXECUTABLE as pre-registered (purge is not purity-terminated)"),
 # 1g PV chain and P_min / standby
 ("pv_gamma_pdc", "gamma", "-0.0047", "1/degC", "PV", L56, "Table 3, Standard module -0.47 %/C; Eq.(8) Pdc = Itr/1000 Pdc0 (1 + gamma (Tcell - Tref)), Tref = 25 C", "table", "none", "INPUT", "PVWatts V5 DC model"),
 ("pv_INOCT", "", "45", "degC", "PV", L56, "Sec 9 Thermal Model: INOCT 45 C (open rack, fixed)", "text", "none", "INPUT", "used with the NOCT relation Tc = Ta + (INOCT-20) G/800 (simplification of the Fuentes model; wind speed not in the BSRN series; stated)"),
 ("pv_system_loss", "", "0.14", "-", "PV", L56, "Table 6 total losses 14 % (Eqn 9); same default as the PVGIS requests (results/data/pvgis/requests.csv)", "table", "none", "INPUT", "applied as a DC derate; inverter not modelled (DC-coupled)"),
 ("pv_albedo", "", "0.2", "-", "PV", "UNSOURCED (pvlib default); see source_needed SN26", "-", "assumption", "none", "INPUT", "UNSOURCED assumption, flagged"),
 ("pv_dcdc_efficiency", "", "1.0", "-", "PV", "UNSOURCED: DC/DC (MPPT) efficiency BLOCKED (SN19)", "-", "assumption", "none", "INPUT", "no converter loss applied; stated limitation; the 14 % L56 derate is the only DC loss"),
 ("pmin_pem_low", "", "0.10", "x P_rated", "PEM", L4, "Sec 2.1.3.1 (also registry c3_Pmin)", "text", "none", "INPUT", "PEM P_min nominal and range lower end"),
 ("pmin_pem_high", "", "0.191", "x P_rated", "PEM", L1, "Sec 3.3 P_min total 19.1 % of rated input (also registry renau_Pmin_total); includes BoP", "text", "none", "INPUT", "PEM P_min range upper end"),
 ("pmin_awe_min", "", "0.156", "x P_rated", "AWE", L21, "abstract: minimum load extended to 15.6 %", "text", "none", "INPUT", "AWE P_min (lowest load reported)"),
 ("standby_power_low", "", "0.01", "x P_rated", "PEM", L4, "Sec 2.1.2 (also registry c3_standby)", "text", "none", "INPUT", "UQ range low end; nominal for POL-B/C/D"),
 ("standby_power_high", "", "0.18", "x P_rated", "PEM", L13, "Table 3: 44, 38, 45 kW hot standby of a 250 kW plant (45/250 = 0.18)", "table", "none", "INPUT", "UQ range high end"),
 # L15 SI (item 1b): bed geometry and static VPSA conditions; isotherm/LDF/cycle-step table are NOT in the SI (main text, not retrieved)
 ("l15si_bed_height", "", "10", "cm", "PEM", "L15 Santos 2026 Ind Eng Chem Res 65:17315 Supporting Information (Europe PMC PMC13495766, CC-BY 4.0, ie6c00750_si_001.pdf read 2026-10-30)", "Table S2 Hbed", "table", "none", "INPUT", "lab-scale packed bed of the VPSA experiments"),
 ("l15si_bed_inner_diameter", "", "1.0", "cm", "PEM", "L15 Santos 2026 Ind Eng Chem Res 65:17315 Supporting Information (Europe PMC PMC13495766, CC-BY 4.0, ie6c00750_si_001.pdf read 2026-10-30)", "Table S2 Di (Do = 1.3 cm)", "table", "none", "INPUT", ""),
 ("l15si_bed_volume", "", "7.85", "cm3", "PEM", "L15 Santos 2026 Ind Eng Chem Res 65:17315 Supporting Information (Europe PMC PMC13495766, CC-BY 4.0, ie6c00750_si_001.pdf read 2026-10-30)", "Table S2 Vbed", "table", "none", "INPUT", ""),
 ("l15si_bed_mass", "", "6.11", "g", "PEM", "L15 Santos 2026 Ind Eng Chem Res 65:17315 Supporting Information (Europe PMC PMC13495766, CC-BY 4.0, ie6c00750_si_001.pdf read 2026-10-30)", "Table S2 mbed (VPSA experiments; 6.17 g in Table S1 breakthrough bed)", "table", "none", "INPUT", ""),
 ("l15si_feed_flow", "", "1.5", "NL/min", "PEM", "L15 Santos 2026 Ind Eng Chem Res 65:17315 Supporting Information (Europe PMC PMC13495766, CC-BY 4.0, ie6c00750_si_001.pdf read 2026-10-30)", "Table S2 Ffeed", "table", "none", "INPUT", ""),
 ("l15si_feed_T_RH", "", "28.5 degC; RH about 80 %", "-", "PEM", "L15 Santos 2026 Ind Eng Chem Res 65:17315 Supporting Information (Europe PMC PMC13495766, CC-BY 4.0, ie6c00750_si_001.pdf read 2026-10-30)", "Table S2", "table", "none", "INPUT", ""),
 ("l15si_PF_P_EV", "", "P/F 1 %; P_EV 0.025 bar", "-", "PEM", "L15 Santos 2026 Ind Eng Chem Res 65:17315 Supporting Information (Europe PMC PMC13495766, CC-BY 4.0, ie6c00750_si_001.pdf read 2026-10-30)", "Table S2", "table", "none", "INPUT", "P/F of the static VPSA experiments (optimum-cycle value not restated)"),
 ("l15si_AA_bulk_density", "", "789.43", "kg/m3", "PEM", "L15 Santos 2026 Ind Eng Chem Res 65:17315 Supporting Information (Europe PMC PMC13495766, CC-BY 4.0, ie6c00750_si_001.pdf read 2026-10-30)", "Table S3 activated alumina AA (commercial AA, CAPEX parameters)", "table", "none", "INPUT", "bulk density used for the cost model, not the lab bed"),
 # baselines (added after item 14; registry hash for item 14 was logged before)
 ("bep_sec_net", "", "67", "kWh/kg", "PEM", "L12 Crespi 2023 Energy Convers Manag 277:116622 (abstract value from cache; full text not read)", "abstract: partial-load net consumption 67 -> 140 kWh/kg; 67 = best operating point", "text", "none", "INPUT", "B-EP constant net SEC (engineering practice)"),
 ("bcl_recovery_primary", "", "98.4", "%", "PEM", "L14 Ligen 2020 (abstract value from cache; full text NOT FOUND, SN16)", "abstract: dryer recovery 98.4 % for the deoxo + VPSA chain", "text", "none", "INPUT", "B-CL constant purification recovery, primary (D34)"),
 # item 13: L32
 ("l32_m_low", "m_low", "0.5521", "kW_H2/kW_el", "PEM", L32, "Sec 3.1 fitted values after Eq.(1); parameter table Sec 7", "text", "none", "VAL", "VT2b: H2-rate function of the source system itself"),
 ("l32_m_high", "m_high", "0.4698", "kW_H2/kW_el", "PEM", L32, "Sec 3.1", "text", "none", "VAL", ""),
 ("l32_m_T", "m_T", "0.1029", "kW_H2/degC", "PEM", L32, "Sec 3.1", "text", "none", "VAL", ""),
 ("l32_k_E", "k_E", "51.4613", "kW_H2", "PEM", L32, "Sec 3.1", "text", "none", "VAL", "Eq.(1): E_el = dt (m_T T + k_E) + dt m_low (P - P_nom) for 0 <= P <= P_nom (rate in kW_H2)"),
 ("l32_P_nom", "P_nom", "100", "kW", "PEM", L32, "Sec 3.1 breakpoint at the nominal system power (P_nom = 100 kW)", "text", "none", "VAL", ""),
 ("l32_LHV_basis", "", "3", "kWh/Nm3", "PEM", L32, "Sec 3.1 energy embodied in the output hydrogen gas is 3 kWh/Nm3 (LHV)", "text", "none", "VAL", ""),
 ("l32_pressurise_time", "", "roughly 5", "min", "PEM", L32, "Sec 2.1 The electrolyzer takes roughly five minutes to pressurize and start up from an off state", "text", "APPROXIMATE (word roughly)", "VAL", "VT2b target; includes start-up, so an upper bound on pressurisation time"),
 ("l32_storage_pressure", "", "25-40", "bar", "PEM", L32, "Sec 2.1 stored in tanks at a pressure between 25 and 40 bar", "text", "none", "VAL", ""),
 ("l32_rated_system", "", "SILYZER 100, 100 kW, 4 stacks x 25 kW", "-", "PEM", L32, "Sec 2.1, Sec 3.3", "text", "none", "INPUT", "scale 100 kW, not MW"),
 # L52
 ("l52_C_text", "C", "717924", "J/K", "PEM", L52, "Sec 3.4 text (resulting in C = 717 924 J/K)", "text", "none", "UQ_RANGE", "MODEL-DERIVED (L52 material-volume estimate), not measured"),
 ("l52_C_table6", "C", "815445", "J/K", "PEM", L52, "Table 6 Heat capacity", "table", "none", "UQ_RANGE", "MODEL-DERIVED; inconsistent with the Sec 3.4 text value"),
 ("l52_UA", "UA", "67", "W/K", "PEM", L52, "Table 6 (row Factor for the heat transfer coefficient, unit W/K; label read from text extraction)", "table", "none", "UQ_RANGE", "MODEL-DERIVED: fitted to a 46 kW stack cool-down (Fig. 9); L52 states it must be adjusted for the larger stack"),
 ("l52_P_stack", "", "701350", "W", "PEM", L52, "Table 3 / Table 6 Nominal power per stack", "table", "none", "UQ_RANGE", "calculated in L52 Sec 3.2"),
 ("l52_stacks", "", "24", "-", "PEM", L52, "Table 3 Stacks", "table", "none", "UQ_RANGE", ""),
 ("l52_plant_power", "", "17.5", "MW", "PEM", L52, "Table 3 Power (input), Sec 3.1 Silyzer 300", "table", "none", "UQ_RANGE", ""),
 # L53 (AWE candidate)
 ("l53_stack_power", "", "3", "MW", "AWE", L53, "D37 reading: plant description", "text", "none", "AWE_CANDIDATE", "not used in PEM runs"),
 ("l53_cells_pressure", "", "326 cells; 16 bar", "-", "AWE", L53, "D37 reading", "text", "none", "AWE_CANDIDATE", ""),
 ("l53_OinH2_range", "", "0.2-0.32", "%", "AWE", L53, "D37 reading (about 7 months of plant data)", "text", "none", "AWE_CANDIDATE", "AWE deoxo feed range"),
 ("l53_HinO2_range", "", "1.8-3", "%", "AWE", L53, "D37 reading", "text", "none", "AWE_CANDIDATE", ""),
 ("l53_C_t", "", "65.44", "MJ/degC", "AWE", L53, "D37 reading", "text", "MODEL-ESTIMATED", "AWE_CANDIDATE", "not a target"),
]
have = set(df["name"])
new = [dict(zip(cols, r)) for r in rows if r[0] not in have]
df = pd.concat([df, pd.DataFrame(new, columns=cols)], ignore_index=True)
df.to_csv(p, index=False)
print("rows added", len(new), "total", len(df))

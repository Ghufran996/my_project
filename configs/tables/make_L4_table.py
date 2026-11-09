"""Writes code/configs/tables/L4_virahsawmy2024.csv (values transcribed from L4 = Virah-Sawmy, Beck, Sturmberg 2024,
Int. J. Hydrogen Energy 72:49-59, doi 10.1016/j.ijhydene.2024.05.360, CC BY; text read from the ScienceDirect full text
(cached by the analyst) and Fig. 2 from the publisher image gr2_lrg.jpg, sha256 in the file).  Frozen before any VER-2R run."""
import csv, hashlib
from pathlib import Path
out = Path(__file__).parent / "L4_virahsawmy2024.csv"
rows = []
def add(table, item, site, curve, value, unit, loc, ext, ud=""):
    rows.append(dict(table=table, item=item, site=site, curve=curve, value=value, unit=unit, location=loc, extraction=ext, u_d=ud))
FIG = "Fig. 2 legend: 'Efficiency curve k: 0.75-0.xx'"
UD = 0.002   # max deviation of the automated colour trace from Eq.10 (0.0014) + half line width (0.0006), see code/smoke/l4_fig2_trace.py
add("Fig2", "eta_max", "", "all", 0.75, "-", FIG + "; Sec 2 text 'maximum efficiency occurs at 30 % of the capacity'", "text (legend) + colour-trace check", UD)
for k, v in ((1, 0.70), (2, 0.60), (3, 0.55)):
    add("Fig2", "eta_min", "", k, v, "-", FIG, "text (legend) + colour-trace check", UD)
add("Eq10", "P_eta_max", "", "all", 0.3, "x P_cap", "Sec 2.1.3.2 text below Eq.(10)", "text")
add("Eq10", "denominator", "", "all", 0.49, "-", "Eq.(10): eta = -d_eta/0.49 ((P_El/P_cap) - P_eta_max)^2 + eta_max", "equation")
add("Table1", "segment1_lower", "", "all", 0.1, "x P_cap", "Table 1", "table"); add("Table1", "segment1_upper", "", "all", 0.3, "x P_cap", "Table 1", "table")
add("Table1", "segment2_lower", "", "all", 0.3, "x P_cap", "Table 1", "table"); add("Table1", "segment2_upper", "", "all", 1.0, "x P_cap", "Table 1", "table")
add("Sec2", "P_El_min", "", "all", 0.1, "x P_cap", "Eqs.(5)-(6) text", "text"); add("Sec2", "P_standby", "", "all", 0.01, "x P_cap", "Sec 2 / Eq.(4) text", "text")
add("Sec2", "startup_energy", "", "all", 5, "min x P_cap", "Sec 2.1.1 text", "text"); add("Sec2", "LHV_H2", "", "all", 0.033, "MWh/kg", "Sec 2.1.3.3", "text")
add("Sec4.1", "fixed_efficiency", "", "all", 0.75, "-", "Sec 4.1", "text")
add("Sec4.2", "ratio_timestep_study", "", "all", 0.75, "electrolyser/farm capacity", "Sec 4.2 'An electrolyzer to renewable energy ratio of 0.75 was used for all the sites'", "text")
add("Sec3", "data", "", "all", "NEM 5-min 2022", "-", "Sec 3 'Power generation data at 5-min intervals were obtained from the National Electricity Market (NEM) for the year 2022' (no DUIDs printed)", "text")
for s, cap, cf in (("Finley", 162, 19), ("Tailem Bend", 108, 17), ("Hamilton", 57.5, 24)):
    add("Table2", "rated_capacity", s, "", cap, "MW", "Table 2", "table"); add("Table2", "capacity_factor_2022", s, "", cf, "%", "Table 2", "table")
t3 = {"Finley": (-4.4, -13.0, -17.4), "Tailem Bend": (-4.6, -13.5, -18.0), "Hamilton": (-5.1, -14.9, -19.9)}
for s, v in t3.items():
    for k, x in zip((1, 2, 3), v):
        add("Table3", "pct_diff_variable_vs_fixed_ratio1", s, k, x, "%", "Table 3 (ratio 1 of the fixed-efficiency scenario as reference point, Sec 4.1)", "table")
t5 = {"Finley": (30.1, 51.7), "Tailem Bend": (47.9, 53.8), "Hamilton": (35.0, 35.2)}
for s, v in t5.items():
    for k, x in zip((1, 3), v):
        add("Table5", "H2_60min_minus_5min", s, k, x, "t/yr", "Table 5 (ratio 0.75, Sec 4.2)", "table")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(out, len(rows), hashlib.sha256(out.read_bytes()).hexdigest())

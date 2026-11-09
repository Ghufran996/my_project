"""Brief 3 item 1h: D13 check, share of AWE plant-model parameters sourced from FULL TEXT read in this project.
The list is the set of AWE parameter classes the plant engine needs (same structure as the PEM engine); status is taken from the registry
and the literature cache (full text read = L28 only)."""
import sys
from pathlib import Path
import pandas as pd
rows = [
 ("rated H2 flow of the AWE system", "L28", True, "registry awe_rated_H2_flow (full text)"),
 ("maximum operating pressure", "L28", True, "awe_max_pressure"),
 ("production threshold pressure", "L28", True, "awe_production_threshold_pressure"),
 ("delivery threshold pressure", "L28", True, "awe_delivery_threshold_pressure"),
 ("initial inerting duration", "L28", True, "awe_inerting_initial_total"),
 ("pre-start duration", "L28", True, "awe_prestart"),
 ("purge duration (procedural)", "L28", True, "awe_purge_duration"),
 ("pressurisation rate vs power set", "L28", True, "awe_pressurisation_PS*"),
 ("depressurisation duration/rate", "L28", True, "awe_depressurisation_*"),
 ("final inerting duration", "L28", True, "awe_final_inerting_total"),
 ("polarisation curve / cell voltage vs j, T (AWE)", "-", False, "no AWE polarisation source registered"),
 ("Faradaic efficiency (AWE)", "-", False, "none registered"),
 ("stack thermal mass and heat loss (AWE)", "L53 (model-estimated, D37 reading, not re-read); L27 abstract only", False, "no full-text value"),
 ("cold-start heat-up duration (AWE)", "L27 abstract only", False, "not read"),
 ("minimum load P_min (AWE)", "L21 abstract (15.6 %)", False, "abstract only"),
 ("hot-standby power (AWE)", "-", False, "none registered"),
 ("O2-in-H2 feed to the deoxo (AWE)", "L53 via D37 reading", False, "not re-read by implementer"),
 ("deoxo / dryer configuration and recovery (AWE)", "L14 abstract only", False, "full text NOT FOUND (SN16)"),
 ("HTO / crossover vs load (AWE)", "L20/L22 abstracts", False, "abstract only"),
 ("drying-train start-up (AWE)", "-", False, "none"),
]
df = pd.DataFrame(rows, columns=["parameter_class", "source", "sourced_from_full_text", "note"])
share = df.sourced_from_full_text.mean()
df.loc[len(df)] = ["SHARE sourced from full text", "", share, f"{int(df.sourced_from_full_text.sum())} of {len(df)}; D13 threshold 0.80 -> " + ("AWE = second system" if share >= 0.8 else "AWE = SENSITIVITY CASE; claims confined to PEM")]
out = Path(__file__).resolve().parents[1] / "results" / "awe_sourcing_check.csv"
df.to_csv(out, index=False)
print(df.tail(2).to_string(index=False))

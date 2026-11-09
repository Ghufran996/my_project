"""VER-8 (supporting, no gate; Brief 3 item 8): pvlib chain (BSRN irradiance, Perez, PVWatts DC, 14 % derate) vs PVGIS PVGIS-SARAH3 hourly PV output
(requests.csv: same coordinates, optimal angles, c-Si, loss 14 %) for the 17 site-years. The two use DIFFERENT irradiance data (measured BSRN vs satellite), so
the difference is chain + data; it is a plausibility/cross-check, not a verification of either. Output results/ver8_pv_pvgis.csv."""
import io, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su.io import CODE_DIR, RESULTS
from h2su.pv import pv_dc_series
y = yaml.safe_load((CODE_DIR / "configs" / "sites.yaml").read_text())
rows = []
for s, v in y["sites"].items():
    for yr in v["years_selected"]:
        d = pv_dc_series(s, yr)
        ours = (d.p_dc_kw / 1500.0 * 1000.0).resample("1h").mean()                      # W per kWp, hourly mean of the hour starting at HH:00
        g = pd.read_csv(io.StringIO((RESULTS / "data" / "pvgis" / f"{s}_{yr}.csv").read_text(encoding="utf-8")))
        g["time"] = pd.to_datetime(g["time"], utc=True).dt.floor("h")
        g = g.set_index("time")["P"]
        j = pd.concat([ours.rename("ours"), g.rename("pvgis")], axis=1).dropna()
        e_o, e_g = j.ours.sum() / 1000.0, j.pvgis.sum() / 1000.0
        rmse = float(np.sqrt(((j.ours - j.pvgis) ** 2).mean()))
        day = j.pvgis > 0
        rows.append(dict(site=s, year=yr, n_hours=len(j), E_ours_kWh_per_kWp=e_o, E_pvgis_kWh_per_kWp=e_g, annual_diff_pct=(e_o - e_g) / e_g * 100, RMSE_W_per_kWp=rmse,
                         NRMSE_pct_of_mean_daytime_pvgis=rmse / j.pvgis[day].mean() * 100, corr=float(j.ours.corr(j.pvgis))))
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "ver8_pv_pvgis.csv", index=False)
print(df.round(2).to_string(index=False))
print("annual diff % range", df.annual_diff_pct.min().round(2), df.annual_diff_pct.max().round(2), "; NRMSE range", df.NRMSE_pct_of_mean_daytime_pvgis.min().round(1), df.NRMSE_pct_of_mean_daytime_pvgis.max().round(1))

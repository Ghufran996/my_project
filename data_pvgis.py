"""Item 3: PVGIS v5_3 hourly series (B-EP input) for the selected site-years (code/configs/sites.yaml).

    python code/data_pvgis.py

API seriescalc, raddatabase PVGIS-SARAH3, pvcalculation=1 (1 kWp, crystalline Si, free-standing, PVGIS default system loss
14 %), optimalangles=1, components=1, usehorizon=1.  Output results/data/pvgis/<site>_<year>.csv and
results/data/pvgis/requests.csv (exact URL, API version, database actually used, rows, status, time stamp).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su.io import CONFIGS, RESULTS  # noqa: E402

API = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
OUT = RESULTS / "data" / "pvgis"


def main():
    sites = yaml.safe_load(open(CONFIGS / "sites.yaml"))["sites"]
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    for st, v in sites.items():
        for y in v["years_selected"]:
            params = dict(lat=round(v["lat"], 4), lon=round(v["lon"], 4), startyear=y, endyear=y, raddatabase="PVGIS-SARAH3",
                          pvcalculation=1, peakpower=1, pvtechchoice="crystSi", mountingplace="free", loss=14, optimalangles=1,
                          components=1, usehorizon=1, outputformat="json")
            r = requests.get(API, params=params, timeout=180)
            rec = dict(site=st, year=y, url=r.url, api_version="v5_3", http=r.status_code, fetched_utc=pd.Timestamp.now("UTC").isoformat())
            if r.status_code == 200:
                j = r.json()
                h = pd.DataFrame(j["outputs"]["hourly"])
                h["time"] = pd.to_datetime(h["time"], format="%Y%m%d:%H%M", utc=True)
                h.to_csv(OUT / f"{st}_{y}.csv", index=False)
                meta = j["inputs"]
                rec.update(rows=len(h), radiation_db=meta["meteo_data"].get("radiation_db"),
                           slope_deg=meta["mounting_system"]["fixed"]["slope"].get("value"),
                           azimuth_deg=meta["mounting_system"]["fixed"]["azimuth"].get("value"),
                           elevation_m=meta["location"].get("elevation"), system_loss_pct=meta["pv_module"].get("system_loss"),
                           pv_tech=meta["pv_module"].get("technology"))
            else:
                rec.update(rows=0, error=r.text[:200])
            log.append(rec)
            print(rec["site"], y, rec["http"], rec.get("rows"), flush=True)
            time.sleep(0.5)
    df = pd.DataFrame(log)
    df.to_csv(OUT / "requests.csv", index=False)
    ok = df[(df.http == 200)]
    exp = ok.year.map(lambda y: 8784 if y % 4 == 0 else 8760)
    print("rows match expected (8760/8784):", bool((ok.rows == exp).all()), "n requests", len(df), "failed", int((df.http != 200).sum()))


if __name__ == "__main__":
    main()

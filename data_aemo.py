"""Item 4b: AEMO NEM 2022 5-min DISPATCH_UNIT_SCADA for the three C3 solar farms (public MMSDM archive, nemweb.com.au).

    python code/data_aemo.py

DUIDs identified from the MMSDM tables DUDETAILSUMMARY + STATION (2022-12 release): FINLYSF1 (station 'Finely Solar Farm',
NSW1), TBSF1 ('Tailem Bend Solar Project 1', SA1), HAMISF1 ('Hamilton Solar Farm', QLD1); matched to L4 Table 2 by name and
state.  Raw monthly zips (untouched) -> results/data/aemo2022/raw ; per-DUID parquet -> results/data/aemo2022/<DUID>_2022.parquet
(SETTLEMENTDATE = end of the 5-min dispatch interval, AEST, SCADAVALUE in MW).  results/data/aemo2022/sources.csv logs URLs and SHA-256.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su.io import RESULTS  # noqa: E402
from h2su.nemweb import read_mms_zip  # noqa: E402

BASE = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2022/MMSDM_2022_{m:02d}/MMSDM_Historical_Data_SQLLoader/DATA/"
DUIDS = {"FINLYSF1": "Finley", "TBSF1": "Tailem Bend", "HAMISF1": "Hamilton"}
RAW = RESULTS / "data" / "aemo2022" / "raw"


def get(url: Path | str, f: Path) -> dict:
    if not f.exists():
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        f.write_bytes(r.content)
    b = f.read_bytes()
    return dict(url=url, file=f.name, bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    log, frames = [], []
    for m in range(1, 13):
        n = f"PUBLIC_DVD_DISPATCH_UNIT_SCADA_2022{m:02d}010000.zip"
        log.append(get(BASE.format(m=m) + n, RAW / n))
        df = read_mms_zip(RAW / n)
        frames.append(df[df.DUID.isin(DUIDS)][["SETTLEMENTDATE", "DUID", "SCADAVALUE"]])
        print(n, len(df), flush=True)
    for n in ("PUBLIC_DVD_DUDETAIL_202212010000.zip",):
        log.append(get(BASE.format(m=12) + n, RAW / n))
    pd.DataFrame(log).to_csv(RAW.parent / "sources.csv", index=False)
    al = pd.concat(frames)
    al["SETTLEMENTDATE"] = pd.to_datetime(al["SETTLEMENTDATE"].str.strip('"'), format="%Y/%m/%d %H:%M:%S")
    al["SCADAVALUE"] = pd.to_numeric(al["SCADAVALUE"])
    for duid, g in al.groupby("DUID"):
        g = g.sort_values("SETTLEMENTDATE").drop_duplicates("SETTLEMENTDATE")
        g[["SETTLEMENTDATE", "SCADAVALUE"]].to_parquet(RAW.parent / f"{duid}_2022.parquet")
        print(duid, len(g), g.SETTLEMENTDATE.min(), g.SETTLEMENTDATE.max())


if __name__ == "__main__":
    main()

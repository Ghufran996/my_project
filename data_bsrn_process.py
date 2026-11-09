"""Item 2b-c: BSRN year screen (D27) and gap-filled 1-min parquet files for the passing years.

    python code/data_bsrn_process.py

Reads results/data/bsrn_raw/<station>/<station>_<YYYY>-<MM>.tab (written by data_bsrn_download.py), screens every
station-year 2013-2018 (daytime = true zenith < 85 deg; valid = all three components present and passing the L43 limits
and closure/ratio tests), writes results/data/bsrn_screen.csv (QC thresholds with source locations in the '#' header),
results/data/bsrn/<station>_<year>.parquet (UTC 1-min; ghi, dni, dhi, ta_c, fill_flag, valid3, day) for passing years and
code/configs/sites.yaml.  Years with a fraction >= 0.95 pass; all passing years are used (no selection among them).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su import bsrn  # noqa: E402
from h2su.io import CONFIGS, RESULTS  # noqa: E402

STATIONS = ["Cabauw", "Carpentras", "Gobabeb"]
YEARS = range(2013, 2019)
RAW = RESULTS / "data" / "bsrn_raw"
OUT = RESULTS / "data" / "bsrn"


def year_frame(st: str, y: int) -> pd.DataFrame:
    parts = [bsrn.load_month(RAW / st / f"{st}_{y}-{m:02d}.tab") for m in range(1, 13)]
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="first")]
    idx = pd.date_range(pd.Timestamp(y, 1, 1, tz="UTC"), pd.Timestamp(y, 12, 31, 23, 59, tz="UTC"), freq="1min")
    return df.reindex(idx)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sites = [], {}
    for st in STATIONS:
        lat, lon = bsrn.header_latlon(RAW / st / f"{st}_2015-06.tab")
        sites[st] = dict(lat=lat, lon=lon, coordinates_source="PANGAEA dataset header 'Coverage' (BSRN station file)", years_screened=[], years_selected=[])
        for y in YEARS:
            ts = time.time()
            df = year_frame(st, y)
            geo = bsrn.solar_geometry(df.index, lat, lon)
            r = bsrn.screen_year(df, geo)
            r.update(station=st, year=y)
            if r["passed"]:
                filled = bsrn.fill_year(df, geo)
                filled.to_parquet(OUT / f"{st}_{y}.parquet", compression="zstd")
                r["frac_day_minutes_filled"] = float(((filled["fill_flag"] > 0) & filled["day"]).sum() / max(1, filled["day"].sum()))
                sites[st]["years_selected"].append(y)
            else:
                r["frac_day_minutes_filled"] = np.nan
            sites[st]["years_screened"].append(y)
            rows.append(r)
            print(f"{st} {y}: n_day={r['n_day_min']} valid_all3={r['frac_valid_all3']:.4f} GHI={r['frac_valid_GHI']:.4f} "
                  f"gap>10min days={r['n_days_gap_gt10min']} longest={r['longest_gap_min']} pass={r['passed']} ({time.time()-ts:.0f} s)", flush=True)
    df = pd.DataFrame(rows)[["station", "year", "n_day_min", "frac_valid_all3", "frac_valid_GHI", "frac_present_all3",
                              "n_days_gap_gt10min", "longest_gap_min", "frac_day_minutes_filled", "passed"]]
    df = df.rename(columns={"passed": "pass"})
    hdr = ["# BSRN year screen (D27). daytime = true zenith < 85 deg (pvlib nrel_numpy). valid = GHI, DNI, DHI present and pass L43 tests.",
           "# L43 = Long and Shi 2008, Open Atmos. Sci. J. 2:23-37, doi 10.2174/1874282300802010023 (open PDF read 2026-10-30).",
           "# Failure of the (non-definitive) closure or diffuse-ratio test invalidates all three components of that minute (implementation choice)."]
    for k, (v, src) in bsrn.QC.items():
        hdr.append(f"# threshold {k} = {v} | {src}")
    hdr.append(f"# pass rule: frac_valid_all3 >= {bsrn.SCREEN_MIN_FRAC} (PL8/D27)")
    p = RESULTS / "data" / "bsrn_screen.csv"
    p.write_text("\n".join(hdr) + "\n" + df.to_csv(index=False, float_format="%.6f"), encoding="utf-8")
    # extension rule when < 3 passing years at a site is applied by hand if triggered (logged); here only report
    short = [s for s, v in sites.items() if len(v["years_selected"]) < 3]
    with open(CONFIGS / "sites.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(window="2013-2018 (D27)", pass_rule=f"frac_valid_all3 >= {bsrn.SCREEN_MIN_FRAC}", sites=sites,
                            sites_below_3_years=short), f, sort_keys=False)
    size = sum(f.stat().st_size for f in OUT.glob("*.parquet")) / 2**20
    msg = f"wall {time.time()-t0:.0f} s; parquet {size:.0f} MiB; sites below 3 passing years: {short}"
    (RESULTS / "data" / "bsrn_process_log.txt").write_text(msg + "\n")
    print(msg)


if __name__ == "__main__":
    main()

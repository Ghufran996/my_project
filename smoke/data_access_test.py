"""Item 5 data-access probe: PANGAEA BSRN listing summary, one-month format test per station, PVGIS hourly, MIDC reachability.
No bulk download: one monthly file per station."""
import json, re, io, collections, sys, os, time
import requests, numpy as np, pandas as pd, pvlib

ROOT = os.environ["H2SU_ROOT"]
OUT = os.path.join(ROOT, "results", "smoke", "data_test")
os.makedirs(OUT, exist_ok=True)
rows = json.load(open(os.path.join(ROOT, "results", "smoke", "pangaea_listing.json")))
by = collections.defaultdict(dict)
for r in rows:
    m = re.search(r"\((\d{4})-(\d{2})\)", r["title"])
    if m and "reference list" not in r["title"]:
        by[r["station"]].setdefault((int(m.group(1)), int(m.group(2))), []).append(r)
TEST_MONTH = (2019, 7)
coords = {}
summary = []
for st, d in by.items():
    yrs = collections.Counter(y for (y, m) in d)
    full = sorted(y for y, c in yrs.items() if c == 12)
    first = d[min(d)][0]
    coords[st] = (first["lat"], first["lon"])
    # longest run of consecutive complete years
    best, cur = [], []
    for y in full:
        cur = cur + [y] if cur and y == cur[-1] + 1 else [y]
        if len(cur) > len(best): best = cur
    # month pick
    key = TEST_MONTH if TEST_MONTH in d else sorted(d)[len(d) // 2]
    rec = d[key][0]
    uri = rec["uri"]; doi = uri.replace("https://doi.org/", "")
    fn = os.path.join(OUT, f"{st}_{key[0]}-{key[1]:02d}.tab")
    t0 = time.time()
    resp = requests.get(f"https://doi.pangaea.de/{doi}?format=textfile", timeout=180)
    status = resp.status_code
    nbytes = len(resp.content)
    open(fn, "wb").write(resp.content)
    txt = resp.content.decode("utf-8", errors="replace")
    head, _, body = txt.partition("*/\n")
    df = pd.read_csv(io.StringIO(body), sep="\t")
    cols = list(df.columns)
    tcol = cols[0]
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    df = df.set_index(tcol)
    pick = lambda p: next((c for c in cols if c.startswith(p) and "std" not in c and "min" not in c and "max" not in c), None)
    ghi, dni, dif = pick("SWD"), pick("DIR"), pick("DIF")
    present = [c for c in (ghi, dni, dif) if c]
    lat, lon = coords[st]
    n_month_min = pd.Timestamp(key[0], key[1], 1).days_in_month * 1440
    idx = pd.date_range(pd.Timestamp(key[0], key[1], 1, tz="UTC"), periods=n_month_min, freq="1min")
    df = df[~df.index.duplicated()].reindex(idx)
    sp = pvlib.solarposition.get_solarposition(idx, lat, lon)
    day = (sp["apparent_zenith"] < 85).values
    ok3 = df[present].notna().all(axis=1).values if len(present) == 3 else np.zeros(len(df), bool)
    okany = df[present].notna().all(axis=1).values
    lim = {ghi: (-10, 1600), dni: (-10, 1400), dif: (-10, 1000)}
    rng = np.ones(len(df), bool)
    for c in present: rng &= ((df[c] > lim[c][0]) & (df[c] < lim[c][1])).values
    rec_out = dict(station=st, lat=lat, lon=lon, test_month=f"{key[0]}-{key[1]:02d}", doi=doi, http=status, bytes=nbytes,
                   rows_in_file=len(body.strip().splitlines()) - 1, timestep_s=int(pd.Series(df.index).diff().median().total_seconds()),
                   components_present=",".join(c.split(" [")[0] for c in present), frac_valid_present_daytime=round(float(okany[day].mean()), 4),
                   frac_valid_all3_allminutes=round(float(ok3.mean()), 4), frac_valid_all3_daytime=round(float(ok3[day].mean()), 4),
                   frac_valid_all3_inrange_daytime=round(float((ok3 & rng)[day].mean()), 4),
                   n_months_listed=len(d), listing_first=f"{min(d)[0]}-{min(d)[1]:02d}", listing_last=f"{max(d)[0]}-{max(d)[1]:02d}",
                   complete_years_by_listing=";".join(map(str, full)), longest_consecutive_complete_run=f"{best[0]}-{best[-1]}" if best else "",
                   n_longest_run=len(best), login_option_values=";".join(sorted({str(x[0]['login']) for x in d.values()})),
                   license="BSRN-1.0 (https://bsrn.awi.de/data/conditions-of-data-release/)")
    summary.append(rec_out)
    print(st, rec_out["test_month"], status, nbytes, rec_out["frac_valid_all3_daytime"], round(time.time() - t0, 1), "s", flush=True)
pd.DataFrame(summary).to_csv(os.path.join(ROOT, "results", "data_access.csv"), index=False)

# PVGIS hourly seriescalc, one year, same coordinates (radiation only; pvcalculation=0 and also PV calc)
pv_rows = []
for st, (lat, lon) in coords.items():
    for ver in ["v5_3"]:
        try:
            r = requests.get(f"https://re.jrc.ec.europa.eu/api/{ver}/seriescalc",
                             params=dict(lat=lat, lon=lon, startyear=2019, endyear=2019, pvcalculation=0, components=1, outputformat="json"), timeout=120)
            ok = r.status_code
            if ok == 200:
                j = r.json()
                n = len(j["outputs"]["hourly"])
                db = j["inputs"].get("meteo_data", {}).get("radiation_db")
                pd.DataFrame(j["outputs"]["hourly"]).to_csv(os.path.join(OUT, f"pvgis_{st}_2019_{ver}.csv"), index=False)
                pv_rows.append(dict(station=st, api=ver, http=ok, n_hourly=n, radiation_db=db, elevation=j["inputs"]["location"].get("elevation")))
            else:
                pv_rows.append(dict(station=st, api=ver, http=ok, error=r.text[:200]))
        except Exception as e:
            pv_rows.append(dict(station=st, api=ver, error=str(e)[:200]))
        time.sleep(0.5)
pd.DataFrame(pv_rows).to_csv(os.path.join(OUT, "pvgis_test.csv"), index=False)
print(pd.DataFrame(pv_rows).to_string())

# MIDC reachability
mid = []
for u in ["https://midcdmz.nrel.gov/apps/sitehome.pl?site=BMS", "https://midcdmz.nrel.gov/", "https://midcdmz.nlr.gov/", "https://doi.org/10.5439/1052221"]:
    try:
        r = requests.get(u, timeout=30, allow_redirects=True)
        mid.append(dict(url=u, http=r.status_code, final=r.url, bytes=len(r.content)))
    except Exception as e:
        mid.append(dict(url=u, error=type(e).__name__ + ": " + str(e)[:150]))
pd.DataFrame(mid).to_csv(os.path.join(OUT, "midc_test.csv"), index=False)
print(pd.DataFrame(mid).to_string())

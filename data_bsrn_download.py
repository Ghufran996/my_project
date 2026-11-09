"""Item 2a: download PANGAEA BSRN monthly 1-min files (untouched) for the D27 sites/years.

    python code/data_bsrn_download.py [--years 2013-2018]

Output: results/data/bsrn_raw/<station>/<station>_<YYYY>-<MM>.tab (as served, never edited) and
results/data/bsrn_citations.csv (PANGAEA DOI, title, bytes, sha256 per month).  Licence BSRN-1.0
(https://bsrn.awi.de/data/conditions-of-data-release/); the dataset citation of every file is inside the file header.
"""
from __future__ import annotations

import argparse, hashlib, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su.io import RESULTS  # noqa: E402

STATIONS = ["Cabauw", "Carpentras", "Gobabeb"]
RAW = RESULTS / "data" / "bsrn_raw"


def listing():
    rows = json.load(open(RESULTS / "smoke" / "pangaea_listing.json"))
    out = {}
    for r in rows:
        m = re.search(r"\((\d{4})-(\d{2})\)", r["title"])
        if m and r["station"] in STATIONS and "reference list" not in r["title"]:
            out.setdefault((r["station"], int(m.group(1)), int(m.group(2))), []).append(r)
    return out


def fetch(job):
    st, y, mo, rec = job
    doi = rec["uri"].replace("https://doi.org/", "")
    f = RAW / st / f"{st}_{y}-{mo:02d}.tab"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists() and f.stat().st_size > 10_000:
        b = f.read_bytes()
        return dict(station=st, year=y, month=mo, doi=doi, title=rec["title"], login=rec["login"], bytes=len(b),
                    sha256=hashlib.sha256(b).hexdigest(), http=200, cached=True)
    for attempt in range(3):
        try:
            r = requests.get(f"https://doi.pangaea.de/{doi}?format=textfile", timeout=180)
            if r.status_code == 200 and len(r.content) > 10_000:
                f.write_bytes(r.content)
                return dict(station=st, year=y, month=mo, doi=doi, title=rec["title"], login=rec["login"], bytes=len(r.content),
                            sha256=hashlib.sha256(r.content).hexdigest(), http=200, cached=False)
            err = r.status_code
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__
        time.sleep(3)
    return dict(station=st, year=y, month=mo, doi=doi, title=rec["title"], login=rec["login"], bytes=0, sha256="", http=err, cached=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2013-2018")
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-"))
    lst = listing()
    jobs, missing = [], []
    for st in STATIONS:
        for y in range(y0, y1 + 1):
            for mo in range(1, 13):
                recs = lst.get((st, y, mo))
                if not recs:
                    missing.append((st, y, mo))
                    continue
                recs = sorted(recs, key=lambda r: (r["login"] is not None and r["login"] != 1))  # prefer login option 1 (no login)
                jobs.append((st, y, mo, recs[0]))
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(fetch, jobs))
    df = pd.DataFrame(res)
    out = RESULTS / "data" / "bsrn_citations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} months requested, {(df.http == 200).sum()} ok, missing in listing: {missing}, wall {time.time()-t0:.0f} s, "
          f"disk {df.bytes.sum()/2**20:.0f} MiB")
    print(df[df.http != 200])


if __name__ == "__main__":
    main()

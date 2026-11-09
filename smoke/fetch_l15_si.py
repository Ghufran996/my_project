"""Brief 3 item 1b: download L15 SI from Europe PMC (PMC13495766), CC-BY 4.0. Run in background."""
import sys, time, requests, pathlib
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
url = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13495766/supplementaryFiles"
t0 = time.time()
with requests.get(url, stream=True, timeout=(30, 1200)) as r:
    print("HTTP", r.status_code, r.headers.get("content-type"), r.headers.get("content-length"), flush=True)
    r.raise_for_status()
    n = 0
    with open(out / "supp.zip", "wb") as f:
        for ch in r.iter_content(1 << 16):
            f.write(ch); n += len(ch)
print("bytes", n, "wall_s", round(time.time() - t0, 1), flush=True)

"""Compute and cache the PV DC power series for all selected site-years (Brief 3 item 5); prints a summary."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
from h2su.pv import pv_dc_series, site_info
import yaml
from h2su.io import CODE_DIR
y = yaml.safe_load((CODE_DIR / "configs" / "sites.yaml").read_text())
for site in y["sites"]:
    si = site_info(site)
    for yr in si["years"]:
        t0 = time.time()
        d = pv_dc_series(site, yr)
        e = d.p_dc_kw.sum() / 60.0
        print(f"{site} {yr} tilt {si['tilt']} az {si['az_pvgis']} NaN {int(d.p_dc_kw.isna().sum())} E_dc {e/1500:.0f} kWh/kWp CF {e/1500/(len(d)/60)*100:.1f}% Pmax {d.p_dc_kw.max():.0f} kW fill {int((d.fill>0).sum())} wall {time.time()-t0:.1f}s", flush=True)

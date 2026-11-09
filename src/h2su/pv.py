"""PV chain (Brief 3 item 5): BSRN 1-min GHI/DNI/DHI -> plane-of-array (Perez 1990 via pvlib) -> cell temperature (NOCT relation with the
PVWatts V5 INOCT, L56) -> DC power (PVWatts V5 Eq.8, gamma and 14 % derate from L56) -> DC power available to the electrolyser.
No converter loss (SN19, unsourced) and no IAM (simplification, stated). Tilt/azimuth: per-site median of the PVGIS optimal angles
(results/data/pvgis/requests.csv); PVGIS azimuth 0 = south -> pvlib 180 + az. Filled minutes (D27) carry fill_flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from .io import CODE_DIR, RESULTS, param
from .shared import block_mean  # noqa: F401


def block_mean_hold(x: np.ndarray, block: int) -> np.ndarray:
    """Block means, held piecewise-constant on the 1-min grid (PL8); the last partial block is averaged separately."""
    n = x.size
    out = np.empty_like(x, dtype=float)
    for s in range(0, n, block):
        out[s:s + block] = x[s:s + block].mean()
    return out


def site_info(site: str) -> dict:
    y = yaml.safe_load((CODE_DIR / "configs" / "sites.yaml").read_text())
    s = y["sites"][site]
    rq = pd.read_csv(RESULTS / "data" / "pvgis" / "requests.csv")
    rq = rq[rq.site == site]
    return dict(lat=s["lat"], lon=s["lon"], years=list(s["years_selected"]), tilt=float(rq.slope_deg.median()), az_pvgis=float(rq.azimuth_deg.median()),
                elevation=float(rq.elevation_m.median()))


def pvwatts_iam(aoi_deg, n_glass: float = 1.526):
    """PVWatts V5 module-cover angular response, standard module (L56 Sec. 8, Eq. 3-6, single slab, absorption neglected as printed), normalised to normal incidence."""
    th1 = np.radians(np.clip(np.asarray(aoi_deg, float), 1e-6, 89.999))
    th2 = np.arcsin(np.sin(th1) / n_glass)
    tau = 1.0 - 0.5 * (np.sin(th2 - th1) ** 2 / np.sin(th2 + th1) ** 2 + np.tan(th2 - th1) ** 2 / np.tan(th2 + th1) ** 2)
    tau0 = 1.0 - ((n_glass - 1.0) / (n_glass + 1.0)) ** 2
    return np.where(np.asarray(aoi_deg, float) >= 90.0, 0.0, np.clip(tau / tau0, 0.0, 1.0))


def pv_dc_series(site: str, year: int, pdc0_kw: float = 1500.0, cache: bool = True, iam: bool = False, dcdc: float | None = None, tag: str = "") -> pd.DataFrame:
    import pvlib
    f = RESULTS / "data" / "pv_dc" / f"{site}_{year}{tag}.parquet"
    if cache and f.exists():
        return pd.read_parquet(f)
    si = site_info(site)
    d = pd.read_parquet(RESULTS / "data" / "bsrn" / f"{site}_{year}.parquet")
    t = d.index
    loc = pvlib.location.Location(si["lat"], si["lon"], altitude=si["elevation"])
    sp = loc.get_solarposition(t, temperature=d.ta_c.fillna(10.0).to_numpy())
    zen = sp["apparent_zenith"].clip(upper=180.0)
    ghi = d.ghi.fillna(0.0).clip(lower=0.0)
    dni = d.dni.fillna(0.0).clip(lower=0.0)
    dhi = d.dhi.fillna(0.0).clip(lower=0.0)
    am = pvlib.atmosphere.get_relative_airmass(zen).fillna(40.0)
    dni_x = pvlib.irradiance.get_extra_radiation(t)
    poa = pvlib.irradiance.get_total_irradiance(si["tilt"], 180.0 + si["az_pvgis"], zen, sp["azimuth"], dni, ghi, dhi, dni_extra=dni_x, airmass=am,
                                                albedo=param("pv_albedo"), model="perez", model_perez="allsitescomposite1990")
    g = poa["poa_global"].fillna(0.0).clip(lower=0.0)
    if iam:                                                      # PVWatts V5: the angle-of-incidence correction is applied to the direct beam only
        aoi = pvlib.irradiance.aoi(si["tilt"], 180.0 + si["az_pvgis"], zen, sp["azimuth"])
        f_iam = pd.Series(pvwatts_iam(aoi.to_numpy()), index=t)
        g = (poa["poa_global"] - poa["poa_direct"].fillna(0.0) + poa["poa_direct"].fillna(0.0) * f_iam).fillna(0.0).clip(lower=0.0)
    ta = d.ta_c.ffill().bfill().fillna(10.0)
    tc = ta + (param("pv_INOCT") - 20.0) * g / 800.0
    pdc = pdc0_kw * g / 1000.0 * (1.0 + param("pv_gamma_pdc") * (tc - 25.0)) * (1.0 - param("pv_system_loss")) * (param("pv_dcdc_efficiency") if dcdc is None else dcdc)
    out = pd.DataFrame({"p_dc_kw": pdc.clip(lower=0.0).astype("float64"), "poa": g.astype("float64"), "zenith": zen.astype("float64"), "ta_c": ta.astype("float64"),
                        "fill": d.fill_flag.astype("int8")}, index=t)
    f.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f)
    return out

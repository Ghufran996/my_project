"""BSRN 1-min radiation: quality control (QCRad, L43), year-validity screen and gap filling (D27).

QC thresholds (L43 = Long and Shi 2008, Open Atmos. Sci. J. 2:23-37, doi 10.2174/1874282300802010023; text read from the
open PDF, journal pages 23-25; S0 = 1368 W/m2, S_a = S0/AU^2, mu0 = cos(SZA), mu0 = 0 for SZA > 90 deg):
  physically possible (BSRN limits, p. 24-25):  GHI  -4 <= x <= S_a*1.5*mu0^1.2 + 100
                                                DIF  -4 <= x <= S_a*0.95*mu0^1.2 + 50
                                                DNI  -4 <= x <= S_a
  closure ratio (p. 25, 'Ratio of Global SW over Sum SW', Sum = DIF + DNI*mu0, tested if Sum > 50 W/m2):
        |GHI/Sum - 1| <= 0.08 for SZA < 75 deg ;  <= 0.15 for 75 <= SZA < 93 deg ; not tested otherwise
  diffuse ratio (p. 25, tested if GHI > 50 W/m2): DIF/GHI < 1.05 for SZA < 75 deg ; < 1.10 for 75 <= SZA < 93 deg
Closure and ratio tests are non-definitive in L43 (the failing instrument is unknown); here a failure invalidates all three
components of that minute (D27 implementation choice, stated in the screen header).
Gap rule (D27): GHI from closure (GHI = DIF + DNI*mu0) where only GHI is invalid; DNI/DHI from the Erbs decomposition
(Erbs, Klein, Duffie 1982, doi 10.1016/0038-092X(82)90302-4, pvlib implementation, Holmgren et al. 2018,
doi 10.21105/joss.00884) where only the components are invalid; remaining daytime minutes by linear interpolation (in time)
of the clear-sky index GHI/GHI_Haurwitz (Haurwitz 1945, doi 10.1175/1520-0469(1945)002<0154:IIRTCA>2.0.CO;2, pvlib
implementation), components then from Erbs.  Fill flags: 0 measured, 1 closure, 2 Erbs components, 3 interpolated.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from .io import read_bsrn_tab

S0 = 1368.0                 # W/m2, L43 p. 23 (definitions)
DAY_ZENITH_DEG = 85.0       # D27 daytime definition
SCREEN_MIN_FRAC = 0.95      # D27 / PL8
QC = {
    "S0_Wm2": (S0, "L43 p.23 Definitions: 'QCRad uses a value of 1368 Wm-2'"),
    "ghi_min": (-4.0, "L43 p.24 Physically possible limits per BSRN, Global SWdn Min"),
    "ghi_max_a": (1.5, "L43 p.25 Global SWdn Max: S_a x 1.5 x mu0^1.2 + 100"),
    "ghi_max_b": (100.0, "L43 p.25 Global SWdn Max: S_a x 1.5 x mu0^1.2 + 100"),
    "dif_min": (-4.0, "L43 p.25 Diffuse SW Min"),
    "dif_max_a": (0.95, "L43 p.25 Diffuse SW Max: S_a x 0.95 x mu0^1.2 + 50"),
    "dif_max_b": (50.0, "L43 p.25 Diffuse SW Max: S_a x 0.95 x mu0^1.2 + 50"),
    "dni_min": (-4.0, "L43 p.25 Direct Normal Min"),
    "dni_max": ("S_a", "L43 p.25 Direct Normal Max: S_a"),
    "closure_tol_lowSZA": (0.08, "L43 p.25 Ratio of Global SW over Sum SW: +/-8 % for SZA < 75 deg, Sum > 50 W/m2"),
    "closure_tol_highSZA": (0.15, "L43 p.25: +/-15 % for 93 deg > SZA > 75 deg, Sum > 50 W/m2"),
    "closure_sum_min": (50.0, "L43 p.25: for Sum SW < 50 W/m2 test not possible"),
    "difratio_max_lowSZA": (1.05, "L43 p.25 Diffuse Ratio: (Dif SW)/(SWdn) < 1.05 for SZA < 75 deg, SWdn > 50 W/m2"),
    "difratio_max_highSZA": (1.10, "L43 p.25: < 1.10 for 93 deg > SZA > 75 deg, SWdn > 50 W/m2"),
    "sza_split_deg": (75.0, "L43 p.25"),
    "sza_max_test_deg": (93.0, "L43 p.25"),
    "exponent": (1.2, "L43 p.25 mu0^1.2"),
}


def header_latlon(path: Path) -> tuple[float, float]:
    txt = Path(path).read_text(encoding="utf-8", errors="replace")[:6000]
    m = re.search(r"Coverage:\s*LATITUDE:\s*([-\d.]+)\s*\*\s*LONGITUDE:\s*([-\d.]+)", txt)
    if not m:
        raise ValueError(f"no coordinates in header of {path}")
    return float(m.group(1)), float(m.group(2))


def _pick(cols, prefix):
    return next(c for c in cols if c.startswith(prefix) and "std" not in c and "min" not in c and "max" not in c)


def load_month(path: Path) -> pd.DataFrame:
    df = read_bsrn_tab(path)
    df = df[~df.index.duplicated(keep="first")]
    out = pd.DataFrame({"ghi": df[_pick(df.columns, "SWD")], "dni": df[_pick(df.columns, "DIR")], "dhi": df[_pick(df.columns, "DIF")]})
    tcol = next((c for c in df.columns if c.startswith("T2")), None)
    out["ta_c"] = df[tcol] if tcol else np.nan
    return out.apply(pd.to_numeric, errors="coerce")


def solar_geometry(idx: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    sp = pvlib.solarposition.get_solarposition(idx, lat, lon, method="nrel_numpy")
    au = pvlib.solarposition.nrel_earthsun_distance(idx)
    out = pd.DataFrame({"zenith": sp["zenith"].to_numpy(), "apparent_zenith": sp["apparent_zenith"].to_numpy()}, index=idx)
    out["au"] = np.asarray(au, float)
    return out


def qc_flags(ghi, dni, dhi, zenith_deg, au) -> dict[str, np.ndarray]:
    """Boolean arrays (True = passes / not testable). NaN inputs fail the presence flags."""
    ghi, dni, dhi, z, au = (np.asarray(a, float) for a in (ghi, dni, dhi, zenith_deg, au))
    mu0 = np.where(z > 90.0, 0.0, np.cos(np.radians(z)))
    sa = S0 / au**2
    e = QC["exponent"][0]
    with np.errstate(invalid="ignore"):
        p_ghi = (ghi >= QC["ghi_min"][0]) & (ghi <= sa * QC["ghi_max_a"][0] * mu0**e + QC["ghi_max_b"][0])
        p_dhi = (dhi >= QC["dif_min"][0]) & (dhi <= sa * QC["dif_max_a"][0] * mu0**e + QC["dif_max_b"][0])
        p_dni = (dni >= QC["dni_min"][0]) & (dni <= sa)
        sm = dhi + dni * mu0
        lo, hi = z < QC["sza_split_deg"][0], (z >= QC["sza_split_deg"][0]) & (z < QC["sza_max_test_deg"][0])
        testable = np.isfinite(ghi) & np.isfinite(sm) & (sm > QC["closure_sum_min"][0]) & (lo | hi)
        tol = np.where(lo, QC["closure_tol_lowSZA"][0], QC["closure_tol_highSZA"][0])
        closure_ok = ~testable | (np.abs(ghi / np.where(sm == 0, np.nan, sm) - 1.0) <= tol)
        dtest = np.isfinite(ghi) & np.isfinite(dhi) & (ghi > QC["closure_sum_min"][0]) & (lo | hi)
        dlim = np.where(lo, QC["difratio_max_lowSZA"][0], QC["difratio_max_highSZA"][0])
        dratio_ok = ~dtest | (dhi / np.where(ghi == 0, np.nan, ghi) < dlim)
    pres_ghi, pres_dni, pres_dhi = np.isfinite(ghi), np.isfinite(dni), np.isfinite(dhi)
    ghi_ok = pres_ghi & p_ghi
    comp_ok = pres_dni & pres_dhi & p_dni & p_dhi
    valid3 = ghi_ok & comp_ok & closure_ok & dratio_ok
    return dict(mu0=mu0, sa=sa, ghi_ok=ghi_ok, comp_ok=comp_ok, closure_ok=closure_ok & dratio_ok, valid3=valid3,
                pres_all=pres_ghi & pres_dni & pres_dhi)


def screen_year(df: pd.DataFrame, geo: pd.DataFrame) -> dict:
    """D27 year-validity metrics over daytime minutes (true zenith < 85 deg)."""
    f = qc_flags(df["ghi"], df["dni"], df["dhi"], geo["zenith"], geo["au"])
    day = (geo["zenith"].to_numpy() < DAY_ZENITH_DEG)
    n = int(day.sum())
    bad = pd.Series(day & ~f["valid3"], index=df.index)
    run_len = bad.groupby((~bad).cumsum()).cumsum()          # length of the current run of invalid daytime minutes
    gaps_by_day = run_len.groupby(df.index.date).max()
    frac_all3 = float(f["valid3"][day].mean()) if n else np.nan
    return dict(n_day_min=n, frac_valid_all3=frac_all3, frac_valid_GHI=float((f["ghi_ok"] & np.isfinite(df["ghi"].to_numpy()))[day].mean()),
                frac_present_all3=float(f["pres_all"][day].mean()), n_days_gap_gt10min=int((gaps_by_day > 10).sum()),
                longest_gap_min=int(run_len.max()), passed=bool(frac_all3 >= SCREEN_MIN_FRAC))


def fill_year(df: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    """D27 gap rule; returns ghi, dni, dhi (W/m2, simulation input), ta_c, fill_flag, valid3, day."""
    f = qc_flags(df["ghi"], df["dni"], df["dhi"], geo["zenith"], geo["au"])
    z = geo["zenith"].to_numpy()
    mu0 = f["mu0"]
    day = z < DAY_ZENITH_DEG
    ghi, dni, dhi = df["ghi"].to_numpy(float).copy(), df["dni"].to_numpy(float).copy(), df["dhi"].to_numpy(float).copy()
    flag = np.zeros(len(df), np.int8)
    A = f["valid3"]
    B = day & ~f["ghi_ok"] & f["comp_ok"]
    C = day & f["ghi_ok"] & ~f["comp_ok"]
    D = day & ~A & ~B & ~C
    ghi[B] = dhi[B] + dni[B] * mu0[B]
    flag[B] = 1
    doy = df.index.dayofyear
    clear = pvlib.clearsky.haurwitz(geo["apparent_zenith"])["ghi"].to_numpy()
    anchor = day & (A | B | C) & (clear > 0)
    kc = pd.Series(np.where(anchor, ghi / np.where(clear > 0, clear, np.nan), np.nan), index=df.index)
    kc_i = kc.where(day & (clear > 0)).interpolate(method="time", limit_direction="both")
    ghi[D] = (kc_i.to_numpy() * clear)[D]
    flag[D] = 3
    need_erbs = C | D
    if need_erbs.any():
        er = pvlib.irradiance.erbs(pd.Series(ghi, index=df.index)[need_erbs], pd.Series(z, index=df.index)[need_erbs], doy[need_erbs])
        dni[need_erbs] = np.asarray(er["dni"], float)
        dhi[need_erbs] = np.asarray(er["dhi"], float)
    flag[C] = 2
    out = pd.DataFrame({"ghi": ghi, "dni": dni, "dhi": dhi, "ta_c": df["ta_c"].to_numpy(), "fill_flag": flag,
                        "valid3": A & day, "day": day}, index=df.index)
    night = ~day
    out.loc[night, ["ghi", "dni", "dhi"]] = 0.0
    for c in ("ghi", "dni", "dhi"):
        out[c] = out[c].clip(lower=0.0).fillna(0.0)
    return out

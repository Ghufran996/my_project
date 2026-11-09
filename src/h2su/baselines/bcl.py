"""B-CL (classical): Faraday-type fixed LHV efficiency (75 %, L4 fixed case) and constant purification recovery (98.4 %, L14 abstract; 98.98 % L15 as
sensitivity), no start-up losses; P_min cut-off 0.1 P_rated; 1-min or hourly (block means of the same series). B-EP (engineering practice): PVGIS hourly
power x 1.5 MWp/1 kWp, constant net SEC 67 kWh/kg (L12), P_min cut-off."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io import RESULTS, param
from ..shared import block_mean


def bcl(p_dc_kw: np.ndarray, block: int = 1, p_rated_kw: float = 1000.0, eta: float | None = None, recovery: float | None = None):
    eta = param("c3_fixed_efficiency") / 100.0 if eta is None else eta
    rec = param("bcl_recovery_primary") / 100.0 if recovery is None else recovery
    lhv = param("c3_LHV") * 1000.0                                   # kWh/kg
    x = np.minimum(np.asarray(p_dc_kw, float), p_rated_kw)
    if block > 1:
        x = block_mean(x, block)
    on = x >= param("pmin_pem_low") * p_rated_kw
    m = np.where(on, eta * x / lhv * block / 60.0, 0.0)             # kg per block
    return dict(M_stack=float(m.sum()), M_del=float((m * rec).sum()), E=float(np.where(on, x, 0.0).sum() * block / 60.0))


def bep(site: str, year: int, pdc0_kw: float = 1500.0, p_rated_kw: float = 1000.0):
    import io
    txt = (RESULTS / "data" / "pvgis" / f"{site}_{year}.csv").read_text(encoding="utf-8")
    df = pd.read_csv(io.StringIO(txt))
    p = np.minimum(df["P"].to_numpy(float) / 1000.0 * pdc0_kw, p_rated_kw)      # PVGIS P [W] for 1 kWp -> kW at 1.5 MWp (14 % loss as in requests.csv)
    on = p >= param("pmin_pem_low") * p_rated_kw
    E = float(np.where(on, p, 0.0).sum())
    return dict(M_del=E / param("bep_sec_net"), E=E)

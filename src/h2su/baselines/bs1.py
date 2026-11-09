"""B-S1 (SOTA C1, Renau 2025 class): steady load-dependent purification loss s(L) = phi + b/L (VT5a fit of L1 Table 2) on the stack H2 of an instantly switching
plant (no start-up mechanism). Independent vectorised implementation (numpy), used as the reference of SC-B item (iii)."""
from __future__ import annotations

import numpy as np

from .. import plantrun as pr


def bs1_annual(p_dc_kw: np.ndarray, p_rated_kw: float = pr.P_RATED_KW, pmin_frac: float | None = None, T_C: float = pr.T_OP_C, dt_min: float = 1.0):
    from ..io import param
    st = pr.stack_tables()
    row = st["mtab"][list(pr.TG).index(T_C)]
    phi, b = pr.psa_fit()
    pmin = (param("pmin_pem_low") if pmin_frac is None else pmin_frac) * p_rated_kw
    pa = np.minimum(p_dc_kw, p_rated_kw)
    on = pa >= pmin
    l = pa / p_rated_kw
    ms = np.where(on, np.interp(l, pr.LG, row) * (p_rated_kw / 1000.0) / 60.0 * dt_min, 0.0)
    s = phi + b / np.maximum(l, 0.10)
    return dict(M_stack=float(ms.sum()), M_del=float((ms * (1.0 - s)).sum()), E_stack=float((np.where(on, pa, 0.0)).sum() * dt_min / 60.0))

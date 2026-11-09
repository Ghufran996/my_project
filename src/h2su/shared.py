"""Code shared by the C3 re-implementation (baselines/bs2_c3.py) and the plant engine (plant.py): VER-2L link (D31, Brief 3 item 7).

block_mean: one block-mean function for every resolution change (5 -> 60 min in VER-2R, 1 -> 5..60 min in the plant study).
account_c3: one power -> H2 / energy accounting function for a given on/standby/off state path and the L4 two-segment chord curve.
"""
from __future__ import annotations

import numpy as np
from numba import njit


def block_mean(x, k: int):
    """Mean over consecutive blocks of k samples (trailing partial block dropped, as in L4 resolution handling)."""
    x = np.asarray(x, float)
    n = len(x) // k * k
    return x[:n].reshape(-1, k).mean(axis=1)


@njit(cache=True)
def _c3_arrays(pre, path, p_cap, a1, b1, a2, b2):
    n = len(pre)
    mdot = np.zeros(n)
    p_el = np.zeros(n)
    n_su = 0
    prev = 0
    for t in range(n):
        s = path[t]
        if s == 2:
            p = min(pre[t], p_cap)
            p_el[t] = p
            mdot[t] = a1 * p + b1 if p <= 0.3 * p_cap else a2 * p + b2
            if prev == 0:
                n_su += 1
        prev = s
    return mdot, p_el, n_su


def account_c3(pre, path, dt_h, p_cap, p_sb, a1, b1, a2, b2):
    """H2 [kg], electrolyser energy [MWh], standby energy [MWh], n_startups, hours_on for a state path (0 off, 1 standby, 2 on) and chord segments
    mdot[kg/h] = a P[MW] + b on P <= 0.3 P_cap (segment 1) and above (segment 2). Per-step values come from a numba loop; the sums use numpy (pairwise),
    so the results equal the pre-refactor numpy accounting bit for bit."""
    path = np.asarray(path)
    mdot, p_el, n_su = _c3_arrays(np.asarray(pre, float), path, p_cap, a1, b1, a2, b2)
    on = path == 2
    return float(np.sum(mdot * dt_h)), float(np.sum(p_el * dt_h)), float(np.sum(path == 1) * p_sb * p_cap * dt_h), int(n_su), float(on.sum() * dt_h)

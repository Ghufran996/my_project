"""Restart state machine STUB (timing probe only; no physics, no parameter values from the literature).

States: 0 OFF, 1 STANDBY, 2 START, 3 PRODUCTION, 4 SHUTDOWN. A restart is counted at every START entry.
Debounce, start duration, standby hold and shutdown duration are dimensionless step counts supplied by the caller.
"""
from __future__ import annotations

import numpy as np
from numba import njit

OFF, STANDBY, START, PRODUCTION, SHUTDOWN = 0, 1, 2, 3, 4


@njit(cache=True)
def run_stub(p_kw: np.ndarray, p_min_kw: float, t_deb: int, n_start: int, n_standby_max: int, n_shutdown: int):
    n = p_kw.size
    state = OFF
    above = 0
    below = 0
    clock = 0
    n_restarts = 0
    steps_in = np.zeros(5, dtype=np.int64)
    state_trace = np.empty(n, dtype=np.int8)
    for k in range(n):
        p = p_kw[k]
        if p >= p_min_kw:
            above += 1
            below = 0
        else:
            below += 1
            above = 0
        if state == OFF or state == STANDBY:
            if above >= t_deb:
                state = START
                clock = 0
                n_restarts += 1
            elif state == STANDBY:
                clock += 1
                if clock >= n_standby_max:
                    state = OFF
        elif state == START:
            clock += 1
            if below >= t_deb:
                state = SHUTDOWN
                clock = 0
            elif clock >= n_start:
                state = PRODUCTION
        elif state == PRODUCTION:
            if below >= t_deb:
                state = SHUTDOWN
                clock = 0
        elif state == SHUTDOWN:
            clock += 1
            if clock >= n_shutdown:
                state = STANDBY
                clock = 0
        steps_in[state] += 1
        state_trace[k] = state
    return n_restarts, steps_in, state_trace

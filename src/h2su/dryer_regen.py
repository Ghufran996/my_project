"""D53 non-trace regeneration form for the L15 dryer (Brief 5 item 5).

Isobaric, isothermal, H2 sorption neglected, water solid-film LDF with the frozen k_LDF, Toth-Aranovich-Donohue isotherm at P_w = y P (L15 eq 2-4), upwind finite volumes.
Total molar flux N(z, t) of the gas follows from the total mole balance at constant gas concentration c = P/(R T):
    dN/dz = - rho_b dq/dt,   N(0, t) = purge flux N0 (dry H2, y_in = 0)
Water balance (conservative form):
    eps c dy/dt + d(N y)/dz = - rho_b dq/dt          <=>   eps c dy/dt = d[N (1 - y)]/dz-type form below
    dq/dt = k_LDF (q*(y P) - q)
Discrete (cell i, face fluxes Nf[0] = N0, Nf[i+1] = Nf[i] - rho_b h dq_i/dt, y_{-1} = y_in):
    eps c h dy_i/dt = Nf[i+1] (1 - y_i) - Nf[i] (1 - y_{i-1})        (uses Nf[i+1] - Nf[i] = - rho_b h dq_i/dt)
so y <= 1 is preserved and the H2 flux Nf (1 - y) is constant at steady state. For y -> 0 the form reduces to the constant-flux trace-species model of dryer.py.
Purge direction: the profile is solved with the purge entering at x = 0. For a purge in the feed direction of the later adsorption step the q profile is used as is;
for a countercurrent purge the q profile is reversed (x -> L - x) before the adsorption step.
"""
from __future__ import annotations

import numpy as np
from scipy import constants as sc
from scipy.integrate import solve_ivp

from . import dryer as dr

R = sc.gas_constant
PPM = 1e-6


def regen_solve(p: dict, iso: dict, N: int, t: np.ndarray, rtol: float = 1e-6, y0: float | None = None, q0: float | None = None, y_in: float = 0.0):
    """p: L_m, eps, rho_bed_kg_m3, p_Pa (regeneration pressure), T_K, N0_mol_m2_s (purge molar flux), kLDF_1_s. Returns dict(t, y (N, nt), q (N, nt), Nout, ...)."""
    T, P = p["T_K"], p["p_Pa"]
    pbar = P / 1e5
    c = P / (R * T)
    eps, rb, k, L = p["eps"], p["rho_bed_kg_m3"], p["kLDF_1_s"], p["L_m"]
    h = L / N
    N0 = p["N0_mol_m2_s"]
    y0 = p["y_init"] if y0 is None else y0
    q0 = p["q_init_mol_kg"] if q0 is None else q0

    def qstar(y):
        return dr.q_water(dr.y_pos(y / PPM * 1.0) * PPM * pbar, T, iso)       # smooth positive part of y (umol/mol units inside y_pos)

    def fluxes(y, q):
        dq = k * (qstar(y) - q)
        Nf = np.empty(N + 1)
        Nf[0] = N0
        Nf[1:] = N0 - rb * h * np.cumsum(dq)
        return dq, Nf

    def rhs(_t, Y):
        y, q = Y[:N], Y[N:2 * N]
        dq, Nf = fluxes(y, q)
        yprev = np.concatenate([[y_in], y[:-1]])
        dy = (Nf[1:] * (1.0 - y) - Nf[:-1] * (1.0 - yprev)) / (eps * c * h)
        return np.concatenate([dy, dq, [Nf[N] * y[-1], Nf[N]]])

    def dqstar_dy(y):
        return dr.dq_water_dp(dr.y_pos(y / PPM) * PPM * pbar, T, iso) * pbar * dr.dy_pos(y / PPM)

    rbh = rb * h
    tri = np.tril(np.ones((N, N)))                       # [i, j] = 1 for j <= i
    tri_lo = np.tril(np.ones((N, N)), -1)                # j <= i - 1

    def jac(_t, Y):
        y, q = Y[:N], Y[N:2 * N]
        dq, Nf = fluxes(y, q)
        a = k * dqstar_dy(y)                             # d dq_j / d y_j
        yprev = np.concatenate([[y_in], y[:-1]])
        w_i = 1.0 - y                                    # (1 - y_i)
        w_p = 1.0 - yprev                                # (1 - y_{i-1})
        # coefficient c_ij = w_i [j <= i] - w_p [j <= i-1]
        C = w_i[:, None] * tri - w_p[:, None] * tri_lo
        den = eps * c * h
        Jyy = (-rbh * a[None, :] * C) / den
        Jyq = (rbh * k * C) / den
        idx = np.arange(N)
        Jyy[idx, idx] += -Nf[1:] / den
        Jyy[idx[1:], idx[:-1]] += Nf[1:N] / den
        n = 2 * N + 2
        J = np.zeros((n, n))
        J[:N, :N] = Jyy
        J[:N, N:2 * N] = Jyq
        J[N + idx, idx] = a
        J[N + idx, N + idx] = -k
        J[2 * N + 1, :N] = -rbh * a
        J[2 * N + 1, N:2 * N] = rbh * k
        J[2 * N, :N] = -rbh * a * y[-1]
        J[2 * N, N - 1] += Nf[N]
        J[2 * N, N:2 * N] = rbh * k * y[-1]
        return J

    Y0 = np.concatenate([np.full(N, y0), np.full(N, q0), [0.0, 0.0]])
    atol = np.concatenate([np.full(N, 1e-9), np.full(N, 1e-5), [1e-12, 1e-12]])
    sol = solve_ivp(rhs, (t[0], t[-1]), Y0, t_eval=t, method="BDF", rtol=rtol, atol=atol, jac=jac)
    if not sol.success:
        raise RuntimeError(f"regeneration integration failed: {sol.message}")
    y, q = sol.y[:N], sol.y[N:2 * N]
    Wout, Tout = sol.y[2 * N], sol.y[2 * N + 1]
    # balances (per unit bed cross-section): water in gas + solid + water out = initial (inlet water = y_in N0 t)
    gas_w = eps * c * h * (y.sum(axis=0) - N * y0)
    sol_w = rb * h * (q.sum(axis=0) - N * q0)
    w_in = y_in * N0 * (sol.t - sol.t[0])
    res_w = gas_w + sol_w + Wout - w_in
    scale = max(abs(rb * h * N * q0), 1e-300)
    # total-mole balance: gas moles constant (isobaric) -> Tout - N0 t = - rb h sum dq (solid release) ; check with the solid change
    res_tot = (Tout - N0 * (sol.t - sol.t[0])) + sol_w
    Nf_end = np.array([fluxes(y[:, j], q[:, j])[1][-1] for j in range(y.shape[1])])
    return dict(t=sol.t, y=y, q=q, y_out=y[-1], N_out=Nf_end, water_balance_rel=np.abs(res_w) / scale, total_balance_rel=np.abs(res_tot) / scale, W_out=Wout, T_out=Tout, h=h, N=N)

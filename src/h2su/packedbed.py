"""VER-6: 1-D packed-bed method-of-lines (finite volume) reference for the lumped deoxo / dryer start-up models.

Every field f obeys (same equation as the COMSOL Coefficient Form PDE, see COMSOL_MATLAB/build_bed_model.m)
        A_f dF/dt + V_f dF/dx - d/dx( D_f dF/dx ) = S_f(F)
with Danckwerts inlet  V_f F - D_f dF/dx = V_f F_in  and zero gradient at the outlet.  Finite volumes, cell averages, MUSCL
(minmod) or first-order upwind convection, central diffusion; the cumulative inlet flux, outlet flux and source are carried as
extra ODE states so that mass and energy balances can be checked exactly (they are linear invariants of the discrete system).
Stiff BDF/Radau integration with a banded Jacobian sparsity pattern (scipy.integrate.solve_ivp).
Constant molar density and superficial velocity (trace-level reaction, isobaric): see packedbed_models.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix


@dataclass
class Field:
    name: str
    A: float                 # accumulation coefficient
    V: float                 # convective coefficient (velocity-like)
    D: float                 # diffusion coefficient
    F_in: float              # inlet value (Danckwerts)
    F0: float                # uniform initial value
    scale: float = 1.0       # magnitude used for the absolute tolerance


@dataclass
class Bed:
    L: float
    fields: list[Field]
    source: Callable[[np.ndarray], np.ndarray]    # F (nf, N) -> S (nf, N), source per unit bed volume
    scheme: str = "muscl"                         # 'muscl' | 'upwind'
    meta: dict = field(default_factory=dict)


def _minmod(a, b):
    return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def _face_values(c: np.ndarray, scheme: str) -> np.ndarray:
    """Upwind-biased value at the outlet face of every cell (flow in +x). c: (N,) -> (N,)"""
    if scheme == "upwind":
        return c
    if scheme == "linup2":                      # second-order linear upwind-biased (Fromm-type, no limiter, smooth)
        f = c.copy()
        f[1:] = c[1:] + 0.5 * (c[1:] - c[:-1])
        return f
    s = np.zeros_like(c)
    s[1:-1] = _minmod(c[1:-1] - c[:-2], c[2:] - c[1:-1])
    s[-1] = c[-1] - c[-2]
    return c + 0.5 * s


def rhs_factory(bed: Bed, N: int):
    nf = len(bed.fields)
    h = bed.L / N
    A = np.array([f.A for f in bed.fields])
    V = np.array([f.V for f in bed.fields])
    D = np.array([f.D for f in bed.fields])
    Fin = np.array([f.F_in for f in bed.fields])

    def rhs(t, y):
        F = y[:nf * N].reshape(nf, N)
        dF = np.empty_like(F)
        ext = np.empty((nf, 3))
        S = bed.source(F)
        for k in range(nf):
            c = F[k]
            fl = np.empty(N + 1)
            fl[0] = V[k] * Fin[k]
            cf = _face_values(c, bed.scheme)
            fl[1:] = V[k] * cf
            if D[k] != 0.0 and N > 1:
                fl[1:N] -= D[k] * (c[1:] - c[:-1]) / h
            dF[k] = (fl[:-1] - fl[1:]) / h + S[k]
            dF[k] /= A[k]
            ext[k] = (fl[0], fl[-1], h * S[k].sum())
        return np.concatenate([dF.ravel(), ext.ravel()])
    return rhs


def jac_sparsity(nf: int, N: int):
    n = nf * N + 3 * nf
    J = lil_matrix((n, n))
    for k in range(nf):
        for i in range(N):
            r = k * N + i
            for kk in range(nf):                    # source couples fields within a cell
                J[r, kk * N + i] = 1
            for j in range(max(0, i - 2), min(N, i + 3)):
                J[r, k * N + j] = 1
    for k in range(nf):
        for e in range(3):
            r = nf * N + 3 * k + e
            for kk in range(nf):
                for i in range(N):
                    J[r, kk * N + i] = 1
    return J.tocsr()


def solve(bed: Bed, N: int, t_eval: np.ndarray, rtol: float = 1e-6, method: str = "BDF", atol_factor: float = 1e-3):
    nf = len(bed.fields)
    y0 = np.concatenate([np.repeat([f.F0 for f in bed.fields], N), np.zeros(3 * nf)])
    atol = np.concatenate([np.repeat([rtol * atol_factor * f.scale for f in bed.fields], N),
                           np.full(3 * nf, rtol * atol_factor)])
    sol = solve_ivp(rhs_factory(bed, N), (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method=method, rtol=rtol, atol=atol,
                    jac_sparsity=jac_sparsity(nf, N))
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    F = sol.y[:nf * N].reshape(nf, N, -1)
    ext = sol.y[nf * N:].reshape(nf, 3, -1)
    h = bed.L / N
    out = dict(t=sol.t, F=F, outlet=None, I_in=ext[:, 0], I_out=ext[:, 1], I_src=ext[:, 2], nfev=sol.nfev, h=h, N=N)
    # outlet value = reconstructed face value of the last cell
    out["outlet"] = np.array([[_face_values(F[k, :, j], bed.scheme)[-1] for j in range(F.shape[2])] for k in range(nf)])
    inv = np.array([bed.fields[k].A * h * F[k].sum(axis=0) for k in range(nf)])
    out["inventory"] = inv
    # balance: d inventory = in - out + source (source sign as in the equation), relative to the largest term
    bal = inv - inv[:, :1] - (out["I_in"] - out["I_out"] + out["I_src"])
    den = np.maximum.reduce([np.abs(inv - inv[:, :1]), np.abs(out["I_in"]), np.abs(out["I_out"]), np.abs(out["I_src"]), np.full_like(inv, 1e-300)])
    out["balance_err"] = np.abs(bal) / den
    return out

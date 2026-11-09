"""L15 dryer model (D46): isothermal 1-D packed bed, water Toth-Aranovich-Donohue isotherm with a solid-film LDF, H2 at local equilibrium.

Equations (identical in COMSOL_MATLAB/model_dryer_startup_regen.m):
    eps dy/dt + u_s dy/dx - d/dx(eps D_ax dy/dx) = - rho_bed k_LDF (q*(y) - q) / (1e-6 c_T)      y: umol H2O / mol gas
    dq/dt = k_LDF (q*(y) - q)                                                                    q: mol H2O / kg adsorbent
    q*(P_w) = [q_m b2 P_w / (1 + (b2 P_w)^f)^(1/f)] [1 / (1 - b3 P_w)^n],   P_w = 1e-6 y p  [bar]      (L15 eq 2)
    b_i = b_i0 exp[Q_i/(R T0) (T0/T - 1)] (eq 3),  f = f0 + alpha (1 - T0/T) (eq 4)
Constant total molar flux N, c_T = p/(R T), u_s = N/c_T (trace-species form of the D46 model; y << 1 required, checked per run).
Axial dispersion: D_ax = 0 (the Wakao-Funazkri correlation could not be sourced from a tool result; see log item 1).
"""
from __future__ import annotations

import numpy as np
from scipy import constants as sc

from . import packedbed as pb
from .constants import M_H2
from .io import load_registry

R = sc.gas_constant
PPM = 1e-6
DELTA_Y = 1e-4          # umol/mol: smoothing of max(y, 0) in the isotherm argument (2e-5 of the 5 umol/mol spec); removes the kink that stalls BDF
P_NORM = 101325.0
T_NORM = 273.15


def reg_value(df, name):
    return float(df.loc[df["name"] == name, "value_or_range"].iloc[0])


def iso_params(df=None):
    """Isotherm constants in SI-compatible units (b in 1/bar, Q in J/mol)."""
    df = load_registry() if df is None else df
    g = lambda n: reg_value(df, n)
    return dict(qm=g("l15_isoW_qm"), f0=g("l15_isoW_f0"), alpha=g("l15_isoW_alpha"), b20=g("l15_isoW_b20"), Q2=g("l15_isoW_Q2") * 1e3,
                b30=g("l15_isoW_b30"), Q3=g("l15_isoW_Q3") * 1e3, n=g("l15_isoW_n"), T0=g("l15_T0"),
                lqm=g("l15_isoH2_qm"), lb10=g("l15_isoH2_b10"), lQ1=g("l15_isoH2_Q1") * 1e3,
                mA=g("l15_magnus_A"), mB=g("l15_magnus_B"), mC=g("l15_magnus_C"))


def psat_bar(T_K, iso):
    """L15 eq 7 (Magnus-Tetens, hPa) -> bar."""
    return iso["mA"] * np.exp(iso["mB"] * (T_K - 273.15) / (T_K - iso["mC"])) * 1e-3


def toth_T(T_K, iso):
    T0 = iso["T0"]
    e = T0 / T_K - 1.0
    b2 = iso["b20"] * np.exp(iso["Q2"] / (R * T0) * e)
    b3 = iso["b30"] * np.exp(iso["Q3"] / (R * T0) * e)
    f = iso["f0"] + iso["alpha"] * (1.0 - T0 / T_K)
    return b2, b3, f


def q_water(pw_bar, T_K, iso):
    b2, b3, f = toth_T(T_K, iso)
    pw = np.maximum(pw_bar, 0.0)
    return iso["qm"] * b2 * pw / (1.0 + (b2 * pw) ** f) ** (1.0 / f) / (1.0 - b3 * pw) ** iso["n"]


def dq_water_dp(pw_bar, T_K, iso):
    """d q*/d P_w (mol/kg/bar), analytic; finite at P_w = 0."""
    b2, b3, f = toth_T(T_K, iso)
    pw = np.maximum(pw_bar, 0.0)
    x = b2 * pw
    g = (1.0 + x ** f) ** (-1.0 / f) * (1.0 - b3 * pw) ** (-iso["n"])
    q = iso["qm"] * x * g
    return iso["qm"] * b2 * (1.0 + x ** f) ** (-1.0 / f - 1.0) * (1.0 - b3 * pw) ** (-iso["n"]) + q * iso["n"] * b3 / (1.0 - b3 * pw)


def q_h2(p_bar, T_K, iso):
    b1 = iso["lb10"] * np.exp(iso["lQ1"] / (R * iso["T0"]) * (iso["T0"] / T_K - 1.0))
    return iso["lqm"] * b1 * p_bar / (1.0 + b1 * p_bar)


def lab_geometry(df):
    L = reg_value(df, "l15si_bed_height") / 100.0
    Di = reg_value(df, "l15si_bed_inner_diameter") / 100.0
    return L, np.pi / 4 * Di ** 2


def design_volume(df):
    D, H = reg_value(df, "l15_design_vessel_Di"), reg_value(df, "l15_design_vessel_Hv")
    return np.pi / 4 * D ** 2 * H


def plant_geometry(df, m_rated_kg_h):
    """L15 scale-up rule: bed volume proportional to the feed H2 mass flow, H/D = 10 (SI Table S3 design)."""
    s = m_rated_kg_h / reg_value(df, "l15_design_feed_mass_flow")
    V = design_volume(df) * s
    D = (4 * V / (np.pi * 10.0)) ** (1 / 3)
    return 10.0 * D, np.pi / 4 * D ** 2, V


def y_pos(y):
    """Smooth positive part of y (umol/mol): 0.5 (y + sqrt(y^2 + DELTA_Y^2)); equals y for y >> DELTA_Y."""
    return 0.5 * (y + np.sqrt(y * y + DELTA_Y * DELTA_Y))


def dy_pos(y):
    return 0.5 * (1.0 + y / np.sqrt(y * y + DELTA_Y * DELTA_Y))


def make_bed(p: dict, iso: dict, N: int | None = None, scheme: str = "upwind") -> pb.Bed:
    """Bed for parameter dict p (names as in params_comsol.csv scope rows)."""
    T = p["T_K"]
    cT = p["p_Pa"] / (R * T)
    us = p["N_flux_mol_m2_s"] / cT
    b2, b3, f = toth_T(T, iso)
    k, rb, eps = p["kLDF_1_s"], p["rho_bed_kg_m3"], p["eps"]
    pbar = p["p_Pa"] / 1e5

    def qstar(y):
        return q_water(y_pos(y) * PPM * pbar, T, iso)

    def source(F):
        y, q = F[0], F[1]
        dq = k * (qstar(y) - q)
        return np.vstack([-rb * dq / (PPM * cT), dq])

    qin = float(qstar(p["yW_in_umol_mol"]))
    fy = pb.Field("y", A=eps, V=us, D=eps * p["Dax_m2_s"], F_in=p["yW_in_umol_mol"], F0=p["yW_init_umol_mol"], scale=max(p["yW_in_umol_mol"], 1.0))
    fq = pb.Field("q", A=1.0, V=0.0, D=0.0, F_in=0.0, F0=p["q_init_mol_kg"], scale=max(qin, 1e-3))
    return pb.Bed(L=p["L_m"], fields=[fy, fq], source=source, scheme=scheme, meta=dict(us=us, cT=cT))


def jac_factory(p: dict, iso: dict, N: int, scheme: str = "upwind"):
    """Analytic sparse Jacobian (first-order upwind structure; exact for scheme='upwind', quasi-Newton matrix for MUSCL)."""
    from scipy.sparse import coo_matrix
    T = p["T_K"]
    cT = p["p_Pa"] / (R * T)
    us = p["N_flux_mol_m2_s"] / cT
    k, rb, eps = p["kLDF_1_s"], p["rho_bed_kg_m3"], p["eps"]
    pbar = p["p_Pa"] / 1e5
    h = p["L_m"] / N
    nf = 2
    idx = np.arange(N)

    def jac(t, Y):
        y = Y[:N]
        dqs = dq_water_dp(y_pos(y) * PPM * pbar, T, iso) * PPM * pbar * dy_pos(y)          # d q*/d y
        r, c, v = [], [], []

        def add(rr, cc, vv):
            rr = np.atleast_1d(rr)
            r.append(rr); c.append(np.atleast_1d(cc)); v.append(np.broadcast_to(vv, rr.shape).astype(float))
        src_y = -rb * k * dqs / (PPM * cT) / eps
        if scheme == "upwind":
            add(idx, idx, -us / h / eps + src_y)
            add(idx[1:], idx[1:] - 1, us / h / eps)
        else:                                  # linup2: face i = 1.5 c_i - 0.5 c_{i-1} (i >= 1), face 0 = c_0, inlet face = F_in
            dg = np.full(N, -1.5 * us / h / eps)
            dg[0] = -us / h / eps
            add(idx, idx, dg + src_y)
            sub = np.full(N - 1, 2.0 * us / h / eps)
            sub[0] = 1.5 * us / h / eps
            add(idx[1:], idx[:-1], sub)
            if N > 2:
                add(idx[2:], idx[:-2], np.full(N - 2, -0.5 * us / h / eps))
        add(idx, N + idx, rb * k / (PPM * cT) / eps)
        add(N + idx, idx, k * dqs)
        add(N + idx, N + idx, -k)
        e0 = nf * N
        if scheme == "upwind" or N < 2:
            add([e0 + 1], [N - 1], us)
        else:
            add([e0 + 1, e0 + 1], [N - 1, N - 2], np.array([1.5 * us, -0.5 * us]))
        add(np.full(N, e0 + 2), idx, h * (-rb * k * dqs / (PPM * cT)))
        add(np.full(N, e0 + 2), N + idx, h * rb * k / (PPM * cT))
        add(np.full(N, e0 + 5), idx, h * k * dqs)
        add(np.full(N, e0 + 5), N + idx, -h * k)
        n = nf * N + 6
        return coo_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(n, n)).tocsc()
    return jac


def solve_bed(bed: pb.Bed, N: int, t: np.ndarray, rtol=1e-6, y0_profiles=None, jac=None):
    """Like packedbed.solve but with optional non-uniform initial profiles (nf, N) and an analytic Jacobian."""
    from scipy.integrate import solve_ivp
    nf = len(bed.fields)
    y0 = np.concatenate([np.repeat([f.F0 for f in bed.fields], N), np.zeros(3 * nf)]) if y0_profiles is None else \
        np.concatenate([np.asarray(y0_profiles, float).reshape(-1), np.zeros(3 * nf)])
    # absolute tolerances: 1e-3 umol/mol (2e-4 of the 5 umol/mol spec) and 1e-5 mol/kg for rtol <= 1e-6, scaled up with rtol above 1e-6.
    # Smaller values make the BDF integrator stall on the non-smooth Toth leading edge (y -> 0): tested, 1e-4 umol/mol needs 65 s vs 0.8 s
    # for the same t_b (relative difference 2e-6).
    sc_ = max(1.0, rtol / 1e-6)
    atol = np.concatenate([np.full(N, 1e-3 * sc_), np.full(N, 1e-5 * sc_), np.full(3 * nf, 1e-9)])
    kw = dict(jac=jac) if jac is not None else dict(jac_sparsity=pb.jac_sparsity(nf, N))
    sol = solve_ivp(pb.rhs_factory(bed, N), (t[0], t[-1]), y0, t_eval=t, method="BDF", rtol=rtol, atol=atol, **kw)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    F = sol.y[:nf * N].reshape(nf, N, -1)
    ext = sol.y[nf * N:].reshape(nf, 3, -1)
    h = bed.L / N
    outlet = np.array([[pb._face_values(F[k, :, j], bed.scheme)[-1] for j in range(F.shape[2])] for k in range(nf)])
    inv = np.array([bed.fields[k].A * h * F[k].sum(axis=0) for k in range(nf)])
    bal = inv - inv[:, :1] - (ext[:, 0] - ext[:, 1] + ext[:, 2])
    den = np.maximum.reduce([np.abs(inv - inv[:, :1]), np.abs(ext[:, 0]), np.abs(ext[:, 1]), np.abs(ext[:, 2]), np.full_like(inv, 1e-300)])
    return dict(t=sol.t, F=F, outlet=outlet, I_in=ext[:, 0], I_out=ext[:, 1], I_src=ext[:, 2], inventory=inv, balance_err=np.abs(bal) / den, h=h, N=N)


def first_crossing(t, y, thr):
    """First time y rises above thr (linear interpolation); nan if it never does; t[0] if y[0] > thr."""
    idx = np.where(y > thr)[0]
    if len(idx) == 0:
        return float("nan")
    k = idx[0]
    if k == 0:
        return float(t[0])
    return float(t[k - 1] + (thr - y[k - 1]) * (t[k] - t[k - 1]) / (y[k] - y[k - 1]))


def run_case(p: dict, iso: dict, N: int = 400, rtol: float = 1e-6, scheme: str = "upwind", y0_profiles=None):
    """Run one dryer case; returns series (outlet) and bookkeeping."""
    t = np.arange(0.0, p["t_end_s"] + 1e-9, p["dt_out_s"])
    bed = make_bed(p, iso, N, scheme)
    sol = solve_bed(bed, N, t, rtol, y0_profiles, jac_factory(p, iso, N, bed.scheme))
    y_out = np.maximum(sol["outlet"][0], 0.0)
    q_prof = sol["F"][1, :, -1]
    y_prof = sol["F"][0, :, -1]
    return dict(t=t, y_out=y_out, y_prof=y_prof, q_prof=q_prof, sol=sol, us=bed.meta["us"], cT=bed.meta["cT"])


def t_b(p: dict, iso: dict, N=400, rtol=1e-6, thr=5.0, scheme="upwind"):
    r = run_case(p, iso, N, rtol, scheme)
    return first_crossing(r["t"], r["y_out"], thr), r


def lab_case(df, iso, row, kLDF, t_end_s=None, dt_out_s=10.0):
    """Lab breakthrough case from a Table S4 row (dict with P_bar, RH_pct, T_C, F_feed_LN_min)."""
    L, A = lab_geometry(df)
    T = row["T_C"] + 273.15
    P = row["P_bar"] * 1e5
    y_in = row["RH_pct"] / 100.0 * psat_bar(T, iso) / row["P_bar"] / PPM
    Vm_N = R * T_NORM / P_NORM
    n_dot = row["F_feed_LN_min"] * 1e-3 / 60.0 / Vm_N
    return dict(L_m=L, A_m2=A, eps=reg_value(df, "l15_bed_porosity"), rho_bed_kg_m3=reg_value(df, "l15_bed_packing_density"), Dax_m2_s=0.0,
                N_flux_mol_m2_s=n_dot / A, p_Pa=P, T_K=T, yW_in_umol_mol=y_in, yW_init_umol_mol=0.0, q_init_mol_kg=0.0, kLDF_1_s=kLDF,
                t_end_s=t_end_s if t_end_s else 8.0 * 3600.0, dt_out_s=dt_out_s)

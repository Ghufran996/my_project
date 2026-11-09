"""VER-6 driver: cases, event definitions (identical to COMSOL_MATLAB/compute_events.m), code verification 6a, grids 6d.

Deoxo (6b), dryer (6c) and lumped-vs-1-D (6e) cases are built only from sourced registry parameters (ver6_cases.py); while
SN17/SN18 are open they are reported BLOCKED and never run on assumed values.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import packedbed as pb
from .io import RESULTS
from .ver6_cases import Case, build

OUT = RESULTS / "ver6"
MESH_LEVELS = (25, 50, 100, 200, 400)
RTOL_BASE = 1e-6
RTOL_VARIANTS = (1e-4, 1e-8)


def levels():
    lv = [(n, RTOL_BASE, f"N{n:03d}") for n in MESH_LEVELS]
    lv += [(MESH_LEVELS[-1], r, f"N{MESH_LEVELS[-1]:03d}_r{r:.0e}") for r in RTOL_VARIANTS]
    return lv


# ------------------------------------------------------------------ events (mirror of compute_events.m)
def last_crossing_time(t, r):
    idx = np.where(r > 1)[0]
    if len(idx) == 0:
        return float(t[0])
    k = idx[-1]
    if k == len(t) - 1:
        return float("nan")
    return float(t[k] + (1 - r[k]) * (t[k + 1] - t[k]) / (r[k + 1] - r[k]))


def trapz_to(t, f, te):
    if np.isnan(te) or te <= t[0]:
        return 0.0
    k = np.where(t <= te)[0][-1]
    s = np.trapezoid(f[:k + 1], t[:k + 1])
    if k < len(t) - 1 and te > t[k]:
        fe = f[k] + (f[k + 1] - f[k]) * (te - t[k]) / (t[k + 1] - t[k])
        s += (f[k] + fe) / 2 * (te - t[k])
    return float(s)


def compute_events(t, s: dict, cfg: dict) -> dict:
    ev = dict(m_H2_consumed_kg=cfg.get("m_consumed_kg", np.nan), m_H2_vented_kg=np.nan, m_H2_purge_kg=np.nan,
              m_H2_repress_kg=cfg.get("repress_kg", np.nan), t_spec_s=np.nan, t_break_s=np.nan,
              mass_balance_err=cfg.get("mass_err", np.nan), energy_balance_err=cfg.get("energy_err", np.nan))
    cols = cfg.get("spec_cols", [])
    if cols:
        r = np.zeros_like(t)
        for c, lim in zip(cols, cfg["spec_lim"]):
            r = np.maximum(r, s[c] / lim)
        ev["t_spec_s"] = last_crossing_time(t, r)
        if not np.isnan(ev["t_spec_s"]):
            ev["m_H2_vented_kg"] = trapz_to(t, s["F_out_mol_s"] * s["y_H2_out"] * cfg["M_H2"], ev["t_spec_s"])
    if cfg.get("purge_all", False):
        ev["m_H2_purge_kg"] = float(np.trapezoid(s["F_out_mol_s"] * s["y_H2_out"] * cfg["M_H2"], t))
    if cfg.get("break_col"):
        ratio = s[cfg["break_col"]] / cfg["y_in"]
        if ratio[-1] >= cfg["break_frac"]:
            ev["t_break_s"] = float(np.trapezoid(1 - ratio, t))
    return ev


# ------------------------------------------------------------------ verification-case model
def build_verif(p: dict, N: int):
    eps, us, k1, K = p["eps"], p["us_m_s"], p["k1_s"], p["K_ads"]
    v = us / eps
    dax = p["inv_Pe"] * v * p["L_m"]
    f = pb.Field("ct", A=eps + (1 - eps) * K, V=us, D=eps * dax, F_in=p["c_in_norm"], F0=0.0, scale=p["c_in_norm"])
    return pb.Bed(L=p["L_m"], fields=[f], source=lambda F: -k1 * F, meta=dict(v=v, dax=dax))


def verif_analytic(case_id: str, p: dict) -> dict:
    da = p["k1_s"] * p["L_m"] / p["us_m_s"]
    v = p["us_m_s"] / p["eps"]
    if case_id == "ver_plugflow":
        return dict(name="conversion X = 1 - exp(-Da)", value=1 - np.exp(-da), basis="absolute")
    if case_id.startswith("ver_disp"):
        pe = 1.0 / p["inv_Pe"]
        a = np.sqrt(1 + 4 * da / pe)
        r = 4 * a * np.exp(pe / 2) / ((1 + a) ** 2 * np.exp(a * pe / 2) - (1 - a) ** 2 * np.exp(-a * pe / 2))
        return dict(name=f"Danckwerts outlet ratio, Pe_L = {pe:g}", value=r, basis="relative")
    if case_id == "ver_breakthrough":
        return dict(name="first moment (L/v)(1 + K(1-eps)/eps)", value=(p["L_m"] / v) * (1 + p["K_ads"] * (1 - p["eps"]) / p["eps"]), basis="relative")
    raise KeyError(case_id)


def run_verif(case: Case, p: dict, g: dict, N: int, rtol: float, grid_id: str, write: bool = True):
    bed = build_verif(p, N)
    t = np.arange(0.0, p["t_end_s"] + 1e-9, p["dt_out_s"])
    t0 = time.time()
    sol = pb.solve(bed, N, t, rtol=rtol)
    wall = time.time() - t0
    cin = p["c_in_norm"]
    n = len(t)
    nan = np.full(n, np.nan)
    series = dict(T_out_K=nan, y_H2_out=nan, y_O2_out=sol["outlet"][0] / cin, y_H2O_out=nan, y_N2_out=nan, F_out_mol_s=nan, p_out_Pa=nan)
    cfg = dict(spec_cols=[], spec_lim=[], break_col="y_O2_out" if case.case_id == "ver_breakthrough" else "", y_in=1.0,
               break_frac=g["event_break_frac"], purge_all=False, M_H2=g["M_H2_kg_mol"], mass_err=float(sol["balance_err"][0, -1]))
    ev = compute_events(t, series, cfg)
    row = dict(case_id=case.case_id, model="python_mol", grid_id=grid_id, rtol=rtol, **ev, wall_s=wall)
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(dict(case_id=case.case_id, model="python_mol", grid_id=grid_id, rtol=rtol, t_s=t, **series))
        df.to_csv(OUT / f"{case.case_id}__python_mol__{grid_id}.csv", index=False)
    return row, series, sol


def params_for(case: Case, rows: list[dict]):
    g = {r["name"]: float(r["value"]) for r in rows if r["scope"] == "global"}
    p = {r["name"]: float(r["value"]) for r in rows if r["scope"] == case.case_id}
    return p, g


def gci(f_fine, f_mid, f_coarse, r=2.0, fs=1.25):
    """Celik et al. 2008 GCI (constant refinement ratio r). Returns (p_obs, gci_fine_pct, monotonic)."""
    e21, e32 = f_mid - f_fine, f_coarse - f_mid
    if e21 == 0 or e32 == 0 or e32 / e21 <= 0:
        return float("nan"), float("nan"), False
    p = np.log(abs(e32 / e21)) / np.log(r)
    return float(p), float(fs * abs(e21 / f_fine) / (r ** p - 1) * 100), True


def run_all_verif():
    cases, rows = build()
    ev_rows, checks, conv = [], [], []
    for case in [c for c in cases if c.status == "READY"]:
        p, g = params_for(case, rows)
        ana = verif_analytic(case.case_id, p)
        scal = {}
        for n, rt, gid in levels():
            row, series, sol = run_verif(case, p, g, n, rt, gid)
            ev_rows.append(row)
            val = (1 - series["y_O2_out"][-1]) if case.case_id == "ver_plugflow" else (row["t_break_s"] if case.case_id == "ver_breakthrough" else series["y_O2_out"][-1])
            scal[gid] = val
            err = abs(val - ana["value"]) if ana["basis"] == "absolute" else abs(val - ana["value"]) / ana["value"]
            checks.append(dict(check="6a", case_id=case.case_id, grid_id=gid, N=n, rtol=rt, quantity=ana["name"], numeric=val,
                               analytic=ana["value"], error=err, basis=ana["basis"], tol=p["tol_check"], passed=bool(err <= p["tol_check"]),
                               mass_balance_err=row["mass_balance_err"]))
        e = {c["grid_id"]: c["error"] for c in checks if c["case_id"] == case.case_id}
        order = float(np.log2(e["N200"] / e["N400"])) if e["N400"] > 0 else float("nan")
        p_o, gci_pct, mono = gci(scal["N400"], scal["N200"], scal["N100"])
        conv.append(dict(case_id=case.case_id, quantity=ana["name"], f_N100=scal["N100"], f_N200=scal["N200"], f_N400=scal["N400"],
                         observed_order_error_ratio=order, p_celik=p_o, gci_fine_pct=gci_pct, monotonic=mono,
                         change_two_finest_pct=abs(scal["N400"] - scal["N200"]) / abs(scal["N400"]) * 100,
                         rtol_change_1e6_vs_1e8_pct=abs(scal["N400"] - scal["N400_r1e-08"]) / abs(scal["N400"]) * 100,
                         rtol_change_1e4_vs_1e6_pct=abs(scal["N400_r1e-04"] - scal["N400"]) / abs(scal["N400"]) * 100))
    return pd.DataFrame(ev_rows), pd.DataFrame(checks), pd.DataFrame(conv)

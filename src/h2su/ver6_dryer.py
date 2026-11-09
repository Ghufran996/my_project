"""VER-6c/6d/6e dryer cases (D46): Python 1-D reference (model 'python_mol'), lumped / tanks-in-series forms (model 'lumped_Nt'), event extraction identical
to COMSOL_MATLAB/model_dryer_startup_regen.m + compute_events.m, schema of code/configs/ver6_schema.csv.

Cases (results/ver6/<case>__<model>__<grid>.csv, rows appended to results/ver6/ver6_events.csv):
  dry_bt2, dry_su_i, dry_regen, dry_su_ii  (ver6_cases / dryer_cases; frozen k_LDF of results/vt6_cal_frozen.json)
Grids: N = 25, 50, 100, 200, 400 at rtol 1e-6 (+ N = 800), N = 400 at rtol 1e-4 and 1e-8. Upwind finite volume (see dryer.py).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import dryer as dr
from .io import RESULTS
from .ver6 import OUT, compute_events, params_for
from .ver6_cases import build

R = dr.R
PPM = dr.PPM
DRY_IDS = ("dry_bt2", "dry_su_i", "dry_regen", "dry_su_ii")
GRIDS = [(25, 1e-6, "N025"), (50, 1e-6, "N050"), (100, 1e-6, "N100"), (200, 1e-6, "N200"), (400, 1e-6, "N400"), (800, 1e-6, "N800"),
         (400, 1e-4, "N400_r1e-04"), (400, 1e-8, "N400_r1e-08")]
LUMPED = [(1, "lumped_N1"), (5, "lumped_N5"), (20, "lumped_N20")]


def stage_params(p: dict, prefix: str) -> dict:
    q = dict(p)
    for n in ("N_flux_mol_m2_s", "p_Pa", "yW_in_umol_mol", "yW_init_umol_mol", "q_init_mol_kg", "t_end_s", "dt_out_s"):
        q[n] = p[prefix + n]
    return q


def run_dryer(p: dict, iso: dict, N: int, rtol: float):
    """Run a dryer case (two-stage aware). Returns dict(t, y_out, mass_err, wall)."""
    t0 = time.time()
    y0prof = None
    if p.get("two_stage", 0.0) != 0.0:
        p1 = stage_params(p, "s1_")
        r1 = dr.run_case(p1, iso, N=N, rtol=rtol)
        # bed reversed (countercurrent purge); gas phase re-pressurised with dry product H2 (backfill, L15 Table 3 stage III): y0 = 0
        y0prof = np.vstack([np.zeros(N), r1["sol"]["F"][1, ::-1, -1]])
    r = dr.run_case(p, iso, N=N, rtol=rtol, y0_profiles=y0prof)
    sol = r["sol"]
    t, y = r["t"], r["y_out"] * PPM
    cT = p["p_Pa"] / (R * p["T_K"])
    nA = p["N_flux_mol_m2_s"] * p["A_m2"]
    h = p["L_m"] / N
    y_int = sol["F"][0].sum(axis=0) * h
    q_int = sol["F"][1].sum(axis=0) * h
    w_in = nA * p["yW_in_umol_mol"] * PPM * (t[-1] - t[0])
    w_out = nA * np.trapezoid(y, t)
    w_acc = p["A_m2"] * (p["eps"] * cT * PPM * (y_int[-1] - y_int[0]) + p["rho_bed_kg_m3"] * (q_int[-1] - q_int[0]))
    mass_err = abs(w_in - w_out - w_acc) / max(w_in, w_out, abs(w_acc), 1e-300)
    return dict(t=t, y_out=y, mass_err=mass_err, wall=time.time() - t0, us=r["us"])


def events(p: dict, iso: dict, g: dict, t, y) -> tuple[dict, dict]:
    """Event dict (VER-6 schema) + the 5 umol/mol first crossing (dryer breakthrough definition of L15 eq 9)."""
    n = len(t)
    series = dict(F_out_mol_s=np.full(n, p["N_flux_mol_m2_s"] * p["A_m2"]), y_H2_out=1 - y, y_H2O_out=y)
    regen = p["purge_all"] != 0
    cfg = dict(spec_cols=[] if regen else ["y_H2O_out"], spec_lim=[] if regen else [g["spec_H2O_umol_mol"] * PPM], M_H2=g["M_H2_kg_mol"],
               purge_all=bool(regen), break_col="y_H2O_out" if p["use_breakthrough"] != 0 else "", y_in=p["yW_in_umol_mol"] * PPM,
               break_frac=g["event_break_frac"])
    if not regen:
        T = p["T_K"]
        b1 = iso["lb10"] * np.exp(iso["lQ1"] / (R * iso["T0"]) * (iso["T0"] / T - 1))
        qH2 = lambda pb: iso["lqm"] * b1 * pb / (1 + b1 * pb)
        v_bed = p["A_m2"] * p["L_m"]
        cfg["repress_kg"] = (max(p["p_Pa"] - p["p_init_Pa"], 0.0) * p["eps"] * v_bed / (R * T)
                             + p["rho_bed_kg_m3"] * v_bed * max(qH2(p["p_Pa"] / 1e5) - qH2(p["p_init_Pa"] / 1e5), 0.0)) * g["M_H2_kg_mol"]
    ev = compute_events(t, series, cfg)
    if not regen:                                   # spec events (time to spec, vented H2) only inside the adsorption window t <= t_spec_window_s
        w = t <= p["t_spec_window_s"]
        evw = compute_events(t[w], {k: v[w] for k, v in series.items()}, dict(cfg, break_col=""))
        ev["t_spec_s"], ev["m_H2_vented_kg"] = evw["t_spec_s"], evw["m_H2_vented_kg"]
    return ev, dict(t_b5_s=dr.first_crossing(t, y / PPM, g["spec_H2O_umol_mol"]))


def run_all(include_lumped: bool = True):
    cases, rows = build()
    iso = dr.iso_params()
    ev_rows, extra = [], []
    OUT.mkdir(parents=True, exist_ok=True)
    for c in cases:
        if c.case_id not in DRY_IDS:
            continue
        p, g = params_for(c, rows)
        for N, rtol, gid in GRIDS:
            if N == 800 and c.case_id not in ("dry_bt2", "dry_su_i"):
                continue
            r = run_dryer(p, iso, N, rtol)
            ev, ex = events(p, iso, g, r["t"], r["y_out"])
            ev["mass_balance_err"] = r["mass_err"]
            ev_rows.append(dict(case_id=c.case_id, model="python_mol", grid_id=gid, rtol=rtol, **ev, wall_s=r["wall"]))
            extra.append(dict(case_id=c.case_id, model="python_mol", grid_id=gid, **ex, y_out_max_mol_mol=float(r["y_out"].max())))
            n = len(r["t"])
            nan = np.full(n, np.nan)
            pd.DataFrame(dict(case_id=c.case_id, model="python_mol", grid_id=gid, rtol=rtol, t_s=r["t"], T_out_K=nan, y_H2_out=1 - r["y_out"], y_O2_out=nan,
                              y_H2O_out=r["y_out"], y_N2_out=nan, F_out_mol_s=p["N_flux_mol_m2_s"] * p["A_m2"], p_out_Pa=p["p_Pa"])).to_csv(
                OUT / f"{c.case_id}__python_mol__{gid}.csv", index=False)
            print(c.case_id, gid, f"wall {r['wall']:.1f}s", flush=True)
        if include_lumped:
            for Nt, gid in LUMPED:
                r = run_dryer(p, iso, Nt, 1e-6)
                ev, ex = events(p, iso, g, r["t"], r["y_out"])
                ev["mass_balance_err"] = r["mass_err"]
                ev_rows.append(dict(case_id=c.case_id, model=gid, grid_id=f"Nt{Nt:03d}", rtol=1e-6, **ev, wall_s=r["wall"]))
                extra.append(dict(case_id=c.case_id, model=gid, grid_id=f"Nt{Nt:03d}", **ex, y_out_max_mol_mol=float(r["y_out"].max())))
                print(c.case_id, gid, f"wall {r['wall']:.1f}s", flush=True)
    new = pd.DataFrame(ev_rows)
    f = OUT / "ver6_events.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    if not old.empty:
        old = old[~old["case_id"].isin(DRY_IDS)]
    pd.concat([old, new], ignore_index=True).to_csv(f, index=False)
    pd.DataFrame(extra).to_csv(RESULTS / "ver6_dryer_extra.csv", index=False)
    return new, pd.DataFrame(extra)


GATE = ["t_break_s", "t_spec_s", "m_H2_vented_kg", "m_H2_purge_kg", "m_H2_repress_kg"]     # D46 outputs per event (schema events)
INFO = ["t_b5_s"]                                                                          # L15 breakpoint (VT6 definition): reported, not gating
QUANT = GATE + INFO + ["mass_balance_err"]


def _val(df, case, model, grid, q):
    r = df[(df.case_id == case) & (df.model == model) & (df.grid_id == grid)]
    return float(r[q].iloc[0]) if len(r) and q in r else np.nan


def verdict():
    """6d (grid, tolerance) and 6e (lumped vs 1-D, D29 revision rule) verdicts -> results/ver6_dryer_verdict.csv."""
    ev = pd.read_csv(OUT / "ver6_events.csv")
    ex = pd.read_csv(RESULTS / "ver6_dryer_extra.csv")
    df = ev.merge(ex[["case_id", "model", "grid_id", "t_b5_s", "y_out_max_mol_mol"]], on=["case_id", "model", "grid_id"], how="left")
    rows = []

    def rel(a, b):
        if np.isnan(a) or np.isnan(b):
            return np.nan
        if abs(b) < 1e-12:
            return 0.0 if abs(a) < 1e-12 else np.inf
        return (a - b) / b

    for case in DRY_IDS:
        sub = df[df.case_id == case]
        if sub.empty:
            continue
        ymax = float(sub[sub.model == "python_mol"]["y_out_max_mol_mol"].max())
        for q in QUANT:
            if q == "mass_balance_err":
                continue
            a2, a4, a8 = (_val(df, case, "python_mol", g, q) for g in ("N200", "N400", "N800"))
            r1, r4, r8 = (_val(df, case, "python_mol", g, q) for g in ("N400_r1e-04", "N400", "N400_r1e-08"))
            if np.isnan(a4):
                continue
            d_grid = rel(a2, a4)
            d_tol = rel(r4, r8)
            rows.append(dict(check="6d", case_id=case, quantity=q, value_N400=a4, change_N200_to_N400=d_grid, change_N400_to_N800=rel(a4, a8),
                             change_rtol_6_vs_8=d_tol, change_rtol_4_vs_6=rel(r1, r4),
                             criterion="grid < 1 % (N200 -> N400), rtol < 0.1 % (1e-6 -> 1e-8)" if q in GATE else "not gating (first-order upwind: change halves per grid doubling)",
                             verdict=("PASS" if (np.isnan(d_grid) or abs(d_grid) < 0.01) and (np.isnan(d_tol) or abs(d_tol) < 0.001) else "FAIL") if q in GATE else "INFO",
                             note=f"max outlet water mole fraction {ymax:.3g}" + ("; TRACE FORM VIOLATED (y > 0.05)" if ymax > 0.05 else "")))
        ref = {q: _val(df, case, "python_mol", "N400", q) for q in QUANT}
        passing = None
        for rev, (Nt, gid) in enumerate([(1, "lumped_N1"), (5, "lumped_N5"), (20, "lumped_N20")]):
            errs = {}
            for q in GATE + INFO:
                v = _val(df, case, gid, f"Nt{Nt:03d}", q)
                errs[q] = rel(v, ref[q])
            ok = all(np.isnan(e) or abs(e) <= 0.10 for q, e in errs.items() if q in GATE)
            if ok and passing is None:
                passing = rev
            for q, e in errs.items():
                rows.append(dict(check="6e", case_id=case, quantity=q, value_N400=ref[q], lumped_model=gid, revision=rev, rel_err_vs_1D=e,
                                 criterion="abs RE <= 10 % per event (D29: <= 2 revisions, N-node tanks in series)",
                                 verdict=(("PASS" if (np.isnan(e) or abs(e) <= 0.10) else "FAIL") if q in GATE else "INFO"),
                                 note=f"max outlet water mole fraction (1-D) {ymax:.3g}" + ("; undefined (0/0 or not applicable)" if np.isnan(e) else "") + ("; TRACE FORM VIOLATED, ill-conditioned H2 balance" if (ymax > 0.05 and q == "m_H2_purge_kg") else "")))
        rows.append(dict(check="6e-summary", case_id=case, quantity="all events", lumped_model="first passing revision", revision=passing if passing is not None else -1,
                         verdict="PASS" if passing is not None else "FAIL (no form within 2 revisions)",
                         note="revision 0 = single-node LDF, 1 = 5 tanks in series, 2 = 20 tanks in series (D29 revision rule; forms fixed before the run)"))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ver6_dryer_verdict.csv", index=False, float_format="%.6g")
    return out

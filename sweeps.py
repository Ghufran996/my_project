"""Brief 4 parameter sweeps on the PEM plant engine (items 5, 7, 8, 9, 10, 14).

    python code/sweeps.py {bop|controller|inventory|polb|resolution|awe} [--workers N]

Each task = one site-year; a worker loads the cached PV series once and runs every spec of the sweep on it (spec = dict(name, pol, t_sb, t_deb, over, off,
block, s1)). Output rows carry the metrics of plantrun.metrics. No value here is new: every parameter override refers to a registry-derived number
(plantrun.build_params) or to the pre-declared grids of PL8 / D40-D43 (cited in each sweep function).
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from h2su import plant as pl  # noqa: E402
from h2su import plantrun as pr  # noqa: E402
from h2su.io import CODE_DIR, RESULTS, param  # noqa: E402
from h2su.pv import block_mean_hold, pv_dc_series  # noqa: E402

KEEP = ["M_stack", "M_del", "M_purge", "M_offspec", "M_deoxo", "M_dryer", "M_blowdown", "dM_inv", "lambda_SU", "lambda_SU_complete", "lambda_SU_abort", "lambda_SS", "eta_del",
        "N_r_attempts", "N_r_completed", "N_r_aborted", "N_r_unfinished", "N_stops", "N_r_diurnal", "N_r_intra", "tau_eq_median", "n_cycles_closed", "E_stack", "E_bop",
        "E_standby", "E_avail", "E_below_min", "min_standby", "pause_min", "closure_H2_rel", "closure_E_rel", "identity_SU_SS_rel", "identity_abort_rel",
        "SEC_del", "SEC_del_BoP_from_PV", "SEC_del_L24_sb0095", "SEC_del_L1_sb0095"]


def site_years():
    y = yaml.safe_load((CODE_DIR / "configs" / "sites.yaml").read_text())
    return [(s, yr) for s, v in y["sites"].items() for yr in v["years_selected"]]


def run_spec(df: pd.DataFrame, spec: dict) -> dict:
    par = pr.build_params(spec["pol"], t_sb_min=spec.get("t_sb", 0.0), t_deb=spec.get("t_deb", 1.0), **spec.get("over", {}))
    sw = pr.switches(**spec.get("off", {}))
    d = df
    blk = spec.get("block", 1)
    if blk > 1:
        d = df.copy()
        d["p_dc_kw"] = block_mean_hold(df.p_dc_kw.to_numpy(), blk)
    if spec.get("months"):                       # window of calendar months (C4-comparable summer windows); non-contiguous months are concatenated
        d = d[d.index.month.isin(spec["months"])]
    t0 = time.perf_counter()
    tot, cyc, nc, _ = pr.run(d, par, sw)
    wall = time.perf_counter() - t0
    m = pr.metrics(tot, cyc, nc, par)
    row = {k: m.get(k, np.nan) for k in KEEP}
    row.update(wall_s=wall, mr_kg_min=par[pl.MRATED_MIN], i_inv=par[pl.I_INV], p_op_bar=par[pl.P_OP], p0_bar=par[pl.P0], theta_spec=par[pl.THETA_SPEC])
    if spec.get("s1"):
        p1, sw1 = pr.s1_config(par)
        t1 = pr.run(d, p1, sw1)
        row["M_del_S1"] = t1[0][pl.T_DEL]
        row["beta_B_S1"] = (t1[0][pl.T_DEL] - m["M_del"]) / m["M_del"] * 100.0
    return row


def _worker(args):
    s, y, specs = args
    df = pv_dc_series(s, y)
    out = []
    for sp in specs:
        r = run_spec(df, sp)
        r.update(site=s, year=y, variant=sp["name"], policy=sp["pol"], **{k: v for k, v in sp.get("tags", {}).items()})
        out.append(r)
    return out


def run_grid(specs: list[dict], workers: int = 12) -> pd.DataFrame:
    tasks = [(s, y, specs) for s, y in site_years()]
    t0 = time.time()
    rows = []
    with mp.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks), 1):
            rows.extend(r)
            print(f"{i}/{len(tasks)} site-years done ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["site", "year", "policy", "variant"]).reset_index(drop=True)
    df.attrs["wall_total_s"] = time.time() - t0
    return df


# ----------------------------------------------------------------------------------------------------------------------------------------- sweeps
def sweep_bop():
    """Item 5 (D41): BoP drawn from the PV bus. Variants: nominal (auxiliary supply), L24 constant share, L1 Table 1 load-dependent share (labelled laboratory-scale)."""
    specs = []
    for pol in "ACD":
        specs += [dict(name="nominal_aux_supply", pol=pol, s1=True), dict(name="BoP_from_PV_L24_constant", pol=pol, over=dict(BOP_MODE=1.0), s1=True),
                  dict(name="BoP_from_PV_L1_table1", pol=pol, over=dict(BOP_MODE=2.0), s1=True)]
    df = run_grid(specs)
    nom = df[df.variant == "nominal_aux_supply"].set_index(["site", "year", "policy"])
    for c in ("lambda_SU", "M_del", "N_r_attempts", "N_r_completed", "N_r_aborted"):
        df["d_" + c] = df.apply(lambda r: r[c] - nom.loc[(r.site, r.year, r.policy), c], axis=1)
    df["d_M_del_pct"] = df.d_M_del / df.apply(lambda r: nom.loc[(r.site, r.year, r.policy), "M_del"], axis=1) * 100.0
    out = RESULTS / "sens"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "bop_pv.csv", index=False)
    return df


def sweep_controller():
    """Item 7: t_deb in {1, 5, 15} min (PL8) and the abort-hold variant (D43b: hold while P < P_min, shutdown only after t_hold = 15 min), all policies."""
    specs = []
    for pol in "ACD":
        for td in (1.0, 5.0, 15.0):
            specs.append(dict(name=f"t_deb_{int(td)}min", pol=pol, t_deb=td, s1=True))
        specs.append(dict(name="abort_hold_t_deb1_hold15", pol=pol, t_deb=1.0, over=dict(HOLD_MODE=1.0), s1=True))
    df = run_grid(specs)
    out = RESULTS / "sens"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "controller.csv", index=False)
    return df


LEVELS = {"i_0.120_L32_end": 0.120, "i_0.195_L32": 0.195, "i_0.509_L13_82kW": 0.509, "i_0.699_nominal": None, "i_0.888_L13_143kW": 0.888, "i_3.178_L28_AWE_pilot": 3.178}


def sweep_inventory():
    """Item 8 (D43d): inventory OAT i in {0.120, 0.195, 0.509, 0.699 nominal, 0.888, 3.178 (L28, 'AWE pilot')} min/bar, POL-A and POL-C, all site-years."""
    specs = []
    for pol in "AC":
        for nm, v in LEVELS.items():
            specs.append(dict(name=nm, pol=pol, over={} if v is None else dict(I_INV=v), tags=dict(i_level=v if v is not None else float(pr.build_params("A")[pl.I_INV]))))
    df = run_grid(specs)
    out = RESULTS / "sens"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "inventory_oat_raw.csv", index=False)
    return df


def sweep_polb():
    """Item 9: POL-B t_sb,max in {15, 30, 60, 120} min x standby power {0.01, 0.095, 0.18} P_rated (PL5 F9, D40-2) + POL-A reference and POL-D; t_su,hot = 0 bounding row (D40-5)."""
    sb_nom = float(pr.build_params("A")[pl.SB_F])
    specs = []
    for sbf in (0.01, sb_nom, 0.18):
        specs.append(dict(name="POL-A", pol="A", over=dict(SB_F=sbf), tags=dict(sb_f=sbf, t_sb=0.0, t_su_hot="nominal")))
        specs.append(dict(name="POL-D", pol="D", over=dict(SB_F=sbf), tags=dict(sb_f=sbf, t_sb=1440.0, t_su_hot="nominal")))
        for tsb in (15.0, 30.0, 60.0, 120.0):
            specs.append(dict(name=f"POL-B_{int(tsb)}", pol="B", t_sb=tsb, over=dict(SB_F=sbf), tags=dict(sb_f=sbf, t_sb=tsb, t_su_hot="nominal")))
            specs.append(dict(name=f"POL-B_{int(tsb)}_tsuhot0", pol="B", t_sb=tsb, over=dict(SB_F=sbf, T_SU_HOT=0.0), tags=dict(sb_f=sbf, t_sb=tsb, t_su_hot="0 (bounding, no source)")))
        specs.append(dict(name="POL-D_tsuhot0", pol="D", over=dict(SB_F=sbf, T_SU_HOT=0.0), tags=dict(sb_f=sbf, t_sb=1440.0, t_su_hot="0 (bounding, no source)")))
    df = run_grid(specs)
    out = RESULTS / "policy"
    out.mkdir(parents=True, exist_ok=True)
    ref = df[(df.variant == "POL-A")].set_index(["site", "year", "sb_f"])
    df["d_M_del_vs_POLA_kg"] = df.apply(lambda r: r.M_del - ref.loc[(r.site, r.year, r.sb_f), "M_del"], axis=1)
    df["d_M_del_vs_POLA_pct"] = df.apply(lambda r: (r.M_del - ref.loc[(r.site, r.year, r.sb_f), "M_del"]) / ref.loc[(r.site, r.year, r.sb_f), "M_del"] * 100.0, axis=1)
    df["E_standby_kWh"] = df["E_standby"]
    df.to_csv(out / "polB_pareto.csv", index=False)
    return df


def sweep_resolution():
    """Item 10 (PL5 F6, H3b; C14 robustness 3 and 4): block means of 1/5/15/30/60 min held on the 1-min solver grid, all site-years, POL-A/C/D; B-S1 on the same input;
    C4 summer windows (JJA for the NH sites, Dec-Feb for Gobabeb; window definition is an implementer reading of 'summer', flagged)."""
    specs = []
    for pol in "ACD":
        for blk in (1, 5, 15, 30, 60):
            specs.append(dict(name=f"dt{blk}", pol=pol, block=blk, s1=True, tags=dict(block_min=blk, window="full year")))
    specs_w = []
    for pol in "ACD":
        for blk in (1, 60):
            specs_w.append(dict(name=f"dt{blk}_summer_NH", pol=pol, block=blk, s1=True, months=[6, 7, 8], tags=dict(block_min=blk, window="JJA")))
            specs_w.append(dict(name=f"dt{blk}_summer_SH", pol=pol, block=blk, s1=True, months=[12, 1, 2], tags=dict(block_min=blk, window="DJF")))
    tasks = [(s, y, specs + (specs_w if True else [])) for s, y in site_years()]
    rows = []
    t0 = time.time()
    with mp.Pool(12) as pool:
        for r in pool.imap_unordered(_worker, tasks):
            rows.extend(r)
    df = pd.DataFrame(rows)
    # keep the window rows that match the hemisphere of the site
    south = df.site.isin(["Gobabeb"])
    keep = (df.window == "full year") | ((df.window == "JJA") & ~south) | ((df.window == "DJF") & south)
    df = df[keep].sort_values(["site", "year", "policy", "variant"]).reset_index(drop=True)
    df.attrs["wall_total_s"] = time.time() - t0
    out = RESULTS / "resolution"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "resolution_raw.csv", index=False)
    return df


def sweep_awe():
    """Item 14 (D13 outcome): PEM engine with the L28 AWE-pilot inventory (3.178 min/bar) and the AWE P_min 0.156 (L21 abstract, flagged); the L28 inerting / purge /
    depressurisation durations are already the nominal engine durations (plantrun.build_params uses the awe_* registry rows). Cabauw and Gobabeb, POL-A and POL-C."""
    pmin_awe = param("pmin_awe_min")            # L21 abstract 15.6 % (flagged: abstract value, full text not read)
    specs = []
    for pol in "AC":
        specs += [dict(name="nominal_PEM", pol=pol, s1=True), dict(name="AWE_procedure_sensitivity", pol=pol, over=dict(I_INV=3.177966101694916, P_MIN_F=pmin_awe), s1=True),
                  dict(name="AWE_inventory_only", pol=pol, over=dict(I_INV=3.177966101694916), s1=True), dict(name="AWE_Pmin_only", pol=pol, over=dict(P_MIN_F=pmin_awe), s1=True)]
    sites = ("Cabauw", "Gobabeb")
    tasks = [(s, y, specs) for s, y in site_years() if s in sites]
    rows = []
    with mp.Pool(12) as pool:
        for r in pool.imap_unordered(_worker, tasks):
            rows.extend(r)
    df = pd.DataFrame(rows).sort_values(["site", "year", "policy", "variant"]).reset_index(drop=True)
    df["label"] = "AWE-procedure sensitivity, not an AWE plant model"
    out = RESULTS / "sens"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "awe_procedure.csv", index=False)
    return df


if __name__ == "__main__":
    mp.freeze_support()
    what = sys.argv[1]
    fn = {"bop": sweep_bop, "controller": sweep_controller, "inventory": sweep_inventory, "polb": sweep_polb, "awe": sweep_awe, "resolution": sweep_resolution}[what]
    df = fn()
    print(what, "rows", len(df), "wall_total_s", df.attrs.get("wall_total_s"))

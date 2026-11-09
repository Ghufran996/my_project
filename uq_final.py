"""Brief 5 item 7: final UQ runs (D16, D49, D50). Checkpoint per task (one .npz per task; existing files are skipped, so the run resumes after any interruption).

    python code/uq_final.py run [--workers N] [--only a|b|scen]    run all tasks in priority order
    python code/uq_final.py smoke                                  one small task (timing, sanity)
    python code/uq_final.py summarise                              items 7c, 7d (after the runs)

(a) Sobol, POL-B60: the same 11 factors as the D47 design (configs/sobol_factors.yaml variant a), N = 1024 Saltelli base samples (calc_second_order False), seeds 11/22/33/44/55,
    3 sites at the a-priori reference year -> results/uq/plant_<site>_B60_seed<seed>_a.npz (columns lambda_SU, M_del, N_r_completed, tau_eq_median, closure_H2, M_stack).
(b) Interval ensemble: the 10 epistemic factors (PV_RATIO fixed at 1.5), scrambled Sobol sequence (scipy.stats.qmc.Sobol, 2^10 = 1024 points, no Saltelli blocks), seeds as above,
    17 site-years x POL-A, POL-B60, POL-C, POL-D -> results/uq/ens_<site>_<year>_<pol>_seed<seed>.npz. Scenario rows PV_RATIO 1.0 and 2.0 with seed 11 only (..._pv1.0.npz, ..._pv2.0.npz).
    Seed-11 base rows additionally carry the B-S3 delivered H2 of the same sample (direct engine run, s3_config) for the beta intervals (item 7d).
The same sample matrix X (per seed) is used for every site-year and policy (paired).
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h2su import plantrun as pr  # noqa: E402
from h2su.io import CODE_DIR, RESULTS  # noqa: E402
from h2su.pv import pv_dc_series  # noqa: E402
from sobol_pipeline import CFG, SEEDS, apply_factors, plant_problem  # noqa: E402
from sweeps import site_years  # noqa: E402

UQ = RESULTS / "uq"
POLS = ("A", "B60", "C", "D")
ENS_FACTORS = [f for f in CFG["variants"]["a"]["factors"] if f != "PV_RATIO"]
ENS_COLS = ["lambda_SU", "lambda_SU_complete", "lambda_SU_abort", "M_stack", "M_del", "eta_del", "N_r_attempts", "N_r_completed", "N_r_aborted", "tau_eq_median", "SEC_del_L24", "closure_H2_rel",
            "closure_E_rel", "M_del_S3"]
SOB_COLS = ["lambda_SU", "M_del", "N_r_completed", "tau_eq_median", "closure_H2_rel", "M_stack"]


def params_for(pol: str) -> np.ndarray:
    return pr.build_params("B", t_sb_min=60.0) if pol == "B60" else pr.build_params(pol)


def thermal_base():
    th = pd.read_csv(RESULTS / "thermal_ranges.csv")
    C0 = float(th[(th.quantity == "C_per_MW") & (th.end.str.startswith("L26 printed"))].value.iloc[0])
    R0 = float(th[(th.quantity == "R_per_MW") & (th.end.str.startswith("L26 printed"))].value.iloc[0])
    return C0, R0


def ens_sample(seed: int) -> np.ndarray:
    lo = np.array([CFG["factors"][f].get("range_a", CFG["factors"][f].get("range"))[0] for f in ENS_FACTORS], float)
    hi = np.array([CFG["factors"][f].get("range_a", CFG["factors"][f].get("range"))[1] for f in ENS_FACTORS], float)
    u = qmc.Sobol(d=len(ENS_FACTORS), scramble=True, seed=seed).random_base2(m=10)
    return qmc.scale(u, lo, hi)


# --------------------------------------------------------------------------------------------------------------------------------------- tasks
def sobol_b60_task(args):
    site, seed = args
    f = UQ / f"plant_{site}_B60_seed{seed}_a.npz"
    if f.exists():
        return str(f), 0.0
    from SALib.sample import sobol as ss
    t0 = time.time()
    year = next(y for s, y in site_years() if s == site)
    df = pv_dc_series(site, year)
    prob = plant_problem("a")
    X = ss.sample(prob, CFG["design"]["N"], calc_second_order=False, scramble=True, seed=seed)
    par0 = params_for("B60")
    C0, R0 = thermal_base()
    sw = pr.switches()
    base = df.p_dc_kw.to_numpy()
    out = np.zeros((len(X), len(SOB_COLS)))
    for i, row in enumerate(X):
        x = dict(zip(prob["names"], row))
        p = apply_factors(par0, x, C0, R0)
        d = df.copy()
        d["p_dc_kw"] = base * (x["PV_RATIO"] / 1.5)
        tot, cyc, nc, _ = pr.run(d, p, sw)
        m = pr.metrics(tot, cyc, nc, p)
        out[i] = [m[c] for c in SOB_COLS]
    np.savez(f, X=X, Y=out, names=np.array(prob["names"]), cols=np.array(SOB_COLS), site=site, year=year, pol="B60", seed=seed, wall_s=time.time() - t0)
    return str(f), time.time() - t0


def ens_task(args):
    site, year, pol, seed, pv, with_s3, n = args
    tag = "" if pv == 1.5 else f"_pv{pv}"
    f = UQ / f"ens_{site}_{year}_{pol}_seed{seed}{tag}.npz"
    if f.exists() and n == 1024:
        return str(f), 0.0
    t0 = time.time()
    df = pv_dc_series(site, year)
    if pv != 1.5:
        df = df.copy()
        df["p_dc_kw"] = df.p_dc_kw.to_numpy() * (pv / 1.5)
    X = ens_sample(seed)[:n]
    par0 = params_for(pol)
    C0, R0 = thermal_base()
    sw = pr.switches()
    out = np.full((len(X), len(ENS_COLS)), np.nan)
    for i, row in enumerate(X):
        x = dict(zip(ENS_FACTORS, row))
        p = apply_factors(par0, x, C0, R0)
        tot, cyc, nc, _ = pr.run(df, p, sw)
        m = pr.metrics(tot, cyc, nc, p)
        for j, c in enumerate(ENS_COLS[:-1]):
            out[i, j] = m["SEC_del_L24_sb0095"] if c == "SEC_del_L24" else m[c]
        if with_s3:
            p3, sw3 = pr.s3_config(p)
            out[i, -1] = pr.run(df, p3, sw3)[0][pr.pl.T_DEL]
    if n == 1024:
        np.savez(f, X=X, Y=out, names=np.array(ENS_FACTORS), cols=np.array(ENS_COLS), site=site, year=year, pol=pol, seed=seed, pv_ratio=pv, wall_s=time.time() - t0)
    return str(f), time.time() - t0


S3_SEEDS = (22, 33, 44, 55)


def s3_closure(tot: np.ndarray) -> float:
    """Hydrogen closure of an engine run from its totals (same expression as plantrun.metrics closure_H2_rel)."""
    stack = tot[pr.pl.T_STACK]
    loss = tot[pr.pl.T_PURGE_] + tot[pr.pl.T_OFFSPEC] + tot[pr.pl.T_DEOXO] + tot[pr.pl.T_DRYER] + tot[pr.pl.T_BLOWDOWN]
    return abs(stack - (tot[pr.pl.T_DEL] + loss + tot[pr.pl.T_DINV])) / stack if stack > 0 else 0.0


def s3_task(args, n: int = 1024, write: bool = True):
    """Brief 7 item 2 (D67a): B-S3 delivered H2 of the seed-22/33/44/55 samples. Same s3_config and same sample matrix X as the stored ens_<...>_seed<seed>.npz
    (X is regenerated by ens_sample(seed) and asserted equal to the stored X; the ens file is read only). One checkpoint per task: s3_<site>_<year>_<pol>_seed<seed>.npz (X, M_del_S3, closure)."""
    site, year, pol, seed = args
    f = UQ / f"s3_{site}_{year}_{pol}_seed{seed}.npz"
    if write and f.exists():
        return str(f), 0.0
    t0 = time.time()
    df = pv_dc_series(site, year)
    X = ens_sample(seed)[:n]
    z = np.load(UQ / f"ens_{site}_{year}_{pol}_seed{seed}.npz", allow_pickle=True)
    assert [str(c) for c in z["names"]] == ENS_FACTORS and np.array_equal(z["X"][:n], X), "sample matrix differs from the stored ensemble"
    par0 = params_for(pol)
    C0, R0 = thermal_base()
    out = np.full(len(X), np.nan)
    clo = np.full(len(X), np.nan)
    for i, row in enumerate(X):
        p = apply_factors(par0, dict(zip(ENS_FACTORS, row)), C0, R0)
        p3, sw3 = pr.s3_config(p)
        tot = pr.run(df, p3, sw3)[0]
        out[i] = tot[pr.pl.T_DEL]
        clo[i] = s3_closure(tot)
    if write:
        tmp = f.with_name(f.stem + "_tmp.npz")
        np.savez(tmp, X=X, M_del_S3=out, closure=clo, names=np.array(ENS_FACTORS), site=site, year=year, pol=pol, seed=seed, wall_s=time.time() - t0)
        tmp.replace(f)
    return (out, clo) if not write else (str(f), time.time() - t0)


def all_tasks(only: str | None = None):
    sy = site_years()
    if only == "s3":
        return [("s3", (s, y, p, sd)) for sd in S3_SEEDS for s, y in sy for p in POLS]
    ref = {s: next(y for ss_, y in sy if ss_ == s) for s in ("Cabauw", "Carpentras", "Gobabeb")}
    tasks = []
    if only in (None, "b"):
        tasks += [("ens", (s, y, p, 11, 1.5, True, 1024)) for s, y in sy for p in POLS]                  # seed 11 with B-S3 (beta)
        tasks += [("ens", (s, y, p, sd, 1.5, False, 1024)) for sd in SEEDS[1:] for s, y in sy for p in POLS]
    if only in (None, "a"):
        tasks += [("sob", (s, sd)) for sd in SEEDS for s in ref]
    if only in (None, "scen"):
        tasks += [("ens", (s, y, p, 11, pv, False, 1024)) for pv in (1.0, 2.0) for s, y in sy for p in POLS]
    return tasks


def _dispatch(t):
    kind, a = t
    if kind == "s3":
        return (f"s3 {a[0]} {a[1]} {a[2]} seed{a[3]}",) + s3_task(a)
    return (sob_name(a) if kind == "sob" else ens_name(a),) + (sobol_b60_task(a) if kind == "sob" else ens_task(a))


def sob_name(a):
    return f"sobol_b60 {a[0]} seed{a[1]}"


def ens_name(a):
    return f"ens {a[0]} {a[1]} {a[2]} seed{a[3]} pv{a[4]}"


def run(workers: int, only: str | None):
    UQ.mkdir(parents=True, exist_ok=True)
    tasks = all_tasks(only)
    t0 = time.time()
    done = 0
    with mp.Pool(workers) as pool:
        for nm, f, w in pool.imap_unordered(_dispatch, tasks):
            done += 1
            print(f"{done}/{len(tasks)} {nm} ({w:.0f} s; elapsed {time.time() - t0:.0f} s)", flush=True)


def smoke():
    t0 = time.time()
    r = ens_task(("Cabauw", 2013, "A", 11, 1.5, True, 16))
    print("16 samples with S3:", time.time() - t0, "s")
    t0 = time.time()
    X = ens_sample(11)
    print("X", X.shape, X.min(axis=0).round(3), X.max(axis=0).round(3))


if __name__ == "__main__":
    mp.freeze_support()
    what = sys.argv[1]
    if what == "run":
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 10
        only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
        run(w, only)
    elif what == "worker":                      # independent OS process i of n (robust alternative to mp.Pool): tasks i, i+n, i+2n, ...
        i, n = int(sys.argv[2]), int(sys.argv[3])
        only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
        tl = all_tasks(only)
        t0 = time.time()
        for j, t in enumerate(tl):
            if j % n == i:
                nm, f, w = _dispatch(t)
                print(f"[w{i}] {j}/{len(tl)} {nm} ({w:.0f} s; elapsed {time.time() - t0:.0f} s)", flush=True)
    elif what == "claimworker":                 # Brief 7 item 2: independent OS process; tasks are claimed dynamically via O_EXCL claim files (restart: stale claims without checkpoint are removed by `clearclaims`)
        import os
        wid = sys.argv[2]
        only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "s3"
        tl = all_tasks(only)
        t0 = time.time()
        for j, t in enumerate(tl):
            a = t[1]
            ck = UQ / f"s3_{a[0]}_{a[1]}_{a[2]}_seed{a[3]}.npz"
            cl = UQ / f"s3_{a[0]}_{a[1]}_{a[2]}_seed{a[3]}.claim"
            if ck.exists():
                continue
            try:
                fd = os.open(cl, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"worker {wid} pid {os.getpid()}".encode())
                os.close(fd)
            except FileExistsError:
                continue
            nm, f, w = _dispatch(t)
            print(f"[w{wid}] {j + 1}/{len(tl)} {nm} ({w:.0f} s; elapsed {time.time() - t0:.0f} s)", flush=True)
        print(f"[w{wid}] finished", flush=True)
    elif what == "clearclaims":
        for c in UQ.glob("s3_*.claim"):
            c.unlink()
        print("claims removed")
    elif what == "smoke":
        smoke()
    elif what == "summarise":
        from uq_final_summary import main as m
        m()

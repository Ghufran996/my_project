"""Brief 4 item 12: Sobol pipeline (SALib 1.6.0), verified on the Ishigami function, then the PEM plant (variant a of D47 only).

    python code/sobol_pipeline.py ishigami          -> results/uq/ishigami_check.csv
    python code/sobol_pipeline.py plant [--workers N]  -> results/uq/plant_*.npz (checkpoint per task), results/uq/sobol_plant.csv (after all tasks)
    python code/sobol_pipeline.py summarise         -> results/uq/sobol_plant.csv, results/uq/sobol_plant_summary.csv

Design (configs/sobol_factors.yaml): N = 1024 Saltelli base samples, first-order and total-order indices (calc_second_order False), 5 scrambled seeds, 1000 bootstrap
resamples, 95 % confidence. Ishigami (a = 7, b = 0.1, x_i ~ U(-pi, pi)): analytic S1 = (0.3139, 0.4424, 0), ST = (0.5576, 0.4424, 0.2437); acceptance abs error <= 0.02 at N = 1024.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from SALib.analyze import sobol as sa  # noqa: E402
from SALib.sample import sobol as ss  # noqa: E402

from h2su import plant as pl  # noqa: E402
from h2su import plantrun as pr  # noqa: E402
from h2su.io import CODE_DIR, RESULTS  # noqa: E402
from h2su.pv import pv_dc_series  # noqa: E402

UQ = RESULTS / "uq"
CFG = yaml.safe_load((CODE_DIR / "configs" / "sobol_factors.yaml").read_text(encoding="utf-8"))
N_BASE = CFG["design"]["N"]
SEEDS = CFG["design"]["seeds"]
NRES = CFG["design"]["bootstrap_resamples"]


# ------------------------------------------------------------------------------------------------------------------------------------ Ishigami
def ishigami(x, a=7.0, b=0.1):
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def ishigami_check():
    UQ.mkdir(parents=True, exist_ok=True)
    a, b = 7.0, 0.1
    v1 = 0.5 * (1 + b * np.pi ** 4 / 5) ** 2
    v2 = a ** 2 / 8
    v13 = 8 * b ** 2 * np.pi ** 8 / 225
    var = 0.5 + v2 + 0.5 * 0 + b * np.pi ** 4 / 5 + b ** 2 * np.pi ** 8 / 18
    s1 = np.array([v1, v2, 0.0]) / var
    st = np.array([v1 + v13, v2, v13]) / var
    problem = dict(num_vars=3, names=["x1", "x2", "x3"], bounds=[[-np.pi, np.pi]] * 3)
    rows = []
    for seed in SEEDS:
        X = ss.sample(problem, N_BASE, calc_second_order=False, scramble=True, seed=seed)
        Y = ishigami(X)
        r = sa.analyze(problem, Y, calc_second_order=False, num_resamples=NRES, conf_level=0.95, seed=seed)
        for i, nm in enumerate(problem["names"]):
            rows.append(dict(seed=seed, N=N_BASE, n_model_runs=len(X), factor=nm, S1=r["S1"][i], S1_conf=r["S1_conf"][i], S1_analytic=s1[i], S1_abs_err=abs(r["S1"][i] - s1[i]),
                             ST=r["ST"][i], ST_conf=r["ST_conf"][i], ST_analytic=st[i], ST_abs_err=abs(r["ST"][i] - st[i])))
    df = pd.DataFrame(rows)
    df["pass_abs_err_le_0.02"] = (df.S1_abs_err <= 0.02) & (df.ST_abs_err <= 0.02)
    mean = df.groupby("factor").agg(S1=("S1", "mean"), ST=("ST", "mean"), S1_analytic=("S1_analytic", "first"), ST_analytic=("ST_analytic", "first")).reset_index()
    mean["S1_abs_err"] = (mean.S1 - mean.S1_analytic).abs()
    mean["ST_abs_err"] = (mean.ST - mean.ST_analytic).abs()
    mean["seed"] = "mean of 5 seeds"
    mean["N"] = N_BASE
    mean["pass_abs_err_le_0.02"] = (mean.S1_abs_err <= 0.02) & (mean.ST_abs_err <= 0.02)
    df = pd.concat([df.assign(seed=df.seed.astype(str)), mean], ignore_index=True)
    df.to_csv(UQ / "ishigami_check.csv", index=False, float_format="%.6g")
    print(mean.round(4).to_string())
    print(df.round(4).to_string())
    single = df[df.seed != "mean of 5 seeds"]
    print("analytic S1", s1.round(4), "ST", st.round(4), "variance", var, "single-seed max abs err", max(single.S1_abs_err.max(), single.ST_abs_err.max()), "single-seed all pass", bool(single["pass_abs_err_le_0.02"].all()), "| 5-seed mean max abs err", max(mean.S1_abs_err.max(), mean.ST_abs_err.max()))
    return df


# ------------------------------------------------------------------------------------------------------------------------------------ plant
def plant_problem(variant: str = "a"):
    names = CFG["variants"][variant]["factors"]
    bounds = []
    for n in names:
        f = CFG["factors"][n]
        bounds.append(f.get(f"range_{variant}", f.get("range")))
    return dict(num_vars=len(names), names=names, bounds=bounds)


def apply_factors(par: np.ndarray, x: dict, base_C: float, base_R: float) -> np.ndarray:
    p = par.copy()
    for k in ("P_MIN_F", "SB_F", "I_INV", "P_OP", "T_PURGE", "T_BLOW", "T_SU_COLD", "THETA_SPEC", "XO2_OVERRIDE"):
        if k in x:
            p[getattr(pl, k)] = x[k]
    C = x.get("C_PER_MW", base_C)
    R = x.get("R_PER_MW", base_R)
    p[pl.TAU_COOL] = R * C / 60.0
    return p


def run_task(args):
    site, pol, seed, variant = args
    f = UQ / f"plant_{site}_{pol}_seed{seed}_{variant}.npz"
    if f.exists():
        return str(f), 0.0
    t0 = time.time()
    year = next(y for s, y in __import__("sweeps").site_years() if s == site)
    df = pv_dc_series(site, year)
    prob = plant_problem(variant)
    X = ss.sample(prob, N_BASE, calc_second_order=False, scramble=True, seed=seed)
    par0 = pr.build_params(pol)
    th = pd.read_csv(RESULTS / "thermal_ranges.csv")
    C0 = float(th[(th.quantity == "C_per_MW") & (th.end.str.startswith("L26 printed"))].value.iloc[0])
    R0 = float(th[(th.quantity == "R_per_MW") & (th.end.str.startswith("L26 printed"))].value.iloc[0])
    sw = pr.switches()
    base_pdc = df.p_dc_kw.to_numpy()
    out = np.zeros((len(X), 5))
    for i, row in enumerate(X):
        x = dict(zip(prob["names"], row))
        p = apply_factors(par0, x, C0, R0)
        d = df
        if "PV_RATIO" in x:
            d = df.copy()
            d["p_dc_kw"] = base_pdc * (x["PV_RATIO"] / 1.5)
        tot, cyc, nc, _ = pr.run(d, p, sw)
        m = pr.metrics(tot, cyc, nc, p)
        out[i] = [m["lambda_SU"], m["M_del"], m["N_r_completed"], m["tau_eq_median"], m["closure_H2_rel"]]
    np.savez(f, X=X, Y=out, names=np.array(prob["names"]), site=site, year=year, pol=pol, seed=seed, wall_s=time.time() - t0)
    return str(f), time.time() - t0


def run_plant(workers: int = 12, variant: str = "a"):
    UQ.mkdir(parents=True, exist_ok=True)
    tasks = [(s, p, sd, variant) for s in ("Cabauw", "Carpentras", "Gobabeb") for p in ("A", "C") for sd in SEEDS]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, (f, w) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            print(f"{i}/{len(tasks)} {Path(f).name} ({w:.0f} s; elapsed {time.time() - t0:.0f} s)", flush=True)


def summarise(variant: str = "a"):
    prob = plant_problem(variant)
    rows = []
    for f in sorted(UQ.glob(f"plant_*_{variant}.npz")):
        z = np.load(f, allow_pickle=True)
        site, pol, seed = str(z["site"]), str(z["pol"]), int(z["seed"])
        for j, out in enumerate(["lambda_SU", "M_del", "N_r_completed", "tau_eq_median"]):
            Y = z["Y"][:, j]
            if not np.isfinite(Y).all():
                Y = np.nan_to_num(Y, nan=np.nanmean(Y))
            r = sa.analyze(prob, Y, calc_second_order=False, num_resamples=NRES, conf_level=0.95, seed=seed)
            for i, nm in enumerate(prob["names"]):
                rows.append(dict(site=site, year=int(z["year"]), policy=pol, seed=seed, output=out, factor=nm, S1=r["S1"][i], S1_conf=r["S1_conf"][i], ST=r["ST"][i], ST_conf=r["ST_conf"][i],
                                 n_runs=len(Y), output_mean=Y.mean(), output_std=Y.std(), closure_max=float(z["Y"][:, 4].max())))
    df = pd.DataFrame(rows)
    df.to_csv(UQ / "sobol_plant.csv", index=False, float_format="%.6g")
    g = df.groupby(["site", "year", "policy", "output", "factor"]).agg(S1_mean=("S1", "mean"), S1_std=("S1", "std"), ST_mean=("ST", "mean"), ST_std=("ST", "std"), ST_conf_mean=("ST_conf", "mean"),
                                                                          n_seeds=("seed", "count"), output_mean=("output_mean", "mean"), output_std=("output_std", "mean")).reset_index()
    g.to_csv(UQ / "sobol_plant_summary.csv", index=False, float_format="%.6g")
    print(g[g.output == "lambda_SU"].round(3).to_string())
    return g


if __name__ == "__main__":
    mp.freeze_support()
    what = sys.argv[1]
    if what == "ishigami":
        ishigami_check()
    elif what == "plant":
        run_plant(int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 12)
    elif what == "summarise":
        summarise()

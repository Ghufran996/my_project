"""Brief 6 item 8 (extended in Brief 7 item 2 with the 272 B-S3 tasks of seeds 22-55): UQ completion check against the task list of uq_final.py (does not trust the file count).

    python code/uq_completion_check.py

For every task of uq_final.all_tasks() the expected checkpoint must exist, load, carry the expected identity (site, year, policy, seed, PV ratio), the expected number of rows
(ensemble 1024; Sobol N*(D+2) = 1024*13 = 13312 for SALib Saltelli with calc_second_order=False, D = 11 factors; R59/R75 text "12 288" was a miscount, D49 states 13 312), the paired sample matrix (ensemble: ens_sample(seed); Sobol: Saltelli sample of the seed), only finite required outputs and
closure_H2_rel, closure_E_rel <= 1e-6 (Sobol files carry closure_H2_rel only). The B-S3 column must be finite for the seed-11 base rows (beta) and is not required elsewhere.
Output: results/uq/completion_check.csv (one row per expected task) and results/uq/completion_check_summary.txt. Files in results/uq/*.npz that are not in the task list are listed
(the 30 POL-A/C Sobol checkpoints of R59 are expected extras and are checked too). Missing or failed tasks are NOT filled or substituted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uq_final as uf  # noqa: E402
from h2su.io import RESULTS  # noqa: E402
from sobol_pipeline import CFG, SEEDS, plant_problem  # noqa: E402

TOL = 1e-6
UQ = RESULTS / "uq"
_cache: dict = {}


def check_ens(a):
    site, year, pol, seed, pv, with_s3, n = a
    tag = "" if pv == 1.5 else f"_pv{pv}"
    f = UQ / f"ens_{site}_{year}_{pol}_seed{seed}{tag}.npz"
    return f, check_common(f, dict(site=site, year=year, pol=pol, seed=seed, pv_ratio=pv), n_rows=1024, ens=True, need_s3=bool(with_s3), seed=seed)


def check_sob(a, pol="B60"):
    site, seed = a
    f = UQ / f"plant_{site}_{pol}_seed{seed}_a.npz"
    return f, check_common(f, dict(site=site, pol=pol, seed=seed), n_rows=CFG["design"]["N"] * (len(plant_problem("a")["names"]) + 2), ens=False, need_s3=False, seed=seed)


def check_common(f: Path, ident: dict, n_rows: int, ens: bool, need_s3: bool, seed: int):
    r = dict(exists=f.exists(), loads=False, identity_ok=False, rows=0, rows_ok=False, finite_ok=False, sample_ok=False, closure_H2_max=np.nan, closure_E_max=np.nan, closure_ok=False, note="")
    if not f.exists():
        r["note"] = "missing"
        return r
    try:
        z = np.load(f, allow_pickle=True)
        X, Y = z["X"], z["Y"]
        cols = [str(c) for c in z["cols"]] if "cols" in z.files else ["lambda_SU", "M_del", "N_r_completed", "tau_eq_median", "closure_H2_rel"][:Y.shape[1]]   # R59 A/C files carry no cols entry (sobol_pipeline.py line 126)
        names = [str(c) for c in z["names"]]
        r["loads"] = True
    except Exception as e:  # noqa: BLE001
        r["note"] = f"load error: {type(e).__name__}: {e}"
        return r
    ok = True
    for k, v in ident.items():
        got = z[k].item() if z[k].shape == () else z[k]
        if isinstance(v, float):
            ok &= abs(float(got) - v) < 1e-12
        elif isinstance(v, int):
            ok &= int(got) == v
        else:
            ok &= str(got) == v
    r["identity_ok"] = bool(ok)
    r["rows"] = int(Y.shape[0])
    r["rows_ok"] = bool(Y.shape[0] == n_rows and X.shape[0] == n_rows and Y.shape[1] == len(cols))
    req = [i for i, c in enumerate(cols) if not (c == "M_del_S3" and not need_s3)]
    r["finite_ok"] = bool(np.isfinite(Y[:, req]).all() and np.isfinite(X).all())
    if ens:
        key = ("ens", seed)
        if key not in _cache:
            _cache[key] = uf.ens_sample(seed)
        exp = _cache[key]
        r["sample_ok"] = bool(names == list(uf.ENS_FACTORS) and X.shape == exp.shape and np.array_equal(X, exp))
    else:
        key = ("sob", seed)
        if key not in _cache:
            from SALib.sample import sobol as ss
            _cache[key] = ss.sample(plant_problem("a"), CFG["design"]["N"], calc_second_order=False, scramble=True, seed=seed)
        exp = _cache[key]
        r["sample_ok"] = bool(names == list(plant_problem("a")["names"]) and X.shape == exp.shape and np.array_equal(X, exp))
    ch = Y[:, cols.index("closure_H2_rel")]
    r["closure_H2_max"] = float(np.nanmax(ch))
    ok = bool(np.isfinite(ch).all() and np.nanmax(ch) <= TOL)
    if "closure_E_rel" in cols:
        ce = Y[:, cols.index("closure_E_rel")]
        r["closure_E_max"] = float(np.nanmax(ce))
        ok = ok and bool(np.isfinite(ce).all() and np.nanmax(ce) <= TOL)
    r["closure_ok"] = ok
    return r


def check_s3(a):
    """B-S3 checkpoint s3_<site>_<year>_<pol>_seed<seed>.npz (X, M_del_S3, closure): identity, 1024 rows, M_del_S3 finite and positive, X equal to ens_sample(seed), closure <= TOL."""
    site, year, pol, seed = a
    f = UQ / f"s3_{site}_{year}_{pol}_seed{seed}.npz"
    r = dict(exists=f.exists(), loads=False, identity_ok=False, rows=0, rows_ok=False, finite_ok=False, sample_ok=False, closure_H2_max=np.nan, closure_E_max=np.nan, closure_ok=False, note="")
    if not f.exists():
        r["note"] = "missing"
        return f, r
    try:
        z = np.load(f, allow_pickle=True)
        X, M, C = z["X"], z["M_del_S3"], z["closure"]
        r["loads"] = True
    except Exception as e:  # noqa: BLE001
        r["note"] = f"load error: {type(e).__name__}: {e}"
        return f, r
    r["identity_ok"] = bool(str(z["site"]) == site and int(z["year"]) == year and str(z["pol"]) == pol and int(z["seed"]) == seed)
    r["rows"] = int(len(M))
    r["rows_ok"] = bool(len(M) == 1024 and X.shape[0] == 1024 and len(C) == 1024)
    r["finite_ok"] = bool(np.isfinite(M).all() and (M > 0).all() and np.isfinite(X).all())
    key = ("ens", seed)
    if key not in _cache:
        _cache[key] = uf.ens_sample(seed)
    r["sample_ok"] = bool([str(n) for n in z["names"]] == list(uf.ENS_FACTORS) and np.array_equal(X, _cache[key]))
    r["closure_H2_max"] = float(np.nanmax(C))
    r["closure_ok"] = bool(np.isfinite(C).all() and np.nanmax(C) <= TOL)
    return f, r


def passed(r):
    return bool(r["exists"] and r["loads"] and r["identity_ok"] and r["rows_ok"] and r["finite_ok"] and r["sample_ok"] and r["closure_ok"])


def main():
    tasks = uf.all_tasks()
    rows, expected = [], set()
    for j, (kind, a) in enumerate(tasks):
        if kind == "ens":
            f, r = check_ens(a)
            grp = "scenario" if a[4] != 1.5 else "ensemble"
            tid = f"ens {a[0]} {a[1]} {a[2]} seed{a[3]} pv{a[4]}"
        else:
            f, r = check_sob(a)
            grp, tid = "sobol_B60", f"sobol_b60 {a[0]} seed{a[1]}"
        expected.add(f.name)
        r.update(task_index=j, group=grp, task=tid, file=f.name)
        r["PASS"] = passed(r)
        rows.append(r)
    for a in [t[1] for t in uf.all_tasks("s3")]:
        f, r = check_s3(a)
        expected.add(f.name)
        r.update(task_index=len(rows), group="B-S3 seeds 22-55", task=f"s3 {a[0]} {a[1]} {a[2]} seed{a[3]}", file=f.name)
        r["PASS"] = passed(r)
        rows.append(r)
    extras = []
    sites = sorted({s for s, _ in uf.site_years()})
    for pol in ("A", "C"):
        for s in sites:
            for sd in SEEDS:
                f, r = check_sob((s, sd), pol)
                expected.add(f.name)
                r.update(task_index=-1, group="sobol_A_C (R59, extra)", task=f"sobol {pol} {s} seed{sd}", file=f.name)
                r["PASS"] = passed(r)
                extras.append(r)
    allrows = pd.DataFrame(rows + extras)
    cols = ["task_index", "group", "task", "file", "PASS", "exists", "loads", "identity_ok", "rows", "rows_ok", "finite_ok", "sample_ok", "closure_H2_max", "closure_E_max", "closure_ok", "note"]
    allrows[cols].to_csv(UQ / "completion_check.csv", index=False, float_format="%.6g")
    unexpected = sorted(p.name for p in UQ.glob("*.npz") if p.name not in expected)
    d = pd.DataFrame(rows)
    lines = [f"expected tasks (uq_final.all_tasks + all_tasks('s3')): {len(d)}  (ensemble {int((d.group == 'ensemble').sum())}, Sobol POL-B60 {int((d.group == 'sobol_B60').sum())}, scenarios {int((d.group == 'scenario').sum())}, "
             f"B-S3 seeds 22-55 {int((d.group == 'B-S3 seeds 22-55').sum())})",
             f"PASS: {int(d.PASS.sum())} of {len(d)}",
             f"missing files: {int((~d.exists).sum())}; load errors: {int((d.exists & ~d.loads).sum())}; identity failures: {int((d.loads & ~d.identity_ok).sum())}; "
             f"row-count failures: {int((d.loads & ~d.rows_ok).sum())}; non-finite: {int((d.loads & ~d.finite_ok).sum())}; sample-matrix mismatches: {int((d.loads & ~d.sample_ok).sum())}; "
             f"closure failures: {int((d.loads & ~d.closure_ok).sum())}",
             f"max closure_H2_rel over expected tasks: {d.closure_H2_max.max():.3e}; max closure_E_rel: {d.closure_E_max.max():.3e} (tolerance {TOL:g})",
             f"extra POL-A/C Sobol checkpoints (R59): {len(extras)} checked, PASS {sum(e['PASS'] for e in extras)}; npz files not in any list: {unexpected}"]
    bad = d[~d.PASS]
    if len(bad):
        lines.append("FAILED / MISSING TASKS:")
        lines += [f"  {r.task_index} {r.task} -> {r.file} {r.note}" for r in bad.itertuples()]
    txt = "\n".join(lines)
    (UQ / "completion_check_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return len(bad) == 0 and not unexpected


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

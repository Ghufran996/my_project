"""Brief 5 item 7c, 7d: summaries of the final UQ runs (works on whatever checkpoints exist; reports the number of seeds / years found per set).

    python code/uq_final_summary.py [intervals|scm|sobol|beta|all]

Outputs: results/uq/intervals.csv, scm_verdict.csv, scs_verdict.csv, sobol_pooled.csv, beta_intervals.csv, sobol_beta.csv, beta_table_check.csv.
Conventions (D49, D50): percentiles per seed over all rows of a site x policy (years pooled with equal weight, 1024 rows per site-year and seed), pooled value = mean of the per-seed percentiles;
SC-M class: P5 >= 1.0 % material, P95 < 0.5 % negligible, otherwise condition-dependent. SC-S: Sobol S_T >= 0.1 -> bootstrap 95 % half-width <= 0.05 for the POOLED 5-seed estimator
(Saltelli blocks of the five seeds concatenated; single-seed half-widths reported beside it); percentile spread (max - min)/mean <= 1 % per site x policy with an absolute floor of 0.01 pp.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h2su import plantrun as pr  # noqa: E402
from h2su.baselines.bcl import bcl  # noqa: E402
from h2su.baselines.bs1 import bs1_annual  # noqa: E402
from h2su.io import RESULTS, param  # noqa: E402
from h2su.pv import pv_dc_series  # noqa: E402
from sweeps import site_years  # noqa: E402

UQ = RESULTS / "uq"
SEEDS = [11, 22, 33, 44, 55]
SITES = ["Cabauw", "Carpentras", "Gobabeb"]
POLS = ["A", "B60", "C", "D"]
OUT = ["lambda_SU", "tau_eq_median", "tau_mean", "eta_del", "M_del"]
MDOT_RATED = float(pr.build_params("A")[pr.pl.MRATED_MIN])            # kg H2 per min at rated stack power (constant: no uncertain factor changes it); rated-min = kg / MDOT_RATED


def load_ens():
    """dict[(site, year, pol, seed, pv)] -> DataFrame of one sample set."""
    d = {}
    for f in sorted(UQ.glob("ens_*.npz")):
        z = np.load(f, allow_pickle=True)
        cols = [str(c) for c in z["cols"]]
        df = pd.DataFrame(z["Y"], columns=cols)
        for i, nme in enumerate(z["names"]):
            df[str(nme)] = z["X"][:, i]
        d[(str(z["site"]), int(z["year"]), str(z["pol"]), int(z["seed"]), float(z["pv_ratio"]))] = df
    return d


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.percentile(x, [5, 50, 95]) if len(x) else np.array([np.nan] * 3)


def intervals():
    ens = load_ens()
    rows = []
    lam_rows = {}
    for pv in (1.5, 1.0, 2.0):
        for s in SITES:
            for pol in POLS:
                keys = [k for k in ens if k[0] == s and k[2] == pol and k[4] == pv]
                years = sorted({k[1] for k in keys})
                allseeds = sorted({k[3] for k in keys})
                for sd in allseeds:
                    ys = [y for y in years if (s, y, pol, sd, pv) in ens]
                    if not ys:
                        continue
                    frame = pd.concat([ens[(s, y, pol, sd, pv)] for y in ys], ignore_index=True)
                    frame["eta_del"] = frame.M_del / frame.M_stack
                    if pv == 1.5:
                        lam_rows.setdefault((s, pol), []).append(frame.lambda_SU.to_numpy())
                    # D66: tau_mean = lambda_SU/100 x M_stack / (N_r_completed x mdot_rated) [rated-min per completed restart, aborted attempts included in the numerator]
                    frame["tau_mean"] = np.where(frame.N_r_completed > 0, frame.lambda_SU / 100.0 * frame.M_stack / (frame.N_r_completed.clip(lower=1) * MDOT_RATED), np.nan)
                    r = dict(site=s, policy=pol, seed=sd, pv_ratio=pv, n_years=len(ys), n_rows=len(frame))
                    for o in OUT:
                        p5, p50, p95 = pct(frame[o])
                        r.update({f"{o}_P5": p5, f"{o}_P50": p50, f"{o}_P95": p95})
                    tm = frame.tau_eq_median.to_numpy()
                    r["share_samples_tau_median_zero"] = float(np.mean(tm[np.isfinite(tm)] == 0.0)) * 100.0 if np.isfinite(tm).any() else np.nan
                    r["closure_H2_max"] = frame.closure_H2_rel.max()
                    r["closure_E_max"] = frame.closure_E_rel.max()
                    rows.append(r)
    d = pd.DataFrame(rows)
    if d.empty:
        print("no ensemble checkpoints yet")
        return d, d
    pooled = d.groupby(["site", "policy", "pv_ratio"]).agg(n_seeds=("seed", "count"), n_years=("n_years", "first"), **{c: (c, "mean") for c in d.columns if c.endswith(("_P5", "_P50", "_P95"))},
                                                          share_samples_tau_median_zero=("share_samples_tau_median_zero", "mean"), closure_H2_max=("closure_H2_max", "max"), closure_E_max=("closure_E_max", "max")).reset_index()
    pooled.insert(2, "seed", "pooled (mean of per-seed percentiles)")
    d["seed"] = d.seed.astype(str)
    allr = pd.concat([d, pooled], ignore_index=True)
    # between-year spread at nominal
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    ann = ann[ann.model == "full"].copy()
    ann["policy"] = ann.policy.replace({"B": "B60"})
    ann = ann[ann.policy.isin(["A", "C", "D"])]
    yr = ann.groupby(["site", "policy"]).agg(nominal_lambda_SU_mean=("lambda_SU", "mean"), nominal_lambda_SU_min_year=("lambda_SU", "min"), nominal_lambda_SU_max_year=("lambda_SU", "max"),
                                             nominal_lambda_SU_sd_between_years=("lambda_SU", "std"), nominal_eta_del_mean=("eta_del", "mean"), n_years_nominal=("year", "count")).reset_index()
    pooled = pooled.merge(yr, on=["site", "policy"], how="left")
    # Brief 7 item 4c: share of pooled rows (all seeds, all site-years, PV ratio 1.5) with lambda_SU below the nominal mean (mean over site-years of the nominal annual value; POL-B60 from nominal/b60_nominal.csv)
    nomf = RESULTS / "nominal" / "b60_nominal.csv"
    nom_mean = {(r.site, r.policy): r.nominal_lambda_SU_mean for r in yr.itertuples()}
    if nomf.exists():
        b6 = pd.read_csv(nomf)
        nom_mean.update({(s_, "B60"): g_.lambda_SU.mean() for s_, g_ in b6.groupby("site")})
    pooled["nominal_lambda_SU_mean_used"] = [nom_mean.get((r.site, r.policy), np.nan) if r.pv_ratio == 1.5 else np.nan for r in pooled.itertuples()]
    pooled["nominal_percentile_in_ensemble"] = [float(np.mean(np.concatenate(lam_rows[(r.site, r.policy)]) < r.nominal_lambda_SU_mean_used)) * 100.0 if r.pv_ratio == 1.5 and np.isfinite(r.nominal_lambda_SU_mean_used) else np.nan
                                                for r in pooled.itertuples()]
    allr.to_csv(UQ / "intervals.csv", index=False, float_format="%.6g")
    pooled.to_csv(UQ / "intervals_pooled.csv", index=False, float_format="%.6g")
    return allr, pooled


def klass(p5, p95):
    return "material" if p5 >= 1.0 else ("negligible" if p95 < 0.5 else "condition-dependent")


def scm_scs():
    allr, pooled = intervals()
    if allr.empty:
        return
    rows, srows = [], []
    d = allr[allr.seed != "pooled (mean of per-seed percentiles)"]
    for (s, pol, pv), g in d.groupby(["site", "policy", "pv_ratio"]):
        p5, p50, p95 = g.lambda_SU_P5.mean(), g.lambda_SU_P50.mean(), g.lambda_SU_P95.mean()
        rows.append(dict(source="interval ensemble (10 factors, PV ratio fixed)", site=s, policy=pol, pv_ratio=pv, n_seeds=len(g), n_years=int(g.n_years.iloc[0]), lambda_SU_P5=p5, lambda_SU_P50=p50, lambda_SU_P95=p95,
                         SC_M_class=klass(p5, p95) if pv == 1.5 else "not assigned (scenario row, seed 11 only; D60 RR1)"))
        if pv == 1.5 and len(g) > 1:
            for q in ("P5", "P50", "P95"):
                v = g[f"lambda_SU_{q}"]
                spread = float(v.max() - v.min())
                rel = spread / float(v.mean()) if v.mean() != 0 else np.nan
                srows.append(dict(kind="percentile spread", site=s, policy=pol, quantity=f"lambda_SU {q}", n_seeds=len(g), mean=float(v.mean()), max_minus_min=spread, rel_spread=rel,
                                  passes=bool(rel <= 0.01 or spread <= 0.01), rule="(max-min)/mean <= 1 % or (max-min) <= 0.01 pp (D49)"))
    # regenerate the D49 table (reference years, A and C, stored Sobol samples, all rows and PV ratio 1.45-1.55)
    for s in SITES:
        for pol in ("A", "C", "B60"):
            P = []
            for sd in SEEDS:
                f = UQ / f"plant_{s}_{pol}_seed{sd}_a.npz"
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=True)
                names = [str(n) for n in z["names"]]
                pvv = z["X"][:, names.index("PV_RATIO")]
                lam = z["Y"][:, 0]
                P.append([*pct(lam), *pct(lam[(pvv >= 1.45) & (pvv <= 1.55)])])
            if P:
                P = np.array(P)
                m = P.mean(axis=0)
                rows.append(dict(source="Sobol design rows (PV ratio 1.0-2.0 inside the percentiles; reference year; D49 table)", site=s, policy=pol, pv_ratio="1.0-2.0 (all rows)", n_seeds=len(P), n_years=1,
                                 lambda_SU_P5=m[0], lambda_SU_P50=m[1], lambda_SU_P95=m[2], SC_M_class=klass(m[0], m[2]), P5_at_pv_1p45_1p55=m[3]))
    pd.DataFrame(rows).to_csv(UQ / "scm_verdict.csv", index=False, float_format="%.6g")
    pd.DataFrame(srows).to_csv(UQ / "scs_percentile_spread.csv", index=False, float_format="%.6g")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    if srows:
        sr = pd.DataFrame(srows)
        print("percentile spread: passes", int(sr.passes.sum()), "of", len(sr))


def sobol():
    from SALib.analyze import sobol as sa
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sobol_pipeline import plant_problem
    prob = plant_problem("a")
    rows = []
    for s in SITES:
        for pol in ("A", "B60", "C"):
            Ys, per_seed = [], []
            for sd in SEEDS:
                f = UQ / f"plant_{s}_{pol}_seed{sd}_a.npz"
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=True)
                Ys.append(z["Y"][:, 0])
                r = sa.analyze(prob, Ys[-1], calc_second_order=False, num_resamples=1000, conf_level=0.95, seed=sd)
                per_seed.append(r)
                for i, nme in enumerate(prob["names"]):
                    rows.append(dict(site=s, policy=pol, estimator=f"single seed {sd}", factor=nme, S1=r["S1"][i], S1_conf=r["S1_conf"][i], ST=r["ST"][i], ST_conf=r["ST_conf"][i], n_seeds=1))
            if len(Ys) >= 2:
                Yp = np.concatenate(Ys)
                r = sa.analyze(prob, Yp, calc_second_order=False, num_resamples=1000, conf_level=0.95, seed=1)
                seed_sd = np.array([x["ST"] for x in per_seed]).std(axis=0, ddof=1)
                for i, nme in enumerate(prob["names"]):
                    rows.append(dict(site=s, policy=pol, estimator="pooled (Saltelli blocks of all seeds concatenated)", factor=nme, S1=r["S1"][i], S1_conf=r["S1_conf"][i], ST=r["ST"][i],
                                     ST_conf=r["ST_conf"][i], n_seeds=len(Ys), ST_seed_sd=seed_sd[i]))
    d = pd.DataFrame(rows)
    if d.empty:
        return
    d.to_csv(UQ / "sobol_pooled.csv", index=False, float_format="%.6g")
    pool = d[d.estimator.str.startswith("pooled")]
    big = pool[pool.ST >= 0.1]
    sing = d[d.estimator.str.startswith("single") & d.set_index(["site", "policy", "factor"]).index.isin(big.set_index(["site", "policy", "factor"]).index)]
    s = pd.DataFrame(dict(n_ST_ge_0p1_pooled=[len(big)], max_halfwidth_pooled=[big.ST_conf.max() if len(big) else np.nan], n_pooled_halfwidth_gt_0p05=[int((big.ST_conf > 0.05).sum())],
                          n_single_seed_estimates=[len(sing)], mean_halfwidth_single=[sing.ST_conf.mean() if len(sing) else np.nan], n_single_halfwidth_gt_0p05=[int((sing.ST_conf > 0.05).sum())],
                          max_seed_sd_ST_pooled_factors=[big.ST_seed_sd.max() if len(big) else np.nan]))
    s.to_csv(UQ / "scs_verdict.csv", index=False, float_format="%.6g")
    print(s.to_string(index=False))
    print(pool[pool.ST >= 0.1].sort_values(["site", "policy", "ST"], ascending=[True, True, False]).round(3).to_string(index=False))


# ----------------------------------------------------------------------------------------------------------------------------------- beta
LEV_P = np.linspace(0.10, 0.191, 11)
LEV_PV = np.linspace(1.0, 2.0, 11)
LEV2_P = np.linspace(0.10, 0.191, 21)           # 2-D table (item 7d): 11 x 11 levels gave a 2.1e-3 maximum interpolation error at Gobabeb (> 1e-3), so 21 x 21 are used
LEV2_PV = np.linspace(1.0, 2.0, 21)


def bep_pmin(site, year, pdc0_kw, pmin):
    import io
    txt = (RESULTS / "data" / "pvgis" / f"{site}_{year}.csv").read_text(encoding="utf-8")
    df = pd.read_csv(io.StringIO(txt))
    p = np.minimum(df["P"].to_numpy(float) / 1000.0 * pdc0_kw, 1000.0)
    on = p >= pmin * 1000.0
    return float(np.where(on, p, 0.0).sum()) / param("bep_sec_net")


def baselines_direct(site, year, pdc, pmin, pv):
    """B-CL (1 min), B-EP, B-S1 at P_min fraction pmin and PV scale pv/1.5 (direct evaluation)."""
    x = pdc * (pv / 1.5)
    eta, rec, lhv = param("c3_fixed_efficiency") / 100.0, param("bcl_recovery_primary") / 100.0, param("c3_LHV") * 1000.0
    xc = np.minimum(x, 1000.0)
    on = xc >= pmin * 1000.0
    cl = float((np.where(on, eta * xc / lhv / 60.0, 0.0) * rec).sum())
    return dict(BCL=cl, BEP=bep_pmin(site, year, 1500.0 * pv / 1.5, pmin), BS1=bs1_annual(x, pmin_frac=pmin)["M_del"])


def tables(site, year, two_d):
    pdc = pv_dc_series(site, year).p_dc_kw.to_numpy()
    if two_d:
        T = {k: np.zeros((len(LEV2_P), len(LEV2_PV))) for k in ("BCL", "BEP", "BS1")}
        for i, pm in enumerate(LEV2_P):
            for j, pv in enumerate(LEV2_PV):
                b = baselines_direct(site, year, pdc, pm, pv)
                for k in T:
                    T[k][i, j] = b[k]
    else:
        T = {k: np.zeros(len(LEV_P)) for k in ("BCL", "BEP", "BS1")}
        for i, pm in enumerate(LEV_P):
            b = baselines_direct(site, year, pdc, pm, 1.5)
            for k in T:
                T[k][i] = b[k]
    return pdc, T


def interp1(T, pm):
    return np.interp(pm, LEV_P, T)


def interp2(T, pm, pv):
    """Bilinear interpolation on (LEV_P, LEV_PV)."""
    from scipy.interpolate import RegularGridInterpolator
    return RegularGridInterpolator((LEV2_P, LEV2_PV), T)(np.column_stack([pm, pv]))


BETA_BASELINES = ["B-CL", "B-EP", "B-S1", "B-S2 curve1", "B-S2 curve2", "B-S2 curve3", "B-S3"]
BETA_ESTIMATOR = "per seed: all site-years of the site concatenated (equal year weight), beta per row, P5/P50/P95; pooled value = mean of the per-seed percentiles (D49, RR1)"
BETA_LABEL = {"B-S2 curve1": "M_del,B-S2 at nominal; interval from the full-model denominator only", "B-S2 curve2": "M_del,B-S2 at nominal; interval from the full-model denominator only",
              "B-S2 curve3": "M_del,B-S2 at nominal; interval from the full-model denominator only", "B-EP": "auxiliaries from an auxiliary supply",
              "B-CL": "baseline evaluated at the row's P_MIN_F against the row's M_del", "B-S1": "baseline evaluated at the row's P_MIN_F against the row's M_del",
              "B-S3": "B-S3 re-run on the same sample row (same s3_config)"}


def s3_column(key, df):
    """B-S3 delivered H2 of one ensemble set: seed 11 from the ens checkpoint (column M_del_S3), seeds 22-55 from s3_<...>.npz (Brief 7 item 2); None when missing."""
    s, y, pol, sd, pv = key
    if sd == 11:
        return df.M_del_S3.to_numpy() if "M_del_S3" in df and not df.M_del_S3.isna().any() else None
    f = UQ / f"s3_{s}_{y}_{pol}_seed{sd}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    if not np.array_equal(z["X"], df[[str(n) for n in z["names"]]].to_numpy()):
        raise RuntimeError(f"sample matrix mismatch {f.name}")
    return z["M_del_S3"]


def beta_pool(rows_by_seed):
    """rows_by_seed: dict seed -> {baseline: list of per-year beta arrays}. Percentiles per seed on the concatenation (before the percentile), then the mean over seeds."""
    out = {}
    for sd, bd in rows_by_seed.items():
        for nm, lst in bd.items():
            v = np.concatenate(lst)
            out.setdefault(nm, {})[sd] = (np.percentile(v, 5), np.percentile(v, 50), np.percentile(v, 95), len(v), len(lst))
    return out


def beta_all():
    ens = load_ens()
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    tabs = {}
    rng = np.random.default_rng(2024)
    chk, sy_rows, seed_rows = [], [], []
    bucket = {}                                           # (site, pol) -> seed -> baseline -> list of arrays
    n_years = {s: len({y for (ss, y, *_r) in ens if ss == s}) for s in SITES}
    for (s, y, pol, sd, pv), df in sorted(ens.items()):
        if pv != 1.5:
            continue
        if (s, y) not in tabs:
            tabs[(s, y)] = tables(s, y, False)
        pdc, T = tabs[(s, y)]
        pm = df.P_MIN_F.to_numpy()
        md = df.M_del.to_numpy()
        b = {"B-CL": interp1(T["BCL"], pm), "B-EP": interp1(T["BEP"], pm), "B-S1": interp1(T["BS1"], pm)}
        a = ann[(ann.site == s) & (ann.year == y) & (ann.policy == (pol if pol != "B60" else "A"))]      # B-S2 is policy-independent in annual.csv (A = C = D)
        for k in (1, 2, 3):
            b[f"B-S2 curve{k}"] = np.full(len(df), float(a[a.model == f"B-S2_curve{k}"].M_del.iloc[0]))
        m3 = s3_column((s, y, pol, sd, pv), df)
        if m3 is not None:
            b["B-S3"] = m3
        for nm, M in b.items():
            beta = (M - md) / md * 100.0
            bucket.setdefault((s, pol), {}).setdefault(sd, {}).setdefault(nm, []).append(beta)
            if sd == 11:
                sy_rows.append(dict(site=s, year=y, policy=pol, baseline=nm, beta_P5=np.percentile(beta, 5), beta_P50=np.percentile(beta, 50), beta_P95=np.percentile(beta, 95), beta_mean=beta.mean(), n=len(beta)))
        if pol == "A" and sd == 11:
            idx = rng.choice(len(df), 50 if (s, y) == ("Cabauw", 2013) else 5, replace=False)
            for i in idx:
                direct = baselines_direct(s, y, pdc, pm[i], 1.5)
                for k, kk in (("BCL", "B-CL"), ("BEP", "B-EP"), ("BS1", "B-S1")):
                    chk.append(dict(seed=sd, site=s, year=y, sample=int(i), P_MIN_F=pm[i], baseline=kk, interpolated=b[kk][i], direct=direct[k], rel_diff=abs(b[kk][i] - direct[k]) / direct[k]))
    # table check extension (Brief 7 item 1): 50 direct evaluations on seeds 22-55 (POL-A, random site-year, seed, row)
    rng2 = np.random.default_rng(2026)
    keys = [k for k in sorted(ens) if k[2] == "A" and k[4] == 1.5 and k[3] != 11]
    for j in rng2.choice(len(keys), 50, replace=True):
        s, y, pol, sd, pv = keys[j]
        pdc, T = tabs[(s, y)]
        df = ens[keys[j]]
        i = int(rng2.integers(len(df)))
        pmi = float(df.P_MIN_F.iloc[i])
        direct = baselines_direct(s, y, pdc, pmi, 1.5)
        for k, kk in (("BCL", "B-CL"), ("BEP", "B-EP"), ("BS1", "B-S1")):
            ip = float(interp1(T[k], np.array([pmi]))[0])
            chk.append(dict(seed=sd, site=s, year=y, sample=i, P_MIN_F=pmi, baseline=kk, interpolated=ip, direct=direct[k], rel_diff=abs(ip - direct[k]) / direct[k]))
    rows = []
    for (s, pol), per_seed in sorted(bucket.items()):
        full = {sd: bd for sd, bd in per_seed.items() if all(len(bd.get(nm, [])) == n_years[s] for nm in ("B-CL",))}
        for nm in BETA_BASELINES:
            have = {sd: bd for sd, bd in full.items() if len(bd.get(nm, [])) == n_years[s]}
            if not have:
                continue
            res = beta_pool({sd: {nm: bd[nm]} for sd, bd in have.items()})[nm]
            P = np.array([[v[0], v[1], v[2]] for v in res.values()])
            rows.append(dict(site=s, policy=pol, baseline=nm, n_seeds=len(res), n_years=n_years[s], estimator=BETA_ESTIMATOR, label=BETA_LABEL[nm],
                             beta_P5=P[:, 0].mean(), beta_P50=P[:, 1].mean(), beta_P95=P[:, 2].mean()))
            sr = dict(site=s, policy=pol, baseline=nm, n_seeds=len(res), seeds=" ".join(str(k) for k in res))
            for sd, v in res.items():
                sr.update({f"P5_seed{sd}": v[0], f"P50_seed{sd}": v[1], f"P95_seed{sd}": v[2]})
            sr.update(spread_P5=P[:, 0].max() - P[:, 0].min(), spread_P50=P[:, 1].max() - P[:, 1].min(), spread_P95=P[:, 2].max() - P[:, 2].min())
            seed_rows.append(sr)
    g = pd.DataFrame(rows)
    if not g.empty:
        pd.DataFrame(sy_rows).to_csv(UQ / "beta_intervals_site_year.csv", index=False, float_format="%.6g")
        g.to_csv(UQ / "beta_intervals.csv", index=False, float_format="%.6g")
        pd.DataFrame(seed_rows).to_csv(UQ / "beta_seed_spread.csv", index=False, float_format="%.6g")
        print(g.drop(columns=["estimator", "label"]).round(2).to_string(index=False))
    if chk:
        c = pd.DataFrame(chk)
        c.to_csv(UQ / "beta_table_check.csv", index=False, float_format="%.6g")
        print("1-D table check: n", len(c), "max rel diff", c.rel_diff.max(), "| seeds 22-55 part: n", int((c.seed != 11).sum()), "max", c[c.seed != 11].rel_diff.max())


def sobol_beta():
    from SALib.analyze import sobol as sa
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sobol_pipeline import plant_problem
    prob = plant_problem("a")
    names = prob["names"]
    rows, chk = [], []
    rng = np.random.default_rng(2025)
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    for s in SITES:
        year = next(y for ss_, y in site_years() if ss_ == s)
        tabs = None
        for pol in ("A", "B60", "C"):
            fs = [UQ / f"plant_{s}_{pol}_seed{sd}_a.npz" for sd in SEEDS if (UQ / f"plant_{s}_{pol}_seed{sd}_a.npz").exists()]
            if not fs:
                continue
            if tabs is None:
                tabs = tables(s, year, True)
            pdc, T = tabs
            Yb = {"B-CL": [], "B-S1": []}
            for f in fs:
                z = np.load(f, allow_pickle=True)
                X = z["X"]
                pm, pv = X[:, names.index("P_MIN_F")], X[:, names.index("PV_RATIO")]
                md = z["Y"][:, 1]
                for k, kk in (("BCL", "B-CL"), ("BS1", "B-S1")):
                    Yb[kk].append((interp2(T[k], pm, pv) - md) / md * 100.0)
                if f == fs[0] and pol == "A":
                    for i in rng.choice(len(X), 50, replace=False):
                        dd = baselines_direct(s, year, pdc, pm[i], pv[i])
                        for k, kk in (("BCL", "B-CL"), ("BS1", "B-S1")):
                            chk.append(dict(site=s, policy=pol, sample=int(i), P_MIN_F=pm[i], PV_RATIO=pv[i], baseline=kk, interpolated=float(interp2(T[k], pm[i:i + 1], pv[i:i + 1])[0]), direct=dd[k],
                                            rel_diff=abs(float(interp2(T[k], pm[i:i + 1], pv[i:i + 1])[0]) - dd[k]) / dd[k]))
            for kk, lst in Yb.items():
                Yp = np.concatenate(lst)
                r = sa.analyze(prob, Yp, calc_second_order=False, num_resamples=1000, conf_level=0.95, seed=1)
                for i, nme in enumerate(names):
                    rows.append(dict(site=s, year=year, policy=pol, output=f"beta_{kk}", factor=nme, S1=r["S1"][i], S1_conf=r["S1_conf"][i], ST=r["ST"][i], ST_conf=r["ST_conf"][i], n_seeds=len(lst),
                                     output_mean=Yp.mean(), output_sd=Yp.std()))
    if rows:
        d = pd.DataFrame(rows)
        d.to_csv(UQ / "sobol_beta.csv", index=False, float_format="%.6g")
        print(d[d.ST >= 0.1].round(3).to_string(index=False))
    if chk:
        c = pd.DataFrame(chk)
        c.to_csv(UQ / "sobol_beta_table_check.csv", index=False, float_format="%.6g")
        print("2-D table check: n", len(c), "max rel diff", c.rel_diff.max())


def main(what="all"):
    if what in ("intervals", "scm", "all"):
        scm_scs()
    if what in ("sobol", "all"):
        sobol()
    if what in ("beta", "all"):
        beta_all()
        sobol_beta()
    if what == "sobolbeta":
        sobol_beta()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")

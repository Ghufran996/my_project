"""Brief 5 item 3 (D51, H2): event structure of lambda_SU. Replaces the F5 daily regression as the evidence for the intra-day question.

    python code/event_structure.py
      -> results/nominal/lambda_by_event_class.csv   lambda_SU split into diurnal completed cycle, intra-day completed cycle, aborted attempts (+ open year-end window)
         results/nominal/binned_day_ratio.csv        site-days binned by intra-day attempts: ratio of sums, n days, share of annual m_SU
         results/nominal/site_days.csv               per-day table (POL-A, POL-C, 17 site-years)
         results/nominal/intraday_slope.csv          PL5 pre-registered slope (descriptive): OLS, R2, 95 % CI clustered by site-year

Cycle classes (D61c, pre-registered PL8): "diurnal" = a completed cycle whose closing START attempt is the first restart of the day after sunrise (engine flag, zenith < 90 deg); "intra-day" = every other
completed cycle; aborted attempts and the open year-end window are separate classes. These are the PRIMARY columns (m_SU_diurnal_completed_kg, m_SU_intraday_completed_kg, share_*, pp_*, n_diurnal, n_intraday).
The night-spanning split (a completed cycle whose window, stop to restart, contains a night, zenith >= 90 deg; the PL4 cycle definition) was chosen after the result and is reported second, labelled post hoc:
columns *_window_post_hoc. The aborted share is identical in both splits.
Negative shares (D61g): under POL-C the aborted-attempt share can be slightly negative (-1.5, -0.5, -0.01 %). Not a bug: m_SU,c is the excess over the steady reference (plant.py header, PL4), so a window
can be slightly negative; the `note` column of lambda_by_event_class.csv states this for POL-C.
Per-day quantities (binned_day_ratio.csv): m_SU per day (kg/d and rated-minutes/d = kg/d divided by the rated H2 production per minute of the plant, PL4) and M_stack per day (kg/d) are the means over the days in the bin.
Conventions: a day is a block of 1440 one-minute steps from the start of the series (UTC). A cycle window's net loss (m_SU) is assigned to the day in which its restart
closes (the day of the last step of the window); aborted-attempt losses carry the window they belong to. Intra-day attempts = START entries of the day minus the first
START entry after sunrise (zenith < 90 deg), floored at 0 (the engine's own diurnal/intra-day rule, plant.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2su import plant as pl  # noqa: E402
from h2su import plantrun as pr  # noqa: E402
from h2su.io import RESULTS  # noqa: E402
from h2su.pv import pv_dc_series  # noqa: E402
from sweeps import site_years  # noqa: E402

BINS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 8, "6-8"), (9, 10 ** 6, ">=9")]


def per_day(s, y, pol):
    df = pv_dc_series(s, y)
    par = pr.build_params(pol)
    tot, cyc, nc, tr = pr.run(df, par, pr.switches(), trace=True)
    n = len(df)
    nd = int(np.ceil(n / 1440))
    day = np.arange(n) // 1440
    ms = np.bincount(day, weights=tr[:, 2], minlength=nd)[:nd]
    md = np.bincount(day, weights=tr[:, 3], minlength=nd)[:nd]
    st = tr[:, 0]
    k_start = np.flatnonzero(np.diff((st == pl.START).astype(int)) == 1) + 1
    if st[0] == pl.START:
        k_start = np.concatenate([[0], k_start])
    n_att = np.bincount(k_start // 1440, minlength=nd)[:nd].astype(float)
    zen = df.zenith.to_numpy()
    first_diurnal = np.zeros(nd)
    seen = set()
    for k in k_start:
        dd = int(k // 1440)
        if dd not in seen and zen[k] < 90.0:
            first_diurnal[dd] = 1
            seen.add(dd)
    c = cyc[:nc]
    dclose = np.minimum((c[:, 1].astype(int) - 1) // 1440, nd - 1)
    msu = c[:, 11] + c[:, 12]
    msu_d = np.bincount(dclose, weights=msu, minlength=nd)[:nd]
    ncl_d = np.bincount(dclose, weights=(c[:, 10] > 0.5).astype(float), minlength=nd)[:nd]
    d = pd.DataFrame(dict(site=s, year=y, policy=pol, day=np.arange(1, nd + 1), N_attempts=n_att, N_intra=np.maximum(n_att - first_diurnal, 0), M_stack_kg=ms, M_del_kg=md, m_SU_kg=msu_d,
                          n_cycles_closed=ncl_d))
    # lambda_SU split
    closed = c[:, 10] > 0.5
    dia_flag = c[:, 6] > 0.5                                           # PRIMARY (PL8): the closing START attempt is the first of the day after sunrise (engine flag)
    cs = np.concatenate([[0], np.cumsum(zen >= 90.0)])                  # night minutes before index k
    i0 = c[:, 0].astype(int)
    i1 = np.minimum(c[:, 1].astype(int), n)
    dia = (cs[i1] - cs[np.minimum(i0, n)]) > 0                          # POST HOC (chosen after the result): the cycle window spans a night (zenith >= 90 deg), i.e. it contains the daily stop
    split = dict(site=s, year=y, policy=pol, M_stack_kg=tot[pl.T_STACK], m_SU_total_kg=float(msu.sum()), m_SU_diurnal_completed_kg=float(c[closed & dia_flag, 12].sum()),
                 m_SU_intraday_completed_kg=float(c[closed & ~dia_flag, 12].sum()), m_SU_aborted_attempts_kg=float(c[:, 11].sum()), m_SU_open_window_year_end_kg=float(c[~closed, 12].sum()),
                 m_SU_diurnal_completed_window_post_hoc_kg=float(c[closed & dia, 12].sum()), m_SU_intraday_completed_window_post_hoc_kg=float(c[closed & ~dia, 12].sum()),
                 n_cycles_closed=int(closed.sum()), n_diurnal=int((closed & dia_flag).sum()), n_intraday=int((closed & ~dia_flag).sum()),
                 n_diurnal_window_post_hoc=int((closed & dia).sum()), n_intraday_window_post_hoc=int((closed & ~dia).sum()), n_aborts=int(tot[pl.T_NABORT]), mr_kg_min=float(par[pl.MRATED_MIN]))
    return d, split


def cluster_ols(x, y, g):
    """OLS y = a + b x with a CR1 cluster-robust covariance (clusters g); returns slope, se, 95 % half-width (t, G-1 d.o.f.), R2, intercept."""
    X = np.column_stack([np.ones_like(x), x])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    u = y - X @ beta
    G = len(np.unique(g))
    meat = np.zeros((2, 2))
    for k in np.unique(g):
        m = g == k
        sg = X[m].T @ u[m]
        meat += np.outer(sg, sg)
    n, kk = len(y), 2
    V = XtXi @ meat @ XtXi * (G / (G - 1)) * ((n - 1) / (n - kk))
    se = float(np.sqrt(V[1, 1]))
    r2 = 1.0 - float((u ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(beta[1]), se, float(stats.t.ppf(0.975, G - 1) * se), r2, float(beta[0]), G


def main():
    days, splits = [], []
    for s, y in site_years():
        for pol in "AC":
            d, sp = per_day(s, y, pol)
            days.append(d)
            splits.append(sp)
        print(s, y, flush=True)
    days = pd.concat(days, ignore_index=True)
    sp = pd.DataFrame(splits)
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    ann = ann[ann.model == "full"].set_index(["site", "year", "policy"])
    sp["lambda_SU_pct"] = sp.m_SU_total_kg / sp.M_stack_kg * 100
    # identity checks
    sp["identity_classes_minus_total_kg"] = sp.m_SU_diurnal_completed_kg + sp.m_SU_intraday_completed_kg + sp.m_SU_aborted_attempts_kg + sp.m_SU_open_window_year_end_kg - sp.m_SU_total_kg
    sp["lambda_SU_engine_pct"] = [ann.loc[(r.site, r.year, r.policy), "lambda_SU"] for r in sp.itertuples()]
    sp["lambda_check_rel"] = (sp.lambda_SU_pct - sp.lambda_SU_engine_pct).abs() / sp.lambda_SU_engine_pct.abs()
    for k in ("diurnal_completed", "intraday_completed", "aborted_attempts", "open_window_year_end"):
        sp[f"share_{k}"] = sp[f"m_SU_{k}_kg"] / sp.m_SU_total_kg
        sp[f"pp_{k}"] = sp[f"m_SU_{k}_kg"] / sp.M_stack_kg * 100
    sp["share_sum_minus_1"] = sp[[f"share_{k}" for k in ("diurnal_completed", "intraday_completed", "aborted_attempts", "open_window_year_end")]].sum(axis=1) - 1
    for k in ("diurnal_completed_window_post_hoc", "intraday_completed_window_post_hoc"):
        sp[f"share_{k}"] = sp[f"m_SU_{k}_kg"] / sp.m_SU_total_kg
        sp[f"pp_{k}"] = sp[f"m_SU_{k}_kg"] / sp.M_stack_kg * 100
    out = RESULTS / "nominal"
    sp.to_csv(out / "lambda_by_event_class_site_year.csv", index=False, float_format="%.9g")
    # site x policy summary: pooled kg shares and mean pp over years
    rows = []
    for (s, pol), g in sp.groupby(["site", "policy"]):
        tot_kg = g.m_SU_total_kg.sum()
        r = dict(site=s, policy=pol, n_years=len(g), lambda_SU_mean_pct=g.lambda_SU_pct.mean(), n_diurnal_mean=g.n_diurnal.mean(), n_intraday_mean=g.n_intraday.mean(), n_aborts_mean=g.n_aborts.mean(),
                 n_diurnal_window_post_hoc_mean=g.n_diurnal_window_post_hoc.mean(), n_intraday_window_post_hoc_mean=g.n_intraday_window_post_hoc.mean())
        for k in ("diurnal_completed", "intraday_completed", "aborted_attempts", "open_window_year_end"):
            r[f"share_{k}"] = g[f"m_SU_{k}_kg"].sum() / tot_kg
            r[f"pp_{k}"] = g[f"pp_{k}"].mean()
        for k in ("diurnal_completed_window_post_hoc", "intraday_completed_window_post_hoc"):
            r[f"share_{k}"] = g[f"m_SU_{k}_kg"].sum() / tot_kg
            r[f"pp_{k}"] = g[f"pp_{k}"].mean()
        r["share_sum_minus_1"] = sum(r[f"share_{k}"] for k in ("diurnal_completed", "intraday_completed", "aborted_attempts", "open_window_year_end")) - 1
        r["max_identity_classes_minus_total_kg"] = g.identity_classes_minus_total_kg.abs().max()
        r["max_lambda_check_rel_vs_engine"] = g.lambda_check_rel.max()
        r["note"] = ("primary split = first restart after sunrise (PL8, D61c); *_window_post_hoc = night-spanning window split chosen after the result"
                     + ("; POL-C aborted share can be negative: m_SU,c is the excess over the steady reference (D61g)" if pol == "C" else ""))
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "lambda_by_event_class.csv", index=False, float_format="%.9g")
    days.to_csv(out / "site_days.csv", index=False, float_format="%.9g")
    # binned ratio of sums
    brow = []
    for (s, pol), g in days.groupby(["site", "policy"]):
        tot_msu = g.m_SU_kg.sum()
        mr = float(sp[(sp.site == s) & (sp.policy == pol)].mr_kg_min.iloc[0])
        noprod = g.M_stack_kg <= 1.0
        for lo, hi, nm in BINS:
            b = g[(~noprod) & (g.N_intra >= lo) & (g.N_intra <= hi)]
            brow.append(dict(site=s, policy=pol, bin_intra_day_attempts=nm, n_days=len(b), sum_m_SU_kg=b.m_SU_kg.sum(), sum_M_stack_kg=b.M_stack_kg.sum(),
                             ratio_lambda_SU_pct=b.m_SU_kg.sum() / b.M_stack_kg.sum() * 100 if b.M_stack_kg.sum() > 0 else np.nan,
                             eta_del=b.M_del_kg.sum() / b.M_stack_kg.sum() if b.M_stack_kg.sum() > 0 else np.nan, share_of_annual_m_SU=b.m_SU_kg.sum() / tot_msu if tot_msu != 0 else np.nan,
                             share_of_days=len(b) / len(g), m_SU_per_day_kg=b.m_SU_kg.sum() / len(b) if len(b) else np.nan,
                             m_SU_per_day_rated_min=b.m_SU_kg.sum() / len(b) / mr if len(b) else np.nan, M_stack_per_day_kg=b.M_stack_kg.sum() / len(b) if len(b) else np.nan))
        b = g[noprod]
        brow.append(dict(site=s, policy=pol, bin_intra_day_attempts="no stack production (M_stack <= 1 kg)", n_days=len(b), sum_m_SU_kg=b.m_SU_kg.sum(), sum_M_stack_kg=b.M_stack_kg.sum(),
                         ratio_lambda_SU_pct=np.nan, eta_del=np.nan, share_of_annual_m_SU=b.m_SU_kg.sum() / tot_msu if tot_msu != 0 else np.nan, share_of_days=len(b) / len(g),
                         m_SU_per_day_kg=b.m_SU_kg.sum() / len(b) if len(b) else np.nan, m_SU_per_day_rated_min=b.m_SU_kg.sum() / len(b) / mr if len(b) else np.nan,
                         M_stack_per_day_kg=b.M_stack_kg.sum() / len(b) if len(b) else np.nan))
    bd = pd.DataFrame(brow)
    chk = bd.groupby(["site", "policy"]).share_of_annual_m_SU.sum()
    bd.to_csv(out / "binned_day_ratio.csv", index=False, float_format="%.6g")
    # pre-registered slope (descriptive), days with >= 1 closed cycle for lambda, M_stack > 1 kg for eta_del
    srow = []
    for (s, pol), g in days.groupby(["site", "policy"]):
        for nm, sel, yv in (("lambda_SU_day_pct", (g.n_cycles_closed > 0) & (g.M_stack_kg > 1.0), g.m_SU_kg / g.M_stack_kg * 100), ("eta_del_day", g.M_stack_kg > 1.0, g.M_del_kg / g.M_stack_kg)):
            gg = g[sel]
            slope, se, hw, r2, icpt, G = cluster_ols(gg.N_intra.to_numpy(float), yv[sel].to_numpy(float), (gg.site + gg.year.astype(str)).to_numpy())
            srow.append(dict(site=s, policy=pol, output=nm, n_days=len(gg), n_clusters=G, slope_per_intraday_attempt=slope, slope_ci95_halfwidth_clustered=hw, intercept=icpt, R2=r2,
                             note="descriptive; daily ratio, small-denominator days included (M_stack > 1 kg); CI clustered by site-year (CR1, t with G-1 d.o.f.)"))
    pd.DataFrame(srow).to_csv(out / "intraday_slope.csv", index=False, float_format="%.6g")
    print(pd.read_csv(out / "lambda_by_event_class.csv").round(4).to_string())
    print("share sums (bins incl. no-production):", chk.round(9).to_dict())
    print(bd[bd.policy == "A"].round(3).to_string())
    print(pd.DataFrame(srow).round(4).to_string())


if __name__ == "__main__":
    main()

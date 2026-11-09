"""Side brief S item 3 (D72b): load shape behind the C3 efficiency-convention term; closed form tested against stored terms.

    python code/c3_load_shape.py   ->   results/resolution/c3_load_shape.csv, c3_load_shape_summary.csv

Load L = min(P/P_cap, 1) from 5-min block means of the plant DC power (results/data/pv_dc/<site>_<year>.parquet, 1.5 MWp, P_cap = ratio x 1.5 MW, ratios 1.0, 0.75, 0.667;
base = PV DC peak) and from the three AEMO 2022 farms (5-min SCADA values, P_cap = Table 2 rated capacity; base = rated capacity). Energy is counted for L >= 0.1.
x = sum(max(L - 0.3, 0)) / sum(L); energy-weighted mean load = sum(L^2) / sum(L). Closed form (curve k): term = -(1 - s_k / 0.75) x 100 [%], s_k = (eta_k(1) - 0.225) / 0.7
(eta_k(1) = eta_min of curve k, results of L4 Fig. 2). Stored term: plant rows = d_vs_fixed75_5min_pct of results/resolution/c3_comparable.csv (same ratio, curve k);
farm rows = L4 Table 3 published value (ratio 1). plant_site_pooled rows (Brief 9 item 1, D77): stored term = 100 x (sum over the site's years of H2_5min curve k / sum of H2_5min fixed75 - 1)
at the same ratio (like for like with the pooled x); plant_site_mean rows keep the mean of the yearly terms. Column stored_basis names the convention. No fitting; curves 2 and 3 are reported as they come out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2su.baselines import bs2_c3 as b2  # noqa: E402
from h2su.io import CONFIGS, RESULTS  # noqa: E402
from sweeps import site_years  # noqa: E402

FARMS = {"Finley": "FINLYSF1", "Tailem Bend": "TBSF1", "Hamilton": "HAMISF1"}
RATIOS = (1.0, 0.75, 2.0 / 3.0)
BASE_PV, BASE_FARM = "PV DC peak (1.5 MWp)", "rated capacity (L4 Table 2)"


def shape(L: np.ndarray):
    m = L >= 0.1
    s = L[m].sum()
    return float(np.maximum(L[m] - 0.3, 0.0).sum() / s), float((L[m] ** 2).sum() / s)


STORED_BASIS = {"plant": "site-year term (c3_comparable.csv)", "farm": "L4 Table 3 published value", "plant_site_mean": "mean of the yearly terms",
                "plant_site_pooled": "pooled sums over site-years: 100 x (sum H2 curve k / sum H2 fixed75 - 1)"}


def pooled_stored_term(sub: pd.DataFrame, k: int) -> float:
    """Pooled like-for-like term of curve k [%]: 100 x (sum over the years of H2_5min curve k / sum of H2_5min fixed75 - 1); sub = c3_comparable rows of one site and ratio."""
    num = sub[sub.scenario == f"curve{k}"].H2_5min_kg.sum()
    den = sub[sub.scenario == "fixed75"].H2_5min_kg.sum()
    return float(100.0 * (num / den - 1.0))


def main():
    tab = pd.read_csv(CONFIGS / "tables" / "L4_virahsawmy2024.csv")
    e = tab[tab.table == "Fig2"]
    emin = {int(r.curve): float(r.value) for r in e[e.item == "eta_min"].itertuples()}
    sk = {k: (emin[k] - 0.225) / 0.7 for k in (1, 2, 3)}
    cc = pd.read_csv(RESULTS / "resolution" / "c3_comparable.csv")
    rows = []
    pooled = {}
    for s, y in site_years():
        p = pd.read_parquet(RESULTS / "data" / "pv_dc" / f"{s}_{y}.parquet").p_dc_kw.to_numpy() / 1000.0   # MW, 1.5 MWp
        p5 = b2.block_mean(p, 5)
        for r in RATIOS:
            L = np.minimum(p5 / (1.5 * r), 1.0)
            pooled.setdefault((s, round(r, 4)), []).append(L)
            x, lm = shape(L)
            d = dict(kind="plant", site=s, year=y, ratio=round(r, 4), base=BASE_PV, x_share_above_0p3=x, mean_load_energy_weighted=lm)
            for k in (1, 2, 3):
                cf = -(1.0 - sk[k] / 0.75) * x * 100.0
                st = cc[(cc.site == s) & (cc.year == y) & (cc.scenario == f"curve{k}") & np.isclose(cc.electrolyser_over_pv_ratio, r)].d_vs_fixed75_5min_pct
                assert len(st) == 1
                d.update({f"closed_form_curve{k}_pct": cf, f"stored_curve{k}_pct": float(st.iloc[0]), f"diff_curve{k}_pp": cf - float(st.iloc[0]),
                          f"stored_source_curve{k}": "results/resolution/c3_comparable.csv"})
            rows.append(d)
    for f, duid in FARMS.items():
        cap = float(tab[(tab.table == "Table2") & (tab.item == "rated_capacity") & (tab.site == f)].value.iloc[0])
        v = pd.read_parquet(RESULTS / "data" / "aemo2022" / f"{duid}_2022.parquet").SCADAVALUE.to_numpy(float)
        assert len(v) == 105120 and np.isfinite(v).all()
        x, lm = shape(np.minimum(v / cap, 1.0))
        d = dict(kind="farm", site=f, year=2022, ratio=1.0, base=BASE_FARM, x_share_above_0p3=x, mean_load_energy_weighted=lm)
        for k in (1, 2, 3):
            cf = -(1.0 - sk[k] / 0.75) * x * 100.0
            pub = float(tab[(tab.table == "Table3") & (tab.site == f) & (tab.curve.astype(str) == str(k))].value.iloc[0])
            d.update({f"closed_form_curve{k}_pct": cf, f"stored_curve{k}_pct": pub, f"diff_curve{k}_pp": cf - pub, f"stored_source_curve{k}": "L4 Table 3 (published)"})
        rows.append(d)
    df = pd.DataFrame(rows)
    m = df[df.kind == "plant"].groupby(["site", "ratio"]).mean(numeric_only=True).reset_index()
    m.insert(0, "kind", "plant_site_mean")
    m["year"] = -1
    m["base"] = BASE_PV
    pr_ = []
    for (s, r), Ls in pooled.items():       # pooled over the site-years of a site (all L concatenated); stored term = mean over the site-years
        x, lm = shape(np.concatenate(Ls))
        d = dict(kind="plant_site_pooled", site=s, year=-1, ratio=r, base=BASE_PV, x_share_above_0p3=x, mean_load_energy_weighted=lm)
        for k in (1, 2, 3):
            cf = -(1.0 - sk[k] / 0.75) * x * 100.0
            sub = cc[(cc.site == s) & np.isclose(cc.electrolyser_over_pv_ratio, r, atol=1e-3)]
            assert len(sub[sub.scenario == f"curve{k}"]) == len(Ls)
            st = pooled_stored_term(sub, k)
            d.update({f"closed_form_curve{k}_pct": cf, f"stored_curve{k}_pct": st, f"diff_curve{k}_pp": cf - st, f"stored_source_curve{k}": "results/resolution/c3_comparable.csv (pooled sums over site-years)"})
        pr_.append(d)
    out = pd.concat([df, m, pd.DataFrame(pr_)], ignore_index=True)
    out.insert(out.columns.get_loc("base") + 1, "stored_basis", out.kind.map(STORED_BASIS))
    out.to_csv(RESULTS / "resolution" / "c3_load_shape.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print("s_k:", {k: round(v, 4) for k, v in sk.items()})
    print(out[~out.kind.isin(["plant", "plant_site_mean"])][["kind", "site", "ratio", "x_share_above_0p3", "mean_load_energy_weighted"] + [f"diff_curve{k}_pp" for k in (1, 2, 3)]].round(4).to_string())
    pp_ = out[out.kind == "plant_site_pooled"]
    for k in (1, 2, 3):
        print(f"curve {k}: plant pooled-site rows max |diff| {pp_[f'diff_curve{k}_pp'].abs().max():.4f} pp")
    for k in (1, 2, 3):
        pl = df[df.kind == "plant"][f"diff_curve{k}_pp"].abs()
        fm = df[df.kind == "farm"][f"diff_curve{k}_pp"].abs()
        print(f"curve {k}: max |diff| plant site-year rows {pl.max():.4f} pp, plant site means {m[f'diff_curve{k}_pp'].abs().max():.4f} pp, farms {fm.max():.4f} pp")
    summary(out)
    return len(out)


def summary(out: pd.DataFrame):
    """Per kind and curve: n, max |diff|, count within 0.04 pp, count with |closed form| < |stored|; Spearman of x vs the curve 1 stored term (9 pooled + 3 farm rows)."""
    from scipy.stats import spearmanr
    rows = []
    for kind in ("plant", "plant_site_mean", "plant_site_pooled", "farm"):
        g = out[out.kind == kind]
        for k in (1, 2, 3):
            d = g[f"diff_curve{k}_pp"].abs()
            smaller = int((g[f"closed_form_curve{k}_pct"].abs() < g[f"stored_curve{k}_pct"].abs()).sum())
            rows.append(dict(kind=kind, curve=k, n=len(g), max_abs_diff_pp=float(d.max()), n_within_0p04_pp=int((d <= 0.04).sum()), n_closed_form_smaller_in_size=smaller,
                             stored_basis=STORED_BASIS[kind]))
    sm = pd.DataFrame(rows)
    sel = out[(out.kind == "plant_site_pooled") | (out.kind == "farm")]
    rho = spearmanr(sel.x_share_above_0p3, sel.stored_curve1_pct)[0]
    sm["spearman_x_vs_curve1_term_12_rows"] = np.nan
    sm.loc[0, "spearman_x_vs_curve1_term_12_rows"] = float(rho)
    sm.loc[0, "spearman_n_rows"] = len(sel)
    sm.to_csv(RESULTS / "resolution" / "c3_load_shape_summary.csv", index=False, float_format="%.6g")
    print(sm.round(4).to_string())
    print("Spearman x vs curve 1 stored term, rows:", len(sel), "rho", round(float(rho), 4))


if __name__ == "__main__":
    print("rows", main())

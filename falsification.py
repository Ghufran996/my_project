"""Brief 5 item 1 (D52, D47, D50a): falsification grid and break-even.

    python code/falsification.py grid        -> results/sens/falsification_grid.csv   (POL-A, POL-C, 17 site-years, theta_spec x i x P_OP full factorial)
    python code/falsification.py breakeven   -> results/sens/breakeven.csv, results/sens/falsification_verdict.txt

theta_spec in rated-minutes {0, 1, 2, 3.25, 6.5, 13, 25, 50, 100}; theta_spec = 0 -> SW_PURGE off; 0 < theta_spec < 6.5 keeps T_PURGE (purge time unchanged);
i in {0.120 (L32), 0.699 (nominal)} min/bar; P_OP in {12, 15.8, 30} bar. Labels: theta_spec < 6.5 "bounding, no source"; theta_spec = 100 "AWE-derived, L14 abstract (D47)";
P_OP = 12 "AWE threshold, PL8 low end"; P_OP = 15.8 "L25 hot-restart pressure (D50a)"; i = 0.120 "L32 end".
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2su import plant as pl  # noqa: E402
from h2su import plantrun as pr  # noqa: E402
from h2su.io import RESULTS  # noqa: E402
from sweeps import run_grid  # noqa: E402

THETA = (0.0, 1.0, 2.0, 3.25, 6.5, 13.0, 25.0, 50.0, 100.0)
INV = (0.120, 0.699)
POP = (12.0, 15.8, 30.0)


def lab_theta(t):
    if t < 6.5:
        return "bounding, no source"
    if t == 100.0:
        return "AWE-derived, L14 abstract (D47)"
    return "L28 procedural 6.5 rated-min (nominal)" if t == 6.5 else "scenario (scaling of the L28 value)"


def lab_pop(p):
    return {12.0: "AWE threshold, PL8 low end", 15.8: "L25 hot-restart pressure (D50a)", 30.0: "L13/L24 (nominal)"}[p]


def grid(workers: int = 10):
    i_nom = float(pr.build_params("A")[pl.I_INV])
    specs = []
    for pol in "AC":
        for t, i, p in itertools.product(THETA, INV, POP):
            ii = i_nom if abs(i - 0.699) < 1e-9 else i                 # 0.699 = the nominal engine value (0.6986), not a rounded copy
            specs.append(dict(name=f"th{t}_i{i}_p{p}", pol=pol, over=dict(THETA_SPEC=t, I_INV=ii, P_OP=p), off=dict(purge=0) if t == 0.0 else {},
                              tags=dict(theta_req=t, i_req=i, p_op_req=p, label_theta=lab_theta(t), label_p_op=lab_pop(p), label_i="L32 end" if i == 0.120 else "nominal (L13)")))
    df = run_grid(specs, workers=workers)
    out = RESULTS / "sens"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "falsification_grid.csv", index=False)
    print("rows", len(df), "max closure H2", df.closure_H2_rel.max(), "max closure E", df.closure_E_rel.max())
    return df


def breakeven():
    g = pd.read_csv(RESULTS / "sens" / "falsification_grid.csv")
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    f = ann[ann.model == "full"]
    mr = float(pr.build_params("A")[pl.MRATED_MIN])
    g["tau_mean_rmin"] = g.lambda_SU / 100.0 * g.M_stack / (g.N_r_completed * mr)
    rows = []
    for (s, pol, i, p), d in g.groupby(["site", "policy", "i_req", "p_op_req"]):
        m = d.groupby("theta_req")[["lambda_SU", "tau_mean_rmin", "tau_eq_median"]].mean().sort_index()
        th, lam = m.index.to_numpy(), m.lambda_SU.to_numpy()
        yrs = d.groupby("theta_req").lambda_SU.min().sort_index().to_numpy()
        thb = tau_m = tau_med = np.nan
        if lam.min() >= 1.0:
            res, note = "not reached: lambda_SU >= 1.0 % over the whole theta_spec range", "site-mean lambda_SU stays >= 1.0 % for theta_spec 0-100"
        elif lam[0] >= 1.0 and lam[-1] < 1.0:
            res, note = "not reached", "non-monotone"
        elif lam[-1] < 1.0:
            res, note = "not reached: lambda_SU < 1.0 % up to theta_spec = 100", "site-mean lambda_SU stays < 1.0 % over the whole range"
        elif lam[0] >= 1.0:
            res, note = "not reached", "unexpected"
        else:
            k = int(np.argmax(lam >= 1.0))                               # first grid point at or above 1.0 % (lam[0] < 1.0 here)
            w = (1.0 - lam[k - 1]) / (lam[k] - lam[k - 1])
            thb = th[k - 1] + w * (th[k] - th[k - 1])
            tau_m = float(np.interp(thb, th, m.tau_mean_rmin.to_numpy()))
            tau_med = float(np.interp(thb, th, m.tau_eq_median.to_numpy()))
            res, note = "reached", "linear interpolation of the site-mean lambda_SU (mean over years) between neighbouring theta_spec grid points"
        a = f[(f.site == s) & (f.policy == pol)]
        cf = 0.01 * a.M_stack.mean() / (a.N_r_completed.mean() * mr)
        rows.append(dict(site=s, policy=pol, i_min_per_bar=i, P_OP_bar=p, status=res, theta_spec_at_1pct_rmin=thb, tau_eq_mean_at_1pct_rmin=tau_m, tau_eq_median_at_1pct_rmin=tau_med,
                         lambda_SU_at_theta0=lam[0], lambda_SU_at_theta6p5=float(np.interp(6.5, th, lam)), lambda_SU_at_theta100=lam[-1], min_site_year_lambda_at_theta0=float(yrs[0]),
                         closed_form_tau_eq_1pct_rmin_D52=cf, interpolation=note, label_i="L32 end" if i == 0.120 else "nominal", label_P_OP=lab_pop(p)))
    be = pd.DataFrame(rows).sort_values(["site", "policy", "i_min_per_bar", "P_OP_bar"])
    be.to_csv(RESULTS / "sens" / "breakeven.csv", index=False, float_format="%.5g")
    gm = g.groupby(["site", "policy", "i_req", "p_op_req", "theta_req"]).lambda_SU.agg(["mean", "min", "max"]).reset_index()
    lines = []
    low = gm[gm["mean"] < 1.0]
    nom = g[(g.i_req == 0.699) & (g.p_op_req == 30.0) & (g.theta_req == 6.5)]
    ref = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    ref = ref[(ref.model == "full") & ref.policy.isin(["A", "C"])].set_index(["site", "year", "policy"])
    dif = max(abs(r.M_del - ref.loc[(r.site, r.year, r.policy), "M_del"]) / ref.loc[(r.site, r.year, r.policy), "M_del"] for r in nom.itertuples())
    lines.append(f"CHECK (0.699, 30 bar, theta 6.5) vs R46 nominal: max relative M_del difference {dif:.2e} over {len(nom)} runs")
    lines.append(f"CHECK closure: max H2 {g.closure_H2_rel.max():.2e}, max energy {g.closure_E_rel.max():.2e} (limit 1e-6)")
    if low.empty:
        lines.append("VERDICT: the site-mean lambda_SU (mean over years) is >= 1.0 % in EVERY grid cell (all sites, POL-A and POL-C): the class does not leave 'material' anywhere on this grid.")
    else:
        for (s, pol), d in low.groupby(["site", "policy"]):
            n_all = len(gm[(gm.site == s) & (gm.policy == pol)])
            cells = "; ".join(f"(i={r.i_req}, P_OP={r.p_op_req}, theta={r.theta_req}: {r.mean:.2f} %)" for r in d.itertuples())
            lines.append(f"VERDICT {s} POL-{pol}: site-mean lambda_SU < 1.0 % in {len(d)} of {n_all} cells: {cells}")
        lines.append("VERDICT (rule): a site x policy leaves the 'material' class (P5 >= 1.0 %) at a grid cell when the site-mean lambda_SU of that cell is < 1.0 %; the grid is deterministic "
                     "(no UQ percentiles), so this is a screening bound on the class, not an SC-M class.")
    anyyr = gm[gm["min"] < 1.0]
    lines.append(f"Cells with at least one site-year below 1.0 %: {len(anyyr)} of {len(gm)}")
    (RESULTS / "sens" / "falsification_verdict.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return be


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    {"grid": grid, "breakeven": breakeven}[sys.argv[1]]()

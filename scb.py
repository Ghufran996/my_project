"""SC-B, VER-4, VER-5 on the plant engine (Brief 3 item 6) -> results/scb_plant.csv.

    python code/scb.py
(i) annual H2 and energy closure <= 1e-6 relative (independent recomputation from the per-step trace, all 17 site-years, POL-A/C/D);
(ii) constant P = 0.6 P_rated all year, initial state PRODUCTION -> N_r = 0, lambda_SU = 0;
(iii) all start-up mechanisms off -> B-S1 annual M_del to <= 1e-9 (17 site-years; B-S1 = independent vectorised implementation);
(iv) washout and inventory inside the engine vs analytic <= 0.1 % (constant-power start from OFF; tanks-in-series integrator);
(v) VER-5: Cabauw, first passing year, the month with the most P_min crossings (rule fixed before the run), 10 s vs 1 min internal step on the same held 1-min input.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su import plant as pl  # noqa: E402
from h2su import plantrun as pr  # noqa: E402
from h2su.baselines.bs1 import bs1_annual  # noqa: E402
from h2su.gas import tanks_in_series_step_response  # noqa: E402
from h2su.io import CODE_DIR, RESULTS  # noqa: E402
from h2su.pv import pv_dc_series  # noqa: E402


def site_years():
    y = yaml.safe_load((CODE_DIR / "configs" / "sites.yaml").read_text())
    return [(s, yr) for s, v in y["sites"].items() for yr in v["years_selected"]]


def independent_stack_recompute(tr, par):
    """M_stack from the trace with a separate numpy lookup (np.interp on the T grid, then on the load grid) instead of the numba bilinear routine."""
    st = pr.stack_tables()
    p_st, T = tr[:, 1], tr[:, 8]
    l = p_st / par[pl.P_RATED]
    # linear in T between the two bracketing grid rows, linear in l within each row
    iT = np.clip(np.searchsorted(pr.TG, T) - 1, 0, pr.TG.size - 2)
    out = np.zeros_like(l)
    for it in np.unique(iT):
        sel = iT == it
        r0 = np.interp(l[sel], pr.LG, st["mtab"][it])
        r1 = np.interp(l[sel], pr.LG, st["mtab"][it + 1])
        f = np.clip((T[sel] - pr.TG[it]) / (pr.TG[it + 1] - pr.TG[it]), 0, 1)
        out[sel] = r0 * (1 - f) + r1 * f
    out = np.where(p_st > 0, out * (par[pl.P_RATED] / 1000.0) / 60.0, 0.0)
    return out.sum()


def main():
    rows = []
    t0 = time.time()
    # (i) closure
    worst_h, worst_e, worst_s = 0.0, 0.0, 0.0
    n_runs = 0
    for s, yr in site_years():
        df = pv_dc_series(s, yr)
        for pol in "ACD":
            par = pr.build_params(pol)
            tot, cyc, nc, tr = pr.run(df, par, pr.switches(), trace=True)
            m = pr.metrics(tot, cyc, nc, par)
            ref = independent_stack_recompute(tr, par)
            fates = tr[:, 3].sum() + tr[:, 4].sum() + tr[:, 5].sum() + tr[:, 6].sum() + tr[:, 9][-1] + (tr[-1, 7] - par[pl.M_INV0])
            # independent H2 closure from the per-step trace: stack = delivered + purge + offspec + (deoxo+dryer) + blowdown + inventory change
            h2c = abs(tr[:, 2].sum() - fates) / tr[:, 2].sum()
            h2_indep = abs(ref - tot[pl.T_STACK]) / tot[pl.T_STACK]
            e_in = float(np.minimum(df.p_dc_kw.to_numpy(), pr.P_RATED_KW).sum() / 60.0)
            e_c = abs(e_in - tot[pl.T_EAVAIL]) / e_in
            e_c2 = abs(tot[pl.T_EAVAIL] - tot[pl.T_ESTACK] - tot[pl.T_EUNUSED]) / tot[pl.T_EAVAIL]
            e_c3 = abs(tr[:, 1].sum() / 60.0 - tot[pl.T_ESTACK]) / tot[pl.T_ESTACK]
            worst_h = max(worst_h, h2c, h2_indep)
            worst_e = max(worst_e, e_c, e_c2, e_c3)
            worst_s = max(worst_s, m["identity_SU_SS_rel"])
            n_runs += 1
    rows.append(dict(check="(i) annual H2 closure + independent Faraday/table recompute", n_runs=n_runs, value=worst_h, tol=1e-6, passed=worst_h <= 1e-6, note="worst of trace-based fate closure and independent M_stack recompute; 17 site-years x POL-A/C/D"))
    rows.append(dict(check="(i) annual energy closure", n_runs=n_runs, value=worst_e, tol=1e-6, passed=worst_e <= 1e-6, note="input E (clipped PV) vs engine E_avail; E_avail = E_stack + E_unused; E_stack vs trace"))
    rows.append(dict(check="(i) D32d identity lambda_SU + lambda_SS = (M_stack - M_del - dM_inv)/M_stack", n_runs=n_runs, value=worst_s, tol=1e-9, passed=worst_s <= 1e-9, note="cycle windows + outside losses vs fate sums"))
    # (ii) constant power
    n = 365 * 1440
    par0 = pr.build_params("A")
    par = pr.build_params("A", INIT_STATE=3, M_INV0=par0[pl.I_INV] * par0[pl.MRATED_MIN] * (par0[pl.P_OP] - par0[pl.P0]))
    df = pd.DataFrame({"p_dc_kw": np.full(n, 600.0), "zenith": 60.0, "ta_c": 15.0, "fill": np.zeros(n, np.int8)})
    tot, cyc, nc, tr = pr.run(df, par, pr.switches())
    m = pr.metrics(tot, cyc, nc, par)
    nr = m["N_r_hot"] + m["N_r_warm"] + m["N_r_cold"]
    rows.append(dict(check="(ii) constant P = 0.6 P_rated, initial PRODUCTION", n_runs=1, value=abs(m["lambda_SU"]) + nr, tol=0.0, passed=bool(nr == 0 and abs(m["lambda_SU"]) == 0.0), note=f"N_r = {nr}, lambda_SU = {m['lambda_SU']}"))
    # (iii) B-S1 equivalence
    worst = 0.0
    for s, yr in site_years():
        df = pv_dc_series(s, yr)
        p1, sw1 = pr.s1_config(pr.build_params("A"))
        tot, cyc, nc, tr = pr.run(df, p1, sw1)
        ref = bs1_annual(df.p_dc_kw.to_numpy())
        worst = max(worst, abs(tot[pl.T_DEL] - ref["M_del"]) / ref["M_del"], abs(tot[pl.T_STACK] - ref["M_stack"]) / ref["M_stack"])
    rows.append(dict(check="(iii) all start-up mechanisms off = B-S1 (M_del and M_stack)", n_runs=17, value=worst, tol=1e-9, passed=worst <= 1e-9, note="engine (numba state machine) vs bs1.py (numpy), 17 site-years"))
    # (iv) analytic (same tests as tests/test_plant.py, evaluated here with numbers)
    par = pr.build_params("A")
    sw = pr.switches(ramp=0, deoxo=0)
    df = pd.DataFrame({"p_dc_kw": np.full(600, 500.0), "zenith": 60.0, "ta_c": 15.0, "fill": np.zeros(600, np.int8)})
    tot, cyc, nc, tr = pr.run(df, par, sw, trace=True)
    mr = par[pl.MRATED_MIN]
    m_full = par[pl.I_INV] * mr * (par[pl.P_OP] - par[pl.P0])
    e_inv = abs(tot[pl.T_DINV] - m_full) / m_full
    e_wash = abs(tot[pl.T_PURGE_] + tot[pl.T_OFFSPEC] - par[pl.THETA_SPEC] * mr) / (par[pl.THETA_SPEC] * mr)
    tau = par[pl.THETA_SPEC] / np.log(1e6 / 300.0)
    t, c = tanks_in_series_step_response(1, tau, 10 * tau, tau / 4000.0, 1.0)
    t_cross = t[np.argmax(1.0 - c <= 300e-6)]
    e_int = abs(t_cross - par[pl.THETA_SPEC]) / par[pl.THETA_SPEC]
    rows.append(dict(check="(iv) inventory filled vs analytic m_full", n_runs=1, value=e_inv, tol=1e-3, passed=e_inv <= 1e-3, note=f"m_full = {m_full:.6f} kg"))
    rows.append(dict(check="(iv) washout: diverted rated-minutes vs theta_spec", n_runs=1, value=e_wash, tol=1e-3, passed=e_wash <= 1e-3, note=f"theta_spec = {par[pl.THETA_SPEC]} rated-min"))
    rows.append(dict(check="(iv) washout threshold crossing vs RK4 tanks-in-series (N = 1)", n_runs=1, value=e_int, tol=1e-3, passed=e_int <= 1e-3, note=f"tau = {tau:.4f} min"))
    # (v) VER-5
    cand = "Cabauw"
    y = yaml.safe_load((CODE_DIR / "configs" / "sites.yaml").read_text())["sites"][cand]["years_selected"][0]
    df = pv_dc_series(cand, y)
    par = pr.build_params("A")
    pa = np.minimum(df.p_dc_kw.to_numpy(), pr.P_RATED_KW)
    ok = pa >= par[pl.P_MIN_F] * par[pl.P_RATED]
    cross = np.flatnonzero(np.diff(ok.astype(int)) != 0)
    months = df.index[cross].month
    cnt = pd.Series(months).value_counts().sort_index()
    mo = int(cnt.idxmax())
    sel = df[df.index.month == mo]
    m1 = pr.metrics(*pr.run(sel, par, pr.switches())[:3], par)
    # 10 s internal step: same held 1-min input repeated 6 times
    rep = pd.DataFrame({c: np.repeat(sel[c].to_numpy(), 6) for c in ("p_dc_kw", "zenith", "ta_c", "fill")})
    p10 = par.copy()
    p10[pl.DT] = 1.0 / 6.0
    t10 = pr.run(rep, p10, pr.switches())
    m10 = pr.metrics(*t10[:3], p10)
    rel = abs(m10["lambda_SU"] - m1["lambda_SU"]) / m1["lambda_SU"]
    rows.append(dict(check=f"(v) VER-5 {cand} {y} month {mo} ({int(cnt.max())} P_min crossings): lambda_SU 10 s vs 1 min", n_runs=2, value=rel, tol=0.01, passed=rel <= 0.01,
                     note=f"lambda_SU 1 min = {m1['lambda_SU']:.6f} %, 10 s = {m10['lambda_SU']:.6f} %; M_del 1 min = {m1['M_del']:.3f}, 10 s = {m10['M_del']:.3f} kg; crossings per month = {cnt.to_dict()}"))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "scb_plant.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 140)
    print(out[["check", "value", "tol", "passed"]].to_string(index=False))
    print(out.note.iloc[-1])
    print("all pass:", bool(out.passed.all()), f"wall {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()

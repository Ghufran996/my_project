"""Brief 5 item 4: PL8 dryer scenarios and dryer event bounds (D53).

    python code/dryer_scenarios.py scenarios   -> results/sens/dryer_scenarios.csv
    python code/dryer_scenarios.py bound       -> results/sens/dryer_event_bound.csv   (column (iii) is filled by vt6b_run.py, item 5)

Scenarios (steady purification term of the dryer, full model and B-S1 on the same constant): s(L) = phi + b/L (nominal, VT5a fit of L1 Table 2);
constant 1.02 % (L15: 100 - 98.98, registry vpsa_recovery_opt) and constant 1.6 % (L14 abstract: 100 - 98.4, registry bcl_recovery_primary). 17 site-years x POL-A/C/D.
Dryer event bounds (dryer start-up stays OUTSIDE the engine, D53): (i) re-pressurisation H2 per completed restart (results/ver6/ver6_events.csv, dry_su_i);
(ii) bound of one extra regeneration purge per restart = P/F (registry vpsa_purge_to_feed) x rated H2 flow x 1.5 h and 2.5 h (L15 Table S4 durations); (iii) off-spec bound from
dry_su_ii (results/ver6_dryer_nontrace.csv) if t_spec > 0. Each in kg/yr and pp of lambda_SU. All are BOUNDS, not results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2su import plant as pl  # noqa: E402
from h2su import plantrun as pr  # noqa: E402
from h2su.io import RESULTS, param  # noqa: E402
from sweeps import run_grid  # noqa: E402


def scenarios():
    c102 = 1.0 - param("vpsa_recovery_opt") / 100.0
    c16 = 1.0 - param("bcl_recovery_primary") / 100.0
    specs = []
    for pol in "ACD":
        specs += [dict(name="s(L) nominal", pol=pol, s1=True, tags=dict(steady_term="s(L) = phi + b/L (VT5a fit, L1 Table 2)")),
                  dict(name="const_1.02pct_L15", pol=pol, over=dict(PHI=c102, B_PSA=0.0), s1=True, tags=dict(steady_term=f"constant {c102 * 100:.2f} % (L15 98.98 % recovery)")),
                  dict(name="const_1.6pct_L14", pol=pol, over=dict(PHI=c16, B_PSA=0.0), s1=True, tags=dict(steady_term=f"constant {c16 * 100:.2f} % (L14 abstract 98.4 % recovery)"))]
    df = run_grid(specs, workers=10)
    df["identity_lamSU_plus_lamSS"] = (df.lambda_SU + df.lambda_SS) - ((df.M_stack - df.M_del - df.dM_inv) / df.M_stack * 100.0)
    nom = df[df.variant == "s(L) nominal"].set_index(["site", "year", "policy"])
    df["d_M_del_pct_vs_nominal"] = [(r.M_del - nom.loc[(r.site, r.year, r.policy), "M_del"]) / nom.loc[(r.site, r.year, r.policy), "M_del"] * 100.0 for r in df.itertuples()]
    df["d_lambda_SU_pp_vs_nominal"] = [r.lambda_SU - nom.loc[(r.site, r.year, r.policy), "lambda_SU"] for r in df.itertuples()]
    out = RESULTS / "sens"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["site", "year", "policy", "variant", "steady_term", "M_stack", "M_del", "eta_del", "lambda_SU", "lambda_SS", "beta_B_S1", "M_del_S1", "d_M_del_pct_vs_nominal", "d_lambda_SU_pp_vs_nominal",
            "closure_H2_rel", "closure_E_rel", "identity_lamSU_plus_lamSS", "N_r_completed"]
    df[cols].to_csv(out / "dryer_scenarios.csv", index=False, float_format="%.8g")
    print("max closure", df.closure_H2_rel.max(), df.closure_E_rel.max(), "max |identity lambda_SU + lambda_SS|", df.identity_lamSU_plus_lamSS.abs().max())
    print(df.groupby(["policy", "variant"])[["eta_del", "lambda_SU", "lambda_SS", "beta_B_S1", "d_M_del_pct_vs_nominal"]].mean().round(4).to_string())
    return df


def bound():
    ev = pd.read_csv(RESULTS / "ver6" / "ver6_events.csv")
    rep = ev[(ev.case_id == "dry_su_i") & (ev.model == "python_mol") & (ev.grid_id == "N400") & (ev.rtol == 1e-6)].m_H2_repress_kg.iloc[0]
    pf = param("vpsa_purge_to_feed") / 100.0
    mr = float(pr.stack_tables()["m_rated_kg_h"])
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    f = ann[ann.model == "full"]
    # (iii): from the non-trace dryer cases if the file exists
    t_spec_h2 = None
    nt = RESULTS / "ver6_dryer_nontrace.csv"
    if nt.exists():
        d = pd.read_csv(nt)
        t_spec_h2 = d
    rows = []
    for r in f.itertuples():
        base = dict(site=r.site, year=r.year, policy=r.policy, N_r_completed=r.N_r_completed, M_stack_kg=r.M_stack, lambda_SU_pct=r.lambda_SU)
        for tag, per_event, desc in (("i_repressurisation", rep, "re-pressurisation H2 per completed restart (dry_su_i, plant bed, results/ver6/ver6_events.csv)"),
                                     ("ii_extra_regeneration_1.5h", pf * mr * 1.5, "one extra regeneration purge per completed restart: P/F x rated H2 flow x 1.5 h (L15 Table S4 BT7)"),
                                     ("ii_extra_regeneration_2.5h", pf * mr * 2.5, "same, 2.5 h (L15 Table S4 BT6)")):
            kg = per_event * r.N_r_completed
            rows.append(dict(base, bound=tag, kg_per_event=per_event, kg_per_year=kg, pp_of_lambda_SU=kg / r.M_stack * 100.0, label="bound (dryer start-up outside the engine, D53)", description=desc))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "sens" / "dryer_event_bound.csv", index=False, float_format="%.8g")
    print(out.groupby(["policy", "bound"]).pp_of_lambda_SU.mean().round(4).to_string())
    return out


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    {"scenarios": scenarios, "bound": bound}[sys.argv[1]]()

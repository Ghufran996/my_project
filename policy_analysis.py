"""Brief 5 item 2: POL-P bounding sensitivity and standby break-even.

    python code/policy_analysis.py polp        -> results/policy/pol_p_bounding.csv, pol_p_decomposition.csv
    python code/policy_analysis.py polp_post   -> adds m_SU_P_kg, m_SU_D_kg, flag m_SU_P <= m_SU_D, share of the POL-D gain from not venting (no new runs; reads pol_p_bounding.csv, Brief 6 item 3, D61f)
    python code/policy_analysis.py breakeven   -> results/policy/standby_breakeven.csv   (no new runs: polB_pareto.csv and nominal/annual.csv)

POL-P (bounding sensitivity, NOT a policy of the plan; label "bounding: no pressure decay, no crossover handling, no source (SN29)"): at a stop no blow-down, no final inerting,
no N2 ingress, the gas inventory is retained, standby power 0, the stack cools by the D32a ODE (engine flag COOL_STANDBY); restart without inerting, purge or off-spec
diversion, thermal ramp by T0. Implemented as the engine policy code 1 (standby) with t_sb,max = infinity, SB_F = 0 and COOL_STANDBY = 1.
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
from h2su.io import RESULTS  # noqa: E402
from sweeps import run_grid  # noqa: E402

LABEL = "bounding: no pressure decay, no crossover handling, no source (SN29)"


def polp():
    specs = [dict(name="POL-A", pol="A", s1=True), dict(name="POL-D", pol="D", s1=True),
             dict(name="POL-P", pol="B", t_sb=1.0e9, over=dict(SB_F=0.0, COOL_STANDBY=1.0), s1=True, tags=dict(label=LABEL))]
    df = run_grid(specs, workers=10)
    df["policy_label"] = df.variant
    out = RESULTS / "policy"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pol_p_bounding.csv", index=False)
    key = ["site", "year"]
    a = df[df.variant == "POL-A"].set_index(key)
    p = df[df.variant == "POL-P"].set_index(key)
    d = df[df.variant == "POL-D"].set_index(key)
    dec = pd.DataFrame(dict(M_del_A=a.M_del, M_del_P=p.M_del, M_del_D=d.M_del, lambda_SU_A=a.lambda_SU, lambda_SU_P=p.lambda_SU, lambda_SU_D=d.lambda_SU,
                            value_of_not_venting_pct=(p.M_del - a.M_del) / a.M_del * 100.0, value_of_heat_pct=(d.M_del - p.M_del) / a.M_del * 100.0,
                            total_D_minus_A_pct=(d.M_del - a.M_del) / a.M_del * 100.0, E_standby_P=p.E_standby, E_standby_D=d.E_standby,
                            N_attempts_A=a.N_r_attempts, N_attempts_P=p.N_r_attempts, N_attempts_D=d.N_r_attempts, N_aborted_P=p.N_r_aborted, N_aborted_D=d.N_r_aborted,
                            closure_H2_P=p.closure_H2_rel, closure_E_P=p.closure_E_rel)).reset_index()
    dec["lambda_P_le_lambda_D"] = dec.lambda_SU_P <= dec.lambda_SU_D + 1e-12
    dec["label"] = LABEL
    dec.to_csv(out / "pol_p_decomposition.csv", index=False, float_format="%.6g")
    print(dec.groupby("site")[["lambda_SU_A", "lambda_SU_P", "lambda_SU_D", "value_of_not_venting_pct", "value_of_heat_pct", "total_D_minus_A_pct"]].mean().round(3).to_string())
    print("POL-P lambda_SU <= POL-D lambda_SU in all site-years:", bool(dec.lambda_P_le_lambda_D.all()), "max closure", dec.closure_H2_P.max(), dec.closure_E_P.max())
    polp_post()
    return dec


def polp_post():
    """Brief 6 item 3 (D61f): the acceptance check is stated on the numerator, m_SU,P <= m_SU,D (the denominators M_stack differ), relative tolerance 1e-9, 17 of 17 site-years;
    the ratio flag lambda_SU,P <= lambda_SU,D is kept with a note (the 3-5e-6 pp excess seen in the ratio is a denominator effect). Share of the POL-D gain from not venting
    = value_of_not_venting_pct / total_D_minus_A_pct (POL-A reference); at site level it is the ratio of the site-mean values."""
    out = RESULTS / "policy"
    b = pd.read_csv(out / "pol_p_bounding.csv")
    b["m_SU_kg"] = b.lambda_SU / 100.0 * b.M_stack                      # lambda_SU = m_SU / M_stack x 100 (definition)
    m = b.pivot(index=["site", "year"], columns="variant", values="m_SU_kg")
    dec = pd.read_csv(out / "pol_p_decomposition.csv")
    dec = dec.drop(columns=[c for c in dec.columns if c in ("m_SU_P_kg", "m_SU_D_kg", "m_SU_P_le_m_SU_D", "share_of_D_gain_from_not_venting", "note_ratio_flag")])
    dec = dec.merge(m.rename(columns={"POL-P": "m_SU_P_kg", "POL-D": "m_SU_D_kg"})[["m_SU_P_kg", "m_SU_D_kg"]].reset_index(), on=["site", "year"], how="left")
    dec["m_SU_P_le_m_SU_D"] = dec.m_SU_P_kg <= dec.m_SU_D_kg * (1.0 + 1e-9)
    dec["share_of_D_gain_from_not_venting"] = dec.value_of_not_venting_pct / dec.total_D_minus_A_pct
    dec["note_ratio_flag"] = "lambda_P_le_lambda_D kept; a 3-5e-6 pp excess of the ratio is a denominator effect (M_stack differs), the check is stated on the numerator m_SU (D61f)"
    dec.to_csv(out / "pol_p_decomposition.csv", index=False, float_format="%.9g")
    site = dec.groupby("site")[["value_of_not_venting_pct", "value_of_heat_pct", "total_D_minus_A_pct"]].mean()
    site["share_of_D_gain_from_not_venting"] = site.value_of_not_venting_pct / site.total_D_minus_A_pct
    site["n_site_years"] = dec.groupby("site").size()
    site["n_m_SU_P_le_m_SU_D"] = dec.groupby("site").m_SU_P_le_m_SU_D.sum()
    site.reset_index().to_csv(out / "pol_p_decomposition_site.csv", index=False, float_format="%.6g")
    print(f"m_SU_P <= m_SU_D in {int(dec.m_SU_P_le_m_SU_D.sum())} of {len(dec)} site-years")
    print(site.round(4).to_string())
    return dec


def breakeven():
    p = pd.read_csv(RESULTS / "policy" / "polB_pareto.csv")
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    f = ann[ann.model == "full"].set_index(["site", "year", "policy"])
    sb_nom = 0.095
    rows = []
    pr_kw = pr.P_RATED_KW
    for s in ("Cabauw", "Carpentras", "Gobabeb"):
        yrs = sorted(p[p.site == s].year.unique())
        # POL-D reference gain per standby fraction (the recovered share is relative to POL-D at the same standby power)
        for sbf, tag in ((0.01, "sb001"), (sb_nom, "sb0095"), (0.18, "sb018")):
            ref_gain = {}
            for variant in ("POL-B_15", "POL-B_30", "POL-B_60", "POL-B_120", "POL-D", "C"):
                dM, E, sec, mmin = [], [], [], []
                for y in yrs:
                    a = p[(p.site == s) & (p.year == y) & (p.variant == "POL-A") & np.isclose(p.sb_f, sbf)].iloc[0]
                    if variant == "C":
                        c = f.loc[(s, y, "C")]
                        m_del, e_sb, stack_e, stack_m, smin = c.M_del, c[f"E_standby_{tag}"], c.E_stack, c.M_stack, c.min_standby
                    else:
                        r = p[(p.site == s) & (p.year == y) & (p.variant == variant) & np.isclose(p.sb_f, sbf)].iloc[0]
                        m_del, e_sb, stack_e, stack_m, smin = r.M_del, r.E_standby, r.E_stack, r.M_stack, r.min_standby
                    dM.append(m_del - a.M_del)
                    E.append(e_sb)
                    sec.append(stack_e / stack_m)
                    mmin.append(smin)
                dM, E, sec, mmin = (np.array(x, float) for x in (dM, E, sec, mmin))
                gain, esb = dM.mean(), E.mean()
                ref_gain[variant] = gain
                rows.append(dict(site=s, policy=variant.replace("POL-", ""), standby_fraction=sbf, n_years=len(yrs), delta_M_del_kg=gain,
                                 delta_M_del_pct=float(np.mean(dM / np.array([p[(p.site == s) & (p.year == y) & (p.variant == "POL-A") & np.isclose(p.sb_f, sbf)].M_del.iloc[0] for y in yrs])) * 100.0),
                                 E_standby_kWh=esb, E_standby_MWh=esb / 1e3, kWh_per_kg_gained=esb / gain if gain > 0 else np.nan, stack_specific_energy_kWh_kg=float(sec.mean()),
                                 breakeven_standby_fraction=float(gain * sec.mean() / (pr_kw * mmin.mean() / 60.0)) if mmin.mean() > 0 else np.nan, standby_hours=float(mmin.mean() / 60.0)))
            for r in rows[-6:]:
                r["recovered_share_of_POL_D_gain"] = r["delta_M_del_kg"] / ref_gain["POL-D"]
    df = pd.DataFrame(rows)
    out = RESULTS / "policy"
    df.to_csv(out / "standby_breakeven.csv", index=False, float_format="%.6g")
    show = df[np.isclose(df.standby_fraction, sb_nom)][["site", "policy", "delta_M_del_pct", "E_standby_MWh", "kWh_per_kg_gained", "breakeven_standby_fraction", "recovered_share_of_POL_D_gain"]]
    print(show.round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    {"polp": polp, "polp_post": polp_post, "breakeven": breakeven}[sys.argv[1]]()

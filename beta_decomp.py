"""Brief 4 item 6: beta decomposition chains exactly as D43a (fixed before any run).

    python code/beta_decomp.py   ->  results/nominal/beta_decomposition.csv

B-CL chain (1-min input, per site-year x policy; M_del = delivered H2 [kg/yr]):
  V0 = B-CL as implemented (75 % LHV, 98.4 % recovery, P_min 0.10, clipping)
  V1 = V0 with the engine steady stack efficiency (polarisation + Faradaic), constant 98.4 % recovery          term "efficiency convention"
  V2 = V1 with the engine dispatch constraints (P_min, clipping)                                               term "dispatch" (0 if B-CL already has them)
  V3 = V2 with s(L) replacing the constant recovery; V3 = B-S1 (assert <= 1e-9)                                term "steady purification model"
  V4 = full engine, start-up purification mechanisms off (blow-down, purge, off-spec, deoxo), ramp / inerting / shutdown sequence / aborts / s(L) on
                                                                                                                term "start-up delays, production forgone"
  V5 = full engine                                                                                             term "start-up purification losses"
  term_k = (M_del,V(k-1) - M_del,V(k)) / M_del,full x 100 (pp); sum_k term_k = beta_B-CL (assert <= 1e-9).
  Reverse order: from the full model, toggles removed in the opposite order: E (fixed 75 % LHV stack), P (constant recovery), L (mechanisms off), D (instant machine) -> B-CL.
B-EP chain: E0 = B-EP (PVGIS hourly, constant net SEC) -> E1 = same rule on hourly block means of the BSRN-chain DC power [irradiance source] -> E2 = same rule on 1-min
  BSRN DC power [resolution] -> E3 = engine B-S1 configuration with the BoP drawn from the PV bus (L24 constant share) and s(L) [SEC convention + steady purification]
  -> E4 = full engine with the BoP drawn from the PV bus (L24) [start-up delays + losses]. Terms normalised by M_del of E4 (convention-matched pair);
  beta_B-EP vs the nominal (auxiliary-supply) full model reported too. Order check: resolution and SEC convention swapped (E1 -> engine on hourly input -> E3).
B-S2 split (D34): carried from results/nominal/annual.csv (dispatch-foresight and modelling parts).
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
from h2su.baselines.bcl import bcl, bep  # noqa: E402
from h2su.io import RESULTS, param  # noqa: E402
from h2su.pv import block_mean_hold, pv_dc_series  # noqa: E402
from h2su.shared import block_mean  # noqa: E402
from sweeps import site_years  # noqa: E402

REC = param("bcl_recovery_primary") / 100.0
ETA = param("c3_fixed_efficiency") / 100.0
LHV = param("c3_LHV") * 1000.0
MECH_OFF = dict(blow=0, purge=0, offspec=0, deoxo=0)


def eng(df, par, sw, mtab=None):
    tot = pr.run(df, par, sw, mtab=mtab)[0]
    return float(tot[pl.T_DEL])


def s1_variant(par, fixedE=False, constP=False, bop=False):
    p, sw = pr.s1_config(par)
    if constP:
        p[pl.PHI], p[pl.B_PSA] = 1.0 - REC, 0.0
    if bop:
        p[pl.BOP_MODE] = 1.0
    return p, sw


def full_variant(par, constP=False, mech_off=False, bop=False):
    p = par.copy()
    sw = pr.switches(**(MECH_OFF if mech_off else {}))
    if constP:
        p[pl.PHI], p[pl.B_PSA] = 1.0 - REC, 0.0
    if bop:
        p[pl.BOP_MODE] = 1.0
    return p, sw


def bep_rule(p_kw, step_h):
    on = np.minimum(p_kw, pr.P_RATED_KW) >= param("pmin_pem_low") * pr.P_RATED_KW
    return float(np.where(on, np.minimum(p_kw, pr.P_RATED_KW), 0.0).sum() * step_h / param("bep_sec_net"))


def main():
    ann = pd.read_csv(RESULTS / "nominal" / "annual.csv")
    fm = pr.fixed_eff_mtab(ETA, LHV)
    rows = []
    for s, y in site_years():
        df = pv_dc_series(s, y)
        pdc = df.p_dc_kw.to_numpy()
        v0 = bcl(pdc, 1)["M_del"]
        e0 = bep(s, y)["M_del"]
        e1 = bep_rule(block_mean(pdc, 60), 1.0)                       # hourly block means of the BSRN-chain DC power, 1-h steps
        e2 = bep_rule(pdc, 1.0 / 60.0)
        dfh = df.copy()
        dfh["p_dc_kw"] = block_mean_hold(pdc, 60)
        for pol in "ACD":
            par = pr.build_params(pol)
            # ---- B-CL chain
            v1 = eng(df, *s1_variant(par, constP=True))
            v2 = v1                                              # B-CL already applies P_min and the rated clipping (checked: engine S1 with constant recovery = V1)
            v3 = eng(df, *s1_variant(par))
            v4 = eng(df, *full_variant(par, mech_off=True))
            v5 = eng(df, *full_variant(par))
            terms = [(v0 - v1) / v5 * 100, (v1 - v2) / v5 * 100, (v2 - v3) / v5 * 100, (v3 - v4) / v5 * 100, (v4 - v5) / v5 * 100]
            beta_cl = (v0 - v5) / v5 * 100
            bs1_ref = pr.stack_tables() and float(ann[(ann.site == s) & (ann.year == y) & (ann.policy == pol) & (ann.model == "B-S1")].M_del.iloc[0])
            # ---- reverse order (engine runs with the fixed-efficiency stack table)
            c1 = eng(df, *full_variant(par), mtab=fm)
            c2 = eng(df, *full_variant(par, constP=True), mtab=fm)
            c3 = eng(df, *full_variant(par, constP=True, mech_off=True), mtab=fm)
            c4 = eng(df, *s1_variant(par, constP=True), mtab=fm)
            rev = [(c1 - v5) / v5 * 100, (c2 - c1) / v5 * 100, (c3 - c2) / v5 * 100, (c4 - c3) / v5 * 100]          # E, P, L, D (reverse order, same sign as the forward terms); c4 must equal V0
            row = dict(site=s, year=y, policy=pol, chain="B-CL", M_V0=v0, M_V1=v1, M_V2=v2, M_V3=v3, M_V4=v4, M_V5_full=v5,
                       term_efficiency_convention_pp=terms[0], term_dispatch_pp=terms[1], term_steady_purification_model_pp=terms[2], term_startup_delays_pp=terms[3],
                       term_startup_purification_losses_pp=terms[4], beta_B_CL_pct=beta_cl, sum_terms_minus_beta=sum(terms) - beta_cl,
                       V3_minus_BS1_rel=abs(v3 - bs1_ref) / bs1_ref, V0_minus_reverse_end_rel=abs(c4 - v0) / v0,
                       rev_term_E_pp=rev[0], rev_term_P_pp=rev[1], rev_term_L_pp=rev[2], rev_term_D_pp=rev[3], rev_sum_minus_beta=sum(rev) - beta_cl)
            fw = {"E": terms[0], "P": terms[2], "D": terms[3], "L": terms[4]}
            two = sorted(fw, key=lambda k: -abs(fw[k]))[:2]
            row["two_largest_forward_terms"] = "+".join(two)
            row.update({f"fwd_{k}_pp": fw[k] for k in fw})
            row.update({f"rev_{k}_pp": dict(E=rev[0], P=rev[1], L=rev[2], D=rev[3])[k] for k in fw})
            rows.append(row)
            # ---- B-EP chain
            e3 = eng(df, *s1_variant(par, bop=True))
            e4 = eng(df, *full_variant(par, bop=True))
            e2s = eng(dfh, *s1_variant(par, bop=True))                    # order swap: SEC convention at hourly resolution first
            tt = [(e0 - e1) / e4 * 100, (e1 - e2) / e4 * 100, (e2 - e3) / e4 * 100, (e3 - e4) / e4 * 100]
            beta_ep = (e0 - e4) / e4 * 100
            nom = float(ann[(ann.site == s) & (ann.year == y) & (ann.policy == pol) & (ann.model == "full")].M_del.iloc[0])
            rows.append(dict(site=s, year=y, policy=pol, chain="B-EP", M_V0=e0, M_V1=e1, M_V2=e2, M_V3=e3, M_V4=e4, M_V5_full=e4,
                             term_irradiance_source_pp=tt[0], term_resolution_pp=tt[1], term_sec_convention_and_steady_purification_pp=tt[2],
                             term_startup_delays_and_losses_pp=tt[3], beta_B_EP_vs_BoPPV_pct=beta_ep, sum_terms_minus_beta=sum(tt) - beta_ep,
                             beta_B_EP_vs_nominal_pct=(e0 - nom) / nom * 100,
                             swap_term_sec_at_hourly_pp=(e1 - e2s) / e4 * 100, swap_term_resolution_in_engine_pp=(e2s - e3) / e4 * 100,
                             swap_sum_minus_original=((e1 - e2s) + (e2s - e3)) / e4 * 100 - (tt[1] + tt[2])))
        print(s, y, flush=True)
    df = pd.DataFrame(rows)
    # B-S2 split (D34) from annual.csv
    sp = ann[ann.model == "full"][["site", "year", "policy"] + [c for c in ann.columns if "S2" in c and c.startswith("beta")]].copy()
    sp.insert(3, "chain", "B-S2 (D34, carried from annual.csv)")
    out = pd.concat([df, sp], ignore_index=True)
    out.to_csv(RESULTS / "nominal" / "beta_decomposition.csv", index=False)
    cl = df[df.chain == "B-CL"]
    ep = df[df.chain == "B-EP"]
    print("B-CL max |sum - beta|", cl.sum_terms_minus_beta.abs().max(), "max V3-BS1", cl.V3_minus_BS1_rel.max(), "max V0 vs reverse end", cl.V0_minus_reverse_end_rel.max(),
          "max rev sum", cl.rev_sum_minus_beta.abs().max())
    print("B-EP max |sum - beta|", ep.sum_terms_minus_beta.abs().max(), "swap check", ep.swap_sum_minus_original.abs().max())
    return out


if __name__ == "__main__":
    main()

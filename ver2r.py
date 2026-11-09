"""VER-2R (D24/D30): C3 (L4) published Table 3 / Table 5 numbers re-run on AEMO NEM 2022 5-min data.

    python code/ver2r.py

Outputs: results/ver2r_milp_check.csv (DP vs MILP, one-week windows), results/ver2r_c3.csv (verdicts by the D24 rule),
results/ver2r_sensitivity.csv (not part of the verdict).  The digitisation file hash is checked against
results/ver2r_digitisation_hash.txt before and after the simulations.

D24 rule (fixed before any run): identity check = 2022 capacity factor within +-1 pp of L4 Table 2 for all three farms, else
INCONCLUSIVE.  Per target: INCONCLUSIVE if the half-width of the band from eta +- u_d is larger than 10 % of |target|;
otherwise PASS if |RE| <= 10 % at the central curve, else FAIL.  SC-R PASS only if the identity check passes and every
included target is PASS.  Table 3 = (H2 variable - H2 fixed 75 %)/H2 fixed at electrolyser/farm ratio 1, 5-min input (header
reading, D24); Table 5 = (H2 at 60 min - H2 at 5 min) at ratio 0.75 (L4 Sec 4.2), 60-min input = block means of the 5-min series.
"""
from __future__ import annotations

import hashlib
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su.baselines import bs2_c3 as b  # noqa: E402
from h2su.io import CONFIGS, RESULTS  # noqa: E402

FARMS = {"Finley": ("FINLYSF1", 162.0, 19.0), "Tailem Bend": ("TBSF1", 108.0, 17.0), "Hamilton": ("HAMISF1", 57.5, 24.0)}
TABLE = CONFIGS / "tables" / "L4_virahsawmy2024.csv"
HASH_FILE = RESULTS / "ver2r_digitisation_hash.txt"
WINDOWS = ["2022-03-07", "2022-06-06", "2022-09-05"]        # pre-declared (ISO weeks 10, 23, 36), before any run
TOL = 0.10


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_farm(duid: str) -> pd.Series:
    d = pd.read_parquet(RESULTS / "data" / "aemo2022" / f"{duid}_2022.parquet").set_index("SETTLEMENTDATE")["SCADAVALUE"]
    idx = pd.date_range("2022-01-01 00:05", "2023-01-01 00:00", freq="5min")
    return d.reindex(idx)


def curves(tab: pd.DataFrame):
    e = tab[tab.table == "Fig2"]
    emax = float(e[e.item == "eta_max"].value.iloc[0])
    ud = float(e[e.item == "eta_max"].u_d.iloc[0])
    emin = {int(r.curve): float(r.value) for r in e[e.item == "eta_min"].itertuples()}
    return emax, emin, ud


def h2(p_mw, step_min, p_cap, curve, **kw):
    return b.solve_dp(p_mw, step_min / 60.0, p_cap, curve, **kw)["H2_kg"]


def main():
    t0 = time.time()
    h0 = sha(TABLE)
    assert h0 in HASH_FILE.read_text(), "digitisation file hash differs from the frozen hash"
    tab = pd.read_csv(TABLE)
    emax, emin, ud = curves(tab)
    series, ident = {}, []
    for f, (duid, cap, cf_tab) in FARMS.items():
        s = load_farm(duid)
        n_missing = int(s.isna().sum())
        x = s.fillna(0.0).clip(lower=0.0).to_numpy()
        series[f] = (s.fillna(0.0).clip(lower=0.0), cap)
        cf = float(np.sum(x) * 5 / 60 / (cap * 8760) * 100)
        ident.append(dict(farm=f, DUID=duid, capacity_MW=cap, CF_model_pct=cf, CF_table2_pct=cf_tab, diff_pp=cf - cf_tab,
                          n_intervals=len(s), n_missing=n_missing, pass_1pp=abs(cf - cf_tab) <= 1.0))
    ident = pd.DataFrame(ident)
    print(ident.to_string(index=False), flush=True)
    id_ok = bool(ident.pass_1pp.all())

    # ---- MILP vs DP on one-week windows
    chk = []
    scen = {"fixed": b.Curve(emax, emax, True), "curve1": b.Curve(emax, emin[1]), "curve3": b.Curve(emax, emin[3])}
    for f, (s, cap) in series.items():
        for w in WINDOWS:
            seg = s[w:pd.Timestamp(w) + pd.Timedelta(days=7) - pd.Timedelta(minutes=5)].to_numpy()
            for name, cv in scen.items():
                for step in (5, 60):
                    if step == 60 and name == "fixed":
                        continue
                    x = seg if step == 5 else b.block_mean(seg, 12)
                    pc = cap * 1.0
                    d = b.solve_dp(x, step / 60, pc, cv)
                    m = b.solve_milp(x, step / 60, pc, cv, time_limit=300)
                    rel = abs(d["objective_MWh"] - m["objective_MWh"]) / max(abs(m["objective_MWh"]), 1e-12) if m["success"] else np.nan
                    chk.append(dict(farm=f, window=w, scenario=name, step_min=step, n_steps=len(x), obj_dp=d["objective_MWh"],
                                    obj_milp=m["objective_MWh"], rel_diff=rel, H2_dp_kg=d["H2_kg"], H2_milp_kg=m.get("H2_kg"),
                                    milp_status=m["status"], milp_success=m["success"]))
                    print(f, w, name, step, rel, flush=True)
    chk = pd.DataFrame(chk)
    chk.to_csv(RESULTS / "ver2r_milp_check.csv", index=False)
    milp_ok = bool((chk.rel_diff <= 1e-6).all())
    print("MILP check max rel diff", chk.rel_diff.max(), "ok", milp_ok, flush=True)

    # ---- targets
    rows = []

    def band_curve(k, pcts):
        vals = []
        for de, dm in itertools.product((-ud, ud), (-ud, ud)):
            vals.append(pcts(b.Curve(emax + de, emin[k] + dm)))
        return min(vals), max(vals)

    t3 = tab[tab.table == "Table3"]
    t5 = tab[tab.table == "Table5"]
    for f, (s, cap) in series.items():
        x5 = s.to_numpy()
        hf = h2(x5, 5, cap * 1.0, scen["fixed"])
        for r in t3[t3.site == f].itertuples():
            k = int(r.curve)
            if k not in (1, 3):
                continue                                    # D24: curves 1 and 3 only
            pc = lambda cv: (h2(x5, 5, cap * 1.0, cv) - hf) / hf * 100
            central = pc(b.Curve(emax, emin[k]))
            lo, hi = band_curve(k, pc)
            rows.append(dict(target=f"Table3 {f} curve{k}", location="L4 Table 3", published=float(r.value), unit="%", model=central,
                             band_lo=lo, band_hi=hi))
        x60 = b.block_mean(x5, 12)
        for r in t5[t5.site == f].itertuples():
            k = int(r.curve)
            pc = lambda cv: (h2(x60, 60, cap * 0.75, cv) - h2(x5, 5, cap * 0.75, cv)) / 1000.0
            central = pc(b.Curve(emax, emin[k]))
            lo, hi = band_curve(k, pc)
            rows.append(dict(target=f"Table5 {f} curve{k}", location="L4 Table 5 (ratio 0.75)", published=float(r.value), unit="t/yr",
                             model=central, band_lo=lo, band_hi=hi))
    res = pd.DataFrame(rows)
    res["RE"] = (res.model - res.published) / res.published.abs()
    res["band_halfwidth_rel"] = (res.band_hi - res.band_lo) / 2 / res.published.abs()
    def verdict(r):
        if r.band_halfwidth_rel > TOL:
            return "INCONCLUSIVE"
        return "PASS" if abs(r.RE) <= TOL else "FAIL"
    res["verdict"] = res.apply(verdict, axis=1)
    sc_r = "PASS" if (id_ok and milp_ok and (res.verdict == "PASS").all()) else ("INCONCLUSIVE" if not id_ok or (res.verdict == "INCONCLUSIVE").any() else "FAIL")
    res["identity_check_pass"] = id_ok
    res["SC_R"] = sc_r
    res.to_csv(RESULTS / "ver2r_c3.csv", index=False)
    ident.to_csv(RESULTS / "ver2r_identity.csv", index=False)
    print(res[["target", "published", "model", "RE", "band_halfwidth_rel", "verdict"]].round(4).to_string(index=False))
    print("identity", id_ok, "MILP", milp_ok, "SC-R", sc_r)

    # ---- sensitivity (reported, not part of the verdict)
    sens = []
    for f, (s, cap) in series.items():
        x5 = s.to_numpy()
        for lab, kw in (("literal_Eq8_allow_sb_off", dict(allow_sb_off=True)),):
            hf = h2(x5, 5, cap, scen["fixed"], **kw)
            for k in (1, 3):
                sens.append(dict(farm=f, variant=lab, curve=k, pct_diff=(h2(x5, 5, cap, b.Curve(emax, emin[k]), **kw) - hf) / hf * 100))
        for k in (1, 3):
            hf = h2(x5, 5, cap, scen["fixed"])
            for de, dm in ((0.005, -0.005), (-0.005, 0.005)):
                cv = b.Curve(emax + de, emin[k] + dm)
                sens.append(dict(farm=f, variant=f"eta_rounding_{de:+.3f}_{dm:+.3f}", curve=k, pct_diff=(h2(x5, 5, cap, cv) - hf) / hf * 100))
    pd.DataFrame(sens).to_csv(RESULTS / "ver2r_sensitivity.csv", index=False)
    assert sha(TABLE) == h0, "digitisation file changed during the run"
    print(f"digitisation hash unchanged ({h0[:12]}...); wall {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()

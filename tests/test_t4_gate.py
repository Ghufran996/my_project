"""Brief 6 item 1: T4/T5 integrity gate with a partial FAKE checkpoint set (synthetic data, not results)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tables_final as tf  # noqa: E402

COLS = ["lambda_SU", "closure_H2_rel", "closure_E_rel"]


def _fake_npz(uq, site, year, pol, seed, closure=1e-12):
    Y = np.column_stack([np.full(4, 9.0), np.full(4, closure), np.full(4, closure)])
    np.savez(uq / f"ens_{site}_{year}_{pol}_seed{seed}.npz", X=np.zeros((4, 1)), Y=Y, names=np.array(["f"]), cols=np.array(COLS))


def _ann():
    rows = []
    for s in ("S1", "S2"):
        for pol in ("A", "C", "D"):
            for y in (2013, 2014):
                rows.append(dict(site=s, year=y, policy=pol, model="full", lambda_SU=10.0, tau_eq_median=20.0, eta_del=0.8, beta_B_CL_1min=40.0, beta_B_EP=-2.0, beta_B_S1=20.0, beta_B_S3=25.0, beta_B_S2_curve1=30.0, beta_B_S2_curve2=20.0, beta_B_S2_curve3=10.0))
    return pd.DataFrame(rows)


def _pooled(uq, n_seeds):
    rows = []
    for s in ("S1", "S2"):
        for pol in ("A", "B60", "C", "D"):
            r = dict(site=s, policy=pol, seed="pooled", pv_ratio=1.5, n_seeds=n_seeds, n_years=2, closure_H2_max=1e-12, closure_E_max=1e-12)
            for o in ("lambda_SU", "tau_eq_median", "eta_del", "M_del"):
                for q in ("P5", "P50", "P95"):
                    r[f"{o}_{q}"] = 123.456
            rows.append(r)
    pd.DataFrame(rows).to_csv(uq / "intervals_pooled.csv", index=False)
    sm = pd.DataFrame([dict(source="interval ensemble", site=s, policy=p, pv_ratio="1.5", SC_M_class="material") for s in ("S1", "S2") for p in ("A", "B60", "C", "D")])
    sm.to_csv(uq / "scm_verdict.csv", index=False)
    b = pd.DataFrame([dict(site=s, policy=p, baseline=bb, n_years=2, beta_P5=1.0, beta_P50=2.0, beta_P95=3.0) for s in ("S1", "S2") for p in ("A", "B60", "C", "D") for bb in tf.BASES])
    b.to_csv(uq / "beta_intervals.csv", index=False)


def test_partial_set_gives_pending(tmp_path):
    exp = {"S1": [2013, 2014], "S2": [2013, 2014]}
    for sd in tf.SEEDS_FINAL:                       # S1/A complete
        for y in exp["S1"]:
            _fake_npz(tmp_path, "S1", y, "A", sd)
    for y in exp["S2"]:                             # S2/A: only 4 of 5 seeds
        for sd in tf.SEEDS_FINAL[:4]:
            _fake_npz(tmp_path, "S2", y, "A", sd)
    _pooled(tmp_path, 5)                            # summary file claims 5 seeds everywhere (stale / partial)
    out = tf.t4_build(_ann(), uq=tmp_path, expected=exp)
    full = out[(out.site == "S1") & (out.policy == "A")].iloc[0]
    part = out[(out.site == "S2") & (out.policy == "A")].iloc[0]
    assert full.lambda_SU_P5 == 123.456 and full.SC_M_class == "material" and full["beta_P5_B-CL"] == 1.0
    assert full["beta_P5_B-S2curve1"] == 1.0 and "not applicable" not in " ".join(map(str, full.values))
    assert part.lambda_SU_P5 == tf.PENDING and part.SC_M_class == tf.PENDING and part["beta_P5_B-CL"] == tf.PENDING and part["beta_P50_B-S2curve2"] == tf.PENDING
    others = out[~((out.site == "S1") & (out.policy == "A"))]
    assert (others.lambda_SU_P5 == tf.PENDING).all()      # no numeric interval for any incomplete set (B60 and D have no files at all)
    t5 = tf.t5_build(_ann(), out)
    assert t5.loc[t5.baseline == "B-S2 curve 1", "beta_POLA_S1_P5"].iloc[0] == 1.0 and t5.loc[t5.baseline == "B-CL", "beta_POLA_S2_P5"].iloc[0] == tf.PENDING and t5.loc[t5.baseline == "B-CL", "beta_POLA_S1_P5"].iloc[0] == 1.0


def test_closure_violation_gives_pending(tmp_path):
    exp = {"S1": [2013]}
    for sd in tf.SEEDS_FINAL:
        _fake_npz(tmp_path, "S1", 2013, "A", sd, closure=1e-3 if sd == 55 else 1e-12)
    assert not tf.set_complete("S1", "A", [2013], uq=tmp_path)


def test_no_files_all_pending(tmp_path):
    out = tf.t4_build(_ann(), uq=tmp_path, expected={"S1": [2013, 2014], "S2": [2013, 2014]})
    assert (out.lambda_SU_P5 == tf.PENDING).all() and (out.SC_M_class == tf.PENDING).all()

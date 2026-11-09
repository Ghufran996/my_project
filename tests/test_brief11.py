"""Brief 11: freeze rule of the VT6 calibration (D81), VER-2L row (i) (D82-3), chain comparison with the expected list (D82, D83)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))
from h2su import freeze  # noqa: E402
from h2su.io import RESULTS  # noqa: E402

FROZEN = RESULTS / "vt6_cal_frozen.json"
SHA = RESULTS / "vt6_cal_frozen.json.sha256"
_NEEDED = {"vt6_cal_frozen.json": "step 3 dryer VT6", "vt6_cal_frozen.json.sha256": "step 3 dryer VT6", "vt6b_l15.csv": "step 26 VT6b", "tables/T2_supplementary.csv": "step 47 tables"}
_MISSING = [f"results/{k} ({v})" for k, v in _NEEDED.items() if not (RESULTS / k).exists()]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="result files absent: " + ", ".join(_MISSING) + "; run `python code/run_all.py final`")


def test_content_hash_ignores_created_and_detects_value_change(tmp_path):
    obj = json.loads(FROZEN.read_text(encoding="utf-8"))
    f1 = tmp_path / "a.json"
    f1.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    h1 = freeze.content_hash(f1)
    obj2 = dict(obj, created="2099-01-01 00:00:00")
    f2 = tmp_path / "b.json"
    f2.write_text(json.dumps(obj2), encoding="utf-8")                          # other formatting and other created
    assert freeze.content_hash(f2) == h1
    obj3 = dict(obj, value=obj["value"] * (1 + 1e-12))
    f3 = tmp_path / "c.json"
    f3.write_text(json.dumps(obj3), encoding="utf-8")
    assert freeze.content_hash(f3) != h1


def test_frozen_file_passes_all_three_assertions_and_a_changed_value_fails(tmp_path):
    k, h = freeze.load_frozen_vt6(FROZEN, SHA)
    assert h == freeze.VT6_CONTENT_HASH and f"{k:.7g}" == "0.005723887"
    bad = tmp_path / "f.json"
    obj = json.loads(FROZEN.read_text(encoding="utf-8"))
    obj["value"] *= 1.001
    bad.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(AssertionError):
        freeze.load_frozen_vt6(bad, SHA)
    cr = tmp_path / "g.json"
    cr.write_text(json.dumps(dict(json.loads(FROZEN.read_text(encoding="utf-8")), created="x")), encoding="utf-8")   # created changed: still passes
    assert freeze.load_frozen_vt6(cr, SHA)[1] == h


def test_freeze_refuses_when_the_file_exists(tmp_path, monkeypatch):
    import vt6_run
    f = tmp_path / "vt6_cal_frozen.json"
    f.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(vt6_run, "FROZEN", f)
    monkeypatch.setattr(vt6_run, "_fit_bt2", lambda: (_ for _ in ()).throw(AssertionError("fit must not run")))
    with pytest.raises(SystemExit) as e:
        vt6_run.cal_freeze()
    assert "REFUSED" in str(e.value)
    assert f.read_text(encoding="utf-8") == "{}"


def test_verify_reads_bt2_only_and_never_writes_the_frozen_file(tmp_path, monkeypatch):
    import hashlib

    import vt6_run
    s4 = vt6_run.S4.copy()
    s4.loc[s4.index != "BT2"] = np.nan                                         # every other Table S4 row unavailable
    monkeypatch.setattr(vt6_run, "S4", s4)
    monkeypatch.setattr(vt6_run, "RESULTS", tmp_path)
    before = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    row = vt6_run.cal_verify()
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == before
    k_frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["value"]
    assert row["passed"] and row["rel_diff"] <= 1e-9 and abs(row["k_rerun"] - k_frozen) / k_frozen <= 1e-9
    out = pd.read_csv(tmp_path / "vt6_cal_verify.csv")
    assert list(out.columns) == ["k_frozen", "k_rerun", "rel_diff", "n_evaluations", "residual_s", "passed"] and bool(out.passed.iloc[0])


def test_ver2l_row_i_is_not_reported_as_passed_when_it_cannot_run(tmp_path):
    import ver2l
    r = ver2l.row_i(pre=tmp_path / "does_not_exist.csv")
    assert np.isnan(r["value"]) and r["passed"] is None
    assert r["note"] == "not re-run: the pre-refactor file was deleted in the Brief 6 cleanup; historical result R32 (Brief 3): 0.0 relative difference, identical verdicts"
    df = pd.DataFrame([r, dict(check="(ii) x", value=0.0, tol=1e-9, passed=True, note="n")])
    p = tmp_path / "o.csv"
    df.to_csv(p, index=False)
    back = pd.read_csv(p)
    assert pd.isna(back.loc[0, "value"]) and pd.isna(back.loc[0, "passed"]) and back.loc[1, "passed"] in (True, "True")


def _write(p: Path, df: pd.DataFrame):
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def test_compare_chain_expected_flags_on_a_synthetic_pair(tmp_path, monkeypatch):
    import compare_chain as cc
    old, res = tmp_path / "old", tmp_path / "res"
    monkeypatch.setattr(cc, "RESULTS", res)
    base_inv = pd.DataFrame(dict(id=["a", "PEM_L13_printed_rated_flow_check", "b"], i_min_per_bar=[1.0, np.nan, 3.0]))
    _write(old / "inventory_ranges.csv", base_inv)
    _write(res / "inventory_ranges.csv", base_inv[base_inv.id != "PEM_L13_printed_rated_flow_check"])
    _write(old / "vt6_l15.csv", pd.DataFrame(dict(t=[1.0, 2.0], frozen_sha256=["x", "x"], wall_s=[1.0, 2.0])))
    _write(res / "vt6_l15.csv", pd.DataFrame(dict(t=[1.0, 2.0], frozen_sha256=["y", "y"], wall_s=[9.0, 9.0])))
    _write(old / "ver2l_link.csv", pd.DataFrame(dict(check=["(i)", "(ii)"], value=[0.0, 0.0], passed=[True, True], note=["n", "m"])))
    _write(res / "ver2l_link.csv", pd.DataFrame(dict(check=["(i)", "(ii)"], value=[np.nan, 0.0], passed=[None, True], note=["N", "m"])))
    _write(old / "sens" / "x.csv", pd.DataFrame(dict(a=[1.0, 2.0], peak_rss_mb=[100.0, 100.0])))
    _write(res / "sens" / "x.csv", pd.DataFrame(dict(a=[1.0, 2.0 * (1 + 1e-12)], peak_rss_mb=[200.0, 200.0])))      # inside rtol; resource column excluded
    _write(res / "vt6_cal_verify.csv", pd.DataFrame(dict(passed=[True])))
    _write(res / "trial_cost_chain.csv", pd.DataFrame(dict(id=["T1"], wall_s_median=[0.5])))
    df, nf, nc = cc.compare(old)
    assert nf == 4
    d = {(r.file, r.column): r for r in df.itertuples()}
    assert d[("vt6_l15.csv", "frozen_sha256")].expected == "yes" and ("vt6_l15.csv", "wall_s") not in d
    assert d[("inventory_ranges.csv", "__row_absent__:PEM_L13_printed_rated_flow_check")].expected == "yes"
    assert all(d[("ver2l_link.csv", c)].expected == "yes" and d[("ver2l_link.csv", c)].rows_differing == 1 for c in ("value", "passed", "note"))
    assert d[("vt6_cal_verify.csv", "__new_file__")].expected == "yes"
    assert d[("trial_cost_chain.csv", "__check_max_wall_s_median_lt_2s__")].expected == "yes"
    assert not ((df.expected == "no").any()), df[df.expected == "no"]
    # an unexpected change is flagged, as is a row (ii) change and a slow trial
    _write(res / "sens" / "x.csv", pd.DataFrame(dict(a=[1.0, 2.1], peak_rss_mb=[200.0, 200.0])))
    _write(res / "ver2l_link.csv", pd.DataFrame(dict(check=["(i)", "(ii)"], value=[np.nan, 1e-3], passed=[None, True], note=["N", "m"])))
    _write(res / "trial_cost_chain.csv", pd.DataFrame(dict(id=["T1"], wall_s_median=[2.5])))
    df2, _, _ = cc.compare(old)
    bad = {(r.file, r.column) for r in df2.itertuples() if r.expected == "no"}
    assert ("sens/x.csv", "a") in bad and ("ver2l_link.csv", "value") in bad and ("trial_cost_chain.csv", "__check_max_wall_s_median_lt_2s__") in bad


def test_compare_chain_brief12_floor_dlog10_and_note(tmp_path, monkeypatch):
    """Brief 12 item 3: D86 absolute floor for balance residuals, D87 dlog10 floor cell and note column."""
    import compare_chain as cc
    old, res = tmp_path / "old", tmp_path / "res"
    monkeypatch.setattr(cc, "RESULTS", res)
    note = cc.DLOG10_NOTE
    _write(old / "vt6b_l15.csv", pd.DataFrame(dict(y_min_pred_umol_mol=[5.0, 0.0], dlog10_scaled=[1.5, np.inf], dlog10_unscaled=[1.2, np.inf], bt_balance_err=[9.9e-15, 1.2e-13])))
    _write(res / "vt6b_l15.csv", pd.DataFrame(dict(y_min_pred_umol_mol=[5.0, 0.0], dlog10_scaled=[1.5, 24.28], dlog10_unscaled=[1.2, 24.5], bt_balance_err=[3.8e-14, 1.1e-13], dlog10_note=["", note])))
    _write(res / "vt6_cal_verify.csv", pd.DataFrame(dict(passed=[True])))
    _write(res / "trial_cost_chain.csv", pd.DataFrame(dict(id=["T1"], wall_s_median=[0.5])))
    df, _, _ = cc.compare(old)
    d = {(r.file, r.column): r for r in df.itertuples()}
    assert d[("vt6b_l15.csv", "bt_balance_err")].expected == "yes" and "D86 round-off" in d[("vt6b_l15.csv", "bt_balance_err")].reason     # a 1e-14 residual change passes
    assert d[("vt6b_l15.csv", "dlog10_scaled")].expected == "yes" and d[("vt6b_l15.csv", "dlog10_unscaled")].expected == "yes"
    assert d[("vt6b_l15.csv", "__new_column__:dlog10_note")].expected == "yes" and d[("vt6b_l15.csv", "__check_dlog10_note__")].expected == "yes"
    assert not (df.expected == "no").any(), df[df.expected == "no"]
    # a 1e-6 change of a residual fails; a changed finite dlog10 fails; a note on the wrong row fails
    _write(res / "vt6b_l15.csv", pd.DataFrame(dict(y_min_pred_umol_mol=[5.0, 0.0], dlog10_scaled=[1.6, 24.28], dlog10_unscaled=[1.2, 24.5], bt_balance_err=[3.8e-14, 1.1e-6], dlog10_note=[note, ""])))
    df2, _, _ = cc.compare(old)
    bad = {(r.file, r.column) for r in df2.itertuples() if r.expected == "no"}
    assert ("vt6b_l15.csv", "bt_balance_err") in bad and ("vt6b_l15.csv", "dlog10_scaled") in bad and ("vt6b_l15.csv", "__check_dlog10_note__") in bad


def test_vt6b_note_function_marks_only_zero_prediction_rows():
    import vt6b_run as v
    s = v.dlog10_note(pd.Series([50.0, 0.0, 0.001143, 0.0]))
    assert list(s) == ["", v.DLOG10_NOTE, "", v.DLOG10_NOTE]
    stored = pd.read_csv(Path(__file__).resolve().parents[2] / "results" / "vt6b_l15.csv", keep_default_na=False)
    z = stored.y_min_pred_umol_mol == 0
    assert z.sum() == 1 and (stored.dlog10_note[z] == v.DLOG10_NOTE).all() and (stored.dlog10_note[~z] == "").all()
    sup = pd.read_csv(Path(__file__).resolve().parents[2] / "results" / "tables" / "T2_supplementary.csv", keep_default_na=False)
    assert (sup.dlog10_note != "").sum() == 1

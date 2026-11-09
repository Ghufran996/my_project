"""Tests of code/compare_comsol.py and export_comsol_params.py on SYNTHETIC data only (never reported as results)."""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "src"))

import compare_comsol as cc  # noqa: E402
from export_comsol_params import export  # noqa: E402
from h2su.io import load_registry  # noqa: E402

# Skipped when the shipped COMSOL parameter files are absent (not part of the code package; written by `python code/export_comsol_params.py`, not a run_all.py step).
_COMSOL = CODE.parent / "COMSOL_MATLAB"
pytestmark = pytest.mark.skipif(not (_COMSOL / "params_comsol.csv").exists() or not (_COMSOL / "cases_comsol.csv").exists(),
                                reason="COMSOL_MATLAB/params_comsol.csv and cases_comsol.csv are absent; they are written by `python code/export_comsol_params.py` (not a run_all.py step)")

FIX = Path(__file__).resolve().parent / "fixtures" / "comsol_synthetic"


def test_fixture_is_marked_synthetic():
    for f in FIX.rglob("*.csv"):
        assert "SYNTHETIC" in f.name or f.name == "ver6_events.csv"
        assert f.read_text().splitlines()[0].startswith("# SYNTHETIC")


def test_compare_known_differences():
    df = cc.compare(FIX / "comsol_outputs", FIX / "python_ver6")
    assert len(df) == 1
    r = df.iloc[0]
    assert r["status"] == "OK"
    assert r["nrmse_y_O2_out"] == pytest.approx(0.1, rel=1e-9)
    assert r["nrmse_T_out_K"] == pytest.approx(0.05, rel=1e-9)
    assert math.isnan(r["nrmse_y_H2O_out"])            # not modelled in the fixture
    assert r["diff_t_spec_s"] == pytest.approx(1.0)
    assert r["RE_m_H2_consumed_kg"] == pytest.approx(0.1, rel=1e-9)
    assert math.isnan(r["RE_m_H2_vented_kg"])


def test_main_writes_csv_and_exit_codes(tmp_path):
    out = tmp_path / "cmp.csv"
    assert cc.main(["--comsol-dir", str(FIX / "comsol_outputs"), "--python-dir", str(FIX / "python_ver6"), "--out", str(out)]) == 0
    assert out.exists() and len(pd.read_csv(out)) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cc.main(["--comsol-dir", str(empty), "--out", str(tmp_path / "none.csv")]) == 2
    assert not (tmp_path / "none.csv").exists()


def test_no_reference_is_reported(tmp_path):
    ref = tmp_path / "pyempty"
    ref.mkdir()
    df = cc.compare(FIX / "comsol_outputs", ref)
    assert df.iloc[0]["status"].startswith("NO_PYTHON_REFERENCE")


def test_export_round_trip(tmp_path):
    """params_comsol.csv reproduces the registry values it takes from parameters.csv; every row has a source."""
    params, cases = export(tmp_path)
    back = pd.read_csv(tmp_path / "params_comsol.csv")
    assert (back["source_id"].astype(str).str.strip() != "").all()
    reg = load_registry().set_index("name")
    for comsol_name, reg_name in (("spec_O2_umol_mol", "iso_O2_max"), ("spec_N2_umol_mol", "iso_N2_max"),
                                  ("spec_H2O_umol_mol", "iso_H2O_max")):
        v = float(back[(back.scope == "global") & (back.name == comsol_name)]["value"].iloc[0])
        assert v == float(reg.loc[reg_name, "value_or_range"])
    # the shipped files equal a fresh export
    for name in ("params_comsol.csv", "cases_comsol.csv"):
        shipped = pd.read_csv(CODE.parent / "COMSOL_MATLAB" / name)
        fresh = pd.read_csv(tmp_path / name)
        pd.testing.assert_frame_equal(shipped, fresh)


def test_script_runs_without_comsol_output(tmp_path):
    r = subprocess.run([sys.executable, "-I", str(CODE / "compare_comsol.py"), "--comsol-dir", str(tmp_path),
                        "--out", str(tmp_path / "x.csv")], capture_output=True, text=True)
    assert r.returncode == 2
    assert "No COMSOL output found" in r.stdout

"""Code-to-code comparison of the user-run COMSOL results (VER-6c) with the Python 1-D reference (VER-6).

    python code/compare_comsol.py [--comsol-dir DIR] [--python-dir DIR] [--out FILE] [--python-model NAME]

Reads COMSOL_MATLAB/outputs/<case>__comsol__<grid>.csv (+ ver6_events.csv) and pairs every file with
results/ver6/<case>__<python-model>__<grid>.csv (+ ver6_events.csv) by case_id and grid_id.  Per pair:
  * NRMSE of the outlet mole fractions (O2, H2O, N2, H2) and of the outlet temperature,
    NRMSE = sqrt(mean((comsol - python)^2)) / (max(python) - min(python)); NaN where the reference range is zero
    or the quantity is not modelled;
  * differences in time to specification (t_spec_s) and in breakthrough time (t_break_s), COMSOL - Python, in s;
  * relative error of the event H2 masses (consumed, vented, purge, re-pressurisation), (COMSOL - Python)/Python.
Result: results/ver6c_comsol_compare.csv.  Exit code 2 with a plain message when no COMSOL output exists.
Verification-case series (ver_*) hold the normalised tracer in the y_O2_out column (see code/configs/ver6_schema.csv).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from h2su.io import RESULTS, ROOT  # noqa: E402

COMSOL_OUT = ROOT / "COMSOL_MATLAB" / "outputs"
PY_DIR = RESULTS / "ver6"
OUT_FILE = RESULTS / "ver6c_comsol_compare.csv"
SERIES_COLS = ["y_O2_out", "y_H2O_out", "y_N2_out", "y_H2_out", "T_out_K"]
EVENT_RE = ["m_H2_consumed_kg", "m_H2_vented_kg", "m_H2_purge_kg", "m_H2_repress_kg"]
EVENT_DIFF = ["t_spec_s", "t_break_s"]


class NoComsolOutput(RuntimeError):
    pass


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def nrmse(ref: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """(NRMSE normalised by the reference range, absolute RMSE); NaN if not defined."""
    m = np.isfinite(ref) & np.isfinite(test)
    if m.sum() == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((test[m] - ref[m]) ** 2)))
    rng = float(np.max(ref[m]) - np.min(ref[m]))
    return (rmse / rng if rng > 0 else float("nan")), rmse


def compare(comsol_dir: Path = COMSOL_OUT, python_dir: Path = PY_DIR, python_model: str = "python_mol") -> pd.DataFrame:
    files = sorted(Path(comsol_dir).glob("*__comsol__*.csv"))
    if not files:
        raise NoComsolOutput(
            f"No COMSOL output found in {comsol_dir}.\n"
            "Run SMOKE_TEST and RUN_ALL in COMSOL_MATLAB (see COMSOL_MATLAB/README.md) and place the 'outputs' folder "
            "back at COMSOL_MATLAB/outputs, then run this script again.")
    ev_c = _read(Path(comsol_dir) / "ver6_events.csv") if (Path(comsol_dir) / "ver6_events.csv").exists() else pd.DataFrame()
    ev_p_path = Path(python_dir) / "ver6_events.csv"
    ev_p = _read(ev_p_path) if ev_p_path.exists() else pd.DataFrame()
    rows = []
    for f in files:
        c = _read(f)
        case_id, grid_id = str(c["case_id"].iloc[0]), str(c["grid_id"].iloc[0])
        row = dict(case_id=case_id, grid_id=grid_id, rtol=float(c["rtol"].iloc[0]), n_points=len(c), status="OK")
        pf = Path(python_dir) / f"{case_id}__{python_model}__{grid_id}.csv"
        if not pf.exists():
            row["status"] = f"NO_PYTHON_REFERENCE ({pf.name})"
            rows.append(row)
            continue
        p = _read(pf)
        tc, tp = c["t_s"].to_numpy(float), p["t_s"].to_numpy(float)
        same_grid = len(tc) == len(tp) and np.allclose(tc, tp, rtol=1e-9, atol=1e-9)
        if not same_grid:
            row["status"] = "TIME_GRID_DIFFERS (COMSOL interpolated onto the Python grid)"
        for col in SERIES_COLS:
            ref = p[col].to_numpy(float)
            val = c[col].to_numpy(float)
            if not same_grid:
                val = np.interp(tp, tc, val, left=np.nan, right=np.nan)
            row[f"nrmse_{col}"], row[f"rmse_{col}"] = nrmse(ref, val)
        ec =ev_c[(ev_c.case_id == case_id) & (ev_c.grid_id == grid_id)] if len(ev_c) else pd.DataFrame()
        ep = (ev_p[(ev_p.case_id == case_id) & (ev_p.grid_id == grid_id) & (ev_p.model == python_model)]
              if len(ev_p) else pd.DataFrame())
        if ec.empty or ep.empty:
            row["status"] += "; EVENTS_MISSING"
        else:
            ec, ep = ec.iloc[0], ep.iloc[0]
            for k in EVENT_DIFF:
                row[f"diff_{k}"] = float(ec[k] - ep[k])
            for k in EVENT_RE:
                ref = float(ep[k])
                row[f"RE_{k}"] = float((ec[k] - ref) / ref) if np.isfinite(ref) and ref != 0 else float("nan")
            row["comsol_mass_balance_err"] = float(ec["mass_balance_err"])
            row["comsol_wall_s"] = float(ec["wall_s"])
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--comsol-dir", default=str(COMSOL_OUT))
    ap.add_argument("--python-dir", default=str(PY_DIR))
    ap.add_argument("--out", default=str(OUT_FILE))
    ap.add_argument("--python-model", default="python_mol")
    a = ap.parse_args(argv)
    try:
        df = compare(Path(a.comsol_dir), Path(a.python_dir), a.python_model)
    except NoComsolOutput as e:
        print(str(e))
        return 2
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    show = [c for c in df.columns if c in ("case_id", "grid_id", "status") or c.startswith("nrmse_") or c.startswith("RE_")
            or c.startswith("diff_")]
    print(df[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

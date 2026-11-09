"""Briefs 9, 11 and 12: numeric comparison of CSV copies taken before `run_all.py final` with the files the chain wrote.

    python code/compare_chain.py COPY_DIR [OUT_CSV]

COPY_DIR holds the copies under the same relative paths as results/ (and `sha256_list.txt`). For every copied CSV the file in results/ is read and compared column by column:
numeric columns with rtol 1e-9 (atol 0; NaN equals NaN), text columns exactly. Resource columns are excluded (name contains 'wall', or equals 'peak_rss_mb'; in cost_summary.csv the
columns ending '_s'). A leading '# ...' comment line (figure data notes) is compared as text. CSV files of the compared scope (results/*.csv and the sub-folders tables, figures/data,
nominal, uq, resolution, sens, policy) that are absent from COPY_DIR are reported as new files (comparison outputs `chain_b*` are ignored).

Output (default results/chain_b12_diff.csv): one row per file and column that differs, with the columns
file, column, rows_differing, max_relative_difference, expected, reason.
Structural differences use the column names __missing_file__, __rows__, __columns__, __comment_line__, __new_file__, __row_absent__:<id>. `expected` is 'yes' only for the changes
of the EXPECTED list below (Brief 11 item 5: D81, D82, Brief 9 items 1-2) and 'no' otherwise; one check row verifies that the trial re-measurement stays in the same Saltelli
branch of the design decision (max wall_s_median of trial_cost_chain.csv < 2 s). Exit code 1 if any row has expected = no.

Brief 12 additions (default output results/chain_b12_diff.csv):
 * D86: columns whose name contains 'balance_err' or starts with 'closure_' are residuals of mass balances; they are compared with an absolute floor instead of rtol: a row passes when
   |new - old| <= 1e-12 or both values are <= 1e-9; passing rows are reported as expected = yes, reason "D86 round-off"; a row beyond the floor is expected = no.
 * D87: dlog10_scaled and dlog10_unscaled of a row with y_min_pred_umol_mol = 0 (new file) changed from inf to the value at the 1e-30 floor: expected yes only for such rows and only when the
   old value was inf; the new column dlog10_note is expected yes (new column) and is checked: the fixed note on the rows with y_min_pred = 0, empty elsewhere.
 * D88: tables/T2_supplementary.csv frozen_sha256; D89: resolution/cost.csv note.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from h2su.io import RESULTS  # noqa: E402

RTOL = 1e-9
ABS_FLOOR = 1e-12      # D86: absolute floor for balance residuals
ABS_BOTH_BELOW = 1e-9
DLOG10_NOTE = "y_min_pred = 0; dlog10 evaluated at the 1e-30 floor, lower bound only"
OUT_COLS = ["file", "column", "rows_differing", "max_relative_difference", "expected", "reason"]
SCOPE_DIRS = ("tables", "figures/data", "nominal", "uq", "resolution", "sens", "policy")
TRIAL_WALL_LIMIT_S = 2.0

# (file, column or '*') -> (reason, allowed row indices or None). A difference is expected only if it matches an entry.
EXPECTED = {
    ("vt6_l15.csv", "frozen_sha256"): ("D81: the column carries the content hash (created excluded) instead of the file hash", None),
    ("tables/T2_supplementary.csv", "frozen_sha256"): ("D81: copy of vt6_l15.csv frozen_sha256 inside the supplementary table (derived, same change)", None),
    ("ver2l_link.csv", "value"): ("D82-3: row (i) not re-run, value empty", {0}),
    ("ver2l_link.csv", "passed"): ("D82-3: row (i) not re-run, pass flag empty", {0}),
    ("ver2l_link.csv", "note"): ("D82-3: row (i) note cites the historical R32 result", {0}),
    ("c3_load_shape.csv", "*"): ("Brief 9 item 1", None),
    ("resolution/c3_load_shape.csv", "*"): ("Brief 9 item 1", None),
    ("resolution/c3_load_shape_summary.csv", "*"): ("Brief 9 item 1", None),
    ("resolution/cost.csv", "note"): ("Brief 9 item 2 and D89: note of the POL-A 1-min row, built from the data", None),
    ("resolution/cost_summary.csv", "*"): ("Brief 9 item 2", None),
}
# D87: the cells (file, column) whose old value is inf and whose new value is the 1e-30-floor value; allowed rows = rows where the NEW file has y_min_pred_umol_mol == 0
DLOG10_FLOOR_CELLS = {("vt6b_l15.csv", "dlog10_scaled"), ("vt6b_l15.csv", "dlog10_unscaled"), ("tables/T2_supplementary.csv", "dlog10_scaled"), ("tables/T2_supplementary.csv", "dlog10_unscaled")}
# D87: columns that exist only in the new file
EXPECTED_NEW_COLUMNS = {
    ("vt6b_l15.csv", "dlog10_note"): "D87: note column for rows with y_min_pred = 0",
    ("tables/T2_supplementary.csv", "dlog10_note"): "D87: note column for rows with y_min_pred = 0 (derived from vt6b_l15.csv)",
}
EXPECTED_NEW_FILES = {
    "vt6_cal_verify.csv": "D81: verify mode of the VT6 calibration (Brief 11 item 1)",
    "trial_cost_chain.csv": "D82-2: re-measurement of the trial cost written by the chain (Brief 11 item 3)",
    "resolution/c3_load_shape_summary.csv": "Brief 9 item 1",
    "resolution/cost_summary.csv": "Brief 9 item 2",
}
# row removals accepted in a file: file -> (id column, id value, reason)
EXPECTED_ROW_REMOVALS = {
    "inventory_ranges.csv": ("id", "PEM_L13_printed_rated_flow_check", "D82-4: BLOCKED provenance note without producer in code/, not a code output"),
}


def _read(path: Path):
    first = path.open(encoding="utf-8").readline()
    comment = first.rstrip("\n") if first.startswith("#") else None
    df = pd.read_csv(path, skiprows=1 if comment is not None else 0)
    return comment, df


def _excluded(file: str, col: str) -> bool:
    c = col.lower()
    return "wall" in c or c == "peak_rss_mb" or (Path(file).name == "cost_summary.csv" and col.endswith("_s"))


def _row(file, column, n, mx, expected, reason):
    return dict(file=file, column=column, rows_differing=n, max_relative_difference=mx, expected=expected, reason=reason)


def _is_residual(col: str) -> bool:
    c = col.lower()
    return "balance_err" in c or c.startswith("closure_")


def _floor_ok(xv, yv):
    """D86: True where |new - old| <= ABS_FLOOR or both values are <= ABS_BOTH_BELOW (NaN equal NaN)."""
    with np.errstate(invalid="ignore"):
        return (np.abs(xv - yv) <= ABS_FLOOR) | ((np.abs(xv) <= ABS_BOTH_BELOW) & (np.abs(yv) <= ABS_BOTH_BELOW)) | (np.isnan(xv) & np.isnan(yv))


def _classify(file, column, idx):
    """Return (expected, reason) for a column difference at the row positions idx."""
    for key in ((file, column), (file, "*"), (Path(file).name, column), (Path(file).name, "*")):
        if key in EXPECTED:
            reason, allowed = EXPECTED[key]
            if allowed is None or set(map(int, idx)) <= allowed:
                return "yes", reason
    return "no", ""


def compare(copy_dir: Path):
    rows, n_files, n_cols = [], 0, 0
    copied = set()
    for old in sorted(copy_dir.rglob("*.csv")):
        rel = old.relative_to(copy_dir).as_posix()
        copied.add(rel)
        new = RESULTS / rel
        n_files += 1
        if not new.exists():
            rows.append(_row(rel, "__missing_file__", -1, np.nan, "no", ""))
            continue
        c0, a = _read(old)
        c1, b = _read(new)
        if rel in EXPECTED_ROW_REMOVALS:
            idc, idv, why = EXPECTED_ROW_REMOVALS[rel]
            if idc in a.columns and (a[idc] == idv).any() and not (b[idc] == idv).any():
                rows.append(_row(rel, f"__row_absent__:{idv}", 1, np.nan, "yes", why))
                a = a[a[idc] != idv].reset_index(drop=True)
        if c0 != c1:
            rows.append(_row(rel, "__comment_line__", 1, np.nan, "no", ""))
        new_cols = [c for c in b.columns if c not in a.columns]
        if [c for c in a.columns if c not in b.columns] or [c for c in new_cols if (rel, c) not in EXPECTED_NEW_COLUMNS]:
            rows.append(_row(rel, "__columns__", -1, np.nan, "no", ""))
        elif list(a.columns) != [c for c in b.columns if c in a.columns]:
            rows.append(_row(rel, "__columns__", -1, np.nan, "no", "column order changed"))
        for c in new_cols:
            if (rel, c) in EXPECTED_NEW_COLUMNS:
                rows.append(_row(rel, f"__new_column__:{c}", -1, np.nan, "yes", EXPECTED_NEW_COLUMNS[(rel, c)]))
                if c == "dlog10_note" and "y_min_pred_umol_mol" in b.columns:
                    z = b["y_min_pred_umol_mol"].to_numpy(float) == 0.0
                    nn = b[c].fillna("").astype(str).to_numpy()
                    okn = bool(np.all(nn[z] == DLOG10_NOTE) and np.all(nn[~z] == ""))
                    rows.append(_row(rel, "__check_dlog10_note__", 0 if okn else int((np.where(z, nn != DLOG10_NOTE, nn != "")).sum()), np.nan, "yes" if okn else "no",
                                     f"D87: note on the {int(z.sum())} row(s) with y_min_pred = 0, empty on the others" if okn else "D87: note missing or misplaced"))
        if len(a) != len(b):
            rows.append(_row(rel, "__rows__", abs(len(a) - len(b)), np.nan, "no", ""))
            continue
        for col in a.columns:
            if col not in b.columns or _excluded(rel, col):
                continue
            n_cols += 1
            x, y = a[col], b[col]
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
                xv, yv = x.to_numpy(float), y.to_numpy(float)
                both_nan = np.isnan(xv) & np.isnan(yv)
                same_inf = np.isinf(xv) & np.isinf(yv) & (np.sign(xv) == np.sign(yv))
                with np.errstate(invalid="ignore", divide="ignore"):
                    den = np.maximum(np.abs(xv), np.abs(yv))
                    rv = np.where(xv == yv, 0.0, np.abs(xv - yv) / den)
                bad = ~(both_nan | same_inf) & ~(rv <= RTOL)
                bad |= np.isnan(xv) ^ np.isnan(yv)
                if bad.any() and _is_residual(col):
                    ok = _floor_ok(xv, yv)
                    nbad = bad & ~ok
                    with np.errstate(invalid="ignore"):
                        mad = float(np.nanmax(np.abs(xv - yv)[bad]))
                    if nbad.any():
                        rows.append(_row(rel, col, int(nbad.sum()), mad, "no", f"D86 floor exceeded (max absolute difference {mad:.3g})"))
                    if (bad & ok).any():
                        rows.append(_row(rel, col, int((bad & ok).sum()), mad, "yes", f"D86 round-off (max absolute difference {mad:.3g}; floor {ABS_FLOOR:g} or both <= {ABS_BOTH_BELOW:g})"))
                elif bad.any() and (rel, col) in DLOG10_FLOOR_CELLS and "y_min_pred_umol_mol" in b.columns:
                    z = b["y_min_pred_umol_mol"].to_numpy(float) == 0.0
                    fine = bad & z & np.isinf(xv)
                    other = bad & ~fine
                    if other.any():
                        rows.append(_row(rel, col, int(other.sum()), np.inf, "no", ""))
                    if fine.any():
                        rows.append(_row(rel, col, int(fine.sum()), np.inf, "yes", "D87: stored inf (written before the 1e-30 floor) replaced by the value at the floor, y_min_pred = 0; lower bound only"))
                elif bad.any():
                    fin = rv[bad & np.isfinite(rv)]
                    ex, why = _classify(rel, col, np.flatnonzero(bad))
                    rows.append(_row(rel, col, int(bad.sum()), float(fin.max()) if fin.size else np.inf, ex, why))
            else:
                bad = ~((x.astype(str) == y.astype(str)) | (x.isna() & y.isna()))
                if bad.any():
                    ex, why = _classify(rel, col, np.flatnonzero(bad.to_numpy()))
                    rows.append(_row(rel, col, int(bad.sum()), np.nan, ex, why))
    # new files in the compared scope
    cands = [p for p in RESULTS.glob("*.csv")] + [p for d in SCOPE_DIRS for p in (RESULTS / d).glob("*.csv")]
    for p in sorted(cands):
        rel = p.relative_to(RESULTS).as_posix()
        if rel in copied or p.name.startswith("chain_b"):
            continue
        if rel in EXPECTED_NEW_FILES:
            rows.append(_row(rel, "__new_file__", -1, np.nan, "yes", EXPECTED_NEW_FILES[rel]))
        else:
            rows.append(_row(rel, "__new_file__", -1, np.nan, "no", ""))
    # check row: trial re-measurement stays in the same Saltelli branch
    tc = RESULTS / "trial_cost_chain.csv"
    if tc.exists():
        w = pd.read_csv(tc)
        m = float(w.wall_s_median.max())
        ok = m < TRIAL_WALL_LIMIT_S
        rows.append(_row("trial_cost_chain.csv", "__check_max_wall_s_median_lt_2s__", 0 if ok else 1, m, "yes" if ok else "no",
                         f"D82-2: max wall_s_median {m:.4f} s {'<' if ok else '>='} {TRIAL_WALL_LIMIT_S} s (direct Saltelli branch {'unchanged' if ok else 'CHANGED'})"))
    else:
        rows.append(_row("trial_cost_chain.csv", "__check_max_wall_s_median_lt_2s__", 1, np.nan, "no", "file absent"))
    return pd.DataFrame(rows, columns=OUT_COLS), n_files, n_cols


if __name__ == "__main__":
    cd = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "chain_b12_diff.csv"
    df, nf, nc = compare(cd)
    df.to_csv(out, index=False, float_format="%.6g")
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 80)
    n_no = int((df.expected == "no").sum())
    print(f"files compared {nf}; columns compared {nc} (resource columns excluded); rtol {RTOL}; rows in output: {len(df)}; rows with expected = no: {n_no}")
    if len(df):
        print(df.to_string(index=False))
    if n_no:
        raise SystemExit(1)

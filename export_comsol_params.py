"""Export the COMSOL package inputs from the registry and the VER-6 case definitions.

    python code/export_comsol_params.py

Writes COMSOL_MATLAB/params_comsol.csv (long format: scope,name,value,unit,source_id,location,extraction,role,note)
and COMSOL_MATLAB/cases_comsol.csv (case_id,model_function,group,status,blocked_by,description).
No other value source exists: every row comes from code/configs/parameters.csv or from h2su.ver6_cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from h2su import ver6_cases  # noqa: E402
from h2su.io import ROOT  # noqa: E402

PKG = ROOT / "COMSOL_MATLAB"
COLS = ["scope", "name", "value", "unit", "source_id", "location", "extraction", "role", "note"]


def export(pkg_dir: Path = PKG) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases, rows = ver6_cases.build()
    params = pd.DataFrame(rows, columns=COLS)
    case_tab = pd.DataFrame([dict(case_id=c.case_id, model_function=c.model_function, group=c.group, status=c.status,
                                  blocked_by=c.blocked_by, description=c.description) for c in cases])
    bad = params[(params["source_id"].isna()) | (params["source_id"].astype(str).str.strip() == "")]
    if not bad.empty:
        raise ValueError(f"rows without source: {bad['name'].tolist()}")
    pkg_dir.mkdir(parents=True, exist_ok=True)
    params.to_csv(pkg_dir / "params_comsol.csv", index=False, float_format="%.15g")
    case_tab.to_csv(pkg_dir / "cases_comsol.csv", index=False)
    return params, case_tab


if __name__ == "__main__":
    p, c = export()
    print(f"params_comsol.csv: {len(p)} rows; cases_comsol.csv: {len(c)} cases "
          f"({(c.status == 'READY').sum()} READY, {(c.status != 'READY').sum()} BLOCKED)")
    print(c[["case_id", "status", "blocked_by"]].to_string(index=False))

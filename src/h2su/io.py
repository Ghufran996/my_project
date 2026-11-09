"""Paths, parameter registry and BSRN file readers."""
from __future__ import annotations

import io as _io
import os
from pathlib import Path

import pandas as pd

# code/src/h2su/io.py -> project root is three levels up from code/
CODE_DIR = Path(__file__).resolve().parents[2]
ROOT = CODE_DIR.parent
CONFIGS = CODE_DIR / "configs"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def load_registry() -> pd.DataFrame:
    """Parameter registry (name, source, location, role ...). Every value carries its source."""
    return pd.read_csv(CONFIGS / "parameters.csv")


def param(name: str):
    """Return the numeric value of a scalar registry entry (raises if non-numeric or missing)."""
    df = load_registry()
    row = df.loc[df["name"] == name]
    if row.empty:
        raise KeyError(f"parameter {name!r} not in registry")
    return float(row.iloc[0]["value_or_range"])


def roles(name_prefix: str = "") -> pd.DataFrame:
    df = load_registry()
    return df[df["name"].str.startswith(name_prefix)][["name", "role", "source_id", "location"]]


def read_bsrn_tab(path: os.PathLike | str) -> pd.DataFrame:
    """Read a PANGAEA BSRN monthly tab file (header block ends with '*/')."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    _, _, body = txt.partition("*/\n")
    df = pd.read_csv(_io.StringIO(body), sep="\t")
    tcol = df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    return df.set_index(tcol)

"""Reader for AEMO MMSDM 'SQLLoader' CSV zips (C/I/D record format)."""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd


def read_mms_zip(path: Path | str) -> pd.DataFrame:
    z = zipfile.ZipFile(path)
    txt = z.read(z.namelist()[0]).decode("latin-1")
    hdr, rows = None, []
    for rec in csv.reader(io.StringIO(txt)):
        if not rec:
            continue
        if rec[0] == "I":
            hdr = [h.strip() for h in rec[4:]]
        elif rec[0] == "D":
            rows.append(rec[4:4 + len(hdr)])
    return pd.DataFrame(rows, columns=hdr)

"""Content hash of a frozen-calibration JSON file (D81).

The hash is the SHA-256 of the canonical JSON text of the file content without the field "created"
(json.dumps(obj, sort_keys=True, separators=(",", ":")), UTF-8). The creation time stamp is therefore not part of the freeze;
every other field is.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def content_hash_obj(obj: dict) -> str:
    body = {k: v for k, v in obj.items() if k != "created"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def content_hash(path) -> str:
    return content_hash_obj(json.loads(Path(path).read_text(encoding="utf-8")))


# D81: the frozen VT6 calibration (BT2, R39). The file written by the chain at 2026-10-31 03:40:49 is the adopted artifact; its content hash is fixed here once.
VT6_CONTENT_HASH = "631609b3feb27a4fa36e0f21ac2dffd8c2d7120e11f7d11c27b5f31a2efd5974"
VT6_R39_VALUE_7G = "0.005723887"


def load_frozen_vt6(frozen_path, sha_path):
    """Return (k_LDF, content_hash) after asserting: content hash of the file == value in the .sha256 file == the hard-coded constant,
    and the calibrated value printed at 7 significant digits equals the value logged in R39."""
    h = content_hash(frozen_path)
    logged = Path(sha_path).read_text(encoding="utf-8").split()[0]
    assert h == logged == VT6_CONTENT_HASH, (h, logged, VT6_CONTENT_HASH)
    value = float(json.loads(Path(frozen_path).read_text(encoding="utf-8"))["value"])
    assert f"{value:.7g}" == VT6_R39_VALUE_7G, (value, VT6_R39_VALUE_7G)
    return value, h

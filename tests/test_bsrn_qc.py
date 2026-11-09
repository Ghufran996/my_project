"""Unit test of the L43 closure formula on three hand-computed minutes (+ limits and fill logic on synthetic data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from h2su import bsrn  # noqa: E402


def test_closure_three_hand_computed_minutes():
    # AU = 1 so S_a = 1368.  Minute 1: SZA 30 deg, mu0 = 0.8660254; DNI 800, DIF 100 -> Sum = 800*0.8660254 + 100 = 792.82032;
    #   GHI 792.82 -> ratio 1.0000 (pass); GHI 900 -> ratio 1.13522 > 1.08 (fail).
    # Minute 3: SZA 80 deg, mu0 = 0.17364818; DNI 300, DIF 60 -> Sum = 52.094454 + 60 = 112.094454;
    #   GHI 125 -> ratio 1.11513 <= 1.15 (pass); GHI 130 -> ratio 1.15976 > 1.15 (fail).
    ghi = np.array([792.82032, 900.0, 125.0, 130.0])
    dni = np.array([800.0, 800.0, 300.0, 300.0])
    dif = np.array([100.0, 100.0, 60.0, 60.0])
    z = np.array([30.0, 30.0, 80.0, 80.0])
    f = bsrn.qc_flags(ghi, dni, dif, z, np.ones(4))
    assert list(f["closure_ok"]) == [True, False, True, False]
    assert np.allclose(f["mu0"], [0.8660254, 0.8660254, 0.17364818, 0.17364818], atol=1e-7)


def test_physical_limits_and_missing():
    f = bsrn.qc_flags([np.nan, 2100.0, 500.0], [500.0, 500.0, 1500.0], [50.0, 50.0, 50.0], [20.0, 20.0, 20.0], np.ones(3))
    assert list(f["ghi_ok"]) == [False, False, True]      # missing; above S_a*1.5*mu0^1.2+100 = 1368*1.5*0.9267+100 = 2002 at SZA 20 deg
    assert list(f["comp_ok"]) == [True, True, False]      # DNI 1500 > S_a
    assert not f["valid3"].any()


def test_fill_rule_flags():
    idx = pd.date_range("2015-06-21 10:00", periods=4, freq="1min", tz="UTC")
    z = np.full(4, 30.0)
    geo = pd.DataFrame({"zenith": z, "apparent_zenith": z, "au": np.ones(4)}, index=idx)
    mu0 = np.cos(np.radians(30.0))
    df = pd.DataFrame({"ghi": [100 + 800 * mu0, np.nan, 700.0, np.nan], "dni": [800.0, 800.0, np.nan, np.nan],
                       "dhi": [100.0, 100.0, np.nan, np.nan], "ta_c": 20.0}, index=idx)
    out = bsrn.fill_year(df, geo)
    assert list(out["fill_flag"]) == [0, 1, 2, 3]
    assert abs(out["ghi"].iloc[1] - (100 + 800 * mu0)) < 1e-9       # closure
    assert out["dni"].iloc[2] > 0 and out["dhi"].iloc[2] > 0          # Erbs components

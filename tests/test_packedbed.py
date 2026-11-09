"""Code-verification tests of the 1-D packed-bed solver on analytic and SYNTHETIC (arbitrary test numbers, no physical claim) cases."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from h2su import packedbed as pb


def test_steady_plug_flow_and_order():
    errs = {}
    for N in (50, 100, 200):
        f = pb.Field("c", A=0.4, V=0.2, D=0.0, F_in=1.0, F0=0.0, scale=1.0)
        bed = pb.Bed(1.0, [f], lambda F: -0.4 * F)
        s = pb.solve(bed, N, np.array([0.0, 30.0]), rtol=1e-8)
        errs[N] = abs((1 - s["outlet"][0, -1]) - (1 - np.exp(-2.0)))
    assert errs[200] < 1e-2 * 0.0864665 * 10          # well below 1 %
    p = np.log2(errs[100] / errs[200])
    assert p > 1.5                                      # MUSCL: observed order close to 2


def test_linear_isotherm_first_moment_without_dispersion():
    f = pb.Field("c", A=0.4 + 0.6 * 5.0, V=0.2, D=0.0, F_in=1.0, F0=0.0)
    bed = pb.Bed(1.0, [f], lambda F: 0 * F)
    t = np.arange(0, 40.001, 0.5)
    s = pb.solve(bed, 400, t, rtol=1e-8)
    m = np.trapezoid(1 - s["outlet"][0], t)
    assert m == pytest.approx(17.0, rel=1e-2)


def _reacting(N=60):
    """SYNTHETIC two-field adiabatic reacting bed: species y (source -r) and temperature T (source +dH r)."""
    A_y, A_T, V_y, V_T, dH = 0.4, 50.0, 0.2, 5.0, 300.0
    def src(F):
        y, T = F
        r = 2.0 * np.exp(-1.0 * (300.0 / T - 1.0)) * y
        return np.array([-r, dH * r])
    fy = pb.Field("y", A_y, V_y, 0.4 * 1e-3, 1.0, 0.0, 1.0)
    fT = pb.Field("T", A_T, V_T, 1e-2, 300.0, 300.0, 300.0)
    return pb.Bed(1.0, [fy, fT], src), dH


def test_mass_and_energy_conservation_synthetic():
    bed, dH = _reacting()
    s = pb.solve(bed, 60, np.linspace(0, 20, 41), rtol=1e-8)
    assert s["balance_err"].max() < 1e-6                 # both fields close (relative to the largest term)
    # energy released equals dH x moles consumed (source integrals of the two fields)
    assert s["I_src"][1, -1] == pytest.approx(-dH * s["I_src"][0, -1], rel=1e-9)


@pytest.mark.parametrize("scheme", ["upwind", "muscl"])
def test_balance_for_both_schemes(scheme):
    bed, _ = _reacting(40)
    bed.scheme = scheme
    s = pb.solve(bed, 40, np.linspace(0, 10, 11), rtol=1e-8)
    assert s["balance_err"].max() < 1e-6

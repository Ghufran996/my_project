"""Tests of the non-trace regeneration form (Brief 5 item 5, D53): y <= 1, water balance, reduction to the constant-flux trace form at y -> 0."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from h2su import dryer as dr  # noqa: E402
from h2su import dryer_regen as rg  # noqa: E402
from h2su.io import load_registry  # noqa: E402


def lab_regen_params(q0, N0, y0):
    df, iso = load_registry(), dr.iso_params()
    L, A = dr.lab_geometry(df)
    return dict(L_m=L, A_m2=A, eps=0.4, rho_bed_kg_m3=790.0, p_Pa=0.025e5, T_K=301.65, N0_mol_m2_s=N0, kLDF_1_s=0.005723887, q_init_mol_kg=q0, y_init=y0), iso


def test_y_bounded_and_water_balance_saturated_start():
    p, iso = lab_regen_params(q0=7.9, N0=0.0142, y0=3.4e-3)
    t = np.arange(0.0, 1800.0 + 1e-9, 30.0)
    r = rg.regen_solve(p, iso, 40, t, rtol=1e-6)
    assert r["y"].max() <= 1.0 + 1e-9 and r["y"].min() >= -1e-9
    assert r["water_balance_rel"].max() < 1e-6
    assert r["total_balance_rel"].max() < 1e-6
    assert r["q"][:, -1].mean() < 7.9                      # desorption happened


def test_reduces_to_trace_form_at_small_y():
    """Very small loading and a large purge: y << 1 everywhere; the outlet water flux must agree with the trace-species constant-flux model of dryer.py."""
    q0, N0 = 1e-4, 50.0
    p, iso = lab_regen_params(q0=q0, N0=N0, y0=0.0)
    t = np.arange(0.0, 20.0 + 1e-9, 0.5)
    N = 40
    r = rg.regen_solve(p, iso, N, t, rtol=1e-8)
    assert r["y"].max() < 1e-2
    pt = dict(L_m=p["L_m"], A_m2=p["A_m2"], eps=0.4, rho_bed_kg_m3=790.0, Dax_m2_s=0.0, N_flux_mol_m2_s=N0, p_Pa=p["p_Pa"], T_K=p["T_K"], yW_in_umol_mol=0.0, yW_init_umol_mol=0.0,
              q_init_mol_kg=q0, kLDF_1_s=p["kLDF_1_s"], t_end_s=20.0, dt_out_s=0.5)
    tr = dr.run_case(pt, iso, N=N, rtol=1e-8)
    # compare the cumulative water removed from the solid (mol per m2 bed cross-section)
    h = p["L_m"] / N
    w_trace = 790.0 * h * (q0 * N - tr["sol"]["F"][1].sum(axis=0))
    w_full = 790.0 * h * (q0 * N - r["q"].sum(axis=0))
    assert np.allclose(w_full[-1], w_trace[-1], rtol=1e-3)

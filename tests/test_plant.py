"""Unit tests of the plant engine, stack tables, PV chain and baselines (Brief 3 item 5; SC-B building blocks)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from h2su import plant as pl  # noqa: E402
from h2su import plantrun as pr  # noqa: E402
from h2su.baselines.bs1 import bs1_annual  # noqa: E402
from h2su.gas import tanks_in_series_step_response  # noqa: E402
from h2su.io import RESULTS  # noqa: E402

# Skipped when the result files of the chain are absent (package without results/): the engine reads results/inventory_ranges.csv (written by step 1 "brief3 chain", inventories.py).
pytestmark = pytest.mark.skipif(not (RESULTS / "inventory_ranges.csv").exists(),
                                reason="results/inventory_ranges.csv is absent; run `python code/run_all.py final` (step 1, brief3 chain: inventories.py item2/item14/item15) to write it")


def synth(p_kw, ta=15.0, zen=60.0):
    n = len(p_kw)
    return pd.DataFrame({"p_dc_kw": np.asarray(p_kw, float), "zenith": np.full(n, zen), "ta_c": np.full(n, ta), "fill": np.zeros(n, np.int8)})


def run(df, par, sw, trace=False):
    tot, cyc, nc, tr = pr.run(df, par, sw, trace=trace)
    return tot, cyc, nc, tr, pr.metrics(tot, cyc, nc, par)


def test_stack_tables_monotone_and_rated():
    st = pr.stack_tables()
    row = st["mtab"][list(pr.TG).index(75.0)]
    assert np.all(np.diff(row) >= -1e-12)
    assert abs(row[-1] - st["m_rated_kg_h"]) < 1e-12
    # rated point: 1 MW at 1.75 A/cm2, 1.80 V: kg/h = eta_F x I M/(2F) 3600 with the engine's own Faraday constant
    assert 19.0 < st["m_rated_kg_h"] < 21.0                 # about 49-52 kWh/kg stack level
    assert abs(st["jtab"][list(pr.TG).index(75.0)][-1] - 1.75) < 1e-3


def test_constant_power_no_restart_no_startup_loss():
    n = 30 * 1440
    par = pr.build_params("A", M_INV0=pr.build_params("A")[pl.I_INV] * pr.build_params("A")[pl.MRATED_MIN] * (30 - 1.01325), INIT_STATE=3)
    tot, cyc, nc, tr, m = run(synth(np.full(n, 600.0)), par, pr.switches())
    assert m["N_r_hot"] + m["N_r_warm"] + m["N_r_cold"] == 0
    assert abs(m["lambda_SU"]) < 1e-12 and m["M_blowdown"] == 0.0


def test_pressurisation_purge_washout_analytic():
    """Constant 500 kW from t = 0 (night start, OFF): inert 8 min, purge 6.5 min vented, pressurisation to m_full, off-spec until theta_spec."""
    par = pr.build_params("A")
    sw = pr.switches(ramp=0, deoxo=0)
    n = 600
    tot, cyc, nc, tr, m = run(synth(np.full(n, 500.0)), par, sw, trace=True)
    mr = par[pl.MRATED_MIN]
    m_full = par[pl.I_INV] * mr * (par[pl.P_OP] - par[pl.P0])
    ms = tr[20, 2]                                                           # kg/min at 0.5 load, T_op
    assert abs(tot[pl.T_PURGE_] + tot[pl.T_OFFSPEC] - par[pl.THETA_SPEC] * mr) / (par[pl.THETA_SPEC] * mr) < 1e-9     # washout: diverted rated-minutes = theta_spec
    assert abs(tot[pl.T_DINV] - m_full) / m_full < 1e-9                       # inventory filled exactly
    assert abs(tot[pl.T_PURGE_] - 6.5 * ms) / (6.5 * ms) < 1e-9               # purge mass = 6.5 min of production
    t_expected = 8 + 6.5 + m_full / ms + (par[pl.THETA_SPEC] * mr - 6.5 * ms) / ms     # minutes until the first delivered hydrogen
    first_del = int(np.argmax(tr[:, 3] > 0))
    assert abs((first_del + 1) - t_expected) <= 1.0 + 1e-9                    # 1-min resolution


def test_washout_matches_tanks_in_series_integrator():
    """N = 1 tank: time constant tau = theta_spec/ln(1e6/300); the RK4 integrator crosses 300 umol/mol at theta_spec within 0.1 %."""
    par = pr.build_params("A")
    tau = par[pl.THETA_SPEC] / np.log(1e6 / 300.0)
    t, c = tanks_in_series_step_response(1, tau, 10.0 * tau, tau / 4000.0, 1.0)  # step of clean H2 into an N2-filled tank: N2 fraction = 1 - c
    resid = 1.0 - c                                                          # N2 fraction
    t_cross = t[np.argmax(resid <= 300e-6)]
    assert abs(t_cross - par[pl.THETA_SPEC]) / par[pl.THETA_SPEC] < 1e-3


def test_ideal_gas_inventory_consistency():
    """The kg-per-bar inventory equals an ideal-gas vessel of V_eff = m_full R T/(M dp): filling it with m_full gives p_op within 0.1 %."""
    from h2su.constants import M_H2, R_GAS
    par = pr.build_params("A")
    m_full = par[pl.I_INV] * par[pl.MRATED_MIN] * (par[pl.P_OP] - par[pl.P0])
    dp = (par[pl.P_OP] - par[pl.P0]) * 1e5
    V = m_full / M_H2 * R_GAS * 298.15 / dp
    p_end = par[pl.P0] * 1e5 + (m_full / M_H2) * R_GAS * 298.15 / V
    assert abs(p_end - par[pl.P_OP] * 1e5) / (par[pl.P_OP] * 1e5) < 1e-3


def test_all_mechanisms_off_equals_bs1_exactly():
    rng = np.random.default_rng(1)
    base = np.clip(1200 * np.sin(np.linspace(0, 12 * np.pi, 3 * 1440)) ** 2, 0, None)
    p = base * np.clip(1 - 0.9 * (rng.random(base.size) < 0.2) * rng.random(base.size), 0, 1)
    par, sw = pr.s1_config(pr.build_params("A"))
    tot, cyc, nc, tr, m = run(synth(p), par, sw)
    ref = bs1_annual(p)
    assert abs(tot[pl.T_DEL] - ref["M_del"]) <= 1e-9 * ref["M_del"]
    assert abs(tot[pl.T_STACK] - ref["M_stack"]) <= 1e-9 * ref["M_stack"]


def test_lambda_identity_and_closure_on_intermittent_input():
    rng = np.random.default_rng(2)
    day = np.clip(1500 * np.sin(np.linspace(0, 20 * np.pi, 10 * 1440)), 0, None)
    p = day * np.clip(1 - 0.95 * (rng.random(day.size) < 0.15), 0, 1)
    for pol in "ACD":
        par = pr.build_params(pol)
        tot, cyc, nc, tr, m = run(synth(p, zen=np.where(day > 0, 60.0, 100.0).mean()), par, pr.switches())
        assert m["identity_SU_SS_rel"] <= 1e-9 and m["closure_H2_rel"] <= 1e-9
        assert m["M_stack"] >= m["M_del"] >= 0 and m["lambda_SU"] >= -1e-9


def test_hot_restart_has_no_blowdown_and_aborts_counted():
    par = pr.build_params("D")
    p = np.concatenate([np.full(300, 800.0), np.zeros(30), np.full(300, 800.0)])
    tot, cyc, nc, tr, m = run(synth(p), par, pr.switches(), trace=True)
    assert m["M_blowdown"] == 0.0                         # 30 min below P_min < 24 h standby
    assert m["N_r_hot"] == 1


def test_pv_chain_zero_at_night_and_bounded():
    from h2su.pv import pv_dc_series
    d = pv_dc_series("Carpentras", 2016)
    assert d.p_dc_kw.min() >= 0 and d.p_dc_kw.max() < 1.2 * 1500
    assert d.p_dc_kw[d.zenith > 100].max() == 0.0

"""Single entry point that regenerates every result and figure produced so far (Stage 2 brief 1).

    python code/run_all.py --help
    python code/run_all.py ver1 | ver1b | timing | tests | all
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402

from h2su.io import RESULTS  # noqa: E402


def _peak_rss_mb() -> float:
    """Peak working set of this process (Windows psapi); NaN elsewhere."""
    if os.name != "nt":
        return float("nan")

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ps = ctypes.WinDLL("psapi", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    ps.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return float("nan")
    return pmc.PeakWorkingSetSize / 2**20


def cmd_ver1(_):
    from h2su import verification as v
    from h2su.figures import plot_ver1
    df = v.run_ver1()
    out = RESULTS / "ver1_rauls.csv"
    df.to_csv(out, index=False)
    v.table2_consistency().to_csv(RESULTS / "ver1_rauls_table2_consistency.csv", index=False)
    fig = plot_ver1(out)
    print("wrote", out, fig)
    main = df[(df.attempt != "heating_only_check") & (df.thermal_basis == "tables")]
    print(main[["attempt", "P_nom_kW", "n_stacks", "reached_75C", "mH2_model_kg", "mH2_published_kg", "RE_pct", "abs_RE_le_10pct"]].round(3).to_string())


def cmd_ver1b(_):
    from h2su import verification as v
    df = v.run_ver1b()
    out = RESULTS / "ver1b_persson.csv"
    df.to_csv(out, index=False)
    print("wrote", out)
    print(df.groupby("T_K").agg(n_within_2pct=("within_2pct", "sum"), n=("within_2pct", "size"), max_abs_diff=("abs_diff", lambda s: s.abs().max())).to_string())


def cmd_timing(a):
    from h2su.statemachine import run_stub
    (RESULTS / "smoke").mkdir(parents=True, exist_ok=True)
    n = 525_600
    t = np.arange(n)
    rng = np.random.default_rng(12345)
    day = np.clip(np.sin(2 * np.pi * (t % 1440 - 360) / 1440), 0, None)       # dummy half-sine day, 0 at night
    p = 1500.0 * day * np.clip(1.0 - 0.6 * np.abs(rng.standard_normal(n)) * (rng.random(n) < 0.3), 0.0, 1.0)   # dummy intermittency, kW
    t0 = time.perf_counter()
    run_stub(p, 100.0, 1, 10, 30, 5)                      # first call includes JIT compilation (or cache load)
    t_first = time.perf_counter() - t0
    walls = []
    for _i in range(5):                                    # >= 5 trial runs (CLAUDE.md 8.3)
        t0 = time.perf_counter()
        nr, steps, _tr = run_stub(p, 100.0, 1, 10, 30, 5)
        walls.append(time.perf_counter() - t0)
    import numba
    lines = [f"timing probe: numba state-machine stub, {n} one-minute steps, dummy power (no physics claim; NOT a result)",
             f"first call (JIT compile or cache load): {t_first:.3f} s",
             f"steady wall time over 5 runs: min {min(walls)*1e3:.2f} ms, median {np.median(walls)*1e3:.2f} ms, max {max(walls)*1e3:.2f} ms",
             f"restart count (dummy input): {int(nr)}; steps per state OFF/STANDBY/START/PRODUCTION/SHUTDOWN: {steps.tolist()}",
             f"peak working set of this process: {_peak_rss_mb():.1f} MiB",
             f"input array {p.nbytes/2**20:.2f} MiB; state trace {_tr.nbytes/2**20:.2f} MiB",
             f"logical CPUs: {os.cpu_count()}; numba threading layer threads: {numba.config.NUMBA_NUM_THREADS}; the stub is single-threaded",
             f"python {sys.version.split()[0]}, numpy {np.__version__}, numba {numba.__version__}"]
    out = RESULTS / "smoke" / "timing.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_ver6(_):
    """VER-6a code verification + 6d grid/tolerance convergence of the Python 1-D reference (verification cases);
    deoxo/dryer cases are reported BLOCKED while SN17/SN18 are open."""
    from h2su import ver6
    from h2su.ver6_cases import build
    ev, chk, conv = ver6.run_all_verif()
    ev.to_csv(RESULTS / "ver6" / "ver6_events.csv", index=False)
    chk.to_csv(RESULTS / "ver6_verification.csv", index=False)
    conv.to_csv(RESULTS / "ver6_convergence.csv", index=False)
    cases, _ = build()
    for c in cases:
        if c.status != "READY":
            print(f"BLOCKED {c.case_id} ({c.model_function}): blocked by {c.blocked_by}; no run on assumed values")
    print(chk[chk.grid_id.isin(["N400"])][["case_id", "quantity", "numeric", "analytic", "error", "tol", "passed", "mass_balance_err"]].to_string(index=False))
    print(conv.round(5).to_string(index=False))


def cmd_ver2r(_):
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parent / "ver2r.py"), run_name="__main__")


def _script(name):
    import runpy
    sys.argv = [name]
    runpy.run_path(str(Path(__file__).resolve().parent / name), run_name="__main__")


def cmd_brief3(a):
    """Brief 3 chain: PV chain, inventories, VT3/VT4, SC-B, VER-2L, VER-8, trial costs, nominal runs (PRELIMINARY)."""
    import subprocess
    py = sys.executable
    here = Path(__file__).resolve().parent
    for cmd in (["pv_all.py"], ["inventories.py", "item2"], ["inventories.py", "item14"], ["inventories.py", "item15"], ["awe_check.py"], ["vt_runs.py", "vt3"], ["vt_runs.py", "vt4"],
                ["scb.py"], ["ver2r.py"], ["ver2l.py"], ["ver8.py"], ["nominal.py", "trial"], ["nominal.py", "nominal"]):
        subprocess.run([py, str(here / cmd[0])] + cmd[1:], check=True)


def cmd_tests(_):
    import pytest
    sys.exit(pytest.main(["-q", str(Path(__file__).resolve().parent / "tests")]))


def cmd_vt6(_):
    """Brief 4 item 1, Brief 11 item 1: L15 consistency, VT6 CAL verify mode (the frozen file is never written), held-out validation with the frozen k_LDF, BT6/BT7 executability."""
    import runpy
    sys.argv = ["vt6_run.py", "all"]
    runpy.run_path(str(Path(__file__).resolve().parent / "vt6_run.py"), run_name="__main__")


def cmd_ver6dryer(_):
    """Brief 4 item 1c: VER-6c/6d/6e dryer cases (Python reference + lumped forms) -> results/ver6/, ver6_dryer_verdict.csv."""
    from h2su import ver6_dryer
    ver6_dryer.run_all()
    print(ver6_dryer.verdict().to_string())



# ---------------------------------------------------------------------------------------------------------------- final (Brief 5 item 11)
FINAL_STEPS = [
    # (name, command (relative to code/), expected outputs (relative to results/))
    ("brief3 chain", ["run_all.py", "brief3"], ["nominal/annual.csv", "nominal/ablation.csv", "scb_plant.csv", "vt2_l28.csv", "vt3.csv", "vt4.csv", "trial_cost_chain.csv"]),
    ("vt1/vt2/vt5a", ["vt_runs.py", "all"], ["vt1_l26.csv", "vt2b_l32.csv", "vt5a_l1.csv"]),
    ("dryer VT6 (CAL verify on BT2 against the frozen file, then held-out validation; D81)", ["run_all.py", "vt6"], ["vt6_cal_verify.csv", "vt6_l15.csv", "vt6_cal_frozen.json"]),
    ("dryer VER-6", ["run_all.py", "ver6dryer"], ["ver6_dryer_verdict.csv"]),
    ("VER-6c notes", ["ver6c_notes.py"], ["ver6c_notes.csv"]),
    ("nominal (Brief 4 reporting)", ["nominal.py", "nominal"], ["nominal/annual.csv", "nominal/ablation_interaction.csv"]),
    ("neutrality check", ["neutrality_check.py"], ["nominal/neutrality_check.csv"]),
    ("BoP from PV", ["sweeps.py", "bop"], ["sens/bop_pv.csv"]),
    ("controller", ["sweeps.py", "controller"], ["sens/controller.csv"]),
    ("inventory OAT", ["sweeps.py", "inventory"], ["sens/inventory_oat_raw.csv"]),
    ("inventory analysis", ["inventory_analysis.py"], ["sens/inventory_oat.csv", "sens/estimator.csv"]),
    ("POL-B sweep", ["sweeps.py", "polb"], ["policy/polB_pareto.csv"]),
    ("resolution", ["sweeps.py", "resolution"], ["resolution/resolution_raw.csv"]),
    ("resolution analysis", ["resolution_analysis.py"], ["resolution/resolution_summary.csv", "resolution/factorial_2x2.csv"]),
    ("AWE procedure", ["sweeps.py", "awe"], ["sens/awe_procedure.csv"]),
    ("PV chain sensitivity", ["sens_pv.py"], ["sens/pv_chain.csv"]),
    ("beta decomposition", ["beta_decomp.py"], ["nominal/beta_decomposition.csv"]),
    ("falsification grid", ["falsification.py", "grid"], ["sens/falsification_grid.csv"]),
    ("break-even", ["falsification.py", "breakeven"], ["sens/breakeven.csv", "sens/falsification_verdict.txt"]),
    ("POL-P", ["policy_analysis.py", "polp"], ["policy/pol_p_bounding.csv", "policy/pol_p_decomposition.csv"]),
    ("POL-P m_SU comparison (Brief 6 item 3)", ["policy_analysis.py", "polp_post"], ["policy/pol_p_decomposition_site.csv"]),
    ("standby break-even", ["policy_analysis.py", "breakeven"], ["policy/standby_breakeven.csv"]),
    ("event structure", ["event_structure.py"], ["nominal/lambda_by_event_class.csv", "nominal/binned_day_ratio.csv", "nominal/intraday_slope.csv"]),
    ("dryer scenarios", ["dryer_scenarios.py", "scenarios"], ["sens/dryer_scenarios.csv"]),
    ("VT6b definitions", ["vt6b_run.py", "defcheck"], ["vt6b_definitions_check.csv"]),
    ("VT6b", ["vt6b_run.py", "vt6b"], ["vt6b_l15.csv"]),
    ("dryer event bounds (i, ii; superseded by the composed file two steps below)", ["dryer_scenarios.py", "bound"], ["sens/dryer_event_bound.csv"]),
    ("dryer plant cases (non-trace), on-spec throughput and off-spec timing", ["vt6b_run.py", "plant"], ["ver6_dryer_nontrace.csv"]),
    ("dryer grid band N = 800 (Brief 6 item 6)", ["dryer_gridband.py", "one", "800"], ["ver6/gridband/countercurrent_1.5h_N800.json"]),
    ("dryer grid band N = 1600", ["dryer_gridband.py", "one", "1600"], ["ver6/gridband/countercurrent_1.5h_N1600.json"]),
    ("dryer grid band N = 3200 (about 15-20 min)", ["dryer_gridband.py", "one", "3200"], ["ver6/gridband/countercurrent_1.5h_N3200.json"]),
    ("dryer grid band summary (order, Richardson, GCI)", ["dryer_gridband.py", "summary"], ["ver6_dryer_gridband.csv", "ver6_dryer_gridband_gci.csv"]),
    ("dryer event bounds composed (per-event kg, POL-A annual pp)", ["dryer_gridband.py", "bound"], ["sens/dryer_event_bound.csv"]),
    ("Ishigami check", ["sobol_pipeline.py", "ishigami"], ["uq/ishigami_check.csv"]),
    ("Sobol PEM A, C (D47, Brief 4)", ["sobol_pipeline.py", "plant"], ["uq/plant_Cabauw_A_seed11_a.npz"]),
    ("final UQ (Brief 5 item 7; resumes from checkpoints; 11 independent worker processes)", ["uq_final.py", "run-workers"], ["uq/ens_Cabauw_2013_A_seed11.npz", "uq/plant_Cabauw_B60_seed11_a.npz"]),
    ("B-S3 seeds 22/33/44/55 (Brief 7 item 2, D67a; 272 tasks, claim-based worker processes, tasks skip by checkpoint)", ["uq_final.py", "run-s3-workers"], ["uq/s3_Cabauw_2013_A_seed22.npz", "uq/s3_Gobabeb_2018_D_seed55.npz"]),
    ("UQ completion check (Brief 6 item 8, extended in Brief 7 item 2 to the 272 B-S3 tasks; exit code 1 if any of the 763 tasks is missing or fails)", ["uq_completion_check.py"], ["uq/completion_check.csv", "uq/completion_check_summary.txt"]),
    ("tau population and tau_mean closed-form check (Brief 7 item 3)", ["tau_population.py"], ["nominal/tau_population.csv", "nominal/tau_mean_closed_form_check.csv"]),
    ("POL-B60 nominal cells and RR5 cells for POL-B60, POL-D (Brief 7 item 4a, b)", ["t4_b60_rr5.py"], ["nominal/b60_nominal.csv", "sens/rr5_b60_d.csv", "sens/rr5_control_polA.csv"]),
    ("C3 comparable quantities incl. ratio 0.75 (Brief 7 item 5a)", ["c3_comparable.py"], ["resolution/c3_comparable.csv"]),
    ("C3 capacity factors, timestep decomposition, delivered H2 60 vs 5 min (Brief 7 item 5b-d)", ["c3_timestep.py"], ["resolution/c3_site_cf.csv", "resolution/c3_timestep_decomposition.csv", "resolution/delivered_60v5.csv"]),
    ("first cycle of each simulated year (side brief S item 2)", ["first_cycle_term.py"], ["nominal/first_cycle_term.csv"]),
    ("C3 load shape and closed-form efficiency-convention term (side brief S item 3)", ["c3_load_shape.py"], ["resolution/c3_load_shape.csv"]),
    ("UQ summaries (Brief 6 item 9; Brief 7 items 1, 3: beta pooled over 5 seeds with years concatenated, tau_mean, nominal percentile)", ["uq_final_summary.py", "all"], ["uq/intervals.csv", "uq/intervals_pooled.csv", "uq/scm_verdict.csv", "uq/scs_verdict.csv", "uq/scs_percentile_spread.csv", "uq/sobol_pooled.csv",
                                                                       "uq/beta_intervals.csv", "uq/beta_seed_spread.csv", "uq/sobol_beta.csv", "uq/beta_table_check.csv", "uq/sobol_beta_table_check.csv"]),
    ("figures F2-F9", ["figures_final.py"], [f"figures/{n}.png" for n in ("component_tests", "restart_fate_mechanisms", "daily_restart_statistics", "resolution_and_model_bias", "sobol_restart_loss",
                                                                       "standby_tradeoff")] + ["figures/data/F6_factorial_by_site_data.csv"]),
    ("tables T1-T6", ["tables_final.py"], [f"tables/{n}.csv" for n in ("T1_parameters_sources", "T2_validation", "T2_supplementary", "T3_sites_scenarios", "T4_lambda_beta_intervals", "T5_baselines", "T6_comparison")]),
    ("tests", ["run_all.py", "tests"], []),
]


STEP_CAP_S = 3600          # per-step cap (PROJECT.md): a step above it is killed and the chain stops


def _run_step(cmd, here, cap):
    """Run one chain step as separate process(es); returns the exit code (max over worker processes). Raises subprocess.TimeoutExpired above the cap."""
    import subprocess
    t0 = time.monotonic()

    def left():
        return max(cap - (time.monotonic() - t0), 1.0)
    if cmd[0] == "uq_final.py" and cmd[1] == "run-workers":
        procs = [subprocess.Popen([sys.executable, str(here / "uq_final.py"), "worker", str(k), "11"]) for k in range(11)]
    elif cmd[0] == "uq_final.py" and cmd[1] == "run-s3-workers":
        rc = subprocess.run([sys.executable, str(here / "uq_final.py"), "clearclaims"], timeout=cap).returncode
        if rc:
            return rc
        procs = [subprocess.Popen([sys.executable, str(here / "uq_final.py"), "claimworker", f"c{k}", "--only", "s3"]) for k in range(11)]
    else:
        return subprocess.run([sys.executable, str(here / cmd[0])] + cmd[1:], timeout=cap).returncode
    try:
        for p in procs:
            p.wait(timeout=left())
    except subprocess.TimeoutExpired:
        for p in procs:
            p.kill()
        raise
    return max(p.returncode for p in procs)


def cmd_final(a):
    """Brief 5 item 11: every Brief 3-5 result, figure and table in dependency order. `final --dry-run` lists steps and expected outputs (existing / missing) and runs nothing.
    Long runs resume from their checkpoints (uq_final.py skips existing .npz). Steps run as separate processes with the same interpreter; --from N starts at step N (1-based).
    Brief 9 item 5: --log FILE writes a real start and end timestamp (written by this process) and the exit code of every step; the chain stops at the first non-zero exit code
    and at any step above the per-step cap of 60 min.
    """
    import datetime as dt
    import subprocess
    here = Path(__file__).resolve().parent
    missing = 0
    log = None
    if not a.dry_run and a.log:
        log = open(a.log, "a", encoding="utf-8", buffering=1)

    def w(msg):
        if log:
            log.write(msg + "\n")

    now = lambda: dt.datetime.now().astimezone().isoformat(timespec="seconds")      # noqa: E731
    t_chain = time.monotonic()
    if log:
        w(f"CHAIN START {now()} from step {a.start} of {len(FINAL_STEPS)}; python {sys.version.split()[0]}; cap per step {STEP_CAP_S} s")
    status = "OK"
    for i, (name, cmd, outs) in enumerate(FINAL_STEPS, 1):
        ex = [(o, (RESULTS / o).exists()) for o in outs]
        missing += sum(1 for _, e in ex if not e)
        print(f"[{i:02d}] {name}: python code/{' '.join(cmd)}")
        for o, e in ex:
            print(f"       {'OK     ' if e else 'MISSING'} results/{o}")
        if a.dry_run or i < a.start:
            continue
        t0, ts = time.monotonic(), now()
        try:
            rc = _run_step(cmd, here, STEP_CAP_S)
        except subprocess.TimeoutExpired:
            rc = -9
            w(f"[{i:02d}] START {ts} | END {now()} | exit TIMEOUT above {STEP_CAP_S} s | {time.monotonic() - t0:.1f} s | {name}")
            status = f"STOPPED at step {i}: per-step cap exceeded"
            break
        w(f"[{i:02d}] START {ts} | END {now()} | exit {rc} | {time.monotonic() - t0:.1f} s | {name}")
        if rc != 0:
            status = f"STOPPED at step {i}: exit code {rc}"
            break
    print(f"steps: {len(FINAL_STEPS)}; missing expected outputs: {missing}")
    if log:
        w(f"CHAIN END {now()} | {status} | wall {time.monotonic() - t_chain:.1f} s")
        log.close()
        if status != "OK":
            raise SystemExit(status)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regenerate results and figures of the h2su project (Stage 2 brief 1 content).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, hlp in [("ver1", cmd_ver1, "VER-1: L26 Rauls heat-up model vs Table 5"), ("ver1b", cmd_ver1b, "VER-1b: L13 Persson Eq.16"),
                          ("timing", cmd_timing, "timing probe of the numba state-machine stub"),
                          ("ver6", cmd_ver6, "VER-6a/6d: 1-D packed-bed reference, verification cases"),
                          ("vt6", cmd_vt6, "L15 VT6 (Brief 4 item 1)"), ("ver6dryer", cmd_ver6dryer, "VER-6c/6d/6e dryer cases (Brief 4 item 1)"),
                          ("ver2r", cmd_ver2r, "VER-2R: C3 (L4) Tables 3 and 5 re-run on AEMO 2022 data"), ("tests", cmd_tests, "pytest suite (VER-3)"), ("brief3", cmd_brief3, "Brief 3 chain (plant engine, SC-B, VER-2L, VER-8, trial, nominal runs)")]:
        sp = sub.add_parser(name, help=hlp)
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("final", help="every Brief 3-5 result, figure and table in dependency order (--dry-run lists outputs)")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--from", dest="start", type=int, default=1)
    sp.add_argument("--log", default=None, help="chain log file: start/end timestamps and exit code per step")
    sp.set_defaults(fn=cmd_final)
    sp = sub.add_parser("all", help="ver1, ver1b, tests, timing")
    sp.set_defaults(fn=lambda a: [cmd_ver1(a), cmd_ver1b(a), cmd_timing(a), cmd_tests(a)])
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

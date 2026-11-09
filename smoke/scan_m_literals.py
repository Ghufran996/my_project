"""Grep-style report of numeric literals in COMSOL_MATLAB/*.m (comments and string contents removed).

    python code/smoke/scan_m_literals.py

Allowed without comment: 0, 1, 2 (indices, sign, stoichiometry 2 H2 per O2 / 2 H2O per O2, halves) and, by name, the
unit conversion ppm = 1e-6.  Everything else is listed with its line and classified by a fixed rule table
(solver / mesh settings, tolerances, output formats).  Any literal that is not classified is printed as UNCLASSIFIED.
Output: results/smoke/comsol_literals_report.txt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from h2su.io import RESULTS, ROOT  # noqa: E402

PKG = ROOT / "COMSOL_MATLAB"
NUM = re.compile(r"(?<![\w.])(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)")
TRIVIAL = {"0", "1", "2", "4"}   # 4 only in the Danckwerts closed-form (4 a exp(..)) and in 4 Da/Pe; see classification below


def strip_code(line: str) -> str:
    out, i, in_str = [], 0, False
    while i < len(line):
        ch = line[i]
        if in_str:
            if ch == "'":
                if i + 1 < len(line) and line[i + 1] == "'":
                    i += 2
                    continue
                in_str = False
                out.append("'")
            i += 1
            continue
        if ch == "%":
            break
        if ch == "'":
            prev = "".join(out).rstrip()[-1:] if "".join(out).strip() else ""
            if prev and (prev.isalnum() or prev in "_)]}'."):
                out.append("'")            # transpose
            else:
                in_str = True
                out.append("'")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


RULES = [  # (file regex, line regex, literal regex, category)
    (r"run_refinement", r"mesh_levels|rtol_base|rtol_variants|sprintf\('N%03d", r".*", "mesh / tolerance levels (solver setting)"),
    (r"export_ver6_csv", r"M\(i, 1\)", r"[3-7]", "column index of the output matrix (array indexing)"),
    (r"run_refinement|export_ver6_csv|RUN_ALL|write_run_log", r"sprintf|fprintf|format", r".*", "output format"),
    (r"solve_bed_model", r"rtol|abs\(", r"1e-6|1e-9", "comparison tolerance of solver read-back / geometry / initial-value checks"),
    (r"SMOKE_TEST", r"1e-3|1e-6|spec\.N|t_end|spec\.L|D'|n = 1:2:199|\(n \* pi\)|t_end / 10|spec\.rtol|spec\.params", r".*", "smoke-test setup (tiny diffusion model, analytic Fourier series)"),
    (r"verif_tracer_run", r"4 \* da|1 - exp|exp\(pe / 2\)|exp\(a \* pe / 2\)|exp\(-a \* pe / 2\)|\(1 \+ a\)|\(1 - a\)", r"4|2", "closed-form Danckwerts expression (analytic reference)"),
    (r"model_deoxo_startup", r"ppm = 1e-6", r"1e-6", "unit conversion umol/mol -> mol/mol"),
    (r"model_dryer_startup_regen", r"ppm = 1e-6", r"1e-6", "unit conversion umol/mol -> mol/mol"),
    (r"model_dryer_startup_regen", r"/ 1e5|1e5\)", r"1e5", "unit conversion Pa -> bar (1 bar = 1e5 Pa, SI definition)"),
    (r"model_dryer_startup_regen", r"linspace\(0, p\.L_m, 201\)", r"201", "number of sample points of the stage-1 profile handed to stage 2 (interpolation resolution; independent of the mesh)"),
    (r"model_dryer_startup_regen", r"1e-300", r"1e-300", "division guard in the relative balance error (no physical meaning)"),
    (r"model_dryer_startup_regen|solve_bed_model", r"init_tol|tol0", r"1e-3", "tolerance factor of the initial-value self-check (1e-3 of the field scale; same factor as before in solve_bed_model)"),
    (r"build_bed_model", r"numelem|hmax|geom|coord|setIndex|'0'", r".*", "mesh / geometry API argument"),
    (r"read_params|read_cases|require_params", r"1e-|detect", r".*", "none"),
]


def main() -> int:
    rows, unclassified = [], []
    for f in sorted(PKG.glob("*.m")):
        for no, raw in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = strip_code(raw)
            # drop the contents of strings that were kept as '...' quotes
            code_ns = re.sub(r"'[^']*'", "''", code)
            for m in NUM.finditer(code_ns):
                lit = m.group(1)
                if lit in {"0", "1", "2"}:
                    continue
                cat = None
                for fr, lr, litr, c in RULES:
                    if re.search(fr, f.name) and re.search(lr, raw) and re.fullmatch(litr, lit):
                        cat = c
                        break
                rows.append((f.name, no, lit, cat or "UNCLASSIFIED", raw.strip()))
                if cat is None:
                    unclassified.append(rows[-1])
    lines = ["Numeric literals in COMSOL_MATLAB/*.m other than 0, 1, 2 (comments and string contents removed)",
             f"Files scanned: {len(list(PKG.glob('*.m')))}; literals found: {len(rows)}; UNCLASSIFIED: {len(unclassified)}", ""]
    for fn, no, lit, cat, src in rows:
        lines.append(f"{fn}:{no}  [{lit}]  {cat}  |  {src[:110]}")
    out = RESULTS / "smoke" / "comsol_literals_report.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:2]))
    for r in unclassified:
        print("UNCLASSIFIED", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Brief 7 item 2 pre-launch trial: 5 rows of one seed-11 task per (site, policy) re-evaluated with s3_task and compared with the stored M_del_S3 (must be identical); wall time per row -> cost estimate.

    python code/uq_s3_trial.py   -> results/uq/s3_trial_cost.csv
(The run of R89 used a copy of this script with absolute paths; the logic is identical.)
"""
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import uq_final as u  # noqa: E402
from sweeps import site_years  # noqa: E402

rows = []
for s, y in [next((a, b) for a, b in site_years() if a == q) for q in ("Cabauw", "Carpentras", "Gobabeb")]:
    for pol in ("A", "B60", "C", "D"):
        z = np.load(u.UQ / f"ens_{s}_{y}_{pol}_seed11.npz", allow_pickle=True)
        stored = z["Y"][:, list(z["cols"]).index("M_del_S3")]
        t0 = time.time()
        out, clo = u.s3_task((s, y, pol, 11), n=5, write=False)
        dt = time.time() - t0
        rows.append(dict(site=s, year=y, pol=pol, n_rows=5, wall_s_5rows=dt, s_per_row=dt / 5, est_s_per_task=dt / 5 * 1024, max_abs_diff=float(np.max(np.abs(out - stored[:5]))),
                         exact=bool(np.array_equal(out, stored[:5])), max_closure=float(clo.max())))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(u.UQ / "s3_trial_cost.csv", index=False)

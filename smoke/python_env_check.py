"""Smoke check: import the pinned stack and write versions."""
import sys, platform, importlib, os
names = ["numpy","scipy","pandas","numba","pvlib","SALib","matplotlib","yaml","pyarrow","requests","pytest"]
out = [f"python {sys.version}", f"platform {platform.platform()}", f"cpu_count {os.cpu_count()}"]
for n in names:
    m = importlib.import_module(n)
    out.append(f"{n} {getattr(m,'__version__','?')} OK")
# numba JIT test
import numba, numpy as np
@numba.njit
def f(x):
    s = 0.0
    for i in range(x.size): s += x[i]
    return s
out.append(f"numba_jit_sum {f(np.arange(10.0))}")
print("\n".join(out))

"""Programmatic colour trace of L4 Fig. 2 (efficiency curves) as a consistency check of the legend values.
Usage: l4_fig2_trace.py <jpg>.  Axis calibration from detected tick marks (y: 0.75 -> 215 px, 0.50 -> 1113.5 px;
x: 0.1 -> 227.5 px, 1.0 -> 1577.5 px).  Prints eta at P/Pcap = 0.1, 0.3, 1.0 (columns +-3 px) and Eq.10 values."""
import sys, hashlib
import numpy as np
from PIL import Image
f = sys.argv[1]
print("sha256", hashlib.sha256(open(f, "rb").read()).hexdigest())
im = np.asarray(Image.open(f).convert("RGB")).astype(int)
r, g, b = im[..., 0], im[..., 1], im[..., 2]
mask = {"curve1_blue": (b > 200) & (r < 90) & (g < 90), "curve2_red": (r > 200) & (g < 90) & (b < 90),
        "curve3_green": (g > 90) & (r < 70) & (b < 70)}
y2eta = lambda y: 0.75 - (y - 215.0) / ((1113.5 - 215.0) / 0.25)
x2p = lambda x: 0.1 + (x - 227.5) / 1500.0
emax = 0.75
for name, m in mask.items():
    out = []
    for p in (0.1, 0.3, 0.9, 1.0):
        xc = 227.5 + (p - 0.1) * 1500.0
        xs = range(int(xc) - 3 - (3 if p == 1.0 else 0), int(xc) + 4 - (3 if p == 1.0 else 0))
        ys = [yy for x in xs for yy in np.where(m[260:1100, x])[0] + 260]   # exclude legend box (y < 250) and axes
        out.append((p, y2eta(np.mean(ys)) if ys else float("nan"), len(ys)))
    print(name, " ".join(f"P={p}: eta={e:.4f} (n={n})" for p, e, n in out))
for k, emin in enumerate((0.70, 0.60, 0.55), 1):
    de = emax - emin
    print(f"Eq.10 curve{k}: " + " ".join(f"P={p}: {-de / 0.49 * (p - 0.3) ** 2 + emax:.4f}" for p in (0.1, 0.3, 0.9, 1.0)))

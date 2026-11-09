"""Brief 3 item 1e: automated trace of L24 (Liu 2026) Fig. 3 OTH(t) (red dashed curve) with axis calibration from detected tick marks.
Three 'independent' variants (line top edge, centre, bottom edge; two colour thresholds). Image: results/sources/L24/img-000.png (page 8 of the
MDPI PDF, pdfimages). Output code/configs/tables/L24_fig3.csv + sha256 file."""
import hashlib, pathlib, numpy as np, pandas as pd
from PIL import Image
root = pathlib.Path(__file__).resolve().parents[2]
im = np.array(Image.open(root / "results/sources/L24/img-000.png").convert("RGB")).astype(int)
# calibration from detected ticks (see log R22): right axis OTH 1000 ppm at y=235.5, 0 at y=2097 ; bottom axis 0 s at x=501.5, 500 s per 443.2 px
y1000, y0 = 235.5, 2097.0
x0, pxs = 501.5, 443.2 / 500.0
ppm = lambda y: (y0 - y) / (y0 - y1000) * 1000.0
tsec = lambda x: (x - x0) / pxs
rows = []
variants = {"edge_top_thr200": ("top", 200, 110), "centre_thr200": ("mid", 200, 110), "edge_bottom_thr200": ("bot", 200, 110),
            "centre_thr170": ("mid", 170, 140)}
trace = {}
for name, (mode, rt, gt) in variants.items():
    red = (im[:, :, 0] > rt) & (im[:, :, 1] < gt) & (im[:, :, 2] < gt)
    red[:, :512] = False; red[:, 3150:] = False; red[:60, :] = False; red[2180:, :] = False
    red[:, 3140:] = red[:, 3140:] & False
    d = {}
    for x in range(512, 3140):
        ys = np.where(red[:, x])[0]
        if ys.size:
            d[x] = ys.min() if mode == "top" else ys.max() if mode == "bot" else ys.mean()
    trace[name] = d
def val(name, t_s, w=6):
    xs = [x for x in range(int(x0 + t_s * pxs) - w, int(x0 + t_s * pxs) + w + 1) if x in trace[name]]
    return np.mean([ppm(trace[name][x]) for x in xs]) if xs else np.nan
# line half width in ppm: red line thickness measured at a flat part
pts = [("300A_start_peak_t=60s", 60), ("300A_end_t=120s", 120), ("600A_end_t=600s", 600), ("900A_end_t=1040s", 1040), ("1200A_end_t=2900s", 2900)]
half_w_ppm = 0.5 * 6 / (y0 - y1000) * 1000.0     # assumed 6 px line width at full resolution -> half in ppm (see note)
for lab, t in pts:
    v = [val(n, t) for n in variants]
    rows.append(dict(point=lab, t_s=t, **{n: round(x, 2) for n, x in zip(variants, v)}, central=round(float(np.nanmean(v)), 2),
                     max_dev=round(float(np.nanmax(v) - np.nanmin(v)), 2), u_d_ppm=round(float(np.nanmax(v) - np.nanmin(v)) + half_w_ppm, 2)))
df = pd.DataFrame(rows)
out = root / "code/configs/tables/L24_fig3.csv"
hdr = ("# Source: L24 Liu et al. 2026 Energies 19:3844 Fig. 3 (OTH vs time), automated trace of the red dashed curve; calibration from detected tick marks "
       "(OTH 1000 ppm at y=235.5 px, 0 ppm at y=2097 px; 0 s at x=501.5 px, 500 s = 443.2 px). Four trace variants; u_d = max deviation among variants + half line width (3 px -> %.2f ppm).\n"
       "# NOTE: ppm resolution of the figure is 1 px = %.3f ppm; the steady values printed in the text (9.33, 7.70, 5.86 ppm) are used for CAL, the figure only documents the 300 A start-up transient.\n"
       % (half_w_ppm, 1000.0 / (y0 - y1000)))
out.write_text(hdr + df.to_csv(index=False), encoding="utf-8")
h = hashlib.sha256(out.read_bytes()).hexdigest()
(root / "results/vt4_digitisation_hash.txt").write_text(f"{h}  code/configs/tables/L24_fig3.csv\n", encoding="utf-8")
print(df.to_string(index=False)); print(h)

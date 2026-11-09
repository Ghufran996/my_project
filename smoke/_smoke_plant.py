import sys, time
sys.path.insert(0, "code/src")
import numpy as np, pandas as pd
from h2su import plantrun as pr, plant as pl
from h2su.pv import pv_dc_series
st = pr.stack_tables()
print("rated kg/h", st["m_rated_kg_h"], "sigma", st["sigma_scale"], "area cm2", st["area_cm2"])
print("kg/h at l=0.1,0.3,1 (75C):", [float(np.interp(l, pr.LG, st["mtab"][list(pr.TG).index(75.0)])) for l in (0.1,0.3,1.0)])
df = pv_dc_series("Cabauw", 2014)
for pol in "ACD":
    par = pr.build_params(pol)
    t0=time.time(); tot,cyc,nc,tr = pr.run(df, par, pr.switches()); t1=time.time()-t0
    m = pr.metrics(tot,cyc,nc,par)
    print(pol, f"wall {t1:.2f}s", {k:(round(v,4) if isinstance(v,float) else v) for k,v in m.items() if k in ("M_stack","M_del","lambda_SU","lambda_SS","eta_del","N_r_hot","N_r_warm","N_r_cold","N_abort","N_stops","tau_eq_median","closure_H2_rel","closure_E_rel","identity_SU_SS_rel","M_blowdown","M_purge","M_offspec","dM_inv")})

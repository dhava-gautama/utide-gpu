import sys; sys.path.insert(0,"/home/dhava/utide-gpu/UTide")
import numpy as np, warnings, time; warnings.filterwarnings("ignore")
import utide
raw=np.loadtxt("/home/dhava/utide-gpu/UTide/notebooks/can1998.dtf")
t=raw[:,0]/86400.0; e=raw[:,5].copy(); e[raw[:,6]==2]=np.nan; e[np.abs(e-9.990)<1e-6]=np.nan
base=np.nan_to_num(e-np.nanmean(e),nan=0.0); nt=len(t); rng=np.random.default_rng(0)
kw=dict(lat=-25,epoch="1998-01-01",verbose=False)
print("solve_many double precision: solver='auto'(normal-eq) vs 'lstsq'")
for S in [1000,5000,20000]:
    X=rng.uniform(0.5,1.5,S)[None,:]*base[:,None]+0.05*rng.standard_normal((nt,S))
    def best(solver):
        utide.solve_many(t,X,gpu=True,solver=solver,**kw)
        b=1e18
        for _ in range(3):
            t0=time.perf_counter(); utide.solve_many(t,X,gpu=True,solver=solver,**kw); b=min(b,time.perf_counter()-t0)
        return b
    a=utide.solve_many(t,X,gpu=True,solver="auto",**kw); l=utide.solve_many(t,X,gpu=True,solver="lstsq",**kw)
    rel=np.nanmax(np.abs(a.A-l.A))/(np.nanmax(np.abs(l.A))+1e-30)
    ta,tl=best("auto"),best("lstsq")
    print(f"  S={S:5d}: lstsq={1000*tl:7.0f}ms  normal-eq={1000*ta:7.0f}ms  speedup={tl/ta:4.1f}x  rel diff={rel:.1e}")

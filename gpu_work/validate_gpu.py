import sys; sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import numpy as np, warnings, time
warnings.filterwarnings("ignore")
import utide

valid = set(n.strip() for n in utide._ut_constants.ut_constants.const.name)
# well-separated, full-rank over a multi-year record
want = ['M2','S2','N2','K2','K1','O1','P1','Q1','M4','MS4','M6','MM','MF','SA','SSA','MU2','J1','OO1']
constit = [c for c in want if c in valid]
np.random.seed(7)

def mk(nt, twodim=False):
    t = np.arange(nt) / 24.0
    fr = {'M2':1/12.42,'S2':1/12.0,'K1':1/23.93,'O1':1/25.82,'N2':1/12.66}
    u = sum(a*np.cos(2*np.pi*f*24*t + p) for (f,a,p) in
            [(fr['M2'],1.0,0.3),(fr['S2'],0.4,1.0),(fr['K1'],0.5,2.0),
             (fr['O1'],0.3,0.5),(fr['N2'],0.2,1.5)]) + 0.1*np.random.randn(nt)
    if twodim:
        v = sum(a*np.cos(2*np.pi*f*24*t + p) for (f,a,p) in
                [(fr['M2'],0.6,0.7),(fr['K1'],0.3,1.1)]) + 0.1*np.random.randn(nt)
        return t, u, v
    return t, u, None

def cmp(label, ccpu, cgpu, keys):
    print(f"  [{label}]")
    ok = True
    for k in keys:
        a = np.atleast_1d(np.asarray(ccpu[k], dtype=float))
        b = np.atleast_1d(np.asarray(cgpu[k], dtype=float))
        # faithful if equal within combined tol; nan must match nan on both sides
        close = np.allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True)
        nan_c, nan_g = int(np.isnan(a).sum()), int(np.isnan(b).sum())
        fin = np.isfinite(a) & np.isfinite(b)
        d = np.max(np.abs(a[fin] - b[fin])) if fin.any() else 0.0
        flag = "OK " if close else "!! "
        if not close: ok = False
        extra = f"  nan(cpu={nan_c},gpu={nan_g})" if (nan_c or nan_g) else ""
        print(f"     {flag}{k:10s} max|Δ|(finite)={d:.3e}{extra}")
    return ok

print("="*64); print("END-TO-END: solve(gpu=False) vs solve(gpu=True)"); print("="*64)
nt = 2*365*24  # 2 years hourly

# 1-D
t, u, _ = mk(nt)
c0 = utide.solve(t, u, lat=45, constit=constit, method="ols", conf_int="linear", verbose=False)
c1 = utide.solve(t, u, lat=45, constit=constit, method="ols", conf_int="linear", verbose=False, gpu=True)
ok1 = cmp("1-D, OLS, linear CI", c0, c1, ["A","g","A_ci","g_ci","mean","slope"])

# 2-D (u, v)
t, u, v = mk(nt, twodim=True)
c0 = utide.solve(t, u, v, lat=45, constit=constit, method="ols", conf_int="linear", verbose=False)
c1 = utide.solve(t, u, v, lat=45, constit=constit, method="ols", conf_int="linear", verbose=False, gpu=True)
ok2 = cmp("2-D, OLS, linear CI", c0, c1, ["Lsmaj","Lsmin","theta","g","Lsmaj_ci","g_ci","umean","vmean"])

# fallback: gpu requested but unsupported (nodal=False -> not [0,0,0,0])  should match CPU
t, u, _ = mk(nt)
c0 = utide.solve(t, u, lat=45, constit=constit, method="ols", conf_int="none", nodal=False, verbose=False)
c1 = utide.solve(t, u, lat=45, constit=constit, method="ols", conf_int="none", nodal=False, verbose=False, gpu=True)
ok3 = cmp("fallback (nodal=False)", c0, c1, ["A","g"])

print(f"\nRESULT: 1D={'PASS' if ok1 else 'FAIL'}  2D={'PASS' if ok2 else 'FAIL'}  fallback={'PASS' if ok3 else 'FAIL'}")

# quick speed check at 10yr
print("\nSpeed (10yr hourly, OLS, no CI):")
nt = 10*365*24; t, u, _ = mk(nt)
for g in (False, True):
    utide.solve(t, u, lat=45, constit=constit, method="ols", conf_int="none", verbose=False, gpu=g)  # warmup
    best = 1e18
    for _ in range(3):
        t0 = time.perf_counter()
        utide.solve(t, u, lat=45, constit=constit, method="ols", conf_int="none", verbose=False, gpu=g)
        best = min(best, time.perf_counter() - t0)
    print(f"  gpu={str(g):5s}: {1000*best:7.0f} ms")

import sys; sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import numpy as np, warnings, time
warnings.filterwarnings("ignore")
import utide

valid = set(n.strip() for n in utide._ut_constants.ut_constants.const.name)
want = ['M2','S2','N2','K2','K1','O1','P1','Q1','M4','M6','MM','MF']
constit = [c for c in want if c in valid]
rng = np.random.default_rng(0)
nt = 2*365*24
t = np.arange(nt)/24.0
fr = {"M2":1/12.42,"S2":1/12.0,"K1":1/23.93,"O1":1/25.82}
base = (np.cos(2*np.pi*fr["M2"]*24*t+0.3)+0.4*np.cos(2*np.pi*fr["S2"]*24*t)
        +0.5*np.cos(2*np.pi*fr["K1"]*24*t+2)+0.3*np.cos(2*np.pi*fr["O1"]*24*t))

def make_X(S, gap_frac=0.15):
    X = (rng.uniform(0.5,1.5,S)[None,:]*base[:,None] + 0.1*rng.standard_normal((nt,S)))
    # per-station random gaps (different pattern each)
    for s in range(S):
        ng = int(gap_frac*nt)
        idx = rng.choice(nt, ng, replace=False)
        X[idx, s] = np.nan
    return X

kw = dict(lat=45, constit=constit, epoch="2000-01-01", verbose=False)

print("VALIDATE gappy solve_many vs per-station solve (different gaps each)")
X = make_X(6)
om = utide.solve_many(t, X, gpu=True, **kw)
worst = 0.0
for s in range(4):
    c = utide.solve(t, X[:, s], method="ols", conf_int="none",
                    lat=45, constit=constit, epoch="2000-01-01", verbose=False)
    o = [list(om.name).index(n) for n in c['name']]
    big = c['A'] > 0.05
    dA = np.max(np.abs(c['A'][big] - om.A[o, s][big]) / c['A'][big])
    dg = np.max(np.abs((c['g'][big] - om.g[o, s][big] + 180) % 360 - 180))
    worst = max(worst, dA)
    print(f"  station {s} ({np.isnan(X[:,s]).sum()} gaps): max rel A={dA:.2e}  phase={dg:.2e} deg")
print(f"  --> {'PASS' if worst < 1e-3 else 'CHECK'}")

# all-nan and too-gappy columns -> NaN
Xedge = make_X(3); Xedge[:, 0] = np.nan          # all nan
Xedge[: nt-5, 1] = np.nan                         # only 5 valid -> underdetermined
om2 = utide.solve_many(t, Xedge, gpu=True, **kw)
print(f"\n  all-nan series -> A all NaN: {np.all(np.isnan(om2.A[:,0]))}")
print(f"  5-valid series -> A all NaN: {np.all(np.isnan(om2.A[:,1]))}")
print(f"  normal series  -> A finite : {np.all(np.isfinite(om2.A[:,2]))}")

# speed: shared-gap (1 group) vs distinct gaps (S groups), 1000 stations
print("\nSPEED (1000 series):")
Xs = make_X(1000, 0.0); g = rng.choice(nt, int(0.1*nt), replace=False); Xs[g,:] = np.nan  # shared gap
t0=time.perf_counter(); utide.solve_many(t, Xs, gpu=True, **kw); print(f"  shared-gap (1 group):   {1000*(time.perf_counter()-t0):.0f} ms")
Xd = make_X(1000, 0.1)  # distinct gaps -> ~1000 groups
t0=time.perf_counter(); utide.solve_many(t, Xd, gpu=True, **kw); print(f"  distinct gaps (~1000 groups): {1000*(time.perf_counter()-t0):.0f} ms")

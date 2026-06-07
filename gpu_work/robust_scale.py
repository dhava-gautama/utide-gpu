import sys; sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import numpy as np, warnings, time
warnings.filterwarnings("ignore")
import utide

valid = set(n.strip() for n in utide._ut_constants.ut_constants.const.name)
want = ['M2','S2','N2','K2','K1','O1','P1','Q1','M4','MS4','M6','MM','MF','MU2','J1','OO1']
constit = [c for c in want if c in valid]
rng = np.random.default_rng(0)
fr = [1/12.42, 1/12.0, 1/23.93, 1/25.82]

def mk(nt):
    t = np.arange(nt) / 24.0
    u = sum(np.cos(2*np.pi*f*24*t) for f in fr) + 0.2*rng.standard_normal(nt)
    idx = rng.choice(nt, max(10, nt//500), replace=False)
    u[idx] += rng.uniform(-6, 6, len(idx))   # outliers
    return t, u

kw = dict(lat=45, constit=constit, method="robust", conf_int="none",
          epoch="2000-01-01", verbose=False)
print("Robust IRLS scaling: CPU vs GPU-single")
print(f"{'nt':>8} {'yr':>4} {'CPU(ms)':>9} {'GPU-s(ms)':>10} {'speedup':>8} {'iters':>6}")
for nd in [365, 365*10, 365*30]:
    nt = nd*24; t, u = mk(nt)
    def best(**extra):
        b = 1e18; c = None
        for _ in range(2):
            t0 = time.perf_counter(); c = utide.solve(t, u, **kw, **extra)
            b = min(b, time.perf_counter() - t0)
        return b, c
    tc, cc = best()
    tg, cg = best(gpu=True, gpu_precision="single")
    print(f"{nt:>8} {nd//365:>4} {1000*tc:>9.0f} {1000*tg:>10.0f} {tc/tg:>7.1f}x {cc.rf.iterations:>6}")

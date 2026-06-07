import sys; sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import numpy as np, warnings, time
warnings.filterwarnings("ignore")
import utide

path = "/home/dhava/utide-gpu/UTide/notebooks/can1998.dtf"
raw = np.loadtxt(path)
seconds, elev, flag = raw[:, 0], raw[:, 5].copy(), raw[:, 6]
elev[flag == 2] = np.nan
elev[np.abs(elev - 9.990) < 1e-6] = np.nan
t = seconds / 86400.0
anom = elev - np.nanmean(elev)
anom = np.nan_to_num(anom, nan=0.0)   # fill the few gaps for the shared batch
nt = len(t)

# Build S synthetic stations sharing the time base: scaled + phase-jittered + noise
rng = np.random.default_rng(0)
def make_X(S):
    scales = rng.uniform(0.5, 1.5, S)
    X = scales[None, :] * anom[:, None] + 0.02 * rng.standard_normal((nt, S))
    return X

# ---- validate solve_many vs per-station solve ----
print("VALIDATE solve_many vs looping solve() (3 stations)")
X = make_X(5)
om = utide.solve_many(t, X, lat=-25, epoch="1998-01-01", gpu=True, verbose=False)
names_m = list(om.name)
worst = 0.0
for s in range(3):
    c = utide.solve(t, X[:, s], lat=-25, method="ols", conf_int="none",
                    epoch="1998-01-01", verbose=False)
    order = [names_m.index(n) for n in c['name']]
    dA = np.max(np.abs(c['A'] - om.A[order, s]) / (np.abs(c['A']) + 1e-9))
    dg = np.max(np.abs(((c['g'] - om.g[order, s] + 180) % 360 - 180)))
    worst = max(worst, dA)
    print(f"  station {s}: max rel A diff={dA:.2e}   max phase diff={dg:.2e} deg")
print(f"  --> {'PASS' if worst < 1e-4 else 'CHECK'}\n")

# ---- benchmark: naive loop vs batched CPU vs batched GPU ----
print("BENCHMARK: amplitudes/phases for many stations (1 yr hourly)")
# time one solve() to extrapolate the naive loop
t0 = time.perf_counter()
utide.solve(t, X[:, 0], lat=-25, method="ols", conf_int="none", epoch="1998-01-01", verbose=False)
per = time.perf_counter() - t0
print(f"  single solve(): {1000*per:.0f} ms  -> naive loop = S x that")
for S in [100, 1000, 5000]:
    X = make_X(S)
    # batched CPU
    t0 = time.perf_counter()
    utide.solve_many(t, X, lat=-25, epoch="1998-01-01", gpu=False, verbose=False)
    cpu = time.perf_counter() - t0
    # batched GPU (warmup then time)
    utide.solve_many(t, X, lat=-25, epoch="1998-01-01", gpu=True, verbose=False)
    t0 = time.perf_counter()
    utide.solve_many(t, X, lat=-25, epoch="1998-01-01", gpu=True, verbose=False)
    gpu = time.perf_counter() - t0
    print(f"  S={S:>5}: naive~{S*per:7.1f}s | batch-CPU={cpu:6.3f}s | batch-GPU={gpu:6.3f}s "
          f"| vs-naive={S*per/gpu:6.0f}x  vs-CPU-batch={cpu/gpu:4.1f}x")

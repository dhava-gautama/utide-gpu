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
anom = np.nan_to_num(elev - np.nanmean(elev), nan=0.0)
# inject a few outliers so the robust method actually does something
rng = np.random.default_rng(0)
idx = rng.choice(len(anom), 40, replace=False)
anom[idx] += rng.uniform(-5, 5, 40)

kw = dict(lat=-25, method="robust", conf_int="none", epoch="1998-01-01", verbose=False)

def run(label, **extra):
    best = 1e18; c = None
    for _ in range(2):
        t0 = time.perf_counter()
        c = utide.solve(t, anom, **kw, **extra)
        best = min(best, time.perf_counter() - t0)
    return c, best

print("Robust IRLS (cauchy) on real data with injected outliers")
c_cpu, t_cpu = run("cpu")
c_gd,  t_gd  = run("gpu-double", gpu=True)
c_gs,  t_gs  = run("gpu-single", gpu=True, gpu_precision="single")

def cmp(c, ref):
    o = [list(c['name']).index(n) for n in ref['name']]
    big = ref['A'] > 0.02
    dA = np.max(np.abs(ref['A'][big] - c['A'][o][big]) / ref['A'][big])
    dg = np.max(np.abs((ref['g'][big] - c['g'][o][big] + 180) % 360 - 180))
    return dA, dg

dA_d, dg_d = cmp(c_gd, c_cpu)
dA_s, dg_s = cmp(c_gs, c_cpu)
print(f"  iterations: cpu={c_cpu.rf.iterations} gpu-double={c_gd.rf.iterations} gpu-single={c_gs.rf.iterations}")
print(f"  gpu-double vs cpu: max rel A={dA_d:.2e}  max phase={dg_d:.2e} deg")
print(f"  gpu-single vs cpu: max rel A={dA_s:.2e}  max phase={dg_s:.2e} deg")
print(f"  time: cpu={1000*t_cpu:.0f}ms  gpu-double={1000*t_gd:.0f}ms ({t_cpu/t_gd:.1f}x)  "
      f"gpu-single={1000*t_gs:.0f}ms ({t_cpu/t_gs:.1f}x)")

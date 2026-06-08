import sys

sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import utide

path = "/home/dhava/utide-gpu/UTide/notebooks/can1998.dtf"
raw = np.loadtxt(path)
seconds, elev, flag = raw[:, 0], raw[:, 5].copy(), raw[:, 6]
elev[flag == 2] = np.nan
elev[np.abs(elev - 9.990) < 1e-6] = np.nan
t = seconds / 86400.0
anom = np.nan_to_num(elev - np.nanmean(elev), nan=0.0)
nt = len(t)

print("=" * 66)
print("FP32 vs FP64 GPU solve  (single series, real data)")
print("=" * 66)
kw = dict(lat=-25, method="ols", conf_int="linear", epoch="1998-01-01", verbose=False)
cpu = utide.solve(t, anom, **kw)
d = utide.solve(t, anom, gpu=True, gpu_precision="double", **kw)
s = utide.solve(t, anom, gpu=True, gpu_precision="single", **kw)
order_d = [list(d["name"]).index(n) for n in cpu["name"]]
order_s = [list(s["name"]).index(n) for n in cpu["name"]]
for tag, c, o in [("double", d, order_d), ("single", s, order_s)]:
    dA = np.max(np.abs(cpu["A"] - c["A"][o]) / (np.abs(cpu["A"]) + 1e-9))
    dg = np.max(np.abs((cpu["g"] - c["g"][o] + 180) % 360 - 180))
    print(
        f"  gpu {tag:6s}: max rel A diff vs CPU = {dA:.2e}   max phase diff = {dg:.2e} deg",
    )

print("\n" + "=" * 66)
print("FP32 vs FP64 batched solve_many  (precision + speed)")
print("=" * 66)
rng = np.random.default_rng(0)


def make_X(S):
    return rng.uniform(0.5, 1.5, S)[None, :] * anom[
        :,
        None,
    ] + 0.02 * rng.standard_normal((nt, S))


for S in [1000, 5000]:
    X = make_X(S)
    od = utide.solve_many(
        t,
        X,
        lat=-25,
        epoch="1998-01-01",
        gpu=True,
        gpu_precision="double",
        verbose=False,
    )
    os_ = utide.solve_many(
        t,
        X,
        lat=-25,
        epoch="1998-01-01",
        gpu=True,
        gpu_precision="single",
        verbose=False,
    )
    relA = np.max(np.abs(od.A - os_.A) / (np.abs(od.A) + 1e-9))

    # speed
    def timeit(prec):
        utide.solve_many(
            t,
            X,
            lat=-25,
            epoch="1998-01-01",
            gpu=True,
            gpu_precision=prec,
            verbose=False,
        )
        best = 1e18
        for _ in range(3):
            t0 = time.perf_counter()
            utide.solve_many(
                t,
                X,
                lat=-25,
                epoch="1998-01-01",
                gpu=True,
                gpu_precision=prec,
                verbose=False,
            )
            best = min(best, time.perf_counter() - t0)
        return best

    td, ts = timeit("double"), timeit("single")
    print(
        f"  S={S:>5}: single vs double max rel A diff = {relA:.2e} | "
        f"double={1000*td:6.0f}ms  single={1000*ts:6.0f}ms  ({td/ts:.1f}x faster)",
    )

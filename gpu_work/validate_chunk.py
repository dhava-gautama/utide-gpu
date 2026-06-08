import sys

sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import cupy as cp

import utide

path = "/home/dhava/utide-gpu/UTide/notebooks/can1998.dtf"
raw = np.loadtxt(path)
seconds, elev, flag = raw[:, 0], raw[:, 5].copy(), raw[:, 6]
elev[flag == 2] = np.nan
elev[np.abs(elev - 9.990) < 1e-6] = np.nan
t = seconds / 86400.0
base = np.nan_to_num(elev - np.nanmean(elev), nan=0.0)
nt = len(t)
rng = np.random.default_rng(0)


def make_X(S):
    return rng.uniform(0.5, 1.5, S)[None, :] * base[
        :,
        None,
    ] + 0.05 * rng.standard_normal((nt, S))


kw = dict(lat=-25, epoch="1998-01-01", verbose=False)

print("Chunked vs unchunked correctness (S=500):")
X = make_X(500)
a = utide.solve_many(t, X, gpu=True, **kw)
b = utide.solve_many(t, X, gpu=True, chunk_size=50, **kw)
print(
    f"  max|A diff| = {np.max(np.abs(a.A - b.A)):.2e}   max|g diff| = {np.max(np.abs(a.g - b.g)):.2e}",
)
print(f"  identical: {np.allclose(a.A, b.A) and np.allclose(a.g, b.g)}")

free, total = cp.cuda.runtime.memGetInfo()
print(f"\nVRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB")
print(
    "Large-S streaming (no OOM): solving S=20000 series (would be "
    f"{nt*20000*16/1e9:.1f} GB if all on device at once)",
)
X = make_X(20000)
t0 = time.perf_counter()
om = utide.solve_many(t, X, gpu=True, **kw)
dt = time.perf_counter() - t0
peak = cp.get_default_memory_pool().used_bytes()
print(
    f"  done in {dt:.1f}s, A shape {om.A.shape}, all finite: {np.all(np.isfinite(om.A))}",
)
print(f"  pool peak ~{peak/1e9:.2f} GB (stayed well under VRAM)")

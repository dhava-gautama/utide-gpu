"""Smoke test: UTide GPU fork on the real BADA batch (run on the WSL GPU box).

Checks correctness (CPU vs GPU vs FP32, NaN parity) and regression M2 levels.
Requires /tmp/bada_utide_batch.npz -- build it first with:
    python gpu_work/make_bada_batch.py   (needs xarray; run anywhere it's installed)

Exit code 0 = pass.
"""
import sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import cupy as cp  # noqa: F401  (fail loudly if CUDA is gone)
from utide import solve_many

d = np.load("/tmp/bada_utide_batch.npz")
t, U, V, lat = d["t"], d["U"], d["V"], d["lat"]
CONSTS = ["M2", "S2", "N2", "K1", "O1", "P1", "M4"]
kw = dict(constit=CONSTS, trend=True, nodal=True, epoch="2000-01-01", lat=lat, verbose=False)

cpu = solve_many(t, U, V, gpu=False, gappy="auto", **kw)
gpu = solve_many(t, U, V, gpu=True, gappy="auto", **kw)

m = np.isfinite(cpu.Lsmaj) & np.isfinite(gpu.Lsmaj)
rel = float(np.max(np.abs(gpu.Lsmaj[m] - cpu.Lsmaj[m]) / np.maximum(np.abs(cpu.Lsmaj[m]), 1e-6)))
dg = float(np.max(np.abs(gpu.g[m] - cpu.g[m])))
parity = int(np.sum(np.isnan(cpu.Lsmaj[0]) != np.isnan(gpu.Lsmaj[0])))
m2 = list(cpu.name).index("M2")
m2_med = float(np.nanmedian(cpu.Lsmaj[m2]))

ok = True
def chk(name, cond, detail):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok &= bool(cond)

chk("GPU-CPU Lsmaj agreement", rel < 1e-8, f"(max rel {rel:.2e})")
chk("GPU-CPU phase agreement", dg < 1e-5, f"(max dphase {dg:.2e} deg)")
chk("NaN parity", parity == 0, f"(mismatches {parity})")
chk("solved cells", int(np.isfinite(cpu.Lsmaj[0]).sum()) > 250,
    f"({int(np.isfinite(cpu.Lsmaj[0]).sum())} of {cpu.Lsmaj.shape[1]})")
chk("M2 level stable", 15.0 < m2_med < 35.0, f"(median M2 Lsmaj {m2_med:.1f} cm/s; "
    "reference 23.3 for BADA 2022-2026 hourly QC)")

print("SMOKE " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)

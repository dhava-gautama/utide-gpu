import sys

sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import cupy as cp
from harmonics_xp import Tables, ut_E_xp

import utide
from utide.harmonics import ut_E

np = __import__("numpy")
tab_np = Tables(np)
tab_cp = Tables(cp)


def sync():
    cp.cuda.Stream.null.synchronize()


def bench(fn, n=5, gpu=False):
    if gpu:
        sync()
    fn()  # warmup (JIT compile)
    if gpu:
        sync()
    best = 1e18
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if gpu:
            sync()
        best = min(best, time.perf_counter() - t0)
    return best


valid = set(n.strip() for n in utide._ut_constants.ut_constants.const.name)
want = [
    "M2",
    "S2",
    "N2",
    "K2",
    "K1",
    "O1",
    "P1",
    "Q1",
    "M4",
    "MS4",
    "MN4",
    "SA",
    "MF",
    "MM",
    "MU2",
    "NU2",
    "2N2",
    "J1",
    "OO1",
    "M6",
    "S4",
    "M3",
    "S6",
    "M8",
    "2MK3",
    "MK3",
    "SK3",
]
constit = [c for c in want if c in valid]

# get lind/frq for this constituent set
nt0 = 8760
t0 = np.arange(nt0) / 24.0
h0 = np.cos(2 * np.pi * t0) + 0.1 * np.random.randn(nt0)
c0 = utide.solve(
    t0,
    h0,
    lat=45,
    constit=constit,
    method="ols",
    conf_int="none",
    verbose=False,
)
lind = c0["aux"]["lind"]
frq = c0["aux"]["frq"]
lat = 45.0

# ---------- 1. VALIDATE port ----------
print("=" * 70)
print("VALIDATION: ut_E_xp vs utide.ut_E")
print("=" * 70)
t = np.arange(20000) / 24.0
tref = t.mean()
E_ref = ut_E(t, tref, frq, lind, lat, [0, 0, 0, 0], [])
E_np = ut_E_xp(np, t, tref, frq, lind, lat, tab_np)
E_cp = cp.asnumpy(ut_E_xp(cp, cp.asarray(t), tref, frq, lind, lat, tab_cp))
print(f"  numpy-port  max|Δ| vs utide : {np.abs(E_np - E_ref).max():.2e}")
print(f"  cupy-port   max|Δ| vs utide : {np.abs(E_cp - E_ref).max():.2e}")

# ---------- 2. BASIS CONSTRUCTION scaling ----------
print("\n" + "=" * 70)
print("BASIS CONSTRUCTION  ut_E  (CPU numpy vs GPU cupy, complex128)")
print("=" * 70)
print(f"{'nt':>9} {'CPU(ms)':>10} {'GPU(ms)':>10} {'GPU+xfer(ms)':>13} {'speedup':>8}")
for nd in [365, 365 * 3, 365 * 10, 365 * 30, 365 * 100]:
    nt = nd * 24
    t = np.arange(nt) / 24.0
    tref = t.mean()
    tc = bench(lambda: ut_E_xp(np, t, tref, frq, lind, lat, tab_np), n=3)
    t_d = cp.asarray(t)
    tg = bench(lambda: ut_E_xp(cp, t_d, tref, frq, lind, lat, tab_cp), n=5, gpu=True)

    def withxfer():
        td = cp.asarray(t)
        E = ut_E_xp(cp, td, tref, frq, lind, lat, tab_cp)
        return cp.asnumpy(E)

    tgx = bench(withxfer, n=5, gpu=True)
    print(f"{nt:>9} {1000*tc:>10.1f} {1000*tg:>10.1f} {1000*tgx:>13.1f} {tc/tg:>7.1f}x")

# ---------- 3. LSTSQ: FP64 vs FP32 ----------
print("\n" + "=" * 70)
print("LEAST SQUARES  lstsq  (single RHS)  CPU vs GPU, FP64 vs FP32")
print("=" * 70)
print(
    f"{'nt':>9} {'CPUf64':>9} {'GPUf64':>9} {'spd':>5} | {'CPUf32':>9} {'GPUf32':>9} {'spd':>5}",
)
for nd in [365, 365 * 10, 365 * 30, 365 * 100]:
    nt = nd * 24
    t = np.arange(nt) / 24.0
    tref = t.mean()
    E = ut_E_xp(np, t, tref, frq, lind, lat, tab_np)
    B = np.hstack((E, E.conj(), np.ones((nt, 1)), (t - tref)[:, None]))  # complex128
    y = (np.cos(2 * np.pi * t) + 0.1 * np.random.randn(nt)).astype(np.complex128)
    B32 = B.astype(np.complex64)
    y32 = y.astype(np.complex64)
    Bd, yd = cp.asarray(B), cp.asarray(y)
    Bd32, yd32 = cp.asarray(B32), cp.asarray(y32)
    c64 = bench(lambda: np.linalg.lstsq(B, y, rcond=None), n=3)
    g64 = bench(lambda: cp.linalg.lstsq(Bd, yd, rcond=None), n=5, gpu=True)
    c32 = bench(lambda: np.linalg.lstsq(B32, y32, rcond=None), n=3)
    g32 = bench(lambda: cp.linalg.lstsq(Bd32, yd32, rcond=None), n=5, gpu=True)
    print(
        f"{nt:>9} {1000*c64:>9.1f} {1000*g64:>9.1f} {c64/g64:>4.1f}x | {1000*c32:>9.1f} {1000*g32:>9.1f} {c32/g32:>4.1f}x",
    )

# ---------- 4. BATCH many stations (the killer app) ----------
print("\n" + "=" * 70)
print("BATCH: many stations, shared time base (1yr hourly, 26 constit)")
print("=" * 70)
nt = 8760
t = np.arange(nt) / 24.0
tref = t.mean()
E = ut_E_xp(np, t, tref, frq, lind, lat, tab_np)
B = np.hstack((E, E.conj(), np.ones((nt, 1)), (t - tref)[:, None]))
Bd = cp.asarray(B)
print(
    f"  per-station utide.solve(): {1000*bench(lambda: utide.solve(t, np.cos(2*np.pi*t)+0.1*np.random.randn(nt), lat=45, constit=constit, method='ols', conf_int='none', verbose=False), n=3):.1f} ms",
)
for S in [100, 1000, 5000]:
    X = (np.random.randn(nt, S) + 1j * 0).astype(np.complex128)
    Xd = cp.asarray(X)
    cpu = bench(lambda: np.linalg.lstsq(B, X, rcond=None), n=2)
    gpu = bench(lambda: cp.linalg.lstsq(Bd, Xd, rcond=None), n=3, gpu=True)
    print(
        f"  S={S:>5}  batched-CPU-lstsq={1000*cpu:>8.1f}ms  batched-GPU-lstsq={1000*gpu:>8.1f}ms  speedup={cpu/gpu:>5.1f}x  (vs naive loop ~{S*0.087:.0f}s)",
    )

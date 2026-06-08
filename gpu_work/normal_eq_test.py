import sys

sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import cupy as cp

import utide
from utide._harmonics_xp import ut_E_xp


def sync():
    cp.cuda.Stream.null.synchronize()


def best(fn, gpu=True, n=4):
    fn()
    sync() if gpu else None
    b = 1e18
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if gpu:
            sync()
        b = min(b, time.perf_counter() - t0)
    return b


nt = 8760
t = np.arange(nt) / 24.0
tref = t.mean()
freqs = utide.harmonics.linearized_freqs(tref)
lind = np.arange(1, 30)
E = ut_E_xp(np, t, tref, freqs[lind], lind, 45, [0, 0, 0, 0])
B = np.hstack((E, E.conj(), np.ones((nt, 1)), (t - tref)[:, None]))
nm = B.shape[1]
print(f"nt={nt}, nm={nm}, cond(B)={np.linalg.cond(B):.1f}")

for prec, cdt in [("fp64", np.complex128), ("fp32", np.complex64)]:
    Bd = cp.asarray(B.astype(cdt))
    print(f"\n--- {prec} ---")
    for S in [1000, 5000, 20000]:
        X = (np.random.randn(nt, S)).astype(cdt)
        Xd = cp.asarray(X)

        # reference: lstsq
        def f_lstsq():
            return cp.linalg.lstsq(Bd, Xd, rcond=None)[0]

        # normal equations: G = B^H B, rhs = B^H X, solve
        def f_normal():
            G = Bd.conj().T @ Bd
            rhs = Bd.conj().T @ Xd
            return cp.linalg.solve(G, rhs)

        tl = best(f_lstsq)
        tn = best(f_normal)
        Ml = cp.asnumpy(f_lstsq())
        Mn = cp.asnumpy(f_normal())
        rel = np.max(np.abs(Ml - Mn)) / (np.max(np.abs(Ml)) + 1e-30)
        print(
            f"  S={S:5d}: lstsq={1000*tl:7.1f}ms  normal-eq={1000*tn:7.1f}ms  "
            f"speedup={tl/tn:4.1f}x  max rel diff={rel:.1e}",
        )

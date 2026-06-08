import sys

sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import utide
from utide.harmonics import ut_E

print("utide:", utide.__file__)

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
print(f"explicit nconstit={len(constit)} -> B cols ~{2*len(constit)+2}\n")

np.random.seed(1)
fr = [1 / 12.42, 1 / 12.0, 1 / 23.93]


def best(fn, n=2):
    b = 1e18
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


print(
    f"{'nt':>9} {'ut_E(ms)':>10} {'lstsq(ms)':>11} {'solve_ols(ms)':>14} {'solve_rob(ms)':>14}",
)
for nd in [365, 365 * 3, 365 * 10, 365 * 30]:
    nt = nd * 24
    t = np.arange(nt) / 24.0
    h = sum(np.cos(2 * np.pi * f * 24 * t) for f in fr) + 0.3 * np.random.randn(nt)
    c0 = utide.solve(
        t,
        h,
        lat=45,
        constit=constit,
        method="ols",
        conf_int="none",
        verbose=False,
    )
    lind = c0["aux"]["lind"]
    frq = c0["aux"]["frq"]
    tref = t.mean()
    tE = best(lambda: ut_E(t, tref, frq, lind, 45, [0, 0, 0, 0], []))
    E = ut_E(t, tref, frq, lind, 45, [0, 0, 0, 0], [])
    B = np.hstack((E, E.conj(), np.ones((nt, 1)), (t - tref)[:, None]))
    tL = best(lambda: np.linalg.lstsq(B, h, rcond=None))
    tO = best(
        lambda: utide.solve(
            t,
            h,
            lat=45,
            constit=constit,
            method="ols",
            conf_int="none",
            verbose=False,
        ),
    )
    try:
        tR = best(
            lambda: utide.solve(
                t,
                h,
                lat=45,
                constit=constit,
                method="robust",
                conf_int="none",
                verbose=False,
            ),
            1,
        )
    except Exception:
        tR = float("nan")
    print(
        f"{nt:>9} {1000*tE:>10.1f} {1000*tL:>11.1f} {1000*tO:>14.1f} {1000*tR:>14.1f}",
    )

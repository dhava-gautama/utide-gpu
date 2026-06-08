import sys

sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import utide
from utide._ut_constants import ut_constants
from utide.astronomy import ut_astron
from utide.harmonics import ut_E


def best(fn, n=3):
    b = 1e18
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


const = ut_constants.const
names = [n.strip() for n in const.name]
# pick constituent index lists of different sizes (valid, resolvable)
allidx = np.array([i for i in range(1, len(names))])  # skip Z0

print("=== Q1: does ut_E cost scale with #constituents, or always ~146? ===")
nt = 10 * 365 * 24
t = np.arange(nt) / 24.0
tref = t.mean()
freqs = utide.harmonics.linearized_freqs(tref)
for nc in [4, 16, 60, 146]:
    lind = allidx[:nc]
    frq = freqs[lind]
    tE = best(lambda: ut_E(t, tref, frq, lind, 45, [0, 0, 0, 0], []))
    print(f"  nc={nc:4d}: ut_E = {1000*tE:7.1f} ms")

print("\n=== Q2: breakdown of basis sub-ops at 10yr (FP64 CPU) ===")
ta = best(lambda: ut_astron(t))
print(f"  ut_astron(t)                : {1000*ta:7.1f} ms  (called TWICE in FUV)")
sat = ut_constants.sat
uu = np.random.rand(162, nt)
tsexp = best(lambda: np.exp(1j * 2 * np.pi * uu))
print(f"  exp over (162, nt) satellites: {1000*tsexp:7.1f} ms")
F = np.random.rand(146, nt) + 1j * np.random.rand(146, nt)
tang = best(lambda: np.angle(F))
tabs = best(lambda: np.abs(F))
print(f"  np.angle (146,nt)            : {1000*tang:7.1f} ms")
print(f"  np.abs   (146,nt)            : {1000*tabs:7.1f} ms")
dood = const.doodson
astro = ut_astron(t)[0]
tV = best(lambda: dood @ astro)
print(f"  doodson@astro (146,6)@(6,nt) : {1000*tV:7.1f} ms")

print("\n=== Q3: full solve breakdown (10yr, auto) ===")
import cProfile
import io
import pstats

h = np.cos(2 * np.pi * t) + 0.2 * np.random.randn(nt)
pr = cProfile.Profile()
pr.enable()
utide.solve(
    t,
    h,
    lat=45,
    method="ols",
    conf_int="linear",
    epoch="python",
    verbose=False,
)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(8)
print(s.getvalue())

print("=== Q4: CPU lstsq driver — numpy(gelsd) vs scipy gelsy/gels ===")
import scipy.linalg as sla

E = ut_E(t, tref, freqs[allidx[:60]], allidx[:60], 45, [0, 0, 0, 0], [])
B = np.hstack((E, E.conj(), np.ones((nt, 1)), (t - tref)[:, None]))
y = (np.cos(2 * np.pi * t) + 0.1 * np.random.randn(nt)).astype(complex)
print(
    f"  numpy.linalg.lstsq (gelsd) : {1000*best(lambda: np.linalg.lstsq(B,y,rcond=None)):7.1f} ms",
)
print(
    f"  scipy lstsq gelsy          : {1000*best(lambda: sla.lstsq(B,y,lapack_driver='gelsy')):7.1f} ms",
)
print(
    f"  scipy lstsq gelss          : {1000*best(lambda: sla.lstsq(B,y,lapack_driver='gelss')):7.1f} ms",
)

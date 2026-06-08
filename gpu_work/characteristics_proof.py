"""
Proof of concept: empirical tidal characteristics (datums) à la DHI/tide_analytics,
implemented cleanly with numpy/scipy and fitting UTide's batch (grid) layout.

Computes standard datums from a water-level series:
  MHW (mean high water), MLW (mean low water), MTL (mean tide level),
  MTR (mean tidal range), ED/FD (mean ebb/flood duration, hours).
"""

import sys

sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from scipy.signal import find_peaks

from utide.utilities import Bunch


def tidal_characteristics(t, h, min_period_hours=2.0, prominence=None):
    """Datums/characteristics from times `t` (days) and water level `h`."""
    t = np.asarray(t, float)
    h = np.asarray(h, float)
    good = np.isfinite(h)
    t, h = t[good], h[good]
    if len(h) < 4:
        return None
    dt_h = np.median(np.diff(t)) * 24.0
    distance = max(1, int(round(min_period_hours / dt_h)))
    hw, _ = find_peaks(h, distance=distance, prominence=prominence)
    lw, _ = find_peaks(-h, distance=distance, prominence=prominence)
    if len(hw) == 0 or len(lw) == 0:
        return None
    MHW, MLW = h[hw].mean(), h[lw].mean()
    # ebb = HW->next LW, flood = LW->next HW (works through occasional non-alternation)
    ev = np.concatenate([hw, lw])
    typ = np.concatenate([np.ones(len(hw)), -np.ones(len(lw))])
    o = np.argsort(ev)
    ev, typ = ev[o], typ[o]
    dur = np.diff(t[ev]) * 24.0
    ebb = dur[(typ[:-1] == 1) & (typ[1:] == -1)]
    flood = dur[(typ[:-1] == -1) & (typ[1:] == 1)]
    return Bunch(
        MHW=MHW,
        MLW=MLW,
        MTL=0.5 * (MHW + MLW),
        MTR=MHW - MLW,
        ED=float(ebb.mean()) if len(ebb) else np.nan,
        FD=float(flood.mean()) if len(flood) else np.nan,
        n_high=len(hw),
        n_low=len(lw),
    )


def tidal_characteristics_many(t, X, **kw):
    """Datums for every column of X (nt, nseries); returns arrays (nseries,)."""
    nser = X.shape[1]
    keys = ["MHW", "MLW", "MTL", "MTR", "ED", "FD"]
    out = {k: np.full(nser, np.nan) for k in keys}
    for s in range(nser):
        c = tidal_characteristics(t, X[:, s], **kw)
        if c is not None:
            for k in keys:
                out[k][s] = c[k]
    return Bunch(**out)


# --- 1. single real record (can1998) ---
raw = np.loadtxt("/home/dhava/utide-gpu/UTide/notebooks/can1998.dtf")
t = raw[:, 0] / 86400.0
h = raw[:, 5].copy()
h[raw[:, 6] == 2] = np.nan
h[np.abs(h - 9.990) < 1e-6] = np.nan
c = tidal_characteristics(t, h)
print("can1998 (real hourly water level):")
for k in ["MHW", "MLW", "MTL", "MTR", "ED", "FD"]:
    print(f"  {k:4s} = {c[k]:7.3f}")
print(f"  ({c.n_high} highs, {c.n_low} lows over the year)")

# --- 2. batched over a grid (same layout as solve_many) ---
ny = nx = 48
nt = 365 * 24
tt = np.arange(nt) / 24.0
yy, xx = np.meshgrid(np.linspace(0, 1, ny), np.linspace(0, 1, nx), indexing="ij")
xx = xx.ravel()
yy = yy.ravel()
amp = 0.5 + 1.5 * xx + 0.3 * np.sin(2 * np.pi * yy)  # M2 amplitude grid
rng = np.random.default_rng(0)
X = (
    amp[None, :] * np.cos(2 * np.pi * (1 / 12.42) * 24 * tt[:, None])
    + 0.3 * amp[None, :] * np.cos(2 * np.pi * (1 / 23.93) * 24 * tt[:, None])
    + 0.03 * rng.standard_normal((nt, ny * nx))
)
t0 = time.perf_counter()
ch = tidal_characteristics_many(tt, X)
dt = time.perf_counter() - t0
print(f"\ngrid datums: {ny*nx} cells in {dt:.2f}s")
print(
    f"  MTR (tidal range) map: min {np.nanmin(ch.MTR):.2f}  max {np.nanmax(ch.MTR):.2f} m",
)
print("  (MTR should track the M2 amplitude field, ~2x amplitude for a dominant M2)")

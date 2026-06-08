"""
utide-gpu on a real tide-gauge record.

Runs the full workflow on the hourly sea-level record shipped with the package
(``notebooks/can1998.dtf``, one year, 1998): harmonic analysis on the GPU,
prediction (reconstruction) versus the observations, and the empirical tidal
datums.
"""

import os
import time

import numpy as np

import utide

HERE = os.path.dirname(__file__)
DTF = os.path.join(HERE, os.pardir, "notebooks", "can1998.dtf")

# --- load + clean the real record -----------------------------------------
raw = np.loadtxt(DTF)                       # seconds year month day hour elev flag
t = raw[:, 0] / 86400.0                     # days since 1998-01-01
h = raw[:, 5].copy()
h[raw[:, 6] == 2] = np.nan                  # flag 2 == bad
h[np.abs(h - 9.990) < 1e-6] = np.nan        # missing sentinel
print(f"loaded {np.isfinite(h).sum()} hourly samples ({np.isnan(h).sum()} gaps)")

# --- 1. harmonic analysis on the GPU --------------------------------------
t0 = time.perf_counter()
coef = utide.solve(t, h, lat=-25, method="ols", conf_int="linear",
                   epoch="1998-01-01", gpu=True, verbose=False)
print(f"\nsolve(gpu=True): {len(coef['name'])} constituents in {time.perf_counter()-t0:.2f}s")
order = np.argsort(coef["A"])[::-1]
print("  top constituents:   amp (m)   phase (deg)")
for i in order[:6]:
    print(f"    {coef['name'][i]:5s}  {coef['A'][i]:7.3f}   {coef['g'][i]:7.1f}")

# --- 2. prediction vs observations ----------------------------------------
tide = utide.reconstruct(t, coef, epoch="1998-01-01", verbose=False)
resid = h - tide.h
rms = np.sqrt(np.nanmean(resid**2))
print(f"\nreconstruct: RMS(observed - predicted) = {rms:.3f} m "
      f"({100*(1-np.nanvar(resid)/np.nanvar(h)):.1f}% of variance explained)")

# --- 3. empirical tidal datums --------------------------------------------
c = utide.tidal_characteristics(t, h)
print("\ntidal datums:")
for k in ["MHW", "MLW", "MTL", "MTR", "ED", "FD"]:
    unit = "h" if k in ("ED", "FD") else "m"
    print(f"    {k:4s} = {c[k]:6.3f} {unit}")

# --- 4. plots (optional) ---------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    win = slice(0, 24 * 14)                  # first two weeks
    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    ax[0].plot(t[win], h[win], ".", ms=3, label="observed")
    ax[0].plot(t[win], tide.h[win], "-", lw=1, label="predicted (utide)")
    ax[0].set_title("Prediction vs observations (first 2 weeks)")
    ax[0].set_ylabel("sea level (m)"); ax[0].legend()
    ax[1].plot(t[win], h[win], "-", lw=0.8, color="0.5")
    m = (c.hw_t >= t[win][0]) & (c.hw_t <= t[win][-1])
    n = (c.lw_t >= t[win][0]) & (c.lw_t <= t[win][-1])
    ax[1].plot(c.hw_t[m], c.hw_h[m], "r^", label="high water")
    ax[1].plot(c.lw_t[n], c.lw_h[n], "bv", label="low water")
    ax[1].axhline(c.MHW, color="r", ls="--", lw=0.8); ax[1].axhline(c.MLW, color="b", ls="--", lw=0.8)
    ax[1].set_title(f"Datums: MHW={c.MHW:.2f} m, MLW={c.MLW:.2f} m, MTR={c.MTR:.2f} m")
    ax[1].set_xlabel("days since 1998-01-01"); ax[1].set_ylabel("sea level (m)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "real_station.png"), dpi=90)
    print("\nsaved figure -> examples/real_station.png")
except ImportError:
    print("(matplotlib not installed; skipping plot)")

"""
Best use case for utide-gpu: tidal harmonic analysis over a whole grid.

Many real datasets are a *field* of tidal time series that share one time
base -- an ocean-model SSH grid, satellite altimetry, a mooring/gauge array.
The standard approach is to loop ``utide.solve`` over every cell, which is
slow. ``solve_many`` builds the harmonic model once and solves every cell in a
single (optionally GPU) batched call.

This example synthesizes a tidal field on a 64x64 grid (1 year of hourly
samples per cell), recovers the constituent amplitude/phase at every cell with
one ``solve_many`` call, and (if matplotlib is available) plots the recovered
M2 co-amplitude map. It also reports the speed-up over the naive per-cell loop.
"""

import time

import numpy as np

import utide

# --- 1. Build a synthetic tidal field on a grid ---------------------------
ny = nx = 64  # 4096 grid cells
ncell = ny * nx
days = 365
nt = days * 24
t = np.arange(nt) / 24.0  # days
epoch = "2000-01-01"

yy, xx = np.meshgrid(np.linspace(0, 1, ny), np.linspace(0, 1, nx), indexing="ij")
xx = xx.ravel()
yy = yy.ravel()  # (ncell,)

# constituent frequencies (cph) and spatially varying amplitude / phase (deg)
cph = {
    "M2": 1 / 12.4206,
    "S2": 1 / 12.0,
    "N2": 1 / 12.6583,
    "K1": 1 / 23.9345,
    "O1": 1 / 25.8193,
}
amp = {
    "M2": 0.5 + 1.5 * xx + 0.3 * np.sin(2 * np.pi * yy),  # grows toward "coast"
    "S2": 0.2 + 0.5 * xx,
    "N2": 0.1 + 0.2 * xx,
    "K1": 0.3 + 0.1 * yy,
    "O1": 0.2 + 0.1 * yy,
}
pha = {  # Greenwich phase lag in degrees, with a spatial gradient
    "M2": 360 * yy % 360,
    "S2": (30 + 300 * yy) % 360,
    "N2": (60 + 200 * xx) % 360,
    "K1": (90 + 180 * xx) % 360,
    "O1": (120 + 90 * yy) % 360,
}

rng = np.random.default_rng(0)
X = np.zeros((nt, ncell))
for c in cph:
    arg = 2 * np.pi * cph[c] * 24 * t[:, None] - np.deg2rad(pha[c])[None, :]
    X += amp[c][None, :] * np.cos(arg)
X += 0.03 * rng.standard_normal((nt, ncell))  # observational noise

# --- 2. Recover constituents everywhere with ONE solve_many call ----------
constit = list(cph)
# Warm up the GPU first so the timing reflects steady-state throughput, not
# the one-time CUDA kernel compilation on the very first call.
utide.solve_many(
    t,
    X[:, :8],
    lat=45,
    constit=constit,
    epoch=epoch,
    gpu=True,
    verbose=False,
)
t0 = time.perf_counter()
out = utide.solve_many(
    t,
    X,
    lat=45,
    constit=constit,
    epoch=epoch,
    gpu=True,
    verbose=True,
)
gpu_t = time.perf_counter() - t0
print(f"\nsolve_many: {ncell} cells in {gpu_t:.2f} s")

# --- 3. The naive per-cell loop: time it AND use it to check correctness ---
m2 = list(out.name).index("M2")
nsample = 20
sample_A = np.empty(nsample)
t0 = time.perf_counter()
for s in range(nsample):
    c = utide.solve(
        t,
        X[:, s],
        lat=45,
        constit=constit,
        method="ols",
        conf_int="none",
        epoch=epoch,
        verbose=False,
    )
    sample_A[s] = c["A"][list(c["name"]).index("M2")]
per_cell = (time.perf_counter() - t0) / nsample
print(
    f"per-cell solve(): {1000*per_cell:.0f} ms  ->  naive loop ~= {per_cell*ncell:.0f} s "
    f"({per_cell*ncell/gpu_t:.0f}x slower than solve_many)",
)

# solve_many must reproduce the per-cell solve exactly (to round-off):
dA = np.max(np.abs(sample_A - out.A[m2, :nsample]) / sample_A)
print(f"solve_many vs per-cell solve(): max rel diff = {dA:.1e}")

# --- 4. Reshape to maps; the recovered field tracks the synthetic truth ---
A_map = out.A[m2].reshape(ny, nx)
g_map = out.g[m2].reshape(ny, nx)
true_A = amp["M2"].reshape(ny, nx)
# (a few % offset from the pure-cosine "truth" is expected: UTide fits a
# nodally-modulated tide, which a pure cosine does not exactly match.)
print(
    f"recovered M2 amplitude tracks the input field to "
    f"{100*np.median(np.abs(A_map-true_A)/true_A):.1f}% (median)",
)

# --- 5. Plot the recovered M2 co-amplitude map (optional) ------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im0 = ax[0].pcolormesh(A_map, shading="auto")
    ax[0].set_title("M2 amplitude (m)")
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].pcolormesh(g_map, shading="auto", cmap="twilight")
    ax[1].set_title("M2 phase (deg)")
    fig.colorbar(im1, ax=ax[1])
    fig.suptitle(f"Tidal constituents over a {ny}x{nx} grid via solve_many")
    fig.tight_layout()
    fig.savefig("m2_grid.png", dpi=90)
    print("saved figure -> m2_grid.png")
except ImportError:
    print("(matplotlib not installed; skipping plot)")

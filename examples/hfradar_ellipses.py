"""2-D tidal-ellipse analysis of an HF-radar surface-current field.

Fits tidal constituents to every grid cell of a total-current netCDF
(`water_u` / `water_v` on time/lat/lon, CMEMS or radar-provider convention)
in ONE `solve_many(t, u, v, gappy="ne")` call — the 2-D fit returns current
ellipse parameters (Lsmaj, Lsmin, theta, g) per constituent per cell, and
`reconstruct_many` predicts the full u/v field from the batch result.

The demo figure in the README was produced with the authors' WERA HF-radar
product (6,566 cells x 10 constituents, 131 days of half-hourly data,
Dec 2025 - Mar 2026, Sunda Strait): 1.9 s on an RTX 4060 vs 8.7 s CPU.

Usage:
    python examples/hfradar_ellipses.py --nc PATH [--out PNG] [--cpu]

Requires numpy, xarray, matplotlib, (optional cartopy for the basemap,
optional cupy for the GPU path).
"""

import argparse
import warnings

import numpy as np
import xarray as xr

# Constituents resolvable in a ~4-month record (Rayleigh criterion):
# S2-K2 and K1-P1 need ~6 months, so they are not used together here.
CONSTS = ["M2", "S2", "N2", "K1", "O1", "MU2", "NU2", "L2", "2N2"]
EPOCH = np.datetime64("2000-01-01T00:00:00")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nc",
        required=True,
        help="total-current netCDF with time/lat/lon + water_u/water_v",
    )
    ap.add_argument("--out", default="hfradar_ellipses.png")
    ap.add_argument("--min-obs", type=int, default=400)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    ds = xr.open_dataset(args.nc)
    t64 = ds.time.values
    U = ds.water_u.values / 100.0  # cm/s -> m/s (drop the rescale if m/s already)
    V = ds.water_v.values / 100.0
    if np.nanmax(np.abs(U)) < 10:  # already m/s
        U, V = ds.water_u.values, ds.water_v.values
    obs = np.isfinite(U) & np.isfinite(V)
    ncell = obs.sum(axis=0)
    ii, jj = np.where(ncell >= args.min_obs)
    t = ((t64.astype("datetime64[s]") - EPOCH) / np.timedelta64(1, "D")).astype(float)
    XU = np.where(obs[:, ii, jj], U[:, ii, jj], np.nan)
    XV = np.where(obs[:, ii, jj], V[:, ii, jj], np.nan)
    lat, lon = ds.lat.values[ii], ds.lon.values[jj]
    print(
        f"[prep] {len(t)} times, {len(ii)} cells, "
        f"finite={np.isfinite(XU).mean():.2f}",
    )

    try:
        import cupy  # noqa: F401

        gpu = not args.cpu
    except ImportError:
        gpu = False
    from utide import reconstruct_many, solve_many

    kw = {
        "constit": CONSTS,
        "trend": False,
        "nodal": True,
        "epoch": "2000-01-01",
        "lat": lat,
        "verbose": False,
    }
    import time as _time

    t0 = _time.perf_counter()
    coef = solve_many(t, XU, XV, gpu=gpu, gappy="ne", **kw)
    print(
        f"[fit] 2-D ellipse fit, {coef.Lsmaj.shape[1]} cells x {len(CONSTS)} "
        f"constituents in {_time.perf_counter() - t0:.2f} s "
        f"({'gpu' if gpu else 'cpu'})",
    )
    pred = reconstruct_many(t, coef, epoch="2000-01-01", gpu=gpu)

    # tide-explained variance per cell
    var_exp = 1 - (
        np.nanstd(XU - pred.u, axis=0) ** 2 + np.nanstd(XV - pred.v, axis=0) ** 2
    ) / (np.nanvar(XU, axis=0) + np.nanvar(XV, axis=0))
    i0 = int(np.nanargmax(var_exp))
    print(
        f"[reconstruct] best cell ({lat[i0]:.3f}N, {lon[i0]:.3f}E): "
        f"tide explains {var_exp[i0] * 100:.0f}% of variance",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        HAVE_CARTOPY = True
    except ImportError:
        HAVE_CARTOPY = False

    names = list(coef.name)
    iK1 = names.index("K1")
    fig = plt.figure(figsize=(13.0, 4.8))
    gs = fig.add_gridspec(1, 2, wspace=0.18)
    extent = [lon.min() - 0.1, lon.max() + 0.1, lat.min() - 0.1, lat.max() + 0.1]
    transform = None
    if HAVE_CARTOPY:
        ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#efe8d8", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#d6e8f5", zorder=0)
        ax.coastlines(resolution="10m", linewidth=0.5, zorder=2)
        transform = ccrs.PlateCarree()
    else:
        ax = fig.add_subplot(gs[0, 0])
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
    a = coef.Lsmaj[iK1] * 100
    fi = np.isfinite(a)
    sc = ax.scatter(
        lon[fi],
        lat[fi],
        c=a[fi],
        s=5,
        cmap="viridis",
        vmin=0,
        vmax=max(25, np.nanpercentile(a[fi], 99)),
        transform=transform,
        zorder=5,
    )
    plt.colorbar(sc, ax=ax, label="K1 Lsmaj (cm/s)", shrink=0.8, pad=0.02)
    ax.set_title(
        f"K1 semi-major axis at {int(fi.sum())} cells " f"(one 2-D solve_many call)",
        fontsize=10,
    )

    axD = fig.add_subplot(gs[0, 1])
    m = np.isfinite(XU[:, i0])
    w = t <= t[0] + 40
    axD.plot(t[w] - t[0], pred.u[w, i0] * 100, "b-", lw=1.1, label="UTide u (pred)")
    axD.plot(
        t[w] - t[0],
        pred.v[w, i0] * 100,
        "-",
        color="tab:orange",
        lw=1.1,
        label="UTide v (pred)",
    )
    wm = w & m
    axD.plot(t[wm] - t[0], XU[wm, i0] * 100, "k.", ms=3, label="obs u")
    axD.plot(t[wm] - t[0], XV[wm, i0] * 100, ".", color="gray", ms=3, label="obs v")
    axD.set_xlabel(f"days since {str(t64[0])[:10]}")
    axD.set_ylabel("current [cm/s]")
    axD.set_title(
        f"Reconstruction at best cell ({lat[i0]:.3f}N, {lon[i0]:.3f}E): "
        f"{var_exp[i0] * 100:.0f}% variance explained",
        fontsize=10,
    )
    axD.legend(fontsize=7)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[figure] wrote {args.out}")


if __name__ == "__main__":
    main()

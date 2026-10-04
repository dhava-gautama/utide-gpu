"""Atmospheric tides from ERA5 surface pressure — UTide beyond the ocean.

Fits the solar thermal tides (S1, S2) plus S4 and the annual/semiannual
cycles to hourly ERA5 mean-sea-level pressure at every grid cell over
Indonesia, in one `solve_many(..., gappy="ne")` call. The S1 field traces
land-sea heating contrast; S2 is the classical migrating atmospheric tide.

Expects ERA5 monthly netCDFs (hourly `msl`, variables valid_time/latitude/
longitude) as downloaded from the Copernicus Climate Data Store (CDS),
e.g. era5_sunda_202212.nc ... in --data-dir.

Usage:
    python examples/era5_atm_tides.py --data-dir DIR_WITH_ERA5_NC [--out PNG] [--cpu]
"""

import argparse
import glob
import time
import warnings

import numpy as np
import xarray as xr

CONSTS = ["S1", "S2", "S4", "SA", "SSA"]
EPOCH = np.datetime64("2000-01-01T00:00:00")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="era5_atm_tides_demo.png")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    files = sorted(glob.glob(args.data_dir + "/**/era5_*.nc", recursive=True))
    if not files:
        raise SystemExit(f"no era5_*.nc under {args.data_dir}")
    dss = [xr.open_dataset(f) for f in files]
    t = np.concatenate([ds.valid_time.values for ds in dss])
    msl = np.concatenate([ds.msl.values for ds in dss], axis=0)  # Pa
    lat0 = dss[0].latitude.values
    lon0 = dss[0].longitude.values
    order = np.argsort(lat0)
    msl = msl[:, order, :]
    lat0, lon0 = lat0[order], lon0
    print(
        f"[prep] {len(files)} files, {len(t)} hourly steps, grid {msl.shape[1]}x{msl.shape[2]}",
    )

    lat, lon = lat0[::2], lon0[::2]  # demo subsampling, 0.5 deg
    X = (msl[:, ::2, ::2] / 100.0).astype(np.float32)  # Pa -> hPa
    nt = X.shape[0]
    t_days = ((t.astype("datetime64[s]") - EPOCH) / np.timedelta64(1, "D")).astype(
        float,
    )

    try:
        import cupy  # noqa: F401
        import cupy.cuda.runtime as _rt

        _rt.getDeviceCount()
        gpu = not args.cpu
    except Exception:
        gpu = False
    from utide import solve_many

    kw = {
        "constit": CONSTS,
        "trend": True,
        "nodal": True,
        "epoch": EPOCH,
        "lat": np.broadcast_to(lat[:, None], (len(lat), len(lon))).ravel(),
        "verbose": False,
    }
    t0 = time.perf_counter()
    coef = solve_many(t_days, X.reshape(nt, -1), gpu=gpu, gappy="ne", **kw)
    print(
        f"[fit] {coef.A.shape[1]} cells x {len(CONSTS)} constituents "
        f"in {time.perf_counter() - t0:.2f}s ({'gpu' if gpu else 'cpu'})",
    )

    names = list(coef.name)
    iS1, iS2 = names.index("S1"), names.index("S2")
    print(
        f"[science] S1 median {np.nanmedian(coef.A[iS1]):.3f} hPa | "
        f"S2 median {np.nanmedian(coef.A[iS2]):.3f} hPa "
        "(classical values ~1.2 and ~1.1 hPa near the equator)",
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
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")

    fig = plt.figure(figsize=(13.0, 5.2))
    for k, (ii, ttl) in enumerate(
        [(iS1, "S1 (diurnal, solar thermal)"), (iS2, "S2 (semidiurnal, solar)")],
    ):
        if HAVE_CARTOPY:
            ax = fig.add_subplot(1, 2, k + 1, projection=ccrs.PlateCarree())
            ax.set_extent(
                [LON.min() - 0.5, LON.max() + 0.5, LAT.min() - 0.5, LAT.max() + 0.5],
                ccrs.PlateCarree(),
            )
            ax.add_feature(
                cfeature.LAND.with_scale("50m"),
                facecolor="#efe8d8",
                zorder=0,
            )
            ax.add_feature(
                cfeature.OCEAN.with_scale("50m"),
                facecolor="#d6e8f5",
                zorder=0,
            )
            ax.coastlines(resolution="50m", linewidth=0.5, zorder=2)
            transform = ccrs.PlateCarree()
        else:
            ax = fig.add_subplot(1, 2, k + 1)
            transform = None
        a = coef.A[ii]
        fi = np.isfinite(a)
        sc = ax.scatter(
            LON.ravel()[fi],
            LAT.ravel()[fi],
            c=a[fi],
            s=14,
            cmap="magma",
            transform=transform,
            zorder=5,
            vmin=0,
        )
        plt.colorbar(sc, ax=ax, label="amplitude [hPa]", shrink=0.8, pad=0.02)
        ax.set_title(
            f"{'AB'[k]} | ERA5 surface-pressure {ttl}\n"
            f"{int(fi.sum())} cells, one solve_many call",
            fontsize=10,
        )
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[figure] wrote {args.out}")


if __name__ == "__main__":
    main()

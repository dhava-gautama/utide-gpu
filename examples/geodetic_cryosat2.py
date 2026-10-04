"""Tidal analysis of geodetic-phase CryoSat-2 altimetry over the Sunda Strait.

Geodetic-phase ground tracks never repeat (369-day cycle, ~15 km spacing), so
this is the no-repeat extreme: observations are binned to 0.25-deg cells
(~95 irregular samples per cell over 3.3 years, each with a unique NaN
pattern) and fitted with one `solve_many(..., gappy="ne")` call. The margin
of the download box is trimmed BEFORE binning -- clipped edge bins have poor
phase coverage and produce degenerate fits.

The signal is `sla_unfiltered + ocean_tide` (the FES-based ocean-tide
correction undone). Comparing the fitted constituents against the same fit of
the `ocean_tide` series quantifies FES2014 error regionally; near the strait
narrows the altimetry agrees with FES2014 and both exceed TPXO9.

Usage:
    python examples/geodetic_cryosat2.py [--data-dir DIR] [--out PNG] [--cpu]

Requires numpy, pandas, matplotlib, (optional cartopy, optional cupy) and the
`copernicusmarine` CLI with free CMEMS credentials for the download step
(skipped if the CSVs are already in --data-dir).
"""

import argparse
import glob
import os
import subprocess
import time
import warnings

import numpy as np
import pandas as pd

DATASET = "cmems_obs-sl_glo_phy-ssh_my_c2n-l3-duacs_PT1S"
BOX = dict(x0=103.5, x1=108.5, y0=-8.5, y1=-4.0)
MARGIN = 0.25           # deg trimmed from the box before binning
T0, T1 = "2022-12-01T00:00:00", "2026-03-25T00:00:00"
BIN = 0.25              # deg; geodetic tracks do not repeat
MIN_OBS = 30
CONSTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
EPOCH = np.datetime64("2000-01-01T00:00:00")


def download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    if glob.glob(os.path.join(data_dir, "*", "*.csv")):
        print("[download] CSVs found, skipping")
        return
    cmd = ["copernicusmarine", "subset", "-i", DATASET,
           "-o", os.path.join(data_dir, "c2n"), "--force-download",
           "-x", str(BOX["x0"]), "-X", str(BOX["x1"]),
           "-y", str(BOX["y0"]), "-Y", str(BOX["y1"]), "-t", T0, "-T", T1]
    print("[download]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_series(data_dir):
    frames = [pd.read_csv(f) for f in glob.glob(os.path.join(data_dir, "*", "*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df = df[~df.value_qc.isin([4, 9])]
    piv = df.pivot_table(index=["time", "longitude", "latitude"], columns="variable",
                         values="value", aggfunc="first").reset_index()
    piv["time"] = pd.to_datetime(piv.time).dt.tz_localize(None)
    piv = piv.dropna(subset=["sla_unfiltered", "ocean_tide"])
    piv["y_obs"] = piv.sla_unfiltered + piv.ocean_tide
    piv["y_mod"] = piv.ocean_tide
    lon_lo, lon_hi = BOX["x0"] + MARGIN, BOX["x1"] - MARGIN
    lat_lo, lat_hi = BOX["y0"] + MARGIN, BOX["y1"] - MARGIN
    piv = piv[(piv.longitude >= lon_lo) & (piv.longitude <= lon_hi)
              & (piv.latitude >= lat_lo) & (piv.latitude <= lat_hi)]

    piv["ilon"] = np.round(piv.longitude / BIN).astype(int)
    piv["ilat"] = np.round(piv.latitude / BIN).astype(int)
    sizes = piv.groupby(["ilon", "ilat"]).size()
    piv = piv[piv.set_index(["ilon", "ilat"]).index.isin(sizes[sizes >= MIN_OBS].index)]
    print(f"[prep] {len(piv)} records in {piv.groupby(['ilon', 'ilat']).ngroups} bins")

    t = np.sort(piv.time.values.astype("datetime64[s]"))
    t_days = ((t - EPOCH) / np.timedelta64(1, "D")).astype(float)
    tix = {tv: i for i, tv in enumerate(t)}
    piv["ti"] = piv.time.map(lambda x: tix[np.datetime64(x, "s")])
    groups = list(piv.groupby(["ilon", "ilat"]))
    nt, S = len(t), len(groups)
    Xobs = np.full((nt, S), np.nan)
    Xmod = np.full((nt, S), np.nan)
    lat = np.zeros(S)
    lon = np.zeros(S)
    for k, (_, grp) in enumerate(groups):
        Xobs[grp.ti.values, k] = grp.y_obs.values
        Xmod[grp.ti.values, k] = grp.y_mod.values
        lat[k] = grp.latitude.mean()
        lon[k] = grp.longitude.mean()
    return t_days, Xobs, Xmod, lat, lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="cryosat2_data")
    ap.add_argument("--out", default="cryosat2_geodetic_demo.png")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    download(args.data_dir)
    t, Xobs, Xmod, lat, lon = build_series(args.data_dir)

    try:
        import cupy  # noqa: F401
        gpu = not args.cpu
    except ImportError:
        gpu = False
    from utide import solve_many

    kw = dict(constit=CONSTS, trend=True, nodal=True, epoch="2000-01-01",
              lat=lat, verbose=False)
    t0 = time.perf_counter()
    coef = solve_many(t, Xobs, gpu=gpu, gappy="ne", **kw)
    coef_mod = solve_many(t, Xmod, gpu=gpu, gappy="ne", **kw)
    print(f"[fit] {coef.A.shape[1]} bins x {len(CONSTS)} constituents "
          f"in {time.perf_counter() - t0:.2f} s ({'gpu' if gpu else 'cpu'})")

    names = list(coef.name)
    m2 = names.index("M2")
    fin = np.isfinite(coef.A[m2]) & np.isfinite(coef_mod.A[m2])
    r = np.corrcoef(coef.A[m2][fin], coef_mod.A[m2][fin])[0, 1]
    med = np.median(np.abs(coef.A[m2][fin] - coef_mod.A[m2][fin])) * 100
    print(f"[validation] M2 vs FES-based model: r={r:.3f}, median |dA|={med:.1f} cm "
          f"over {int(fin.sum())} bins (differences quantify FES2014 error)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAVE_CARTOPY = True
    except ImportError:
        HAVE_CARTOPY = False

    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.22)
    extent = [lon.min() - 0.3, lon.max() + 0.3, lat.min() - 0.3, lat.max() + 0.3]
    transform = None
    if HAVE_CARTOPY:
        axA = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        axA.set_extent(extent, ccrs.PlateCarree())
        axA.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#efe8d8", zorder=0)
        axA.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#d6e8f5", zorder=0)
        axA.coastlines(resolution="10m", linewidth=0.5, zorder=2)
        transform = ccrs.PlateCarree()
    else:
        axA = fig.add_subplot(gs[0, 0])
        axA.set_xlim(extent[:2])
        axA.set_ylim(extent[2:])
    sc = axA.scatter(lon, lat, c=coef.A[m2] * 100, s=14, cmap="viridis", vmin=0,
                     vmax=max(60, np.nanpercentile(coef.A[m2][fin], 99)),
                     edgecolors="k", linewidths=0.2, transform=transform, zorder=5)
    plt.colorbar(sc, ax=axA, label="fitted M2 amplitude (cm)", shrink=0.8, pad=0.02)
    axA.set_title("A | CryoSat-2 geodetic-phase M2 (2022-12 – 2026-03)\n"
                  f"{coef.A.shape[1]} bins — one solve_many(gappy='ne') call",
                  fontsize=10)
    axB = fig.add_subplot(gs[0, 1])
    lims = [0, max(coef.A[m2][fin].max(), coef_mod.A[m2][fin].max()) * 110]
    axB.plot(lims, lims, "k--", lw=0.8)
    axB.scatter(coef_mod.A[m2][fin] * 100, coef.A[m2][fin] * 100, s=12, alpha=0.65)
    axB.set_xlabel("FES-based model M2 amp [cm]")
    axB.set_ylabel("CryoSat-2-fitted M2 amp [cm]")
    axB.set_title(f"B | per-bin M2: fit vs FES model\nr={r:.3f}, median |dA|={med:.1f} cm",
                  fontsize=10)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[figure] wrote {args.out}")


if __name__ == "__main__":
    main()

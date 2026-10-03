"""Harmonic analysis of real satellite altimetry over the Sunda Strait.

Downloads CMEMS DUACS L3 multi-year along-track sea level (Sentinel-6a LR and
Jason-3n, 1 Hz), builds one gappy point series per ~3 km along-track location
(`sla_unfiltered + ocean_tide`, i.e. the main ocean-tide correction undone so
the tide is back in the observable), fits 12 constituents for every point in a
single `solve_many(gappy="ne")` call, and writes a validation figure against
the FES-based DUACS tide model.

The altimetry use case is the extreme one for the batched solver: every point
series has its own irregular sample times (~75 observations over 3.3 years on
a shared 33,000-step time base), so series cannot be grouped by gap pattern
and the masked normal-equations path (`gappy="ne"`) is what keeps it batched.

Usage:
    python examples/altimetry_sunda.py [--data-dir DIR] [--out PNG] [--cpu]

Requires: numpy, pandas, matplotlib, xarray-free; optional cupy for the GPU
path; the `copernicusmarine` CLI (https://github.com/mercatorocean/copernicus-marine)
with free CMEMS credentials for the download step (skipped if the CSVs are
already in --data-dir).
"""

import argparse
import glob
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

DATASETS = {
    "s6a": "cmems_obs-sl_glo_phy-ssh_my_s6a-lr-l3-duacs_PT1S",
    "j3n": "cmems_obs-sl_glo_phy-ssh_my_j3n-l3-duacs_PT1S",
}
BOX = dict(x0=103.5, x1=108.5, y0=-8.5, y1=-4.0)  # Sunda Strait + shelves
T0, T1 = "2022-12-01T00:00:00", "2026-03-25T00:00:00"
BIN = 0.03          # deg, ~3.3 km along-track clustering
MIN_OBS = 25
CONSTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MU2", "NU2", "L2", "2N2"]
EPOCH = np.datetime64("2000-01-01T00:00:00")


def download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    have = glob.glob(os.path.join(data_dir, "*", "*.csv"))
    if have:
        print(f"[download] found {len(have)} CSV(s) under {data_dir}, skipping")
        return
    for tag, ds in DATASETS.items():
        cmd = [
            "copernicusmarine", "subset", "-i", ds,
            "-o", os.path.join(data_dir, tag), "--force-download",
            "-x", str(BOX["x0"]), "-X", str(BOX["x1"]),
            "-y", str(BOX["y0"]), "-Y", str(BOX["y1"]),
            "-t", T0, "-T", T1,
        ]
        print("[download]", " ".join(cmd))
        subprocess.run(cmd, check=True)


def build_series(data_dir):
    frames = [pd.read_csv(f) for f in glob.glob(os.path.join(data_dir, "*", "*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df = df[~df.value_qc.isin([4, 9])]  # drop only explicitly bad/missing flags
    piv = df.pivot_table(index=["time", "longitude", "latitude"], columns="variable",
                         values="value", aggfunc="first").reset_index()
    piv["time"] = pd.to_datetime(piv.time).dt.tz_localize(None)
    piv = piv.dropna(subset=["sla_unfiltered", "ocean_tide"])
    piv["y_obs"] = piv.sla_unfiltered + piv.ocean_tide   # tide back in the observable
    piv["y_mod"] = piv.ocean_tide                        # FES-based model, for comparison

    piv["ilon"] = np.round(piv.longitude / BIN).astype(int)
    piv["ilat"] = np.round(piv.latitude / BIN).astype(int)
    sizes = piv.groupby(["ilon", "ilat"]).size()
    keep = sizes[sizes >= MIN_OBS].index
    piv = piv[piv.set_index(["ilon", "ilat"]).index.isin(keep)]
    print(f"[prep] {len(piv)} records in {len(keep)} point series "
          f"(median {int(sizes[sizes.index.isin(keep)].median())} obs each)")

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
    ap.add_argument("--data-dir", default="altimetry_data")
    ap.add_argument("--out", default="altimetry_utide_demo.png")
    ap.add_argument("--cpu", action="store_true", help="skip the GPU even if available")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    download(args.data_dir)
    t, Xobs, Xmod, lat, lon = build_series(args.data_dir)

    try:
        import cupy  # noqa: F401
        gpu = not args.cpu
    except ImportError:
        gpu = False
    from utide import solve_many, reconstruct_many

    kw = dict(constit=CONSTS, trend=True, nodal=True, epoch="2000-01-01",
              lat=lat, verbose=False)
    t0 = time.perf_counter()
    coef = solve_many(t, Xobs, gpu=gpu, gappy="ne", **kw)
    print(f"[fit] {coef.A.shape[1]} series x {len(CONSTS)} constituents "
          f"in {time.perf_counter() - t0:.2f} s ({'gpu' if gpu else 'cpu'})")
    coef_mod = solve_many(t, Xmod, gpu=gpu, gappy="ne", **kw)
    pred = reconstruct_many(t, coef, epoch="2000-01-01", gpu=gpu)

    names = list(coef.name)
    A_obs, g_obs = coef.A, coef.g
    A_mod, g_mod = coef_mod.A, coef_mod.g
    pdiff = (g_obs - g_mod + 180) % 360 - 180
    m2 = names.index("M2")
    fin = np.isfinite(A_obs[m2]) & np.isfinite(A_mod[m2])
    r = np.corrcoef(A_obs[m2][fin], A_mod[m2][fin])[0, 1]
    rmsd = np.sqrt(np.mean((A_obs[m2][fin] - A_mod[m2][fin]) ** 2)) * 100
    print(f"[validation] M2 vs FES-based model: r={r:.3f}, rmsd={rmsd:.1f} cm "
          f"over {int(fin.sum())} points")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 10.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], height_ratios=[1, 1.05, 0.8],
                          hspace=0.42, wspace=0.24)
    axA = fig.add_subplot(gs[:, 0])
    sc = axA.scatter(lon, lat, c=A_obs[m2] * 100,
                     s=6 + np.sum(np.isfinite(A_obs), axis=0) / 4, cmap="viridis",
                     edgecolors="k", linewidths=0.2)
    plt.colorbar(sc, ax=axA, label="fitted M2 amplitude (cm)")
    axA.set_xlabel("Longitude [deg E]")
    axA.set_ylabel("Latitude [deg N]")
    axA.set_title("A | UTide harmonic fit of along-track altimetry (S6a + J3n)\n"
                  f"{A_obs.shape[1]} point series, one solve_many(gappy='ne') call",
                  fontsize=10)
    axB = fig.add_subplot(gs[0, 1])
    lims = [0, max(A_obs[m2][fin].max(), A_mod[m2][fin].max()) * 105]
    axB.plot(lims, lims, "k--", lw=0.8)
    axB.scatter(A_mod[m2][fin] * 100, A_obs[m2][fin] * 100, s=9, alpha=0.6)
    axB.set_xlabel("FES-based model M2 amp [cm]")
    axB.set_ylabel("altimetry-fitted M2 amp [cm]")
    axB.set_title(f"B | per-point M2 amplitude: fit vs model\nr={r:.3f}, rmsd={rmsd:.1f} cm",
                  fontsize=10)
    axC = fig.add_subplot(gs[1, 1])
    med_dA, med_dp = [], []
    for i in range(len(names)):
        fi = np.isfinite(A_obs[i]) & np.isfinite(A_mod[i])
        med_dA.append(np.median(np.abs(A_obs[i][fi] - A_mod[i][fi])) * 100)
        med_dp.append(np.median(np.abs(pdiff[i][fi])))
    x = np.arange(len(names))
    axC.bar(x - 0.2, med_dA, 0.4, label="median |dA| vs model [cm]")
    axC.bar(x + 0.2, med_dp, 0.4, color="tab:orange", label="median |dphase| [deg]")
    axC.set_xticks(x, names)
    axC.legend(fontsize=8)
    axC.set_title("C | altimetry fit vs FES model, all points", fontsize=10)
    axD = fig.add_subplot(gs[2, 1])
    i0 = int(np.argmax(np.sum(np.isfinite(A_obs), axis=0)))
    mask = np.isfinite(Xobs[:, i0])
    w = t <= t[0] + 100.0  # first 100 days
    axD.plot(t[w] - t[0], pred[w, i0] * 100, "b-", lw=1.1, label="UTide prediction")
    axD.plot(t[w & mask] - t[0], Xobs[w & mask, i0] * 100, "k.", ms=5, label="altimetry obs")
    axD.plot(t[w & mask] - t[0], Xmod[w & mask, i0] * 100, "r.", ms=3.5, alpha=0.65,
             label="FES-based model")
    res = np.nanstd((Xobs[w & mask, i0] - pred[w & mask, i0]) * 100)
    axD.set_xlabel(f"days since {str(t[0])[:10]}")
    axD.set_ylabel("sea level [cm]")
    axD.set_title(f"D | best-observed point ({lat[i0]:.2f}N, {lon[i0]:.2f}E): "
                  f"residual std {res:.1f} cm", fontsize=10)
    axD.legend(fontsize=7)
    fig.suptitle("UTide on real satellite altimetry — Sunda Strait", fontsize=13)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[figure] wrote {args.out}")


if __name__ == "__main__":
    main()

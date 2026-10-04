"""One solve_many call across a whole tide-gauge network (NOAA CO-OPS).

Downloads 2019-2021 verified hourly heights for every CO-OPS water-level
station (public API, threaded), stacks all stations on one shared time base
(each station keeps its own NaN gap pattern), and fits 10 constituents for
the whole network in a single `solve_many(..., gappy="ne")` call.

Usage:
    python examples/coops_network.py [--data-dir DIR] [--out PNG] [--cpu]
"""

import argparse
import json
import os
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

YEARS = [2019, 2020, 2021]
BASE = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        "?product=hourly_height&station={st}&datum=MSL&units=metric"
        "&time_zone=gmt&format=json&begin_date={b}&end_date={e}")
CONSTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "SA", "SSA"]
EPOCH = np.datetime64("2000-01-01T00:00:00")


def download(data_dir, min_obs=500, workers=10):
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, "coops_network_long.csv")
    if os.path.exists(cache):
        print("[download] cached, skipping")
        return pd.read_csv(cache)
    with urllib.request.urlopen("https://api.tidesandcurrents.noaa.gov/mdapi/prod/"
                                "webapi/stations.json?type=waterlevels",
                                timeout=90) as r:
        meta = json.loads(r.read().decode())
    stas = [(s["id"], s.get("lat"), s.get("lng")) for s in meta.get("stations", [])
            if s.get("lat") is not None]
    print(f"[download] {len(stas)} candidate stations")

    def fetch(s):
        st, lat, lng = s
        rows = []
        for y in YEARS:
            try:
                with urllib.request.urlopen(
                        BASE.format(st=st, b=f"{y}0101", e=f"{y}1231"),
                        timeout=60) as r:
                    d = json.loads(r.read().decode())
                for row in d.get("data", []):
                    try:
                        rows.append((st, row["t"], float(row["v"])))
                    except (ValueError, KeyError):
                        pass
            except Exception:  # noqa: BLE001
                pass
        return st, lat, lng, rows

    recs, keep = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for st, lat, lng, rows in ex.map(fetch, stas):
            if len(rows) >= min_obs:
                keep.append((st, lat, lng, len(rows)))
                recs.extend(rows)
    long = pd.DataFrame(recs, columns=["station", "time", "v"])
    long["time"] = pd.to_datetime(long.time)
    long = long.drop_duplicates(["station", "time"])
    long.to_csv(cache, index=False)
    pd.DataFrame(keep, columns=["station", "lat", "lng", "n"]).to_csv(
        cache.replace("_long.csv", "_meta.csv"), index=False)
    print(f"[download] {len(keep)} usable stations, {len(long)} rows -> {cache}")
    return long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="coops_network_data")
    ap.add_argument("--out", default="coops_network_demo.png")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    long = download(args.data_dir)
    meta = pd.read_csv(os.path.join(args.data_dir, "coops_network_meta.csv"),
                       dtype={"station": str})
    long["time"] = pd.to_datetime(long.time)
    t = np.sort(long.time.unique())
    t_days = ((t.astype("datetime64[s]") - EPOCH) / np.timedelta64(1, "D")).astype(float)
    tix = {tv: i for i, tv in enumerate(t)}
    long["ti"] = long.time.map(lambda x: tix[np.datetime64(x, "s")])
    stas = meta.station.astype(str).tolist()
    X = np.full((len(t), len(stas)), np.nan)
    long["station"] = long["station"].astype(str)
    lat = meta.lat.values.astype(float)
    lon = meta.lng.values.astype(float)
    sidx = {st: k for k, st in enumerate(stas)}
    for st, grp in long.groupby("station"):
        X[grp.ti.values, sidx[st]] = grp.v.values
    print(f"[prep] {len(t)} shared hours, {len(stas)} stations")

    try:
        import cupy  # noqa: F401
        import cupy.cuda.runtime as _rt
        _rt.getDeviceCount()
        gpu = not args.cpu
    except Exception:
        gpu = False
    from utide import solve_many

    kw = dict(constit=CONSTS, trend=True, nodal=True, epoch=EPOCH,
              lat=lat, verbose=False)
    t0 = time.perf_counter()
    coef = solve_many(t_days, X, gpu=gpu, gappy="ne", **kw)
    print(f"[fit] {coef.A.shape[1]} stations x {len(CONSTS)} constituents "
          f"in {time.perf_counter() - t0:.2f}s ({'gpu' if gpu else 'cpu'})")

    names = list(coef.name)
    iM2 = names.index("M2")
    fin = np.isfinite(coef.A[iM2])
    top = np.argsort(coef.A[iM2][fin])[::-1][:5]
    print("[science] top M2 stations (cm):",
          [(stas[np.flatnonzero(fin)[i]], round(float(coef.A[iM2][fin][i]) * 100, 1))
           for i in top])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAVE_CARTOPY = True
    except ImportError:
        HAVE_CARTOPY = False

    fig = plt.figure(figsize=(13.0, 5.6))
    for k, (ii, ttl) in enumerate([(iM2, "M2 (semidiurnal, lunar)"),
                                   (names.index("K1"), "K1 (diurnal)")]):
        if HAVE_CARTOPY:
            ax = fig.add_subplot(1, 2, k + 1, projection=ccrs.PlateCarree())
            ax.set_global()
            ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#efe8d8",
                           zorder=0)
            ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#d6e8f5",
                           zorder=0)
            ax.coastlines(resolution="110m", linewidth=0.4, zorder=2)
            transform = ccrs.PlateCarree()
        else:
            ax = fig.add_subplot(1, 2, k + 1)
            transform = None
        a = coef.A[ii] * 100
        fi = np.isfinite(a)
        sc = ax.scatter(lon[fi], lat[fi], c=a[fi], s=6 + meta.n.values[fi] / 400,
                        cmap="plasma", transform=transform, zorder=5, vmin=0)
        plt.colorbar(sc, ax=ax, label="amplitude [cm]", shrink=0.75, pad=0.02)
        ax.set_title(f"{'CD'[k]} | NOAA CO-OPS network: {ttl}\n"
                     f"{int(fi.sum())} stations, 2019-2021, one solve_many call",
                     fontsize=10)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[figure] wrote {args.out}")


if __name__ == "__main__":
    main()

"""Generate notebooks/gpu_batch_real_example.ipynb (valid nbformat 4)."""

import json


def md(*l):
    return {"cell_type": "markdown", "metadata": {}, "source": [x + "\n" for x in l]}


def code(s):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [x + "\n" for x in s.strip("\n").split("\n")],
    }


cells = [
    md(
        "# The GPU batch use case on real data: 39 NOAA tide gauges",
        "",
        "One year of hourly water level at 39 [NOAA CO-OPS](https://tidesandcurrents.noaa.gov)",
        "stations is stored as a single `(n_times, n_stations)` array. One `solve_many`",
        "call recovers the tidal constituents at **every** station, from which we map the",
        "M2 amplitude and classify each station's tidal regime — and the result matches the",
        "known oceanography (huge tides in Cook Inlet and the Bay of Fundy; a diurnal Gulf",
        "of Mexico).",
        "",
        "*Data: NOAA CO-OPS, public domain.*",
    ),
    code(
        "%matplotlib inline\n"
        "import time\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import utide",
    ),
    md("## 1. Load the stations"),
    code(
        'd = np.load("../examples/data/noaa_hourly_2023.npz", allow_pickle=True)\n'
        't = d["t_days"].astype(float)\n'
        'X = d["levels"].astype(float)          # (n_times, n_stations), metres\n'
        'lats, lons, names = d["lats"], d["lons"], d["names"]\n'
        "lat0 = float(np.median(lats))\n"
        "print(X.shape, '-> 39 stations x 8760 hours')",
    ),
    md(
        "## 2. One `solve_many` call for every station",
        "",
        "`solve_many` uses a single latitude for the (small) nodal correction; the median",
        "of the stations is a fine choice for a one-year fit.",
    ),
    code(
        'kw = dict(lat=lat0, epoch="2023-01-01", verbose=False)\n'
        "utide.solve_many(t, X[:, :4], gpu=True, **kw)        # warm up the GPU\n"
        "t0 = time.perf_counter()\n"
        "out = utide.solve_many(t, X, gpu=True, **kw)\n"
        "gpu_t = time.perf_counter() - t0\n"
        "\n"
        'm2 = list(out.name).index("M2")\n'
        "ns = 10\n"
        "sa = np.empty(ns)\n"
        "t0 = time.perf_counter()\n"
        "for s in range(ns):\n"
        '    c = utide.solve(t, X[:, s], method="ols", conf_int="none", **kw)\n'
        '    sa[s] = c["A"][list(c["name"]).index("M2")]\n'
        "per = (time.perf_counter() - t0) / ns\n"
        'print(f"solve_many: {X.shape[1]} stations in {gpu_t:.2f}s")\n'
        'print(f"per-station solve(): {1000*per:.0f} ms -> naive loop ~= {per*X.shape[1]:.1f}s "\n'
        '      f"({per*X.shape[1]/gpu_t:.0f}x slower)")\n'
        'print(f"solve_many vs per-station solve(): max rel diff = "\n'
        '      f"{np.max(np.abs(sa - out.A[m2, :ns]) / sa):.1e}")',
    ),
    md(
        "## 3. Maps and tidal classification",
        "",
        "From the per-station amplitudes we form the tidal *form factor*",
        "F = (K1+O1)/(M2+S2): F < 0.25 is semidiurnal, F > 1.5 is (mixed-)diurnal.",
        "",
        "*(Install `cartopy` for the coastline basemap; without it the points are",
        "plotted on plain longitude/latitude axes.)*",
    ),
    code(
        "def amp(n): return out.A[list(out.name).index(n)]\n"
        'M2, S2, K1, O1 = amp("M2"), amp("S2"), amp("K1"), amp("O1")\n'
        "form = (K1 + O1) / (M2 + S2)\n"
        "hi, lo = np.argmax(M2), np.argmin(M2)\n"
        'print(f"largest M2: {names[hi]} ({M2[hi]:.2f} m); smallest: {names[lo]} ({M2[lo]:.3f} m)")\n'
        'print("diurnal/mixed-diurnal (form>1.5):", ", ".join(np.asarray(names)[form > 1.5]))\n'
        "\n"
        "try:\n"
        "    import cartopy.crs as ccrs, cartopy.feature as cfeature\n"
        '    proj = ccrs.PlateCarree(); spk = {"projection": proj}\n'
        "except ImportError:\n"
        "    proj = spk = None\n"
        "fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=spk)\n"
        'for a, (title, c, cmap) in zip(ax, [("M2 amplitude (m)", M2, "viridis"),\n'
        '        ("Tidal form factor (K1+O1)/(M2+S2)", np.clip(form, 0, 4), "coolwarm")]):\n'
        "    if proj is not None:\n"
        '        a.add_feature(cfeature.LAND, facecolor="0.93")\n'
        "        a.add_feature(cfeature.COASTLINE, linewidth=0.4)\n"
        '        a.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="0.7")\n'
        "        a.set_extent([-180, -64, 16, 64], crs=proj)\n"
        '        sc = a.scatter(lons, lats, c=c, s=55, cmap=cmap, edgecolor="k", lw=0.3, transform=proj, zorder=5)\n'
        "    else:\n"
        '        sc = a.scatter(lons, lats, c=c, s=55, cmap=cmap, edgecolor="k", lw=0.3); a.set_xlim(-180, -64)\n'
        "    a.set_title(title); fig.colorbar(sc, ax=a, shrink=0.8)\n"
        "fig.tight_layout(); plt.show()",
    ),
    md(
        "That co-amplitude map and tidal classification for all 39 stations came from a",
        "single `solve_many` call. Swap in thousands of model-grid cells or altimetry",
        "points and the same one call scales to them — see `gpu_batch_example.ipynb`.",
    ),
]
for i, c in enumerate(cells):
    c["id"] = f"cell{i}"
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("notebooks/gpu_batch_real_example.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/gpu_batch_real_example.ipynb with", len(cells), "cells")

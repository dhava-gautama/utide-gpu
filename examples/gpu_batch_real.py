"""
The GPU batch use case on REAL data.

Loads one year of hourly water level for 39 NOAA tide-gauge stations (a single
(n_times, n_stations) array) and recovers the tidal constituents at every
station with one ``solve_many`` call. Produces a US co-amplitude map of M2 and
a tidal-type map (form factor), and reports the speed-up over the per-station
loop.

Data: NOAA CO-OPS (https://tidesandcurrents.noaa.gov), public domain.
"""
import os
import time

import numpy as np

import utide

HERE = os.path.dirname(__file__)
d = np.load(os.path.join(HERE, "data", "noaa_hourly_2023.npz"), allow_pickle=True)
t = d["t_days"].astype(float)
X = d["levels"].astype(float)
lats, lons, names = d["lats"], d["lons"], d["names"]
nt, nstn = X.shape
lat0 = float(np.median(lats))
print(f"{nstn} stations x {nt} hours (2023 hourly)")

# --- one solve_many call for every station --------------------------------
kw = dict(lat=lat0, epoch="2023-01-01", verbose=False)
utide.solve_many(t, X[:, :4], gpu=True, **kw)              # warm up
t0 = time.perf_counter()
out = utide.solve_many(t, X, gpu=True, **kw)
gpu_t = time.perf_counter() - t0
print(f"solve_many: {nstn} stations in {gpu_t:.2f}s")

# naive per-station loop (timed on a sample) + exact-agreement check
m2 = list(out.name).index("M2")
ns = 10
sa = np.empty(ns)
t0 = time.perf_counter()
for s in range(ns):
    c = utide.solve(t, X[:, s], method="ols", conf_int="none", **kw)
    sa[s] = c["A"][list(c["name"]).index("M2")]
per = (time.perf_counter() - t0) / ns
print(f"per-station solve(): {1000*per:.0f} ms -> naive loop ~= {per*nstn:.1f}s "
      f"({per*nstn/gpu_t:.0f}x slower)")
print(f"solve_many vs per-station solve(): max rel diff = "
      f"{np.max(np.abs(sa - out.A[m2, :ns]) / sa):.1e}")

# --- real tidal insight: amplitudes + form factor -------------------------
def amp(name):
    return out.A[list(out.name).index(name)] if name in list(out.name) else np.zeros(nstn)

M2, S2, K1, O1 = amp("M2"), amp("S2"), amp("K1"), amp("O1")
form = (K1 + O1) / (M2 + S2)        # tidal form factor
kind = np.where(form < 0.25, "semidiurnal",
        np.where(form < 1.5, "mixed (semi)",
         np.where(form < 3.0, "mixed (diurnal)", "diurnal")))
hi, lo = np.argmax(M2), np.argmin(M2)
print(f"\nlargest M2: {names[hi]} ({M2[hi]:.2f} m);  smallest: {names[lo]} ({M2[lo]:.3f} m)")
print(f"diurnal/mixed-diurnal stations (form>1.5): "
      f"{', '.join(np.asarray(names)[form > 1.5][:6])} ...")

# --- maps (with a coastline basemap if cartopy is available) --------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree()
        spk = {"projection": proj}
    except ImportError:
        proj = spk = None

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=spk)
    panels = [("M2 amplitude (m)", M2, "viridis"),
              ("Tidal form factor  (K1+O1)/(M2+S2)", np.clip(form, 0, 4), "coolwarm")]
    for a, (title, c, cmap) in zip(ax, panels):
        if proj is not None:
            a.add_feature(cfeature.LAND, facecolor="0.93")
            a.add_feature(cfeature.OCEAN, facecolor="0.99")
            a.add_feature(cfeature.COASTLINE, linewidth=0.4)
            a.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="0.7")
            a.set_extent([-180, -64, 16, 64], crs=proj)
            sc = a.scatter(lons, lats, c=c, s=55, cmap=cmap, edgecolor="k",
                           lw=0.3, transform=proj, zorder=5)
        else:
            sc = a.scatter(lons, lats, c=c, s=55, cmap=cmap, edgecolor="k", lw=0.3)
            a.set_xlim(-180, -64); a.set_xlabel("longitude"); a.set_ylabel("latitude")
        a.set_title(title); fig.colorbar(sc, ax=a, shrink=0.8)
    fig.suptitle(f"Tides at {nstn} NOAA stations from one solve_many call "
                 f"({per*nstn/gpu_t:.0f}x faster than looping)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "noaa_tides.png"), dpi=95)
    print("\nsaved figure -> examples/noaa_tides.png")
except ImportError:
    print("(matplotlib not installed; skipping plot)")

"""Build the real-data BADA batch used by smoke_bada.py (run on a box with xarray).

Reads the hourly QC file from the radarMaritim repo (native path preferred,
/mnt/h drvfs fallback), selects sea cells with >= 720 hourly observations,
and writes /tmp/bada_utide_batch.npz (U, V, per-cell lat/lon, days since
2000-01-01). Regenerate whenever the source netCDF changes.
"""

import os
import sys

import numpy as np

for p in (os.path.expanduser("~/radarMaritim"), "/mnt/h/rocket/radarMaritim"):
    if os.path.isdir(p):
        REPO = p
        break
else:
    raise SystemExit(
        "radarMaritim repo not found (looked in ~/radarMaritim, /mnt/h/rocket/radarMaritim)",
    )

sys.path.insert(0, os.path.join(REPO, "profiling", "CODAR_BADA"))
import xarray as xr
from common import F_DOMAIN, F_HOURLY

h = xr.open_dataset(F_HOURLY)
dom = xr.open_dataset(F_DOMAIN)
t = h.time.values
Ur, Vr = h.U_raw.values, h.V_raw.values
glon = np.asarray(dom.longitude.values, float)
glat = np.asarray(dom.latitude.values, float)
obs = np.isfinite(Ur) & np.isfinite(Vr)
n_obs = obs.sum(axis=0)
sel = [c for c in range(n_obs.size) if n_obs.reshape(-1)[c] >= 720]
ii, jj = np.unravel_index(np.asarray(sel), n_obs.shape)
XU = np.stack([Ur[:, i, j] for i, j in zip(ii, jj, strict=False)], axis=1)
XV = np.stack([Vr[:, i, j] for i, j in zip(ii, jj, strict=False)], axis=1)
t_days = ((t - np.datetime64("2000-01-01T00:00:00")) / np.timedelta64(1, "D")).astype(
    np.float64,
)
np.savez_compressed(
    "/tmp/bada_utide_batch.npz",
    t=t_days,
    U=XU,
    V=XV,
    lat=glat[ii],
    lon=glon[jj],
    ii=ii,
    jj=jj,
)
print(
    f"wrote /tmp/bada_utide_batch.npz: nt={len(t)} S={len(sel)} "
    f"({t[0]} .. {t[-1]})",
)

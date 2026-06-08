"""
Fetch one year of hourly water level for a set of NOAA CO-OPS stations and save
a single (n_times, n_stations) dataset for the solve_many real-data example.

Data: NOAA CO-OPS (https://tidesandcurrents.noaa.gov), public domain.
"""

import json
import urllib.request
from datetime import datetime

import numpy as np

YEAR = 2023
# Stations spanning very different tidal regimes (huge semidiurnal in the Gulf
# of Maine / Bay of Fundy & Cook Inlet; diurnal in the Gulf of Mexico; mixed on
# the Pacific coast).
STATIONS = [
    "8410140",
    "8418150",
    "8443970",
    "8447930",
    "8449130",
    "8461490",
    "8510560",
    "8516945",
    "8531680",
    "8534720",
    "8557380",
    "8574680",
    "8594900",
    "8638610",
    "8651370",
    "8656483",
    "8665530",
    "8670870",
    "8720218",
    "8723214",
    "8724580",
    "8729108",
    "8735180",
    "8761724",
    "8770570",
    "8771341",
    "8779770",
    "9410230",
    "9410660",
    "9413450",
    "9414290",
    "9418767",
    "9435380",
    "9444900",
    "9447130",
    "9455920",
    "9457292",
    "9461380",
    "1612340",
    "1617433",
]

start = datetime(YEAR, 1, 1)
end = datetime(YEAR, 12, 31, 23)
nt = int((end - start).total_seconds() // 3600) + 1
t_days = np.arange(nt) / 24.0  # days since YEAR-01-01

URL = (
    "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
    "product=hourly_height&application=utide_gpu_example&begin_date={y}0101&"
    "end_date={y}1231&datum=MSL&station={s}&time_zone=gmt&units=metric&format=json"
)

levels, ids, names, lats, lons = [], [], [], [], []
for s in STATIONS:
    try:
        with urllib.request.urlopen(URL.format(y=YEAR, s=s), timeout=60) as r:
            d = json.load(r)
        if "data" not in d:
            print(f"  {s}: no data ({d.get('error', {}).get('message', '?')[:40]})")
            continue
        col = np.full(nt, np.nan)
        for row in d["data"]:
            v = row.get("v", "")
            if v == "":
                continue
            dt = datetime.strptime(row["t"], "%Y-%m-%d %H:%M")
            i = int((dt - start).total_seconds() // 3600)
            if 0 <= i < nt:
                col[i] = float(v)
        ngood = int(np.isfinite(col).sum())
        if ngood < nt * 0.5:
            print(f"  {s}: only {ngood} good samples, skipping")
            continue
        m = d["metadata"]
        levels.append(col.astype(np.float32))
        ids.append(s)
        names.append(m["name"])
        lats.append(float(m["lat"]))
        lons.append(float(m["lon"]))
        print(f"  {s} {m['name']:24s} lat={float(m['lat']):6.2f} good={ngood}/{nt}")
    except Exception as e:  # noqa: BLE001
        print(f"  {s}: FAILED ({type(e).__name__})")

X = np.column_stack(levels)
out = f"examples/data/noaa_hourly_{YEAR}.npz"
np.savez_compressed(
    out,
    t_days=t_days.astype(np.float32),
    levels=X,
    ids=np.array(ids),
    names=np.array(names),
    lats=np.array(lats),
    lons=np.array(lons),
    year=YEAR,
)
print(
    f"\nsaved {out}: {X.shape[1]} stations x {X.shape[0]} hours "
    f"({np.isnan(X).mean()*100:.1f}% gaps overall)",
)

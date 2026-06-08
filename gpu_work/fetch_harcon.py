"""Fetch NOAA official harmonic constants for the example stations (public domain)."""
import json, urllib.request
import numpy as np

d = np.load("examples/data/noaa_hourly_2023.npz", allow_pickle=True)
ids = [str(x) for x in d["ids"]]
out = {}
for sid in ids:
    url = f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{sid}/harcon.json?units=metric"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            j = json.load(r)
        hc = {h["name"]: [round(h["amplitude"], 4), round(h["phase_GMT"], 2)]
              for h in j.get("HarmonicConstituents", []) if h["amplitude"] > 0}
        if hc:
            out[sid] = hc
            print(f"  {sid}: {len(hc)} constituents")
    except Exception as e:  # noqa
        print(f"  {sid}: FAILED ({type(e).__name__})")
with open("examples/data/noaa_harcon.json", "w") as f:
    json.dump(out, f, indent=0, sort_keys=True)
print(f"\nsaved examples/data/noaa_harcon.json: {len(out)} stations")

"""
Can UTide recover tides from satellite-altimetry-style sampling?

Altimeters revisit a point once per repeat cycle (Jason/TOPEX ~9.9156 days),
far slower than the tides, so constituents alias to long apparent periods.
UTide builds its basis at the true sample times, so the aliasing is handled
implicitly -- IF the record is long enough to separate the aliased frequencies.

Here we take a real tide (Boston's constituents from one year of hourly data),
sample it on a Jason-style orbit for ~25 years, add ~3 cm of noise, and see how
well UTide recovers the constituents.
"""
import sys; sys.path.insert(0, "/mnt/hdd/rocket/UTide")
import warnings; warnings.filterwarnings("ignore")
import numpy as np

import utide

# --- a real tide: Boston constituents from 1 year of hourly data ----------
d = np.load("examples/data/noaa_hourly_2023.npz", allow_pickle=True)
ids = [str(x) for x in d["ids"]]
s = ids.index("8443970")  # Boston
t_h = d["t_days"].astype(float)
h = d["levels"][:, s].astype(float)
constit = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
truth = utide.solve(t_h, h, lat=42.35, constit=constit, method="ols",
                    conf_int="linear", epoch="2023-01-01", verbose=False)

# --- sample that tide on a Jason-style 9.9156-day repeat for 25 years ------
REPEAT = 9.9156
years = 25
t_sat = np.arange(0, years * 365.25, REPEAT)
rng = np.random.default_rng(0)
true_at_sat = utide.reconstruct(t_sat, truth, epoch="2023-01-01", min_SNR=0, verbose=False).h
obs = true_at_sat + 0.03 * rng.standard_normal(t_sat.size)   # ~3 cm altimeter noise
print(f"satellite sampling: {t_sat.size} samples over {years} yr "
      f"(every {REPEAT} d); Nyquist period {2*REPEAT:.1f} d")

# --- recover from the satellite-sampled series -----------------------------
rec = utide.solve(t_sat, obs, lat=42.35, constit=constit, method="ols",
                  conf_int="linear", epoch="2023-01-01", verbose=False)
ti = {n: i for i, n in enumerate(truth["name"])}
ri = {n: i for i, n in enumerate(rec["name"])}
print(f"\n{'constit':7} {'true A':>8} {'sat A':>8} {'dA%':>6}  {'true g':>7} {'sat g':>7} {'dg':>6}")
for c in constit:
    tA, tg = truth["A"][ti[c]], truth["g"][ti[c]]
    rA, rg = rec["A"][ri[c]], rec["g"][ri[c]]
    dg = (tg - rg + 180) % 360 - 180
    flag = "" if abs(rA - tA) / tA < 0.1 and abs(dg) < 10 else "  <-- off"
    print(f"{c:7} {tA:8.3f} {rA:8.3f} {100*(rA-tA)/tA:6.1f}  {tg:7.1f} {rg:7.1f} {dg:6.1f}{flag}")

# --- how recovery depends on record length (aliasing needs long records) ---
print("\nRecord-length sweep (median |amp error| over the 8 constituents):")
print(f"  {'years':>5} {'n_samp':>7} {'M2 err':>8} {'K1 err':>8} {'P1 err':>8} {'S2 err':>8}")
for yr in [2, 5, 12, 25]:
    ts = np.arange(0, yr * 365.25, REPEAT)
    obs_y = utide.reconstruct(ts, truth, epoch="2023-01-01", min_SNR=0, verbose=False).h \
        + 0.03 * rng.standard_normal(ts.size)
    r = utide.solve(ts, obs_y, lat=42.35, constit=constit, method="ols",
                    conf_int="none", epoch="2023-01-01", verbose=False)
    ix = {n: i for i, n in enumerate(r["name"])}
    def e(c):
        return 100 * abs(r["A"][ix[c]] - truth["A"][ti[c]]) / truth["A"][ti[c]]
    print(f"  {yr:5d} {ts.size:7d} {e('M2'):7.1f}% {e('K1'):7.1f}% {e('P1'):7.1f}% {e('S2'):7.1f}%")

# --- the real use case: a satellite TRACK of points, solved at once --------
import time

npts = 4000
scales = rng.uniform(0.4, 1.6, npts)
X = scales[None, :] * true_at_sat[:, None] + 0.03 * rng.standard_normal((t_sat.size, npts))
for g in (False, True):
    utide.solve_many(t_sat, X[:, :8], lat=42.35, constit=constit, gpu=g, epoch="2023-01-01", verbose=False)
    t0 = time.perf_counter()
    om = utide.solve_many(t_sat, X, lat=42.35, constit=constit, gpu=g, epoch="2023-01-01", verbose=False)
    dt = time.perf_counter() - t0
    print(f"  solve_many {npts} track points  gpu={str(g):5s}: {1000*dt:6.0f} ms")
m2i = list(om.name).index("M2")
trueM2 = scales * truth["A"][ti["M2"]]
print(f"  M2 recovery across the track: median |err| = "
      f"{100*np.median(np.abs(om.A[m2i]-trueM2)/trueM2):.1f}%")

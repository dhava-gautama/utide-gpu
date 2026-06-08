"""
Validation against NOAA's official harmonic constants.

For each of the 39 NOAA stations, UTide's harmonic analysis of one year (2023)
of hourly data is compared with NOAA's published, accepted harmonic constants
(derived from ~19 years of record). Amplitudes agree to a few percent and
Greenwich phases to a couple of degrees for the constituents that carry real
signal -- a strong check that the analysis is correct on real data.

Data and harmonic constants: NOAA CO-OPS (public domain).
"""
import json
import os

import numpy as np

import utide

HERE = os.path.dirname(__file__)
d = np.load(os.path.join(HERE, "data", "noaa_hourly_2023.npz"), allow_pickle=True)
harcon = json.load(open(os.path.join(HERE, "data", "noaa_harcon.json")))
t = d["t_days"].astype(float)
X = d["levels"].astype(float)
ids, names, lats = [str(x) for x in d["ids"]], d["names"], d["lats"]

CONSTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "M4"]
uA, nA, uG, nG, ci = [], [], [], [], []
for s in range(X.shape[1]):
    coef = utide.solve(t, X[:, s], lat=float(lats[s]), method="ols",
                       conf_int="none", epoch="2023-01-01", verbose=False)
    i_of = {n: i for i, n in enumerate(coef["name"])}
    hc = harcon[ids[s]]
    for k, c in enumerate(CONSTS):
        if c in i_of and c in hc:
            i = i_of[c]
            uA.append(coef["A"][i]); nA.append(hc[c][0])
            uG.append(coef["g"][i]); nG.append(hc[c][1]); ci.append(k)
uA, nA, uG, nG, ci = map(np.array, (uA, nA, uG, nG, ci))

rel = np.abs(uA - nA) / nA
dphi = np.abs((uG - nG + 180) % 360 - 180)
big = nA > 0.05                       # phase is only meaningful where there is signal
print(f"compared {len(uA)} (station, constituent) pairs across {X.shape[1]} stations")
print(f"  amplitude: median |error| = {100*np.median(rel):.1f}% "
      f"(90th pct {100*np.percentile(rel, 90):.1f}%)")
print(f"  phase (amp>5cm): median |error| = {np.median(dphi[big]):.1f} deg "
      f"(90th pct {np.percentile(dphi[big], 90):.1f} deg)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for k, c in enumerate(CONSTS):
        m = ci == k
        ax[0].scatter(nA[m], uA[m], s=18, color=cmap(k % 10), label=c)
        ax[1].scatter(nG[m][nA[m] > 0.05], uG[m][nA[m] > 0.05], s=18, color=cmap(k % 10))
    lim = [min(nA.min(), uA.min()) * 0.8, max(nA.max(), uA.max()) * 1.2]
    ax[0].plot(lim, lim, "k--", lw=0.8); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("NOAA amplitude (m)"); ax[0].set_ylabel("UTide amplitude (m)")
    ax[0].set_title("Amplitude"); ax[0].legend(ncol=3, fontsize=8)
    ax[1].plot([0, 360], [0, 360], "k--", lw=0.8); ax[1].set_xlim(0, 360); ax[1].set_ylim(0, 360)
    ax[1].set_xlabel("NOAA phase (deg)"); ax[1].set_ylabel("UTide phase (deg)")
    ax[1].set_title("Greenwich phase (amp > 5 cm)")
    fig.suptitle("UTide (1 yr, 2023) vs NOAA official harmonic constants — 39 stations")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "noaa_validation.png"), dpi=95)
    print("\nsaved figure -> examples/noaa_validation.png")
except ImportError:
    print("(matplotlib not installed; skipping plot)")

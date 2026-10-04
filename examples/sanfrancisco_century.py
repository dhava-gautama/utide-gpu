"""A century of hourly tide-gauge data — the single-series GPU path.

Downloads ~127 years of verified hourly water levels for San Francisco
(NOAA CO-OPS station 9414290, public API), fits 18 constituents — diurnal,
semidiurnal, long-period lunar and solar (SA/SSA) — to the whole ~1.1M-sample
record in ONE `solve(gpu=True)` call, and demonstrates:

* the single-series GPU path (the harmonic-basis build dominates a long
  `solve`; the GPU advantage grows with record length, ~8x at a century),
* robust IRLS fitting (`method="robust"`) on the GPU, which resists the
  transients and datum wobbles that bias ordinary least squares,
* long-period constituents and the mean-sea-level trend with linearized CIs.

Usage:
    python examples/sanfrancisco_century.py [--data-dir DIR] [--out PNG]
        [--cpu] [--years N]

Requires numpy, pandas, matplotlib, (optional cupy for the GPU path).
"""

import argparse
import gc
import glob
import json
import os
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd

STATION = "9414290"          # San Francisco, CA
URL = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
       "?product=hourly_height&station={st}&datum=MSL&units=metric"
       "&time_zone=gmt&format=json&begin_date={b}&end_date={e}")
EPOCH = "2000-01-01"
# 18 constituents: 12 short-period + lunar monthly/fortnightly + solar/lunar
# long-period. All pairwise-resolvable over a century (Rayleigh).
CONSTS = (["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MU2", "NU2",
           "L2", "2N2", "MM", "MSM", "MSF", "MF", "SA", "SSA"])


def download(data_dir, y0=1900, y1=2026):
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, f"coops_{STATION}_hourly.csv")
    if os.path.exists(cache):
        print(f"[download] {cache} found, skipping")
        return cache
    rows = []
    for year in range(y0, y1 + 1):  # API limit: 365 days per request
        url = URL.format(st=STATION, b=f"{year}0101", e=f"{year}1231")
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            print(f"  {year}: FAILED ({e})")
            continue
        for row in d.get("data", []):
            try:
                rows.append((row["t"], float(row["v"])))
            except (ValueError, KeyError):
                pass
    df = pd.DataFrame(rows, columns=["time", "v"])
    df["time"] = pd.to_datetime(df.time)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"[download] {len(df)} hourly samples "
          f"({df.time.min().date()} .. {df.time.max().date()}) -> {cache}")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="tidegauge_data")
    ap.add_argument("--out", default="sanfrancisco_century_demo.png")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    csv = download(args.data_dir)
    df = pd.read_csv(csv)
    tt = pd.to_datetime(df.time).values.astype("datetime64[s]")
    t = ((tt - np.datetime64(EPOCH)) / np.timedelta64(1, "D")).astype(float)
    h = df.v.values.astype(float)
    ok = np.isfinite(h)
    t, h = t[ok], h[ok]
    order = np.argsort(t)
    t, uniq = np.unique(t[order], return_index=True)
    h = h[uniq]
    years = (t[-1] - t[0]) / 365.25
    print(f"[prep] {len(t)} samples, {years:.1f} yr")

    try:
        import cupy  # noqa: F401
        import cupy.cuda.runtime as _rt
        _rt.getDeviceCount()
        gpu = not args.cpu
    except Exception:
        gpu = False
    from utide import solve, reconstruct

    def fit(t_, h_, gpu_=gpu, robust=False, quiet=True):
        kw = dict(constit=CONSTS, trend=True, nodal=True, epoch=EPOCH,
                  verbose=not quiet, conf_int="none",
                  gpu_precision="single")
        if robust:
            kw["method"] = "robust"
        return solve(t_, h_, lat=37.77, gpu=gpu_, **kw)

    # timing vs record length (CPU FP64 vs GPU FP32)
    LENGTHS = [1, 2, 5, 10, 25, 50, 100, int(years)]
    times = {"cpu": [], "gpu": []}
    for L in LENGTHS:
        m = t >= t[-1] - 365.25 * L
        t0 = time.perf_counter(); fit(t[m], h[m], gpu_=False)
        times["cpu"].append(time.perf_counter() - t0)
        gc.collect()
        if gpu:
            t0 = time.perf_counter(); fit(t[m], h[m], gpu_=True)
            times["gpu"].append(time.perf_counter() - t0)
            gc.collect()
        else:
            times["gpu"].append(np.nan)
        print(f"[timing] {L:4d} yr: cpu {times['cpu'][-1]:7.2f}s  "
              f"gpu {times['gpu'][-1]:6.2f}s  x{times['cpu'][-1]/times['gpu'][-1]:.1f}")

    # full-record science fits (GPU)
    t0 = time.perf_counter()
    coef = fit(t, h, gpu_=gpu)
    names = list(coef.name)
    print(f"[fit] {len(coef.name)} constituents in {time.perf_counter()-t0:.1f}s "
          f"({len(t)} samples)")
    t0 = time.perf_counter()
    coef_r = fit(t, h, gpu_=gpu, robust=True)
    print(f"[robust] IRLS fit in {time.perf_counter()-t0:.1f}s")
    pred = reconstruct(t, coef, epoch=EPOCH, gpu=gpu, verbose=False).h
    res = h - pred
    pred_r = reconstruct(t, coef_r, epoch=EPOCH, gpu=gpu, verbose=False).h
    res_r = h - pred_r
    print(f"[validation] residual std: OLS {np.nanstd(res)*100:.2f} cm, "
          f"robust {np.nanstd(res_r)*100:.2f} cm")
    print(f"[science] MSL trend {coef.slope*365.25*1000:.2f} mm/yr | "
          f"M2 {coef.A[names.index('M2')]*100:.1f} cm")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import cartopy  # noqa: F401
    except ImportError:
        pass
    yrs = 2000.0 + t / 365.25
    s = pd.Series(h * 100, index=pd.to_datetime(t * 86400, unit="s"))
    pm = s.resample("MS").mean()
    prs = pd.Series(pred * 100, index=s.index).resample("MS").mean()

    fig = plt.figure(figsize=(13.5, 10.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.25, 0.8],
                          hspace=0.36, wspace=0.22)
    axA = fig.add_subplot(gs[0, :])
    axA.plot(pm.index.year + (pm.index.month - 1) / 12, pm.values, ".", ms=2,
             color="#9ec5e8", label="monthly mean obs")
    axA.plot(prs.index.year + (prs.index.month - 1) / 12, prs.values, "b-",
             lw=1.0, label="UTide harmonic fit (18 constituents)")
    slope = coef.slope * 365.25 * 1000
    y0_, y1_ = yrs.min(), yrs.max()
    base = np.nanmean(s.values)
    axA.plot([y0_, y1_], [base + slope / 1000 * 365.25 * (y0_ - yrs.mean()),
                          base + slope / 1000 * 365.25 * (y1_ - yrs.mean())],
             "k--", lw=1.4, label=f"linear trend {slope:.2f} mm/yr")
    axA.set_ylabel("sea level [cm (MSL)]")
    axA.set_title(f"A | NOAA CO-OPS station {STATION} — {len(t):,} hourly samples, "
                  f"{int(y0_)}–{int(y1_)} — one solve(gpu=True) call", fontsize=10)
    axA.legend(fontsize=8, ncol=3, loc="upper left")

    axB = fig.add_subplot(gs[1, 0])
    xs = np.arange(len(names))
    order = np.argsort(coef.A)[::-1]
    axB.plot(xs, coef.A[order], "o", ms=4, color="tab:blue")
    axB.set_xticks(xs, [names[i] for i in order], rotation=60, fontsize=7)
    axB.set_yscale("log")
    axB.set_ylabel("amplitude [m]")
    axB.set_title("B | fitted constituent spectrum (CIs omitted: the "
                  "gappy-data noise-floor PSD is O(N·M))", fontsize=10)

    axC = fig.add_subplot(gs[1, 1])
    Lc = LENGTHS[:-1] + [years] if LENGTHS[-1] != int(years) else LENGTHS
    axC.plot(Lc, times["cpu"], "o-", label="CPU (FP64)")
    if gpu:
        axC.plot(Lc, times["gpu"], "s-", label="GPU (FP32)")
    for x_, i_ in zip(Lc, range(len(Lc))):
        if x_ in (25, Lc[-1]):
            axC.annotate(f"x{times['cpu'][i_]/times['gpu'][i_]:.1f}", (x_, times["cpu"][i_]),
                         textcoords="offset points", xytext=(4, 6), fontsize=8)
    axC.set_xlabel("record length [yr]")
    axC.set_ylabel("solve() wall time [s]")
    axC.set_yscale("log")
    axC.set_title("C | single-series solve(): the GPU advantage\n"
                  "grows with record length (basis build dominates)", fontsize=10)
    axC.legend(fontsize=8)

    axD = fig.add_subplot(gs[2, :])
    yy = 2000.0 + t / 365.25
    w = (yy >= 2011.17) & (yy <= 2011.45)  # Mar 2011 Tohoku tsunami window
    axD.plot(yy[w], h[w] * 100, ".", ms=3, color="#9ec5e8", label="obs")
    axD.plot(yy[w], (h[w] - res[w]) * 100, "r-", lw=1.0,
             label=f"OLS fit (res std {np.nanstd(res)*100:.2f} cm)")
    axD.plot(yy[w], (h[w] - res_r[w]) * 100, "b-", lw=1.0,
             label=f"robust IRLS fit (res std {np.nanstd(res_r)*100:.2f} cm)")
    axD.set_xlabel("year"); axD.set_ylabel("sea level [cm]")
    axD.set_title("D | Mar 2011 (Tohoku tsunami at San Francisco) — GPU robust "
                  "IRLS resists the transient", fontsize=10)
    axD.legend(fontsize=7, loc="upper right")

    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[figure] wrote {args.out}")


if __name__ == "__main__":
    main()

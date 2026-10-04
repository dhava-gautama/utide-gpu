"""Hold-out tide prediction: fit the past, predict the future.

Trains UTide on 1900-2015 of the San Francisco hourly record (NOAA CO-OPS
9414290, public API) and predicts the fully withheld 2016-2026 decade,
reporting R-squared and RMSE on ~88,560 never-seen samples. This is the
prediction half of UTide's purpose: `solve` analyses, `reconstruct` predicts.

Usage:
    python examples/sf_prediction_skill.py [--data-dir DIR] [--out PNG]
        [--train-end 2016] [--cpu]
"""

import argparse
import json
import os
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd

STATION = "9414290"
URL = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
       "?product=hourly_height&station={st}&datum=MSL&units=metric"
       "&time_zone=gmt&format=json&begin_date={b}&end_date={e}")
EPOCH = "2000-01-01"
CONSTS = (["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MU2", "NU2",
           "L2", "2N2", "MM", "MSM", "MSF", "MF", "SA", "SSA"])


def download(data_dir, y0=1900, y1=2026):
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, f"coops_{STATION}_hourly.csv")
    if os.path.exists(cache):
        return cache
    rows = []
    for year in range(y0, y1 + 1):  # API limit: 365 days per request
        try:
            with urllib.request.urlopen(
                    URL.format(st=STATION, b=f"{year}0101", e=f"{year}1231"),
                    timeout=60) as r:
                d = json.loads(r.read().decode())
            for row in d.get("data", []):
                try:
                    rows.append((row["t"], float(row["v"])))
                except (ValueError, KeyError):
                    pass
        except Exception:  # noqa: BLE001
            pass
    df = pd.DataFrame(rows, columns=["time", "v"])
    df["time"] = pd.to_datetime(df.time)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"[download] {len(df)} hourly samples -> {cache}")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="tidegauge_data")
    ap.add_argument("--out", default="sf_prediction_skill_demo.png")
    ap.add_argument("--train-end", type=int, default=2016,
                    help="first full year held out for testing")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    df = pd.read_csv(download(args.data_dir))
    tt = pd.to_datetime(df.time).values.astype("datetime64[s]")
    t = ((tt - np.datetime64(EPOCH)) / np.timedelta64(1, "D")).astype(float)
    h = df.v.values.astype(float)
    ok = np.isfinite(h)
    t, h = t[ok], h[ok]
    order = np.argsort(t)
    t, uniq = np.unique(t[order], return_index=True)
    h = h[uniq]

    split = np.datetime64(f"{args.train_end}-01-01", "s")
    td = (split - np.datetime64(EPOCH)) / np.timedelta64(1, "D")
    mtr = t < td

    try:
        import cupy  # noqa: F401
        import cupy.cuda.runtime as _rt
        _rt.getDeviceCount()
        gpu = not args.cpu
    except Exception:
        gpu = False
    from utide import solve, reconstruct

    kw = dict(constit=CONSTS, trend=True, nodal=True, epoch=EPOCH, lat=37.77,
              verbose=False, conf_int="none", gpu_precision="single")
    t0 = time.perf_counter()
    coef = solve(t[mtr], h[mtr], gpu=gpu, **kw)
    print(f"[fit] train {float(mtr.sum()):,} samples "
          f"({(t[mtr][-1]-t[mtr][0])/365.25:.1f} yr) in "
          f"{time.perf_counter()-t0:.2f}s ({'gpu' if gpu else 'cpu'})")
    t0 = time.perf_counter()
    pred = reconstruct(t[~mtr], coef, epoch=EPOCH, gpu=gpu, verbose=False).h
    print(f"[predict] {int((~mtr).sum()):,} held-out samples in "
          f"{time.perf_counter()-t0:.2f}s")

    obs = h[~mtr]
    r2 = 1 - np.var(obs - pred) / np.var(obs)
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)) * 100)
    print(f"[skill] R2={r2:.4f}  RMSE={rmse:.1f} cm  (test: "
          f"{(t[~mtr][0]):.1f}..{t[~mtr][-1]:.1f} days since {EPOCH})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yrs = 2000.0 + t / 365.25
    fig = plt.figure(figsize=(13.0, 5.2))
    axA = fig.add_subplot(1, 2, 1)
    axA.plot(yrs[~mtr], pred * 100, "b-", lw=0.4,
             label="UTide prediction (fit 1900-2015)")
    axA.plot(yrs[~mtr], obs * 100, ".", ms=0.8, color="#9ec5e8",
             label="obs (held out)")
    axA.set_ylabel("sea level [cm (MSL)]")
    axA.set_title(f"A | {yrs[~mtr].min():.1f}-{yrs[~mtr].max():.1f} hold-out "
                  f"prediction - R2={r2:.4f}, RMSE={rmse:.1f} cm", fontsize=10)
    axA.legend(fontsize=8, loc="upper left")
    axB = fig.add_subplot(1, 2, 2)
    yte = yrs[~mtr]
    w = (yte >= 2021.0) & (yte <= 2021.25)
    axB.plot(yte[w], pred[w] * 100, "b-", lw=1.1, label="prediction")
    axB.plot(yte[w], obs[w] * 100, "k.", ms=3, label="obs")
    axB.set_xlabel("year"); axB.set_ylabel("sea level [cm]")
    axB.set_title("B | 90-day zoom", fontsize=10)
    axB.legend(fontsize=8)
    fig.suptitle(f"Hold-out tide prediction - {STATION} San Francisco: "
                 f"train 1900-{args.train_end-1}, predict {args.train_end}-2026",
                 fontsize=13)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[figure] wrote {args.out}")


if __name__ == "__main__":
    main()

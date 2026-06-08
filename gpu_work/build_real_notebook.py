"""Generate notebooks/real_station_example.ipynb (valid nbformat 4)."""

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
        "# utide-gpu on a real tide-gauge record",
        "",
        "This notebook runs the whole workflow on the hourly sea-level record shipped",
        "with the package (`can1998.dtf` — one year of observations, 1998): harmonic",
        "analysis on the GPU, prediction versus the observations, and the empirical",
        "tidal datums.",
        "",
        "*(This is a single station, so the GPU is not faster here than the CPU — set",
        "`gpu=False` to compare. The GPU win shows up when analysing a whole field of",
        "series at once; see `gpu_batch_example.ipynb`.)*",
    ),
    code(
        "%matplotlib inline\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import utide",
    ),
    md("## 1. Load and clean the record"),
    code(
        'raw = np.loadtxt("can1998.dtf")            # seconds year month day hour elev flag\n'
        "t = raw[:, 0] / 86400.0                     # days since 1998-01-01\n"
        "h = raw[:, 5].copy()\n"
        "h[raw[:, 6] == 2] = np.nan                  # flag 2 == bad\n"
        "h[np.abs(h - 9.990) < 1e-6] = np.nan        # missing sentinel\n"
        'print(f"{np.isfinite(h).sum()} hourly samples, {np.isnan(h).sum()} gaps")',
    ),
    md(
        "## 2. Harmonic analysis on the GPU",
        "",
        "`gpu=True` runs the basis construction and solve on the GPU; results come back",
        "on the host and match the CPU result to round-off.",
    ),
    code(
        'coef = utide.solve(t, h, lat=-25, method="ols", conf_int="linear",\n'
        '                   epoch="1998-01-01", gpu=True, verbose=False)\n'
        'order = np.argsort(coef["A"])[::-1]\n'
        "print(f\"{len(coef['name'])} constituents\\n\")\n"
        'print("constituent   amp (m)   phase (deg)")\n'
        "for i in order[:8]:\n"
        "    print(f\"   {coef['name'][i]:5s}    {coef['A'][i]:7.3f}    {coef['g'][i]:7.1f}\")",
    ),
    md(
        "## 3. Prediction vs observations",
        "",
        "`reconstruct` predicts the tide from the fitted constituents.",
    ),
    code(
        'tide = utide.reconstruct(t, coef, epoch="1998-01-01", verbose=False)\n'
        "resid = h - tide.h\n"
        "rms = np.sqrt(np.nanmean(resid**2))\n"
        'print(f"RMS(observed - predicted) = {rms:.3f} m,  "\n'
        '      f"{100*(1-np.nanvar(resid)/np.nanvar(h)):.1f}% of variance explained")\n'
        "\n"
        "w = slice(0, 24*14)\n"
        "plt.figure(figsize=(11, 3.5))\n"
        'plt.plot(t[w], h[w], ".", ms=3, label="observed")\n'
        'plt.plot(t[w], tide.h[w], "-", lw=1, label="predicted (utide)")\n'
        'plt.xlabel("days since 1998-01-01"); plt.ylabel("sea level (m)")\n'
        'plt.title("Prediction vs observations (first 2 weeks)"); plt.legend(); plt.show()',
    ),
    md(
        "## 4. Empirical tidal datums",
        "",
        "`tidal_characteristics` extracts standard datums directly from the record by",
        "detecting its high and low waters.",
    ),
    code(
        "c = utide.tidal_characteristics(t, h)\n"
        'for k in ["MHW", "MLW", "MTL", "MTR", "ED", "FD"]:\n'
        "    print(f\"   {k:4s} = {c[k]:6.3f} {'h' if k in ('ED','FD') else 'm'}\")\n"
        "\n"
        "w = slice(0, 24*14)\n"
        "plt.figure(figsize=(11, 3.5))\n"
        'plt.plot(t[w], h[w], "-", lw=0.8, color="0.5")\n'
        "m = (c.hw_t >= t[w][0]) & (c.hw_t <= t[w][-1])\n"
        "n = (c.lw_t >= t[w][0]) & (c.lw_t <= t[w][-1])\n"
        'plt.plot(c.hw_t[m], c.hw_h[m], "r^", label="high water")\n'
        'plt.plot(c.lw_t[n], c.lw_h[n], "bv", label="low water")\n'
        'plt.axhline(c.MHW, color="r", ls="--", lw=0.8); plt.axhline(c.MLW, color="b", ls="--", lw=0.8)\n'
        'plt.xlabel("days since 1998-01-01"); plt.ylabel("sea level (m)")\n'
        'plt.title(f"Datums: MHW={c.MHW:.2f} m, MLW={c.MLW:.2f} m, MTR={c.MTR:.2f} m")\n'
        "plt.legend(); plt.show()",
    ),
    md(
        "To analyse a whole **field** of stations like this at once — extracting",
        "constituents and datums everywhere in one call — pass an `(n_times, n_series)`",
        "array to `utide.solve_many` and `utide.tidal_characteristics_many`.",
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
with open("notebooks/real_station_example.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/real_station_example.ipynb with", len(cells), "cells")

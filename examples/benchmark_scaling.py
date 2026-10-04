"""Performance scaling: why the GPU path exists, in two panels.

Left: single-series `solve(gpu=True)` wall time vs record length — the
harmonic-basis build dominates and grows with record length, so the GPU
advantage grows from ~1x at 1 year to ~10-30x at a century.

Right: batched `solve_many` wall time vs number of series (1 year of hourly
data each) — the naive per-series loop, CPU solve_many, GPU FP64 and GPU
FP32. Beyond GPU memory, series stream through in chunks (`chunk_size`).

Everything is synthetic (a fixed multi-constituent signal with random
amplitudes + noise), so the figure is reproducible anywhere.

Usage:
    python examples/benchmark_scaling.py [--out PNG] [--cpu] [--s-max 10000]
"""

import argparse
import gc
import time
import warnings

import numpy as np

CONSTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
EPOCH = "2000-01-01"


def synth(nt, S, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(nt) / 24.0
    fr = [1/12.42, 1/12.0, 1/12.66, 1/11.97, 1/23.93, 1/25.82, 1/163.6, 1/438.3]
    base = sum(np.cos(2*np.pi*f*24*t + i*0.7) for i, f in enumerate(fr))
    X = (rng.uniform(0.5, 1.5, S)[None, :] * base[:, None]
         + 0.05 * rng.standard_normal((nt, S)))
    return t, X


def cleanup():
    gc.collect()
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark_scaling.png")
    ap.add_argument("--cpu", action="store_true", help="skip all GPU timings")
    ap.add_argument("--s-max", type=int, default=10000)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    try:
        import cupy  # noqa: F401
        import cupy.cuda.runtime as _rt
        _rt.getDeviceCount()
        gpu = not args.cpu
    except Exception:
        gpu = False
    from utide import solve, solve_many

    def fit1(t, h, gpu_):
        return solve(t, h, lat=37.8, constit=CONSTS, trend=True, nodal=True,
                     epoch=EPOCH, conf_int="none", gpu=gpu_,
                     gpu_precision="single", verbose=False)

    # ---- A. solve() vs record length ----
    rec_len, rec_cpu, rec_gpu = [], [], []
    for L in [1, 2, 5, 10, 25, 50, 100, 126]:
        t, X = synth(int(8766 * L), 1)
        t0 = time.perf_counter(); fit1(t, X[:, 0], False)
        rec_cpu.append(time.perf_counter() - t0)
        cleanup()
        if gpu:
            t0 = time.perf_counter(); fit1(t, X[:, 0], True)
            rec_gpu.append(time.perf_counter() - t0)
            cleanup()
        rec_len.append(L)
        print(f"[solve] {L:4d} yr: cpu {rec_cpu[-1]:7.2f}s  "
              f"gpu {rec_gpu[-1] if rec_gpu else float('nan'):6.2f}s", flush=True)
        del X

    # ---- B. solve_many vs number of series ----
    S_LIST = [10, 100, 500, 1000, 2000, 5000]
    if args.s_max > 5000:
        S_LIST += [s for s in (10000, 20000) if s <= args.s_max]
    sm = {"S": [], "cpu": [], "gpu64": [], "gpu32": [], "loop": []}
    for S in S_LIST:
        t, X = synth(8766, S)
        t0 = time.perf_counter()
        solve_many(t, X, gpu=False, gappy="ne", constit=CONSTS, trend=True,
                   nodal=True, epoch=EPOCH, lat=37.8, verbose=False)
        sm["cpu"].append(time.perf_counter() - t0)
        cleanup()
        if gpu:
            t0 = time.perf_counter()
            solve_many(t, X, gpu=True, gappy="ne", constit=CONSTS, trend=True,
                       nodal=True, epoch=EPOCH, lat=37.8, verbose=False)
            sm["gpu64"].append(time.perf_counter() - t0)
            cleanup()
            t0 = time.perf_counter()
            solve_many(t, X, gpu=True, gappy="ne", gpu_precision="single",
                       constit=CONSTS, trend=True, nodal=True, epoch=EPOCH,
                       lat=37.8, verbose=False)
            sm["gpu32"].append(time.perf_counter() - t0)
            cleanup()
        else:
            sm["gpu64"].append(np.nan)
            sm["gpu32"].append(np.nan)
        if S <= 100:  # naive per-series loop only where it is not absurd
            t0 = time.perf_counter()
            for k in range(S):
                fit1(t, X[:, k], False)
            sm["loop"].append(time.perf_counter() - t0)
            cleanup()
        else:
            sm["loop"].append(np.nan)
        sm["S"].append(S)
        print(f"[solve_many] S={S:5d}: cpu {sm['cpu'][-1]:6.2f}s  "
              f"gpu64 {sm['gpu64'][-1]:6.2f}s  gpu32 {sm['gpu32'][-1]:6.2f}s  "
              f"loop {sm['loop'][-1]:7.2f}s", flush=True)
        del X
        cleanup()

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.0, 5.2))
    axA = fig.add_subplot(1, 2, 1)
    axA.plot(rec_len, rec_cpu, "o-", label="CPU (FP64)")
    if gpu:
        axA.plot(rec_len, rec_gpu, "s-", label="GPU (FP32)")
        for x_, yc, yg in zip(rec_len, rec_cpu, rec_gpu):
            if x_ in (25, 100, 126):
                axA.annotate(f"x{yc/yg:.0f}", (x_, yc), textcoords="offset points",
                             xytext=(4, 5), fontsize=8)
    axC = axA
    axA.set_xlabel("record length [yr] (hourly)")
    axA.set_ylabel("solve() wall time [s]")
    axA.set_yscale("log")
    axA.set_title("A | single series: solve() time vs record length\n"
                  "basis build dominates and grows with length", fontsize=10)
    axA.legend(fontsize=8)

    axB = fig.add_subplot(1, 2, 2)
    axB.plot(sm["S"], sm["loop"], "^--", ms=4, label="per-series loop (CPU)")
    axB.plot(sm["S"], sm["cpu"], "o-", label="solve_many CPU")
    if gpu:
        axB.plot(sm["S"], sm["gpu64"], "s-", label="solve_many GPU FP64")
        axB.plot(sm["S"], sm["gpu32"], "D-", label="solve_many GPU FP32")
    axB.set_xlabel("number of series (1 yr of hourly data each)")
    axB.set_ylabel("solve_many wall time [s]")
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_title("B | batched field: solve_many vs series count\n"
                  "series stream through the GPU in chunks", fontsize=10)
    axB.legend(fontsize=8)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[figure] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

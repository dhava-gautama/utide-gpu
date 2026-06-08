# UTide

[![gha](https://github.com/wesleybowman/UTide/actions/workflows/tests.yml/badge.svg)](https://github.com/wesleybowman/UTide/actions)
[![license](https://anaconda.org/conda-forge/utide/badges/license.svg)](https://choosealicense.com/licenses/mit/)
[![downloads](https://anaconda.org/conda-forge/utide/badges/downloads.svg)](https://anaconda.org/conda-forge/utide)
[![anaconda_cloud](https://anaconda.org/conda-forge/utide/badges/version.svg)](https://anaconda.org/conda-forge/utide)

Python re-implementation of the Matlab package UTide.

Still in heavy development\--everything is subject to change!

> **utide-gpu fork:** this fork adds an optional GPU (CuPy) backend and a
> batched `solve_many` solver on top of upstream UTide; see
> [GPU acceleration](#gpu-acceleration) below. The CPU behaviour is unchanged.

Note: the user interface differs from the Matlab version, so consult the
Python function docstrings to see how to specify parameters. Some
functionality from the Matlab version is not yet available. For more
information see:

    Codiga, D.L., 2011. Unified Tidal Analysis and Prediction Using the
    UTide Matlab Functions. Technical Report 2011-01. Graduate School
    of Oceanography, University of Rhode Island, Narragansett, RI.
    59pp.
    ftp://www.po.gso.uri.edu/pub/downloads/codiga/pubs/2011Codiga-UTide-Report.pdf

    UTide v1p0 9/2011 d.codiga@gso.uri.edu
    http://www.po.gso.uri.edu/~codiga/utide/utide.htm

# Installation

This fork provides the full upstream UTide API **plus** the optional GPU
backend and `solve_many`, and installs as the `utide` package. Install it from
source:

``` shell
pip install git+https://github.com/dhava-gautama/utide-gpu.git
```

For the GPU features, also install CuPy for your CUDA version. This is
optional\--without it everything runs on the CPU exactly as upstream:

``` shell
pip install cupy-cuda12x      # or cupy-cuda11x, etc., to match your CUDA
```

The upstream, CPU-only package is on PyPI and conda-forge if you do not need
the GPU additions:

``` shell
pip install utide
# or
conda install utide --channel conda-forge
```

The public functions can be imported using

```python
from utide import solve, solve_many, reconstruct
```

A sample call would be

```python
from utide import solve

coef = solve(
    time,
    time_series_u,
    time_series_v,
    lat=30,
    nodal=False,
    trend=False,
    method="ols",
    conf_int="linear",
    Rayleigh_min=0.95,
)
```

For more examples see the
[notebooks](https://nbviewer.jupyter.org/github/wesleybowman/UTide/tree/master/notebooks/)
folder.

# GPU acceleration

This fork adds an **optional GPU backend** (via [CuPy](https://cupy.dev)) and a
**batched** solver, on top of the standard UTide API. The GPU is strictly
opt-in; with `gpu=False` the CPU path is byte-identical to upstream.

```python
from utide import solve, solve_many

# Single series on the GPU (results returned on the host, identical to CPU):
coef = solve(t, h, lat=45, method="ols", conf_int="linear", gpu=True)

# Many series sharing one time base, fit in a single batched solve:
out = solve_many(t, X, lat=45, gpu=True)          # X is (ntimes, nseries)

# Optional single precision for a large extra speedup on consumer GPUs:
out = solve_many(t, X, lat=45, gpu=True, gpu_precision="single")
```

Highlights:

- `solve(..., gpu=True)` accelerates harmonic-basis construction (the dominant
  cost) and the least-squares solve, with automatic CPU fallback for the option
  combinations not yet supported on the GPU. Robust fitting
  (`method="robust"`) also runs on the device.
- `solve_many` fits many series with a shared time base in one solve\--far
  faster than looping `solve`\--and handles per-series gaps and streams large
  batches within available GPU memory.
- `gpu_precision="single"` runs the basis and solve in float32 for a large
  speedup where the GPU's double-precision throughput is limited, at reduced
  numerical precision (intended for screening, not final-precision work).

Requires CuPy with a working CUDA device, e.g. `pip install cupy-cuda12x`. If
CuPy is not installed, importing and using UTide on the CPU is unaffected.

# Use cases

The GPU backend and `solve_many` pay off most when you have **many tidal time
series that share one time base** — an ocean-model SSH grid, satellite
altimetry, or an array of tide gauges / moorings. `solve_many` builds the
harmonic model once and solves every series in a single batched call.

![M2 co-tidal maps recovered over a grid with one solve_many call](examples/m2_grid.png)

*M2 amplitude and phase recovered for every cell of a 64×64 grid (1 year of
hourly data per cell) with a single `solve_many` call — about 240× faster than
looping `solve`, and matching the per-cell result to round-off. See
[`notebooks/gpu_batch_example.ipynb`](notebooks/gpu_batch_example.ipynb).*

**Where it shines**

- **A field of series (the big one).** `solve_many(t, X)` with `X` shaped
  `(n_times, n_series)` returns amplitudes/phases for every series at once —
  ~100×+ faster than looping `solve`, with per-series gap handling and
  streaming of batches larger than GPU memory.
- **Long, high-rate records.** `solve(t, h, gpu=True)` accelerates the
  harmonic-basis construction, which dominates the cost of a single long fit.
- **First-pass screening of huge datasets.** `gpu_precision="single"` trades a
  few digits of precision for a large extra speed-up.

**When the CPU is fine**

- A single, short record (≲ a year): the GPU's setup cost is not worth it; plain
  `solve(...)` is the right tool.

A runnable version of the figure above is in
[`examples/gpu_batch_grid.py`](examples/gpu_batch_grid.py).

# Tidal datums

Alongside harmonic analysis, UTide can compute standard **empirical tidal
datums** directly from a water-level series — mean high/low water (MHW/MLW),
mean tide level (MTL), mean tidal range (MTR), and mean ebb/flood durations
(ED/FD):

```python
from utide import tidal_characteristics, tidal_characteristics_many

c = tidal_characteristics(t, h)          # one series -> MHW, MLW, MTL, MTR, ED, FD
maps = tidal_characteristics_many(t, X)  # a field (n_times, n_series) -> arrays
```

This pairs naturally with `solve_many`: constituent maps *and* datum maps over
the same grid. The datum set follows DHI's
[tide_analytics](https://github.com/DHI/tide_analytics); this is an independent
NumPy/SciPy implementation.

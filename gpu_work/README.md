# UTide GPU backend — internal notes

Status: **working & validated, kept internal** (not published). 2026-06-07.

## What was added to the package
- `utide/_backend.py` — lazy, optional CuPy selection (`get_xp`, `asnumpy`). Importing
  utide never imports cupy unless `gpu=True` is actually requested.
- `utide/_harmonics_xp.py` — backend-agnostic (`numpy`/`cupy`) `ut_E` basis. CPU path
  reproduces `utide.harmonics.ut_E` to ~1e-15; GPU path validated to ~1e-10.
- `utide/_solve.py` — `solve(..., gpu=True)` (GPU basis + OLS lstsq, host fallback for
  inference / linearized-time options) and new batched `solve_many()`.
- `tests/test_gpu.py` — auto-skips when CuPy/CUDA absent; CPU suite stays green.

## Usage
```python
import utide

# Single series on GPU (results returned on host, identical to CPU to ~1e-11):
coef = utide.solve(t, h, lat=45, method="ols", conf_int="linear", gpu=True)

# Many series sharing one time base (the big win): X is (nt, S)
out = utide.solve_many(t, X, lat=45, gpu=True)  # out.A, out.g are (nc, S)
```

## Measured on RTX 4060 (see bench_gpu.py, validate_*.py)
- Basis construction (`ut_E`) is ~75-80% of `solve` time; GPU 1.3×(1yr)→8.7×(100yr).
- `solve(gpu=True)`: ~3.3× end-to-end at 10yr hourly.
- `solve_many`: 498× (1000 stations) / 691× (5000) vs naive per-series loop.
- Consumer-GPU FP64 is heavily throttled (the complex matmul in the basis is ~40× slower
  in FP64 than FP32 on the 4060). `gpu_precision='single'` now runs BOTH the basis (mixed:
  FP64 astronomical reduction, FP32 matmul/transcendentals) and the solve in float32:
  - basis build alone: ~4-4.6× faster than FP64.
  - `solve_many`: ~2.4× at S=1000, ~9.8× at S=5000 (the FP32 basis removed the bottleneck).
  - precision cost: single-series ~1e-4 rel amp / ~0.005° phase; batched ~1.5e-3 (S=1000)
    to ~4.5e-3 (S=5000). Screening, not final-precision work.

## Run the benchmarks/validation (on the WSL GPU box)
```bash
rsync -az --exclude='.git' --exclude='__pycache__' ./ dhava@100.118.127.87:~/utide-gpu/UTide/
ssh dhava@100.118.127.87 'cd ~/utide-gpu && source venv/bin/activate && \
  python -u gpu_work/validate_real.py && python -u gpu_work/validate_many.py'
# env: uv venv at ~/utide-gpu/venv with numpy/scipy/cupy-cuda12x[ctk]/pytest/pandas
```

## Done (hardening, 2026-06-08)
- FP32 opt-in: `gpu_precision='single'` runs the mixed-precision basis + FP32 solve on
  `solve`/`solve_many`. Big batch win (up to ~10×); see measured numbers above.
- GPU robust IRLS: `robustfit` is now backend-agnostic (dispatches on the input array
  module), so `solve(method='robust', gpu=True)` runs the whole reweighting loop on the
  device. Correct (matches CPU to 6.6e-13 double / 2.5e-7 single). Speed: break-even at
  1yr, ~2.8× at 10yr, ~3.9× at 30yr (single) -- it pays off for long records only.
  (Also fixed `andrews`/`huber`/`logistic` weight functions, which used Python `max()` on
  an array and were latently broken; and a `x.size`->`X.size` typo in the 1-D reshape.)
- Fixed `robustfit` crash on rank-deficient design matrices. (Root cause was *not* rcond=1
  as first thought — it's that `np.linalg.lstsq` omits the residual sum when rank < ncols,
  which happens with collinear/unresolvable constituents; `rsumsq[0]` then raised IndexError.
  Now the residual sum is computed directly in that case. `tests/test_robustfit.py`.)

- Gappy/masked batch input in `solve_many`: series are grouped by valid-sample
  pattern and each group solved on its own rows (subselected from one shared basis
  build); all-NaN / under-determined series return NaN. Exact (matches per-series
  `solve` to ~1e-14). No-gap or shared-gap data = 1 group (fast, 1000 series 0.6s);
  worst case of 1000 fully-distinct gap patterns ~11.5s (still ~10x vs naive loop).

- VRAM chunking in `solve_many` (`chunk_size`, auto from free VRAM): the model
  matrix stays resident while series stream through, so very large S doesn't OOM
  (S=20000 fine; chunked result identical to unchunked, ~1e-15). Chunks over
  *series*, not time -- a single huge-nt series whose basis won't fit is still
  unhandled (see TODO).

## TODO before any publish (hardening)
- Time-axis chunking for a single series with huge nt (basis intermediates won't
  fit); would need normal-equations accumulation, FP64 only for stability.
- Confidence intervals + inference on the GPU path -- low value (CIs are ~1% of
  runtime and the per-constituent loop is scalar-heavy; better left on host).
- solve_many many-distinct-groups case is loop-bound (small lstsq launches);
  could batch equal-sized groups if it matters.

This `gpu_work/` dir is scratch (benchmarks/validation), not part of the package.

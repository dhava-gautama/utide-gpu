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
out = utide.solve_many(t, X, lat=45, gpu=True)   # out.A, out.g are (nc, S)
```

## Measured on RTX 4060 (see bench_gpu.py, validate_*.py)
- Basis construction (`ut_E`) is ~75-80% of `solve` time; GPU 1.3×(1yr)→8.7×(100yr).
- `solve(gpu=True)`: ~3.3× end-to-end at 10yr hourly.
- `solve_many`: 498× (1000 stations) / 691× (5000) vs naive per-series loop.
- Consumer-GPU FP64 is weak: complex128 lstsq alone is ~break-even, FP32 lstsq is 7-10×.
  But end-to-end `gpu_precision='single'` only gives ~2.2× at S=1000 and ~1.1× at S=5000,
  because the (FP64) basis build + host transfers dominate, not the solve. Precision cost:
  single-series ~9e-5 rel on amplitudes; batched ~1e-3 (S=1000) to ~7e-3 (S=5000). Use it
  for quick screening, not when you need full precision.

## Run the benchmarks/validation (on the WSL GPU box)
```bash
rsync -az --exclude='.git' --exclude='__pycache__' ./ dhava@100.118.127.87:~/utide-gpu/UTide/
ssh dhava@100.118.127.87 'cd ~/utide-gpu && source venv/bin/activate && \
  python -u gpu_work/validate_real.py && python -u gpu_work/validate_many.py'
# env: uv venv at ~/utide-gpu/venv with numpy/scipy/cupy-cuda12x[ctk]/pytest/pandas
```

## Done (hardening, 2026-06-08)
- FP32 opt-in: `gpu_precision='single'` on `solve`/`solve_many` (modest benefit, see above).
- Fixed `robustfit` crash on rank-deficient design matrices. (Root cause was *not* rcond=1
  as first thought — it's that `np.linalg.lstsq` omits the residual sum when rank < ncols,
  which happens with collinear/unresolvable constituents; `rsumsq[0]` then raised IndexError.
  Now the residual sum is computed directly in that case. `tests/test_robustfit.py`.)

## TODO before any publish (hardening)
- Build the basis itself in FP32 (it dominates; biggest remaining lever, needs FP32 tables).
- GPU robust method (IRLS currently runs on host).
- Gappy/masked batch input in `solve_many` (currently drops shared-NaN rows).
- Larger-than-VRAM chunking for huge nt or S.
- Confidence intervals + inference on the GPU path (currently host).

This `gpu_work/` dir is scratch (benchmarks/validation), not part of the package.

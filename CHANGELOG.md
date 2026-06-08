# Changelog

This is the changelog for the `utide-gpu` fork. See the upstream project,
[wesleybowman/UTide](https://github.com/wesleybowman/UTide), for the history of
the base package.

## v0.4.1

- Empirical tidal datums: `tidal_characteristics` / `tidal_characteristics_many`
  (MHW, MLW, MTL, MTR, ED, FD) and `tidal_form_factor` (diurnal/semidiurnal
  classification). Feature set inspired by
  [DHI/tide_analytics](https://github.com/DHI/tide_analytics).
- GPU prediction: `reconstruct(..., gpu=True)` and batched `reconstruct_many`
  (predict a whole `solve_many` field in one call).
- `solve_many`: per-station latitude support (a latitude array is grouped into
  bands); opt-in normal-equations solver (`solver="normal"`).
- Validation against NOAA's official harmonic constants across 39 stations:
  amplitudes agree to ~2.2% and Greenwich phases to ~0.6° (median).
- Real-data examples and notebooks: a single tide-gauge station, a 39-station
  NOAA batch (co-amplitude and tidal-type maps), and a synthetic grid.
- Packaging: distribution renamed to `utide-gpu` (still imports as `utide`);
  clean Tests + pre-commit CI.

## v0.4.0

- Optional GPU (CuPy) backend: `solve(..., gpu=True)` accelerates harmonic-basis
  construction and the OLS/robust solve, with automatic CPU fallback.
- `solve_many`: batched fit for many series sharing one time base, with
  per-series gap handling and streaming of batches larger than GPU memory.
- `gpu_precision="single"`: mixed-precision (float32) path for a large speedup
  on consumer GPUs.
- CPU behaviour unchanged from upstream UTide.

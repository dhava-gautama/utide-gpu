# Changelog

This is the changelog for the `utide-gpu` fork. See the upstream project,
[wesleybowman/UTide](https://github.com/wesleybowman/UTide), for the history of
the base package.

## v0.5.0

- `solve_many(gappy=...)`: masked batched normal-equations path for series
  with distinct NaN gap patterns (no grouping by pattern; Jacobi
  preconditioning, residual and coefficient-magnitude gates, truncated-SVD
  retry, VRAM-bounded series chunking). `gappy="auto"` enables it when a
  latitude band has more than 8 distinct gap patterns.
- Basis LRU cache (512 MB): repeated `solve_many` calls on the same time base
  skip the harmonic-basis rebuild (~75-80% of solve time).
- `reconstruct_many`: 2-D (u, v) support — predict a whole current field
  from the ellipse-parameter batch result.
- GPU scaling validated: single-series `solve` ~46x CPU at a 126-year hourly
  record; 6,566-cell 2-D radar fit in 1.9 s; 20,000-series chunked streaming
  in 14.3 s.
- Real-data demo suite with verified scripts: Sentinel-6a/Jason-3n altimetry,
  WERA HF-radar ellipses, geodetic CryoSat-2 (FES2014 error quantification),
  a 127-year tide-gauge record (MSL trend 1.96 mm/yr vs NOAA's published
  1.94), ERA5 atmospheric tides (S1 ~1.0 hPa, S2 ~1.3 hPa), the CO-OPS
  station network, and hold-out prediction skill (R2 = 0.971).
- CI: tests green on Python 3.11-3.13; pre-commit clean repo-wide.

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

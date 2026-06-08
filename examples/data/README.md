# Example data

`noaa_hourly_2023.npz` — one year (2023) of hourly verified water level at 39
NOAA CO-OPS tide-gauge stations, assembled into a single `(n_times, n_stations)`
array for the `solve_many` real-data example
([`../gpu_batch_real.py`](../gpu_batch_real.py),
[`../../notebooks/gpu_batch_real_example.ipynb`](../../notebooks/gpu_batch_real_example.ipynb)).

Arrays: `t_days` (hours since 2023-01-01, in days), `levels` (m, MSL datum),
`ids`, `names`, `lats`, `lons`, `year`.

**Source:** NOAA CO-OPS, https://tidesandcurrents.noaa.gov — U.S. Government
work, public domain. Retrieved via the CO-OPS Data API
(`product=hourly_height`). See `../../gpu_work/fetch_noaa.py` for the fetch
script.

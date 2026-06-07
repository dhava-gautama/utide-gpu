"""
GPU-backend tests. These are skipped automatically when CuPy and a working
CUDA device are not available, so the suite still passes on CPU-only machines.
"""

import numpy as np
import pytest

from utide import solve, solve_many

# Skip the whole module unless CuPy + a device are present.
cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # noqa: BLE001
    pytest.skip("no CUDA device", allow_module_level=True)


CONSTIT = ["M2", "S2", "N2", "K1", "O1", "P1", "Q1", "M4", "M6", "MM", "MF"]


def _series(nt=2 * 365 * 24, twodim=False, seed=11):
    rng = np.random.default_rng(seed)
    t = np.arange(nt) / 24.0
    fr = {"M2": 1 / 12.42, "S2": 1 / 12.0, "K1": 1 / 23.93, "O1": 1 / 25.82}
    u = (
        np.cos(2 * np.pi * fr["M2"] * 24 * t + 0.3)
        + 0.4 * np.cos(2 * np.pi * fr["S2"] * 24 * t + 1.0)
        + 0.5 * np.cos(2 * np.pi * fr["K1"] * 24 * t + 2.0)
        + 0.1 * rng.standard_normal(nt)
    )
    if twodim:
        v = 0.6 * np.cos(2 * np.pi * fr["M2"] * 24 * t + 0.7) + 0.1 * rng.standard_normal(nt)
        return t, u, v
    return t, u, None


def test_gpu_matches_cpu_1d():
    t, u, _ = _series()
    kw = dict(lat=45, constit=CONSTIT, method="ols", conf_int="linear", verbose=False)
    c0 = solve(t, u, **kw)
    c1 = solve(t, u, gpu=True, **kw)
    assert np.allclose(c0["A"], c1["A"], rtol=1e-5, atol=1e-8)
    assert np.allclose(c0["g"], c1["g"], rtol=1e-5, atol=1e-6)
    assert np.allclose(c0["A_ci"], c1["A_ci"], rtol=1e-4, atol=1e-8, equal_nan=True)


def test_gpu_matches_cpu_2d():
    t, u, v = _series(twodim=True)
    kw = dict(lat=45, constit=CONSTIT, method="ols", conf_int="linear", verbose=False)
    c0 = solve(t, u, v, **kw)
    c1 = solve(t, u, v, gpu=True, **kw)
    assert np.allclose(c0["Lsmaj"], c1["Lsmaj"], rtol=1e-5, atol=1e-8)
    assert np.allclose(c0["g"], c1["g"], rtol=1e-5, atol=1e-6)


def test_gpu_fallback_unsupported_option():
    # nodal=False is not the GPU basis path; must fall back and still be correct.
    t, u, _ = _series()
    kw = dict(lat=45, constit=CONSTIT, method="ols", conf_int="none",
              nodal=False, verbose=False)
    c0 = solve(t, u, **kw)
    c1 = solve(t, u, gpu=True, **kw)
    assert np.allclose(c0["A"], c1["A"], rtol=1e-10, atol=1e-12)


def test_solve_many_matches_solve():
    t, u, _ = _series()
    X = np.column_stack([u, 0.7 * u, 1.3 * u])
    om = solve_many(t, X, lat=45, constit=CONSTIT, gpu=True, verbose=False)
    c = solve(t, u, lat=45, constit=CONSTIT, method="ols", conf_int="none", verbose=False)
    order = [list(om.name).index(n) for n in c["name"]]
    assert np.allclose(c["A"], om.A[order, 0], rtol=1e-5, atol=1e-8)
    # phase wrapped difference
    dg = (c["g"] - om.g[order, 0] + 180) % 360 - 180
    assert np.allclose(dg, 0, atol=1e-4)

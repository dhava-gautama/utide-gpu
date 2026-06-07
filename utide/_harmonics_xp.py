"""
Backend-agnostic (NumPy or CuPy) implementation of the complex exponential
basis ``ut_E`` and its nodal/satellite + astronomical-argument helpers.

The CPU (``xp=numpy``) path reproduces :func:`utide.harmonics.ut_E` to machine
precision; the GPU (``xp=cupy``) path runs the same math on the device. This is
where UTide spends the large majority of ``solve`` time, so it is the primary
target for acceleration.

Only the two most common ``ngflgs`` configurations are supported here:

* ``[0, 0, 0, 0]`` -- full nodal/satellite correction + Greenwich phase
  (the default), and
* ``ngflgs[1] and ngflgs[3]`` -- no nodal correction and raw (non-Greenwich)
  phase.

For any other configuration (the linearized-time approximations) callers
should fall back to the CPU :func:`utide.harmonics.ut_E`.
"""

import numpy as np

from ._ut_constants import ut_constants
from .astronomy import _coefs as _coefs_np

const = ut_constants.const
sat = ut_constants.sat
shallow = ut_constants.shallow

# Shallow-water bookkeeping (mirrors the module-level code in harmonics.py).
_nshallow = np.ma.masked_invalid(const.nshallow)
_ishallow = np.ma.masked_invalid(const.ishallow)
_not_shallow = _ishallow.mask
_nshallow = _nshallow.compressed().astype(int)
_ishallow = _ishallow.compressed().astype(int) - 1
_kshallow = np.nonzero(~_not_shallow)[0]


class _Tables:
    """Constant tables transferred to the chosen backend once."""

    def __init__(self, xp):
        self.xp = xp
        self.deldood = xp.asarray(sat.deldood)
        self.phcorr = xp.asarray(sat.phcorr)
        self.amprat = xp.asarray(sat.amprat)
        self.ilatfac = xp.asarray(sat.ilatfac)
        self.doodson = xp.asarray(const.doodson)
        self.semi = xp.asarray(const.semi)
        self.coefs = xp.asarray(_coefs_np)
        # Per-shallow-constituent gather indices/coefs, pre-built on device.
        self._shallow = []
        for i0, nshal, k in zip(_ishallow, _nshallow, _kshallow, strict=False):
            ik = slice(i0, i0 + nshal)
            j = xp.asarray(shallow.iname[ik] - 1)
            coef = xp.asarray(shallow.coef[ik])[:, None]
            self._shallow.append((int(k), j, coef, xp.abs(coef)))
        # Selection matrix (nfreq x nsat): satsel[iconst[s], s] = 1, so the
        # per-constituent satellite sum is a single matmul (complex-safe on GPU).
        nfreq = const.isat.shape[0]
        nsat = sat.iconst.shape[0]
        satsel = np.zeros((nfreq, nsat))
        satsel[sat.iconst - 1, np.arange(nsat)] = 1.0
        self.satsel = xp.asarray(satsel)
        self.satsel32 = self.satsel.astype("float32")
        self.nfreq = nfreq


# One cached table set per backend module (numpy / cupy).
_table_cache = {}


def get_tables(xp):
    key = xp.__name__
    tab = _table_cache.get(key)
    if tab is None:
        tab = _Tables(xp)
        _table_cache[key] = tab
    return tab


def ut_astron_xp(xp, jd, tab):
    jd = xp.atleast_1d(jd).ravel()
    daten = 693595.5  # Python epoch offset from the 1899-12-31 noon reference.
    d = jd - daten
    D = d / 10000
    args = xp.vstack((xp.ones(jd.shape), d, D * D, D**3))
    astro = xp.fmod((tab.coefs @ args) / 360, 1)
    tau = jd % 1 + astro[1, :] - astro[0, :]
    astro = xp.vstack((tau, astro))
    return astro


def _FUV_default(xp, t, lind, lat, tab, single=False):
    """Port of harmonics.FUV for the default ngflgs=[0, 0, 0, 0] path.

    If ``single``, the expensive complex matmul and transcendentals run in
    float32 while the astronomical reduction (which involves large day
    numbers) stays in float64 for phase accuracy. F and U are returned in
    float32; V is always returned in float64.
    """
    # ---- nodal/satellite correction (F, U) ----
    astro = ut_astron_xp(xp, t, tab)                      # always FP64
    if abs(lat) < 5:
        lat = np.sign(lat) * 5
    slat = np.sin(np.deg2rad(lat))
    rr = tab.amprat.copy()
    rr = xp.where(tab.ilatfac == 1, rr * 0.36309 * (1.0 - 5.0 * slat**2) / slat, rr)
    rr = xp.where(tab.ilatfac == 2, rr * 2.59808 * slat, rr)

    uu = tab.deldood @ astro[3:6, :] + tab.phcorr[:, None]  # FP64 reduction
    uu = xp.fmod(uu, 1)
    mat = rr[:, None] * xp.exp(1j * 2 * np.pi * uu)        # (nsat, nt) complex

    # F[ii] = 1 + sum_{s: iconst[s]==ii} mat[s]   via selection-matrix matmul
    if single:
        F = 1.0 + tab.satsel32 @ mat.astype(xp.complex64)  # FP32 (the big win)
    else:
        F = 1.0 + tab.satsel @ mat                         # (nfreq, nt) complex
    U = xp.angle(F) / (2 * np.pi)
    F = xp.abs(F)

    for k, j, coef, acoef in tab._shallow:
        if single:
            coef, acoef = coef.astype(U.dtype), acoef.astype(F.dtype)
        F[k, :] = xp.prod(F[j, :] ** acoef, axis=0)
        U[k, :] = xp.sum(U[j, :] * coef, axis=0)

    lind_x = xp.asarray(lind)
    F = F[lind_x, :].T
    U = U[lind_x, :].T

    # ---- Greenwich astronomical argument (V): always FP64 for phase accuracy ----
    astro = ut_astron_xp(xp, t, tab)
    V = tab.doodson @ astro + tab.semi[:, None]            # (nfreq, nt)
    V = xp.fmod(V, 1)
    for k, j, coef, _acoef in tab._shallow:
        V[k, :] = xp.sum(V[j, :] * coef, axis=0)
    V = V[lind_x, :].T
    return F, U, V


def gpu_supported(ngflgs):
    """True if this ngflgs configuration has a device implementation."""
    return list(ngflgs) == [0, 0, 0, 0] or (ngflgs[1] and ngflgs[3])


def ut_E_xp(xp, t, tref, frq, lind, lat, ngflgs, tab=None, precision="double"):
    """Complex exponential basis on the chosen backend (see module docstring).

    ``precision='single'`` uses the mixed-precision path (FP64 astronomical
    reduction, FP32 matmul/transcendentals) and returns a complex64 basis.
    """
    if tab is None:
        tab = get_tables(xp)
    single = precision == "single"
    t = xp.atleast_1d(t)
    frq = xp.atleast_1d(xp.asarray(frq))

    if ngflgs[1] and ngflgs[3]:
        # No nodal correction, raw (non-Greenwich) phase.
        V = (24 * (t - tref))[:, None] * frq[None, :]
        E = xp.exp(1j * V * 2 * np.pi)
        return E.astype(xp.complex64) if single else E

    if list(ngflgs) != [0, 0, 0, 0]:
        raise NotImplementedError(
            f"GPU basis not implemented for ngflgs={list(ngflgs)}; "
            "use the CPU path for linearized-time options.",
        )

    F, U, V = _FUV_default(xp, t, _to_host_int(lind), lat, tab, single=single)
    if single:
        # Keep V (the phase) FP64 through the add, then store as complex64.
        phase = U + V.astype(xp.float32)
        E = (F * xp.exp(1j * 2 * np.pi * phase)).astype(xp.complex64)
    else:
        E = F * xp.exp(1j * (U + V) * 2 * np.pi)
    return E


def _to_host_int(lind):
    return np.asarray(lind).astype(int)

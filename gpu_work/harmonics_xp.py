"""
Backend-agnostic (numpy OR cupy) port of UTide's ut_astron / FUV / ut_E.

Pass xp=numpy for CPU, xp=cupy for GPU. Constant tables are transferred to the
target device once via Tables(xp). The goal is a faithful reproduction of
utide.harmonics.ut_E so we can (a) validate numerical agreement and
(b) benchmark CPU vs GPU on the dominant basis-construction cost.
"""
import numpy as np
from utide._ut_constants import ut_constants
from utide.astronomy import _coefs as _coefs_np

const = ut_constants.const
sat = ut_constants.sat
shallow = ut_constants.shallow

# Shallow-water bookkeeping (mirrors harmonics.py module-level code)
_nshallow = np.ma.masked_invalid(const.nshallow)
_ishallow = np.ma.masked_invalid(const.ishallow)
_not_shallow = _ishallow.mask
_nshallow = _nshallow.compressed().astype(int)
_ishallow = _ishallow.compressed().astype(int) - 1
_kshallow = np.nonzero(~_not_shallow)[0]


class Tables:
    """Constant tables transferred to the chosen backend once."""
    def __init__(self, xp):
        self.xp = xp
        self.coefs = xp.asarray(_coefs_np)
        self.deldood = xp.asarray(sat.deldood)
        self.phcorr = xp.asarray(sat.phcorr)
        self.amprat = xp.asarray(sat.amprat)
        self.ilatfac = xp.asarray(sat.ilatfac)
        self.iconst = xp.asarray(sat.iconst - 1)
        self.doodson = xp.asarray(const.doodson)
        self.semi = xp.asarray(const.semi)
        self.shallow_iname = xp.asarray(shallow.iname - 1)
        self.shallow_coef = xp.asarray(shallow.coef)
        # selection matrix (nfreq x nsat): satsel[iconst[s], s] = 1
        nfreq = const.isat.shape[0]
        nsat = sat.iconst.shape[0]
        satsel = np.zeros((nfreq, nsat))
        satsel[sat.iconst - 1, np.arange(nsat)] = 1.0
        self.satsel = xp.asarray(satsel)
        # index lists kept on host (used for Python-level loops)
        self.ishallow = _ishallow
        self.nshallow = _nshallow
        self.kshallow = _kshallow
        self.nsat = sat.iconst.shape[0]


def ut_astron_xp(xp, jd, tab):
    jd = xp.atleast_1d(jd).ravel()
    daten = 693595.5
    d = jd - daten
    D = d / 10000
    args = xp.vstack((xp.ones(jd.shape), d, D * D, D**3))
    astro = xp.fmod((tab.coefs @ args) / 360, 1)
    tau = jd % 1 + astro[1, :] - astro[0, :]
    astro = xp.vstack((tau, astro))
    return astro


def FUV_xp(xp, t, tref, lind, lat, tab):
    """Faithful port of harmonics.FUV for the default ngflgs=[0,0,0,0] path."""
    t = xp.atleast_1d(t).ravel()
    nt = len(t)

    # ---- nodsat correction ----
    astro = ut_astron_xp(xp, t, tab)
    if abs(lat) < 5:
        lat = np.sign(lat) * 5
    slat = np.sin(np.deg2rad(lat))
    rr = tab.amprat.copy()
    rr = xp.where(tab.ilatfac == 1, rr * 0.36309 * (1.0 - 5.0 * slat**2) / slat, rr)
    rr = xp.where(tab.ilatfac == 2, rr * 2.59808 * slat, rr)

    uu = tab.deldood @ astro[3:6, :] + tab.phcorr[:, None]
    uu = xp.fmod(uu, 1)
    mat = rr[:, None] * xp.exp(1j * 2 * np.pi * uu)          # (nsat, nt) complex

    nfreq = const.isat.shape[0]                              # 146
    # segment-sum of satellite contributions per constituent, as a single
    # selection-matrix multiply (works for complex on numpy and cupy):
    #   F[ii] = 1 + sum_{s: iconst[s]==ii} mat[s]
    F = 1.0 + tab.satsel @ mat                               # (nfreq, nt)

    U = xp.angle(F) / (2 * np.pi)
    F = xp.abs(F)

    for i0, nshal, k in zip(tab.ishallow, tab.nshallow, tab.kshallow):
        ik = slice(i0, i0 + nshal)
        j = xp.asarray(shallow.iname[ik] - 1)
        exp1 = xp.asarray(shallow.coef[ik])[:, None]
        exp2 = xp.abs(exp1)
        F[int(k), :] = xp.prod(F[j, :] ** exp2, axis=0)
        U[int(k), :] = xp.sum(U[j, :] * exp1, axis=0)

    lind_x = xp.asarray(lind)
    F = F[lind_x, :].T
    U = U[lind_x, :].T

    # ---- gwch (astronomical argument V) ----
    astro = ut_astron_xp(xp, t, tab)
    V = tab.doodson @ astro + tab.semi[:, None]             # (146, nt)
    V = xp.fmod(V, 1)
    for i0, nshal, k in zip(tab.ishallow, tab.nshallow, tab.kshallow):
        ik = slice(i0, i0 + nshal)
        j = xp.asarray(shallow.iname[ik] - 1)
        exp1 = xp.asarray(shallow.coef[ik])[:, None]
        V[int(k), :] = xp.sum(V[j, :] * exp1, axis=0)
    V = V[lind_x, :].T
    return F, U, V


def ut_E_xp(xp, t, tref, frq, lind, lat, tab):
    t = xp.atleast_1d(t)
    F, U, V = FUV_xp(xp, t, tref, lind, lat, tab)
    E = F * xp.exp(1j * (U + V) * 2 * np.pi)
    return E

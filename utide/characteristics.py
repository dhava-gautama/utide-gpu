"""
Empirical tidal datums and characteristics from a water-level time series.

These are standard, non-harmonic tidal quantities computed directly from the
series by detecting its high and low waters: the datums MHW / MLW / MTL / MTR
and the mean ebb / flood durations. They complement the harmonic analysis in
:func:`utide.solve` (constituent amplitudes and phases).

The feature set and abbreviations follow DHI's ``tide_analytics``
(https://github.com/DHI/tide_analytics, MIT). This is an independent
NumPy/SciPy implementation; it does not depend on that package.

============  ====================================================
Abbreviation  Description
============  ====================================================
MHW           mean high water (mean of the high-water levels)
MLW           mean low water (mean of the low-water levels)
MTL           mean tide level, ``(MHW + MLW) / 2``
MTR           mean tidal range, ``MHW - MLW``
ED            mean ebb duration, high water to the next low water [hours]
FD            mean flood duration, low water to the next high water [hours]
============  ====================================================
"""

import numpy as np
from scipy.signal import find_peaks

from ._time_conversion import _normalize_time
from .utilities import Bunch

__all__ = ["tidal_characteristics", "tidal_characteristics_many", "tidal_form_factor"]

_FORM_EDGES = [0.25, 1.5, 3.0]
_FORM_LABELS = np.array(
    ["semidiurnal", "mixed (mainly semidiurnal)", "mixed (mainly diurnal)", "diurnal"],
)


def tidal_form_factor(coef, classify=False):
    """
    Tidal form factor ``F = (K1 + O1) / (M2 + S2)`` from a harmonic solution.

    The form factor classifies the tidal regime: ``F < 0.25`` semidiurnal,
    ``0.25-1.5`` mixed (mainly semidiurnal), ``1.5-3`` mixed (mainly diurnal),
    ``> 3`` diurnal.

    Works on a single ``coef`` from :func:`utide.solve` (returns a float) or on
    a :func:`utide.solve_many` result (returns one value per series).

    Parameters
    ----------
    coef : `Bunch`
        Output of ``solve`` or ``solve_many`` (must contain the K1, O1, M2, S2
        amplitudes in ``coef['A']``).
    classify : bool, optional
        If True, also return the regime label(s).

    Returns
    -------
    F : float or ndarray
        The form factor.
    regime : str or ndarray of str
        Only if ``classify`` is True.
    """
    names = list(coef["name"])
    A = np.asarray(coef["A"])

    def amp(n):
        return A[names.index(n)] if n in names else 0.0 * A[0]

    F = (amp("K1") + amp("O1")) / (amp("M2") + amp("S2"))
    if not classify:
        return F
    return F, _FORM_LABELS[np.digitize(F, _FORM_EDGES)]

_KEYS = ("MHW", "MLW", "MTL", "MTR", "ED", "FD")


def _as_days(t, epoch):
    """Return time as a float array in days. Datetime-like input is converted."""
    t = np.atleast_1d(t)
    if t.dtype.kind in ("M", "m", "O"):  # datetime64, timedelta64, or objects
        return np.asarray(_normalize_time(t, epoch), dtype=float)
    return t.astype(float)


def tidal_characteristics(t, h, min_period_hours=2.0, prominence=None, epoch=None):
    """
    Tidal datums and ebb/flood durations from a water-level series.

    Parameters
    ----------
    t : array_like
        Times in days (e.g. the datenum array passed to :func:`utide.solve`),
        or a datetime / ``np.datetime64`` / pandas datetime array. Monotonic;
        spacing may be irregular.
    h : array_like
        Water level (surface elevation), same length as ``t``. NaNs are ignored.
    min_period_hours : float, optional
        Minimum separation between successive high (or low) waters, in hours;
        passed to :func:`scipy.signal.find_peaks` as ``distance``. Set it a
        little below the expected ebb/flood duration. Default 2 hours.
    prominence : float or None, optional
        Minimum peak prominence (same units as ``h``). Default None.
    epoch : optional
        Passed to the time normalizer when ``t`` is datetime-like (see
        :func:`utide.solve`). Ignored when ``t`` is already in days.

    Returns
    -------
    out : `Bunch` or None
        Fields ``MHW``, ``MLW``, ``MTL``, ``MTR`` (same units as ``h``),
        ``ED``, ``FD`` (hours), the integer counts ``n_high`` / ``n_low``, and
        the detected ``hw_t`` / ``hw_h`` / ``lw_t`` / ``lw_h`` (high/low water
        times in days and levels). Returns ``None`` if no high or low water is
        found.
    """
    t = _as_days(t, epoch)
    h = np.asarray(h, dtype=float)
    good = np.isfinite(t) & np.isfinite(h)
    t, h = t[good], h[good]
    if len(h) < 4:
        return None

    dt_h = np.median(np.diff(t)) * 24.0
    distance = max(1, int(round(min_period_hours / dt_h)))
    hw, _ = find_peaks(h, distance=distance, prominence=prominence)
    lw, _ = find_peaks(-h, distance=distance, prominence=prominence)
    if len(hw) == 0 or len(lw) == 0:
        return None

    MHW, MLW = float(h[hw].mean()), float(h[lw].mean())

    # Ebb = high water -> next low water; flood = low water -> next high water.
    # Merge the extrema in time and keep the valid HW->LW / LW->HW transitions
    # (robust to the occasional non-alternating pair).
    idx = np.concatenate([hw, lw])
    kind = np.concatenate([np.ones(len(hw)), -np.ones(len(lw))])
    order = np.argsort(idx)
    idx, kind = idx[order], kind[order]
    dur = np.diff(t[idx]) * 24.0
    ebb = dur[(kind[:-1] == 1) & (kind[1:] == -1)]
    flood = dur[(kind[:-1] == -1) & (kind[1:] == 1)]

    return Bunch(
        MHW=MHW,
        MLW=MLW,
        MTL=0.5 * (MHW + MLW),
        MTR=MHW - MLW,
        ED=float(ebb.mean()) if len(ebb) else np.nan,
        FD=float(flood.mean()) if len(flood) else np.nan,
        n_high=len(hw),
        n_low=len(lw),
        hw_t=t[hw],
        hw_h=h[hw],
        lw_t=t[lw],
        lw_h=h[lw],
    )


def tidal_characteristics_many(t, X, min_period_hours=2.0, prominence=None, epoch=None):
    """
    Tidal datums for every series sharing one time base (batch of :func:`tidal_characteristics`).

    Parameters
    ----------
    t : array_like
        Times shared by every series (days or datetime-like); see
        :func:`tidal_characteristics`.
    X : array_like, shape (n_times,) or (n_times, n_series)
        One or more water-level series (columns are series).
    min_period_hours, prominence, epoch
        As in :func:`tidal_characteristics`.

    Returns
    -------
    out : `Bunch`
        ``MHW``, ``MLW``, ``MTL``, ``MTR``, ``ED``, ``FD`` and ``n_high`` /
        ``n_low``, each a length ``n_series`` array; series with no detectable
        tide are NaN (counts 0). Peak detection is per series, so this loops
        rather than running on the GPU.
    """
    X = np.atleast_1d(X)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    nser = X.shape[1]
    out = Bunch(**{k: np.full(nser, np.nan) for k in _KEYS})
    out.n_high = np.zeros(nser, dtype=int)
    out.n_low = np.zeros(nser, dtype=int)
    for s in range(nser):
        c = tidal_characteristics(
            t, X[:, s], min_period_hours=min_period_hours,
            prominence=prominence, epoch=epoch,
        )
        if c is not None:
            for k in _KEYS:
                out[k][s] = c[k]
            out.n_high[s] = c.n_high
            out.n_low[s] = c.n_low
    return out

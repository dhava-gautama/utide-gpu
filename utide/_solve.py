"""
Central module for calculating the tidal amplitudes, phases, etc.
"""

import numpy as np

from ._time_conversion import _normalize_time
from .confidence import _confidence
from .constituent_selection import ut_cnstitsel
from .diagnostics import _PE, _SNR, ut_diagn
from .ellipse_params import ut_cs2cep
from .harmonics import ut_E
from .robustfit import robustfit
from .utilities import Bunch

default_opts = {
    "constit": "auto",
    "order_constit": None,
    "conf_int": "linear",
    "method": "ols",
    "trend": True,
    "phase": "Greenwich",
    "nodal": True,
    "infer": None,
    "MC_n": 200,
    "Rayleigh_min": 1,
    "robust_kw": {"weight_function": "cauchy"},
    "white": False,
    "verbose": True,
    "epoch": None,
    "gpu": False,
    "gpu_precision": "double",
}


def _process_opts(opts, is_2D):
    newopts = Bunch(default_opts)
    newopts.update_values(strict=True, **opts)
    # TODO: add more validations.
    newopts.infer = validate_infer(newopts.infer, is_2D)
    snr = newopts.conf_int != "none"
    newopts.order_constit = validate_order_constit(newopts.order_constit, snr)

    compat_opts = _translate_opts(newopts)

    return compat_opts


def _translate_opts(opts):
    # Temporary shim between new-style options and Matlab heritage.
    # Here or elsewhere, proper validation remains to be added.
    oldopts = Bunch()
    oldopts.cnstit = opts.constit
    oldopts.ordercnstit = opts.order_constit
    oldopts.infer = opts.infer  # we will not use the matlab names, though

    oldopts.conf_int = True
    if opts.conf_int == "linear":
        oldopts.linci = True
    elif opts.conf_int == "MC":
        oldopts.linci = False
    elif opts.conf_int == "none":
        oldopts.conf_int = False
        oldopts.nodiagn = 1
    else:
        raise ValueError("'conf_int' must be 'linear', 'MC', or 'none'")

    oldopts.notrend = not opts.trend
    oldopts["nodesatlint"] = False
    oldopts["nodesatnone"] = False
    oldopts["gwchlint"] = False
    oldopts["gwchnone"] = False
    if opts.nodal == "linear_time":
        oldopts["nodsatlint"] = True
    elif not opts.nodal:
        oldopts["nodsatnone"] = True
    if opts.phase == "linear_time":
        oldopts["gwchlint"] = True
    elif opts.phase == "raw":
        oldopts["gwchnone"] = True
    # Otherwise it should be default, 'Greenwich.'
    oldopts.rmin = opts.Rayleigh_min
    oldopts.white = opts.white
    oldopts.newopts = opts  # So we can access new opts via the single "opt."
    oldopts["RunTimeDisp"] = opts.verbose
    oldopts.epoch = opts.epoch
    return oldopts


def validate_infer(infer, is_2D):
    if infer is None or infer == "none":
        return None
    required_keys = {"inferred_names", "reference_names", "amp_ratios", "phase_offsets"}
    keys = set(infer.keys())
    if keys < required_keys:
        raise ValueError(f"infer option must include {required_keys:s}")
    nI = len(infer.inferred_names)
    if len(infer.reference_names) != nI:
        raise ValueError("inferred_names must be same" "  length as reference_names")
    nratios = 2 * nI if is_2D else nI
    if len(infer.amp_ratios) != nratios or len(infer.phase_offsets) != nratios:
        raise ValueError(f"ratios and offsets need to have length {nratios:d}")
    if "approximate" not in infer:
        infer.approximate = False
    return infer


def validate_order_constit(arg, have_snr):
    available = ["PE", "frequency"]
    if have_snr:
        available.append("SNR")
    if arg is None:
        return "PE"
    if isinstance(arg, str) and arg in available:
        return arg
    if not isinstance(arg, str) and np.iterable(arg):
        return arg  # TODO: add checking of its elements
    raise ValueError(
        f"order_constit must be one of {available} or"
        f" a sequence of constituents, not '{arg}'",
    )


def solve(t, u, v=None, lat=None, **opts):
    """
    Calculate amplitude, phase, confidence intervals of tidal constituents.

    Parameters
    ----------
    t : array_like
        Time in days since `epoch`, or np.datetime64 array, or pandas datetime array.
    u : array_like
        Sea-surface height, velocity component, etc.
    v : {None, array_like}, optional
        If `u` is a velocity component, `v` is the orthogonal component.
    lat : float, required
        Latitude in degrees.
    epoch : {string, `datetime.date`, `datetime.datetime`}, if datenum is provided in t.
        Default `None` if `t` is `datetime`, `np.datetime64`, or `pd.datetime array.`
        Optional valid strings are
            - 'python' : if `t` is days since '0000-12-31'
            - 'matlab' : if `t` is days since '0000-00-00'
        Or, an arbitrary date in the form 'YYYY-MM-DD'.
    constit : {'auto', sequence}, optional
        List of strings with standard letter abbreviations of
        tidal constituents; or 'auto' to let the list be determined
        based on the time span.
    conf_int : {'linear', 'MC', 'none'}, optional
        If not 'none' (string), calculate linearized confidence
        intervals, or use a Monte-Carlo simulation.
    method : {'ols', 'robust'}, optional
        Solve with ordinary least squares, or with a robust algorithm.
    trend : bool, optional
        True (default) to include a linear trend in the model.
    phase : {'Greenwich', 'linear_time', 'raw'}, optional
        Give Greenwich-referenced phase lags, an approximation
        using linearized times, or raw lags.
    nodal : {True, False, 'linear_time'}, optional
        True (default) to include nodal/satellite corrections;
        'linear_time' to use the linearized time approximation;
        False to omit nodal corrections.

    Returns
    -------
    coef : Bunch
        Data container with all configuration and solution information:

    Other Parameters
    ----------------
    infer : {None, dict or Bunch}, optional; default is None.
        If not None, the items are:

        **inferred_names** : {sequence of N strings}
            inferred constituent names
        **reference_names** : {sequence of N strings}
            reference constituent names
        **amp_ratios** : {sequence, N or 2N floats}
            amplitude ratios (unitless)
        **phase_offsets** : {sequence, N or 2N floats}
            phase offsets (degrees)
        **approximate** : {bool, optional (default is False)}
            use approximate method

        amp_ratios and phase_offsets have length N for a scalar
        time series, or 2N for a vector series.

    order_constit : {'PE', 'SNR', 'frequency', sequence}, optional
        The default is 'PE' (percent energy) order, returning results ordered from
        high energy to low.
        The 'SNR' order is from high signal-to-noise ratio to low, and is
        available only if `conf_int` is not 'none'. The
        'frequency' order is from low to high frequency. Alternatively, a
        sequence of constituent names may be supplied, typically the same list as
        given in the *constit* option.
    MC_n : integer, optional
        Not yet implemented.
    robust_kw : dict, optional
        Keyword arguments for `robustfit`, if `method` is 'robust'.
    Rayleigh_min : float
        Minimum conventional Rayleigh criterion for automatic
        constituent selection; default is 1.
    white : bool
        If False (default), use band-averaged spectra from the
        residuals in the confidence limit estimates; if True,
        assume a white background spectrum.
    verbose : {True, False}, optional
        True (default) turns on verbose output. False emits no messages.
    gpu : {False, True}, optional
        If True, run basis construction and (for ``method='ols'``) the
        least-squares solve on the GPU via CuPy, returning results on the
        host. Requires CuPy with a working CUDA device. Falls back to the
        CPU automatically for option combinations not yet supported on the
        GPU (inference, or the linearized-time nodal/phase approximations).
        Default is False.
    gpu_precision : {'double', 'single'}, optional
        Precision of the GPU computation. 'double' (default) matches the CPU
        result to round-off. 'single' runs the harmonic basis and the solve
        in float32 (the astronomical-argument reduction stays float64 for
        phase accuracy) for a large speedup on consumer GPUs whose FP64
        throughput is throttled, at reduced precision -- typically 3-5
        significant digits in amplitude, ~0.005 deg in phase, and worse for
        large batches or ill-conditioned fits. Use it for screening, not for
        final-precision results. Ignored when ``gpu`` is False.

    Note
    ----
    `utide.reconstruct` requires the calculation of confidence intervals.

    Notes
    -----

    To be added: much additional explanation.

    There will also be more "Other Parameters".

    """

    compat_opts = _process_opts(opts, v is not None)

    coef = _solv1(t, u, v, lat, **compat_opts)

    return coef


def solve_many(
    t,
    u,
    v=None,
    lat=None,
    gpu=True,
    epoch=None,
    constit="auto",
    trend=True,
    nodal=True,
    phase="Greenwich",
    Rayleigh_min=1,
    verbose=True,
    gpu_precision="double",
    chunk_size=None,
    solver="lstsq",
):
    """
    Vectorized OLS tidal fit for many series sharing one time base.

    Batch counterpart to :func:`solve`. Given ``S`` series sampled at the
    same times ``t``, it builds the harmonic model **once** and solves all
    series in a single (optionally GPU) least-squares call -- far faster than
    looping :func:`solve` over each series. Returns amplitudes and Greenwich
    phases per constituent per series.

    Confidence intervals, robust fitting, and inference are **not** computed
    here; use :func:`solve` per series for those.

    Parameters
    ----------
    t : array_like (nt,)
        Times shared by every series (days since ``epoch``, or datetime-like).
    u : array_like (nt,) or (nt, S)
        One or more scalar series (columns are series).
    v : array_like, optional
        Orthogonal component(s), same shape as ``u``, for a 2-D (u, v) fit.
    lat : float
        Latitude in degrees (required).
    gpu : bool, optional
        Use the CuPy backend if available (default True). Falls back to NumPy
        if CuPy / a CUDA device is unavailable.
    epoch, constit, trend, nodal, phase, Rayleigh_min, verbose
        As in :func:`solve`.
    gpu_precision : {'double', 'single'}, optional
        Precision of the batched GPU solve; 'single' is much faster on
        consumer GPUs at reduced precision. As in :func:`solve`.
    chunk_size : int or None, optional
        Maximum number of series solved per device call. ``None`` (default)
        picks a size from free GPU memory so very large ``S`` (or long
        records) does not exhaust VRAM; the model matrix stays resident while
        series stream through. Ignored on the CPU.
    solver : {'lstsq', 'normal'}, optional
        'lstsq' (default) is the maximally stable least-squares solver.
        'normal' uses a normal-equations (Cholesky) solve on the
        double-precision path, which matches ``lstsq`` to round-off and is
        ~1.5-2x faster for a few thousand well-conditioned series; it falls
        back to ``lstsq`` if the design is rank-deficient or badly
        conditioned. Note that for very large batches it can be *slower* on
        consumer GPUs with throttled double precision (the per-chunk matmuls
        dominate), so it is opt-in rather than the default. Single precision
        always uses ``lstsq``.

    Returns
    -------
    out : `Bunch`
        ``name`` (nc,), ``frq`` (nc,), and per-series results shaped ``(nc, S)``:
        ``A`` and ``g`` for scalar input, or ``Lsmaj``, ``Lsmin``, ``theta``,
        ``g`` for (u, v) input. Also ``mean`` (S,) [or ``umean``/``vmean``],
        and ``slope`` (S,) [or ``uslope``/``vslope``] when ``trend`` is True.

    Notes
    -----
    Per-series gaps are supported: series are grouped by their valid-sample
    pattern and each group is solved with its own model matrix (no-gap data is
    a single group, the fast path). All series share the constituent set
    selected from the full time base. Series that are entirely NaN, or that
    have fewer valid samples than model parameters, are returned as NaN. Rows
    where ``t`` itself is NaN are dropped for all series.
    """
    from ._backend import asnumpy, get_xp
    from ._harmonics_xp import gpu_supported, ut_E_xp

    if lat is None:
        raise ValueError("Latitude must be supplied")

    t = _normalize_time(np.atleast_1d(t), epoch).astype(float)
    twodim = v is not None
    u = np.atleast_1d(u)
    U = u[:, np.newaxis] if u.ndim == 1 else u
    if twodim:
        v = np.atleast_1d(v)
        V = v[:, np.newaxis] if v.ndim == 1 else v
        X = U + 1j * V
    else:
        X = U.astype(float)

    # Drop rows where the (shared) time is invalid.
    tfin = np.isfinite(t)
    t = t[tfin]
    X = X[tfin]
    nt, S = X.shape

    lor = np.ptp(t)
    tref = 0.5 * (t[0] + t[-1])

    # Translate the nodal/phase flags to ngflgs (mirrors _translate_opts).
    ngflgs = [
        nodal == "linear_time",
        not nodal,
        phase == "linear_time",
        phase == "raw",
    ]

    cnstit, coef = ut_cnstitsel(tref, Rayleigh_min / (24 * lor), constit, None)
    nNR = coef.nNR
    nm = 2 * nNR + 1 + (1 if trend else 0)

    use_gpu = bool(gpu) and gpu_supported(ngflgs)
    xp = get_xp(use_gpu)
    gpu_single = use_gpu and gpu_precision == "single"

    # Build the full model matrix once; gap groups subselect rows from it
    # rather than rebuilding the (expensive) basis per group.
    if use_gpu:
        E = ut_E_xp(
            xp, xp.asarray(t), tref, cnstit.NR.frq, cnstit.NR.lind, lat,
            ngflgs, precision="single" if gpu_single else "double",
        )
    else:
        E = ut_E(t, tref, cnstit.NR.frq, cnstit.NR.lind, lat, ngflgs, [])
    B_full = xp.hstack((E, E.conj(), xp.ones((nt, 1), dtype=E.real.dtype)))
    if trend:
        tc = xp.asarray((t - tref) / lor)[:, np.newaxis].astype(E.real.dtype)
        B_full = xp.hstack((B_full, tc))

    def _auto_chunk(nrows):
        # Columns per solve, bounded so X-on-device + lstsq workspace fit VRAM.
        if not use_gpu:
            return 1 << 30  # host RAM: no chunking needed
        free, _total = xp.cuda.runtime.memGetInfo()
        per_col = nrows * B_full.dtype.itemsize * 4  # X + workspace headroom
        budget = max(int(free * 0.25), 64 << 20)
        return max(1, budget // per_col)

    def _solve_group(rows, cols):
        Bg = B_full if rows.all() else B_full[xp.asarray(np.flatnonzero(rows))]
        cc = chunk_size or _auto_chunk(Bg.shape[0])
        # Normal-equations path: build and gate the Gram matrix ONCE per group
        # (it is shared across all column chunks); the expensive B^H B is not
        # recomputed per chunk. Cholesky gates it -- if the design is rank-
        # deficient / badly conditioned it raises and we fall back to lstsq.
        use_normal = solver != "lstsq" and not gpu_single
        G = BgH = None
        if use_normal:
            G = Bg.conj().T @ Bg
            try:
                xp.linalg.cholesky(G)
            except Exception:  # noqa: BLE001  (LinAlgError, backend-dependent)
                use_normal = False
            else:
                BgH = Bg.conj().T
        parts = []
        for i in range(0, len(cols), cc):
            Xg = xp.asarray(X[np.ix_(rows, cols[i : i + cc])], dtype=B_full.dtype)
            if use_normal:
                Mg = xp.linalg.solve(G, BgH @ Xg)
            else:
                try:
                    Mg = xp.linalg.lstsq(Bg, Xg, rcond=None)[0]
                except TypeError:
                    Mg = xp.linalg.lstsq(Bg, Xg)[0]
            parts.append(asnumpy(Mg))
        Mg = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        return Mg.astype(np.complex128) if Mg.dtype == np.complex64 else Mg

    # Solve, batching stations that share the same valid-sample mask so the
    # model matrix is built and factored once per distinct gap pattern. The
    # common no-gap case is a single group; columns that are all-NaN or
    # under-determined are left as NaN.
    M = np.full((nm, S), np.nan, dtype=np.complex128)
    present = np.isfinite(X)
    full = present.all(axis=0)
    ngroups = 0
    cols_full = np.flatnonzero(full)
    if cols_full.size:
        allrows = np.ones(nt, dtype=bool)
        M[:, cols_full] = _solve_group(allrows, cols_full)
        ngroups += 1

    gappy = np.flatnonzero(~full)
    n_dropped = 0
    if gappy.size:
        keys = np.packbits(present[:, gappy], axis=0).T
        groups = {}
        for j, col in enumerate(gappy):
            if not present[:, col].any():
                n_dropped += 1
                continue
            groups.setdefault(keys[j].tobytes(), []).append(col)
        for cols in groups.values():
            rows = present[:, cols[0]]
            if int(rows.sum()) <= nm:
                n_dropped += len(cols)
                continue
            M[:, np.asarray(cols)] = _solve_group(rows, np.asarray(cols))
            ngroups += 1

    if verbose:
        where = "gpu" if use_gpu else "cpu"
        extra = f", {ngroups} gap groups" if gappy.size else ""
        drop = f", {n_dropped} series too gappy (NaN)" if n_dropped else ""
        print(
            f"solve_many: {S} series, {nNR} constituents, {nt} times "
            f"[{where}]{extra}{drop} ...",
        )

    ap = M[:nNR]
    am = M[nNR : 2 * nNR]
    Xu = np.real(ap + am)
    Yu = -np.imag(ap - am)

    out = Bunch(
        name=coef.name,
        frq=coef.aux.frq,
        lat=lat,
        nseries=S,
        aux=Bunch(
            lind=coef.aux.lind,
            reftime=tref,
            lat=lat,
            ngflgs=list(ngflgs),
            twodim=twodim,
            trend=trend,
            lor=lor,
        ),
    )
    if not twodim:
        A, _, _, g = ut_cs2cep(Xu, Yu)
        out.A, out.g = A, g
        if trend:
            out.mean = np.real(M[-2])
            out.slope = np.real(M[-1]) / lor
        else:
            out.mean = np.real(M[-1])
    else:
        Xv = np.imag(ap + am)
        Yv = np.real(ap - am)
        Lsmaj, Lsmin, theta, g = ut_cs2cep(Xu, Yu, Xv, Yv)
        out.Lsmaj, out.Lsmin, out.theta, out.g = Lsmaj, Lsmin, theta, g
        if trend:
            out.umean = np.real(M[-2])
            out.vmean = np.imag(M[-2])
            out.uslope = np.real(M[-1]) / lor
            out.vslope = np.imag(M[-1]) / lor
        else:
            out.umean = np.real(M[-1])
            out.vmean = np.imag(M[-1])

    if u.ndim == 1:
        # collapse singleton series axis for convenience
        for k in ("A", "g", "Lsmaj", "Lsmin", "theta"):
            if k in out:
                out[k] = out[k][:, 0]
    return out


def _solv1(tin, uin, vin, lat, **opts):
    # The following returns a possibly modified copy of tin (ndarray).
    # t, u, v are fully edited ndarrays (unless v is None).
    packed = _slvinit(tin, uin, vin, lat, **opts)
    tin, t, u, v, tref, lor, elor, opt = packed
    nt = len(t)
    if opt["RunTimeDisp"]:
        print("solve: ", end="")

    # opt['cnstit'] = cnstit
    cnstit, coef = ut_cnstitsel(
        tref,
        opt["rmin"] / (24 * lor),
        opt["cnstit"],
        opt["infer"],
    )

    # a function we don't need
    # coef.aux.rundescr = ut_rundescr(opt,nNR,nR,nI,t,tgd,uvgd,lat)

    coef.aux.opt = opt
    coef.aux.lat = lat

    if opt["RunTimeDisp"]:
        print("matrix prep ... ", end="")

    ngflgs = [opt["nodsatlint"], opt["nodsatnone"], opt["gwchlint"], opt["gwchnone"]]

    E_args = (lat, ngflgs, opt.prefilt)

    # Select the array backend. The GPU path accelerates basis construction
    # and (for OLS) the least-squares solve; it is used only for the
    # configurations it supports, falling back to NumPy otherwise.
    from ._backend import asnumpy, get_xp, is_gpu_array
    from ._harmonics_xp import gpu_supported, ut_E_xp

    want_gpu = bool(opt.newopts.gpu)
    use_gpu = want_gpu and opt.infer is None and gpu_supported(ngflgs)
    if want_gpu and not use_gpu and opt["RunTimeDisp"]:
        print("(gpu unsupported for these options; using cpu) ", end="")
    xp = get_xp(use_gpu)

    # Make the model array, starting with the harmonics.
    gpu_single = use_gpu and opt.newopts.gpu_precision == "single"
    if use_gpu:
        t_dev = xp.asarray(t)
        E = ut_E_xp(
            xp, t_dev, tref, cnstit.NR.frq, cnstit.NR.lind, lat, ngflgs,
            precision="single" if gpu_single else "double",
        )
    else:
        E = ut_E(t, tref, cnstit.NR.frq, cnstit.NR.lind, *E_args)

    # Positive and negative frequencies
    B = xp.hstack((E, E.conj()))

    if opt.infer is not None:
        Etilp = np.empty((nt, coef.nR), dtype=complex)
        Etilm = np.empty((nt, coef.nR), dtype=complex)

        if not opt.infer.approximate:
            for k, ref in enumerate(cnstit.R):
                E = ut_E(t, tref, ref.frq, ref.lind, *E_args)
                # (nt,1)
                Q = ut_E(t, tref, ref.I.frq, ref.I.lind, *E_args) / E
                # (nt,ni)
                Qsum_p = (Q * ref.I.Rp).sum(axis=1)
                Etilp[:, k] = E[:, 0] * (1 + Qsum_p)
                Qsum_m = (Q * np.conj(ref.I.Rm)).sum(axis=1)
                Etilm[:, k] = E[:, 0] * (1 + Qsum_m)

        else:
            # Approximate inference.
            Q = np.empty((coef.nR,), dtype=float)
            beta = np.empty((coef.nR,), dtype=float)

            for k, ref in enumerate(cnstit.R):
                E = ut_E(t, tref, ref.frq, ref.lind, *E_args)[:, 0]
                Etilp[:, k] = E
                Etilm[:, k] = E
                num = ut_E(tref, tref, ref.I.frq, ref.I.lind, *E_args).real
                den = ut_E(tref, tref, ref.frq, ref.lind, *E_args).real
                Q[k] = (num / den)[0, 0]
                arg = np.pi * lor * 24 * (ref.I.frq - ref.frq) * (nt + 1) / nt
                beta[k] = np.sin(arg) / arg

        B = np.hstack((B, Etilp, np.conj(Etilm)))

    # add the mean (match B's real dtype so a complex64 basis stays complex64)
    B = xp.hstack((B, xp.ones((nt, 1), dtype=B.real.dtype)))

    if not opt["notrend"]:
        trend_col = xp.asarray((t - tref) / lor)[:, np.newaxis].astype(B.real.dtype)
        B = xp.hstack((B, trend_col))

    # nm = B.shape[1]  # 2*(nNR + nR) + 1, plus 1 if trend is included.

    if opt["RunTimeDisp"]:
        print("solution ... ", end="")

    if opt["twodim"]:
        xraw = u + 1j * v
    else:
        xraw = u

    if opt.newopts.method == "ols":
        # Model coefficients (on device if use_gpu). In single-precision GPU
        # mode B is already complex64, so the solve runs in float32 here.
        xraw_solve = xp.asarray(xraw, dtype=B.dtype) if use_gpu else xraw
        try:
            m = xp.linalg.lstsq(B, xraw_solve, rcond=None)[0]
        except TypeError:
            m = xp.linalg.lstsq(B, xraw_solve)[0]
        W = np.ones(nt)  # Uniform weighting; we could use a scalar 1, or None.
        # Return to host for the remaining (CPU) pipeline.
        B = asnumpy(B)
        m = asnumpy(m)
    elif use_gpu:
        # Robust IRLS on the device: the whole reweighting loop stays on the
        # GPU (B is already a device array). Bring results back to the host.
        rf = robustfit(B, xp.asarray(xraw, dtype=B.dtype), **opt.newopts.robust_kw)
        m = asnumpy(rf.b)
        W = asnumpy(rf.w)
        for _k in list(rf.keys()):
            if is_gpu_array(rf[_k]):
                rf[_k] = asnumpy(rf[_k])
        coef.rf = rf
        B = asnumpy(B)
    else:
        rf = robustfit(B, xraw, **opt.newopts.robust_kw)
        m = rf.b
        W = rf.w
        coef.rf = rf
    # Promote a single-precision (complex64) GPU result to complex128 for the
    # host confidence/diagnostics pipeline, which expects double precision.
    if B.dtype == np.complex64:
        B = B.astype(np.complex128)
        m = np.asarray(m).astype(np.complex128)
    coef.weights = W

    xmod = np.dot(B, m)  # Model fit.

    if not opt["twodim"]:
        xmod = np.real(xmod)

    e = W * (xraw - xmod)  # Weighted residuals.

    nI, nR, nNR = coef.nI, coef.nR, coef.nNR

    ap = np.hstack((m[:nNR], m[2 * nNR : 2 * nNR + nR]))
    i0 = 2 * nNR + nR
    am = np.hstack((m[nNR : 2 * nNR], m[i0 : i0 + nR]))

    Xu = np.real(ap + am)
    Yu = -np.imag(ap - am)

    if not opt["twodim"]:
        coef["A"], _, _, coef["g"] = ut_cs2cep(Xu, Yu)
        Xv = []
        Yv = []

    else:
        Xv = np.imag(ap + am)
        Yv = np.real(ap - am)
        packed = ut_cs2cep(Xu, Yu, Xv, Yv)
        coef["Lsmaj"], coef["Lsmin"], coef["theta"], coef["g"] = packed

    # Mean and trend.
    if opt["twodim"]:
        if opt["notrend"]:
            coef["umean"] = np.real(m[-1])
            coef["vmean"] = np.imag(m[-1])
        else:
            coef["umean"] = np.real(m[-2])
            coef["vmean"] = np.imag(m[-2])
            coef["uslope"] = np.real(m[-1]) / lor
            coef["vslope"] = np.imag(m[-1]) / lor
    else:
        if opt["notrend"]:
            coef["mean"] = np.real(m[-1])
        else:
            coef["mean"] = np.real(m[-2])
            coef["slope"] = np.real(m[-1]) / lor

    if opt.infer:
        # complex coefficients
        apI = np.empty((nI,), dtype=complex)
        amI = np.empty((nI,), dtype=complex)
        ind = 0

        for k, ref in enumerate(cnstit.R):
            apI[ind : ind + ref.nI] = ref.I.Rp * ap[nNR + k]
            amI[ind : ind + ref.nI] = ref.I.Rm * am[nNR + k]
            ind += ref.nI

        XuI = (apI + amI).real
        YuI = -(apI - amI).imag

        if not opt.twodim:
            A, _, _, g = ut_cs2cep(XuI, YuI)
            coef.A = np.hstack((coef.A, A))
            coef.g = np.hstack((coef.g, g))
        else:
            XvI = (apI + amI).imag
            YvI = (apI - amI).real
            Lsmaj, Lsmin, theta, g = ut_cs2cep(XuI, YuI, XvI, YvI)
            coef.Lsmaj = np.hstack((coef.Lsmaj, Lsmaj))
            coef.Lsmin = np.hstack((coef.Lsmin, Lsmin))
            coef.theta = np.hstack((coef.theta, theta))
            coef.g = np.hstack((coef.g, g))

    if opt["conf_int"]:
        coef = _confidence(
            coef,
            cnstit,
            opt,
            t,
            e,
            tin,
            elor,
            xraw,
            xmod,
            W,
            m,
            B,
            Xu,
            Yu,
            Xv,
            Yv,
        )

    # Diagnostics.
    if not opt["nodiagn"]:
        coef = ut_diagn(coef)
        # Adds a diagn dictionary, always sorted by energy.
        # This doesn't seem very useful.  Let's directly add the variables
        # to the base coef structure.  Then they can be sorted with everything
        # else.
        coef["PE"] = _PE(coef)
        coef["SNR"] = _SNR(coef)

    # Re-order constituents.
    coef = _reorder(coef, opt)
    # This might have added PE if it was not already present.

    if opt["RunTimeDisp"]:
        print("done.")

    return coef


def _reorder(coef, opt):
    if opt["ordercnstit"] == "PE":
        # Default: order by decreasing energy.
        if "PE" not in coef:
            coef["PE"] = _PE(coef)
        ind = coef["PE"].argsort()[::-1]

    elif opt["ordercnstit"] == "frequency":
        ind = coef["aux"]["frq"].argsort()

    elif opt["ordercnstit"] == "SNR":
        # If we are here, we should be guaranteed to have SNR already.
        ind = coef["SNR"].argsort()[::-1]
    else:
        namelist = list(coef["name"])
        ilist = [namelist.index(name) for name in opt["ordercnstit"]]
        ind = np.array(ilist, dtype=int)

    arrays = "name PE SNR A A_ci g g_ci Lsmaj Lsmaj_ci Lsmin Lsmin_ci theta theta_ci"
    reorderlist = [a for a in arrays.split() if a in coef]

    for key in reorderlist:
        coef[key] = coef[key][ind]

    coef["aux"]["frq"] = coef["aux"]["frq"][ind]
    coef["aux"]["lind"] = coef["aux"]["lind"][ind]
    return coef


def _slvinit(tin, uin, vin, lat, **opts):
    if lat is None:
        raise ValueError("Latitude must be supplied")

    # Supporting only 1-D arrays for now; we can add "group"
    # support later.
    if tin.shape != uin.shape or tin.ndim != 1 or uin.ndim != 1:
        raise ValueError("t and u must be 1-D arrays")

    if vin is not None and vin.shape != uin.shape:
        raise ValueError("v must have the same shape as u")

    opt = Bunch(twodim=(vin is not None))

    # Step 0: apply epoch to time.
    tin = _normalize_time(tin, opts["epoch"])

    # Step 1: remove invalid times from tin, uin, vin
    tin = np.ma.masked_invalid(tin)
    uin = np.ma.masked_invalid(uin)
    if vin is not None:
        vin = np.ma.masked_invalid(vin)
    if np.ma.is_masked(tin):
        goodmask = ~np.ma.getmaskarray(tin)
        uin = uin.compress(goodmask)
        if vin is not None:
            vin = vin.compress(goodmask)

    tin = tin.compressed()  # No longer masked.

    # Step 2: generate t, u, v from edited tin, uin, vin.
    v = None
    if np.ma.is_masked(uin) or np.ma.is_masked(vin):
        mask = np.ma.getmaskarray(uin)
        if vin is not None:
            mask = np.ma.mask_or(np.ma.getmaskarray(vin), mask)
        goodmask = ~mask
        t = tin.compress(goodmask)
        u = uin.compress(goodmask).filled()
        if vin is not None:
            v = vin.compress(goodmask).filled()
    else:
        t = tin
        u = uin.filled()
        if vin is not None:
            v = vin.filled()

    # Now t, u, v, tin are clean ndarrays; uin and vin are masked,
    # but don't necessarily have masked values.

    # Are the times equally spaced?
    eps = np.finfo(np.float64).eps
    if np.var(np.unique(np.diff(tin))) < eps:
        opt["equi"] = True  # based on times; u/v can still have nans ("gappy")
        lor = np.ptp(tin)
        ntgood = len(tin)
        elor = lor * ntgood / (ntgood - 1)
        tref = 0.5 * (tin[0] + tin[-1])
    else:
        opt["equi"] = False
        lor = np.ptp(t)
        nt = len(t)
        elor = lor * nt / (nt - 1)
        tref = 0.5 * (t[0] + t[-1])

    # Options.
    opt["conf_int"] = True
    opt["cnstit"] = "auto"
    opt["notrend"] = 0
    opt["prefilt"] = []
    opt["nodsatlint"] = 0
    opt["nodsatnone"] = 0
    opt["gwchlint"] = 0
    opt["gwchnone"] = 0
    opt["infer"] = None
    opt["inferaprx"] = 0
    opt["rmin"] = 1
    opt["method"] = "ols"
    opt["tunrdn"] = 1
    opt["linci"] = False
    opt["white"] = 0
    opt["nrlzn"] = 200
    opt["lsfrqosmp"] = 1
    opt["nodiagn"] = 0
    opt["diagnplots"] = 0
    opt["diagnminsnr"] = 2
    opt["ordercnstit"] = None
    opt["runtimedisp"] = "yyy"

    # Update the default opt dictionary with the kwargs,
    # ensuring that every kwarg key matches a key in opt.
    for key, item in opts.items():
        try:
            opt[key] = item
        except KeyError:
            print(f"solve: unrecognized input: {key}")

    return tin, t, u, v, tref, lor, elor, opt

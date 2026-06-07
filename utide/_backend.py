"""
Optional GPU (CuPy) array-module backend for UTide.

CPU (NumPy) is always available; the GPU path requires CuPy and a working
CUDA device, and is strictly opt-in via ``solve(..., gpu=True)``. This module
never imports cupy unless the GPU is actually requested, so importing utide on
a machine without CuPy has zero cost and zero risk.
"""

import numpy as np

_cupy = None
_checked = False
_import_error = None


def get_xp(gpu):
    """Return the array module to use: ``numpy`` for CPU, ``cupy`` for GPU.

    Raises a clear ``RuntimeError`` if ``gpu`` is True but CuPy / a CUDA
    device is not available.
    """
    global _cupy, _checked, _import_error
    if not gpu:
        return np
    if not _checked:
        _checked = True
        try:
            import cupy as cp

            # Touch the device so a missing driver/toolkit fails loudly here.
            cp.cuda.runtime.getDeviceCount()
            _cupy = cp
        except Exception as e:  # noqa: BLE001
            _cupy = None
            _import_error = e
    if _cupy is None:
        raise RuntimeError(
            "solve(gpu=True) requires CuPy with a working CUDA device, e.g. "
            "`pip install cupy-cuda12x`.  CuPy import / device check failed: "
            f"{_import_error!r}",
        )
    return _cupy


def asnumpy(a):
    """Return ``a`` as a host NumPy array, whether it is numpy or cupy."""
    if type(a).__module__.split(".")[0] == "cupy":
        return a.get()
    return np.asarray(a)


def is_gpu_array(a):
    return type(a).__module__.split(".")[0] == "cupy"

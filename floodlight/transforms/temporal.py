from copy import deepcopy
from typing import Union

import numpy as np
import scipy.interpolate

from floodlight import Code, XY
from floodlight.core.property import (
    BaseProperty,
    DyadicProperty,
    PlayerProperty,
    TeamProperty,
)


def _resample_array(
    values: np.ndarray,
    src_framerate: int,
    tgt_framerate: int,
    interp_method: Union[str, None],
    order: int,
    k: int,
) -> np.ndarray:
    """Core numerical resample routine.

    Parameters
    ----------
    values: np.ndarray
        Two-dimensional source array of shape ``(T, M)``. Callers flatten
        higher-rank payloads to ``(T, M)`` before calling.
    src_framerate, tgt_framerate: int
        Source and target framerates in Hz.
    interp_method: str or None
        Interpolation method for the upsample path; ignored on the identity
        and downsample paths.
    order, k: int
        Polynomial order / spline degree used by the respective methods.

    Returns
    -------
    out: np.ndarray
        Two-dimensional array of shape ``(N_out, M)``. Identity and
        downsample paths preserve ``values.dtype``; the upsample path
        returns ``float64`` to accommodate NaN.
    """
    # Empty-input short-circuit: bypass framerate math for T == 0 arrays.
    if values.shape[0] == 0:
        return values.copy()

    if tgt_framerate == src_framerate:
        resampled = deepcopy(values)
    elif tgt_framerate < src_framerate:
        resampled = _downsample(values, tgt_framerate, src_framerate)
    else:
        resampled = _upsample_columns(
            values, src_framerate, tgt_framerate, interp_method, order, k
        )

    return resampled


def _upsample_columns(
    values: np.ndarray,
    src_framerate: int,
    tgt_framerate: int,
    interp_method: Union[str, None],
    order: int,
    k: int,
) -> np.ndarray:
    """Per-column upsample helper.

    Parameters
    ----------
    values: np.ndarray
        Two-dimensional source array of shape ``(T, M)``.
    src_framerate, tgt_framerate: int
        Source and target framerates in Hz; ``tgt_framerate > src_framerate``.
    interp_method: str or None
        Interpolation method. ``None`` fills only target positions whose
        time coincides with a source time and leaves the rest NaN.
    order, k: int
        Polynomial order / spline degree used by the respective methods.

    Returns
    -------
    out: np.ndarray
        Two-dimensional array of shape ``(N_out, M)`` and dtype ``float64``.
    """
    N_src = values.shape[0]
    # Endpoint-inclusive length: last source sample aligns with last target sample.
    N_out = (N_src - 1) * tgt_framerate // src_framerate + 1
    M = values.shape[1]

    min_required = {
        None: 1,
        "linear": 2,
        "nearest": 1,
        "polynomial": order + 1,
        "spline": k + 1,
    }[interp_method]
    if N_src < min_required:
        raise ValueError(
            f"Method {interp_method!r} requires at least {min_required} source "
            f"samples, got {N_src}."
        )

    src_times = np.arange(N_src, dtype=np.int64) / src_framerate
    tgt_times = np.arange(N_out, dtype=np.int64) / tgt_framerate
    out = np.full((N_out, M), np.nan, dtype=np.float64)

    if interp_method is None:
        # Fill only targets coinciding with source times; leave rest NaN.
        aligned_targets = (np.arange(N_out) * src_framerate) % tgt_framerate == 0
        aligned_src_idx = (np.arange(N_out) * src_framerate) // tgt_framerate

    for j in range(M):
        column_f = np.asarray(values[:, j], dtype=np.float64)
        valid = ~np.isnan(column_f)
        if not np.any(valid):
            continue

        xp = src_times[valid]
        fp = column_f[valid]

        if interp_method is None:
            fill = aligned_targets & valid[aligned_src_idx]
            out[fill, j] = column_f[aligned_src_idx[fill]]
        elif interp_method == "linear":
            out[:, j] = np.interp(tgt_times, xp, fp, left=np.nan, right=np.nan)
        elif interp_method == "polynomial":
            if xp.size < order + 1:
                continue
            spl = scipy.interpolate.make_interp_spline(xp, fp, k=order)
            out[:, j] = spl(tgt_times, extrapolate=False)
        elif interp_method == "spline":
            if xp.size < k + 1:
                continue
            spl = scipy.interpolate.make_interp_spline(xp, fp, k=k)
            out[:, j] = spl(tgt_times, extrapolate=False)
        elif interp_method == "nearest":
            # Earlier-source tie-break at midpoints.
            idx_right = np.searchsorted(xp, tgt_times, side="left")
            idx_right = np.clip(idx_right, 1, xp.size - 1)
            idx_left = idx_right - 1
            d_left = tgt_times - xp[idx_left]
            d_right = xp[idx_right] - tgt_times
            choose_left = d_left <= d_right
            chosen = np.where(choose_left, idx_left, idx_right)
            in_range = (tgt_times >= xp[0]) & (tgt_times <= xp[-1])
            out[:, j] = np.where(in_range, fp[chosen], np.nan)

    return out


def _downsample(
    values: np.ndarray,
    tgt_framerate: int,
    src_framerate: int,
) -> np.ndarray:
    """Nearest-sample downsample helper.

    Parameters
    ----------
    values: np.ndarray
        Two-dimensional source array of shape ``(T, M)``.
    tgt_framerate, src_framerate: int
        Target and source framerates in Hz; ``tgt_framerate < src_framerate``.

    Returns
    -------
    out: np.ndarray
        Two-dimensional array of shape ``(N_out, M)``; preserves ``values.dtype``.
    """
    N_src = values.shape[0]
    N_out = (N_src - 1) * tgt_framerate // src_framerate + 1
    # +tgt_framerate//2 rounds to the nearest source index instead of flooring.
    src_idx = (np.arange(N_out) * src_framerate + tgt_framerate // 2) // tgt_framerate
    src_idx = np.clip(src_idx, 0, N_src - 1)
    resampled = values[src_idx].copy()
    return resampled


def resample(
    obj: Union[XY, Code, TeamProperty, PlayerProperty, DyadicProperty],
    target_framerate: int,
    interp_method: Union[str, None] = None,
    order: int = 3,
    k: int = 3,
) -> Union[XY, Code, TeamProperty, PlayerProperty, DyadicProperty]:
    """Resample a floodlight core object to a new framerate.

    Rescales any framerate-bearing core object (``XY``, ``Code``,
    ``TeamProperty``, ``PlayerProperty``, or ``DyadicProperty``) to a target framerate.

    Parameters
    ----------
    obj: XY | Code | TeamProperty | PlayerProperty | DyadicProperty
        Framerate-bearing core object to resample.
    target_framerate: int
        Target framerate in Hz. Must be a positive integer.
    interp_method : str or None, optional
        Interpolation method for the upsample path. Must be one of
        ``"linear"``, ``"polynomial"``, ``"spline"``, ``"nearest"`` or ``None``.
        Default ``None``: new rows are left NaN, only target samples that coincide with
        source samples take source values. Ignored on the identity and downsample paths.
    order: int, optional
        Polynomial order for ``interp_method="polynomial"``. Default 3.
    k: int, optional
        Spline degree for ``interp_method="spline"``. Default 3.

    Returns
    -------
    obj_resampled: same type as ``obj``
        New object with ``framerate`` rescaled to ``target_framerate``.

    Notes
    -----
    In case of ``target_framerate == obj.framerate`` ``resample()`` returns a deep copy
    of ``obj``.

    Pre-existing NaN gaps in the source pass through unchanged on both the
    upsample and downsample paths.

    The downsample path uses an integer-math nearest-sample index formula;
    ``interp_method`` is ignored there. No anti-aliasing filter is applied.
    """
    ALLOWED_METHODS = ("linear", "polynomial", "spline", "nearest")

    if not isinstance(obj, (XY, Code, BaseProperty)):
        raise ValueError(
            f"Expected obj to be one of (XY, Code, TeamProperty, "
            f"PlayerProperty, DyadicProperty), got {type(obj).__name__}."
        )

    if obj.framerate is None:
        raise ValueError("Expected obj.framerate to be set, got None.")

    if (
        not isinstance(target_framerate, (int, np.integer))
        or isinstance(target_framerate, bool)
        or target_framerate <= 0
    ):
        raise ValueError(
            f"Expected target_framerate to be a positive integer, "
            f"got {target_framerate!r}."
        )

    if interp_method is not None and interp_method not in ALLOWED_METHODS:
        raise ValueError(
            f"Expected interp_method to be None or one of {ALLOWED_METHODS}, "
            f"got {interp_method!r}."
        )

    if isinstance(obj, XY):
        resampled_array = _resample_array(
            obj.xy, obj.framerate, target_framerate, interp_method, order, k
        )
        resampled = XY(
            xy=resampled_array,
            framerate=target_framerate,
            direction=deepcopy(obj.direction),
        )
    elif isinstance(obj, Code):
        code_2d = np.asarray(obj.code).reshape(-1, 1)
        resampled_2d = _resample_array(
            code_2d, obj.framerate, target_framerate, interp_method, order, k
        )
        resampled = Code(
            code=resampled_2d.reshape(-1),
            name=deepcopy(obj.name),
            definitions=deepcopy(obj.definitions),
            framerate=target_framerate,
        )
    elif isinstance(obj, TeamProperty):
        prop_2d = np.asarray(obj.property).reshape(-1, 1)
        resampled_2d = _resample_array(
            prop_2d, obj.framerate, target_framerate, interp_method, order, k
        )
        resampled = TeamProperty(
            property=resampled_2d.reshape(-1),
            name=deepcopy(obj.name),
            framerate=target_framerate,
        )
    elif isinstance(obj, PlayerProperty):
        resampled_array = _resample_array(
            obj.property, obj.framerate, target_framerate, interp_method, order, k
        )
        resampled = PlayerProperty(
            property=resampled_array,
            name=deepcopy(obj.name),
            framerate=target_framerate,
        )
    elif isinstance(obj, DyadicProperty):
        prop = np.asarray(obj.property)
        T, N1, N2 = prop.shape
        prop_2d = prop.reshape(T, N1 * N2)
        resampled_2d = _resample_array(
            prop_2d, obj.framerate, target_framerate, interp_method, order, k
        )
        N_out = resampled_2d.shape[0]
        resampled = DyadicProperty(
            property=resampled_2d.reshape(N_out, N1, N2),
            name=deepcopy(obj.name),
            framerate=target_framerate,
        )

    return resampled

from typing import List

import numpy as np
import scipy.interpolate

from floodlight import XY


def _get_nan_gaps(data: np.ndarray, max_gap: int = None) -> np.ndarray:
    """Returns start and end indices of missing data gaps bounded by valid data.

    Parameters
    ----------
    data: np.ndarray
        Array of shape (T,) potentially containing missing values.
    max_gap: int, optional
        Maximum gap length to include. Gaps longer than this are excluded.
        If None, all bounded gaps are returned.

    Returns
    -------
    gaps: np.ndarray
        Two-dimensional array of shape (M, 2) and form
        ``[[gap_start_idx, gap_end_idx]]`` containing start and end indices of M
        missing data gaps bounded by valid data, ordered ascendingly. End indices are
        exclusive. Returns an empty array of shape (0, 2) if no bounded gaps are found.
    """
    # find change points between NaN and non-NaN
    is_nan = np.isnan(data)

    # no NaN or all NaN -> no gaps to fill
    if not np.any(is_nan) or np.all(is_nan):
        return np.empty((0, 2), dtype=int)

    # indices where nans and numbers are next to each other
    change_points = np.where(np.diff(is_nan, prepend=np.nan, append=np.nan))[0]
    sequences = np.array(
        [
            (change_points[i], change_points[i + 1])
            for i in range(len(change_points) - 1)
        ]
    )

    # keep only NaN sequences (check first element of each sequence)
    nan_sequences = [seq for seq in sequences if np.isnan(data[seq[0]])]

    if len(nan_sequences) == 0:
        return np.empty((0, 2), dtype=int)

    # first valid index and last valid index
    valid_indices = np.where(~is_nan)[0]
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]

    # keep only bounded gaps (not leading or trailing NaN)
    gaps = [
        seq for seq in nan_sequences if seq[0] > first_valid and seq[1] - 1 < last_valid
    ]

    # filter by max_gap
    if max_gap is not None:
        gaps = [seq for seq in gaps if seq[1] - seq[0] <= max_gap]

    if len(gaps) == 0:
        return np.empty((0, 2), dtype=int)

    return np.array(gaps)


def _interpolate_column(
    column: np.ndarray,
    method: str,
    order: int = 3,
    k: int = 3,
    max_gap: int = None,
) -> np.ndarray:
    """Interpolates missing data gaps in a 1D array using the specified method.

    Parameters
    ----------
    column: np.ndarray
        Array of shape (T,) potentially containing missing values.
    method: str
        Interpolation method. One of ``"linear"``, ``"polynomial"``, or ``"spline"``.
    order: int, optional
        Polynomial order. Only used when ``method="polynomial"``. Default is 3 (cubic).
    k: int, optional
        Spline degree. Only used when ``method="spline"``. Default is 3 (cubic).
    max_gap: int, optional
        Maximum gap length to interpolate. Gaps longer than this are left unchanged.

    Returns
    -------
    result: np.ndarray
        Array of shape (T,) with bounded gaps filled via the specified method.
    """
    result = column.copy()
    gaps = _get_nan_gaps(result, max_gap)

    if len(gaps) == 0:
        return result

    valid_mask = ~np.isnan(result)
    valid_idx = np.where(valid_mask)[0]
    valid_vals = result[valid_idx]

    # build interpolator based on method
    if method == "linear":
        gap_indices = np.concatenate([np.arange(start, end) for start, end in gaps])
        result[gap_indices] = np.interp(gap_indices, valid_idx, valid_vals)
        return result
    elif method == "polynomial":
        if len(valid_idx) < order + 1:
            return result
        interp_func = scipy.interpolate.interp1d(valid_idx, valid_vals, kind=order)
    elif method == "spline":
        if len(valid_idx) < k + 1:
            return result
        interp_func = scipy.interpolate.make_interp_spline(valid_idx, valid_vals, k=k)
    else:
        raise ValueError(
            f"Unknown interpolation method '{method}'. "
            "Expected 'linear', 'polynomial', or 'spline'."
        )

    # fill each gap
    gap_indices = np.concatenate([np.arange(s, e) for s, e in gaps])
    result[gap_indices] = interp_func(gap_indices)

    return result


def interpolate_linear(
    xy: XY,
    xIDs: List[int] = None,
    max_gap: int = None,
) -> XY:
    """Linearly interpolates gaps in XY tracking data along the temporal axis.

    For each column, missing data gaps that are bounded by valid data on both sides are
    filled using linear interpolation via `numpy.interp <https://numpy.org/doc/stable/
    reference/generated/numpy.interp.html>`_. Leading and trailing missing values are
    never interpolated.

    Parameters
    ----------
    xy: XY
        Floodlight XY Data object.
    xIDs: list of int, optional
        Player indices to interpolate. Each xID maps to two columns in the XY object
        (columns ``2 * xID`` and ``2 * xID + 1``). If None, all columns are
        interpolated. Default is None.
    max_gap: int, optional
        Maximum gap length (in frames) to interpolate. Gaps longer than this are
        left unchanged. If None, all bounded gaps are interpolated regardless of
        length. Default is None.

    Returns
    -------
    xy_interpolated: XY
        XY object with linearly interpolated position data.

    Notes
    -----
    Interpolation is strictly temporal (along axis 0). Each column is interpolated
    independently. Only gaps bounded by valid data on both sides are filled;
    leading (before the first valid value) and trailing (after the last valid value)
    missing values are preserved. This prevents extrapolation, e.g. for substituted
    players who are not yet on the field.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from floodlight import XY
    >>> from floodlight.transforms.interpolation import interpolate_linear

    Create an XY object with missing data gaps.

    >>> t = np.linspace(-5, 5, 1000)
    >>> player_x = np.sin(t) * t
    >>> player_x[200:220] = np.nan
    >>> player_x[500:530] = np.nan
    >>> player_y = t
    >>> xy = XY(np.transpose(np.stack((player_x, player_y))), framerate=20)

    Apply linear interpolation with default settings.

    >>> xy_interp = interpolate_linear(xy)
    >>> plt.plot(xy.x, 'o', markersize=2)
    >>> plt.plot(xy_interp.x, linewidth=2)
    >>> plt.legend(("Raw", "Interpolated"))
    >>> plt.show()

    .. image:: ../../_img/interpolate_linear_default_example.png

    Apply linear interpolation with a maximum gap length of 25 frames.

    >>> xy_interp = interpolate_linear(xy, max_gap=25)
    >>> plt.plot(xy.x, 'o', markersize=2)
    >>> plt.plot(xy_interp.x, linewidth=2)
    >>> plt.legend(("Raw", "Interpolated"))
    >>> plt.show()

    .. image:: ../../_img/interpolate_linear_max_gap_example.png
    """
    # handle empty XY
    if xy.xy.size == 0:
        return XY(xy=xy.xy.copy(), framerate=xy.framerate, direction=xy.direction)

    # determine columns to process
    if xIDs is not None:
        for xID in xIDs:
            if xID not in range(xy.N):
                raise ValueError(
                    f"xID {xID} is out of range. Expected 0 <= xID < {xy.N}."
                )
        cols = []
        for xID in xIDs:
            cols.extend([2 * xID, 2 * xID + 1])
    else:
        cols = list(range(xy.xy.shape[1]))

    # start with a float copy so non-selected columns are preserved
    xy_out = np.array(xy.xy, dtype=float)

    for i in cols:
        xy_out[:, i] = _interpolate_column(
            xy_out[:, i], method="linear", max_gap=max_gap
        )

    return XY(xy=xy_out, framerate=xy.framerate, direction=xy.direction)


def interpolate_polynomial(
    xy: XY,
    order: int = 3,
    xIDs: List[int] = None,
    max_gap: int = None,
) -> XY:
    """Interpolates gaps in XY tracking data using piecewise polynomial
    interpolation along the temporal axis.

    For each column, gaps that are bounded by valid data on both sides are filled
    using piecewise polynomial interpolation via `scipy.interpolate.interp1d
    <https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.interpolate.interp1d.html>`_. Leading and trailing missing values are not
    extrapolated.

    Parameters
    ----------
    xy: XY
        Floodlight XY Data object.
    order: int, optional
        Polynomial order for interpolation. Higher orders produce smoother curves but
        require more valid data points (at least ``order + 1``). Default is 3 (cubic).
    xIDs: list of int, optional
        Player indices to interpolate. Each xID maps to two columns in the XY object
        (columns ``2 * xID`` and ``2 * xID + 1``). If None, all columns are
        interpolated. Default is None.
    max_gap: int, optional
        Maximum gap length (in frames) to interpolate. Gaps longer than this are
        left unchanged. If None, all bounded gaps are interpolated regardless of
        length. Default is None.

    Returns
    -------
    xy_interpolated: XY
        XY object with polynomial-interpolated position data.

    Notes
    -----
    Interpolation is strictly temporal (along axis 0). Each column is interpolated
    independently. Only gaps bounded by valid data on both sides are filled;
    leading (before the first valid value) and trailing (after the last valid
    value) missing values are preserved. This prevents extrapolation, e.g. for
    substituted players who are not yet on the field.

    The interpolator is constructed from all valid data points in the column. If the
    total number of valid points is less than ``order + 1``, the column is returned
    unchanged. Interpolation quality depends on the density of valid data surrounding
    each gap.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from floodlight import XY
    >>> from floodlight.transforms.interpolation import interpolate_polynomial

    Create an XY object with missing data gaps.

    >>> t = np.linspace(-5, 5, 1000)
    >>> player_x = np.sin(t) * t
    >>> player_x[200:220] = np.nan
    >>> player_x[500:530] = np.nan
    >>> player_y = t
    >>> xy = XY(np.transpose(np.stack((player_x, player_y))), framerate=20)

    Apply polynomial interpolation with default settings (cubic).

    >>> xy_interp = interpolate_polynomial(xy)
    >>> plt.plot(xy.x, 'o', markersize=2)
    >>> plt.plot(xy_interp.x, linewidth=2)
    >>> plt.legend(("Raw", "Interpolated"))
    >>> plt.show()

    .. image:: ../../_img/interpolate_polynomial_default_example.png

    Apply polynomial interpolation with a maximum gap length of 25 frames.

    >>> xy_interp = interpolate_polynomial(xy, max_gap=25)
    >>> plt.plot(xy.x, 'o', markersize=2)
    >>> plt.plot(xy_interp.x, linewidth=2)
    >>> plt.legend(("Raw", "Interpolated"))
    >>> plt.show()

    .. image:: ../../_img/interpolate_polynomial_max_gap_example.png
    """
    # handle empty XY
    if xy.xy.size == 0:
        return XY(xy=xy.xy.copy(), framerate=xy.framerate, direction=xy.direction)

    # determine columns to process
    if xIDs is not None:
        for xID in xIDs:
            if xID not in range(xy.N):
                raise ValueError(
                    f"xID {xID} is out of range. Expected 0 <= xID < {xy.N}."
                )
        cols = []
        for xID in xIDs:
            cols.extend([2 * xID, 2 * xID + 1])
    else:
        cols = list(range(xy.xy.shape[1]))

    # start with a float copy so non-selected columns are preserved
    xy_out = np.array(xy.xy, dtype=float)

    for i in cols:
        xy_out[:, i] = _interpolate_column(
            xy_out[:, i], method="polynomial", order=order, max_gap=max_gap
        )

    return XY(xy=xy_out, framerate=xy.framerate, direction=xy.direction)


def interpolate_spline(
    xy: XY,
    k: int = 3,
    xIDs: List[int] = None,
    max_gap: int = None,
) -> XY:
    """Interpolates gaps in XY tracking data using spline interpolation along the
    temporal axis.

    For each column, gaps that are bounded by valid data on both sides are filled
    using spline interpolation via `scipy.interpolate.make_interp_spline
    <https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.interpolate.make_interp_spline.html>`_. Leading and trailing missing values
    are never interpolated.

    Parameters
    ----------
    xy: XY
        Floodlight XY Data object.
    k: int, optional
        Spline degree. Higher degrees produce smoother curves but require more valid
        data points (at least ``k + 1``). Default is 3 (cubic spline).
    xIDs: list of int, optional
        Player indices to interpolate. Each xID maps to two columns in the XY object
        (columns ``2 * xID`` and ``2 * xID + 1``). If None, all columns are
        interpolated. Default is None.
    max_gap: int, optional
        Maximum gap length (in frames) to interpolate. Gaps longer than this are
        left unchanged. If None, all bounded gaps are interpolated regardless of
        length. Default is None.

    Returns
    -------
    xy_interpolated: XY
        XY object with spline-interpolated position data.

    Notes
    -----
    Interpolation is strictly temporal (along axis 0). Each column is interpolated
    independently. Only gaps bounded by valid data on both sides are filled;
    leading (before the first valid value) and trailing (after the last valid
    value) missing values are preserved. This prevents extrapolation, e.g. for
    substituted players who are not yet on the field.

    Unlike piecewise polynomial interpolation, spline interpolation guarantees
    smoothness at the joints between segments (C2 continuity for cubic splines). This
    produces more realistic trajectories for tracking data.

    The interpolator is constructed from all valid data points in the column. If the
    total number of valid points is less than ``k + 1``, the column is returned
    unchanged. Interpolation quality depends on the density of valid data surrounding
    each gap.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from floodlight import XY
    >>> from floodlight.transforms.interpolation import interpolate_spline

    Create an XY object with missing data gaps.

    >>> t = np.linspace(-5, 5, 1000)
    >>> player_x = np.sin(t) * t
    >>> player_x[200:220] = np.nan
    >>> player_x[500:530] = np.nan
    >>> player_y = t
    >>> xy = XY(np.transpose(np.stack((player_x, player_y))), framerate=20)

    Apply spline interpolation with default settings (cubic).

    >>> xy_interp = interpolate_spline(xy)
    >>> plt.plot(xy.x, 'o', markersize=2)
    >>> plt.plot(xy_interp.x, linewidth=2)
    >>> plt.legend(("Raw", "Interpolated"))
    >>> plt.show()

    .. image:: ../../_img/interpolate_spline_default_example.png

    Apply spline interpolation with a maximum gap length of 25 frames.

    >>> xy_interp = interpolate_spline(xy, max_gap=25)
    >>> plt.plot(xy.x, 'o', markersize=2)
    >>> plt.plot(xy_interp.x, linewidth=2)
    >>> plt.legend(("Raw", "Interpolated"))
    >>> plt.show()

    .. image:: ../../_img/interpolate_spline_max_gap_example.png
    """
    # handle empty XY
    if xy.xy.size == 0:
        return XY(xy=xy.xy.copy(), framerate=xy.framerate, direction=xy.direction)

    # determine columns to process
    if xIDs is not None:
        for xID in xIDs:
            if xID not in range(xy.N):
                raise ValueError(
                    f"xID {xID} is out of range. Expected 0 <= xID < {xy.N}."
                )
        cols = []
        for xID in xIDs:
            cols.extend([2 * xID, 2 * xID + 1])
    else:
        cols = list(range(xy.xy.shape[1]))

    # start with a float copy so non-selected columns are preserved
    xy_out = np.array(xy.xy, dtype=float)

    for i in cols:
        xy_out[:, i] = _interpolate_column(
            xy_out[:, i], method="spline", k=k, max_gap=max_gap
        )

    return XY(xy=xy_out, framerate=xy.framerate, direction=xy.direction)

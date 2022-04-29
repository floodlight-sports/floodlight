import warnings
from floodlight.utils.types import Numeric

import numpy as np
import scipy.signal

from floodlight import XY


def _get_sequences_without_nans(data: np.ndarray) -> np.ndarray:
    """Returns start and end indices of continuous non-NaN sequences.

    Parameters
    ----------
    data: np.ndarray
        Array of shape (T x 1) potentially containing NaNs.

    Returns
    -------
    sequences: np.ndarray
        Two-dimensional array with (n x 2) entries of n sequences without NaNs of the
        form ``[[sequence_start_idx, sequence_end_idx]]`` ordered by the respective
        sequence start indices.
    """

    if data.ndim != 1:
        raise ValueError(
            f"Input data must be one-dimensional. Data has {data.ndim} dimension."
        )

    if None in data:
        data = np.array(data, dtype=float)
        warnings.warn(
            "Values in data of Type 'NoneType' were found. 'NoneType' values are "
            "treated as 'np.NaN'."
        )

    # indices where nans and numbers are next to each other
    change_points = np.where(np.diff(np.isnan(data), prepend=np.nan, append=np.nan))[0]
    # sequences without nans
    sequences = np.array(
        [
            (change_points[i], change_points[i + 1])
            for i in range(len(change_points) - 1)
        ]
    )

    is_nan = np.where(np.isnan(data[sequences[:, 0]]), False, True).reshape((-1, 1))
    labeled_sequences = np.hstack((sequences, is_nan))

    return labeled_sequences


def _filter_sequence_butterworth_lowpass(
    signal: np.ndarray, order: int = 3, cutoff: Numeric = 1, framerate: Numeric = None
) -> np.ndarray:
    """Filters a signal with a analog Butterworth low pass filter.

    Parameters
    ----------
    signal: np.ndarray
        Sequence of coordinates in a xy-Object.
    order: int
        The order of the filter.
    cutoff: Numeric
        The critical frequency. If ``framerate`` is not specified, the cutoff is
        normalized from 0 to 1, where 1 is the Nyquist frequency.
    framerate: Numeric, optional
        The sampling frequency of the data. If not specified, the ``cutoff`` is
        normalized from 0 to 1, where 1 is the Nyquist frequency.

    Returns
    -------
    singal_filtered: np.array
        Signal filtered by the Butterworth filter
    """

    coeffs = scipy.signal.butter(
        order,
        cutoff,
        btype="lowpass",
        output="ba",
        fs=framerate,
    )

    signal_filtered = scipy.signal.filtfilt(
        coeffs[0], coeffs[1], signal, method="pad", axis=0
    )

    return signal_filtered


def filter_xy_butterworth_lowpass(
    xy: XY,
    order: int = 3,
    cutoff: Numeric = 1,
    remove_short_seqs: bool = True,
) -> XY:
    """Filters the position data inside a XY-object with a digital Butterworth lowpass
    filter.

    Parameters
    ----------
    xy: XY
        List of XY-objects from any floodlight parser.
    order: int
        The order of the filter. Higher order will cut off the signal harder. Default is
        3.
    cutoff: Numeric
        The critical frequency. If ``framerate`` is not specified, the cutoff is
        normalized from 0 to 1, where 1 is the Nyquist frequency. Default is 1.
    remove_short_seqs: bool
        If True, sequences that are to short for the Filter with the specified settings
        are removed from the data. If False, they are kept unfiltered. Default is True


    Returns
    -------
    xy_filtered: XY
        XY object with position data filtered by designed Butterworth low pass filter.
    """

    min_signal_len = 3 * (order + 1)
    framerate = xy.framerate

    xy_filt = np.empty(xy.xy.shape)
    for i, column in enumerate(xy.xy.T):
        sequences = _get_sequences_without_nans(column)

        col_filt = np.full(column.shape, np.nan)

        for seq in sequences[sequences[:, 2] == 1]:
            if np.diff(seq[0:2]) > min_signal_len:
                col_filt[seq[0] : seq[1]] = _filter_sequence_butterworth_lowpass(
                    column[seq[0] : seq[1]], order, cutoff, framerate
                )

            elif remove_short_seqs is False:
                col_filt[seq[0] : seq[1]] = column[seq[0] : seq[1]]
            else:
                pass

        xy_filt[:, i] = col_filt

    xy_filtered = XY(xy=xy_filt, framerate=xy.framerate, direction=xy.direction)

    return xy_filtered


def filter_xy_savgol_lowpass(
    xy: XY,
    window_length: int = 5,
    poly_order: Numeric = 3,
    remove_short_seqs: bool = True,
) -> XY:
    """Filters the position data inside a XY-object with a Savitzky-Golay filter.
    filter.

    Parameters
    ----------
    xy: XY
        List of XY-objects from any floodlight parser.
    window_length: int
        The length of the filter window. Default is 5.
    poly_order: Numeric
        The order of the polynomial used to fit the samples. ``poly_order`` must be less
        than ``window_length``. Default is 3
    remove_short_seqs: bool
        If True, sequences that are to short for the Filter with the specified settings
        are removed from the data. If False, they are kept unfiltered. Default is True.


    Returns
    -------
    xy_filtered: XY
        XY object with position data filtered by designed Butterworth low pass filter.
    """

    min_signal_len = window_length

    xy_filt = np.empty(xy.xy.shape)
    for i, column in enumerate(xy.xy.T):
        sequences = _get_sequences_without_nans(column)

        col_filt = np.full(column.shape, np.nan)

        for seq in sequences[sequences[:, 2] == 1]:
            if np.diff(seq[0:2]) > min_signal_len:
                col_filt[seq[0] : seq[1]] = scipy.signal.savgol_filter(
                    column[seq[0] : seq[1]], window_length, poly_order
                )
            elif remove_short_seqs is False:
                col_filt[seq[0] : seq[1]] = column[seq[0] : seq[1]]
            else:
                pass

        xy_filt[:, i] = col_filt

    xy_filtered = XY(xy=xy_filt, framerate=xy.framerate, direction=xy.direction)

    return xy_filtered

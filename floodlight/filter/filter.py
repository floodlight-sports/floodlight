from typing import Dict
from floodlight.utils.types import Numeric

import numpy as np
import scipy.signal

from floodlight import XY


def _get_sequences_without_nans(data_column: np.ndarray) -> np.ndarray:
    """Gets indices of sequences without NaNs from a sequence of data.

    Parameters
    ----------
    data_column: np.ndarray
        One-dimensional array of data with alternating sequences of Numbers and NaNs

    Returns
    -------
    sequences: np.ndarray
        Two-dimensional array with nx2 entrys of n sequences without NaNs and n[0] being
        the first frame of the sequences and n[1] being the last frame (excluded) of the
        sequence.
    """

    # check if even or odd sequences contain NaNs
    if np.isnan(data_column[0]):
        first_sequence = 1
    else:
        first_sequence = 0

    # indices where nans and numbers are next to each other
    change_points = np.where(
        np.diff(np.isnan(data_column), prepend=np.nan, append=np.nan)
    )[0]
    # sequences without nans
    sequences = np.array(
        [
            (change_points[i], change_points[i + 1])
            for i in range(len(change_points) - 1)
        ]
    )[first_sequence::2]

    return sequences


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
        The sampling frequency of the data.

    Returns
    -------
    filt_signal: np.array
        Signal filtered by the Butterworth filter
    """

    coeffs = scipy.signal.butter(
        order,
        cutoff,
        btype="lowpass",
        output="ba",
        fs=framerate,
    )

    filt_signal = scipy.signal.filtfilt(
        coeffs[0], coeffs[1], signal, method="pad", axis=0
    )

    return filt_signal


def filter_xy_butterworth_lowpass(
    xy: XY,
    team_links: Dict[str, int],
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
    team_links: Dict[str, int]
        Link-dictionary of the form ``links[identifier-ID] = xID``.
    order: int
        The order of the filter.
    cutoff: Numeric
        The critical frequency. If ``framerate`` is not specified, the cutoff is
        normalized from 0 to 1, where 1 is the Nyquist frequency.
    remove_short_seqs: bool
        If True, sequences that are to short for the Filter with the specified settings
        are removed from the data. If False, they are kept unfiltered


    Returns
    -------
    xy_filtered: XY
        XY object with position data filtered by designed Butterworth low pass filter.
    """

    min_signal_len = 3 * (order + 1)
    framerate = xy.framerate

    xy_filt = np.empty(xy.xy.shape)
    for player in team_links:
        # players x-index in xy-Object
        pos_idx = team_links[player] * 2
        pos_idy = team_links[player] * 2 + 1

        data_col_x = xy[:, pos_idx]
        data_col_y = xy[:, pos_idy]

        sequences = _get_sequences_without_nans(data_col_x)

        col_filt_x = np.full(data_col_x.shape, np.nan)
        col_filt_y = np.full(data_col_y.shape, np.nan)

        for seq in sequences:
            if np.diff(seq) > min_signal_len:
                col_filt_x[seq[0] : seq[1]] = _filter_sequence_butterworth_lowpass(
                    data_col_x[seq[0] : seq[1]], order, cutoff, framerate
                )
                col_filt_y[seq[0] : seq[1]] = _filter_sequence_butterworth_lowpass(
                    data_col_y[seq[0] : seq[1]], order, cutoff, framerate
                )
            elif remove_short_seqs is False:
                col_filt_x[seq[0] : seq[1]] = data_col_x[seq[0], seq[1]]
                col_filt_y[seq[0] : seq[1]] = data_col_y[seq[0], seq[1]]
            else:
                pass

        xy_filt[:, pos_idx] = col_filt_x
        xy_filt[:, pos_idy] = col_filt_y

    xy_filtered = XY(xy=xy_filt, framerate=xy.framerate, direction=xy.direction)

    return xy_filtered


def filter_xy_savgol_lowpass(
    xy: XY,
    team_links: Dict[str, int],
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
    team_links: Dict[str, int]
        Link-dictionary of the form ``links[identifier-ID] = xID``.
    window_length: int
        The length of the filter window.
    poly_order: Numeric
        The order of the polynomial used to fit the samples. ``poly_order`` must be less
        than ``window_length``
    remove_short_seqs: bool
        If True, sequences that are to short for the Filter with the specified settings
        are removed from the data. If False, they are kept unfiltered


    Returns
    -------
    xy_filtered: XY
        XY object with position data filtered by designed Butterworth low pass filter.
    """

    min_signal_len = window_length

    xy_filt = np.empty(xy.xy.shape)
    for player in team_links:
        # players x-index in xy-Object
        pos_idx = team_links[player] * 2
        pos_idy = team_links[player] * 2 + 1

        data_col_x = xy[:, pos_idx]
        data_col_y = xy[:, pos_idy]

        sequences = _get_sequences_without_nans(data_col_x)

        col_filt_x = np.full(data_col_x.shape, np.nan)
        col_filt_y = np.full(data_col_y.shape, np.nan)

        for seq in sequences:
            if np.diff(seq) > min_signal_len:
                col_filt_x[seq[0] : seq[1]] = scipy.signal.savgol_filter(
                    data_col_x[seq[0] : seq[1]], window_length, poly_order
                )
                col_filt_y[seq[0] : seq[1]] = scipy.signal.savgol_filter(
                    data_col_y[seq[0] : seq[1]], window_length, poly_order
                )
            elif remove_short_seqs is False:
                col_filt_x[seq[0] : seq[1]] = data_col_x[seq[0], seq[1]]
                col_filt_y[seq[0] : seq[1]] = data_col_y[seq[0], seq[1]]
            else:
                pass

        xy_filt[:, pos_idx] = col_filt_x
        xy_filt[:, pos_idy] = col_filt_y

    xy_filtered = XY(xy=xy_filt, framerate=xy.framerate, direction=xy.direction)

    return xy_filtered

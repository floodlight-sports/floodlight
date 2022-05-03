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
        form `[[sequence_start_idx, sequence_end_idx]]` ordered by the respective
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
    """Wrapper for the `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/
    generated/scipy.signal.butter.html>`__ and `scipy.signal.filtfilt <https://docs.scip
    y.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html>`__ functions.
    Filters the incoming data with da digital Butterworth lowpass filter.

    Parameters
    ----------
    signal: np.ndarray
        Sequence of coordinates in a xy-Object.
    order: int
        The order of the filter. This is the argument ``N`` from the `scipy.signal.
        butter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter
        .html>`__ function.
    cutoff: Numeric
        The critical cutoff frequency. This is the argument ``Wn`` from the
        `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/generated/scipy
        .signal.butter.html>`__ function.
    framerate: Numeric, optional
        The sampling frequency of the data. This is the argument ``fs`` from the
        `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.butter.html>`__ function.

    Returns
    -------
    signal_filtered: np.array
        Signal filtered by the Butterworth filter.
    """
    # Calculation of filter coefficients
    coeffs = scipy.signal.butter(
        order,
        cutoff,
        btype="lowpass",
        output="ba",
        fs=framerate,
    )
    # applying the filter to the data
    signal_filtered = scipy.signal.filtfilt(
        coeffs[0], coeffs[1], signal, method="pad", axis=0
    )

    return signal_filtered


def butterworth_lowpass(
    xy: XY,
    order: int = 3,
    cutoff: Numeric = 1,
    remove_short_seqs: bool = True,
) -> XY:
    """Filters the position data inside a XY-object with a digital Butterworth lowpass-
    filter [1]_. This is a wrapper for the `scipy.filter.butter
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`_
    and the `scipy.signal.filtfilt <https://docs.scipy.org/doc/scipy/reference/
    generated/scipy.signal.filtfilt.html>`_ functions.

    Parameters
    ----------
    xy: XY
        Floodlight.XY-object.
    order: int
        The order of the filter. This is the argument ``N`` from the `scipy.signal.
        butter <https://docs.scipy.org/doc/scipy/reference/generated/ scipy.signal.
        butter.html>`_ function.
    cutoff: Numeric
        The critical cutoff frequency. This is the argument ``Wn`` from the
        `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/generated/
        scipy.signal.butter.html>`_
        function.
    remove_short_seqs: bool
        If True, sequences that are to short for the filter with the specified settings
        are replaced with np.NaNs. If False, they are kept unfiltered. Default is True.

    Returns
    -------
    xy_filtered: XY
        XY object with position data filtered by designed Butterworth low pass filter.

    Notes
    -----
    The values of the input data are assumed to be numerical. Missing data is assumed
    to be either np.NaN or None. The Butterworth-filter requires a minimum signal length
    depending on the settings. A signal is a sequence of data in the XY-object that is
    not interrupted by missing values. The minimum signal length is defined as
    :math:`3 \\cdot (order + 1)`. The treatment of signals shorter than the minimum
    signal length are specified with the ``remove_short_sequence``-argument, where True
    will replace these sequences with np.NaNs ond False will keep the sequences in the
    data unfiltered.

    References
    ----------
        .. [1] `Butterworth, S. (1930). On the theory of filter amplifiers. Wireless
            Engineer, 3, 536-541. <https://www.changpuak.ch/electronics/downloads/
            On_the_Theory_of_Filter_Amplifiers.pdf>`_
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


def savgol_lowpass(
    xy: XY,
    window_length: int = 5,
    poly_order: Numeric = 3,
    remove_short_seqs: bool = True,
) -> XY:
    """Filters the position data inside a XY-object with a Savitzky-Golay-filter [2]_.
    This is a wrapper for the `scipy.filter.savgol
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`__
    function.

    Parameters
    ----------
    xy: XY
        Floodlight.XY-object.
    window_length: int
        The length of the filter window. This is the argument ``window_length`` from the
        `scipy.filter.savgol <https://docs.scipy.org/doc/scipy/reference/generated/
        scipy.signal.savgol_filter.html>`__ function. Default is 5.
    poly_order: Numeric
        The order of the polynomial used to fit the samples. ``poly_order`` must be less
        than ``window_length``. Default is 3. This is the argument ``polyorder`` from
        the `scipy.filter.savgol <https://docs.scipy.org/doc/scipy/reference/generated/
        scipy.signal.savgol_filter.html>`__ function. Default is 5.
    remove_short_seqs: bool
        If True, sequences that are to short for the Filter with the specified settings
        are removed from the data. If False, they are kept unfiltered. Default is True.

    Returns
    -------
    xy_filtered: XY
        XY object with position data filtered by designed Butterworth low pass filter.

    Notes
    -----
    The values of the input data are assumed to be numerical. Missing data is assumed
    to be either np.NaN or None. The Savitzky-Golay-filter requires a minimum signal
    length depending on the settings. A signal is a sequence of data in the XY-object
    that is not interrupted by missing values. The minimum signal length is defined as
    the ``window_length``. The treatment of signals shorter than the minimum signal
    length are specified with the ``remove_short_sequence``-argument, where True will
    replace these sequences with np.NaNs ond False will keep the sequences in the data
    unfiltered.

    References
    ----------
        .. [2] `Savitzky, A.; Golay, M.J. (1964). Smoothing and differentiation of data
            by simplified least squares procedures. Analytical Chemistry, 36(1), 1627-
            1639. <https://pubs.acs.org/doi/abs/10.1021/ac60214a047>`_

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

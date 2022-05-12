import scipy.signal
import numpy as np

from floodlight import XY
from floodlight.utils.types import Numeric


def _get_sequences_without_nans(data: np.ndarray) -> np.ndarray:
    """Returns start and end indices of continuous non-NaN sequences.

    Parameters
    ----------
    data: np.ndarray
        Array of shape (T x 1) potentially containing NaNs.

    Returns
    -------
    sequences: np.ndarray
        Two-dimensional array of shape (N x 3) and form
        ``[[sequence_start_idx, sequence_end_idx, is_sequence_nan]]`` containing start
        and end indices of N alternating sequences of NaN and non-NaN entries of the
        original data, ordered ascendingly, along with a bool that indicates if that
        sequence contains NaN (False) of non-NaN (True) values.
    """
    if data.ndim != 1:
        raise ValueError(
            f"Expected input data to be one-dimensional. Got {data.ndim}-dimensional"
            f"data instead."
        )

    # Convert possible None-types in data to np.NaN
    data = np.array(data, dtype=float)

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
    """Filters the incoming signal with a digital Butterworth lowpass filter.

    Wrapper for combined application of the `scipy.signal.butter <https://docs.scipy.
    org/doc/scipy/reference/generated/scipy.signal.butter.html>`__ and `scipy.signal.
    filtfilt <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.
    filtfilt.html>`_ functions.

    Parameters
    ----------
    signal: np.ndarray
        Array of shape (T, N) containing the signal to be smoothed with T frames and N
        independent signals. Corresponds to the argument ``x`` from the `scipy.signal.
        filtfilt <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.
        filtfilt.html>`_ function.
    order: int, optional
        The order of the filter. Corresponds to the argument ``N`` from the `scipy.
        signal. butter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.
        signal.butter.html>`_ function. Default is 3.
    cutoff: Numeric, optional
        The critical cutoff frequency. Corresponds to the argument ``Wn`` from the
        `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/generated/scipy
        .signal.butter.html>`_ function. Default is 1.
    framerate: Numeric, optional
        The sampling frequency of the signal. Corresponds to the argument ``fs`` from
        the `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.butter.html>`_ function.

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
    """Applies a digital Butterworth lowpass-filter [1]_ to a XY data object.

    For filtering, the `scipy.filter.butter <https://docs.scipy.org/doc/scipy/reference/
    generated/scipy.signal.butter.html>`_ and the `scipy.signal.filtfilt <https://docs.
    scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html>`_ functions are
    used. This function provides a convenience access to both functions, directly
    applying the filter to all non-NaN sequences in all columns.

    Parameters
    ----------
    xy: XY
        Floodlight XY Data object.
    order: int, optional
        The order of the filter. Corresponds to the argument ``N`` from the `scipy.
        signal.butter <https://docs.scipy.org/doc/scipy/reference/generated/ scipy.
        signal.butter.html>`_ function. Default is 3
    cutoff: Numeric, optional
        The critical cutoff frequency. Corresponds to the argument ``Wn`` from the
        `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/generated/
        scipy.signal.butter.html>`_ function. Default is 1.
    remove_short_seqs: bool, optional
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
    # minimum signal length a filter with this specs can be applied on
    min_signal_len = 3 * (order + 1)
    framerate = xy.framerate

    # pre-allocate space for filtered data
    xy_filt = np.empty(xy.xy.shape)
    # loop through the xy-object columns
    for i, column in enumerate(np.transpose(xy.xy)):
        # extract indices of alternating NaN/non-NaN sequences
        sequences = _get_sequences_without_nans(column)
        # pre-allocate space for filtered column
        col_filt = np.full(column.shape, np.nan)

        # loop through filterable sequences
        for start, end, _ in sequences[
            np.all(
                (
                    sequences[:, 2] == 1,
                    np.squeeze(np.diff(sequences[:, 0:2]) >= min_signal_len),
                ),
                axis=0,
            )
        ]:
            # apply filter to the sequence and enter filtered data to their
            # respective indices in the data
            col_filt[start:end] = _filter_sequence_butterworth_lowpass(
                column[start:end], order, cutoff, framerate
            )
        # check treatment of sequences that don't meet minimum signal length
        if remove_short_seqs is False:
            # enter short sequences unfiltered to their respective indices in the data
            for start, end, _ in sequences[
                np.all(
                    (
                        sequences[:, 2] == 1,
                        np.squeeze(np.diff(sequences[:, 0:2]) < min_signal_len),
                    ),
                    axis=0,
                )
            ]:
                col_filt[start:end] = column[start:end]

        # enter filtered data into respective column
        xy_filt[:, i] = col_filt

    # create new XY-data object with filtered data
    xy_filtered = XY(xy=xy_filt, framerate=xy.framerate, direction=xy.direction)

    return xy_filtered


def savgol_lowpass(
    xy: XY,
    window_length: int = 5,
    poly_order: Numeric = 3,
    remove_short_seqs: bool = True,
) -> XY:
    """Applies a Savitzky-Golay lowpass-filter [2]_ to a XY data object.

    For filtering, the `scipy.filter.savgol <https://docs.scipy.org/doc/scipy/reference/
    generated/scipy.signal.butter.html>`_ function is used. This function provides a
    convenient access to the function, directly applying the filter to all non-NaN
    sequences in all columns.

    Parameters
    ----------
    xy: XY
        Floodlight XY Data object.
    window_length: int, optional
        The length of the filter window. Corresponds to the argument ``window_length``
        from the `scipy.filter.savgol <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.savgol_filter.html>`_ function. Default is 5.
    poly_order: Numeric, optional
        The order of the polynomial used to fit the samples. ``poly_order`` must be less
        than ``window_length``. Default is 3. Corresponds to the argument ``polyorder``
        from the `scipy.filter.savgol <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.savgol_filter.html>`_ function. Default is 5.
    remove_short_seqs: bool, optional
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
    # minimum signal length a filter with this specs can be applied on
    min_signal_len = window_length

    # pre-allocate space for filtered data
    xy_filt = np.empty(xy.xy.shape)
    # loop through the xy-object columns
    for i, column in enumerate(np.transpose(xy.xy)):
        # extract indices of alternating NaN/non-NaN sequences
        sequences = _get_sequences_without_nans(column)
        # pre-allocate space for filtered column
        col_filt = np.full(column.shape, np.nan)

        # loop through filterable sequences
        for start, end, _ in sequences[
            np.all(
                (
                    sequences[:, 2] == 1,
                    np.squeeze(np.diff(sequences[:, 0:2]) >= min_signal_len),
                ),
                axis=0,
            )
        ]:
            # apply filter to the sequence and enter filtered data to their
            # respective indices in the data
            col_filt[start:end] = scipy.signal.savgol_filter(
                column[start:end], window_length, poly_order
            )
        # check treatment of sequences that don't meet minimum signal length
        if remove_short_seqs is False:
            # enter short sequences unfiltered to their respective indices in the data
            for start, end, _ in sequences[
                np.all(
                    (
                        sequences[:, 2] == 1,
                        np.squeeze(np.diff(sequences[:, 0:2]) < min_signal_len),
                    ),
                    axis=0,
                )
            ]:
                col_filt[start:end] = column[start:end]

        # enter filtered data into respective column
        xy_filt[:, i] = col_filt

    # create new XY-data object with filtered data
    xy_filtered = XY(xy=xy_filt, framerate=xy.framerate, direction=xy.direction)

    return xy_filtered

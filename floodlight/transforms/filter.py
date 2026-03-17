from typing import Tuple

import scipy.signal
import numpy as np

from floodlight import XY
from floodlight.utils.types import Numeric


def _get_filterable_and_short_sequences(
    data: np.ndarray, min_signal_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns start and end indices of continuous, filterable sequences and sequences
    to short for filtering with the specified filter.

    Parameters
    ----------
    data: np.ndarray
        Array of shape (T,) potentially containing NaNs.
    min_signal_len: int
        The minimum signal length that the specified filter can be applied on.

    Returns
    -------
    filterable_sequences: np.ndarray
        Two-dimensional array of shape (N, 2) and form
        ``[[sequence_start_idx, sequence_end_idx]]`` containing start and end indices of
         N filterable sequences in the original data, ordered ascendingly. A sequence is
         filterable when it doesn't contain NaNs and is at least as long as the minimum
         window length of the specified filter.
    short_sequences: np.ndarray
        Two-dimensional array of shape (N, 2) and form
        ``[[sequence_start_idx, sequence_end_idx]]`` containing start and end indices of
         N sequences in the original data that don't contain NaNs but are to short to
         apply the specified filter on.
    """
    if data.ndim != 1:
        raise ValueError(
            f"Expected input data to be one-dimensional. Got {data.ndim}-dimensional "
            f"data instead."
        )

    # Convert possible None-types in data to np.NaN
    data = np.array(data, dtype=float)

    # indices where nans and numbers are next to each other
    change_points = np.where(np.diff(np.isnan(data), prepend=np.nan, append=np.nan))[0]
    sequences = np.array(
        [
            (change_points[i], change_points[i + 1])
            for i in range(len(change_points) - 1)
        ]
    )

    # which sequences contain NaNs
    seq_is_nan = np.where(np.isnan(data[sequences[:, 0]]), False, True)
    # remove sequences containing NaNs
    non_nan_sequences = sequences[seq_is_nan == 1]
    # split remaining sequences into filterable and short
    filterable_sequences = non_nan_sequences[
        (non_nan_sequences[:, 1] - non_nan_sequences[:, 0]) > min_signal_len
    ]
    short_sequences = non_nan_sequences[
        (non_nan_sequences[:, 1] - non_nan_sequences[:, 0]) <= min_signal_len
    ]

    return filterable_sequences, short_sequences


def _filter_sequence_butterworth_lowpass(
    signal: np.ndarray,
    order: int = 3,
    Wn: Numeric = 1,
    framerate: Numeric = None,
    **kwargs,
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
    Wn: Numeric, optional
        The critical cutoff frequency. Corresponds to the argument ``Wn`` from the
        `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/generated/scipy
        .signal.butter.html>`_ function. Default is 1.
    framerate: Numeric, optional
        The sampling frequency of the signal. Corresponds to the argument ``fs`` from
        the `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.butter.html>`_ function.
    kwargs:
        Optional arguments {'padtype', 'padlen', 'method', 'irlen'} that can be passed
        to the `scipy.signal.filtfilt <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.filtfilt.html>`_ function.

    Returns
    -------
    signal_filtered: np.array
        Signal filtered by the Butterworth filter.
    """
    # Calculation of filter coefficients
    coeffs = scipy.signal.butter(
        order,
        Wn,
        btype="lowpass",
        output="ba",
        fs=framerate,
    )
    # applying the filter to the data
    signal_filtered = scipy.signal.filtfilt(
        coeffs[0], coeffs[1], signal, axis=0, **kwargs
    )

    return signal_filtered


def butterworth_lowpass(
    xy: XY, order: int = 3, Wn: Numeric = 1, remove_short_seqs: bool = False, **kwargs
) -> XY:
    """Applies a digital Butterworth lowpass-filter to a XY data object. [1]_

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
        signal.butter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.
        signal.butter.html>`_ function. Default is 3
    Wn: Numeric, optional
        The normalized critical cutoff frequency. Corresponds to the argument ``Wn``
        from the `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.butter.html>`_ function. Default is 1.
    remove_short_seqs: bool, optional
        If True, sequences that are to short for the filter with the specified settings
        are replaced with np.NaNs. If False, they are kept unfiltered. Default is False.
    kwargs:
        Optional arguments {'padtype', 'padlen', 'method', 'irlen'} that can be passed
        to the `scipy.signal.filtfilt <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.filtfilt.html>`_ function.
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

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from floodlight import XY
    >>> from floodlight.transforms.filter import butterworth_lowpass

    We first generate a noisy XY-object to smooth.

    >>> t = np.linspace(-5, 5, 1000)
    >>> player_x = np.sin(t) * t + np.random.rand(1000)
    >>> player_x[450:495] = np.NaN
    >>> player_x[505:550] = np.NaN
    >>> player_y = t + np.random.randn()
    >>> xy = XY(np.transpose(np.stack((player_x, player_y))), framerate=20)

    Apply the Butterworth lowpass filter with its default settings.

    >>> xy_filt = butterworth_lowpass(xy)
    >>> plt.plot(xy.x)
    >>> plt.plot(xy_filt.x, linewidth=3)
    >>> plt.legend(("Raw", "Smoothed"))
    >>> plt.show()

    .. image:: ../../_img/butterworth_default_example.png


    Apply the same filter but remove the sequence that is to short to filter.

    >>> xy_filt = butterworth_lowpass(xy, remove_short_seqs=True)
    >>> plt.plot(xy.x)
    >>> plt.plot(xy_filt.x, linewidth=3)
    >>> plt.legend(("Raw", "Smoothed"))
    >>> plt.show()

    .. image:: ../../_img/butterworth_removed_short_example.png

    Apply the filter with different specifications.

    >>> xy_filt = butterworth_lowpass(xy, order=5, Wn=4)
    >>> plt.plot(xy.x)
    >>> plt.plot(xy_filt.x, linewidth=3)
    >>> plt.legend(("Raw", "Smoothed"))
    >>> plt.show()

    .. image:: ../../_img/butterworth_adjusted_example.png

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
        # extract indices of filterable and short sequences
        seqs_filt, seqs_short = _get_filterable_and_short_sequences(
            column, min_signal_len
        )
        # pre-allocate space for filtered column
        col_filt = np.full(column.shape, np.nan)

        # loop through filterable sequences
        for start, end in seqs_filt:
            # apply filter to the sequence and enter filtered data to their
            # respective indices in the data
            col_filt[start:end] = _filter_sequence_butterworth_lowpass(
                column[start:end], order, Wn, framerate, **kwargs
            )
        # check treatment of sequences that don't meet minimum signal length
        if remove_short_seqs is False:
            # enter short sequences unfiltered to their respective indices in the data
            for start, end in seqs_short:
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
    remove_short_seqs: bool = False,
    **kwargs,
) -> XY:
    """Applies a Savitzky-Golay lowpass-filter to a XY data object. [2]_

    For filtering, the `scipy.filter.savgol <https://docs.scipy.org/doc/scipy/reference/
    generated/scipy.signal.savgol_filter.html>`_ function is used. This function
    provides a convenient access to the function, directly applying the filter to all
    non-NaN sequences in all columns.

    Parameters
    ----------
    xy: XY
        Floodlight XY Data object.
    window_length: int, optional
        The length of the filter window. Corresponds to the argument ``window_length``
        from the `scipy.filter.savgol <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.savgol_filter.html>`_ function. Default is 5.
    polyorder: Numeric, optional
        The order of the polynomial used to fit the samples. ``poly_order`` must be less
        than ``window_length``. Default is 3. Corresponds to the argument ``polyorder``
        from the `scipy.filter.savgol <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.savgol_filter.html>`_ function. Default is 5.
    remove_short_seqs: bool, optional
        If True, sequences that are to short for the Filter with the specified settings
        are removed from the data. If False, they are kept unfiltered. Default is False.
    kwargs:
        Optional arguments {'deriv', 'delta', 'mode', 'cval'} that can be passed to
        the `scipy.signal.savgol <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.savgol_filter.html>`_ function.

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

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from floodlight import XY
    >>> from floodlight.transforms.filter import savgol_lowpass

    We first generate a noisy XY-object to smooth.

    >>> t = np.linspace(-5, 5, 1000)
    >>> player_x = np.sin(t) * t + np.random.rand(1000)
    >>> player_x[450:495] = np.NaN
    >>> player_x[505:550] = np.NaN
    >>> player_y = t + np.random.randn()
    >>> xy = XY(np.transpose(np.stack((player_x, player_y))), framerate=20)

    Apply the Savgol lowpass filter with its default settings.

    >>> xy_filt = savgol_lowpass(xy)
    >>> plt.plot(xy.x)
    >>> plt.plot(xy_filt.x, linewidth=3)
    >>> plt.legend(("Raw", "Smoothed"))
    >>> plt.show()

    .. image:: ../../_img/savgol_default_example.png


    Apply the filter with a longer window lengh and remove the sequence that is to short
    to filter.

    >>> xy_filt = savgol_lowpass(xy, window_length=12, remove_short_seqs=True)
    >>> plt.plot(xy.x)
    >>> plt.plot(xy_filt.x, linewidth=3)
    >>> plt.legend(("Raw", "Smoothed"))
    >>> plt.show()

    .. image:: ../../_img/savgol_removed_short_example.png


    Apply the filter with different specifications.

    >>> xy_filt = savgol_lowpass(xy, window_length=50, poly_order=5)
    >>> plt.plot(xy.x)
    >>> plt.plot(xy_filt.x, linewidth=3)
    >>> plt.legend(("Raw", "Smoothed"))
    >>> plt.show()

    .. image:: ../../_img/savgol_adjusted_example.png

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
        seqs_filt, seqs_short = _get_filterable_and_short_sequences(
            column, min_signal_len
        )
        # pre-allocate space for filtered column
        col_filt = np.full(column.shape, np.nan)

        # loop through filterable sequences
        for start, end in seqs_filt:
            # apply filter to the sequence and enter filtered data to their
            # respective indices in the data
            col_filt[start:end] = scipy.signal.savgol_filter(
                column[start:end], window_length, poly_order, **kwargs
            )
        # check treatment of sequences that don't meet minimum signal length
        if remove_short_seqs is False:
            # enter short sequences unfiltered to their respective indices in the data
            for start, end in seqs_short:
                col_filt[start:end] = column[start:end]

        # enter filtered data into respective column
        xy_filt[:, i] = col_filt

    # create new XY-data object with filtered data
    xy_filtered = XY(xy=xy_filt, framerate=xy.framerate, direction=xy.direction)

    return xy_filtered


def _kalman_filter_1d(
    signal: np.ndarray,
    dt: float,
    process_noise: float = 1.0,
    measurement_noise: float = 0.04,
) -> np.ndarray:
    """Applies a forward-only 1D Kalman filter with a constant-velocity model.

    The state vector is ``[position, velocity]`` and only position is observed.
    NaN values in the input are handled natively: when an observation is missing,
    only the predict step runs (covariance grows), and the output remains NaN.
    The filter state is preserved across gaps so that velocity and uncertainty
    information carries over when observations resume.

    Parameters
    ----------
    signal: np.ndarray
        Array of shape (T,) containing the signal to be smoothed. May contain
        NaNs (missing observations).
    dt: float
        Time step between frames in seconds (1 / framerate).
    process_noise: float, optional
        Process noise intensity (acceleration variance in m²/s⁴) that controls
        how much the model trusts the constant-velocity prediction. Larger
        values allow faster changes in velocity. Default corresponds to
        :math:`\\sigma_a = 1\\,\\mathrm{m/s^2}`, which is a conservative smoothing
        prior.
    measurement_noise: float, optional
        Measurement noise variance (in m²) that controls how much the model
        trusts the observed positions. Larger values produce smoother output.
        Default is 0.04, corresponding to 0.20 m (20 cm) RMSE, a conservative
        estimate for common local or optical tracking systems during high-dynamic
        situations.

    Returns
    -------
    signal_filtered: np.ndarray
        Filtered signal of shape (T,). Frames where the input is NaN remain
        NaN in the output.
    """
    T = len(signal)
    signal = np.array(signal, dtype=float)

    # state transition matrix (constant velocity model)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    # observation matrix (only position is observed)
    H = np.array([[1.0, 0.0]])
    # process noise covariance (standard CV discretization assuming
    # piecewise-constant acceleration noise with intensity process_noise)
    Q = process_noise * np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]])
    # measurement noise covariance
    R = np.array([[measurement_noise]])

    filtered = np.full(T, np.nan)

    # find first non-NaN observation to initialize state
    initialized = False
    x = None
    P = None

    for t in range(T):
        obs = signal[t]

        if not initialized:
            if np.isnan(obs):
                continue
            # initialize state from first observation
            x = np.array([obs, 0.0])
            P = np.diag([measurement_noise, measurement_noise / dt])
            filtered[t] = obs
            initialized = True
            continue

        # predict
        x = F @ x
        P = F @ P @ F.T + Q

        if np.isnan(obs):
            # no observation: predict only, output stays NaN
            continue

        # update with observation
        y = obs - H @ x  # innovation
        S = H @ P @ H.T + R  # innovation covariance
        K = P @ H.T / S[0, 0]  # Kalman gain (S is scalar)
        x = x + K.ravel() * y[0]
        P = (np.eye(2) - K @ H) @ P

        filtered[t] = x[0]

    return filtered


def kalman(
    xy: XY,
    process_noise: float = 1.0,
    measurement_noise: float = 0.04,
) -> XY:
    """Applies a forward Kalman filter to a XY data object. [3]_

    Uses a constant-velocity motion model where the state vector consists of
    position and velocity. Only positions are observed. The filter smooths noisy
    position data by combining predictions from the motion model with the
    observed measurements.

    Parameters
    ----------
    xy: XY
        Floodlight XY Data object. Must have ``framerate`` set.
    process_noise: float, optional
        Process noise intensity (acceleration variance in m²/s⁴) that controls
        how much the model trusts the constant-velocity prediction. Larger
        values allow faster changes in velocity. Default corresponds to
        :math:`\\sigma_a = 1\\,\\mathrm{m/s^2}`, which is a conservative smoothing
        prior.
    measurement_noise: float, optional
        Measurement noise variance (in m²) controlling how much the model
        trusts the observed positions. Larger values produce smoother output.
        Default is 0.04, corresponding to 0.20 m RMSE, a conservative
        estimate for common optical [4]_ and local [5]_ tracking systems
        during high-dynamic situations.

    Returns
    -------
    xy_filtered: XY
        XY object with position data filtered by the Kalman filter.

    Notes
    -----
    The Kalman filter requires ``xy.framerate`` to be set in order to compute
    the time interval between frames.

    The default noise parameters assume positions are given in meters. If
    positions use a different unit (e.g. centimeters), the noise parameters
    must be adjusted accordingly (e.g. ``measurement_noise = 20.0**2`` for
    20 cm RMSE in centimeter units).

    Unlike :func:`~floodlight.transforms.filter.butterworth_lowpass` and
    :func:`~floodlight.transforms.filter.savgol_lowpass`, the Kalman filter
    handles missing data (NaN) natively. When an observation is missing, the
    filter runs a predict-only step, maintaining its internal state (velocity
    estimate, covariance) across gaps. Frames with missing input data remain
    NaN in the output — no gap-filling is performed.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from floodlight import XY
    >>> from floodlight.transforms.filter import kalman

    Generate a noisy XY-object to smooth.

    >>> t = np.linspace(-5, 5, 1000)
    >>> player_x = np.sin(t) * t + np.random.rand(1000)
    >>> player_x[450:495] = np.NaN
    >>> player_x[505:550] = np.NaN
    >>> player_y = t + np.random.randn()
    >>> xy = XY(np.transpose(np.stack((player_x, player_y))), framerate=20)

    Apply the Kalman filter with default settings.

    >>> xy_filt = kalman(xy)
    >>> plt.plot(xy.x)
    >>> plt.plot(xy_filt.x, linewidth=3)
    >>> plt.legend(("Raw", "Smoothed"))
    >>> plt.show()

    .. image:: ../../_img/kalman_default_example.png


    Apply the filter with increased measurement noise for stronger smoothing.

    >>> xy_filt = kalman(xy, measurement_noise=1.0)
    >>> plt.plot(xy.x)
    >>> plt.plot(xy_filt.x, linewidth=3)
    >>> plt.legend(("Raw", "Smoothed"))
    >>> plt.show()

    .. image:: ../../_img/kalman_adjusted_example.png

    References
    ----------
        .. [3] `Kalman, R. E. (1960). A New Approach to Linear Filtering and
            Prediction Problems. Journal of Basic Engineering, 82(1), 35-45.
            <https://doi.org/10.1115/1.3662552>`_
        .. [4] `Linke, D., Link, D., & Lames, M. (2020). Football-specific
            validity of TRACAB's optical video tracking systems. PLoS ONE,
            15(3), e0230179. <https://doi.org/10.1371/journal.pone.0230179>`_
        .. [5] `Blauberger, P., Marzilger, R., & Lames, M. (2021). Validation
            of player and ball tracking with a local positioning system.
            Sensors, 21(4), 1465. <https://doi.org/10.3390/s21041465>`_
    """
    if xy.framerate is None:
        raise ValueError(
            "The Kalman filter requires xy.framerate to be set in order to "
            "compute the time step between frames."
        )

    dt = 1.0 / xy.framerate

    # pre-allocate space for filtered data
    xy_filt = np.empty(xy.xy.shape)
    # loop through the xy-object columns
    for i, column in enumerate(np.transpose(xy.xy)):
        xy_filt[:, i] = _kalman_filter_1d(column, dt, process_noise, measurement_noise)

    # create new XY-data object with filtered data
    xy_filtered = XY(xy=xy_filt, framerate=xy.framerate, direction=xy.direction)

    return xy_filtered

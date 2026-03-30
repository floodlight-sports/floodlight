import pytest
import numpy as np

from floodlight import XY
from floodlight.transforms import filter


@pytest.mark.unit
def test_get_sequences(example_sequence: np.ndarray) -> None:
    # Arrange
    sequence = example_sequence

    # Act
    seq_filt, seq_short = filter._get_filterable_and_short_sequences(sequence, 2)

    # Assert
    assert np.all(
        (
            np.array_equal(seq_filt, np.array([[2, 5]])),
            np.array_equal(seq_short, np.array([[7, 9], [10, 11]])),
        )
    )


@pytest.mark.unit
def test_get_sequences_empty(example_sequence_empty: np.ndarray) -> None:
    # Arrange
    sequence = example_sequence_empty

    # Act
    with pytest.raises(
        ValueError,
        match="Expected input data to be one-dimensional. Got 0-dimensional data "
        "instead.",
    ):
        filter._get_filterable_and_short_sequences(sequence, 2)


@pytest.mark.unit
def test_get_sequences_two_dimensional(
    example_sequence_two_dimensional: np.ndarray,
) -> None:
    # Arrange
    sequence = example_sequence_two_dimensional

    # Act
    with pytest.raises(
        ValueError,
        match="Expected input data to be one-dimensional. Got 2-dimensional "
        "data instead.",
    ):
        filter._get_filterable_and_short_sequences(sequence, 2)


@pytest.mark.unit
def test_get_sequences_full(example_sequence_full: np.ndarray) -> None:
    # Arrange
    sequence = example_sequence_full

    # Act
    seq_filt, seq_short = filter._get_filterable_and_short_sequences(sequence, 2)

    # Assert
    assert np.all(
        (
            np.array_equal(seq_filt, np.array([[0, 5]])),
            np.array_equal(seq_short, np.empty((0, 2))),
        )
    )


@pytest.mark.unit
def test_get_sequences_nan(example_sequence_nan: np.ndarray) -> None:
    # Arrange
    sequence = example_sequence_nan

    # Act
    seq_filt, seq_short = filter._get_filterable_and_short_sequences(sequence, 2)

    # Assert
    assert np.all(
        (
            np.array_equal(seq_filt, np.empty((0, 2))),
            np.array_equal(seq_short, np.empty((0, 2))),
        )
    )


@pytest.mark.unit
def test_butterworth_lowpass_remove_seqs_false(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_filt = filter.butterworth_lowpass(data)

    # Assert
    assert np.array_equal(
        np.round(data_filt, 2),
        np.array(
            [
                [np.nan, -8.66, np.nan, 1.18],
                [np.nan, -6.29, np.nan, 2.28],
                [-5.07, -4.31, np.nan, 3.32],
                [-2.7, -1.95, np.nan, 4.26],
                [np.nan, -0.13, np.nan, 5.07],
                [np.nan, 2.31, np.nan, 5.71],
                [1.43, 3.74, np.nan, 6.15],
                [3.54, 6.53, np.nan, 6.39],
                [5.61, 8.07, np.nan, 6.41],
                [7.61, 10.53, np.nan, 6.25],
                [9.52, np.nan, np.nan, 5.93],
                [11.34, np.nan, np.nan, 5.49],
                [13.07, np.nan, np.nan, 5.0],
                [14.72, 14.88, np.nan, 4.51],
                [16.29, 17.05, np.nan, 4.08],
                [17.81, 18.37, np.nan, 3.78],
                [19.29, 19.27, np.nan, 3.63],
                [20.75, 20.46, np.nan, 3.67],
                [22.2, 22.61, np.nan, 3.92],
                [23.63, 23.54, np.nan, 4.38],
                [25.06, 25.25, np.nan, 5.02],
                [26.45, 25.95, np.nan, 5.82],
                [27.82, 28.06, np.nan, 6.74],
                [np.nan, 29.55, np.nan, 7.74],
                [30.06, np.nan, np.nan, 8.77],
            ]
        ),
        equal_nan=True,
    )


@pytest.mark.unit
def test_butterworth_lowpass_remove_seqs_true(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_filt = filter.butterworth_lowpass(data, remove_short_seqs=True)

    # Assert
    assert np.array_equal(
        np.round(data_filt, 2),
        np.array(
            [
                [np.nan, np.nan, np.nan, 1.18],
                [np.nan, np.nan, np.nan, 2.28],
                [np.nan, np.nan, np.nan, 3.32],
                [np.nan, np.nan, np.nan, 4.26],
                [np.nan, np.nan, np.nan, 5.07],
                [np.nan, np.nan, np.nan, 5.71],
                [1.43, np.nan, np.nan, 6.15],
                [3.54, np.nan, np.nan, 6.39],
                [5.61, np.nan, np.nan, 6.41],
                [7.61, np.nan, np.nan, 6.25],
                [9.52, np.nan, np.nan, 5.93],
                [11.34, np.nan, np.nan, 5.49],
                [13.07, np.nan, np.nan, 5.0],
                [14.72, np.nan, np.nan, 4.51],
                [16.29, np.nan, np.nan, 4.08],
                [17.81, np.nan, np.nan, 3.78],
                [19.29, np.nan, np.nan, 3.63],
                [20.75, np.nan, np.nan, 3.67],
                [22.2, np.nan, np.nan, 3.92],
                [23.63, np.nan, np.nan, 4.38],
                [25.06, np.nan, np.nan, 5.02],
                [26.45, np.nan, np.nan, 5.82],
                [27.82, np.nan, np.nan, 6.74],
                [np.nan, np.nan, np.nan, 7.74],
                [np.nan, np.nan, np.nan, 8.77],
            ]
        ),
        equal_nan=True,
    )


@pytest.mark.unit
def test_butterworth_lowpass_short_remove_seqs_false(
    example_xy_filter_short: XY,
) -> None:
    # Arrange
    data = example_xy_filter_short

    # Act
    data_filt = filter.butterworth_lowpass(data)

    # Assert
    assert np.array_equal(data, data_filt, equal_nan=True)


@pytest.mark.unit
def test_butterworth_lowpass_short_remove_seqs_true(
    example_xy_filter_short: XY,
) -> None:
    # Arrange
    data = example_xy_filter_short

    # Act
    data_filt = filter.butterworth_lowpass(data, remove_short_seqs=True)

    # Assert
    assert np.array_equal(
        data_filt,
        np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
        equal_nan=True,
    )


@pytest.mark.unit
def test_butterworth_lowpass_one_frame(example_xy_filter_one_frame: XY) -> None:
    # Arrange
    data = example_xy_filter_one_frame

    # Act
    with pytest.raises(
        ValueError,
        match="Expected input data to be one-dimensional. Got 0-dimensional data "
        "instead.",
    ):
        # Assert
        filter.butterworth_lowpass(data)


@pytest.mark.unit
def test_butterworth_lowpass_empty(example_xy_filter_empty: XY) -> None:
    # Arrange
    data = example_xy_filter_empty

    # Act
    data_filt = filter.butterworth_lowpass(data, remove_short_seqs=True)

    # Assert
    assert np.array_equal(data, data_filt, equal_nan=True)


@pytest.mark.unit
def test_savgol_lowpass_remove_seqs_false(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_filt = filter.savgol_lowpass(data)

    # Assert
    assert np.array_equal(
        np.round(data_filt, 2),
        np.array(
            [
                [np.nan, -8.64, np.nan, 1],
                [np.nan, -6.39, np.nan, 2],
                [-5.07, -4.17, np.nan, 3],
                [-2.7, -2.13, np.nan, 4],
                [np.nan, 0.11, np.nan, 5],
                [np.nan, 1.97, np.nan, 6],
                [1.6, 4.17, np.nan, 7],
                [4.86, 6.12, np.nan, 8.17],
                [7.42, 8.34, np.nan, 8.66],
                [8.98, 10.46, np.nan, 8.17],
                [10.64, np.nan, np.nan, 7],
                [11.88, np.nan, np.nan, 6],
                [13.49, np.nan, np.nan, 5],
                [14.81, 14.88, np.nan, 4],
                [16.11, 17.07, np.nan, 3],
                [17.2, 18.35, np.nan, 1.83],
                [18.58, 19.27, np.nan, 1.34],
                [20.23, 20.7, np.nan, 1.83],
                [21.76, 22.25, np.nan, 3],
                [23.02, 23.86, np.nan, 4],
                [24.33, 24.89, np.nan, 5],
                [25.67, 26.33, np.nan, 6],
                [27.15, 27.81, np.nan, 7],
                [np.nan, 29.61, np.nan, 8],
                [30.06, np.nan, np.nan, 9],
            ]
        ),
        equal_nan=True,
    )


@pytest.mark.unit
def test_savgol_lowpass_remove_seqs_true(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_filt = filter.savgol_lowpass(data, remove_short_seqs=True)

    # Assert
    assert np.array_equal(
        np.round(data_filt, 2),
        np.array(
            [
                [np.nan, -8.64, np.nan, 1],
                [np.nan, -6.39, np.nan, 2],
                [np.nan, -4.17, np.nan, 3],
                [np.nan, -2.13, np.nan, 4],
                [np.nan, 0.11, np.nan, 5],
                [np.nan, 1.97, np.nan, 6],
                [1.6, 4.17, np.nan, 7],
                [4.86, 6.12, np.nan, 8.17],
                [7.42, 8.34, np.nan, 8.66],
                [8.98, 10.46, np.nan, 8.17],
                [10.64, np.nan, np.nan, 7],
                [11.88, np.nan, np.nan, 6],
                [13.49, np.nan, np.nan, 5],
                [14.81, 14.88, np.nan, 4],
                [16.11, 17.07, np.nan, 3],
                [17.2, 18.35, np.nan, 1.83],
                [18.58, 19.27, np.nan, 1.34],
                [20.23, 20.7, np.nan, 1.83],
                [21.76, 22.25, np.nan, 3],
                [23.02, 23.86, np.nan, 4],
                [24.33, 24.89, np.nan, 5],
                [25.67, 26.33, np.nan, 6],
                [27.15, 27.81, np.nan, 7],
                [np.nan, 29.61, np.nan, 8],
                [np.nan, np.nan, np.nan, 9],
            ]
        ),
        equal_nan=True,
    )


@pytest.mark.unit
def test_savgol_lowpass_short_remove_seqs_false(example_xy_filter_short: XY) -> None:
    # Arrange
    data = example_xy_filter_short

    # Act
    data_filt = filter.savgol_lowpass(data)

    # Assert
    assert np.array_equal(data, data_filt, equal_nan=True)


@pytest.mark.unit
def test_savgol_lowpass_short_remove_seqs_true(example_xy_filter_short: XY) -> None:
    # Arrange
    data = example_xy_filter_short

    # Act
    data_filt = filter.savgol_lowpass(data, remove_short_seqs=True)

    # Assert
    assert np.array_equal(
        data_filt,
        np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
        equal_nan=True,
    )


@pytest.mark.unit
def test_savgol_lowpass_one_frame(example_xy_filter_one_frame: XY) -> None:
    # Arrange
    data = example_xy_filter_one_frame

    # Act
    with pytest.raises(
        ValueError,
        match="Expected input data to be one-dimensional. Got 0-dimensional data "
        "instead.",
    ):
        # Assert
        filter.savgol_lowpass(data)


@pytest.mark.unit
def test_savgol_lowpass_empty(example_xy_filter_empty: XY) -> None:
    # Arrange
    data = example_xy_filter_empty

    # Act
    data_filt = filter.savgol_lowpass(data, remove_short_seqs=True)

    # Assert
    assert np.array_equal(data, data_filt, equal_nan=True)


@pytest.mark.unit
def test_fir_lowpass_remove_seqs_false(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_filt = filter.fir_lowpass(data, numtaps=3)

    # Assert
    assert np.array_equal(
        np.round(data_filt, 2),
        np.array(
            [
                [np.nan, -8.66, np.nan, 1.0],
                [np.nan, -6.34, np.nan, 2.0],
                [-5.07, -4.27, np.nan, 3.0],
                [-2.7, -2.01, np.nan, 4.0],
                [np.nan, -0.06, np.nan, 5.0],
                [np.nan, 2.19, np.nan, 6.0],
                [1.53, 3.9, np.nan, 7.0],
                [4.92, 6.38, np.nan, 7.99],
                [7.08, 8.18, np.nan, 8.75],
                [9.26, 10.53, np.nan, 7.99],
                [10.28, np.nan, np.nan, 7.0],
                [12.15, np.nan, np.nan, 6.0],
                [13.31, np.nan, np.nan, 5.0],
                [14.84, 14.88, np.nan, 4.0],
                [16.17, 16.94, np.nan, 3.0],
                [17.14, 18.31, np.nan, 2.01],
                [18.59, 19.31, np.nan, 1.25],
                [20.27, 20.58, np.nan, 2.01],
                [21.7, 22.46, np.nan, 3.0],
                [23.08, 23.63, np.nan, 4.0],
                [24.28, 25.13, np.nan, 5.0],
                [25.73, 26.12, np.nan, 6.0],
                [27.13, 27.99, np.nan, 7.0],
                [np.nan, 29.55, np.nan, 8.0],
                [30.06, np.nan, np.nan, 9.0],
            ]
        ),
        equal_nan=True,
    )


@pytest.mark.unit
def test_fir_lowpass_remove_seqs_true(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act - default numtaps=21, all sequences are too short
    data_filt = filter.fir_lowpass(data, remove_short_seqs=True)

    # Assert - all values should be NaN since no sequence meets min length
    assert np.all(np.isnan(data_filt.xy))


@pytest.mark.unit
def test_fir_lowpass_short_remove_seqs_false(
    example_xy_filter_short: XY,
) -> None:
    # Arrange
    data = example_xy_filter_short

    # Act
    data_filt = filter.fir_lowpass(data)

    # Assert
    assert np.array_equal(data, data_filt, equal_nan=True)


@pytest.mark.unit
def test_fir_lowpass_short_remove_seqs_true(
    example_xy_filter_short: XY,
) -> None:
    # Arrange
    data = example_xy_filter_short

    # Act
    data_filt = filter.fir_lowpass(data, remove_short_seqs=True)

    # Assert
    assert np.array_equal(
        data_filt,
        np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
        equal_nan=True,
    )


@pytest.mark.unit
def test_fir_lowpass_empty(example_xy_filter_empty: XY) -> None:
    # Arrange
    data = example_xy_filter_empty

    # Act
    data_filt = filter.fir_lowpass(data, remove_short_seqs=True)

    # Assert
    assert np.array_equal(data, data_filt, equal_nan=True)


@pytest.mark.unit
def test_kalman_default(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_filt = filter.kalman(data)

    # Assert
    assert np.array_equal(
        np.round(data_filt, 2),
        np.array(
            [
                [np.nan, -8.66, np.nan, 1.0],
                [np.nan, -7.45, np.nan, 1.51],
                [-5.07, -6.22, np.nan, 2.09],
                [-3.86, -4.64, np.nan, 2.8],
                [np.nan, -2.84, np.nan, 3.67],
                [np.nan, -0.64, np.nan, 4.67],
                [-0.95, 1.49, np.nan, 5.76],
                [1.88, 3.95, np.nan, 6.88],
                [4.32, 6.23, np.nan, 8.01],
                [6.81, 8.63, np.nan, 8.54],
                [8.77, np.nan, np.nan, 8.61],
                [10.84, np.nan, np.nan, 8.36],
                [12.61, np.nan, np.nan, 7.88],
                [14.37, 15.18, np.nan, 7.23],
                [16.03, 16.93, np.nan, 6.45],
                [17.49, 18.54, np.nan, 5.58],
                [18.95, 19.96, np.nan, 4.65],
                [20.47, 21.34, np.nan, 4.08],
                [21.96, 22.89, np.nan, 3.8],
                [23.43, 24.31, np.nan, 3.75],
                [24.83, 25.79, np.nan, 3.9],
                [26.24, 27.11, np.nan, 4.22],
                [27.64, 28.58, np.nan, 4.67],
                [np.nan, 30.04, np.nan, 5.23],
                [30.54, np.nan, np.nan, 5.9],
            ]
        ),
        equal_nan=True,
    )


@pytest.mark.unit
def test_kalman_nan_preservation(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_filt = filter.kalman(data)

    # Assert - NaN positions in input must remain NaN in output
    input_nans = np.isnan(np.array(data.xy, dtype=float))
    output_nans = np.isnan(data_filt.xy)
    assert np.array_equal(input_nans, output_nans)


@pytest.mark.unit
def test_kalman_all_nan() -> None:
    # Arrange
    data = XY(
        np.array(
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        ),
        framerate=20,
    )

    # Act
    data_filt = filter.kalman(data)

    # Assert
    assert np.all(np.isnan(data_filt.xy))


@pytest.mark.unit
def test_kalman_no_framerate() -> None:
    # Arrange
    data = XY(np.array([[1.0, 2.0], [3.0, 4.0]]))

    # Act
    with pytest.raises(
        ValueError,
        match="The Kalman filter requires xy.framerate to be set",
    ):
        filter.kalman(data)


@pytest.mark.unit
def test_kalman_single_observation() -> None:
    # Arrange
    data = XY(
        np.array(
            [
                [np.nan, np.nan],
                [5.0, 3.0],
                [np.nan, np.nan],
            ]
        ),
        framerate=20,
    )

    # Act
    data_filt = filter.kalman(data)

    # Assert - only the single observation frame should be non-NaN
    assert np.array_equal(
        data_filt.xy,
        np.array(
            [
                [np.nan, np.nan],
                [5.0, 3.0],
                [np.nan, np.nan],
            ]
        ),
        equal_nan=True,
    )


@pytest.mark.unit
def test_kalman_empty(example_xy_filter_empty: XY) -> None:
    # Arrange
    data = example_xy_filter_empty

    # Act
    data_filt = filter.kalman(data)

    # Assert
    assert np.array_equal(data, data_filt, equal_nan=True)

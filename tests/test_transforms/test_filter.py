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
                [np.NaN, -8.66, np.NaN, 1.18],
                [np.NaN, -6.29, np.NaN, 2.28],
                [-5.07, -4.31, np.NaN, 3.32],
                [-2.7, -1.95, np.NaN, 4.26],
                [np.NaN, -0.13, np.NaN, 5.07],
                [np.NaN, 2.31, np.NaN, 5.71],
                [1.43, 3.74, np.NaN, 6.15],
                [3.54, 6.53, np.NaN, 6.39],
                [5.61, 8.07, np.NaN, 6.41],
                [7.61, 10.53, np.NaN, 6.25],
                [9.52, np.NaN, np.NaN, 5.93],
                [11.34, np.NaN, np.NaN, 5.49],
                [13.07, np.NaN, np.NaN, 5.0],
                [14.72, 14.88, np.NaN, 4.51],
                [16.29, 17.05, np.NaN, 4.08],
                [17.81, 18.37, np.NaN, 3.78],
                [19.29, 19.27, np.NaN, 3.63],
                [20.75, 20.46, np.NaN, 3.67],
                [22.2, 22.61, np.NaN, 3.92],
                [23.63, 23.54, np.NaN, 4.38],
                [25.06, 25.25, np.NaN, 5.02],
                [26.45, 25.95, np.NaN, 5.82],
                [27.82, 28.06, np.NaN, 6.74],
                [np.NaN, 29.55, np.NaN, 7.74],
                [30.06, np.NaN, np.NaN, 8.77],
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
                [np.NaN, np.NaN, np.NaN, 1.18],
                [np.NaN, np.NaN, np.NaN, 2.28],
                [np.NaN, np.NaN, np.NaN, 3.32],
                [np.NaN, np.NaN, np.NaN, 4.26],
                [np.NaN, np.NaN, np.NaN, 5.07],
                [np.NaN, np.NaN, np.NaN, 5.71],
                [1.43, np.NaN, np.NaN, 6.15],
                [3.54, np.NaN, np.NaN, 6.39],
                [5.61, np.NaN, np.NaN, 6.41],
                [7.61, np.NaN, np.NaN, 6.25],
                [9.52, np.NaN, np.NaN, 5.93],
                [11.34, np.NaN, np.NaN, 5.49],
                [13.07, np.NaN, np.NaN, 5.0],
                [14.72, np.NaN, np.NaN, 4.51],
                [16.29, np.NaN, np.NaN, 4.08],
                [17.81, np.NaN, np.NaN, 3.78],
                [19.29, np.NaN, np.NaN, 3.63],
                [20.75, np.NaN, np.NaN, 3.67],
                [22.2, np.NaN, np.NaN, 3.92],
                [23.63, np.NaN, np.NaN, 4.38],
                [25.06, np.NaN, np.NaN, 5.02],
                [26.45, np.NaN, np.NaN, 5.82],
                [27.82, np.NaN, np.NaN, 6.74],
                [np.NaN, np.NaN, np.NaN, 7.74],
                [np.NaN, np.NaN, np.NaN, 8.77],
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
        np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]),
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
                [np.NaN, -8.64, np.NaN, 1],
                [np.NaN, -6.39, np.NaN, 2],
                [-5.07, -4.17, np.NaN, 3],
                [-2.7, -2.13, np.NaN, 4],
                [np.NaN, 0.11, np.NaN, 5],
                [np.NaN, 1.97, np.NaN, 6],
                [1.6, 4.17, np.NaN, 7],
                [4.86, 6.12, np.NaN, 8.17],
                [7.42, 8.34, np.NaN, 8.66],
                [8.98, 10.46, np.NaN, 8.17],
                [10.64, np.NaN, np.NaN, 7],
                [11.88, np.NaN, np.NaN, 6],
                [13.49, np.NaN, np.NaN, 5],
                [14.81, 14.88, np.NaN, 4],
                [16.11, 17.07, np.NaN, 3],
                [17.2, 18.35, np.NaN, 1.83],
                [18.58, 19.27, np.NaN, 1.34],
                [20.23, 20.7, np.NaN, 1.83],
                [21.76, 22.25, np.NaN, 3],
                [23.02, 23.86, np.NaN, 4],
                [24.33, 24.89, np.NaN, 5],
                [25.67, 26.33, np.NaN, 6],
                [27.15, 27.81, np.NaN, 7],
                [np.NaN, 29.61, np.NaN, 8],
                [30.06, np.NaN, np.NaN, 9],
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
                [np.NaN, -8.64, np.NaN, 1],
                [np.NaN, -6.39, np.NaN, 2],
                [np.NaN, -4.17, np.NaN, 3],
                [np.NaN, -2.13, np.NaN, 4],
                [np.NaN, 0.11, np.NaN, 5],
                [np.NaN, 1.97, np.NaN, 6],
                [1.6, 4.17, np.NaN, 7],
                [4.86, 6.12, np.NaN, 8.17],
                [7.42, 8.34, np.NaN, 8.66],
                [8.98, 10.46, np.NaN, 8.17],
                [10.64, np.NaN, np.NaN, 7],
                [11.88, np.NaN, np.NaN, 6],
                [13.49, np.NaN, np.NaN, 5],
                [14.81, 14.88, np.NaN, 4],
                [16.11, 17.07, np.NaN, 3],
                [17.2, 18.35, np.NaN, 1.83],
                [18.58, 19.27, np.NaN, 1.34],
                [20.23, 20.7, np.NaN, 1.83],
                [21.76, 22.25, np.NaN, 3],
                [23.02, 23.86, np.NaN, 4],
                [24.33, 24.89, np.NaN, 5],
                [25.67, 26.33, np.NaN, 6],
                [27.15, 27.81, np.NaN, 7],
                [np.NaN, 29.61, np.NaN, 8],
                [np.NaN, np.NaN, np.NaN, 9],
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
        np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]),
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

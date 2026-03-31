import numpy as np
import pytest
from floodlight import XY
from floodlight.transforms import interpolation


@pytest.mark.unit
def test_interpolate_linear_default(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_interp = interpolation.interpolate_linear(data)

    # Assert
    assert np.array_equal(
        np.round(data_interp.xy, 2),
        np.array(
            [
                [np.nan, -8.66, np.nan, 1.0],
                [np.nan, -6.29, np.nan, 2.0],
                [-5.07, -4.31, np.nan, 3.0],
                [-2.7, -1.95, np.nan, 4.0],
                [-1.29, -0.13, np.nan, 5.0],
                [0.12, 2.31, np.nan, 6.0],
                [1.53, 3.74, np.nan, 7.0],
                [5.13, 6.53, np.nan, 8.0],
                [7.02, 8.07, np.nan, 9.0],
                [9.48, 10.53, np.nan, 8.0],
                [10.09, 11.62, np.nan, 7.0],
                [12.31, 12.7, np.nan, 6.0],
                [13.22, 13.79, np.nan, 5.0],
                [14.88, 14.88, np.nan, 4.0],
                [16.23, 17.05, np.nan, 3.0],
                [17.06, 18.37, np.nan, 2.0],
                [18.56, 19.27, np.nan, 1.0],
                [20.32, 20.46, np.nan, 2.0],
                [21.7, 22.61, np.nan, 3.0],
                [23.11, 23.54, np.nan, 4.0],
                [24.23, 25.25, np.nan, 5.0],
                [25.74, 25.95, np.nan, 6.0],
                [27.13, 28.06, np.nan, 7.0],
                [28.6, 29.55, np.nan, 8.0],
                [30.06, np.nan, np.nan, 9.0],
            ]
        ),
        equal_nan=True,
    )


@pytest.mark.unit
def test_interpolate_linear_max_gap(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act - col 1 has a 3-frame gap (rows 10-12), col 0 has a 2-frame gap (rows 4-5)
    data_interp = interpolation.interpolate_linear(data, max_gap=2)

    # Assert - 2-frame gap in col 0 is filled
    assert not np.isnan(data_interp.xy[4, 0])
    assert not np.isnan(data_interp.xy[5, 0])
    # 3-frame gap in col 1 is NOT filled
    assert np.all(np.isnan(data_interp.xy[10:13, 1]))


@pytest.mark.unit
def test_interpolate_linear_xIDs(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act - only interpolate player 1 (cols 2, 3)
    data_interp = interpolation.interpolate_linear(data, xIDs=[1])

    # Assert - player 0 columns unchanged (gaps still NaN)
    assert np.all(np.isnan(data_interp.xy[4:6, 0]))
    assert np.all(np.isnan(data_interp.xy[10:13, 1]))
    # player 1 col 2 is all-NaN (no bounded gaps) so stays NaN
    assert np.all(np.isnan(data_interp.xy[:, 2]))
    # player 1 col 3 has no NaN gaps so stays the same
    assert np.array_equal(data_interp.xy[:, 3], data.xy[:, 3])


@pytest.mark.unit
def test_interpolate_linear_all_nan() -> None:
    # Arrange
    data = XY(np.full((5, 4), np.nan), framerate=20)

    # Act
    data_interp = interpolation.interpolate_linear(data)

    # Assert
    assert np.all(np.isnan(data_interp.xy))


@pytest.mark.unit
def test_interpolate_linear_no_nan() -> None:
    # Arrange
    data = XY(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float), framerate=20)

    # Act
    data_interp = interpolation.interpolate_linear(data)

    # Assert
    assert np.array_equal(data_interp.xy, data.xy)


@pytest.mark.unit
def test_interpolate_linear_empty(example_xy_filter_empty: XY) -> None:
    # Arrange
    data = example_xy_filter_empty

    # Act
    data_interp = interpolation.interpolate_linear(data)

    # Assert
    assert np.array_equal(data, data_interp, equal_nan=True)


@pytest.mark.unit
def test_interpolate_linear_invalid_xID(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act & Assert
    with pytest.raises(ValueError):
        interpolation.interpolate_linear(data, xIDs=[5])


@pytest.mark.unit
def test_interpolate_linear_preserves_properties(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_interp = interpolation.interpolate_linear(data)

    # Assert
    assert data_interp.xy.shape == data.xy.shape
    assert data_interp.framerate == data.framerate
    assert data_interp.direction == data.direction


# --- Polynomial interpolation tests ---


@pytest.mark.unit
def test_interpolate_polynomial_default(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_interp = interpolation.interpolate_polynomial(data)

    # Assert
    assert np.array_equal(
        np.round(data_interp.xy, 2),
        np.array(
            [
                [np.nan, -8.66, np.nan, 1.0],
                [np.nan, -6.29, np.nan, 2.0],
                [-5.07, -4.31, np.nan, 3.0],
                [-2.7, -1.95, np.nan, 4.0],
                [-1.71, -0.13, np.nan, 5.0],
                [-0.74, 2.31, np.nan, 6.0],
                [1.53, 3.74, np.nan, 7.0],
                [5.13, 6.53, np.nan, 8.0],
                [7.02, 8.07, np.nan, 9.0],
                [9.48, 10.53, np.nan, 8.0],
                [10.09, 12.16, np.nan, 7.0],
                [12.31, 12.91, np.nan, 6.0],
                [13.22, 13.55, np.nan, 5.0],
                [14.88, 14.88, np.nan, 4.0],
                [16.23, 17.05, np.nan, 3.0],
                [17.06, 18.37, np.nan, 2.0],
                [18.56, 19.27, np.nan, 1.0],
                [20.32, 20.46, np.nan, 2.0],
                [21.7, 22.61, np.nan, 3.0],
                [23.11, 23.54, np.nan, 4.0],
                [24.23, 25.25, np.nan, 5.0],
                [25.74, 25.95, np.nan, 6.0],
                [27.13, 28.06, np.nan, 7.0],
                [28.46, 29.55, np.nan, 8.0],
                [30.06, np.nan, np.nan, 9.0],
            ]
        ),
        equal_nan=True,
    )


@pytest.mark.unit
def test_interpolate_polynomial_max_gap(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_interp = interpolation.interpolate_polynomial(data, max_gap=2)

    # Assert - 2-frame gap in col 0 is filled
    assert not np.isnan(data_interp.xy[4, 0])
    assert not np.isnan(data_interp.xy[5, 0])
    # 3-frame gap in col 1 is NOT filled
    assert np.all(np.isnan(data_interp.xy[10:13, 1]))


@pytest.mark.unit
def test_interpolate_polynomial_xIDs(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act - only interpolate player 1 (cols 2, 3)
    data_interp = interpolation.interpolate_polynomial(data, xIDs=[1])

    # Assert - player 0 columns unchanged (gaps still NaN)
    assert np.all(np.isnan(data_interp.xy[4:6, 0]))
    assert np.all(np.isnan(data_interp.xy[10:13, 1]))


@pytest.mark.unit
def test_interpolate_polynomial_all_nan() -> None:
    # Arrange
    data = XY(np.full((5, 4), np.nan), framerate=20)

    # Act
    data_interp = interpolation.interpolate_polynomial(data)

    # Assert
    assert np.all(np.isnan(data_interp.xy))


@pytest.mark.unit
def test_interpolate_polynomial_no_nan() -> None:
    # Arrange
    data = XY(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float), framerate=20)

    # Act
    data_interp = interpolation.interpolate_polynomial(data)

    # Assert
    assert np.array_equal(data_interp.xy, data.xy)


@pytest.mark.unit
def test_interpolate_polynomial_empty(example_xy_filter_empty: XY) -> None:
    # Arrange
    data = example_xy_filter_empty

    # Act
    data_interp = interpolation.interpolate_polynomial(data)

    # Assert
    assert np.array_equal(data, data_interp, equal_nan=True)


@pytest.mark.unit
def test_interpolate_polynomial_invalid_xID(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act & Assert
    with pytest.raises(ValueError):
        interpolation.interpolate_polynomial(data, xIDs=[5])


@pytest.mark.unit
def test_interpolate_polynomial_insufficient_data() -> None:
    # Arrange - 3 valid points, order=3 needs 4
    data = XY(
        np.array(
            [
                [1.0, np.nan],
                [np.nan, np.nan],
                [3.0, np.nan],
                [np.nan, np.nan],
                [5.0, np.nan],
            ]
        ),
        framerate=20,
    )

    # Act
    data_interp = interpolation.interpolate_polynomial(data, order=3)

    # Assert - gaps not filled because insufficient data
    assert np.isnan(data_interp.xy[1, 0])
    assert np.isnan(data_interp.xy[3, 0])


# --- Spline interpolation tests ---


@pytest.mark.unit
def test_interpolate_spline_default(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_interp = interpolation.interpolate_spline(data)

    # Assert
    assert np.array_equal(
        np.round(data_interp.xy, 2),
        np.array(
            [
                [np.nan, -8.66, np.nan, 1.0],
                [np.nan, -6.29, np.nan, 2.0],
                [-5.07, -4.31, np.nan, 3.0],
                [-2.7, -1.95, np.nan, 4.0],
                [-1.71, -0.13, np.nan, 5.0],
                [-0.74, 2.31, np.nan, 6.0],
                [1.53, 3.74, np.nan, 7.0],
                [5.13, 6.53, np.nan, 8.0],
                [7.02, 8.07, np.nan, 9.0],
                [9.48, 10.53, np.nan, 8.0],
                [10.09, 12.16, np.nan, 7.0],
                [12.31, 12.91, np.nan, 6.0],
                [13.22, 13.55, np.nan, 5.0],
                [14.88, 14.88, np.nan, 4.0],
                [16.23, 17.05, np.nan, 3.0],
                [17.06, 18.37, np.nan, 2.0],
                [18.56, 19.27, np.nan, 1.0],
                [20.32, 20.46, np.nan, 2.0],
                [21.7, 22.61, np.nan, 3.0],
                [23.11, 23.54, np.nan, 4.0],
                [24.23, 25.25, np.nan, 5.0],
                [25.74, 25.95, np.nan, 6.0],
                [27.13, 28.06, np.nan, 7.0],
                [28.46, 29.55, np.nan, 8.0],
                [30.06, np.nan, np.nan, 9.0],
            ]
        ),
        equal_nan=True,
    )


@pytest.mark.unit
def test_interpolate_spline_max_gap(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act
    data_interp = interpolation.interpolate_spline(data, max_gap=2)

    # Assert - 2-frame gap in col 0 is filled
    assert not np.isnan(data_interp.xy[4, 0])
    assert not np.isnan(data_interp.xy[5, 0])
    # 3-frame gap in col 1 is NOT filled
    assert np.all(np.isnan(data_interp.xy[10:13, 1]))


@pytest.mark.unit
def test_interpolate_spline_xIDs(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act - only interpolate player 1 (cols 2, 3)
    data_interp = interpolation.interpolate_spline(data, xIDs=[1])

    # Assert - player 0 columns unchanged (gaps still NaN)
    assert np.all(np.isnan(data_interp.xy[4:6, 0]))
    assert np.all(np.isnan(data_interp.xy[10:13, 1]))


@pytest.mark.unit
def test_interpolate_spline_all_nan() -> None:
    # Arrange
    data = XY(np.full((5, 4), np.nan), framerate=20)

    # Act
    data_interp = interpolation.interpolate_spline(data)

    # Assert
    assert np.all(np.isnan(data_interp.xy))


@pytest.mark.unit
def test_interpolate_spline_no_nan() -> None:
    # Arrange
    data = XY(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float), framerate=20)

    # Act
    data_interp = interpolation.interpolate_spline(data)

    # Assert
    assert np.array_equal(data_interp.xy, data.xy)


@pytest.mark.unit
def test_interpolate_spline_empty(example_xy_filter_empty: XY) -> None:
    # Arrange
    data = example_xy_filter_empty

    # Act
    data_interp = interpolation.interpolate_spline(data)

    # Assert
    assert np.array_equal(data, data_interp, equal_nan=True)


@pytest.mark.unit
def test_interpolate_spline_invalid_xID(example_xy_filter: XY) -> None:
    # Arrange
    data = example_xy_filter

    # Act & Assert
    with pytest.raises(ValueError):
        interpolation.interpolate_spline(data, xIDs=[5])


@pytest.mark.unit
def test_interpolate_spline_insufficient_data() -> None:
    # Arrange - 3 valid points, k=3 needs 4
    data = XY(
        np.array(
            [
                [1.0, np.nan],
                [np.nan, np.nan],
                [3.0, np.nan],
                [np.nan, np.nan],
                [5.0, np.nan],
            ]
        ),
        framerate=20,
    )

    # Act
    data_interp = interpolation.interpolate_spline(data, k=3)

    # Assert - gaps not filled because insufficient data
    assert np.isnan(data_interp.xy[1, 0])
    assert np.isnan(data_interp.xy[3, 0])

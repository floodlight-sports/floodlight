import pytest
import numpy as np


from floodlight.core.xy import XY


# Test def x(self) function
@pytest.mark.unit
def test_x_pos_int(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    # Act
    x_position = data.x

    # Assert
    assert np.array_equal(x_position, np.array([[1, 3], [5, 7]]))


@pytest.mark.unit
def test_x_none(example_xy_data_none: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_none)

    # Act
    x_position = data.x

    # Assert
    assert np.array_equal(x_position, np.array([[None, None], [None, None]]))


@pytest.mark.unit
def test_x_neg_int(example_xy_data_negative: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_negative)

    # Act
    x_position = data.x

    # Assert
    assert np.array_equal(x_position, np.array([[-1, -3], [-5, -7]]))


@pytest.mark.unit
def test_x_pos_float(example_xy_data_float: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_float)

    # Act
    x_position = data.x

    # Assert
    assert np.array_equal(
        x_position,
        np.array([[1.0, 0.00000000000000001], [2.32843476297480273847, 7.5]]),
    )


@pytest.mark.unit
def test_x_string(example_xy_data_string: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_string)

    # Act
    x_position = data.x

    # Assert
    assert np.array_equal(x_position, np.array([["1", "3"], ["5", "7"]]))


# Test def translate(self, shift: Tuple[numeric, numeric]
@pytest.mark.unit
def test_translate_pos_int(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    # Act
    data.translate([2, 2])
    translated_data = data.xy

    # Assert
    assert np.array_equal(translated_data, np.array([[3, 4, 5, 6], [7, 8, 9, 10]]))

@pytest.mark.unit
def test_rotate(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    # Act
    data.rotate(90)
    rotated_data_pos = data.xy
    data = XY(example_xy_data_pos_int)
    data.rotate(-90)
    rotated_data_neg = data.xy
    data = XY(example_xy_data_pos_int)

    # Assert
    assert np.array_equal(rotated_data_pos, np.array([[-2., 1., -4., 3.], [-6., 5., -8., 7.]]))
    assert np.array_equal(rotated_data_neg, np.array([[2., -1., 4., -3.], [6., -5., 8., -7.]]))

@pytest.mark.unit
def test_rotate_assert(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    # Act
    with pytest.raises(ValueError, match=f"Expected angle to be between -360 and 360"):
        data.rotate(367)
        rotated_data = data.xy

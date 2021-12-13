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


# Test def frame(self, t)
@pytest.mark.unit
def test_frame(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    assert np.array_equal(data.frame(0), np.array([1, 2, 3, 4]))
    with pytest.raises(IndexError):
        data.frame(2)


# Test def player(self, xID)
@pytest.mark.unit
def test_player(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    assert np.array_equal(data.player(0), np.array([[1, 2], [5, 6]]))


# Test def point(self, t, xID)
@pytest.mark.unit
def test_point(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    assert np.array_equal(data.point(1, 0), np.array([5, 6]))


# Test def translate(self, shift: Tuple[numeric, numeric]
@pytest.mark.unit
def test_translate_pos_int(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    # Act
    data.translate((2, 2))
    translated_data = data.xy

    # Assert
    assert np.array_equal(translated_data, np.array([[3, 4, 5, 6], [7, 8, 9, 10]]))


# Test def scale(self, factor, axis)
@pytest.mark.unit
def test_scale_pos_int(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    # Act + Assert
    data.scale(factor=2)
    assert np.array_equal(data.xy, np.array([[2, 4, 6, 8], [10, 12, 14, 16]]))
    data.scale(factor=-1, axis=0)
    assert np.array_equal(data.xy, np.array([[-2, 4, -6, 8], [-10, 12, -14, 16]]))
    data.scale(factor=0, axis=1)
    assert np.array_equal(data.xy, np.array([[-2, 0, -6, 0], [-10, 0, -14, 0]]))
    with pytest.raises(ValueError):
        data.scale(factor=1, axis=2)


# Test def reflect(self, factor, axis)
@pytest.mark.unit
def test_reflect_pos_int(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    # Act + Assert
    data.reflect(axis=1)
    assert np.array_equal(data.xy, np.array([[-1, 2, -3, 4], [-5, 6, -7, 8]]))
    data.reflect(axis=0)
    assert np.array_equal(data.xy, np.array([[-1, -2, -3, -4], [-5, -6, -7, -8]]))
    with pytest.raises(ValueError):
        data.reflect(axis=2)


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
    assert np.array_equal(
        rotated_data_pos, np.array([[-2.0, 1.0, -4.0, 3.0], [-6.0, 5.0, -8.0, 7.0]])
    )
    assert np.array_equal(
        rotated_data_neg, np.array([[2.0, -1.0, 4.0, -3.0], [6.0, -5.0, 8.0, -7.0]])
    )


@pytest.mark.unit
def test_rotate_assert(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    # Act
    with pytest.raises(ValueError):
        data.rotate(367)


# Test def slice(startframe, endframe, inplace) method
@pytest.mark.unit
def test_slice(example_xy_data_pos_int: np.array) -> None:
    # Arrange
    xy = XY(example_xy_data_pos_int)

    # check copying
    xy_deep_copy = xy.slice()
    assert xy is not xy_deep_copy
    assert xy.xy is not xy_deep_copy.xy
    assert (xy.xy == xy_deep_copy.xy).all()

    # slicing
    xy_short = xy.slice(endframe=1)
    xy_none = xy.slice(startframe=1, endframe=1)
    assert np.array_equal(xy_short.xy, np.array([[1, 2, 3, 4]]))
    assert len(xy_none.xy) == 0

    # inplace
    xy.slice(endframe=1, inplace=True)
    assert np.array_equal(xy.xy, np.array([[1, 2, 3, 4]]))

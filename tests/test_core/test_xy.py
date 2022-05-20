import matplotlib
import matplotlib.pyplot as plt
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


# Test def N(self) property
@pytest.mark.unit
def test_N(example_xy_data_pos_int: np.ndarray) -> None:
    uneven_array = np.array([[1, 2, 3], [5, 6, 7]])

    xy1 = XY(example_xy_data_pos_int)
    xy2 = XY(uneven_array)

    assert xy1.N == 2
    with pytest.raises(ValueError):
        print(xy2.N)


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
    data.translate((2, 2.2))

    # Assert
    assert data.xy.dtype == np.float32
    assert np.allclose(data, np.array([[3.0, 4.2, 5.0, 6.2], [7.0, 8.2, 9.0, 10.2]]))


# Test def scale(self, factor, axis)
@pytest.mark.unit
def test_scale_pos_int(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    # Act + Assert
    data.scale(factor=2.0)
    assert data.xy.dtype == np.float32
    assert np.array_equal(
        data.xy, np.array([[2.0, 4.0, 6.0, 8.0], [10.0, 12.0, 14.0, 16.0]])
    )

    data.scale(factor=-1, axis="x")
    assert np.array_equal(
        data.xy, np.array([[-2.0, 4.0, -6.0, 8.0], [-10.0, 12.0, -14.0, 16.0]])
    )

    data.scale(factor=0, axis="y")
    assert np.array_equal(
        data.xy, np.array([[-2.0, 0.0, -6.0, 0.0], [-10.0, 0.0, -14.0, 0.0]])
    )

    data.scale(factor=1.01, axis="x")
    assert np.allclose(
        data.xy, np.array([[-2.02, 0.0, -6.06, 0.0], [-10.1, 0.0, -14.14, 0.0]])
    )

    with pytest.raises(ValueError):
        data.scale(factor=1, axis="z")


# Test def reflect(self, factor, axis)
@pytest.mark.unit
def test_reflect_pos_int(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange
    data = XY(example_xy_data_pos_int)

    # Act + Assert
    data.reflect(axis="y")
    assert np.array_equal(data.xy, np.array([[-1, 2, -3, 4], [-5, 6, -7, 8]]))
    data.reflect(axis="x")
    assert np.array_equal(data.xy, np.array([[-1, -2, -3, -4], [-5, -6, -7, -8]]))
    with pytest.raises(ValueError):
        data.reflect(axis="z")


# Test rotate method for correct rotations and type conversion
@pytest.mark.unit
def test_rotate_angles_and_type(example_xy_data_pos_int: np.ndarray) -> None:
    # Arrange + Acts
    data = XY(example_xy_data_pos_int.copy())
    data.rotate(90)
    rotated_data_90 = data

    data = XY(example_xy_data_pos_int.copy())
    data.rotate(-90)
    rotated_data_neg_90 = data

    data = XY(example_xy_data_pos_int.copy())
    data.rotate(0)
    rotated_data_0 = data

    data = XY(example_xy_data_pos_int.copy())
    data.rotate(37.5)
    rotated_data_float = data

    data = XY(example_xy_data_pos_int.copy())
    data.rotate(-142)
    rotated_data_neg_142 = data

    # Assert
    assert rotated_data_90.xy.dtype == np.float32
    assert np.array_equal(
        rotated_data_90.xy, np.array([[-2.0, 1.0, -4.0, 3.0], [-6.0, 5.0, -8.0, 7.0]])
    )
    assert np.array_equal(
        rotated_data_neg_90.xy,
        np.array([[2.0, -1.0, 4.0, -3.0], [6.0, -5.0, 8.0, -7.0]]),
    )
    assert np.array_equal(
        rotated_data_0.xy, np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    )
    assert np.allclose(
        rotated_data_float.xy,
        np.array([[-0.424, 2.195, -0.055, 5.0], [0.314, 7.804, 0.683, 10.608]]),
    )
    assert np.allclose(
        rotated_data_neg_142.xy,
        np.array([[0.443, -2.192, 0.099, -4.999], [-0.246, -7.806, -0.591, -10.614]]),
    )


# Test rotate method for correct nan and exception handling
@pytest.mark.unit
def test_rotate_nan_and_exceptions(example_xy_data_float_with_nans: np.ndarray) -> None:
    # Arrange + Acts
    data = XY(example_xy_data_float_with_nans)
    data.rotate(90)

    # Assert
    assert data.xy.dtype == np.float64
    with pytest.raises(ValueError):
        data.rotate(367)
    assert np.array_equal(
        data.xy,
        np.array(
            [
                [np.nan, np.nan, -1.000e17, 6.123e00],
                [-6.000e00, 2.328e00, np.nan, np.nan],
            ]
        ),
        equal_nan=True,
    )


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


# Test plot method
@pytest.mark.plot
def test_plot_return_for_position_arg_without_axes(example_xy_object):
    # Arrange
    xy = example_xy_object

    # Act
    ax = xy.plot(t=0, plot_type="positions")

    # Assert
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


@pytest.mark.plot
def test_plot_return_for_position_arg_with_axes(example_xy_object):
    # Arrange
    xy = example_xy_object
    axes = plt.subplots()[1]

    # Act
    ax = xy.plot(t=0, plot_type="positions", ax=axes)

    # Assert
    assert ax == axes
    plt.close()


@pytest.mark.plot
def test_plot_return_for_trajectories_arg_without_axes(example_xy_object):
    # Arrange
    xy = example_xy_object

    # Act
    ax = xy.plot(t=(0, 4), plot_type="trajectories")

    # Assert
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


@pytest.mark.plot
def test_plot_return_for_trajectories_arg_with_axes(example_xy_object):
    # Arrange
    xy = example_xy_object
    axes = plt.subplots()[1]

    # Act
    ax = xy.plot(t=(0, 4), plot_type="trajectories", ax=axes)

    # Assert
    assert ax == axes
    plt.close()


@pytest.mark.plot
def test_plot_value_error_for_unknown_plot_type(example_xy_object):
    # Arrange
    xy = example_xy_object

    # Assert
    with pytest.raises(ValueError):
        xy.plot(t=(0, 4), plot_type="unkown plot type")
    plt.close()

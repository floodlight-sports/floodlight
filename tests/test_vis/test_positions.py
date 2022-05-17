import matplotlib
import matplotlib.pyplot as plt
import pytest

from floodlight.vis.positions import plot_positions, plot_trajectories


# Test plot_positions( xy, frame: int, ball: bool, ax: matplotlib.axes, **kwargs)
@pytest.mark.plot
def test_plot_positions_return_without_axes(example_xy_object):
    # Act
    ax = plot_positions(example_xy_object, frame=0, ball=False, ax=None)
    # Assert
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


@pytest.mark.plot
def test_plot_positions_return_with_axes(example_xy_object):
    # Arrange
    axes = plt.subplots()[1]
    # Act
    ax = plot_positions(example_xy_object, frame=0, ball=False, ax=axes)
    # Assert
    assert ax == axes
    plt.close()


# Test plot_trajectories(xy, frame: int, ball: bool, ax: matplotlib.axes, **kwargs)
@pytest.mark.plot
def test_plot_trajectoriess_return_matplotlib_axes_without_ax(example_xy_object):
    # Act
    ax = plot_trajectories(
        example_xy_object, start_frame=0, end_frame=4, ball=False, ax=None
    )
    # Assert
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


@pytest.mark.plot
def test_plot_trajectories_return_matplotlib_axes_with_ax(example_xy_object):
    # Arrange
    axes = plt.subplots()[1]
    # Act
    ax = plot_trajectories(
        example_xy_object, start_frame=0, end_frame=4, ball=False, ax=axes
    )
    # Assert
    assert ax == axes
    plt.close()


@pytest.mark.plot
def test_plot_trajectories_default_color(example_xy_object):
    # Act
    plot_trajectories(
        example_xy_object, start_frame=0, end_frame=4, ball=False, ax=None
    )
    # Assert
    for line in plt.gca().lines:
        assert line.get_color() == "black"
    plt.close()


@pytest.mark.plot
def test_plot_trajectories_default_color_ball_true(example_xy_object):
    # Act
    plot_trajectories(example_xy_object, start_frame=0, end_frame=4, ball=True, ax=None)
    # Assert
    for line in plt.gca().lines:
        assert line.get_color() == "grey"
    plt.close()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from floodlight.vis.pitches import plot_handball_pitch, plot_football_pitch


# Test def plot_*_pitch(
#     xlim: Tuple[Numeric, Numeric],
#     ylim: Tuple[Numeric, Numeric],
#     (length: Numeric),
#     (width: Numeric),
#     unit: str,
#     color_scheme: str,
#     show_axis_ticks: bool,
#     ax: matplotlib.axes,
#     **kwargs,) -> matplotib.axes
# Test return
# football


@pytest.mark.plot
def test_plot_football_pitch_return_matplotlib_axes(
    example_input_plot_football_pitch,
) -> None:
    # Act
    ax = plot_football_pitch(*example_input_plot_football_pitch)
    # Assert
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


# handball
@pytest.mark.plot
def test_plot_handball_pitch_return_matplotlib_axes(
    example_input_plot_handball_pitch,
) -> None:
    # Act
    ax = plot_handball_pitch(*example_input_plot_handball_pitch)
    # Assert
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


# Test ticks
# football
@pytest.mark.plot
def test_plot_football_pitch_show_axis_ticks_default(
    example_input_plot_football_pitch,
) -> None:
    # Act
    ax = plot_football_pitch(*example_input_plot_football_pitch)
    # Assert
    assert ax.get_xticks() == []
    assert ax.get_yticks() == []
    plt.close()


@pytest.mark.plot
def test_plot_football_pitch_show_axis_ticks_True(
    example_input_plot_football_pitch_axis_ticks,
) -> None:
    # Act
    ax = plot_football_pitch(*example_input_plot_football_pitch_axis_ticks)
    # Assert
    assert np.array_equal(
        np.array(ax.get_xticks()), np.array([-20, 0, 20, 40, 60, 80, 100, 120])
    )
    assert np.array_equal(
        np.array(ax.get_yticks()),
        np.array([-10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    )
    plt.close()


# handball
@pytest.mark.plot
def test_plot_handball_pitch_show_axis_ticks_default(
    example_input_plot_handball_pitch,
) -> None:
    # Act
    ax = plot_handball_pitch(*example_input_plot_handball_pitch)
    # Assert
    assert ax.get_xticks() == []
    assert ax.get_yticks() == []
    plt.close()


@pytest.mark.plot
def test_plot_handball_pitch_show_axis_ticks_True(
    example_input_plot_handball_pitch_axis_ticks,
) -> None:
    # Act
    ax = plot_handball_pitch(*example_input_plot_handball_pitch_axis_ticks)
    # Assert
    assert np.array_equal(
        np.array(ax.get_xticks()), np.array([-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    )
    assert np.array_equal(
        np.array(ax.get_yticks()),
        np.array([-2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5]),
    )
    plt.close()

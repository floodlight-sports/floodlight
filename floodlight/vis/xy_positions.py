from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def handle_ax(func):
    """Decorator function that handles if matplotlib.axes are given as an argument or
        not.

    Parameters
    ---------
    func: xy_positions.<method>
        Function object that needs a matplotlib.axes as an argument. If ax == None an
        axes is created an passed to func as a keyworded argument.

    Returns
    ------
    func: xy_positions.<method>()
        Decorated function.
    """

    def wrapper(*args, **kwargs):

        if not kwargs.get("ax"):  # If no matplotlib.axes is not given (ax == None)
            kwargs.pop("ax")  # an axes is created.
            ax = plt.subplots()[1]
            return func(*args, ax=ax, **kwargs)

        return func(*args, **kwargs)  # If matplotlib.axes is given nothing changes and
        # the function is returned with the given *args
        # and **kwargs

    return wrapper


@handle_ax
def plot_positions(
    xy: np.ndarray, frame: int, ball: bool, ax: matplotlib.axes, **kwargs
) -> matplotlib.axes:
    """Plots positions for a given frame of the xy-array on a matplotlib.axes.

    Parameters
    -----------
    xy: np.ndarray
        Full data array containing x- and y-coordinates, where each player's
        coordinates occupy two consecutive columns.
    frame: int
        Frame for which the positions are plotted.
    ball: bool
        If set to False marker="o". Else marker="."
    ax: matplotlib.axes
        Axes from matplotlib library on which the positions are plotted.
    kwargs:
        Optional keyworded arguments e.g. {'color', 'zorder', 'marker'}
        which can be used for the plot functions from matplotlib. The kwargs are
        only passed to all the plot functions of matplotlib.

    Returns
    -------
    matplotib.axes
        A matplotlib.axes on which x and y-positions of a given frame from the
        xy-array are plotted.
    """

    # kwargs which are used to configure the plot with default values.
    # All other kwargs are just getting passed to the scatter() method.
    marker = kwargs.pop("marker", "o" if not ball else ".")
    color = kwargs.pop("color", "black" if not ball else "grey")
    zorder = kwargs.pop("zorder", 1)

    # plotting the positions
    # if ball is false
    if not ball:
        ax.scatter(
            x=xy[frame, ::2],
            y=xy[frame, 1::2],
            marker=marker,
            color=color,
            zorder=zorder,
            **kwargs,
        )
    # if ball is true
    elif ball:
        ax.scatter(
            x=xy[frame, ::2],
            y=xy[frame, 1::2],
            marker=marker,
            color=color,
            zorder=zorder,
            **kwargs,
        )

    return ax


@handle_ax
def plot_trajectories(
    xy: np.ndarray,
    frame_range: Tuple[int, int],
    ball: bool,
    ax: matplotlib.axes,
    **kwargs,
) -> matplotlib.axes:
    """Plots trajectories for a given range of frames  of the xyy-array on a
    matplotlib.axes.

    Parameters
    -----------
    xy: np.ndarray
        Full data array containing x- and y-coordinates, where each player's
        coordinates occupy two consecutive columns.
    frame_range: Tuple[int, int]
        From frame_range[0] to frame_range[1] a trajectory is plotted.
    ball: bool
        If set to False marker="o". Else marker="."
    ax: matplotlib.axes
        Axes from matplotlib library on which the positions are plotted.
    kwargs:
        Optional keyworded arguments e.g.{'linewidth', 'zorder', 'linestyle',
        'alpha'} which can be used for the plot functions from matplotlib.
        The kwargs are only passed to all the plot functions of matplotlib.

    Returns
    -------
    matplotib.axes
        A matplotlib.axes on which trajectories of a given frame from the
        xy-array are plotted.
    """

    # kwargs which are used to configure the plot with default values.
    # All other kwargs are just getting passed to the plot() method.
    color = kwargs.pop("color", "black")
    zorder = kwargs.pop("zorder", 1)
    # ball trajectories are thinner per default
    linewidth = kwargs.pop("linewidth", 1 if not ball else 0.5)

    # iterating over every object (for instance players) in the xy-array and plot the
    # movement over the given frame_range
    for i in range(0, len(xy[0]), 2):
        x = xy[frame_range[0] : frame_range[1], i]
        y = xy[frame_range[0] : frame_range[1], i + 1]
        ax.plot(x, y, color=color, zorder=zorder, linewidth=linewidth, **kwargs)

    return ax

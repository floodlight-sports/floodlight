import matplotlib
import numpy as np


def plot_positions(
    xy: np.ndarray, frame: int, ball: bool, ax: matplotlib.axes, **kwargs
) -> matplotlib.axes:
    """
        Plots positions for a given frame on a matplotlib.axes.
    plot
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

        Returns
        -------
        matplotib.axes
            A matplotlib.axes on which x and y-positions of a given frame are plotted.
    """

    # kwargs which are used to configure the plot with default values.
    # All other kwargs are just getting passed to the scatter() method.
    marker = kwargs.pop("marker", "o" if not ball else ".")
    color = kwargs.pop("color", "black")
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

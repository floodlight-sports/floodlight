from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Union

import matplotlib
import numpy as np

from floodlight.utils.types import Numeric
from floodlight.vis.positions import plot_positions, plot_trajectories


@dataclass
class XY:
    """Spatio-temporal data fragment. Core class of floodlight.

    Parameters
    ----------
    xy: np.ndarray
        Full data array containing x- and y-coordinates, where each player's coordinates
        occupy two consecutive columns.
    framerate: int, optional
        Temporal resolution of data in frames per second/Hertz.
    direction: {'lr', 'rl'}, optional
        Playing direction of players in data fragment, should be either
        'lr' (left-to-right) or 'rl' (right-to-left).

    Attributes
    ----------
    x: np.array
        X-data array, where each player's x-coordinates occupy one column.
    y: np.array
        Y-data array, where each player's y-coordinates occupy one column.
    N: int
        The object's number of players.
    """

    xy: np.ndarray
    framerate: int = None
    direction: str = None

    def __str__(self):
        return f"Floodlight XY object of shape {self.xy.shape}"

    def __len__(self):
        return len(self.xy)

    def __getitem__(self, key):
        return self.xy[key]

    def __setitem__(self, key, value):
        self.xy[key] = value

    @property
    def N(self) -> int:
        n_columns = self.xy.shape[1]
        if (n_columns % 2) != 0:
            raise ValueError(f"XY has an odd number of columns ({n_columns})")
        return n_columns // 2

    @property
    def x(self) -> np.array:
        """X-data array, where each player's x-coordinates occupy one column."""
        return self.xy[:, ::2]

    @x.setter
    def x(self, x_data: np.ndarray):
        self.xy[:, ::2] = x_data

    @property
    def y(self) -> np.array:
        """Y-data array, where each player's y-coordinates occupy one column."""
        return self.xy[:, 1::2]

    @y.setter
    def y(self, y_data: np.ndarray):
        self.xy[:, 1::2] = y_data

    def frame(self, t: int) -> np.ndarray:
        """Returns data for given frame *t*.

        Parameters
        ----------
        t : int
            Frame index.

        Returns
        -------
        frame : np.ndarray
            One-dimensional xy-data row for given frame.
        """
        return self.xy[t, :]

    def player(self, xID: int) -> np.ndarray:
        """Returns data for player with given player index *xID*.

        Parameters
        ----------
        xID : int
            Player index.

        Returns
        -------
        player : np.ndarray
            Two-dimensional xy-data for given player.

        """
        return self.xy[:, xID * 2 : xID * 2 + 2]

    def point(self, t: int, xID: int) -> np.ndarray:
        """Returns data for a point determined by frame *t* and player index *xID*.

        Parameters
        ----------
        t: int
            Frame index.
        xID: int
            Player index.

        Returns
        -------
        point : np.ndarray
            Point-data of shape (2,)
        """
        return self.xy[t, xID * 2 : xID * 2 + 2]

    def translate(self, shift: Tuple[Numeric, Numeric]):
        """Translates data by shift vector.

        Parameters
        ----------
        shift : list or array-like
            Shift vector of form v = (x, y). Any iterable data type with two numeric
            entries is accepted.

        Notes
        -----
        Executing this method will cast the object's xy attribute to dtype np.float32 if
        it previously has a non-floating dtype.
        """
        # cast to float
        if self.xy.dtype not in [np.float_, np.float64, np.float32, float]:
            self.xy = self.xy.astype(np.float32, copy=False)

        self.x = np.round(self.x + shift[0], 3)
        self.y = np.round(self.y + shift[1], 3)

    def scale(self, factor: float, axis: str = None):
        """Scales data by a given factor and optionally selected axis.

        Parameters
        ----------
        factor : float
            Scaling factor.
        axis : {None, 'x', 'y'}, optional
            Name of scaling axis. If set to 'x' data is scaled on x-axis, if set to 'y'
            data is scaled on y-axis. If None, data is scaled in both directions
            (default).

        Notes
        -----
        Executing this method will cast the object's xy attribute to dtype np.float32 if
        it previously has a non-floating dtype.
        """
        # cast to float
        if self.xy.dtype not in [np.float_, np.float64, np.float32, float]:
            self.xy = self.xy.astype(np.float32, copy=False)

        if axis is None:
            self.xy = np.round(self.xy * factor, 3)
        elif axis == "x":
            self.x = np.round(self.x * factor, 3)
        elif axis == "y":
            self.y = np.round(self.y * factor, 3)
        else:
            raise ValueError(f"Expected axis to be one of ('x', 'y', None), got {axis}")

    def reflect(self, axis: str):
        """Reflects data on given `axis`.

        Parameters
        ----------
        axis : {'x', 'y'}
            Name of reflection axis. If set to "x", data is reflected on x-axis,
            if set to "y", data is reflected on y-axis.
        """
        if axis == "x":
            self.scale(factor=-1, axis="y")
        elif axis == "y":
            self.scale(factor=-1, axis="x")
        else:
            raise ValueError(f"Expected axis to be one of ('x', 'y'), got {axis}")

    def rotate(self, alpha: float):
        """Rotates data on given angle 'alpha' around the origin.

        Parameters
        ----------
        alpha: float
            Rotation angle in degrees. Alpha must be between -360 and 360. If positive
            alpha, data is rotated in counter clockwise direction around the origin. If
            negative, data is rotated in clockwise direction around the origin.

        Notes
        -----
        Executing this method will cast the object's xy attribute to dtype np.float32 if
        it previously has a non-floating dtype.
        """
        if not (-360 <= alpha <= 360):
            raise ValueError(
                f"Expected alpha to be from -360 to 360, got {alpha} instead"
            )
        # cast to float
        if self.xy.dtype not in [np.float_, np.float64, np.float32, float]:
            self.xy = self.xy.astype(np.float32, copy=False)

        # construct rotation matrix
        phi = np.radians(alpha)
        cos = np.cos(phi)
        sin = np.sin(phi)
        r = np.array([[cos, -sin], [sin, cos]]).transpose()

        # perform player-wise rotation - this correctly handles nan's compared to
        # block matrix approach
        for p in range(self.N):
            columns = (p * 2, p * 2 + 1)
            self.xy[:, columns] = np.round(self.xy[:, columns] @ r, 3)

    def slice(
        self, startframe: int = None, endframe: int = None, inplace: bool = False
    ):
        """Return copy of object with sliced data. Mimics numpy's array slicing.

        Parameters
        ----------
        startframe : int, optional
            Start of slice. Defaults to beginning of segment.
        endframe : int, optional
            End of slice (endframe is excluded). Defaults to end of segment.
        inplace: bool, optional
            If set to ``False`` (default), a new object is returned, otherwise the
            operation is performed in place on the called object.

        Returns
        -------
        xy_sliced: Union[XY, None]
        """
        sliced_data = self.xy[startframe:endframe, :].copy()
        xy_sliced = None

        if inplace:
            self.xy = sliced_data
        else:
            xy_sliced = XY(
                xy=sliced_data,
                framerate=deepcopy(self.framerate),
                direction=deepcopy(self.direction),
            )

        return xy_sliced

    def plot(
        self,
        t: Union[int, Tuple[int, int]],
        plot_type: str = "positions",
        ball: bool = False,
        ax: matplotlib.axes = None,
        **kwargs,
    ) -> matplotlib.axes:
        """Plots a snapshot or time intervall of the object's spatiotemporal data on a
        matplotlib axes.

        Parameters
        ----------
        t: Union[int, Tuple [int, int]]
            Frame for which postions should be plotted if plot_type == 'positions', or a
            Tuple that has the form (start_frame, end_frame) if
            plot_type == 'trajectories'.
        plot_type: str, optional
            One of {'positions', 'trajectories'}. Determines which plotting function is
            called. Defaults to 'positions'.
        ball: bool, optional
            Boolean indicating whether this object is storing ball data. If set to True,
            the styling is adjusted accordingly. Defaults to False.
        ax: matplotlib.axes, optional
            Axes from matplotlib library to plot on. Defaults to None.
        kwargs:
            Optional keyworded arguments e.g. {'color', 'zorder', 'marker', 'linestyle',
            'alpha'} which can be used for the plot functions from matplotlib.
            The kwargs are only passed to all the plot functions of matplotlib. If not
            given default values are used (see floodlight.vis.positions).


        Returns
        -------
        axes: matplotlib.axes
            Axes from matplotlib library on which the specified plot type is plotted.

        Notes
        -----
        The kwargs are only passed to the plot functions of matplotlib. To customize the
        plots have a look at
        `matplotlib
        <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.
        For example in order to modify the color of the points and lines pass a color
        name or rgb-value (`matplotlib colors
        <https://matplotlib.org/3.5.0/tutorials/colors/colors.html>`_) to the keyworded
        argument 'color'. The same principle applies to other kwargs like 'zorder',
        'marker' and 'linestyle'.

        Examples
        --------
        - :ref:`Positions plot <positions-plot-label>`
        - :ref:`Trajectories plot <trajectories-plot-label>`

        """

        plot_types = ["positions", "trajectories"]

        # call visualization function based on plot_type
        if plot_type == "positions":
            return plot_positions(self, t, ball, ax=ax, **kwargs)
        elif plot_type == "trajectories":
            return plot_trajectories(self, t[0], t[1], ball, ax=ax, **kwargs)
        else:
            raise ValueError(
                f"Expected plot_type to be one of {plot_types}, got {plot_type} "
                "instead."
            )

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

    Attributes
    ----------
    xy: np.ndarray
        Full data array containing x- and y-coordinates, where each player's coordinates
        occupy two consecutive columns.
    x: np.ndarray
        X-data array, where each player's x-coordinates occupy one column.
    y: np.ndarray
        Y-data array, where each player's y-coordinates occupy one column.
    framerate: int, optional
        Temporal resolution of data in frames per second/Hertz.
    direction: str, optional
        Playing direction of players in data fragment, should be either
        'lr' (left-to-right) or 'rl' (right-to-left).
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
    def x(self):
        return self.xy[:, ::2]

    @x.setter
    def x(self, x_data: np.ndarray):
        self.xy[:, ::2] = x_data

    @property
    def y(self):
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
        """
        self.x += shift[0]
        self.y += shift[1]

    def scale(self, factor: float, axis: int = None):
        """Scales data by a given factor and optionally selected axis.

        Parameters
        ----------
        factor : float
            Scaling factor.
        axis : int, optional
            Index of scaling axis. If set to 0, data is scaled on x-axis, if set to 1,
            data is scaled on y-axis. If none is provided, data is scaled in both
            directions (default).
        """
        if axis is None:
            self.xy *= factor
        elif axis == 0:
            self.x *= factor
        elif axis == 1:
            self.y *= factor
        else:
            raise ValueError(f"Expected axis to be one of {0, 1, None}, got {axis}")

    def reflect(self, axis: int):
        """Reflects data on given `axis`.

        Parameters
        ----------
        axis : int
            Index of reflection axis. If set to 0, data is reflected on x-axis,
            if set to 1, data is reflected on y-axis.
        """
        if axis == 0:
            self.scale(factor=-1, axis=1)
        elif axis == 1:
            self.scale(factor=-1, axis=0)
        else:
            raise ValueError(f"Expected axis to be one of {0, 1}, got {axis}")

    def rotate(self, alpha: float):
        """Rotates data on given angle 'alpha' around the origin.

        Parameters
        ----------
        alpha: float
            Rotation angle in degrees. Alpha must be between -360 and 360. If positive
            alpha, data is rotated in counter clockwise direction around the origin. If
            negative, data is rotated in clockwise direction around the origin.
        """
        if not (-360 <= alpha <= 360):
            raise ValueError(
                f"Expected alpha to be from -360 to 360, got {alpha} instead"
            )

        phi = np.radians(alpha)
        cos = np.cos(phi)
        sin = np.sin(phi)

        # construct rotation matrix
        r = np.array([[cos, -sin], [sin, cos]]).transpose()

        # construct block-diagonal rotation matrix to match number of players
        n_players = int(self.xy.shape[1] / 2)
        r_diag = np.kron(np.eye(n_players), r)

        # perform rotation
        self.xy = np.round(np.dot(self.xy, r_diag), 3)

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
        """Wrapper function that calls the actual plotting functions based on the
        selected plot type.

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
            If ball == True the points and lines are adjusted accordingly.
            Defaults to False.
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
            Specified plot function which returns a matplotlib.axes object.

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
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from floodlight.core.pitch import Pitch
        >>> from floodlight.core.xy import XY

        >>> # positions
        >>> pos = np.array(
        >>>     [[35,5,35,63,25,25,25,50],
        >>>     [45,10,45,55,35,20,35,45],
        >>>     [55,10,55,55,45,20,45,45],
        >>>     [88.5,20,88.5,30,88.5,40,88.5,50]])

        >>> xy_pos = XY(pos) # create XY object

        >>> # create Pitch object
        >>> football_pitch = Pitch(xlim=(0,105), ylim=(0, 68), unit="m",
        >>> sport="football")

        >>> # create matplotlib.axes
        >>> ax = plt.subplots()[1]

        >>> # plot pitch on ax
        >>> football_pitch.plot(color_scheme="standard", ax=ax)

        >>> # plot positions on ax
        >>> xy_pos.plot(plot_type="positions", t=0, ax=ax)
        >>> plt.show()

        .. image:: ../../_img/positions_example.png

        >>> # plot trajectories from frame 0 to 4 on ax
        >>> xy_pos.plot(plot_type="trajectories", t=(0,4), ball= False, ax=ax)
        >>> plt.show()

        .. image:: ../../_img/trajectories_example.png

        """

        plot_types = ["positions", "trajectories"]

        # call visualization function based on plot_type
        if plot_type == "positions":
            return plot_positions(self, t, ball, ax=ax, **kwargs)
        elif plot_type == "trajectories":
            return plot_trajectories(self, t[0], t[1], ball, ax=ax, **kwargs)
        else:
            raise ValueError(
                f"Unknown plot type: {plot_type}, choose one of " f"{plot_types}"
            )

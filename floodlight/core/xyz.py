from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Union

import matplotlib
import numpy as np

from floodlight.utils.types import Numeric
from floodlight.vis.positions import plot_positions, plot_trajectories


@dataclass
class XYZ:
    """Spatio-temporal data fragment. Core class of floodlight.

    Parameters
    ----------
    xyz: np.ndarray
        Full data array containing x-, y- and z-coordinates, where each player's coordinates
        occupy three consecutive columns.
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
    z: np.array
        Z-data array, where each player's z-coordinates occupy one column.
    N: int
        The object's number of players.
    """

    xyz: np.ndarray
    framerate: int = None
    direction: str = None

    def __str__(self):
        return f"Floodlight XYZ object of shape {self.xyz.shape}"

    def __len__(self):
        return len(self.xyz)

    def __getitem__(self, key):
        return self.xyz[key]

    def __setitem__(self, key, value):
        self.xyz[key] = value

    @property
    def N(self) -> int:
        n_columns = self.xyz.shape[1]
        if (n_columns % 3) != 0:
            raise ValueError(f"XYZ has an odd number of columns ({n_columns})")
        return n_columns // 3

    @property
    def x(self) -> np.array:
        """X-data array, where each player's x-coordinates occupy one column."""
        return self.xyz[:, ::3]

    @x.setter
    def x(self, x_data: np.ndarray):
        self.xyz[:, ::3] = x_data

    @property
    def y(self) -> np.array:
        """Y-data array, where each player's y-coordinates occupy one column."""
        return self.xyz[:, 1::3]

    @y.setter
    def y(self, y_data: np.ndarray):
        self.xyz[:, 1::3] = y_data

    @property
    def z(self) -> np.array:
        """Z-data array, where each player's z-coordinates occupy one column."""
        return self.xyz[:, 2::3]

    @z.setter
    def z(self, z_data: np.ndarray):
        self.xyz[:, 2::3] = z_data

    def frame(self, t: int) -> np.ndarray:
        """Returns data for given frame *t*.

        Parameters
        ----------
        t : int
            Frame index.

        Returns
        -------
        frame : np.ndarray
            One-dimensional xyz-data row for given frame.
        """
        return self.xyz[t, :]

    def player(self, xID: int) -> np.ndarray:
        """Returns data for player with given player index *xID*.

        Parameters
        ----------
        xID : int
            Player index.

        Returns
        -------
        player : np.ndarray
            Three-dimensional xyz-data for given player.

        """
        return self.xyz[:, xID * 3 : xID * 3 + 3]

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
            Point-data of shape (3,)
        """
        return self.xyz[t, xID * 3 : xID * 3 + 3]
    def translate(self, shift: Tuple[Numeric, Numeric, Numeric]):
        """Translates data by shift vector.

        Parameters
        ----------
        shift : list or array-like
            Shift vector of form v = (x, y, z). Any iterable data type with three numeric
            entries is accepted.

        Notes
        -----
        Executing this method will cast the object's xyz attribute to dtype np.float32 if
        it previously has a non-floating dtype.
        """
        # cast to float
        if self.xyz.dtype not in [np.float_, np.float64, np.float32, float]:
            self.xyz = self.xyz.astype(np.float32, copy=False)

        self.x = np.round(self.x + shift[0], 3)
        self.y = np.round(self.y + shift[1], 3)
        self.z = np.round(self.z + shift[2], 3)

    def scale(self, factor: float, axis: str = None):
        """Scales data by a given factor and optionally selected axis.

        Parameters
        ----------
        factor : float
            Scaling factor.
        axis : {None, 'x', 'y', 'z'}, optional
            Name of scaling axis. If set to 'x' data is scaled on x-axis, if set to 'y'
            data is scaled on y-axis, if set to 'z' data is scaled on z-axis.
            If None, data is scaled in all directions (default).

        Notes
        -----
        Executing this method will cast the object's xyz attribute to dtype np.float32 if
        it previously has a non-floating dtype.
        """
        # cast to float
        if self.xyz.dtype not in [np.float_, np.float64, np.float32, float]:
            self.xyz = self.xyz.astype(np.float32, copy=False)

        if axis is None:
            self.xyz = np.round(self.xyz * factor, 3)
        elif axis == "x":
            self.x = np.round(self.x * factor, 3)
        elif axis == "y":
            self.y = np.round(self.y * factor, 3)
        elif axis == "z":
            self.z = np.round(self.z * factor, 3)
        else:
            raise ValueError(f"Expected axis to be one of ('x', 'y', 'z', None), got {axis}")

    def reflect(self, axis: str):
        """Reflects data on given `axis`.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Name of reflection axis. If set to "x", data is reflected on x-axis,
            if set to "y", data is reflected on y-axis, if set to "z", data is reflected on z-axis.
        """
        if axis == "x":
            self.scale(factor=-1, axis="y")
        elif axis == "y":
            self.scale(factor=-1, axis="x")
        elif axis == "z":
            self.scale(factor=-1, axis="z")
        else:
            raise ValueError(f"Expected axis to be one of ('x', 'y', 'z'), got {axis}")

    def rotate(self, alpha: float, axis: str = 'z'):
        """Rotates data on given angle 'alpha' around the origin in a specified plane.

        Parameters
        ----------
        alpha: float
            Rotation angle in degrees. Alpha must be between -360 and 360. If positive
            alpha, data is rotated in counter clockwise direction around the origin. If
            negative, data is rotated in clockwise direction around the origin.
        axis: str, optional
            The axis of rotation, should be one of {'x', 'y', 'z'}. Defaults to 'z'.

        Notes
        -----
        Executing this method will cast the object's xyz attribute to dtype np.float32 if
        it previously has a non-floating dtype.
        """
        if not (-360 <= alpha <= 360):
            raise ValueError(
                f"Expected alpha to be from -360 to 360, got {alpha} instead"
            )
        # cast to float
        if self.xyz.dtype not in [np.float_, np.float64, np.float32, float]:
            self.xyz = self.xyz.astype(np.float32, copy=False)

        # construct rotation matrix
        phi = np.radians(alpha)
        cos = np.cos(phi)
        sin = np.sin(phi)

        if axis == 'z':
            r = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]).transpose()
        elif axis == 'y':
            r = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]).transpose()
        elif axis == 'x':
            r = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]]).transpose()
        else:
            raise ValueError(f"Expected axis to be one of ('x', 'y', 'z'), got {axis}")

        # perform player-wise rotation
        for p in range(self.N):
            columns = (p * 3, p * 3 + 1, p * 3 + 2)
            self.xyz[:, columns] = np.round(self.xyz[:, columns] @ r, 3)

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
        xyz_sliced: Union[XYZ, None]
        """
        sliced_data = self.xyz[startframe:endframe, :].copy()
        xyz_sliced = None

        if inplace:
            self.xyz = sliced_data
        else:
            xyz_sliced = XYZ(
                xyz=sliced_data,
                framerate=deepcopy(self.framerate),
                direction=deepcopy(self.direction),
            )

        return xyz_sliced


    def plot(
        self,
        t: Union[int, Tuple[int, int]],
        plot_type: str = "positions",
        ball: bool = False,
        ax: matplotlib.axes = None,
        **kwargs,
    ) -> matplotlib.axes:
        """Plots a snapshot or time interval of the object's spatiotemporal data on a
        matplotlib axes.

        Parameters
        ----------
        t: Union[int, Tuple [int, int]]
            Frame for which positions should be plotted if plot_type == 'positions', or a
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
        This method needs to be adapted for 3D plotting. The current implementation
        may only work for 2D plotting.

        """

        plot_types = ["positions", "trajectories"]

        # 3D plotting needs to be implemented here. The current code is for 2D.
        # Example adaptation for 3D plotting:
        # if plot_type == "positions":
        #     return plot_positions_3d(self, t, ball, ax=ax, **kwargs)
        # elif plot_type == "trajectories":
        #     return plot_trajectories_3d(self, t[0], t[1], ball, ax=ax, **kwargs)
        # else:
        #     raise ValueError(
        #         f"Expected plot_type to be one of {plot_types}, got {plot_type} "
        #         "instead."
        #     )

        # call visualization function based on plot_type
        if plot_type == "positions":
            # Placeholder for 3D positions plot function
            return plot_positions(self, t, ball, ax=ax, **kwargs)
        elif plot_type == "trajectories":
            # Placeholder for 3D trajectories plot function
            return plot_trajectories(self, t[0], t[1], ball, ax=ax, **kwargs)
        else:
            raise ValueError(
                f"Expected plot_type to be one of {plot_types}, got {plot_type} "
                "instead."
            )

# End of XYZ class

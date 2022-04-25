from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from floodlight.utils.types import Numeric


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

    @property
    def N(self) -> int:
        """Returns the object's number of players."""
        n_columns = self.xy.shape[1]
        if (n_columns % 2) != 0:
            raise ValueError(f"XY has an odd number of columns ({n_columns})")
        return n_columns // 2

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
        """
        if axis is None:
            self.xy *= factor
        elif axis == "x":
            self.x *= factor
        elif axis == "y":
            self.y *= factor
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

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from floodlight.utils.typing import Numeric


@dataclass
class XY:
    """
    Spatio-temporal data fragment. Core class of floodlight.

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
        """
        Returns data for given frame *t*.

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
        """
        Returns data for player with given player index *xID*.

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
        """
        Returns data for a point determined by frame *t* and player index *xID*.

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
        """"""
        pass

    def permute(self, i: int, j: int):
        """"""
        pass

    def downsample(self):
        pass

    def upsample(self):
        pass

    def check_nan(self):
        """"""
        pass

    def fill_nan(self):
        """"""
        pass

from dataclasses import dataclass

import numpy as np

from floodlight.utils.types import Numeric
from floodlight import Pitch, XY
from floodlight.core.property import PlayerProperty


@dataclass
class DistanceModel:
    """Class for calculating distances of players on the pitch.

    Parameters
    __________
    pitch: Pitch
        Pitch object of the data. Should at least include information about the data's
        unit. If pitch.unit is 'percent', length and width of the pitch are also needed.

    """

    pitch: Pitch = None
    length: Numeric = None
    width: Numeric = None
    unit: str = None

    def __init__(self, pitch):
        self.unit = pitch.unit
        self.length = pitch.length
        self.width = pitch.width
        self._distance_euclidean = None
        self.framerate = None

    def __str__(self):
        return (
            f"Floodlight ModelDistance Class with unit [{self.unit}],"
            f"length {self.length} and width {self.width}"
        )

    def fit(self, xy: XY, direction: str = None):
        """Fits a model to calculate euclidean distances to an xy object.

        Parameters
        ----------
        xy: XY
            Floodlight XY Data object.
        direction: str
            One of {'x', 'y', 'plane'} of None.
            Indicates the direction to calculate the distance in.
            'x' will only calculate distance in the direction of the x-coordinate.
            'y' will only calculate distance in the direction of the y-coordinate.
            'plane' or None will calculate distance in both directions.

        """

        distance_euclidean = None

        if self.unit == "m":
            unit_factor = (1, 1)
        elif self.unit == "cm":
            unit_factor = (100, 100)
        elif self.unit == "percent":
            unit_factor = (self.length / 100, self.width / 100)
        else:
            raise ValueError(
                "Invalid argument: 'unit' has to be 'm', 'cm' or 'percent'!"
            )

        if direction is None or direction == "plane":
            differences_xy = np.gradient(xy.xy, axis=0)
            distance_euclidean = np.hypot(
                differences_xy[:, ::2] * unit_factor[0],
                differences_xy[:, 1::2] * unit_factor[1],
            )
        elif direction == "x":
            distance_euclidean = np.gradient(xy.x * unit_factor[0], axis=0)
        elif direction == "y":
            distance_euclidean = np.gradient(xy.y * unit_factor[1], axis=0)
        else:
            raise ValueError(
                "Invalid argument: 'direction' has to be 'plane', 'x', 'y' or None!"
            )

        self._distance_euclidean = distance_euclidean
        if xy.framerate is not None:
            self.framerate = xy.framerate

    @property
    def distance(self):
        distance = PlayerProperty(
            property=self._distance_euclidean, name="distance", framerate=self.framerate
        )
        return distance

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

    def fit(self, xy: XY, difference: str = "central", direction: str = "plane"):
        """Fits a model to calculate euclidean distances to an xy object.

        Parameters
        ----------
        xy: XY
            Floodlight XY Data object.
        difference: str
            The method of differentiation. One of {'central', 'forward'}.\n
            'central' will differentiate using the central difference method:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{1}-y_{-1}}{t_{1} - t_{-1}}

            'forward' will differentiate using the forward difference method and append
            a '0' at the end of the array along axis 1:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{1}-y_{0}}{t_{1} - t_{0}}


        direction: str
            One of {'x', 'y', 'plane'}.
            Indicates the direction to calculate the distance in.
            'x' will only calculate distance in the direction of the x-coordinate.
            'y' will only calculate distance in the direction of the y-coordinate.
            'plane' or None will calculate distance in both directions.

        """

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

        if direction == "plane":
            if difference == "central":
                differences_xy = np.gradient(xy.xy, axis=0)
            elif difference == "forward":
                differences_xy = np.diff(xy.xy, axis=0, append=0)
            else:
                raise ValueError(
                    "Invalid argument: 'difference' has to be 'central' or 'forward'!"
                )
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
                "Invalid argument: 'direction' has to be 'plane', 'x', 'y'!"
            )

        self._distance_euclidean = distance_euclidean
        if xy.framerate is not None:
            self.framerate = xy.framerate

    def distance_covered(self):
        distance = PlayerProperty(
            property=self._distance_euclidean,
            name="distance_covered",
            framerate=self.framerate,
        )
        return distance

    def cumulative_distance_covered(self):
        cum_dist = np.nancumsum(self._distance_euclidean, axis=0)
        cumulative_distance = PlayerProperty(
            property=cum_dist,
            name="cumulative_distance_covered",
            framerate=self.framerate,
        )
        return cumulative_distance


@dataclass
class VelocityModel:
    """Class for calculating velocities of players on the pitch.

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
        self.pitch = pitch
        self._velocity = None
        self.framerate = None

    def __str__(self):
        return (
            f"Floodlight VelocityModel Class with unit [{self.unit}],"
            f"length {self.length} and width {self.width}"
        )

    def fit(
        self,
        xy: XY,
        difference: str = "central",
        direction: str = "plane",
        velocity_unit: str = "m/s",
    ):
        """Fits a model to calculate euclidean distances to an xy object.

        Parameters
        ----------
        xy: XY
            Floodlight XY Data object.
        difference: str
            The method of differentiation. One of {'central', 'forward'}.\n
            'central' will differentiate using the central difference method:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{1}-y_{-1}}{t_{1} - t_{-1}}

            'forward' will differentiate using the forward difference method and append
            a '0' at the end of the array along axis 1:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{1}-y_{0}}{t_{1} - t_{0}}

        direction: str
            One of {'x', 'y', 'plane'}.
            Indicates the direction to calculate the distance in.
            'x' will only calculate distance in the direction of the x-coordinate.
            'y' will only calculate distance in the direction of the y-coordinate.
            'plane' or None will calculate distance in both directions.
        velocity_unit: str
            One of {"m/s", "km/h"}.
            Unit in which the velocity is calculated.
            "m/s" will calculate the velocity in meters per second.
            "km/h" will calculate the velocity in kilometers per hour.

        """

        distance_model = DistanceModel(self.pitch)
        distance_model.fit(xy, difference=difference, direction=direction)
        distance_euclidean = distance_model.distance_covered()

        if velocity_unit == "m/s":
            velocity = distance_euclidean.property * distance_euclidean.framerate
        elif velocity_unit == "km/h":
            velocity = (
                distance_euclidean.property * distance_euclidean.framerate
            ) * 3.6
        else:
            raise ValueError(
                "Invalid argument: 'velocity_unit' has to be 'm/s' or 'km/h'!"
            )

        self._velocity = velocity
        self.framerate = distance_euclidean.framerate

    def velocity(self):
        velocity = PlayerProperty(
            property=self._velocity,
            name="velocity",
            framerate=self.framerate,
        )
        return velocity


@dataclass
class AccelerationModel:
    """Class for calculating accelerations of players on the pitch.

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
        self.pitch = pitch
        self._acceleration = None
        self.framerate = None

    def __str__(self):
        return (
            f"Floodlight AccelerationModel Class with unit [{self.unit}],"
            f"length {self.length} and width {self.width}"
        )

    def fit(
        self,
        xy: XY,
        difference: str = "central",
        direction: str = "plane",
    ):
        """Fits a model to calculate euclidean distances to an xy object.

        Parameters
        ----------
        xy: XY
            Floodlight XY Data object.
        difference: str
            The method of differentiation. One of {'central', 'forward'}.\n
            'central' will differentiate using the central difference method:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{1}-y_{-1}}{t_{1} - t_{-1}}

            'forward' will differentiate using the forward difference method and append
            a '0' at the end of the array along axis 1:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{1}-y_{0}}{t_{1} - t_{0}}

        direction: str
            One of {'x', 'y', 'plane'}.
            Indicates the direction to calculate the distance in.
            'x' will only calculate distance in the direction of the x-coordinate.
            'y' will only calculate distance in the direction of the y-coordinate.
            'plane' or None will calculate distance in both directions.
        """

        velocity_model = VelocityModel(self.pitch)
        velocity_model.fit(xy, difference=difference, direction=direction)
        velocity = velocity_model.velocity()

        if difference == "central":
            acceleration = np.gradient(velocity.property, axis=0) * velocity.framerate
        else:
            acceleration = (
                np.diff(velocity.property, axis=0, append=0) * velocity.framerate
            )

        self._acceleration = acceleration
        self.framerate = velocity.framerate

    def acceleration(self):
        acceleration = PlayerProperty(
            property=self._acceleration,
            name="acceleration",
            framerate=self.framerate,
        )
        return acceleration

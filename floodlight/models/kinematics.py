import warnings

import numpy as np

from floodlight import Pitch, XY
from floodlight.core.property import PlayerProperty


class DistanceModel:
    """Class for calculating distances of players on the pitch.

    Parameters
    __________
    pitch: Pitch
        Pitch object of the data.

    """

    def __init__(self, pitch: Pitch):
        self.pitch = pitch
        self._framerate = None
        self._distance_euclidean = None

    def fit(self, xy: XY, difference: str = "central", axis: str = None):
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


        axis: str or None
            One of {'x', 'y', None}.
            Indicates the direction to calculate the distance in.
            'x' will only calculate distance in the direction of the x-coordinate.
            'y' will only calculate distance in the direction of the y-coordinate.
            None will calculate distance in the x/y-plane.

        """

        self._framerate = xy.framerate

        if self.pitch.unit == "percent":
            raise warnings.warn(
                "Pitch unit is percent. Euclidean calculations may be distorted!"
            )

        if axis is None:
            if difference == "central":
                differences_xy = np.gradient(xy.xy, axis=0)
            elif difference == "forward":
                differences_xy = np.diff(xy.xy, axis=0, append=0)
            else:
                raise ValueError(
                    "Invalid argument: 'difference' has to be 'central' or 'forward'!"
                )
            distance_euclidean = np.hypot(
                differences_xy[:, ::2],
                differences_xy[:, 1::2],
            )
        elif axis == "x":
            distance_euclidean = np.gradient(xy.x, axis=0)
        elif axis == "y":
            distance_euclidean = np.gradient(xy.y, axis=0)
        else:
            raise ValueError("Invalid argument: 'axis' has to be None, 'x' or 'y'!")

        self._distance_euclidean = distance_euclidean

    def distance_covered(self):
        distance = PlayerProperty(
            property=self._distance_euclidean,
            name="distance_covered",
            framerate=self._framerate,
        )
        return distance

    def cumulative_distance_covered(self):
        cum_dist = np.nancumsum(self._distance_euclidean, axis=0)
        cumulative_distance = PlayerProperty(
            property=cum_dist,
            name="cumulative_distance_covered",
            framerate=self._framerate,
        )
        return cumulative_distance


class VelocityModel:
    """Class for calculating velocities of players on the pitch.

    Parameters
    __________
    pitch: Pitch
        Pitch object of the data. Should at least include information about the data's
        unit. If pitch.unit is 'percent', length and width of the pitch are also needed.

    """

    def __init__(self, pitch):
        self.pitch = pitch
        self._framerate = None
        self._velocity = None

    def fit(
        self,
        xy: XY,
        difference: str = "central",
        axis: str = None,
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

        axis: str or None
            One of {'x', 'y', None}.
            Indicates the direction to calculate the distance in.
            'x' will only calculate distance in the direction of the x-coordinate.
            'y' will only calculate distance in the direction of the y-coordinate.
            None will calculate distance in both directions.
        """

        self._framerate = xy.framerate

        distance_model = DistanceModel(self.pitch)
        distance_model.fit(xy, difference=difference, axis=axis)
        distance_euclidean = distance_model.distance_covered()

        velocity = distance_euclidean.property * distance_euclidean.framerate

        self._velocity = velocity

    def velocity(self):
        velocity = PlayerProperty(
            property=self._velocity,
            name="velocity",
            framerate=self._framerate,
        )
        return velocity


class AccelerationModel:
    """Class for calculating accelerations of players on the pitch.

    Parameters
    __________
    pitch: Pitch
        Pitch object of the data. Should at least include information about the data's
        unit. If pitch.unit is 'percent', length and width of the pitch are also needed.

    """

    def __init__(self, pitch):
        self.pitch = pitch
        self._framerate = None
        self._acceleration = None

    def fit(
        self,
        xy: XY,
        difference: str = "central",
        axis: str = None,
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

        axis: str
            One of {'x', 'y', None}.
            Indicates the direction to calculate the distance in.
            'x' will only calculate distance in the direction of the x-coordinate.
            'y' will only calculate distance in the direction of the y-coordinate.
            None will calculate distance in both directions.
        """

        self._framerate = xy.framerate

        velocity_model = VelocityModel(self.pitch)
        velocity_model.fit(xy, difference=difference, axis=axis)
        velocity = velocity_model.velocity()

        if difference == "central":
            acceleration = np.gradient(velocity.property, axis=0) * velocity.framerate
        else:
            acceleration = (
                np.diff(velocity.property, axis=0, append=0) * velocity.framerate
            )

        self._acceleration = acceleration

    def acceleration(self):
        acceleration = PlayerProperty(
            property=self._acceleration,
            name="acceleration",
            framerate=self._framerate,
        )
        return acceleration

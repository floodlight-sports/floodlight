import numpy as np

from floodlight import XY
from floodlight.core.property import PlayerProperty
from floodlight.models.base import BaseModel


class DistanceModel(BaseModel):
    """Computations for euclidean distances of all players.

    Upon calling the :func:`~DistanceModel.fit`-method, this model calculates the
    framewise euclidean distance for each player. The following calculations can
    subsequently be queried by calling the corresponding methods:

        - Frame-wise Distance Covered --> :func:`~DistanceModel.distance_covered`
        - Cumulative Distance Covered -->
        :func:`~DistanceModel.cumulative_distance_covered`

    Notes
    -----
    The model calculates the distance in the unit of the corresponding position data.
    This may lead to distortions when used in combination with models that use SI units
    or when the unit is percent.

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.kinematics import DistanceModel

    >>> xy = XY(np.array(((0, 0), (0, 1), (1, 1), (2, 2))))

    >>> dm = DistanceModel()
    >>> dm.fit(xy)

    >>> dm.distance_covered()
    PlayerProperty(property=array([[1.        ],
       [0.70710678],
       [1.11803399],
       [1.41421356]]), name='distance_covered', framerate=20)
    >>> dm.cumulative_distance_covered()
    PlayerProperty(property=array([[1.        ],
       [0.70710678],
       [1.11803399],
       [1.41421356]]), name='distance_covered', framerate=20)
    """

    def __init__(self):
        super().__init__()
        self._distance_euclidean_ = None

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


            axis: {None, 'x', 'y'}, optional
                Optional argument that restricts distance calculation to either the x-
                or y-dimension of the data. If set to None (default), distances are
                calculated in both dimensions.
        """

        if axis is None:
            if difference == "central":
                differences_xy = np.gradient(xy.xy, axis=0)
            elif difference == "forward":
                differences_xy = np.diff(xy.xy, axis=0, append=0)
            else:
                raise ValueError(
                    f"Expected axis to be one of (None, 'x', 'y'), got {axis}."
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
            raise ValueError(
                f"Expected axis to be one of (None, 'x', 'y'), got {axis}."
            )

        self._distance_euclidean_ = PlayerProperty(
            property=distance_euclidean, name="distance_covered", framerate=xy.framerate
        )

    def distance_covered(self) -> PlayerProperty:
        """Returns the frame-wise distance covered as computed by the fit method.

        Returns
        -------
        distance_euclidean: XY
            An XY object of shape (T, N), where T is the total number of frames and N is
            the number of players. The columns contain the frame-wise euclidean distance
            covered.
        """
        return self._distance_euclidean_

    def cumulative_distance_covered(self) -> PlayerProperty:
        """Returns the cumulative distance covered.

        Returns
        -------
        cumulative_distance_euclidean: XY
            An XY object of shape (T, N), where T is the total number of frames and N is
            the number of players. The columns contain the cumulative euclidean distance
            covered calculated by numpy.nancumsum() over axis=0.
        """
        dist = self._distance_euclidean_.property
        cum_dist = np.nancumsum(dist, axis=0)
        cumulative_distance = PlayerProperty(
            property=cum_dist,
            name="cumulative_distance_covered",
            framerate=self._distance_euclidean_.framerate,
        )
        return cumulative_distance


class VelocityModel(BaseModel):
    """Computations for velocities of all players.

    Upon calling the :func:`~VelocityModel.fit`-method, this model calculates the
    framewise velocity for each player. The calculation can subsequently be queried by
    calling the corresponding method:

        - Frame-wise velocity --> :func:`~VelocityModel.velocity`

    Notes
    -----
    The model calculates the velocity in the unit of the corresponding position data per
    second. This may lead to distortions when used in combination with models that use
    SI units or when the unit is percent.

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.kinematics import VelocityModel

    >>> xy = XY(np.array(((0, 0), (0, 1), (1, 1), (2, 2))))

    >>> vm = VelocityModel()
    >>> vm.fit(xy)

    >>> vm.velocity()
    PlayerProperty(property=array([[20.        ],
       [14.14213562],
       [22.36067977],
       [28.28427125]]), name='velocity', framerate=20)
    """

    def __init__(self):
        super().__init__()
        self._velocity_ = None

    def fit(
        self,
        xy: XY,
        difference: str = "central",
        axis: str = None,
    ):
        """Fits a model to calculate velocities to an xy object.

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


            axis: {None, 'x', 'y'}, optional
                Optional argument that restricts distance calculation to either the x-
                or y-dimension of the data. If set to None (default), distances are
                calculated in both dimensions.
        """

        distance_model = DistanceModel()
        distance_model.fit(xy, difference=difference, axis=axis)
        distance_euclidean = distance_model.distance_covered()

        velocity = distance_euclidean.property * distance_euclidean.framerate

        self._velocity_ = PlayerProperty(
            property=velocity, name="velocity", framerate=xy.framerate
        )

    def velocity(self) -> PlayerProperty:
        """Returns the frame-wise velocity as computed by the fit method.

        Returns
        -------
        velocity: XY
            An XY object of shape (T, N), where T is the total number of frames and N is
            the number of players. The columns contain the frame-wise velocity.
        """
        return self._velocity_


class AccelerationModel(BaseModel):
    """Computations for velocities of all players.

    Upon calling the :func:`~AccelerationModel.fit`-method, this model calculates the
    framewise acceleration for each player. The calculation can subsequently be queried
    by calling the corresponding method:

        - Frame-wise acceleration --> :func:`~AccelerationModel.acceleration`

    Notes
    -----
    The model calculates the velocity in the unit of the corresponding position data per
    second. This may lead to distortions when used in combination with models that use
    SI units or when the unit is percent.

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.kinematics import AccelerationModel

    >>> xy = XY(np.array(((0, 0), (0, 1), (1, 1), (2, 2))))

    >>> am = AccelerationModel()
    >>> am.fit(xy)

    >>> am.acceleration()
    PlayerProperty(property=array([[-117.15728753],
       [  23.60679775],
       [ 141.42135624],
       [ 118.47182945]]), name='acceleration', framerate=20)
    """

    def __init__(self):
        super().__init__()
        self._acceleration_ = None

    def fit(
        self,
        xy: XY,
        difference: str = "central",
        axis: str = None,
    ):
        """Fits a model to calculate acceleration to an xy object.

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


            axis: {None, 'x', 'y'}, optional
                Optional argument that restricts distance calculation to either the x-
                or y-dimension of the data. If set to None (default), distances are
                calculated in both dimensions.
        """

        velocity_model = VelocityModel()
        velocity_model.fit(xy, difference=difference, axis=axis)
        velocity = velocity_model.velocity()

        if difference == "central":
            acceleration = np.gradient(velocity.property, axis=0) * velocity.framerate
        else:
            acceleration = (
                np.diff(velocity.property, axis=0, append=0) * velocity.framerate
            )

        self._acceleration_ = PlayerProperty(
            property=acceleration,
            name="acceleration",
            framerate=xy.framerate,
        )

    def acceleration(self) -> PlayerProperty:
        """Returns the frame-wise acceleration as computed by the fit method.

        Returns
        -------
        acceleration: XY
            An XY object of shape (T, N), where T is the total number of frames and N is
            the number of players. The columns contain the frame-wise acceleration.
        """
        return self._acceleration_

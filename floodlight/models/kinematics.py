import numpy as np

from floodlight import XY
from floodlight.core.property import PlayerProperty
from floodlight.models.base import BaseModel, requires_fit


class DistanceModel(BaseModel):
    """Computations for Euclidean distances of all players.

    Upon calling the :func:`~DistanceModel.fit`-method, this model calculates the
    frame-wise Euclidean distance for each player. The following calculations can
    subsequently be queried by calling the corresponding methods:

        - Frame-wise Distance Covered --> :func:`~DistanceModel.distance_covered`
        - Cumulative Distance Covered --> :func:`~DistanceModel.cumulative_distance_\
covered`

    Notes
    -----
    For input data in metrical units, the output equals the input unit.
    Differences between frames can be calculated with two different methods:

        *Central difference method* (recommended) allows for differenciation without
        temporal shift:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{1}-y_{-1}}{t_{1} - t_{-1}}

        The first and last frame are padded with linear extrapolation.\n
        *Backward difference method* calculates the difference between each consecutive
        frame:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{0}-y_{-1}}{t_{0} - t_{-1}}

        The first frame is padded prepending a '0' at the beginning of the array along
        axis=1.

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
       [1.41421356]]), name='distance_covered'0)
    >>> dm.cumulative_distance_covered()
    PlayerProperty(property=array([[1.        ],
       [0.70710678],
       [1.11803399],
       [1.41421356]]), name='distance_covered')
    """

    def __init__(self):
        super().__init__()
        self._distance_euclidean_ = None

    def fit(self, xy: XY, difference: str = "central", axis: str = None):
        """Fits a model calculating Euclidean distances of each player to an XY object.

        Parameters
        ----------
        xy: XY
            Floodlight XY Data object.
        difference: {'central', 'backward'}, optional
            The method of differentiation. 'central' will differentiate using the
            central difference method, 'backward' will differentiate using the backward
            difference method as described in the Notes.
        axis: {None, 'x', 'y'}, optional
            Optional argument that restricts distance calculation to either the x-
            or y-dimension of the data. If set to None (default), distances are
            calculated in both dimensions.
        """

        if axis is None:
            if difference == "central":
                differences_xy = np.gradient(xy.xy, axis=0)
            elif difference == "backward":
                differences_xy = np.diff(xy.xy, axis=0, prepend=xy.xy[0].reshape(1, -1))
            else:
                raise ValueError(
                    f"Expected axis to be one of (None, 'x', 'y'), got {axis}."
                )
            distance_euclidean = np.hypot(
                differences_xy[:, ::2],
                differences_xy[:, 1::2],
            )
        elif axis == "x":
            if difference == "central":
                distance_euclidean = np.gradient(xy.x, axis=0)
            elif difference == "backward":
                distance_euclidean = np.diff(
                    xy.x, axis=0, prepend=xy.x[0].reshape(1, -1)
                )
        elif axis == "y":
            if difference == "central":
                distance_euclidean = np.gradient(xy.y, axis=0)
            if difference == "backward":
                distance_euclidean = np.diff(
                    xy.y, axis=0, prepend=xy.y[0].reshape(1, -1)
                )
        else:
            raise ValueError(
                f"Expected axis to be one of (None, 'x', 'y'), got {axis}."
            )

        self._distance_euclidean_ = PlayerProperty(
            property=distance_euclidean, name="distance_covered", framerate=xy.framerate
        )

    @requires_fit
    def distance_covered(self) -> PlayerProperty:
        """Returns the frame-wise distance covered as computed by the fit method.

        Returns
        -------
        distance_euclidean: PlayerProperty
            A PlayerProperty object of shape (T, N), where T is the total number of
            frames and N is the number of players. The columns contain the frame-wise
            Euclidean distance covered.
        """
        return self._distance_euclidean_

    @requires_fit
    def cumulative_distance_covered(self) -> PlayerProperty:
        """Returns the cumulative distance covered.

        Returns
        -------
        cumulative_distance_euclidean: PlayerProperty
            A PlayerProperty object of shape (T, N), where T is the total number of
            frames and N is the number of players. The columns contain the cumulative
            Euclidean distance covered calculated by numpy.nancumsum() over axis=0.
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
    frame-wise velocity for each player. The calculation can subsequently be queried by
    calling the corresponding method:

        - Frame-wise velocity --> :func:`~VelocityModel.velocity`

    Notes
    -----
    For input data in metrical units, the output equals the input unit.
    Differences between frames can be calculated with two different methods:

        *Central difference method* (recommended) allows for differenciation without
        temporal shift:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{1}-y_{-1}}{t_{1} - t_{-1}}

        The first and last frame are padded with linear extrapolation.\n
        *Backward difference method* calculates the difference between each consecutive
        frame:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{0}-y_{-1}}{t_{0} - t_{-1}}

        The first frame is padded prepending a '0' at the beginning of the array along
        axis=1.

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.kinematics import VelocityModel

    >>> xy = XY(np.array(((0, 0), (0, 1), (1, 1), (2, 2))), framerate=20)

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
        """Fits a model calculating velocities of each player to an XY object.

        Parameters
        ----------
        xy: XY
            Floodlight XY Data object.
        difference: {'central', 'backward'}, optional
            The method of differentiation. 'central' will differentiate using the
            central difference method, 'backward' will differentiate using the backward
            difference method as described in the Notes.
        axis: {None, 'x', 'y'}, optional
            Optional argument that restricts distance calculation to either the x-
            or y-dimension of the data. If set to None (default), distances are
            calculated in both dimensions.
        """

        distance_model = DistanceModel()
        distance_model.fit(xy, difference=difference, axis=axis)
        distance_euclidean = distance_model.distance_covered()

        velocity = np.multiply(
            distance_euclidean.property, distance_euclidean.framerate
        )

        self._velocity_ = PlayerProperty(
            property=velocity, name="velocity", framerate=xy.framerate
        )

    @requires_fit
    def velocity(self) -> PlayerProperty:
        """Returns the frame-wise velocity as computed by the fit method.

        Returns
        -------
        velocity: PlayerProperty
            A PlayerProperty object of shape (T, N), where T is the total number of
            frames and N is the number of players. The columns contain the frame-wise
            velocity.
        """
        return self._velocity_


class AccelerationModel(BaseModel):
    """Computations for velocities of all players.

    Upon calling the :func:`~AccelerationModel.fit`-method, this model calculates the
    frame-wise acceleration for each player. The calculation can subsequently be queried
    by calling the corresponding method:

        - Frame-wise acceleration --> :func:`~AccelerationModel.acceleration`

    Notes
    -----
    For input data in metrical units, the output equals the input unit.
    Differences between frames can be calculated with two different methods:

        *Central difference method* (recommended) allows for differenciation without
        temporal shift:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{1}-y_{-1}}{t_{1} - t_{-1}}

        The first and last frame are padded with linear extrapolation.\n
        *Backward difference method* calculates the difference between each consecutive
        frame:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{0}-y_{-1}}{t_{0} - t_{-1}}

        The first frame is padded prepending a '0' at the beginning of the array along
        axis=1.

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.kinematics import AccelerationModel

    >>> xy = XY(np.array(((0, 0), (0, 1), (1, 1), (2, 2))), framerate=20)

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
        """Fits a model calculating accelerations of each player to an XY object.

        Parameters
        ----------
        xy: XY
            Floodlight XY Data object.
        difference: {'central', 'backward'}, optional
            The method of differentiation. 'central' will differentiate using the
            central difference method, 'backward' will differentiate using the backward
            difference method as described in the Notes.
        axis: {None, 'x', 'y'}, optional
            Optional argument that restricts distance calculation to either the x-
            or y-dimension of the data. If set to None (default), distances are
            calculated in both dimensions.
        """

        velocity_model = VelocityModel()
        velocity_model.fit(xy, difference=difference, axis=axis)
        velocity = velocity_model.velocity()

        if difference == "central":
            acceleration = np.multiply(
                np.gradient(velocity.property, axis=0), velocity.framerate
            )
        else:
            acceleration = np.multiply(
                np.diff(
                    velocity.property,
                    axis=0,
                    prepend=velocity.property[0].reshape(1, -1),
                ),
                velocity.framerate,
            )

        self._acceleration_ = PlayerProperty(
            property=acceleration,
            name="acceleration",
            framerate=xy.framerate,
        )

    @requires_fit
    def acceleration(self) -> PlayerProperty:
        """Returns the frame-wise acceleration as computed by the fit method.

        Returns
        -------
        acceleration: PlayerProperty
            A PlayerProperty object of shape (T, N), where T is the total number of
            frames and N is the number of players. The columns contain the frame-wise
            acceleration.
        """
        return self._acceleration_

import numpy as np
from scipy.spatial.distance import cdist
import math

from floodlight import XY
from floodlight.core.property import TeamProperty
from floodlight.models.base import BaseModel, requires_fit


MIN_COORDINATE_PAIRS = 2  # Minimum points for valid distance calculation


class DistanceModel(BaseModel):
    """Computations based on distances between players, including distances to nearest mates,
    distances to nearest opponents, and team spread.

    Upon calling the :func:`~DistanceModel.fit`-method, this model performs the following
    calculations based on the given data:

        - Distance to Nearest Mate --> :func:`~DistanceModel.distance_to_nearest_mate`
        - Distance to Nearest Opponents --> :func:`~DistanceModel.distance_to_nearest_opponents`
        - Team Spread --> :func:`~DistanceModel.team_spread`

    Notes
    -----
    The calculations are performed as follows:

    - Distance to Nearest Mate:
        For each player, the Euclidean distance to the nearest other player in the same frame
        is computed.

    - Distance to Nearest Opponents:
        For each player in `xy1`, the Euclidean distance to the nearest opponent in `xy2` is
        computed, and vice versa.

    - Team Spread:
        The Frobenius norm of the lower triangular matrix of player-to-player distances is
        computed to represent the spread of the team.

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.geometry import DistanceModel

    >>> xy1 = XY(np.array(((1, 1, 2, -2), (1.5, np.nan, np.nan, -0))))
    >>> xy2 = XY(np.array(((2, 2, -1, -1), (0.5, np.nan, np.nan, 1))))
    >>> dm = DistanceModel()

    # Calculate distances to nearest mates
    >>> dm.fit(xy1)
    >>> dm.distance_to_nearest_mate()
    TeamProperty(property=array([1.58113883, nan]), name='distance_to_nearest_mate', framerate=None)

    # Calculate distances to nearest opponents
    >>> dm.fit(xy1, xy2)
    >>> dm.distance_to_nearest_opponents()
    (TeamProperty(property=array([1.0, 2.0]), name='distance_to_nearest_opponents_1', framerate=None),
     TeamProperty(property=array([1.5, 2.5]), name='distance_to_nearest_opponents_2', framerate=None))

    # Calculate team spread
    >>> dm.fit(xy1)
    >>> dm.team_spread()
    TeamProperty(property=array([3.0, nan]), name='team_spread', framerate=None)
    """

    def __init__(self):
        super().__init__()
        self._dtnm = None
        self._dtno1 = None
        self._dtno2 = None
        self._spread = None

    def fit(self, xy1: XY, xy2: XY = None) -> None:
        """
        Fit the model to the given data and calculate distances and spread.

        Parameters
        ----------
        xy1 : XY
            Player spatiotemporal data for which the calculations are based.
        xy2 : XY, optional
            Second set of player spatiotemporal data used for calculating distances to
            opponents. If not provided, distances to nearest mates and team spread are calculated.

        Returns
        -------
        None
        """
        if xy2 is None:
            self._dtnm = self._calc_dtnm(xy1)
            self._spread = self._calc_team_spread(xy1)
        else:
            self._dtno1, self._dtno2 = self._calc_dtno(xy1, xy2)

    # ensure there are enough needed valid coordinaates pairs
    def count_valid_pairs(points):
        # Split list into pairs and filter for valid pairs
        pairs = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
        valid_pairs = [
            pair
            for pair in pairs
            if not (isinstance(pair[0], float) and math.isnan(pair[0]))
            and not (isinstance(pair[1], float) and math.isnan(pair[1]))
        ]

        return len(valid_pairs)

    def _calc_dtnm(self, xyobj: XY) -> TeamProperty:
        """Calculate distance to nearest mate."""
        dtnm = np.full((len(xyobj), 1), np.nan)
        for t in range(len(xyobj)):
            points = xyobj.frame(t)[~np.isnan(xyobj.frame(t))]

            if DistanceModel.count_valid_pairs(points) < MIN_COORDINATE_PAIRS:
                continue
            pairwise_distances = cdist(points.reshape(-1, 2), points.reshape(-1, 2))
            pairwise_distances[pairwise_distances == 0] = np.nan
            if np.isnan(pairwise_distances).all():
                dtnm[t] = np.nan  # or some other value if you prefer
            else:
                dtnm[t] = np.nanmin(pairwise_distances, axis=1).mean()

        return TeamProperty(
            property=dtnm, name="distance_to_nearest_mate", framerate=xyobj.framerate
        )

    def _calc_dtno(self, xyobj1: XY, xyobj2: XY) -> tuple[TeamProperty, TeamProperty]:
        """Calculate distances of xyobj1 to nearest opponents xyobj2."""
        dtnm1 = np.full((len(xyobj1), 1), np.nan)
        dtnm2 = np.full((len(xyobj1), 1), np.nan)
        for t in range(len(xyobj1)):
            points1 = xyobj1.frame(t)[~np.isnan(xyobj1.frame(t))]
            points2 = xyobj2.frame(t)[~np.isnan(xyobj2.frame(t))]
            if len(points1) < 2 or len(points2) < 2:
                continue
            pairwise_distances = cdist(points1.reshape(-1, 2), points2.reshape(-1, 2))
            dtnm1[t] = np.nanmin(pairwise_distances, axis=1).mean()
            dtnm2[t] = np.nanmin(pairwise_distances, axis=0).mean()
        return (
            TeamProperty(
                property=dtnm1,
                name="distance_to_nearest_opponents_1",
                framerate=xyobj1.framerate,
            ),
            TeamProperty(
                property=dtnm2,
                name="distance_to_nearest_opponents_2",
                framerate=xyobj2.framerate,
            ),
        )

    def _calc_team_spread(self, xyobj: XY) -> TeamProperty:
        """Calculate team spread."""
        spread = np.full((len(xyobj), 1), np.nan)
        for t in range(len(xyobj)):
            points = xyobj.frame(t)[~np.isnan(xyobj.frame(t))]
            if len(points) < 2:
                continue
            pairwise_distances = cdist(points.reshape(-1, 2), points.reshape(-1, 2))
            spread[t] = np.linalg.norm(np.tril(pairwise_distances), ord="fro")
        return TeamProperty(
            property=spread, name="team_spread", framerate=xyobj.framerate
        )

    @requires_fit
    def distance_to_nearest_mate(self) -> TeamProperty:
        """Returns the distance to the nearest mate as computed by the fit method."""
        return self._dtnm

    @requires_fit
    def distance_to_nearest_opponents(self) -> tuple[TeamProperty, TeamProperty]:
        """Returns the distance to nearest opponents for both xy1 and xy2 as computed by the fit method."""
        return self._dtno1, self._dtno2

    @requires_fit
    def team_spread(self) -> TeamProperty:
        """Returns the team spread as computed by the fit method."""
        return self._spread

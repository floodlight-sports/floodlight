import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, QhullError
import numpy as np
from floodlight import XY
from floodlight.core.property import TeamProperty
from floodlight.models.base import BaseModel, requires_fit


class SpatialMetricsModel(BaseModel):
    """Computations for spatial metrics such as convex hull areas and
    effective playing space.

    Upon calling the :func:`~SpatialMetricsModel.fit` method, this model
    performs the following calculations based on the given data:

        - Convex Hull Area of each team
        - Effective Playing Space (EPS)

    Notes
    -----
    The calculations are performed as follows:

    - Convex Hull Area:
        For each frame, the convex hull area of all available points is
        computed to represent the area covered by a team.

    - Effective Playing Space (EPS):
        For each frame, the convex hull area that encompasses all players
        from both teams is computed to represent the effective playing space.

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.geometry import SpatialMetricsModel
    >>> xy1 = XY(np.array(((1, 1, 2, -2), (1.5, np.nan, np.nan, -0))))
    >>> xy2 = XY(np.array(((2, 2, -1, -1), (0.5, np.nan, np.nan, 1))))
    >>> smm = SpatialMetricsModel()

    # Calculate convex hull areas for a team
    >>> smm.fit(xy1)
    >>> smm.convex_hull_area()
    TeamProperty(property=array([area_value1, nan]), name='convex_hull_area',
                 framerate=None)

    # Calculate effective playing space
    >>> smm.fit(xy1, xy2)
    >>> smm.effective_playing_space()
    TeamProperty(property=array([eps_value1, eps_value2]),
                 name='effective_playing_space', framerate=None)
    """

    def __init__(self):
        super().__init__()
        self._chulls = None
        self._eps = None

    def fit(self, *xy_objects: XY):
        """Fit the model to the given data and calculate spatial metrics.

        Parameters
        ----------
        *xy_objects : XY
            One or more XY objects. The first XY object is used for
            calculating convex hull areas. If a second XY object is provided,
            it is used for calculating effective playing space.
        """
        if len(xy_objects) == 1:
            self._chulls = self._calc_chulls(xy_objects[0])
        elif len(xy_objects) == 2:
            self._eps = self._calc_eps(xy_objects[0], xy_objects[1])
        else:
            raise ValueError("Expected one or two XY objects.")

    def _calc_chulls(self, xyobj: XY) -> TeamProperty:
        """Calculate convex hull area for each frame of all available points."""
        chulls = np.full((len(xyobj), 1), np.nan)
        for t in range(len(xyobj)):
            points = xyobj.frame(t)[~np.isnan(xyobj.frame(t))]
            if len(points) < 6:
                continue
            chulls[t] = ConvexHull(points.reshape(-1, 2)).volume
        return TeamProperty(
            property=chulls, name='convex_hull_area', framerate=xyobj.framerate
        )

    def _calc_eps(self, xyobj1: XY, xyobj2: XY) -> TeamProperty:
        """Calculate effective playing space for each frame of both teams."""
        eps = np.full((len(xyobj1), 1), np.nan)
        for t in range(len(xyobj1)):
            points1 = xyobj1.frame(t)[~np.isnan(xyobj1.frame(t))]
            points2 = xyobj2.frame(t)[~np.isnan(xyobj2.frame(t))]
            if len(points1) + len(points2) < 6:
                continue
            eps[t] = ConvexHull(
                np.concatenate((points1, points2)).reshape(-1, 2)
            ).volume
        return TeamProperty(
            property=eps, name='effective_playing_space',
            framerate=xyobj1.framerate
        )

    @requires_fit
    def convex_hull_area(self) -> TeamProperty:
        """Returns the convex hull area as computed by the fit method."""
        return self._chulls

    @requires_fit
    def effective_playing_space(self) -> TeamProperty:
        """Returns the effective playing space as computed by the fit method."""
        return self._eps

    def plot_convex_hull(self, xyobj: XY, frame: int, ax: plt.Axes):
        """Plot the convex hull on an existing football pitch.

        Parameters
        ----------
        xyobj : XY
            Player spatiotemporal data for which the convex hull is
            calculated.
        frame : int
            The frame index to plot.
        ax : plt.Axes
            The matplotlib axes to plot on. Must be initialized with a
            football pitch.
        """
        if ax is None:
            raise ValueError("An existing matplotlib Axes object must be provided.")

        points = xyobj.frame(frame)
        if points is None or np.all(np.isnan(points)):
            print("No data available for the specified frame.")
            return

        if len(points) < 6:
            print("Not enough points to form a convex hull.")
            return

        points = points.reshape(-1, 2)

        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])
            ax.plot(hull_points[:, 0], hull_points[:, 1], 'r-', lw=2, label='Convex Hull')
            ax.fill(hull_points[:, 0], hull_points[:, 1], 'r', alpha=0.3)
        except (QhullError, IndexError):
            print("ConvexHull computation failed.")

        ax.set_title(f'Convex Hull for Frame {frame}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()

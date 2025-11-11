import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, QhullError
import numpy as np
import math
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

    def _calc_chulls(self, xyobj: XY) -> TeamProperty:
        """Calculate convex hull area for each frame of all available points."""
        chulls = np.full((len(xyobj), 1), np.nan)
        for t in range(len(xyobj)):
            points = xyobj.frame(t)[~np.isnan(xyobj.frame(t))]
            if SpatialMetricsModel.count_valid_pairs(points) < 3:
                continue
            reshaped_points = points.reshape(-1, 2)
            # make sure the area of convex hull is not 0
            if len(np.unique(reshaped_points, axis=0)) > 2:
                chulls[t] = ConvexHull(reshaped_points).volume
        return TeamProperty(
            property=chulls, name="convex_hull_area", framerate=xyobj.framerate
        )

    def _calc_eps(self, xyobj1: XY, xyobj2: XY) -> TeamProperty:
        """Calculate effective playing space for each frame of both teams."""
        eps = np.full((len(xyobj1), 1), np.nan)
        for t in range(len(xyobj1)):
            points1 = xyobj1.frame(t)[~np.isnan(xyobj1.frame(t))]
            points2 = xyobj2.frame(t)[~np.isnan(xyobj2.frame(t))]

            # ensure there are enough needed valid coordinaates pairs
            if (
                SpatialMetricsModel.count_valid_pairs(points1) < 1
                or SpatialMetricsModel.count_valid_pairs(points2) < 1
            ):
                # Log a warning if entire second team is NaN (optional)
                if len(points2) == 0:
                    print(f"Warning: Frame {t} - Second team data is all NaN.")
                continue

            # Proceed with convex hull calculation if valid points are sufficient
            if len(points1) + len(points2) >= 6:
                eps[t] = ConvexHull(
                    np.concatenate((points1, points2)).reshape(-1, 2)
                ).volume

        return TeamProperty(
            property=eps, name="effective_playing_space", framerate=xyobj1.framerate
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
        """
        Plot the convex hull on an existing football pitch.

        Parameters
        ----------
        xyobj : XY
            Player spatiotemporal data for which the convex hull is calculated.
        frame : int
            The frame index to plot.
        ax : matplotlib.axes._axes.Axes
            The matplotlib axes to plot on. Must be initialized with a football pitch.

        Examples
        --------
        Plotting a convex hull for player positions on a football pitch:

        import numpy as np
        import matplotlib.pyplot as plt
        from floodlight.core.xy import XY
        from floodlight.models.spatialMetricsModel import SpatialMetricsModel
        from floodlight.vis.pitches import plot_football_pitch
        from floodlight.vis.positions import plot_positions

        # Initialize some player position data
        data = np.array([
            [
                10, 42, 59, 43, 61,
                21, 63, 57, 36, 57,
                18, 27, 15, 24, 18,
                11, 51, 49, 39, 1,
                57, 58
            ],
            [
                22, 29, 32, 107, 54,
                25, 50, 8, 40, 69,
                21, 25, 98, 64, 101,
                102, 88, 36, 93, 55,
                57, 43
            ]
        ])
        xy = XY(data)

        # Initialize the plot
        fig, ax = plt.subplots()

        # Plot the football pitch
        plot_football_pitch(
            xlim=(0, 108), ylim=(0, 68), length=108, width=68, unit='m',
            color_scheme='standard', show_axis_ticks=False, ax=ax
        )

        # Plot positions for the first frame
        plot_positions(xy=xy, frame=0, ball=False, ax=ax)

        # Instantiate the SpatialMetricsModel
        smm = SpatialMetricsModel()

        # Fit the model with position data
        smm.fit(xy)

        # Plot convex hull for the first frame using the existing ax
        smm.plot_convex_hull(xy, ax=ax, frame=0)

        plt.show()  # Display the plot

        .. image:: ../../_img/sample_plot_convex_hull_on_pitch.png
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
            ax.plot(
                hull_points[:, 0], hull_points[:, 1], "r-", lw=2, label="Convex Hull"
            )
            ax.fill(hull_points[:, 0], hull_points[:, 1], "r", alpha=0.3)
        except (QhullError, IndexError):
            print("ConvexHull computation failed.")

        ax.set_title(f"Convex Hull for Frame {frame}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend()

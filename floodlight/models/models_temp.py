import warnings
import numpy as np
from scipy.spatial.distance import cdist
from floodlight import XY
from floodlight.core.property import TeamProperty, PlayerProperty
from floodlight.models.base import BaseModel, requires_fit
from typing import Tuple

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

    def fit(self, xy1: XY, xy2: XY = None):
        """Fit the model to the given data and calculate distances and spread.

        Parameters
        ----------
        xy1: XY
            Player spatiotemporal data for which the calculations are based.
        xy2: XY, optional
            Optional second set of player spatiotemporal data used for calculating distances
            to opponents.
        """
        if xy2 is None:
            # Calculate distance to nearest mate and team spread
            self._dtnm = self._calc_dtnm(xy1)
            self._spread = self._calc_team_spread(xy1)
        else:
            # Calculate distance to nearest opponents
            self._dtno1, self._dtno2 = self._calc_dtno(xy1, xy2)

    def _calc_dtnm(self, xyobj: XY) -> TeamProperty:
        """Calculate distance to nearest mate."""
        dtnm = np.full((len(xyobj), 1), np.nan)
        for t in range(len(xyobj)):
            points = xyobj.frame(t)[~np.isnan(xyobj.frame(t))]
            if len(points) < 4: # changed from 2 to 4 beacause we need 2 coordinates ( so 2 *2 (x and y))
                continue
            pairwise_distances = cdist(points.reshape(-1, 2), points.reshape(-1, 2))
            pairwise_distances[pairwise_distances == 0] = np.nan
            dtnm[t] = np.nanmin(pairwise_distances, axis=1).mean()
        return TeamProperty(property=dtnm, name='distance_to_nearest_mate', framerate=xyobj.framerate)


    def _calc_dtno(self, xyobj1: XY, xyobj2: XY) -> Tuple[TeamProperty, TeamProperty]:
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
        return (TeamProperty(property=dtnm1, name='distance_to_nearest_opponents_1', framerate=xyobj1.framerate),
                TeamProperty(property=dtnm2, name='distance_to_nearest_opponents_2', framerate=xyobj2.framerate))

    def _calc_team_spread(self, xyobj: XY) -> TeamProperty:
        """Calculate team spread."""
        spread = np.full((len(xyobj), 1), np.nan)
        for t in range(len(xyobj)):
            points = xyobj.frame(t)[~np.isnan(xyobj.frame(t))]
            if len(points) < 2:
                continue
            pairwise_distances = cdist(points.reshape(-1, 2), points.reshape(-1, 2))
            spread[t] = np.linalg.norm(np.tril(pairwise_distances), ord="fro")
        return TeamProperty(property=spread, name='team_spread', framerate=xyobj.framerate)

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


####
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial import ConvexHull, convex_hull_plot_2d

class SpatialMetricsModel(BaseModel):
    """Computations for spatial metrics such as convex hull areas and effective playing space.

    Upon calling the :func:`~SpatialMetricsModel.fit` method, this model performs the following
    calculations based on the given data:

        - Convex Hull Area of each team --> :func:`~SpatialMetricsModel.convex_hull_area`
        - Effective Playing Space (EPS) --> :func:`~SpatialMetricsModel.effective_playing_space`

    Notes
    -----
    The calculations are performed as follows:

    - Convex Hull Area:
        For each frame, the convex hull area of all available points is computed to represent
        the area covered by a team.

    - Effective Playing Space (EPS):
        For each frame, the convex hull area that encompasses all players from both teams
        is computed to represent the effective playing space.

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
    TeamProperty(property=array([area_value1, nan]), name='convex_hull_area', framerate=None)

    # Calculate effective playing space
    >>> smm.fit(xy1, xy2)
    >>> smm.effective_playing_space()
    TeamProperty(property=array([eps_value1, eps_value2]), name='effective_playing_space', framerate=None)
    """

    def __init__(self):
        super().__init__()
        self._chulls = None
        self._eps = None

    def fit(self, *xy_objects: XY):
        """Fit the model to the given data and calculate spatial metrics.

        Parameters
        ----------
        *xy_objects: XY
            One or more XY objects. The first XY object is used for calculating convex hull areas.
            If a second XY object is provided, it is used for calculating effective playing space.
        """
        if len(xy_objects) == 1:
            # Calculate convex hull areas
            self._chulls = self._calc_chulls(xy_objects[0])
        elif len(xy_objects) == 2:
            # Calculate effective playing space
            self._eps = self._calc_eps(xy_objects[0], xy_objects[1])
        else:
            raise ValueError("Expected one or two XY objects.")

    def _calc_chulls(self, xyobj: XY) -> TeamProperty:
        """Calculate convex hull area for each frame of all available points."""
        chulls = np.full((len(xyobj), 1), np.nan)
        for t in range(len(xyobj)):
            points = xyobj.frame(t)[~np.isnan(xyobj.frame(t))]
            if len(points) < 6:  # At least three points are needed for a convex hull (6 values because x and y coordinates)
                continue
            chulls[t] = ConvexHull(points.reshape(-1, 2)).volume
        return TeamProperty(property=chulls, name='convex_hull_area', framerate=xyobj.framerate)

    def _calc_eps(self, xyobj1: XY, xyobj2: XY) -> TeamProperty:
        """Calculate effective playing space for each frame of all available points of both teams."""
        eps = np.full((len(xyobj1), 1), np.nan)
        for t in range(len(xyobj1)):
            points1 = xyobj1.frame(t)[~np.isnan(xyobj1.frame(t))]
            points2 = xyobj2.frame(t)[~np.isnan(xyobj2.frame(t))]
            if len(points1) + len(points2) < 6:  # At least three points in total are needed for a convex hull
                continue
            eps[t] = ConvexHull(np.concatenate((points1, points2)).reshape(-1, 2)).volume
        return TeamProperty(property=eps, name='effective_playing_space', framerate=xyobj1.framerate)

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
        xyobj: XY
            Player spatiotemporal data for which the convex hull is calculated.
        frame: int
            The frame index to plot.
        ax: plt.Axes
            The matplotlib axes to plot on. Must be initialized with a football pitch.
        """
        # Ensure the provided axes are not None
        if ax is None:
            raise ValueError("An existing matplotlib Axes object must be provided.")

        # Get the data for the specified frame
        points = xyobj.frame(frame)
        if points is None or np.all(np.isnan(points)):
            print("No data available for the specified frame.")
            return

        if len(points) < 6:
            print("Not enough points to form a convex hull.")
            return

        points = points.reshape(-1, 2)  # Reshape to (n_points, 2)

        try:
            hull = ConvexHull(points)
            # Plotting the convex hull
            hull_points = points[hull.vertices]  
            # Append the first point to the end to close the polygon
            hull_points = np.vstack([hull_points, hull_points[0]])
            # Plot the boundary of the convex hull
            ax.plot(hull_points[:, 0], hull_points[:, 1], 'r-', lw=2, label='Convex Hull')
            # Fill the area inside the convex hull
            ax.fill(hull_points[:, 0], hull_points[:, 1], 'r', alpha=0.3)
            
        except (QhullError, IndexError):
            # Handle cases where ConvexHull fails due to insufficient points
            print("ConvexHull computation failed.")

        ax.set_title(f'Convex Hull for Frame {frame}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()


##### FOR TESTING

from floodlight.vis.pitches import plot_football_pitch
from floodlight.vis.positions import plot_positions

def generate_random_xy_data(num_frames, num_points, xlim=(0, 108), ylim=(0, 68)):
    """Generate random XY data for testing."""
    data = np.empty((num_frames, num_points * 2))  
    data[:] = np.nan  # Fill with NaNs

    for frame in range(num_frames):
        x_coords = np.random.uniform(xlim[0], xlim[1], num_points)
        y_coords = np.random.uniform(ylim[0], ylim[1], num_points)
        
        data[frame, :num_points] = x_coords  # X coordinates for the current frame
        data[frame, num_points:] = y_coords  # Y coordinates for the current frame

    return XY(data)

def test_distance_model():
    """Test the DistanceModel class with realistic scenarios including 11 players per team."""
    
    # Generate sample data for two teams with 11 players each and 3 frames
    xy1 = generate_random_xy_data(num_frames=3, num_points=11)
    xy2 = generate_random_xy_data(num_frames=3, num_points=11, xlim=(50, 158), ylim=(50, 158))
    
    # Initialize DistanceModel
    dm = DistanceModel()

    # Test distance to nearest mate
    print("Testing distance to nearest mate:")
    dm.fit(xy1)
    dtnm = dm.distance_to_nearest_mate()
    print("Distance to Nearest Mate:")
    print(dtnm.property)

    # Test team spread
    print("\nTesting team spread:")
    spread = dm.team_spread()
    print("Team Spread:")
    print(spread.property)

    # Test distance to nearest opponents
    print("\nTesting distance to nearest opponents:")
    dm.fit(xy1, xy2)
    dtno1, dtno2 = dm.distance_to_nearest_opponents()
    print("Distance to Nearest Opponents (xy1 to xy2):")
    print(dtno1.property)
    print("Distance to Nearest Opponents (xy2 to xy1):")
    print(dtno2.property)

if __name__ == "__main__":
    # Generate sample data for team 1
    xy1 = generate_random_xy_data(num_frames=3, num_points=11)
    print(xy1.y) 
    print(xy1.x)  

    # Plot football pitch and player positions
    fig, ax = plt.subplots()
    plot_football_pitch(
        xlim=(0, 108), ylim=(0, 68), length=108, width=68, unit='m',
        color_scheme='standard', show_axis_ticks=False, ax=ax
    )
    plot_positions(xy=xy1, frame=0, ball=False, ax=ax)

    # Instantiate and use SpatialMetricsModel
    smm = SpatialMetricsModel()
    smm.fit(xy1)
    smm.plot_convex_hull(xy1, ax=ax, frame=0)

    plt.show()  

    # Run the distance model test function
    test_distance_model()

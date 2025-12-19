import warnings

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, QhullError
import matplotlib.pyplot as plt

from floodlight import XY
from floodlight.core.property import TeamProperty, PlayerProperty
from floodlight.models.base import BaseModel, requires_fit


class CentroidModel(BaseModel):
    """Computations based on the geometric center of all players, commonly referred to
    as a team's *centroid*.

    Upon calling the :func:`~CentroidModel.fit`-method, this model calculates a team's
    centroid. The following calculations can subsequently be queried by calling the
    corresponding methods:

        - Centroid [1]_ --> :func:`~CentroidModel.centroid`
        - Centroid Distance --> :func:`~CentroidModel.centroid_distance`
        - Stretch Index [2]_ --> :func:`~CentroidModel.stretch_index`

    Notes
    -----
    Team centroids are computed as the arithmetic mean of all player positions (based on
    *numpy*'s nanmean function). For a fixed point in time and :math:`N` players with
    corresponding positions :math:`x_1, \\dots, x_N \\in \\mathbb{R}^2`, the centroid is
    calculated as

        .. math::
            C = \\frac{1}{N} \\sum_i^N x_i.

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.geometry import CentroidModel

    >>> xy = XY(np.array(((1, 1, 2, -2), (1.5, np.nan, np.nan, -0))))
    >>> cm = CentroidModel()
    >>> cm.fit(xy)
    >>> cm.centroid()
    XY(xy=array([[ 1.5, -0.5],
       [ 1.5,  0. ]]), framerate=None, direction=None)
    >>> cm.stretch_index(xy)
    TeamProperty(property=array([1.5811388, nan]), name='stretch_index', framerate=None)
    >>> cm.stretch_index(xy, axis='x')
    TeamProperty(property=array([0.5, 0.]), name='stretch_index', framerate=None)

    References
    ----------
        .. [1] `Sampaio, J., & Maçãs, V. (2012). Measuring tactical behaviour in
            football. International Journal of Sports Medicine, 33(05), 395-401.
            <https://www.thieme-connect.de/products/ejournals/abstract/10.1055/s-0031-13
            01320>`_
        .. [2] `Bourbousson, J., Sève, C., & McGarry, T. (2010). Space–time coordination
            dynamics in basketball: Part 2. The interaction between the two teams.
            Journal of Sports Sciences, 28(3), 349-358.
            <https://www.tandfonline.com/doi/full/10.1080/02640410903503640>`_
    """

    def __init__(self):
        super().__init__()
        # model parameter
        self._centroid_ = None

    def fit(self, xy: XY, exclude_xIDs: list = None):
        """Fit the model to the given data and calculate team centroids.

        Parameters
        ----------
        xy: XY
            Player spatiotemporal data for which the centroid is calculated.
        exclude_xIDs: list, optional
            A list of xIDs to be excluded from computation. This can be useful if one
            would like, for example, to exclude goalkeepers from analysis.
        """
        if not exclude_xIDs:
            exclude_xIDs = []
        # boolean for column inclusion, initialize to True for all columns
        include = np.full((xy.N * 2), True)

        # exclude columns according to exclude_xIDs
        for xID in exclude_xIDs:
            if xID not in range(0, xy.N):
                raise ValueError(
                    f"Expected entries of exclude_xIDs to be in range 0 to {xy.N}, "
                    f"got {xID}."
                )
            exclude_start = xID * 2
            exclude_end = exclude_start + 2
            include[exclude_start:exclude_end] = False

        with warnings.catch_warnings():
            # supress warnings caused by empty slices
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # calculate centroid
            centroids = np.nanmean(xy.xy[:, include].reshape((len(xy), -1, 2)), axis=1)

        # wrap as XY object
        self._centroid_ = XY(
            xy=centroids, framerate=xy.framerate, direction=xy.direction
        )

    @requires_fit
    def centroid(self) -> XY:
        """Returns the team centroid positions as computed by the fit method.

        Returns
        -------
        centroid: XY
            An XY object of shape (T, 2), where T is the total number of frames. The two
            columns contain the centroids' x- and y-coordinates, respectively.
        """
        return self._centroid_

    @requires_fit
    def centroid_distance(self, xy: XY, axis: str = None) -> PlayerProperty:
        """Calculates the Euclidean distance of each player to the fitted centroids.

        Parameters
        ----------
        xy: XY
            Player spatiotemporal data for which the distances to the fitted centroids
            are calculated.
        axis: {None, 'x', 'y'}, optional
            Optional argument that restricts distance calculation to either the x- or
            y-dimension of the data. If set to None (default), distances are calculated
            in both dimensions.

        Returns
        -------
        centroid_distance: PlayerProperty
            A PlayerProperty object of shape (T, N), where T is the total number of
            frames. Each column contains the distances to the team centroid of the
            player with corresponding xID.
        """
        # check matching lengths
        T = len(self._centroid_)
        if len(xy) != T:
            raise ValueError(
                f"Length of xy ({len(xy)}) does not match length of fitted centroids "
                f"({T})."
            )

        # calculate distances on specified axis
        distances = np.full((T, xy.N), np.nan)
        if axis is None:
            for t in range(T):
                distances[t] = cdist(
                    self._centroid_[t].reshape(-1, 2), xy[t].reshape(-1, 2)
                )
        elif axis == "x":
            for t in range(T):
                distances[t] = cdist(
                    self._centroid_.x[t].reshape(-1, 1), xy.x[t].reshape(-1, 1)
                )
        elif axis == "y":
            for t in range(T):
                distances[t] = cdist(
                    self._centroid_.y[t].reshape(-1, 1), xy.y[t].reshape(-1, 1)
                )
        else:
            raise ValueError(
                f"Expected axis to be one of (None, 'x', 'y'), got {axis}."
            )

        # wrap as PlayerProperty
        centroid_distance = PlayerProperty(
            property=distances,
            name="centroid_distance",
            framerate=xy.framerate,
        )

        return centroid_distance

    @requires_fit
    def stretch_index(self, xy: XY, axis: str = None) -> TeamProperty:
        """Calculates the *Stretch Index*, i.e., the mean distance of all players to the
        team centroid.

        Parameters
        ----------
        xy: XY
            Player spatiotemporal data for which the stretch index is calculated.
        axis: {None, 'x', 'y'}, optional
            Optional argument that restricts stretch index calculation to either the x-
            or y-dimension of the data. If set to None (default), the stretch index is
            calculated in both dimensions.

        Returns
        -------
        stretch_index: TeamProperty
            A TeamProperty object of shape (T,), where T is the total number of
            frames. Each entry contains the stretch index of that particular frame.
        """
        # get player distances from centroid
        centroid_distances = self.centroid_distance(xy=xy, axis=axis)

        with warnings.catch_warnings():
            # supress warnings caused by empty slices
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # calculate stretch index
            stretch_index = np.nanmean(centroid_distances.property, axis=1)

        # wrap as TeamProperty object
        stretch_index = TeamProperty(
            property=stretch_index, name="stretch_index", framerate=xy.framerate
        )

        return stretch_index


class NearestMateModel(BaseModel):
    """Computations for within-team distance metrics.

    Upon calling the :func:`~NearestMateModel.fit`-method, this model
    calculates pairwise distances between all players for each frame. The
    following calculations can subsequently be queried by calling the
    corresponding methods:

        - Distance to Nearest Mate
          --> :func:`~NearestMateModel.distance_to_nearest_mate`
        - Team Spread [5]_ --> :func:`~NearestMateModel.team_spread`

    Notes
    -----
    The calculations are performed as follows:

    - *Distance to Nearest Mate (DTNM)*:
        For each player in each frame, the Euclidean distance to their nearest
        teammate is computed.

    - *Team Spread*:
        The Frobenius norm of the lower triangular matrix of all pairwise player
        distances, representing the overall dispersion of the team.

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.geometry import NearestMateModel

    >>> xy = XY(np.array(((1, 1, 2, -2), (1.5, np.nan, np.nan, -0))))
    >>> nmm = NearestMateModel()
    >>> nmm.fit(xy)

    >>> dtnm = nmm.distance_to_nearest_mate()
    >>> dtnm
    PlayerProperty(property=array([[3.16227766, 3.16227766],
           [nan, nan]]), name='distance_to_nearest_mate', framerate=None)
    >>> dtnm.property.mean(axis=1)  # Mean distance per frame
    array([3.16227766, nan])

    >>> nmm.team_spread()
    TeamProperty(property=array([3.16227766, nan]), name='team_spread', framerate=None)

    References
    ----------
        .. [5] `Bartlett, R., Button, C., Robins, M., Dutt-Mazumder, A., & Kennedy,
            G. (2012). Analysing team coordination patterns from player movement
            trajectories in soccer: Methodological considerations. International
            Journal of Performance Analysis in Sport, 12(2), 398-424.
            <https://www.tandfonline.com/doi/abs/10.1080/24748668.2012.11868607>`_


    """

    def __init__(self):
        super().__init__()
        self._pairwise_distances_ = None
        self._framerate = None

    def fit(self, xy: XY):
        """Fit the model to the given data and calculate pairwise distances.

        Parameters
        ----------
        xy: XY
            Player spatiotemporal data for which the pairwise distances are
            calculated.
        """
        self._framerate = xy.framerate
        # Initialize distance array with fixed shape (T, N, N)
        self._pairwise_distances_ = np.full((len(xy), xy.N, xy.N), np.nan)

        for t in range(len(xy)):
            # Get coordinates for all players
            coords = xy.frame(t).reshape(-1, 2)

            # Calculate pairwise distances (NaN propagates naturally)
            self._pairwise_distances_[t] = cdist(coords, coords)
            # Set diagonal to NaN (self-distances are meaningless)
            np.fill_diagonal(self._pairwise_distances_[t], np.nan)

    @requires_fit
    def distance_to_nearest_mate(self) -> PlayerProperty:
        """Calculates the distance to the nearest teammate for each player.

        Returns
        -------
        distance_to_nearest_mate: PlayerProperty
            A PlayerProperty object of shape (T, N), where T is the total number
            of frames and N is the number of players. Each entry contains the
            distance to the nearest teammate for that player in that frame.
        """
        T, N, _ = self._pairwise_distances_.shape
        dtnm = np.full((T, N), np.nan)

        with warnings.catch_warnings():
            # suppress warnings caused by all-NaN slices
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for t in range(T):
                # Calculate minimum distance per player (NaN values ignored)
                dtnm[t] = np.nanmin(self._pairwise_distances_[t], axis=1)

        return PlayerProperty(
            property=dtnm,
            name="distance_to_nearest_mate",
            framerate=self._framerate,
        )

    @requires_fit
    def team_spread(self) -> TeamProperty:
        """Calculates the team spread (Frobenius norm of distance matrix).

        Returns
        -------
        spread: TeamProperty
            A TeamProperty object of shape (T,), where T is the total number of
            frames. Each entry contains the team spread (Frobenius norm of the
            pairwise distance matrix) for that frame.
        """
        T = len(self._pairwise_distances_)
        spread = np.full(T, np.nan)

        for t in range(T):
            distances = self._pairwise_distances_[t]
            # Check if all distances are NaN (no valid players)
            if np.isnan(distances).all():
                continue
            # Calculate Frobenius norm of lower triangular matrix
            # Replace NaN with 0 (missing distances don't contribute)
            spread[t] = np.linalg.norm(np.nan_to_num(np.tril(distances)), ord="fro")

        return TeamProperty(
            property=spread, name="team_spread", framerate=self._framerate
        )


class NearestOpponentModel(BaseModel):
    """Computations for between-team distance metrics.

    Upon calling the :func:`~NearestOpponentModel.fit`-method, this model
    calculates pairwise distances between players of opposing teams. The
    following calculations can subsequently be queried:

        - Distance to Nearest Opponent [6]_
          --> :func:`~NearestOpponentModel.distance_to_nearest_opponent`

    Notes
    -----
    For each player in each frame, the Euclidean distance to their nearest
    opponent is computed.

    References
    ----------

        .. [6] `Gonçalves, B., Marcelino, R., Torres-Ronda, L., Torrents, C., & Sampaio,
            J. (2016). Effects of emphasising opposition and cooperation on collective
            movement behaviour during football small-sided games. Journal of sports
            sciences, 34(14), 1346-1354.
            <https://www.tandfonline.com/doi/full/10.1080/02640414.2016.1143111>`_


    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.geometry import NearestOpponentModel

    >>> xy1 = XY(np.array(((1, 1, 2, -2), (1.5, np.nan, np.nan, -0))))
    >>> xy2 = XY(np.array(((2, 2, -1, -1), (0.5, np.nan, np.nan, 1))))
    >>> nom = NearestOpponentModel()
    >>> nom.fit(xy1, xy2)

    >>> dtno1, dtno2 = nom.distance_to_nearest_opponent()
    >>> dtno1
    PlayerProperty(property=array([[1.41421356, 3.16227766],
           [nan, nan]]), name='distance_to_nearest_opponent', framerate=None)
    >>> dtno1.property.mean(axis=1)  # Mean distance per frame
    array([2.28824561, nan])
    """

    def __init__(self):
        super().__init__()
        self._pairwise_distances_ = None
        self._framerate = None

    def fit(self, xy1: XY, xy2: XY):
        """Fit the model to the given data and calculate pairwise distances.

        Parameters
        ----------
        xy1: XY
            Player spatiotemporal data for the first team.
        xy2: XY
            Player spatiotemporal data for the second team.
        """
        self._framerate = xy1.framerate
        # Initialize distance array with fixed shape (T, N1, N2)
        self._pairwise_distances_ = np.full((len(xy1), xy1.N, xy2.N), np.nan)

        for t in range(len(xy1)):
            # Get coordinates for all players (NaN values preserved)
            coords1 = xy1.frame(t).reshape(-1, 2)
            coords2 = xy2.frame(t).reshape(-1, 2)

            # Calculate pairwise distances (NaN propagates naturally)
            self._pairwise_distances_[t] = cdist(coords1, coords2)

    @requires_fit
    def distance_to_nearest_opponent(
        self,
    ) -> tuple[PlayerProperty, PlayerProperty]:
        """Calculates distance to nearest opponent for each player on both teams.

        Returns
        -------
        distance_to_nearest_opponent: tuple[PlayerProperty, PlayerProperty]
            A tuple of two PlayerProperty objects of shape (T, N) containing distances
            to nearest opponent for each player in the first and second team for each
            frame.
        """
        T, N1, N2 = self._pairwise_distances_.shape
        dtno1 = np.full((T, N1), np.nan)
        dtno2 = np.full((T, N2), np.nan)

        with warnings.catch_warnings():
            # suppress warnings caused by all-NaN slices
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for t in range(T):
                distances = self._pairwise_distances_[t]

                # Calculate minimum distance per player for each team (NaN ignored)
                dtno1[t] = np.nanmin(distances, axis=1)
                dtno2[t] = np.nanmin(distances, axis=0)

        return (
            PlayerProperty(
                property=dtno1,
                name="distance_to_nearest_opponent",
                framerate=self._framerate,
            ),
            PlayerProperty(
                property=dtno2,
                name="distance_to_nearest_opponent",
                framerate=self._framerate,
            ),
        )


class ConvexHullModel(BaseModel):
    """Computations based on the convex hull of player positions.

    Upon calling the :func:`~ConvexHullModel.fit` method, this model calculates convex
    hull objects for each frame. The following calculations can subsequently be queried
    by calling the corresponding methods:

        - Convex Hull Area --> :func:`~ConvexHullModel.convex_hull_area`
        - Convex Hull Visualization --> :func:`~ConvexHullModel.plot`

    Notes
    -----
    The convex hull is computed using scipy's `ConvexHull class
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.
    html>`_ and can be understood as the minimal convex area containing all (outfield)
    players.

    The convex hull is also known in the literature under the terms 'surface area',
    'coverage area' and 'playing area' [3]_.

    When multiple XY objects are provided, all players from all teams are combined and
    the convex hull encompasses all of them. This is commonly referred to as the
    *Effective Area of Play (EAP)* [4]_.

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.geometry import ConvexHullModel

    Single team convex hull (excluding goalkeeper):

    >>> xy = XY(np.array([[0, 0, 10, 0, 10, 10, 0, 10],
    ...                   [0, 0, 10, 0,  5, 10, 5,  5]]))
    >>> chm = ConvexHullModel()
    >>> chm.fit(xy, exclude_xIDs=[[0]])
    >>> chull = chm.convex_hull_area()
    >>> chull
    TeamProperty(property=array([50., 12.5]), name='convex_hull_area', framerate=None)

    Effective Area of Play (both teams):

    >>> xy_home = XY(np.array([[0, 10, 20, 20], [10, 20, 10, 20]]))
    >>> xy_away = XY(np.array([[40, 50, 60, 60], [50, 70, 50, 60]]))
    >>> chm = ConvexHullModel()
    >>> chm.fit([xy_home, xy_away])
    >>> eap = chm.convex_hull_area()
    >>> eap
    TeamProperty(property=array([400., 200.]), name='convex_hull_area', framerate=None)

    References
    ----------
        .. [3] `Moura, F. A., Martins, L. E. B., Anido, R. D. O., De Barros, R. M. L., &
            Cunha, S. A. (2012). Quantitative analysis of Brazilian football players'
            organisation on the pitch. Sports biomechanics, 11(1), 85-96.
            <https://www.tandfonline.com/doi/full/10.1080/14763141.2011.637123>`_
        .. [4] `Clemente, M. F., Couceiro, S. M., Martins, F. M., Mendes, R., &
            Figueiredo, A. J. (2013). Measuring Collective Behaviour in Football Teams:
            Inspecting the impact of each half of the match on ball possession.
            International Journal of Performance Analysis in Sport, 13(3), 678-689.
            <https://www.tandfonline.com/doi/abs/10.1080/24748668.2013.11868680>`_
    """

    def __init__(self):
        super().__init__()
        self._convex_hulls_ = None
        self._framerate = None

    def fit(self, xy: XY | list[XY], exclude_xIDs: list[list] | None = None):
        """Fit the model to the given data and calculate convex hulls.

        Parameters
        ----------
        xy : XY or list[XY]
            Single XY object or list of XY objects. If list, all XY objects will be
            combined and the convex hull will encompass all players (effective playing
            space).
        exclude_xIDs : list[list], optional
            For each XY object, a list of xIDs to exclude from computation. This can be
            useful to exclude goalkeepers from analysis. Length must match number of XY
            objects.

            Examples:

            - Single XY with `xID=0` excluded: ``[[0]]``
            - Two XYs with both `xID=0` excluded: ``[[0], [0]]``
        """
        # Normalize inputs to lists
        xy_list = [xy] if isinstance(xy, XY) else xy

        if exclude_xIDs is None:
            exclude_xIDs = [None] * len(xy_list)
        elif len(exclude_xIDs) != len(xy_list):
            raise ValueError(
                f"exclude_xIDs length ({len(exclude_xIDs)}) must match "
                f"number of XY objects ({len(xy_list)})"
            )

        T = len(xy_list[0])
        self._framerate = xy_list[0].framerate

        # Validate all XY objects have same length
        for i, xy_obj in enumerate(xy_list[1:], start=1):
            if len(xy_obj) != T:
                raise ValueError(
                    f"All XY objects must have same length. "
                    f"xy[0] has {T} frames, xy[{i}] has {len(xy_obj)} frames"
                )

        # Concatenate all XY arrays
        xy_arrays = [xy_obj.xy for xy_obj in xy_list]
        combined_xy = np.hstack(xy_arrays)

        # Create validity mask
        valid_mask = self._create_validity_mask(xy_list, exclude_xIDs)

        # Calculate convex hull for each frame
        self._convex_hulls_ = []
        for t in range(T):
            hull = self._calculate_convex_hull_for_frame(combined_xy[t], valid_mask)
            self._convex_hulls_.append(hull)

    def _create_validity_mask(
        self, xy_list: list[XY], exclude_xIDs: list[list | None]
    ) -> np.ndarray:
        """Create a boolean mask indicating which coordinates to include.

        Parameters
        ----------
        xy_list : list[XY]
            List of XY objects.
        exclude_xIDs : list[list | None]
            Exclusion lists for each XY object.

        Returns
        -------
        valid_mask : np.ndarray
            Boolean array where True = include coordinate, False = exclude.
            Shape matches total number of coordinates in concatenated XY.
        """
        masks = []

        for xy_obj, excl in zip(xy_list, exclude_xIDs):
            # Start with all True (include all)
            mask = np.ones(xy_obj.N * 2, dtype=bool)
            if excl is not None:
                for xid in excl:
                    if xid < 0 or xid >= xy_obj.N:
                        raise ValueError(
                            f"xID {xid} out of range [0, {xy_obj.N-1}] "
                            f"for XY object with {xy_obj.N} players"
                        )
                    # Exclude both x and y coordinates
                    mask[xid * 2] = False
                    mask[xid * 2 + 1] = False

            masks.append(mask)

        # Concatenate all masks
        return np.concatenate(masks)

    def _calculate_convex_hull_for_frame(
        self, frame_data: np.ndarray, valid_mask: np.ndarray
    ) -> ConvexHull | None:
        """Calculate convex hull for a single frame.

        Parameters
        ----------
        frame_data : np.ndarray
            Frame data (1D array of all coordinates).
        valid_mask : np.ndarray
            Boolean mask indicating which coordinates to include.

        Returns
        -------
        hull : ConvexHull or None
            ConvexHull object if successful, None if insufficient valid points.
        """

        MIN_POINTS_FOR_CHULL = 3

        # Exclude players (apply mask)
        masked_data = frame_data[valid_mask]

        # Reshape to (N, 2) coordinate pairs
        points = masked_data.reshape(-1, 2)

        # Exclude points with NaN
        valid_points = points[~np.isnan(points).any(axis=1)]

        # Check for at least 3 points
        if len(valid_points) < MIN_POINTS_FOR_CHULL:
            return None

        # Calculate ConvexHull (scipy handles collinearity/duplicates)
        try:
            return ConvexHull(valid_points)
        except QhullError:
            # QhullError for collinear points, duplicates, or other geometric issues
            return None

    @requires_fit
    def convex_hull_area(self) -> TeamProperty:
        """Calculates the area enclosed by the convex hull.

        Returns
        -------
        convex_hull_area : TeamProperty
            A TeamProperty object of shape (T,), where T is the total number of frames.
            Each entry contains the area enclosed by the convex hull for that frame.
            Frames with insufficient valid points have NaN values.

        Notes
        -----
        If the model was fitted with:

        - Single XY object: returns the team's convex hull area
        - Multiple XY objects: returns the effective area of play (EAP)
        """

        areas = np.array(
            [
                hull.volume if hull is not None else np.nan
                for hull in self._convex_hulls_
            ]
        )

        return TeamProperty(
            property=areas, name="convex_hull_area", framerate=self._framerate
        )

    @requires_fit
    def plot(
        self, t: int, ax=None, fill: bool = True, fill_alpha: float = 0.3, **kwargs
    ):
        """Plot the convex hull for a given time point on a matplotlib axes.

        Parameters
        ----------
        t : int
            Frame index to plot.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new figure and axes are created.
        fill : bool, optional
            Whether to fill the convex hull polygon. Default is True.
        fill_alpha : float, optional
            Transparency of the fill (0=transparent, 1=opaque). Default is 0.3.
            Only used if fill=True.
        **kwargs : optional
            Additional keyword arguments passed to the line plot.
            Common options:

            - color : str, default 'black'
            - alpha : float, default 1.0 (line transparency)
            - linewidth : float, default 2
            - linestyle : str (e.g., '--', ':')
            - Any other matplotlib.axes.Axes.plot() parameters

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object with the convex hull plotted.

        Notes
        -----
        The kwargs are passed to the plot function of matplotlib for drawing the
        hull boundary. To customize the plots have a look at
        `matplotlib
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from floodlight import Pitch
        >>>
        >>> pitch = Pitch(
        ... xlim=(0, 105), ylim=(0, 68), sport="football", unit="m", boundaries="fixed"
        ... )
        >>>
        >>> fig, ax = plt.subplots()
        >>> pitch.plot(ax=ax)
        >>>
        >>> # Filled hull with custom color
        >>> chm.plot(t=0, ax=ax, color='blue', fill_alpha=0.2)

        .. image:: ../../_img/sample_chm_plot_filled.png

        >>> # Dashed outline, no fill
        >>> chm.plot(t=0, ax=ax, fill=False, linestyle="--", linewidth=3)

        .. image:: ../../_img/sample_chm_plot_dashed.png
        """

        ax = ax or plt.subplots()[1]

        hull = self._convex_hulls_[t]

        # Handle None hull - return axes
        if hull is None:
            return ax

        # Extract plotting parameters with defaults
        color = kwargs.pop("color", "black")
        linewidth = kwargs.pop("linewidth", 2)

        # Get hull vertices
        points = hull.points
        vertices = hull.vertices

        # Close the polygon
        vertices_closed = np.append(vertices, vertices[0])
        hull_points = points[vertices_closed]

        # Plot boundary
        ax.plot(
            hull_points[:, 0],
            hull_points[:, 1],
            color=color,
            linewidth=linewidth,
            **kwargs,
        )

        # Fill if requested
        if fill:
            ax.fill(hull_points[:, 0], hull_points[:, 1], color=color, alpha=fill_alpha)

        return ax

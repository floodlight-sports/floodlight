import warnings

import numpy
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull

from floodlight import XY
from floodlight.core.property import TeamProperty, PlayerProperty
from floodlight.models.base import BaseModel, requires_fit
from floodlight.models.utils import _exclude_x_ids


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
        # array to indicate which players are to be excluded from the calculation
        include = _exclude_x_ids(xy, exclude_xIDs)

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
            A TeamProperty object of shape (T, 1), where T is the total number of
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


class ConvexHullModel(BaseModel):
    """Class for convex hull calculation.

    Upon calling the :func:`~ConvexHullModel.fit`-method, this model creates a
    ConvexHull object of the team for every frame.

    Notes
    -----
    The convex hull is computed using the
    `ConvexHull class
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.
    html>`_ from scipy.spatial and can be understood as the minimal convex area
    containing all outfield players [reference].

    The convex hull is also known in the literature under the terms ‘surface area’,
    ‘coverage area’ and ‘playing area’ [1].

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.geometry import ConvexHullModel

    >>> xy = XY(np.array(   [0, 0, 10, 0, 10, 10, 0, 10, 5, 5], # square (10x10)
    >>>                     [0, 0, 10, 0, 5, 10, 2, 2, 3, 3],   # triangle ((1/2)x10x10)
    >>>                     [0, 5, 5, 10, 10, 5, 5, 0, 5, 5]))  # rhombus (5x10)
    >>> exclude_xIDs = [4]
    >>> chull = ConvexHullModel()
    >>> chull.fit(xy=xy, exclude_xIDs=exclude_xIDs)
    >>> chull.area_convex_hull()
    TeamProperty(property=array([100, 50, 50]), name='area_convex_hull', framerate=None)

    The area enclosed by the convex hull can be visualised using matplotlib as follows:

    >>> import matplotlib.pyplot as plt
    >>> c_hulls=[]
    >>> c_hulls = chull.convex_hulls
    >>> plt.plot(c_hulls[0].points[:, 0], c_hulls[0].points[:, 1], 'o')
    >>> for simplex in c_hulls[0].simplices:
    >>>     plt.plot(c_hulls[0].points[simplex, 0], c_hulls[0].points[simplex, 1], 'k-')
    >>> plt.show()

    .. image:: ../../_img/convex_hull.png

    References
    ----------
        .. [1] `Low, B., Coutinho, D., Gonçalves, B., Rein, R., Memmert, D., & Sampaio,
            J. (2020). A Systematic Review of Collective Tactical Behaviours in Football
            Using Positional Data. Sports Medicine, 50(2), 343–385.
            <https://link.springer.com/article/10.1007/s40279-019-01194-7>`_
    """

    def __init__(self):
        super().__init__()
        # model parameter
        self.convex_hulls = None

    def fit(self, xy: XY, exclude_xIDs: list = None, **kwargs):
        """
        Fit the model to the given data and create ConvexHull objects from scipy.spatial
        for each frame.

        Parameters
        ----------
        xy: XY
            Player spatiotemporal data for which the ConvexHull object is created.
        exclude_xIDs: list, optional
            A list of xIDs to be excluded from computation. This can be useful if one
            would like, for example, to exclude goalkeepers from analysis.
        kwargs:
            Optional keyworded arguments e.g. {'incremental: bool = False',
            'qhull_options: str' = None} which are passed to the ConvexHull objects
            (`scipy.spatial.ConvexHull
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.
            html>`_).
        """

        # array to indicate which players are to be excluded from the calculation
        include = _exclude_x_ids(xy, exclude_xIDs)

        # check for NaN's in xy since ConvexHull from scipy can not handle NaN values
        if np.isnan(xy[:, include]).any():
            raise ValueError("xy array contains NaN")

        convex_hulls = list(range(len(xy)))

        # create and add ConvexHull objects to list
        for t in range(len(xy)):
            convex_hulls[t] = ConvexHull(xy.frame(t)[include].reshape(-1, 2), **kwargs)

        self.convex_hulls = convex_hulls

    @requires_fit
    def area_convex_hull(self) -> TeamProperty:
        """Calculates the area enclosed by the convex hull.

        Returns
        -------
        area_convex_hulls: TeamProperty
            A TeamProperty object of shape (T, 1), where T is the total number of
            frames. Each entry contains the area enclosed by the convex hull of that
            particular frame.
        """
        # List of the areas of the convex hulls for each image
        areas = np.full((len(self.convex_hulls), 1), np.nan)

        # add area of the convex hull for every frame to list
        for i, c_hull in enumerate(self.convex_hulls):
            areas[i] = c_hull.volume  # for 2D area = volume

        # generate TeamProperty object and add calculated areas as property
        area_convex_hulls = TeamProperty(
            property=np.concatenate(areas), name="area_convex_hull"
        )

        return area_convex_hulls

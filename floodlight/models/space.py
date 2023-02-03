from typing import Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.spatial.distance import cdist

from floodlight import XY, Pitch, TeamProperty, PlayerProperty
from floodlight.models.base import BaseModel, requires_fit


class DiscreteVoronoiModel(BaseModel):
    """Calculates discretized versions of the Voronoi tessellation commonly used to
    assess space control.

    Upon instantiation, this model creates a mesh grid that spans the entire pitch with
    a fixed number of mesh points. When calling the
    :func:`~DiscreteVoronoiModel.fit`-method, closest players to the respective mesh
    points are evaluated and their control assigned to players. Thus, cumulative
    controls and controlled areas are calculated on a discretization of the pitch. The
    following calculations can subsequently be queried by calling the corresponding
    methods:

        - Player Space Control --> :func:`~DiscreteVoronoiModel.player_controls`
        - Team Space Control --> :func:`~DiscreteVoronoiModel.team_controls`

    Furthermore, the following plotting methods are available to visualize the model:

        - Plot controlled areas --> :func:`~DiscreteVoronoiModel.plot`
        - Plot mesh grid --> :func:`~DiscreteVoronoiModel.plot_mesh`

    Parameters
    ----------
    pitch: Pitch
        A floodlight Pitch object corresponding to the XY data that will be supplied to
        the model. The mesh created during instantiation will span this pitch.
    mesh: {'square', 'hexagonal'}, optional
        A string indicating the type of mesh that will be generated. 'square' will
        generate a grid-like mesh with square cell shapes (default). 'hexagonal' will
        generate a mesh with hexagonal cell shapes where mesh points have equidistant
        neighbours.
    xpoints: int, optional
        The number of mesh grid points used in x-direction. Must be in range [10, 1000]
        and defaults to 100. The number of messh grid points in y-direction will be
        inferred automatically to match the shape of the pitch and produce regular mesh
        cell shapes.

    Notes
    -----
    The original work by Taki and Hasegawa proposed to use Voronoi tessellations for
    assessing player dominant regions [1]_. This approach has later been simplified by
    using the Euclidean distance when allocating space to players [2]_ , [3]_.
    Instead of computing algebraic Voronoi regions, this model discretizes the problem
    by sampling space control on a finite number of mesh points across the pitch. This
    runs much faster and can be easier to handle. If an appropriate number of mesh
    points is chosen, the resulting error is expected to be negligible given the common
    spatial inaccuracies of tracking data as well as variations in moving players'
    centers of masses.

    References
    ----------
        .. [1] `Taki, T., & Hasegawa, J. (2000). Visualization of dominant region in
            team games and its application to teamwork analysis. Proceedings Computer
            Graphics International 2000, 227–235.
            <https://ieeexplore.ieee.org/document/852338>`_
        .. [2] `Fonseca, S., Milho, J., Travassos, B., & Araújo, D. (2012). Spatial
            dynamics of team sports exposed by Voronoi diagrams. Human Movement
            Science, 31(6), 1652–1659. <https://doi.org/10.1016/j.humov.2012.04.006>`_
        .. [3] `Rein, R., Raabe, D., & Memmert, D. (2017). “Which pass is better?”
            Novel approaches to assess passing effectiveness in elite soccer. Human
            Movement Science, 55, 172–181.
            <https://doi.org/10.1016/j.humov.2017.07.010>`_

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY, Pitch
    >>> from floodlight.models.space import DiscreteVoronoiModel

    >>> # create data and fit model
    >>> xy1 = XY(np.array(((10, 10, 20, 80, 30, 40), (10, 10, np.nan, np.nan, 35, 35))))
    >>> xy2 = XY(np.array(((90, 90, 80, 20, 75, 80), (90, 90, 75, 25, 80, 70))))
    >>> pitch = Pitch.from_template("opta", length=105, width=68)
    >>> dvm = DiscreteVoronoiModel(pitch)
    >>> dvm.fit(xy1, xy2)

    >>> # print player controls [%] for first team
    >>> player_control1, player_control2 = dvm.player_controls()
    >>> print(player_control1.property)
    [[10.63 19.32 21.71]
     [10.35  0.   36.56]]

    >>> # print team controls [%] for first team
    >>> team_control1, team_control2 = dvm.team_controls()
    >>> print(team_control1.property)
    [[51.66]
     [46.91]]
    """

    def __init__(self, pitch: Pitch, mesh: str = "square", xpoints: int = 100):
        super().__init__(pitch)

        # input parameter
        self._mesh_type = mesh
        self._xpoints = xpoints

        # model parameter
        self._meshx_ = None
        self._meshy_ = None
        self._xpolysize_ = None
        self._ypolysize_ = None
        self._T_ = None
        self._N1_ = None
        self._N2_ = None
        self._framerate = None
        self._cell_controls_ = None

        # checks
        valid_mesh_types = ["square", "hexagonal"]
        if mesh not in valid_mesh_types:
            raise ValueError(
                f"Invalid mesh type. Expected one of {valid_mesh_types}, got {mesh}"
            )
        if xpoints < 10 or xpoints > 1000:
            raise ValueError(
                f"Expected xpoints to be in range [10, 1000], got {xpoints}"
            )

        # generate mesh
        self._generate_mesh(mesh, xpoints)

    def _generate_mesh(self, mesh: str = "square", xpoints: int = 100) -> None:
        """Generates a np.meshgrid for a given mesh type."""
        # param
        self._meshx_ = None
        self._meshy_ = None
        self._xpolysize_ = None
        self._ypolysize_ = None
        xmin, xmax = self._pitch.xlim
        ymin, ymax = self._pitch.ylim

        if mesh == "square":
            # determine square size
            self._xpolysize_ = (xmax - xmin) / xpoints
            self._ypolysize_ = self._xpolysize_

            # derive number of points in y direction
            ypoints = round((ymax - ymin) / self._ypolysize_)
            # re-adjust ypolysize for stretching/rounding in y direction
            self._ypolysize_ = (ymax - ymin) / ypoints

            # get padding
            xpad = self._xpolysize_ * 0.5
            ypad = self._ypolysize_ * 0.5

            # create unilateral and two-dimensional grid points
            x = np.linspace(xmin + xpad, xmax - xpad, xpoints)
            y = np.linspace(ymax - ypad, ymin + ypad, ypoints)
            self._meshx_, self._meshy_ = np.meshgrid(x, y)

        elif mesh == "hexagonal":
            # longitudinal spacing of polygons (minus half polygon that's out of bounds)
            xspace = (xmax - xmin) / (xpoints - 0.5)
            # hexagon size (= radius of outer circumcircle)
            self._xpolysize_ = xspace / np.sqrt(3)
            self._ypolysize_ = self._xpolysize_
            # lateral spacing of polygons (by formula)
            yspace = self._xpolysize_ * 1.5
            # longitudinal padding, also offset for odd rows of polygons
            xpad = xspace * 0.5

            # derive number of points in y direction
            ypoints = round((ymax - ymin) / yspace) + 1

            # unilateral and two-dimensional grid points
            x = np.linspace(xmin, xmax - xpad, xpoints)
            y = np.linspace(ymax, ymin, ypoints)
            self._meshx_, self._meshy_ = np.meshgrid(x, y)

            # add offset for odd rows
            self._meshx_[1::2, :] += xpad

    def _calc_cell_controls(self, xy1: XY, xy2: XY):
        """Calculates xID of closest player to each mesh point at each time point and
        stores results in self._cell_controls"""
        # bin
        T = len(xy1)
        self._cell_controls_ = np.full(
            # shape is: time x (mesh shape)
            (T, self._meshx_.shape[0], self._meshx_.shape[1]),
            np.nan,
        )

        # loop
        for t in range(T):
            # stack and reshape player and mesh coordinates to (M x 2) arrays
            player_points = np.hstack((xy1.frame(t), xy2.frame(t))).reshape(-1, 2)
            mesh_points = np.stack((self._meshx_, self._meshy_), axis=2).reshape(-1, 2)

            # calculate pairwise distances and determine closest player
            pairwise_distances = cdist(mesh_points, player_points)
            closest_player_index = np.nanargmin(pairwise_distances, axis=1)
            self._cell_controls_[t] = closest_player_index.reshape(self._meshx_.shape)

    def fit(self, xy1: XY, xy2: XY):
        """Fit the model to the given data and calculate control values for mesh points.

        Parameters
        ----------
        xy1: XY
            Player spatiotemporal data of the first team.
        xy2: XY
            Player spatiotemporal data of the second team.
        """
        # derive parameters
        self._N1_ = xy1.N
        self._N2_ = xy2.N
        self._T_ = len(xy1)
        self._framerate = xy1.framerate
        # invoke control calculation
        self._calc_cell_controls(xy1, xy2)

    @requires_fit
    def player_controls(self) -> Tuple[PlayerProperty, PlayerProperty]:
        """Returns the percentage of mesh points controlled by each player of the first
        and second team.

        Returns
        -------
        player_controls: Tuple[PlayerProperty, PlayerProperty]
            One Property object for each team (corresponding to the fitted xy1 and xy2)
            of shape (n_frames x n_players), respectively. Property objets contain the
            percentage of points controlled by each player on the pitch.
        """
        # infer number of mesh cells
        number_of_cells = self._cell_controls_.shape[1] * self._cell_controls_.shape[2]

        # xID ranges for both team's players if stacked together
        range1 = range(self._N1_)
        range2 = range(self._N1_, self._N1_ + self._N2_)

        # for each xID count number of cell controls in each mesh through time
        counts1 = [np.sum(self._cell_controls_ == xID, axis=(1, 2)) for xID in range1]
        counts2 = [np.sum(self._cell_controls_ == xID, axis=(1, 2)) for xID in range2]

        # transform to arrays and normalize
        counts1 = np.array(counts1).transpose()
        counts2 = np.array(counts2).transpose()

        # transform to percentages
        percentages1 = np.round(100 * counts1 / number_of_cells, 2)
        percentages2 = np.round(100 * counts2 / number_of_cells, 2)

        # create objects
        property1 = PlayerProperty(
            property=percentages1, name="space control", framerate=self._framerate
        )
        property2 = PlayerProperty(
            property=percentages2, name="space control", framerate=self._framerate
        )

        return property1, property2

    @requires_fit
    def team_controls(self) -> Tuple[TeamProperty, TeamProperty]:
        """Returns the percentage of mesh points controlled by the first and second
        team.

        Returns
        -------
        team_controls: Tuple[TeamProperty, TeamProperty]
            One Property object for each team (corresponding to the fitted xy1 and xy2)
            of shape (n_frames x 1), respectively. Property objets contain the
            percentage of points controlled by each team on the pitch.
        """
        # infer number of mesh cells
        number_of_cells = self._cell_controls_.shape[1] * self._cell_controls_.shape[2]

        # count number of cell controls for a team in each mesh through time
        counts1 = np.sum(self._cell_controls_ < self._N1_, axis=(1, 2))
        counts2 = np.sum(self._cell_controls_ >= self._N1_, axis=(1, 2))

        # transform to arrays and normalize
        counts1 = np.array(counts1).reshape(-1, 1)
        counts2 = np.array(counts2).reshape(-1, 1)

        # transform to percentages
        percentages1 = np.round(100 * counts1 / number_of_cells, 2)
        percentages2 = np.round(100 * counts2 / number_of_cells, 2)

        # create objects
        property1 = TeamProperty(
            property=percentages1, name="space control", framerate=self._framerate
        )
        property2 = TeamProperty(
            property=percentages2, name="space control", framerate=self._framerate
        )

        return property1, property2

    @requires_fit
    def plot(
        self,
        t: int = 0,
        team_colors: Tuple[str, str] = ("red", "blue"),
        ax: matplotlib.axes = None,
        **kwargs,
    ) -> matplotlib.axes:
        """Plots the fitted mesh grid colored by team controls for a given time point
        on a matplotlib axes.

        Parameters
        ----------
        t: int, optional
            Frame for which controls are plotted. Defaults to 0.
        team_colors: Tuple[str, str], optional
            Tuple of two colors in a format accepted by matplotlib that is used to
            color team specific control areas. Defaults to ('red', 'blue').
        ax: matplotlib.axes, optional
            Axes from matplotlib library to plot on. Defaults to None.
        kwargs:
            Optional keyworded arguments e.g. {'zorder', 'ec', 'alpha'} which can be
            used for the plot functions from matplotlib. The kwargs are only passed to
            all the plot functions of matplotlib. If not given default values are used.

        Returns
        -------
        axes: matplotlib.axes
            Axes from matplotlib library with plot.

        Notes
        -----
        The kwargs are only passed to the plot functions of matplotlib. To customize the
        plots have a look at
        `matplotlib
        <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.

        Examples
        --------
        Given a DiscreteVoronoiModel that has already been fitted:

        >>> # fitted_dvm_model has square mesh
        >>> ax = pitch.plot(color_scheme="bw")
        >>> fitted_dvm_model.plot(ax=ax)

        .. image:: ../../_img/sample_dvm_plot_square.png

        >>> # fitted_dvm_model has hexagonal mesh
        >>> ax = pitch.plot(color_scheme="bw")
        >>> fitted_dvm_model.plot(ax=ax)

        .. image:: ../../_img/sample_dvm_plot_hex.png
        """
        # get ax
        ax = ax or plt.subplots()[1]

        # get colors and construct team color vector
        team_color1, team_color2 = team_colors
        color_vector = [team_color1] * self._N1_ + [team_color2] * self._N2_

        # call plot by mesh type
        if self._mesh_type == "square":
            ax = self._plot_square(t, color_vector, ax=ax, **kwargs)
        elif self._mesh_type == "hexagonal":
            ax = self._plot_hexagonal(t, color_vector, ax=ax, **kwargs)

        return ax

    def _plot_square(
        self,
        t: int = 0,
        team_colors: Tuple[str, str] = None,
        ax: matplotlib.axes = None,
        **kwargs,
    ) -> matplotlib.axes:
        """Plots square mesh grid controls in given color."""
        # handle kwargs
        ec = kwargs.pop("ec", "grey")
        alpha = kwargs.pop("alpha", 0.3)

        # offset to shift rectangle position from bottom left corner to center
        xoffset = -(self._xpolysize_ * 0.5)
        yoffset = -(self._ypolysize_ * 0.5)
        # loop through mesh points and plot Rectangle patch
        for i, j in np.ndindex(self._meshx_.shape):
            poly = plt.Rectangle(
                (self._meshx_[i, j] + xoffset, self._meshy_[i, j] + yoffset),
                width=self._xpolysize_,
                height=self._ypolysize_,
                fc=team_colors[int(self._cell_controls_[t, i, j])],
                ec=ec,
                alpha=alpha,
                **kwargs,
            )
            ax.add_patch(poly)

        return ax

    def _plot_hexagonal(
        self,
        t: int = 0,
        team_colors: Tuple[str, str] = None,
        ax: matplotlib.axes = None,
        **kwargs,
    ) -> matplotlib.axes:
        """Plots hexagonal mesh grid controls in given color."""
        # handle kwargs
        ec = kwargs.pop("ec", "grey")
        alpha = kwargs.pop("alpha", 0.3)

        # hexagons are regular polygons with 6 vertices
        n_vertices = 6
        # loop through mesh points and plot RegularPolygon patch
        for (i, j), x in np.ndenumerate(self._meshx_):
            poly = RegularPolygon(
                (x, self._meshy_[i, j]),
                numVertices=n_vertices,
                radius=self._xpolysize_,
                fc=team_colors[int(self._cell_controls_[t, i, j])],
                ec=ec,
                alpha=alpha,
                **kwargs,
            )
            ax.add_patch(poly)

        return ax

    def plot_mesh(self, ax: matplotlib.axes = None) -> matplotlib.axes:
        """Plots the generated mesh on a matplotlib.axes.

        Parameters
        ----------
        ax: matplotlib.axes, optional
            Matplotlib axes on which the mesh points are plotted. If ax is None, a
            default-sized matplotlib.axes object is created.

        Returns
        -------
        axes: matplotlib.axes
            Matplotlib axes on which the mesh points are plotted.

        Examples
        --------
        Given a DiscreteVoronoiModel that has already been fitted:

        >>> ax = pitch.plot(color_scheme="bw")
        >>> fitted_dvm_model.plot_mesh(ax=ax)

        .. image:: ../../_img/sample_dvm_plot_hex_mesh.png
        """
        # get ax
        ax = ax or plt.subplots()[1]
        # plot mesh
        ax.plot(self._meshx_, self._meshy_, "ok", markersize=0.5)

        return ax

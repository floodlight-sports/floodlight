from typing import Tuple
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.spatial.distance import cdist

from floodlight import XY, Pitch, TeamProperty, PlayerProperty
from floodlight.models.base import BaseModel, requires_fit
from floodlight.models.kinematics import VelocityVectorModel


class SpaceControlModel(BaseModel):
    """
    Computes space control on a discretized pitch using different underlying models.

    The SpaceControlModel allows for different definitions of control:
    - Euclidean Voronoi tessellation: Each mesh point is assigned to the nearest player.
    - Taki-Hasegawa motion model: Each mesh point is assigned to the player who can
      reach it fastest, considering initial velocity and maximum acceleration.

    Upon instantiation, this model creates a mesh grid that spans the entire pitch with
    a fixed number of mesh points. When calling the :func:`~SpaceControlModel.fit`
    method, each mesh point is assigned to the controlling player according to the
    selected model. Cumulative controls and controlled areas are then computed across
    time.

    The following calculations can subsequently be queried by calling the corresponding
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
    max_acceleration : float, optional
        Maximum acceleration used in the Taki-Hasegawa model. Defaults to 4.0 m/s².
    model : {'euclidean', 'taki_hasegawa'}, optional
        String identifier for the control model to be used. Defaults to 'euclidean'.

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

    In addition to the Euclidean model, this class supports a dynamic motion-based
    model based on the formulation proposed by Taki and Hasegawa [1]_.
    This approach assigns each mesh point to the player who can reach it fastest,
    considering both current velocity and maximum acceleration. It is particularly
    suitable for analyzing movement intentions and temporal dominance structures.

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
    >>> from floodlight.models.space import SpaceControlModel

    >>> # create data and fit model using Euclidean control
    >>> xy1 = XY(np.array(((10, 10, 20, 80, 30, 40), (10, 10, np.nan, np.nan, 35, 35))))
    >>> xy2 = XY(np.array(((90, 90, 80, 20, 75, 80), (90, 90, 75, 25, 80, 70))))
    >>> pitch = Pitch.from_template("opta", length=105, width=68)
    >>> scm = SpaceControlModel(pitch, model="euclidean")
    >>> scm.fit(xy1, xy2)

    >>> # get player controls [%]
    >>> player_control1, player_control2 = scm.player_controls()
    >>> print(player_control1.property)
    [[10.63 19.32 21.71]
     [10.35  0.   36.56]]

    >>> # get team controls [%]
    >>> team_control1, team_control2 = scm.team_controls()
    >>> print(team_control1.property)
    [[51.66]
     [46.91]]
    """

    # Supported space control model identifiers
    EUCLIDEAN = "euclidean"
    TAKI_HASEGAWA = "taki_hasegawa"

    def __init__(
        self,
        pitch: Pitch,
        mesh: str = "square",
        xpoints: int = 100,
        max_acceleration: float = 4.0,
        model: str = EUCLIDEAN,
    ):
        super().__init__(pitch)

        # input parameter
        self._mesh_type = mesh
        self._xpoints = xpoints
        self._max_acceleration = max_acceleration
        self._model = model
        self._pitch = pitch

        valid_models = [self.EUCLIDEAN, self.TAKI_HASEGAWA]
        if self._model not in valid_models:
            raise ValueError(
                f"Invalid model type. Expected one of {valid_models}, "
                f"got '{self._model}'"
            )

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
        self._velocityvector1 = None
        self._velocityvector2 = None

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
        """
        Calculates the xID of the controlling player for each mesh point at each
        time point and stores the results in self._cell_controls.
        """
        # bin
        n_frames = len(xy1)
        self._cell_controls_ = np.full(
            # shape is: time x (mesh shape)
            (n_frames, self._meshx_.shape[0], self._meshx_.shape[1]),
            np.nan,
        )

        mesh_points = np.stack((self._meshx_, self._meshy_), axis=2).reshape(-1, 2)

        if self._model == self.EUCLIDEAN:
            # loop
            for t in range(n_frames):
                # stack and reshape player and mesh coordinates to (M x 2) arrays
                player_points = np.hstack((xy1.frame(t), xy2.frame(t))).reshape(-1, 2)

                # calculate pairwise distances and determine closest player
                pairwise_distances = cdist(mesh_points, player_points)
                closest_player_index = self._masked_nanargmin(pairwise_distances)

                self._cell_controls_[t] = closest_player_index.reshape(
                    self._meshx_.shape
                )

        elif self._model == self.TAKI_HASEGAWA:
            # loop
            for frame in range(n_frames):
                # combine stacked positions and velocities into (P x 4) array
                pos1 = xy1.xy[frame].reshape(-1, 2)
                pos2 = xy2.xy[frame].reshape(-1, 2)
                player_positions = np.vstack((pos1, pos2))
                vel1 = self._velocityvector1[frame]
                vel2 = self._velocityvector2[frame]
                player_velocities = np.vstack((vel1, vel2))
                player_positions_velocities = np.hstack(
                    (player_positions, player_velocities)
                )

                # compute shortest arrival times and identify fastest player
                pairwise_times = self._calculate_shortest_times(
                    mesh_points, player_positions_velocities
                )

                fastest_player_index = self._masked_nanargmin(pairwise_times)

                self._cell_controls_[frame] = fastest_player_index.reshape(
                    self._meshx_.shape
                )

    def fit(self, xy1: XY, xy2: XY):
        """Fit the model to the given data and calculate control values for mesh points.

        Parameters
        ----------
        xy1: XY
            Player spatiotemporal data of the first team.
        xy2: XY
            Player spatiotemporal data of the second team.
        """

        # Check if both inputs are valid XY objects
        if not isinstance(xy1, XY) or not isinstance(xy2, XY):
            raise TypeError("Both inputs must be valid XY objects.")

        # check if xy1 and xy2 are valid
        if len(xy1) != len(xy2):
            raise ValueError("XY objects must have the same number of frames.")

        # check for out-of-bounds player positions
        mask1 = self._check_positions_within_pitch(xy1, self._pitch)
        mask2 = self._check_positions_within_pitch(xy2, self._pitch)
        if not np.all(mask1):
            warnings.warn(
                "Some player positions in xy1 lie outside the pitch boundaries. "
                f"Check the provided pitch dimensions (length={self._pitch.length}, "
                f"width={self._pitch.width})."
            )
        if not np.all(mask2):
            warnings.warn(
                "Some player positions in xy2 lie outside the pitch boundaries. "
                f"Check the provided pitch dimensions (length={self._pitch.length}, "
                f"width={self._pitch.width})."
            )

        # sanitize XY data to handle NaNs properly
        xy1 = self._sanitize_nan_coordinates(xy1)
        xy2 = self._sanitize_nan_coordinates(xy2)

        # derive parameters
        self._N1_ = xy1.N
        self._N2_ = xy2.N
        self._T_ = len(xy1)
        self._framerate = xy1.framerate

        if self._model == self.TAKI_HASEGAWA:
            self._compute_velocities(xy1, xy2)

        # invoke control calculation
        self._calc_cell_controls(xy1, xy2)

    @staticmethod
    def _check_positions_within_pitch(xy: XY, pitch: Pitch) -> np.ndarray:
        """Return a mask of shape (T, N) where True = position within pitch bounds."""
        x = xy.xy[:, ::2]
        y = xy.xy[:, 1::2]
        in_x_bounds = (x >= pitch.xlim[0]) & (x <= pitch.xlim[1]) | np.isnan(x)
        in_y_bounds = (y >= pitch.ylim[0]) & (y <= pitch.ylim[1]) | np.isnan(y)
        return in_x_bounds & in_y_bounds

    @staticmethod
    def _sanitize_nan_coordinates(xy: XY) -> XY:
        """Ensure NaN consistency by setting both x and y to NaN if either is NaN."""
        # Create a copy of the xy array and convert to float to avoid issues with NaN
        sanitized_xy_array = xy.xy.copy().astype(float)

        # Detect all x or y that are nan (shape: T, N*2)
        x_nan = np.isnan(sanitized_xy_array[:, ::2])
        y_nan = np.isnan(sanitized_xy_array[:, 1::2])

        # Combine masks to get players with either x or y nan
        any_nan = x_nan | y_nan

        # Broadcast to both x and y components
        sanitized_xy_array[:, ::2][any_nan] = np.nan
        sanitized_xy_array[:, 1::2][any_nan] = np.nan

        return XY(sanitized_xy_array, framerate=xy.framerate, direction=xy.direction)

    @staticmethod
    def _masked_nanargmin(array: np.ndarray) -> np.ndarray:
        """
        Return index of minimum value in each row, ignoring rows with only NaNs or infs.
        Sets result to np.nan for rows without finite values.
        """

        # set times to np.inf for invalid player positions (e.g., NaN)
        invalid_mask = np.all(~np.isfinite(array), axis=1)
        result = np.full(array.shape[0], np.nan)
        valid_rows = ~invalid_mask
        result[valid_rows] = np.nanargmin(array[valid_rows], axis=1)
        return result

    def _compute_velocities(self, xy1: XY, xy2: XY) -> None:
        """
        Compute and store player velocity vectors required by dynamic models.
        If velocities have already been computed, this method does nothing.
        """
        if self._velocityvector1 is not None and self._velocityvector2 is not None:
            return

        vvm1 = VelocityVectorModel()
        vvm2 = VelocityVectorModel()
        vvm1.fit(xy1)
        vvm2.fit(xy2)
        self._velocityvector1 = vvm1.velocityvector().property
        self._velocityvector2 = vvm2.velocityvector().property

    def _calculate_shortest_times(
        self, mesh_points: np.ndarray, player_points: np.ndarray
    ) -> np.ndarray:
        """Compute shortest arrival times from players to mesh points."""

        # extract player positions and velocities from input array
        pos = player_points[:, :2]
        vel = player_points[:, 2:]

        # replace NaN velocities with zeros for stable projection computation
        vel = np.nan_to_num(vel, nan=0.0)

        # compute vector from each player to each mesh point
        d = mesh_points[:, np.newaxis, :] - pos[np.newaxis, :, :]

        # compute Euclidean distances between players and mesh points
        d_norm = np.linalg.norm(d, axis=2, keepdims=True)
        d_norm_squeezed = np.squeeze(d_norm, axis=2)

        # compute unit direction vectors from players to mesh points
        d_hat = np.divide(d, d_norm, out=np.zeros_like(d), where=d_norm != 0)

        # project player velocities onto direction vectors toward mesh points
        v_proj = np.sum(d_hat * vel[np.newaxis, :, :], axis=2)

        # set uniform acceleration magnitude for all players and directions
        a_proj = self._max_acceleration

        # compute discriminant and identify valid (real-valued) solutions
        disc = v_proj**2 + 2 * a_proj * d_norm_squeezed
        valid = disc >= 0

        # compute square root of discriminant only for valid entries
        sqrt_disc = np.zeros_like(disc)
        sqrt_disc[valid] = np.sqrt(disc[valid])

        # solve quadratic equation for time using the quadratic formula
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-v_proj + sqrt_disc) / a_proj
            t2 = (-v_proj - sqrt_disc) / a_proj

        # discard invalid or negative time solutions
        t1[~valid | (t1 < 0)] = np.inf
        t2[~valid | (t2 < 0)] = np.inf

        # choose the smaller valid time as the shortest arrival time
        times = np.minimum(t1, t2)

        # set time to 0.0 if player is already at the mesh point
        times[d_norm_squeezed == 0] = 0.0

        return times

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
        Given a SpaceControlModel that has already been fitted:

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
            control_value = self._cell_controls_[t, i, j]
            # skip NaN values
            if np.isnan(control_value):
                continue

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
            control_value = self._cell_controls_[t, i, j]
            # skip NaN values
            if np.isnan(control_value):
                continue

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
        Given a SpaceControlModel that has already been fitted:

        >>> ax = pitch.plot(color_scheme="bw")
        >>> fitted_dvm_model.plot_mesh(ax=ax)

        .. image:: ../../_img/sample_dvm_plot_hex_mesh.png
        """
        # get ax
        ax = ax or plt.subplots()[1]
        # plot mesh
        ax.plot(self._meshx_, self._meshy_, "ok", markersize=0.5)

        return ax

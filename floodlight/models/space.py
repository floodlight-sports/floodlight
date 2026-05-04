from typing import Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.spatial.distance import cdist

from floodlight import XY, Pitch, TeamProperty, PlayerProperty
from floodlight.models.base import BaseModel, requires_fit
from floodlight.models.kinematics import VelocityModel


class DiscreteVoronoiModel(BaseModel):
    """Calculates discretized dominant regions commonly used to assess space control.

    Upon instantiation, this model creates a mesh grid that spans the entire pitch with
    a fixed number of mesh points. When calling the
    :func:`~DiscreteVoronoiModel.fit`-method, the controlling player for each mesh
    point is determined according to the selected movement model. By default, the
    Euclidean distance is used, which corresponds to a classical Voronoi tessellation.
    Alternatively, motion-based models can be selected that account for player velocity
    and acceleration. Cumulative controls and controlled areas are then calculated on a
    discretization of the pitch. The following calculations can subsequently be queried
    by calling the corresponding methods:

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
        and defaults to 100. The number of mesh grid points in y-direction will be
        inferred automatically to match the shape of the pitch and produce regular mesh
        cell shapes.
    motion_model: {'euclidean', 'taki_hasegawa', 'fujimura_sugihara'}, optional
        The motion model used to assign mesh points to players. 'euclidean'
        (default) assigns each mesh point to the nearest player by Euclidean distance.
        'taki_hasegawa' assigns each mesh point to the player who can reach it fastest,
        considering initial velocity and maximum acceleration.
        'fujimura_sugihara' uses an exponential velocity decay model where players
        approach a terminal velocity with a drag coefficient.
    max_acceleration: float, optional
        Maximum player acceleration in m/s². Only used by 'taki_hasegawa'.
        Defaults to 4.2 according to [5]_.
    vmax: float, optional
        Terminal velocity in m/s used by 'fujimura_sugihara'. Defaults to
        7.8, according to [4]_.
    alpha: float, optional
        Drag coefficient controlling how quickly players approach terminal
        velocity used by 'fujimura_sugihara'. Defaults to 1.3 according to [4]_.

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

    In addition to the Euclidean model, this class supports the motion-based model by
    Taki and Hasegawa [1]_. This approach assigns each mesh point to the player who can
    reach it fastest, considering both current velocity and maximum acceleration. For
    each player at position :math:`\\mathbf{p}` with velocity :math:`\\mathbf{v}`, the
    minimum arrival time :math:`t^*` to a target point :math:`\\mathbf{x}` under the
    isotropic acceleration constraint :math:`\\|\\mathbf{a}\\| \\leq a_{max}` is the
    smallest :math:`t \\ge 0` satisfying:

    .. math::

        \\|\\mathbf{x} - \\mathbf{p} - \\mathbf{v} t\\| \\leq \\frac{1}{2} a_{max} t^2

    This is solved numerically via bisection on the feasibility function
    :math:`g(t) = \\|\\mathbf{x} - \\mathbf{p} - \\mathbf{v} t\\|
    - \\frac{1}{2} a_{max} t^2`, using a velocity-projection quadratic as lower bound
    and a stop-then-go strategy as upper bound for initial bracketing.

    The model by Fujimura and Sugihara [4]_ extends this by modeling player motion with
    an exponential decay of acceleration instead of constant acceleration. A player's
    velocity approaches a terminal speed :math:`v_{max}` with drag coefficient
    :math:`\\alpha`. The center of the reachable region by time :math:`t` is displaced
    by :math:`\\varphi(t) = (1 - e^{-\\alpha t}) / \\alpha` along the initial velocity
    direction, while the reachable radius grows as
    :math:`v_{max} (t - \\varphi(t))`. The arrival time is the smallest :math:`t \\ge 0`
    satisfying:

    .. math::

        \\|\\mathbf{x} - \\mathbf{p} - \\varphi(t) \\mathbf{v}\\|
        \\leq v_{max} (t - \\varphi(t))

    .. note::
        Computation time scales with the number of mesh points, players, and frames.
        Motion-based models (``'taki_hasegawa'``, ``'fujimura_sugihara'``) are
        significantly more expensive than the Euclidean model. For high-resolution
        meshes and long sequences, fitting may take several minutes.

    References
    ----------
        .. [1] `Taki, T., & Hasegawa, J. (2000). Visualization of dominant region in
            team games and its application to teamwork analysis. Proceedings Computer
            Graphics International 2000, 227–235.
            <https://ieeexplore.ieee.org/document/852338>`_
        .. [2] `Fonseca, S., Milho, J., Travassos, B., & Araújo, D. (2012). Spatial
            dynamics of team sports exposed by Voronoi diagrams. Human Movement
            Science, 31(6), 1652–1659. <https://doi.org/10.1016/j.humov.2012.04.006>`_
        .. [3] `Rein, R., Raabe, D., & Memmert, D. (2017). "Which pass is better?"
            Novel approaches to assess passing effectiveness in elite soccer. Human
            Movement Science, 55, 172–181.
            <https://doi.org/10.1016/j.humov.2017.07.010>`_
        .. [4] `Fujimura, A., & Sugihara, K. (2005). Geometric analysis and
            quantitative evaluation of sport teamwork. Systems and Computers in
            Japan, 36(6), 49–58. <https://doi.org/10.1002/scj.20254>`_
        .. [5] `Brefeld, U., Lasek, J., & Mair, S. (2019). Probabilistic movement
            models and zones of control. Machine Learning, 108(1), 127–147.
            <https://doi.org/10.1007/s10994-018-5725-1>`_

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

    def __init__(
        self,
        pitch: Pitch,
        mesh: str = "square",
        xpoints: int = 100,
        motion_model: str = "euclidean",
        max_acceleration: float = 4.2,
        vmax: float = 7.8,
        alpha: float = 1.3,
    ):
        super().__init__(pitch)

        # input parameter
        self._mesh_type = mesh
        self._xpoints = xpoints
        self._motion_model = motion_model
        self._max_acceleration = max_acceleration
        self._vmax = vmax
        self._alpha = alpha

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

        # velocity and speed for motion-based models, not checked by is_fitted
        self._vel1 = None
        self._vel2 = None
        self._speed1 = None
        self._speed2 = None
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
        valid_motion_models = ["euclidean", "taki_hasegawa", "fujimura_sugihara"]
        if motion_model not in valid_motion_models:
            raise ValueError(
                f"Invalid motion_model. Expected one of "
                f"{valid_motion_models}, got '{motion_model}'"
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

    def _compute_velocities(self, xy1: XY, xy2: XY):
        """Compute velocity vectors and speeds for both teams using VelocityModel."""
        vm = VelocityModel()

        vm.fit(xy1, axis="x")
        vx1 = vm.velocity().property
        vm.fit(xy1, axis="y")
        vy1 = vm.velocity().property
        vm.fit(xy1)
        self._vel1 = np.stack((vx1, vy1), axis=2)
        self._speed1 = vm.velocity().property

        vm.fit(xy2, axis="x")
        vx2 = vm.velocity().property
        vm.fit(xy2, axis="y")
        vy2 = vm.velocity().property
        vm.fit(xy2)
        self._vel2 = np.stack((vx2, vy2), axis=2)
        self._speed2 = vm.velocity().property

    @staticmethod
    def _bisect_roots(
        g_func,
        t_lo: np.ndarray,
        t_hi: np.ndarray,
        tol: float = 1e-2,
        max_iter: int = 10,
    ) -> np.ndarray:
        """Find roots of a feasibility function via bracketed bisection.

        Solves for the smallest ``t`` such that ``g_func(t) <= 0`` using
        bisection on the bracket ``[t_lo, t_hi]``. Resets ``t_lo`` to 0 where
        numerically already feasible. Callers must provide proven analytical
        bounds that guarantee ``t_lo`` is infeasible and ``t_hi`` is feasible.

        Parameters
        ----------
        g_func: callable
            Feasibility function mapping ``np.ndarray`` of shape (K,) to
            ``np.ndarray`` of shape (K,). Negative values indicate feasibility.
        t_lo: np.ndarray
            Array of shape (K,) with analytical lower bounds (infeasible).
        t_hi: np.ndarray
            Array of shape (K,) with analytical upper bounds (feasible).
        tol: float, optional
            Convergence tolerance in seconds. Defaults to 1e-2.
        max_iter: int, optional
            Maximum number of bisection iterations. Each iteration halves the
            bracket, so 10 iterations reduce it by a factor of 2^10 = 1024.
            Iteration stops early if all brackets are below ``tol``.
            Defaults to 10.

        Returns
        -------
        roots: np.ndarray
            Array of shape (K,) with the root estimates (upper bound of final
            bracket).
        """
        # reset lower bound to 0 where numerically already feasible
        g_lo = g_func(t_lo)
        t_lo[g_lo <= 0.0] = 0.0

        # fixed-iteration bisection with global early stop
        for _ in range(max_iter):
            t_mid = 0.5 * (t_lo + t_hi)
            g_mid = g_func(t_mid)
            feasible = g_mid <= 0.0
            t_hi[feasible] = t_mid[feasible]
            t_lo[~feasible] = t_mid[~feasible]
            if np.max(t_hi - t_lo) < tol:
                break

        return t_hi

    def _taki_hasegawa_arrival_time(
        self,
        mesh_points: np.ndarray,
        player_positions: np.ndarray,
        player_velocities: np.ndarray,
        player_speeds: np.ndarray,
    ) -> np.ndarray:
        """Compute arrival times using the Taki & Hasegawa (2000) kinematic model.

        Finds the minimum time for each player to reach each mesh point under the
        isotropic acceleration constraint ``||a|| <= max_acceleration``. Uses
        bisection root-finding on the feasibility function with analytical lower
        and upper bounds for tight initial bracketing. See class-level Notes for
        the mathematical formulation.

        Parameters
        ----------
        mesh_points: np.ndarray
            Array of shape (M, 2) with mesh point coordinates.
        player_positions: np.ndarray
            Array of shape (P, 2) with player coordinates (may contain NaN).
        player_velocities: np.ndarray
            Array of shape (P, 2) with player velocity vectors.
        player_speeds: np.ndarray
            Array of shape (P,) with player scalar speeds.

        Returns
        -------
        arrival_times: np.ndarray
            Array of shape (M, P) with arrival times. NaN where player position
            data is invalid.
        """
        M = mesh_points.shape[0]
        P = player_positions.shape[0]
        amax = self._max_acceleration
        arrival_times = np.full((P, M), np.nan, dtype=float)

        for p in range(P):
            pos = player_positions[p]
            vel = player_velocities[p]
            speed = float(player_speeds[p])

            # skip NaN players
            if np.any(np.isnan(pos)):
                continue

            # NaN velocity at valid position (gradient edge effect)
            if np.any(np.isnan(vel)):
                vel = np.zeros(2, dtype=float)
                speed = 0.0

            # displacement and distance from player to each mesh point
            disp = mesh_points - pos  # (M, 2)
            dist = np.linalg.norm(disp, axis=1)  # (M,)

            # filter out zero-distance points to avoid division by zero
            # in direction computation; assign them t=0
            solve = dist > 0.0
            arrival_times[p, ~solve] = 0.0

            r = disp[solve]  # (K, 2)
            d = dist[solve]  # (K,)

            # lower bound: velocity projection quadratic
            direction = r / d[:, np.newaxis]
            v_proj = np.sum(vel * direction, axis=1)
            t_lo = (-v_proj + np.sqrt(v_proj**2 + 2.0 * amax * d)) / amax

            # stationary player: both bounds are exact, skip bisection
            if speed == 0.0:
                arrival_times[p, solve] = t_lo
                continue

            # upper bound: stop-then-go strategy
            t_stop = speed / amax
            stop_pos = pos + vel * t_stop - 0.5 * (vel / speed) * amax * t_stop**2
            d_go = np.linalg.norm(mesh_points[solve] - stop_pos, axis=1)
            t_hi = t_stop + np.sqrt(2.0 * d_go / amax)

            # feasibility function: g(t) <= 0 means reachable by time t
            def _g(t_vec):
                rt = r - vel[np.newaxis, :] * t_vec[:, np.newaxis]
                return np.linalg.norm(rt, axis=1) - 0.5 * amax * t_vec**2

            arrival_times[p, solve] = DiscreteVoronoiModel._bisect_roots(
                _g,
                t_lo,
                t_hi,
            )

        return np.transpose(arrival_times)

    def _fujimura_sugihara_arrival_time(
        self,
        mesh_points: np.ndarray,
        player_positions: np.ndarray,
        player_velocities: np.ndarray,
        player_speeds: np.ndarray,
    ) -> np.ndarray:
        """Compute arrival times using the Fujimura & Sugihara (2005) model.

        Models player motion with an exponential velocity decay toward a terminal
        speed ``vmax`` with drag coefficient ``alpha``. Uses bisection
        root-finding on the feasibility function with analytical bounds. See
        class-level Notes for the mathematical formulation.

        Parameters
        ----------
        mesh_points: np.ndarray
            Array of shape (M, 2) with mesh point coordinates.
        player_positions: np.ndarray
            Array of shape (P, 2) with player coordinates (may contain NaN).
        player_velocities: np.ndarray
            Array of shape (P, 2) with player velocity vectors.
        player_speeds: np.ndarray
            Array of shape (P,) with player scalar speeds.

        Returns
        -------
        arrival_times: np.ndarray
            Array of shape (M, P) with arrival times. NaN where player position
            data is invalid.
        """
        M = mesh_points.shape[0]
        P = player_positions.shape[0]
        vmax = self._vmax
        alpha = self._alpha
        arrival_times = np.full((P, M), np.nan, dtype=float)

        for p in range(P):
            pos = player_positions[p]
            vel = player_velocities[p]
            speed = float(player_speeds[p])

            # skip NaN players
            if np.any(np.isnan(pos)):
                continue

            # NaN velocity at valid position (gradient edge effect)
            if np.any(np.isnan(vel)):
                vel = np.zeros(2, dtype=float)
                speed = 0.0

            # displacement and distance from player to each mesh point
            disp = mesh_points - pos  # (M, 2)
            dist = np.linalg.norm(disp, axis=1)  # (M,)

            # filter out zero-distance points; assign them t=0
            solve = dist > 0.0
            arrival_times[p, ~solve] = 0.0

            r = disp[solve]  # (K, 2)
            d = dist[solve]  # (K,)

            # lower bound: best case distance reduction
            t_lo = np.maximum(0.0, d - speed / alpha) / vmax

            # upper bound: conservative estimate
            t_hi = 1.0 / alpha + d / vmax + speed / (alpha * vmax)

            # phi(t) = (1 - exp(-alpha * t)) / alpha
            def _phi(t_vec):
                return (1.0 - np.exp(-alpha * t_vec)) / alpha

            # feasibility function: g(t) <= 0 means reachable by time t
            def _g(t_vec):
                ph = _phi(t_vec)
                rt = r - vel[np.newaxis, :] * ph[:, np.newaxis]
                return np.linalg.norm(rt, axis=1) - vmax * (t_vec - ph)

            arrival_times[p, solve] = DiscreteVoronoiModel._bisect_roots(
                _g,
                t_lo,
                t_hi,
            )

        return np.transpose(arrival_times)

    def _calc_cell_controls(self, xy1: XY, xy2: XY):
        """Calculate the xID of the controlling player for each mesh point at each
        frame.

        Dispatches to the appropriate arrival-time method based on
        ``self._motion_model`` and assigns each cell to the player with
        the lowest cost. All-NaN frames are skipped.
        """
        T = len(xy1)
        self._cell_controls_ = np.full(
            (T, self._meshx_.shape[0], self._meshx_.shape[1]),
            np.nan,
        )

        # flatten mesh to (M, 2) — computed once outside the loop
        mesh_points = np.stack((self._meshx_, self._meshy_), axis=2).reshape(-1, 2)

        # pre-stack player positions from both teams: (T, P, 2)
        all_positions = np.hstack((xy1.xy, xy2.xy)).reshape(T, -1, 2)

        # pre-stack velocities and speeds for motion-based models
        if self._motion_model in ["taki_hasegawa", "fujimura_sugihara"]:
            all_velocities = np.concatenate(
                (self._vel1, self._vel2), axis=1
            )  # (T, P, 2)
            all_speeds = np.concatenate((self._speed1, self._speed2), axis=1)  # (T, P)

        for t in range(T):
            player_positions = all_positions[t]

            # skip all-NaN frames (e.g., halftime, breaks)
            if np.all(np.isnan(player_positions)):
                continue

            # compute cost matrix based on motion model type
            if self._motion_model == "euclidean":
                costs = cdist(mesh_points, player_positions)
            elif self._motion_model == "taki_hasegawa":
                costs = self._taki_hasegawa_arrival_time(
                    mesh_points,
                    player_positions,
                    all_velocities[t],
                    all_speeds[t],
                )
            elif self._motion_model == "fujimura_sugihara":
                costs = self._fujimura_sugihara_arrival_time(
                    mesh_points,
                    player_positions,
                    all_velocities[t],
                    all_speeds[t],
                )

            # assign each mesh cell to the player with lowest cost
            self._cell_controls_[t] = np.nanargmin(costs, axis=1).reshape(
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
        # derive parameters
        self._N1_ = xy1.N
        self._N2_ = xy2.N
        self._T_ = len(xy1)
        self._framerate = xy1.framerate

        # compute velocities for motion-based models
        if self._motion_model in ["taki_hasegawa", "fujimura_sugihara"]:
            self._compute_velocities(xy1, xy2)

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
            of shape (n_frames x n_players), respectively. Property objects contain the
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
            of shape (n_frames x 1), respectively. Property objects contain the
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
            control_value = self._cell_controls_[t, i, j]
            if np.isnan(control_value):
                continue
            poly = plt.Rectangle(
                (self._meshx_[i, j] + xoffset, self._meshy_[i, j] + yoffset),
                width=self._xpolysize_,
                height=self._ypolysize_,
                fc=team_colors[int(control_value)],
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
            if np.isnan(control_value):
                continue
            poly = RegularPolygon(
                (x, self._meshy_[i, j]),
                numVertices=n_vertices,
                radius=self._xpolysize_,
                fc=team_colors[int(control_value)],
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

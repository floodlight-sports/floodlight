import numpy as np

from scipy.constants import g

from floodlight.utils.types import Numeric

from floodlight.core.xy import XY
from floodlight.core.property import PlayerProperty
from floodlight.models.base import BaseModel
from floodlight.models.kinematics import VelocityModel, AccelerationModel


class MetabolicPowerModel(BaseModel):
    """Class for calculating Metabolic Power of players on the pitch.

    Notes
    -----
    Metabolic Power is defined as the energy expenditure over time necessary to move at
    a certain speed, and is calculated as the product of energy cost of transport per
    unit body mass and distance [:math:`\\frac{J}{kg \\cdot m}`] and velocity
    [:math:`\\frac{m}{s}`]. Metabolic Power and Energy cost of walking is calculated
    according to [1]_. Energy cost of running is calculated with the updated formula of
    [2]_.

    Examples
    --------
    >>> import numpy as np
    >>> from floodlight import XY
    >>> from floodlight.models.kinetics import MetabolicPowerModel

    >>> xy = XY(np.array(((0, 0), (0, 1), (1, 1), (2, 2))), framerate=20)

    >>> metabolic_power_model = MetabolicPowerModel()
    >>> metabolic_power_model.fit(xy)

    >>> metabolic_power_model.metabolic_power()
    PlayerProperty(property=array([[1164.59773017],
           [ 185.59792131],
           [9448.10007077],
           [8593.05199423]]), name='metabolic_power', framerate=20)
    >>> metabolic_power_model.cumulative_equivalent_distance()
    PlayerProperty(property=array([[ 323.49936949],
       [ 375.05434763],
       [2999.52658952],
       [5386.4854768 ]]), name='cumulative_equivalent_distance', framerate=20)

    References
    ----------
        .. [1] `di Prampero P.E., Osgnach C. (2018). Metabolic power in team sports -
            Part 1: An update. International Journal of Sports Medicine, 39(08),
            581-587.
            <https://www.thieme-connect.de/products/ejournals/abstract/10.1055/
            a-0592-7660>`_
        .. [2] `Minetti, A.E., Parvei, G. (2018). Update and extension of the
            ‘Equivalent Slope’ of speed changing level locomotion in humans: A
            computational model for shuttle running. Journal Experimental Biology,
            221:jeb.182303.
            <https://journals.biologists.com/jeb/article/221/15/jeb182303/19414/Update-and-
            extension-of-the-equivalent-slope-of>`_
    """

    def __init__(self):
        super().__init__()
        self._metabolic_power_ = None

    @staticmethod
    def _calc_es(vel, acc):
        """Calculates equivalent slope based on the formula by di Prampero & Osgnach
        (2018)

        Parameters
        ----------
        vel: np.array
            velocity
        acc: np.array
            acceleration

        Returns
        -------
        es: np.array
            equivalent slope
        """
        # Coefficient of air resistance from di Prampero (2018).
        k = 0.0037
        es = (acc / g) + ((k * np.square(vel)) / g)
        return es

    @staticmethod
    def _calc_em(es):
        """Calculates equivalent mass based on the formula by di Prampero & Osgnach
        (2018)

        Parameters
        ----------
        es: np.array
            equivalent slope

        Returns
        -------
        em: np.array
            equivalent mass
        """

        em = np.sqrt(np.square(es) + 1)
        return em

    @staticmethod
    def _calc_v_trans(es: np.ndarray) -> np.ndarray:
        """Calculate the walking to running transition velocity at a certain equivalent
        slope based on the formular of di Prampero (2018).

        Parameters
        ----------
        es: np.array
            equivalent slope

        Returns
        -------
        v_trans: np.array
            Array with the respective transition velocity

        """
        # Coefficients of polynomial to calculate the walk-run-transition
        # velocity based on the equivalent slope from di Prampero (2018).
        coeff = np.array((-107.05, 113.13, -1.13, -15.84, -1.7, 2.27))
        es_power = np.stack(
            (
                np.power(es, 5),
                np.power(es, 4),
                np.power(es, 3),
                np.power(es, 2),
                es,
                np.ones(es.shape),
            ),
            axis=-1,
        )

        v_trans = np.matmul(es_power, coeff)

        return v_trans

    @staticmethod
    def _is_running(vel: np.ndarray, es: np.ndarray) -> np.ndarray:
        """
        Checks if athlete is walking or running based on the model of di Prampero
        (2018).
        Parameters
        ----------
        vel: np.array
            Velocity
        es: np.array
            Equivalent slope
        Returns
        -------
        is_running: bool
            True: Athlete is running
            False: Athlete is walking
        """
        # Calculate walk-run-transition velocity
        v_trans = MetabolicPowerModel._calc_v_trans(es)

        is_running = (vel >= v_trans) | (vel > 2.5)

        return is_running

    @staticmethod
    def _get_interpolation_weight_matrix(es: np.ndarray) -> np.ndarray:
        """Calculates interpolation weight matrix.

        Parameters
        ----------
        es: np.array
            Equivalent slope

        Returns
        -------
        W: np.array
            Interpolation weight matrix with 3 dimensions (T frames, N players,
            len(EDGES)=8 polynomials. Values range between 0 and 1 and indicate
            how much the polynomial of energy cost of walking is weighted for that
            time and player
        """
        # Edges of equivalent slope for using the corresponding polynomial to calculate
        # energy cost of walking at a certain velocity from di Prampero (2018).
        EDGES = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4])
        # Number of frames
        T = es.shape[0]
        # Number of players
        N = es.shape[1]
        # Pre-allocated interpolation weight matrix with 3 dimensions (T frames, N
        # players, len(EDGES)=8 polynomials)
        W = np.zeros((T, N, len(EDGES)))

        # Index of each ES regarding its position in EDGES.
        # E.g. ES = 0.25 -> ES will be sorted between EDGES[5] and EDGES[6],
        # idxs = 6
        idxs = EDGES.searchsorted(es)

        # mask for non-edge cases (ES outside of EDGES)
        mask = (idxs > 0) & (idxs < 8)

        # Initialize grids for appropriate indexing of W along axis=0 (time) and
        # axis=1 (player)
        grid_t, grid_n = np.mgrid[0:T, 0:N]

        # Fill W with the right interpolation weights for each time t (axis=0),
        # player n (axis=1) and polynomial (axis=2)
        W[grid_t[mask], grid_n[mask], idxs[mask] - 1] = (
            EDGES[idxs[mask]] - es[mask]
        ) * 10
        W[grid_t[mask], grid_n[mask], idxs[mask]] = (
            es[mask] - EDGES[idxs[mask] - 1]
        ) * 10

        # Fill edge cases (ES outside of EDGES) with 1 because they are not
        # interpolated
        W[idxs == 0, 0] = 1
        W[idxs == 8, 7] = 1

        return W

    @staticmethod
    def _calc_ecw(es: np.ndarray, vel: np.ndarray, em: np.ndarray) -> np.ndarray:
        """Calculates energy cost of walking based on formula (13), (14) and table
        1 in di Prampero & Osgnach (2018).

        Parameters
        ----------
        es: np.array
            Equivalent slope
        vel: np.array
            Velocity
        em: np.array
            Equivalent mass

        Returns
        -------
        ECW: np.array
            Energy cost of walking

        """
        # Coefficients of polynomials to calculate energy cost of walking from di
        # Prampero (2018).
        COEFF = np.array(
            [
                [0.28, -1.66, 3.81, -3.96, 4.01],
                [0.03, -0.15, 0.98, -2.25, 3.14],
                [0.69, -3.21, 5.94, -5.07, 2.79],
                [1.25, -6.57, 13.14, -11.15, 5.35],
                [0.68, -4.17, 10.17, -10.31, 8.66],
                [3.80, -14.91, 22.94, -14.53, 11.24],
                [44.95, -122.88, 126.94, -57.46, 21.39],
                [94.62, -213.94, 184.43, -68.49, 25.04],
            ]
        )
        # Interpolation weight matrix
        W = MetabolicPowerModel._get_interpolation_weight_matrix(es)

        # Matrix product of EDGES and W, ie. weighted factors in polynomials
        WC = np.matmul(W, COEFF)

        # Calcualte vel^4 + vel^3 + vel^2 + vel + 1 for every frame and player
        V = np.stack(
            (
                np.power(vel, 4),
                np.power(vel, 3),
                np.power(vel, 2),
                vel,
                np.ones(vel.shape),
            ),
            axis=-1,
        )

        # Multiply WC and V. Calculate sum of terms. Multiply with em
        ECW = np.multiply(np.multiply(WC, V).sum(axis=2), em)

        return ECW

    @staticmethod
    def _calc_ecr(es: np.ndarray, em: np.ndarray, eccr: Numeric = 3.6) -> np.ndarray:
        """Calculates Energy cost of running based on formula (3) and (4) from
        Minetti & Parvei (2018).

        Parameters
        ----------
        es: np.array
            Equivalent slope
        em: np.array
            Equivalent mass
        eccr: Numeric
            Energy cost of constant running. Standard is set to 3.6
            :math:`\\frac{J}{kg \\cdot m}` according to di Prampero (2018). Can differ
            for different turfs.

        Returns
        -------
        ecr: np.array
            Energy cost of running
        """
        # Cost of negative gradient from Minetti (2018)
        def _cng(es: np.ndarray):
            return -8.34 * es + eccr * np.exp(13 * es)

        # Cost of positive gradient
        def _cpg(es: np.ndarray):
            return 39.5 * es + eccr * np.exp(-4 * es)

        # Energy cost of running. Where es < 0 calculate cost of negative gradient.
        # Where es >= 0 calculate cost of positive gradient.
        ecr = np.piecewise(es, [es < 0, es >= 0], [_cng, _cpg]) * em

        return ecr

    @staticmethod
    def _calc_ecl(
        es: np.ndarray, vel: np.ndarray, em: np.ndarray, eccr: Numeric = 3.6
    ) -> np.ndarray:
        """Calculate Energy cost of locomotion.

        Parameters
        ----------
        es: np.array
            Equivalent slope
        vel: np.array
            Velocity
        em: np.array
            Equivalent mass
        eccr: Numeric
            Energy cost of constant running. Standard is set to 3.6
            :math:`\\frac{J}{kg \\cdot m}` according to di Prampero (2018). Can differ
            for different turfs.

        Returns
        -------
        ecl: np.array
            Energy cost of locomotion
        """
        # Check where locomotion is running
        running = MetabolicPowerModel._is_running(vel, es)
        # Calculate energy cost of walking for entire array
        ecl = MetabolicPowerModel._calc_ecw(es, vel, em)
        # Substitute ecw with energy cost of running where locomotion is running
        ecl[running] = MetabolicPowerModel._calc_ecr(es[running], em[running], eccr)

        return ecl

    @staticmethod
    def _calc_metabolic_power(
        es: np.ndarray,
        vel: np.ndarray,
        em: np.ndarray,
        framerate: Numeric,
        eccr: Numeric = 3.6,
    ) -> np.ndarray:
        """Calculates metabolic power as the product of energy cost of locomotion
        and velocity.

        Parameters
        ----------
        es: np.array
            Equivalent slope
        vel: np.array
            Velocity
        em: np.array
            Equivalent mass

        Returns
        -------
        metp: np.array
            Metabolic power
        """
        # Calculate energy cost of locomotion
        ecl = MetabolicPowerModel._calc_ecl(es, vel, em, eccr)
        # Calculate metabolic power as product of ecl and velocity (m/s)
        metp = ecl * vel / framerate

        return metp

    def fit(
        self, xy: XY, difference: str = "central", axis: str = None, eccr: Numeric = 3.6
    ):
        """Fit the model to the given data and calculate metabolic power for every
        player.

        Notes
        -----
        To give appropriate results, unit of coordinates must be in meter.

        Parameters
        ----------
        xy: XY
            Floodlight XY Data object.
        difference: str
            The method of differentiation. One of {'central', 'forward'}.\n
            'central' will differentiate using the central difference method:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{1}-y_{-1}}{t_{1} - t_{-1}}

            'forward' will differentiate using the forward difference method and append
            a '0' at the end of the array along axis 1:

                .. math::

                    y^{\\prime}(t_{0}) = \\frac{y_{1}-y_{0}}{t_{1} - t_{0}}

        axis: {None, 'x', 'y'}, optional
                Optional argument that restricts distance calculation to either the x-
                or y-dimension of the data. If set to None (default), distances are
                calculated in both dimensions.
        eccr: Numeric
            Energy cost of constant running. Standard is set to 3.6
            :math:`\\frac{J}{kg \\cdot m}` according to di Prampero (2018). Can differ
            for different turfs.
        """

        # Velocity
        velocity_model = VelocityModel()
        velocity_model.fit(xy, difference=difference, axis=axis)
        velocity = velocity_model.velocity()

        # Acceleration
        acceleration_model = AccelerationModel()
        acceleration_model.fit(xy, difference=difference, axis=axis)
        acceleration = acceleration_model.acceleration()

        # Equivalent slope
        equivalent_slope = MetabolicPowerModel._calc_es(
            velocity.property, acceleration.property
        )
        # Equivalent mass
        equivalent_mass = MetabolicPowerModel._calc_em(equivalent_slope)

        # Metabolic power
        metabolic_power = MetabolicPowerModel._calc_metabolic_power(
            equivalent_slope, velocity.property, equivalent_mass, xy.framerate, eccr
        )

        self._metabolic_power_ = PlayerProperty(
            property=metabolic_power,
            name="metabolic_power",
            framerate=xy.framerate,
        )

    def metabolic_power(self) -> PlayerProperty:
        """Returns the frame-wise metabolic power as computed by the fit method.

        Returns
        -------
        metabolic_power: PlayerProperty
            An Player Property object of shape (T, N), where T is the total number of
            frames and N is the number of players. The columns contain the frame-wise
            metabolic power.
        """
        metabolic_power = self._metabolic_power_
        return metabolic_power

    def cumulative_metabolic_power(self) -> PlayerProperty:
        """Returns the cumulative metabolic power.

        Returns
        -------
        metabolic_power: PlayerProperty
            An Player Property object of shape (T, N), where T is the total number of
            frames and N is the number of players. The columns contain the frame-wise
            metabolic power calculated by numpy.nancumsum() over axis=0.
        """
        cum_metp = np.nancumsum(self._metabolic_power_.property, axis=0)
        cumulative_metabolic_power = PlayerProperty(
            property=cum_metp,
            name="cumulative_metabolic_power",
            framerate=self._metabolic_power_.framerate,
        )
        return cumulative_metabolic_power

    def equivalent_distance(self, eccr: Numeric = 3.6) -> PlayerProperty:
        """Returns frame-wise equivalent distance, defined as the distance a player
        could have run if moving at a constant speed and calculated as the fraction of
        metabolic work and the cost of constant running.

        Parameters
        ----------
        eccr: Numeric
            Energy cost of constant running. Standard is set to 3.6
            :math:`\\frac{J}{kg \\cdot m}` according to di Prampero (2018). Can differ
            for different turfs.

        Returns
        -------
        equivalent_distance: PlayerProperty
            PlayerProperty of the instantaneous equivalent distance covered.
        """

        eq_dist = self._metabolic_power_.property / eccr
        cumulative_metabolic_power = PlayerProperty(
            property=eq_dist,
            name="equivalent_distance",
            framerate=self._metabolic_power_.framerate,
        )
        return cumulative_metabolic_power

    def cumulative_equivalent_distance(self, eccr: int = 3.6) -> PlayerProperty:
        """Returns cumulative equivalent distance defined as the distance a player
        could have run if moving at a constant speed and calculated as the fraction
        of metabolic work and the cost of constant running.

        Parameters
        ----------
        eccr: int
            Energy cost of constant running. Standard is set to 3.6
            :math:`\\frac{J}{kg \\cdot m}` according to di Prampero (2018). Can differ
            for different turfs.

        Returns
        -------
        cumulative_equivalent_distance: PlayerProperty
            PlayerProperty of the cumulative equivalent distance covered calculated by
            numpy.nancumsum() over axis=0.
        """

        cum_metp = np.nancumsum(self._metabolic_power_.property, axis=0)
        cum_eqdist = cum_metp / eccr

        cumulative_equivalent_distance = PlayerProperty(
            property=cum_eqdist,
            name="cumulative_equivalent_distance",
            framerate=self._metabolic_power_.framerate,
        )
        return cumulative_equivalent_distance

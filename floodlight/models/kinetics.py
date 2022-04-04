import numpy as np

from scipy.constants import g

from floodlight.core.xy import XY
from floodlight.core.pitch import Pitch
from floodlight.core.property import PlayerProperty
from floodlight.models.kinematics import VelocityModel, AccelerationModel


class MetabolicPowerModel:
    """Class for calculating Metabolic Power of players on the pitch. Metabolic Power
    is defined as the energy expenditure over time necessary to move at a certain speed,
    and is calculated as the product of energy cost of transport per unit body mass and
    distance [:math:`\\frac{J}{kg \\cdot m}`] and velocity [:math:`\\frac{m}{s}`].
    Metabolic Power and Energy cost of walking is calculated according to di Prampero &
    Osgnach (2018). Energy cost of running is calculated with the updated formula of
    Minetti & Parvei (2018).

    Parameters
    __________
    pitch: Pitch
        Pitch object of the data. Should at least include information about the data's
        unit. If pitch.unit is 'percent', length and width of the pitch are also needed.

    References
    __________
    di Prampero PE, Osgnach C. Metabolic power in team sports - Part 1: An update. Int J
    Sports Med. 2018;39(08):581-587. doi:10.1055/a-0592-7660
    Minetti, AE, Parvei, G. Update and extension of the ‘Equivalent Slope’ of speed
    changing level locomotion in humans: A computational model for shuttle running. J
    Exp Biol. 2018;221:jeb.182303. doi: 10.1242/jeb.182303
    """

    def __init__(self, pitch: Pitch):
        self.pitch = pitch
        self._framerate = None
        self._metabolic_power = None
        # Edges of equivalent slope for using the corresponding polynomial to calculate
        # energy cost of walking at a certain velocity from di Prampero (2018).
        self.EDGES = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4])
        # Coefficients of polynomials to calculate energy cost of walking from di
        # Prampero (2018).
        self.COEFF = np.array(
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
        # Coefficient of air resistance from di Prampero (2018).
        self.k = 0.0037

    def fit(
        self,
        xy: XY,
        difference: str = "central",
        axis: str = None,
    ):
        """Fits a model to calculate metabolic power from a XY-object.

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

        axis: str
            One of {'x', 'y', None}.
            Indicates the direction to calculate the distance in.
            'x' will only calculate distance in the direction of the x-coordinate.
            'y' will only calculate distance in the direction of the y-coordinate.
            None will calculate distance in both directions.
        """

        self._framerate = xy.framerate

        def _calc_v_trans(es: np.ndarray[float]):
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

        def _is_running(vel: np.ndarray[float], es: np.ndarray[float]):
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
            v_trans = _calc_v_trans(es)

            is_running = (vel >= v_trans) | (vel > 2.5)

            return is_running

        def _get_interpolation_weight_matrix(es: np.ndarray[float]):
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
            # Number of frames
            T = es.shape[0]
            # Number of players
            N = es.shape[1]
            # Pre-allocated interpolation weight matrix with 3 dimensions (T frames, N
            # players, len(EDGES)=8 polynomials)
            W = np.zeros((T, N, len(self.EDGES)))

            # Index of each ES regarding its position in EDGES.
            # E.g. ES = 0.25 -> ES will be sorted between EDGES[5] and EDGES[6],
            # idxs = 6
            idxs = self.EDGES.searchsorted(es)

            # mask for non-edge cases (ES outside of EDGES)
            mask = (idxs > 0) & (idxs < 8)

            # Initialize grids for appropriate indexing of W along axis=0 (time) and
            # axis=1 (player)
            grid_t, grid_n = np.mgrid[0:T, 0:N]

            # Fill W with the right interpolation weights for each time t (axis=0),
            # player n (axis=1) and polynomial (axis=2)
            W[grid_t[mask], grid_n[mask], idxs[mask] - 1] = (
                self.EDGES[idxs[mask]] - es[mask]
            ) * 10
            W[grid_t[mask], grid_n[mask], idxs[mask]] = (
                es[mask] - self.EDGES[idxs[mask] - 1]
            ) * 10

            # Fill edge cases (ES outside of EDGES) with 1 because they are not
            # interpolated
            W[idxs == 0, 0] = 1
            W[idxs == 8, 7] = 1

            return W

        def calc_ecw(
            es: np.ndarray[float], vel: np.ndarray[float], em: np.ndarray[float]
        ):
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

            References
            ----------
            di Prampero PE, Osgnach C. Metabolic power in team sports - Part 1: An
            update. Int J Sports Med. 2018;39(08):581-587. doi:10.1055/a-0592-7660
            """
            # Interpolation weight matrix
            W = _get_interpolation_weight_matrix(es)

            # Matrix product of EDGES and W, ie. weighted factors in polynomials
            WC = np.matmul(W, self.COEFF)

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

        def calc_ecr(es: np.ndarray[float], em: np.ndarray[float]):
            """Calculates Energy cost of running based on formula (3) and (4) from
            Minetti & Parvei (2018).

            Parameters
            ----------
            es: np.array
                Equivalent slope
            em: np.array
                Equivalent mass

            Returns
            -------
            ecr: np.array
                Energy cost of running
            References
            ----------
            Minetti, AE, Parvei, G. Update and extension of the ‘Equivalent Slope’ of
            speed changing level locomotion in humans: A computational model for shuttle
            running. J Exp Biol. 2018;221:jeb.182303. doi: 10.1242/jeb.182303
            """
            # Cost of negative gradient from Minetti (2018)
            def _cng(es: np.ndarray[float]):
                return -8.34 * es + 3.6 * np.exp(13 * es)

            # Cost of positive gradient
            def _cpg(es: np.ndarray[float]):
                return 39.5 * es + 3.6 * np.exp(-4 * es)

            # Energy cost of running. Where es < 0 calculate cost of negative gradient.
            # Where es >= 0 calculate cost of positive gradient.
            ecr = np.piecewise(es, [es < 0, es >= 0], [_cng, _cpg]) * em

            return ecr

        def calc_ecl(
            es: np.ndarray[float], vel: np.ndarray[float], em: np.ndarray[float]
        ):
            """Calculate Energy cost of locomotion.

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
            ecl: np.array
                Energy cost of locomotion
            """
            # Check where locomotion is running
            running = _is_running(vel, es)
            # Calculate energy cost of walking for entire array
            ecl = calc_ecw(es, vel, em)
            # Substitute ecw with energy cost of running where locomotion is running
            ecl[running] = calc_ecr(es[running], em[running])

            return ecl

        def calc_metabolic_power(
            es: np.ndarray[float], vel: np.ndarray[float], em: np.ndarray[float]
        ):
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
            ecl = calc_ecl(es, vel, em)
            # Calculate metabolic power as product of ecl and velocity (m/s)
            metp = ecl * vel / self._framerate

            return metp

        # Velocity
        velocity_model = VelocityModel(self.pitch)
        velocity_model.fit(xy, difference=difference, axis=axis)
        velocity = velocity_model.velocity()

        # Acceleration
        acceleration_model = AccelerationModel(self.pitch)
        acceleration_model.fit(xy, difference=difference, axis=axis)
        acceleration = acceleration_model.acceleration()

        # Equivalent slope
        equivalent_slope = (acceleration.property / g) + (
            (self.k * np.square(velocity.property)) / g
        )

        # Equivalent mass
        equivalent_mass = np.sqrt(np.square(equivalent_slope) + 1)

        # Metabolic power
        metabolic_power = calc_metabolic_power(
            equivalent_slope, velocity.property, equivalent_mass
        )

        self._metabolic_power = metabolic_power

    def metabolic_power(self):
        metabolic_power = PlayerProperty(
            property=self._metabolic_power,
            name="metabolic_power",
            framerate=self._framerate,
        )
        return metabolic_power

    def cumulative_metabolic_power(self):
        cum_metp = np.nancumsum(self._metabolic_power, axis=0)
        cumulative_metabolic_power = PlayerProperty(
            property=cum_metp,
            name="cumulative_metabolic_power",
            framerate=self._framerate,
        )
        return cumulative_metabolic_power

    def equivalent_distance(self, eccr: int = 3.6):
        """Instantaneous equivalent distance, defined as the distance a player could
        have run if moving at a constant speed and calculated as the fraction of
        metabolic work and the cost of constant running.

        Parameters
        ----------
        eccr: int
            Energy cost of constant running. Standard is set to 3.6
            :math:`\\frac{J}{kg \\cdot m}` according to di Prampero (2018). Can differ
            for different turfs.

        Returns
        -------
        equivalent_distance: PlayerProperty
            PlayerProperty of the instantaneous equivalent distance covered.
        """

        eq_dist = self._metabolic_power / eccr
        cumulative_metabolic_power = PlayerProperty(
            property=eq_dist, name="equivalent_distance", framerate=self._framerate
        )
        return cumulative_metabolic_power

    def cumulative_equivalent_distance(self, eccr: int = 3.6):
        """
        Cumulative equivalent distance defined as the distance a player could have run
        if moving at a constant speed and calculated as the fraction of metabolic work
        and the cost of constant running.

        Parameters
        ----------
        eccr: int
            Energy cost of constant running. Standard is set to 3.6
            :math:`\\frac{J}{kg \\cdot m}` according to di Prampero (2018). Can differ
            for different turfs.

        Returns
        -------
        cumulative_equivalent_distance: PlayerProperty
            PlayerProperty of the cumulative equivalent distance covered.
        """

        cum_metp = np.nancumsum(self._metabolic_power, axis=0)
        cum_eqdist = cum_metp / eccr

        cumulative_equivalent_distance = PlayerProperty(
            property=cum_eqdist,
            name="cumulative_equivalent_distance",
            framerate=self._framerate,
        )
        return cumulative_equivalent_distance

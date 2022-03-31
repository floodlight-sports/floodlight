import numpy as np

from scipy.constants import g

from floodlight.core.xy import XY
from floodlight.core.property import PlayerProperty
from floodlight.models.kinematics import VelocityModel, AccelerationModel


class MetabolicPowerModel:
    """Class for calculating Metabolic Power of players on the pitch. Metabolic Power
    is defined as the energy expenditure over time necessary to move at a certain speed,
    and is calculated as the product of energy cost of transport per unit body mass and
    distance [:math:`\\frac{J}{kg \\cdot m}`] and velocity [:math:`\\frac{m}{s}`].


    Parameters
    __________
    pitch: Pitch
        Pitch object of the data. Should at least include information about the data's
        unit. If pitch.unit is 'percent', length and width of the pitch are also needed.

    References
    __________
    di Prampero PE, Osgnach C. Metabolic power in team sports - Part 1: An update. Int J
    Sports Med. 2018;39(08):581-587. doi:10.1055/a-0592-7660

    """

    def __init__(self, pitch):
        self.pitch = pitch
        self._framerate = None
        self._metabolic_power = None
        self.EDGES = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4])
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

        def _calc_v_trans(es):
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

        def _is_running(vel, es):
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

            v_trans = _calc_v_trans(es)

            is_running = (vel >= v_trans) | (vel > 2.5)

            return is_running

        def _get_interpolation_weight_matrix(es):
            T = es.shape[0]
            N = es.shape[1]
            W = np.zeros((T, N, len(self.EDGES)))
            # dim = (T, N, zones)

            idxs = self.EDGES.searchsorted(es)

            # mask for non-edge casees
            mask = (idxs > 0) & (idxs < 8)

            grid_t, grid_n = np.mgrid[0:T, 0:N]

            W[grid_t[mask], grid_n[mask], idxs[mask] - 1] = (
                self.EDGES[idxs[mask]] - es[mask]
            ) * 10
            W[grid_t[mask], grid_n[mask], idxs[mask]] = (
                es[mask] - self.EDGES[idxs[mask] - 1]
            ) * 10

            # edge cases
            W[idxs == 0, 0] = 1
            W[idxs == 8, 7] = 1

            # maybe not needed:
            # W = W.round(3)

            return W

        def calc_ecw(es, vel, em):
            W = _get_interpolation_weight_matrix(es)

            WC = np.matmul(W, self.COEFF)
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

            ECW = np.multiply(np.multiply(WC, V).sum(axis=2), em)

            return ECW

        def calc_ecr(es, em):
            def _cng(es):
                return -8.34 * es + 3.6 * np.exp(13 * es)

            def _cpg(es):
                return 39.5 * es + 3.6 * np.exp(-4 * es)

            ecr = np.piecewise(es, [es <= 0, es > 0], [_cng, _cpg]) * em

            return ecr

        def calc_ecl(es, vel, em):
            running = _is_running(vel, es)

            ecl = calc_ecw(es, vel, em)
            ecl[running] = calc_ecr(es[running], em[running])

            return ecl

        def calc_metabolic_power(es, vel, em):
            ecl = calc_ecl(es, vel, em)
            metabolic_power = ecl * vel / 20

            return metabolic_power

        self._framerate = xy.framerate

        velocity_model = VelocityModel(self.pitch)
        velocity_model.fit(xy, difference=difference, axis=axis)
        velocity = velocity_model.velocity()

        acceleration_model = AccelerationModel(self.pitch)
        acceleration_model.fit(xy, difference=difference, axis=axis)
        acceleration = acceleration_model.acceleration()

        equivalent_slope = (acceleration.property / g) + (
            (self.k * np.square(velocity.property)) / g
        )

        equivalent_mass = np.sqrt(np.square(equivalent_slope) + 1)

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

import numpy as np

from scipy.constants import g

from floodlight.utils.types import Numeric
from floodlight.core.xy import XY
from floodlight.core.pitch import Pitch
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

    pitch: Pitch = None
    length: Numeric = None
    width: Numeric = None
    unit: str = None

    def __init__(self, pitch):
        self.unit = pitch.unit
        self.length = pitch.length
        self.width = pitch.width
        self.pitch = pitch
        self._metabolic_power = None
        self.framerate = None

    def __str__(self):
        return (
            f"Floodlight MetabolicPowerModel Class with unit [{self.unit}],"
            f"length {self.length} and width {self.width}"
        )

    def fit(
        self,
        xy: XY,
        difference: str = "central",
        direction: str = "plane",
    ):
        """Fits a model to calculate euclidean distances to an xy object.

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

        direction: str
            One of {'x', 'y', 'plane'}.
            Indicates the direction to calculate the distance in.
            'x' will only calculate distance in the direction of the x-coordinate.
            'y' will only calculate distance in the direction of the y-coordinate.
            'plane' or None will calculate distance in both directions.
        """

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
            is_running = (
                vel
                > -107.05 * np.power(es, 5)
                + 113.13 * np.power(es, 4)
                - 1.13 * np.power(es, 3)
                - 15.84 * np.power(es, 2)
                - 1.7 * es
                + 2.27
            ) | (vel > 2.5)

            return is_running

        def _ecr_minetti(acc):
            """
            Calculates the energy cost of running based on the model of Minetti (2018).
            Parameters
            ----------
            acc: np.array
                acceleration

            Returns
            -------
            ecr: np.array
                Energy cost of running
            """
            ecr = np.where(
                acc > 0,
                0.102
                * np.sqrt(np.square(acc) + 96.2)
                * (4.03 * acc + 3.6 * np.exp(-0.408 * acc)),
                0.102
                * np.sqrt(np.square(acc) + 96.2)
                * (-0.85 * acc + 3.6 * np.exp(1.33 * acc)),
            )

            return ecr

        def _ecw_n03(vel):
            """
            Calculates energy cost of walking at a equivalent slope of -0.3 based on the
            model of di Prampero (2018).

            Parameters
            ----------
            vel: np.array
                velocity

            Returns
            -------
            ecw: np.array
                Energy cost of walking
            """
            ecw = (
                0.28 * np.power(vel, 4)
                - 1.66 * np.power(vel, 3)
                + 3.81 * np.power(vel, 2)
                - 3.96 * vel
                + 4.01
            )

            return ecw

        def _ecw_n02(vel):
            """
            Calculates energy cost of walking at a equivalent slope of -0.2 based on the
            model of di Prampero (2018)

            Parameters
            ----------
            vel: np.array
                velocity

            Returns
            -------
            ecw: np.array
                Energy cost of walking
            """
            ecw = (
                0.03 * np.power(vel, 4)
                - 0.15 * np.power(vel, 3)
                + 0.98 * np.power(vel, 2)
                - 2.25 * vel
                + 3.14
            )

            return ecw

        def _ecw_n01(vel):
            """
            Calculates energy cost of walking at a equivalent slope of -0.1 based on the
            model of di Prampero (2018).

            Parameters
            ----------
            vel: np.array
                velocity

            Returns
            -------
            ecw: np.array
                Energy cost of walking
            """
            ecw = (
                0.69 * np.power(vel, 4)
                - 3.21 * np.power(vel, 3)
                + 5.94 * np.power(vel, 2)
                - 5.07 * vel
                + 2.79
            )

            return ecw

        def _ecw_00(vel):
            """
            Calculates energy cost of walking at a equivalent slope of 0 based on the
            model of di Prampero (2018).

            Parameters
            ----------
            vel: np.array
                velocity

            Returns
            -------
            ecw: np.array
                Energy cost of walking
            """
            ecw = (
                1.25 * np.power(vel, 4)
                - 6.57 * np.power(vel, 3)
                + 13.14 * np.power(vel, 2)
                - 11.15 * vel
                + 5.35
            )

            return ecw

        def _ecw_01(vel):
            """
            Calculates energy cost of walking at a equivalent slope of 0.1 based on the
            model of di Prampero (2018).

            Parameters
            ----------
            vel: np.array
                velocity

            Returns
            -------
            ecw: np.array
                Energy cost of walking
            """
            ecw = (
                0.68 * np.power(vel, 4)
                - 4.17 * np.power(vel, 3)
                + 10.17 * np.power(vel, 2)
                - 10.31 * vel
                + 8.66
            )

            return ecw

        def _ecw_02(vel):
            """
            Calculates energy cost of walking at a equivalent slope of 0.2 based on the
            model of di Prampero (2018).

            Parameters
            ----------
            vel: np.array
                velocity

            Returns
            -------
            ecw: np.array
                Energy cost of walking
            """
            ecw = (
                3.80 * np.power(vel, 4)
                - 14.91 * np.power(vel, 3)
                + 22.94 * np.power(vel, 2)
                - 14.53 * vel
                + 11.24
            )

            return ecw

        def _ecw_03(vel):
            """
            Calculates energy cost of walking at a equivalent slope of 0.3 based on the
            model of di Prampero (2018).

            Parameters
            ----------
            vel: np.array
                velocity

            Returns
            -------
            ecw: np.array
                Energy cost of walking
            """
            ecw = (
                44.95 * np.power(vel, 4)
                - 122.88 * np.power(vel, 3)
                + 126.94 * np.power(vel, 2)
                - 57.46 * vel
                + 21.39
            )

            return ecw

        def _ecw_04(vel):
            """
            Calculates energy cost of walking at a equivalent slope of 0.4 based on the
            model of di Prampero (2018).

            Parameters
            ----------
            vel: np.array
                velocity

            Returns
            -------
            ecw: np.array
                Energy cost of walking
            """
            ecw = (
                94.62 * np.power(vel, 4)
                - 213.94 * np.power(vel, 3)
                + 184.43 * np.power(vel, 2)
                - 68.49 * vel
                + 25.04
            )

            return ecw

        def _ecw_interpol(ecw_lower, ecw_higher, es_lower, es_higher, es, em):
            """
            Interpolates the Energy cost of walking between the higher and lower ecw
            formula.

            Parameters
            ----------
            ecw_lower: np.array
                Energy cost of running calculated based on the lower equivalent slope.
            ecw_higher: np.array
                Energy cost of running calculated based on the higher equivalent slope.
            es_lower: float
                Lower equivalent slope
            es_higher: float
                higher equivalent slope
            es: np.array
                equivalent slope
            em: np.array
                equivalent mass

            Returns
            -------
            ecq: np.array
                Interpolated energy cost of walking
            """

            ecw_interpolated = (
                ecw_lower
                + (ecw_higher - ecw_lower)
                * ((es - es_lower) / (es_higher - es_lower))
                * em
            )

            return ecw_interpolated

        def _ecr_mixed(vel, acc, es, em):
            """
            Calculates the energy cost of running based on the model of di Prampero
            (2018) with the formula of Minetti (2018) for running.

            Parameters
            ----------
            vel: np.array
                velocity
            acc: np.array
                acceleration
            es: np.array
                equivalent slope
            em: np.array
            equivalent mass

            Returns
            -------
            ecr: np.array
                Energy cost of running
            """

            ecr = np.where(
                _is_running(vel, es),
                # calculate ecr with Minetti
                _ecr_minetti(acc),
                # when walking differentiate between different es
                np.where(
                    es < -0.3,
                    _ecw_n03(vel) * em,
                    np.where(
                        (es >= -0.3) & (es < -0.2),
                        _ecw_interpol(_ecw_n03(vel), _ecw_n02(vel), -0.3, -0.2, es, em),
                        np.where(
                            (es >= -0.2) & (es < -0.1),
                            _ecw_interpol(
                                _ecw_n02(vel), _ecw_n01(vel), -0.2, -0.1, es, em
                            ),
                            np.where(
                                (es >= -0.1) & (es < 0),
                                _ecw_interpol(
                                    _ecw_n01(vel), _ecw_00(vel), -0.1, 0, es, em
                                ),
                                np.where(
                                    (es >= 0) & (es < 0.1),
                                    _ecw_interpol(
                                        _ecw_00(vel), _ecw_01(vel), 0, 0.1, es, em
                                    ),
                                    np.where(
                                        (es >= 0.1) & (es < 0.2),
                                        _ecw_interpol(
                                            _ecw_01(vel), _ecw_02(vel), 0.1, 0.2, es, em
                                        ),
                                        np.where(
                                            (es >= 0.2) & (es < 0.3),
                                            _ecw_interpol(
                                                _ecw_02(vel),
                                                _ecw_03(vel),
                                                0.2,
                                                0.3,
                                                es,
                                                em,
                                            ),
                                            np.where(
                                                (es >= 0.3) & (es < 0.4),
                                                _ecw_interpol(
                                                    _ecw_03(vel),
                                                    _ecw_04(vel),
                                                    0,
                                                    0.1,
                                                    es,
                                                    em,
                                                ),
                                                _ecw_04(vel) * em,
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )

            return ecr

        velocity_model = VelocityModel(self.pitch)
        velocity_model.fit(xy, difference=difference, direction=direction)
        velocity = velocity_model.velocity()

        acceleration_model = AccelerationModel(self.pitch)
        acceleration_model.fit(xy, difference=difference, direction=direction)
        acceleration = acceleration_model.acceleration()

        # constant of air resistance
        k = 0.0037
        equivalent_slope = (acceleration.property / g) + (
            (k * np.square(velocity.property)) / g
        )

        equivalent_mass = np.sqrt(np.square(equivalent_slope) + 1)

        energy_cost_running = _ecr_mixed(
            velocity.property, acceleration.property, equivalent_slope, equivalent_mass
        )

        metabolic_power = energy_cost_running * velocity.property

        self._metabolic_power = metabolic_power / xy.framerate
  # constants
  EDGES = np.array([-0.3, -0.2, -0.1,  0,  0.1,  0.2,  0.3,  0.4])
  COEFF = np.array([
              [0.28, -1.66, 3.81, -3.96, 4.01],
              [0.03, -0.15, 0.98, -2.25, 3.14],
              [0.69, -3.21, 5.94, -5.07, 2.79],
              [1.25, -6.57, 13.14, -11.15, 5.35],
              [0.68, -4.17, 10.17, -10.31, 8.66],
              [3.80, -14.91, 22.94, -14.53, 11.24],
              [44.95, -122.88, 126.94, -57.46, 21.39],
              [94.62, -213.94, 184.43, -68.49, 25.04],
          ])
  
  
  
  def get_interpolation_weight_matrix(es):
      T = es.shape[0]
      N = es.shape[1]
      W = np.zeros((T, N, len(EDGES)))
      # dim = (T, N, zones)
  
      idxs = EDGES.searchsorted(es)
  
      # mask for non-edge casees
      mask = (idxs > 0) & (idxs < 8)
  
      grid_t, grid_n = np.mgrid[0:T, 0:N]
  
      W[grid_t[mask], grid_n[mask], idxs[mask]-1] = (EDGES[idxs[mask]] - es[mask]) * 10
      W[grid_t[mask], grid_n[mask], idxs[mask]] = (es[mask] - EDGES[idxs[mask]-1]) * 10
  
      # edge cases
      W[idxs == 0, 0] = 1
      W[idxs == 8, 7] = 1
  
      # maybe not needed:
      W = W.round(3)
      
      return W
  
  
  def calc_ecw(es, vel, em):
      W = get_interpolation_weight_matrix(es)
  
      WC = np.matmul(W, COEFF)
      V = np.stack(
          (
              np.power(vel, 4),
              np.power(vel, 3),
              np.power(vel, 2),
              vel,
              np.ones(vel.shape)
          ), axis=-1
      )
      
      ECW = np.multiply(np.multiply(WC, V).sum(axis=2), em)
      
      return ECW
    def metabolic_power(self):
        metabolic_power = PlayerProperty(
            property=self._metabolic_power,
            name="metabolic_power",
            framerate=self.framerate,
        )
        return metabolic_power

    def cumulative_metabolic_power(self):
        cum_metp = np.nancumsum(self._metabolic_power, axis=0)
        cumulative_metabolic_power = PlayerProperty(
            property=cum_metp,
            name="cumulative_metabolic_power",
            framerate=self.framerate,
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
            property=eq_dist, name="equivalent_distance", framerate=self.framerate
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
            framerate=self.framerate,
        )
        return cumulative_equivalent_distance

import pytest
import numpy as np

from floodlight.models.kinetics import MetabolicPowerModel


@pytest.mark.unit
def test_calc_es(example_velocity, example_acceleration) -> None:
    # Arrange
    velocity = example_velocity
    acceleration = example_acceleration

    # Act
    equivalent_slope = MetabolicPowerModel._calc_es(velocity, acceleration)

    # Assert
    assert np.array_equal(
        np.round(equivalent_slope, 3),
        np.array(((0.184, 0.5), (0.069, 0.122), (-0.049, -0.273))),
    )


@pytest.mark.unit
def test_calc_em(example_equivalent_slope) -> None:
    # Arrange
    equivalent_slope = example_equivalent_slope

    # Act
    equivalent_mass = MetabolicPowerModel._calc_em(equivalent_slope)

    # Assert
    assert np.array_equal(
        np.round(equivalent_mass, 3),
        np.array(((1, 1.011), (1.006, 1.02), (1.118, 1.118))),
    )


@pytest.mark.unit
def test_calc_v_trans(example_equivalent_slope) -> None:
    # Arrange
    equivalent_slope = example_equivalent_slope

    # Act
    v_trans = MetabolicPowerModel._calc_v_trans(equivalent_slope)

    # Assert
    assert np.array_equal(
        np.round(v_trans, 3), np.array(((2.27, 1.704), (2.285, 1.434), (1.044, 9.717)))
    )


@pytest.mark.unit
def test_is_running(example_velocity, example_equivalent_slope) -> None:
    # Arrange
    velocity = example_velocity
    equivalent_slope = example_equivalent_slope

    # Act
    is_running = MetabolicPowerModel._is_running(velocity, equivalent_slope)

    # Assert
    assert np.array_equal(
        is_running, np.array(((False, False), (True, True), (True, False)))
    )


@pytest.mark.unit
def test_get_interpolation_matrix(example_equivalent_slope) -> None:
    # Arrange
    equivalent_slope = example_equivalent_slope

    # Act
    W = MetabolicPowerModel._get_interpolation_weight_matrix(equivalent_slope)

    # Assert
    assert np.array_equal(
        np.round(W, 3),
        np.array(
            (
                ([0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0.5, 0.5, 0, 0]),
                ([0, 0.1, 0.9, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]),
                ([0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0]),
            )
        ),
    )


@pytest.mark.unit
def test_calc_ecw(
    example_equivalent_slope, example_velocity, example_equivalent_mass
) -> None:
    # Arrange
    equivalent_slope = example_equivalent_slope
    velocity = example_velocity
    equivalent_mass = example_equivalent_mass

    # Act
    ecw = MetabolicPowerModel._calc_ecw(equivalent_slope, velocity, equivalent_mass)

    # Assert
    assert np.array_equal(
        np.round(ecw, 3), np.array(((2.02, 8.962), (6.746, 1043.807), (992.779, 3.013)))
    )


@pytest.mark.unit
def test_calc_ecr(example_equivalent_slope, example_equivalent_mass) -> None:
    # Arrange
    equivalent_slope = example_equivalent_slope
    equivalent_mass = example_equivalent_mass

    # Act
    ecr = MetabolicPowerModel._calc_ecr(equivalent_slope, equivalent_mass)

    # Assert
    assert np.array_equal(
        np.round(ecr, 3), np.array(((3.6, 7.988), (1.79, 9.708), (22.625, 4.668)))
    )


@pytest.mark.unit
def test_calc_ecl(
    example_equivalent_slope, example_velocity, example_equivalent_mass
) -> None:
    # Arrange
    equivalent_slope = example_equivalent_slope
    velocity = example_velocity
    equivalent_mass = example_equivalent_mass

    # Act
    ecl = MetabolicPowerModel._calc_ecl(equivalent_slope, velocity, equivalent_mass)

    # Assert
    assert np.array_equal(
        np.round(ecl, 3), np.array(((2.02, 8.962), (1.79, 9.708), (22.625, 3.013)))
    )


@pytest.mark.unit
def test_calc_metabolic_power(
    example_equivalent_slope,
    example_velocity,
    example_equivalent_mass,
) -> None:
    # Arrange
    equivalent_slope = example_equivalent_slope
    velocity = example_velocity
    equivalent_mass = example_equivalent_mass
    framerate = 20

    # Act
    metabolic_power = MetabolicPowerModel._calc_metabolic_power(
        equivalent_slope, velocity, equivalent_mass, framerate
    )

    # Assert
    assert np.array_equal(
        np.round(metabolic_power, 3),
        np.array(((2.020, 0.896), (5.011, 48.540), (52.038, 6.931))),
    )


@pytest.mark.unit
def test_metabolic_power(example_pitch_dfl, example_xy_object_kinetics) -> None:
    # Arrange
    xy = example_xy_object_kinetics

    # Act
    metp_model = MetabolicPowerModel()
    metp_model.fit(xy)
    metabolic_power = metp_model.metabolic_power()

    # Assert
    assert np.array_equal(
        np.round(metabolic_power, 3),
        np.array(((9.177, 4.452), (9.306, 4.988), (9.439, 5.570))),
    )


@pytest.mark.unit
def test_cumulative_metabolic_power(
    example_pitch_dfl, example_xy_object_kinetics
) -> None:
    # Arrange
    xy = example_xy_object_kinetics

    # Act
    metp_model = MetabolicPowerModel()
    metp_model.fit(xy)
    cumulative_metabolic_power = metp_model.cumulative_metabolic_power()

    # Assert
    assert np.array_equal(
        np.round(cumulative_metabolic_power, 3),
        np.array(((0.459, 0.223), (0.924, 0.472), (1.396, 0.751))),
    )


@pytest.mark.unit
def test_equivalent_distance(example_pitch_dfl, example_xy_object_kinetics) -> None:
    # Arrange
    xy = example_xy_object_kinetics

    # Act
    metp_model = MetabolicPowerModel()
    metp_model.fit(xy)
    equivalent_distance = metp_model.equivalent_distance()

    # Assert
    assert np.array_equal(
        np.round(equivalent_distance, 3),
        np.array(((2.549, 1.237), (2.585, 1.386), (2.622, 1.547))),
    )


@pytest.mark.unit
def test_cumulative_equivalent_distance(
    example_pitch_dfl, example_xy_object_kinetics
) -> None:
    # Arrange
    xy = example_xy_object_kinetics

    # Act
    metp_model = MetabolicPowerModel()
    metp_model.fit(xy)
    cumulative_equivalent_distance = metp_model.cumulative_equivalent_distance()

    # Assert
    assert np.array_equal(
        np.round(cumulative_equivalent_distance, 3),
        np.array(((0.127, 0.062), (0.257, 0.131), (0.388, 0.208))),
    )

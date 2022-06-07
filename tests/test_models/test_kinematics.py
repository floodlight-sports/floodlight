import pytest
import numpy as np

from floodlight.models.kinematics import DistanceModel, VelocityModel, AccelerationModel


# Differences in the kinematic models can be calculated via central or backward
# difference methode. This is specified in the respective models .fit()-method.
# As this has no impact on the calculations in the other class methods, only the
# .fit()-methods are tested with both difference methods. The other class methods are
# tested with the default difference methode (i.e., 'central').


@pytest.mark.unit
def test_distance_model_fit_difference_central(example_xy_object_kinematics) -> None:
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    dist_model = DistanceModel()
    dist_model.fit(xy)
    distance_covered = dist_model._distance_euclidean_

    # Assert
    assert np.array_equal(
        np.round(distance_covered, 3),
        np.array(((1, 0), (1.118, np.NaN), (1.414, 0))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_distance_model_fit_difference_backward(example_xy_object_kinematics) -> None:
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    dist_model = DistanceModel()
    dist_model.fit(xy, difference="backward")
    distance_covered = dist_model._distance_euclidean_

    # Assert
    assert np.array_equal(
        np.round(distance_covered, 3),
        np.array(((0, 0), (1, np.NaN), (1.414, 0))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_distance_covered(example_xy_object_kinematics) -> None:
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    dist_model = DistanceModel()
    dist_model.fit(xy)
    distance_covered = dist_model.distance_covered()

    # Assert
    assert np.array_equal(
        np.round(distance_covered, 3),
        np.array(((1, 0), (1.118, np.NaN), (1.414, 0))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_cumulative_distance_covered(example_xy_object_kinematics) -> None:
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    dist_model = DistanceModel()
    dist_model.fit(xy)
    distance_covered = dist_model.cumulative_distance_covered()

    # Assert
    assert np.array_equal(
        np.round(distance_covered, 3),
        np.array(((1, 0), (2.118, 0), (3.532, 0))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_velocity_model_fit_difference_central(example_xy_object_kinematics) -> None:
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    vel_model = VelocityModel()
    vel_model.fit(xy)
    velocity = vel_model._velocity_

    # Assert
    assert np.array_equal(
        np.round(velocity, 3),
        np.array(((20, 0), (22.361, np.NaN), (28.284, 0))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_velocity_model_fit_difference_backward(example_xy_object_kinematics) -> None:
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    vel_model = VelocityModel()
    vel_model.fit(xy, difference="backward")
    velocity = vel_model._velocity_

    # Assert
    assert np.array_equal(
        np.round(velocity, 3),
        np.array(((0, 0), (20, np.NaN), (28.284, 0))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_velocity(example_xy_object_kinematics) -> None:
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    vel_model = VelocityModel()
    vel_model.fit(xy)
    velocity = vel_model.velocity()

    # Assert
    assert np.array_equal(
        np.round(velocity, 3),
        np.array(((20, 0), (22.361, np.NaN), (28.284, 0))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_acceleration_model_difference_central(example_xy_object_kinematics) -> None:
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    acc_model = AccelerationModel()
    acc_model.fit(xy)
    acceleration = acc_model._acceleration_

    # Assert
    assert np.array_equal(
        np.round(acceleration, 3),
        np.array(((47.214, 0), (82.843, np.NaN), (118.472, 0))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_acceleration_model_difference_backward(example_xy_object_kinematics) -> None:
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    acc_model = AccelerationModel()
    acc_model.fit(xy, difference="backward")
    acceleration = acc_model._acceleration_

    # Assert
    assert np.array_equal(
        np.round(acceleration, 3),
        np.array(((0, 0), (400, np.NaN), (165.685, 0))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_acceleration(example_xy_object_kinematics) -> None:
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    acc_model = AccelerationModel()
    acc_model.fit(xy)
    acceleration = acc_model.acceleration()

    # Assert
    assert np.array_equal(
        np.round(acceleration, 3),
        np.array(((47.214, 0), (82.843, np.NaN), (118.472, 0))),
        equal_nan=True,
    )

import pytest
import numpy as np

from floodlight.models.kinematics import DistanceModel, VelocityModel, AccelerationModel


@pytest.mark.unit
def test_distance_covered_difference_central(example_xy_object_kinematics):
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    dist_model = DistanceModel()
    dist_model.fit(xy)
    distance_covered = dist_model.distance_covered()

    # Assert
    assert np.array_equal(
        np.round(distance_covered, 3),
        np.array(((1, np.NaN), (1.118, 1.414), (1.414, np.NaN))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_cumulative_distance_covered_difference_central(example_xy_object_kinematics):
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    dist_model = DistanceModel()
    dist_model.fit(xy)
    distance_covered = dist_model.cumulative_distance_covered()

    # Assert
    assert np.array_equal(
        np.round(distance_covered, 3),
        np.array(((1, 0), (2.118, 1.414), (3.532, 1.414))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_velocity_difference_central(example_xy_object_kinematics):
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    vel_model = VelocityModel()
    vel_model.fit(xy)
    velocity = vel_model.velocity()

    # Assert
    assert np.array_equal(
        np.round(velocity, 3),
        np.array(((20, np.NaN), (22.361, 28.284), (28.284, np.NaN))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_acceleration_difference_central(example_xy_object_kinematics):
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    acc_model = AccelerationModel()
    acc_model.fit(xy)
    acceleration = acc_model.acceleration()

    # Assert
    assert np.array_equal(
        np.round(acceleration, 3),
        np.array(((47.214, np.NaN), (82.843, np.NaN), (118.472, np.NaN))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_distance_covered_difference_backward(example_xy_object_kinematics):
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    dist_model = DistanceModel()
    dist_model.fit(xy, difference="backward")
    distance_covered = dist_model.distance_covered()

    # Assert
    assert np.array_equal(
        np.round(distance_covered, 3),
        np.array(((0, 1.414), (1, np.NaN), (1.414, np.NaN))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_cumulative_distance_covered_difference_backward(example_xy_object_kinematics):
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    dist_model = DistanceModel()
    dist_model.fit(xy, difference="backward")
    distance_covered = dist_model.cumulative_distance_covered()

    # Assert
    assert np.array_equal(
        np.round(distance_covered, 3),
        np.array(((0, 1.414), (1, 1.414), (2.414, 1.414))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_velocity_difference_backward(example_xy_object_kinematics):
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    vel_model = VelocityModel()
    vel_model.fit(xy, difference="backward")
    velocity = vel_model.velocity()

    # Assert
    assert np.array_equal(
        np.round(velocity, 3),
        np.array(((0, 28.284), (20, np.NaN), (28.284, np.NaN))),
        equal_nan=True,
    )


@pytest.mark.unit
def test_acceleration_difference_backward(example_xy_object_kinematics):
    # Arrange
    xy = example_xy_object_kinematics

    # Act
    acc_model = AccelerationModel()
    acc_model.fit(xy, difference="backward")
    acceleration = acc_model.acceleration()

    # Assert
    assert np.array_equal(
        np.round(acceleration, 3),
        np.array(((400, np.NaN), (165.685, np.NaN), (-565.685, np.NaN))),
        equal_nan=True,
    )

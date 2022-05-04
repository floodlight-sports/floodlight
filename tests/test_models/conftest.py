import pytest
import numpy as np
from floodlight import Pitch, XY


@pytest.fixture()
def example_xy_object_kinetics() -> XY:
    xy = XY(
        xy=np.array(
            (
                (37.586, 10.144, 32.343, 7.752),
                (37.694, 10.144, 32.318, 7.731),
                (37.803, 10.145, 32.285, 7.708),
            )
        ),
        framerate=20,
    )
    return xy


# sample data for testing geometry models
@pytest.fixture()
def example_xy_object_geometry():
    xy = XY(np.array(((1, 1, 2, -2, np.nan, -2), (1.5, np.nan, np.nan, -0, 1, 1))))
    return xy


# sample data for testing kinematic models
@pytest.fixture()
def example_xy_object_kinematics():
    xy = XY(
        xy=np.array(((0, 0, -1, 1), (0, 1, np.NAN, np.NAN), (1, 2, 1, -1))),
        framerate=20,
    )
    return xy


@pytest.fixture()
def example_equivalent_slope() -> np.ndarray:
    equivalent_slope = np.array(((0, 0.15), (-0.11, 0.2), (0.5, -0.5)))
    return equivalent_slope


@pytest.fixture()
def example_velocity() -> np.ndarray:
    velocity = np.array(((1, 0.1), (2.8, 5), (2.3, 2.3)))
    return velocity


@pytest.fixture()
def example_acceleration() -> np.ndarray:
    acceleration = np.array(((1.8, 4.9), (0.65, 1.1), (-0.5, -2.7)))
    return acceleration


@pytest.fixture()
def example_equivalent_mass() -> np.ndarray:
    equivalent_mass = np.array(((1, 1.011), (1.006, 1.02), (1.118, 1.118)))
    return equivalent_mass


@pytest.fixture()
def example_pitch_dfl():
    pitch = Pitch.from_template("dfl", length=100, width=50)
    return pitch

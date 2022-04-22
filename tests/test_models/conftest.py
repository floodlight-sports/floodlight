import numpy as np
import pytest

from floodlight import XY


# sample data for testing kinematic models
@pytest.fixture()
def example_xy_object_kinematics():
    xy = XY(
        xy=np.array(((0, 0, -1, 1), (0, 1, np.NAN, np.NAN), (1, 2, 1, -1))),
        framerate=20,
    )
    return xy

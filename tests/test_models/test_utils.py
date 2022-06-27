import pytest
import numpy as np

from floodlight.models.utils import _exclude_x_ids


@pytest.mark.unit
def test_exclude_x_ids(example_xy_object_geometry_convex_hull) -> None:
    # Arrange
    xy = example_xy_object_geometry_convex_hull
    xIDs_to_exclude = [0, 4]

    # Act
    include_x_ids = _exclude_x_ids(xy, xIDs_to_exclude)

    # Assert
    assert np.array_equal(
        include_x_ids,
        np.array([False, False, True, True, True, True, True, True, False, False]),
    )

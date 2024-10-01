import pytest
import numpy as np
from floodlight import XY
from floodlight.models.geometry import SpatialMetricsModel
from floodlight.core.property import TeamProperty

@pytest.mark.unit
def test_convex_hull_area():
    # Create dummy XY data with 3 frames and 3 points (6 values per frame for x and y)
    xy_data = np.array([
        [0, 2, 4, 0, 2, 4],
        [1, 3, 5, 1, 3, 5],
        [2, 4, 6, 2, 4, 6]
    ])
    xy = XY(xy_data)

    # Instantiate the model and fit it
    smm = SpatialMetricsModel()
    smm.fit(xy)

    # Retrieve convex hull area
    convex_hull_result = smm.convex_hull_area()

    # Assert that the result is a TeamProperty instance and not NaN
    assert isinstance(convex_hull_result, TeamProperty)
    assert not np.isnan(convex_hull_result.property).all()

@pytest.mark.unit
def test_effective_playing_space():
    # Create dummy XY data for two teams with 3 frames and 3 points each
    xy_data1 = np.array([
        [0, 2, 4, 0, 2, 4],
        [1, 3, 5, 1, 3, 5],
        [2, 4, 6, 2, 4, 6]
    ])
    xy_data2 = np.array([
        [10, 12, 14, 10, 12, 14],
        [11, 13, 15, 11, 13, 15],
        [12, 14, 16, 12, 14, 16]
    ])

    xy1 = XY(xy_data1)
    xy2 = XY(xy_data2)

    # Instantiate the model and fit it
    smm = SpatialMetricsModel()
    smm.fit(xy1, xy2)

    # Retrieve effective playing space
    eps_result = smm.effective_playing_space()

    # Assert that the result is a TeamProperty instance and not NaN
    assert isinstance(eps_result, TeamProperty)
    assert not np.isnan(eps_result.property).all()

@pytest.mark.unit
def test_invalid_xy_data():
    # Test with invalid XY data (all NaNs)
    xy_data = np.full((3, 6), np.nan)
    xy = XY(xy_data)

    # Instantiate the model and fit it
    smm = SpatialMetricsModel()
    smm.fit(xy)

    # Convex hull area should be all NaNs due to invalid data
    convex_hull_result = smm.convex_hull_area()

    # Assert that the convex hull area is NaN for all frames
    assert np.isnan(convex_hull_result.property).all()

@pytest.mark.unit
def test_convex_hull_area_not_enough_points():
    # Test with insufficient points for convex hull calculation
    xy_data = np.array([
        [1, 2, np.nan, 1, 2, np.nan],  # 2 points only, insufficient for convex hull
        [2, 3, np.nan, 2, 3, np.nan],
        [3, 4, np.nan, 3, 4, np.nan]
    ])
    xy = XY(xy_data)

    # Instantiate the model and fit it
    smm = SpatialMetricsModel()
    smm.fit(xy)

    # Convex hull area should be NaN due to insufficient points
    convex_hull_result = smm.convex_hull_area()

    # Assert that the convex hull area is NaN
    assert np.isnan(convex_hull_result.property).all()

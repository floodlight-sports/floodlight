import numpy as np
from floodlight import XY
from floodlight.models.distance_model import DistanceModel
import pytest


@pytest.mark.unit
def test_distance_model_nearest_mate(xy_data):
    """Test the DistanceModel for distance to nearest mate."""
    dm = DistanceModel()
    dm.fit(xy_data)
    dtnm = dm.distance_to_nearest_mate()

    assert dtnm is not None
    assert isinstance(dtnm.property, np.ndarray)
    assert dtnm.property.shape[0] == 2 


@pytest.mark.unit
def test_distance_model_team_spread(xy_data):
    """Test the DistanceModel for team spread calculation."""
    dm = DistanceModel()
    dm.fit(xy_data)
    spread = dm.team_spread()

    assert spread is not None
    assert isinstance(spread.property, np.ndarray)
    assert spread.property.shape[0] == 2  


@pytest.mark.unit
def test_distance_model_opponents(xy_data):
    """Test the DistanceModel for distance to nearest opponents."""
    opponent_data = np.array(
        [
            [
                50,
                70,
                80,
                60,
                75,
                55,
                85,
                65,
                55,
                70,
                60,
                75,
                45,
                85,
                50,
                90,
                70,
                65,
                80,
                60,
                55,
                65,
            ],
            [
                75,
                90,
                85,
                95,
                60,
                70,
                75,
                90,
                80,
                85,
                65,
                75,
                95,
                80,
                90,
                100,
                70,
                80,
                95,
                85,
                90,
                75,
            ],
        ]
    )
    xy_opponents = XY(opponent_data)

    dm = DistanceModel()
    dm.fit(xy_data, xy_opponents)
    dtno1, dtno2 = dm.distance_to_nearest_opponents()

    assert dtno1 is not None
    assert dtno2 is not None
    assert isinstance(dtno1.property, np.ndarray)
    assert isinstance(dtno2.property, np.ndarray)
    assert dtno1.property.shape[0] == 2  
    assert dtno2.property.shape[0] == 2


@pytest.mark.unit
def test_distance_model_nearest_mate_all_nan(xy_all_nan):
    """Test the DistanceModel for distance to nearest mate with NaN data."""
    dm = DistanceModel()
    dm.fit(xy_all_nan)
    dtnm = dm.distance_to_nearest_mate()

    assert np.all(np.isnan(dtnm.property)) and dtnm.property.shape == (2, 1)
    assert isinstance(dtnm.property, np.ndarray)
    assert dtnm.property.shape[0] == 2


@pytest.mark.unit
def test_distance_model_team_spread_all_nan(xy_all_nan):
    """Test the DistanceModel for team spread calculation with NaN data."""
    dm = DistanceModel()
    dm.fit(xy_all_nan)
    spread = dm.team_spread()

    assert np.all(np.isnan(spread.property))
    assert spread.property.shape == (2, 1)
    assert isinstance(spread.property, np.ndarray)
    assert spread.property.shape[0] == 2


@pytest.mark.unit
def test_distance_model_opponents_to_all_nan(xy_all_nan):
    """Test the DistanceModel for distance to nearest opponents with NaN data."""
    opponent_data = np.array(
        [
            [
                50,
                70,
                80,
                60,
                75,
                55,
                85,
                65,
                55,
                70,
                60,
                75,
                45,
                85,
                50,
                90,
                70,
                65,
                80,
                60,
                55,
                65,
            ],
            [
                75,
                90,
                85,
                95,
                60,
                70,
                75,
                90,
                80,
                85,
                65,
                75,
                95,
                80,
                90,
                100,
                70,
                80,
                95,
                85,
                90,
                75,
            ],
        ]
    )
    xy_opponents = XY(opponent_data)

    dm = DistanceModel()
    dm.fit(xy_all_nan, xy_opponents)
    dtno1, dtno2 = dm.distance_to_nearest_opponents()

    assert np.all(np.isnan(dtno1.property))
    assert dtno1.property.shape == (2, 1)
    assert np.all(np.isnan(dtno2.property))
    assert dtno2.property.shape == (2, 1)
    assert isinstance(dtno1.property, np.ndarray)
    assert isinstance(dtno2.property, np.ndarray)
    assert dtno1.property.shape[0] == 2
    assert dtno2.property.shape[0] == 2


@pytest.mark.unit
def test_distance_model_nearest_mate_all_zeros(xy_all_zeros):
    """Test the DistanceModel for distance to nearest mate with zero data."""
    dm = DistanceModel()
    dm.fit(xy_all_zeros)
    dtnm = dm.distance_to_nearest_mate()

    assert np.all(np.isnan(dtnm.property)) and dtnm.property.shape == (2, 1)
    assert isinstance(dtnm.property, np.ndarray)
    assert dtnm.property.shape[0] == 2


@pytest.mark.unit
def test_distance_model_team_spread_all_zeros(xy_all_zeros):
    """Test the DistanceModel for team spread calculation with zero data."""
    dm = DistanceModel()
    dm.fit(xy_all_zeros)
    spread = dm.team_spread()

    assert spread.property.shape == (2, 1)
    assert np.all(spread.property == 0)
    assert isinstance(spread.property, np.ndarray)
    assert spread.property.shape[0] == 2


@pytest.mark.unit
def test_distance_model_opponents_to_all_zeros(xy_all_zeros):
    """Test the DistanceModel for distance to nearest opponents with zero data."""
    opponent_data = np.array(
        [
            [
                1,
                0,
                80,
                60,
                75,
                55,
                85,
                65,
                55,
                70,
                60,
                75,
                45,
                85,
                50,
                90,
                70,
                65,
                80,
                60,
                55,
                65,
            ],
            [
                1,
                0,
                85,
                95,
                60,
                70,
                75,
                90,
                80,
                85,
                65,
                75,
                95,
                80,
                90,
                100,
                70,
                80,
                95,
                85,
                90,
                75,
            ],
        ]
    )
    expected_dtno1 = np.array([[1.0], [1.0]])
    xy_opponents = XY(opponent_data)

    dm = DistanceModel()
    dm.fit(xy_all_zeros, xy_opponents)
    dtno1, dtno2 = dm.distance_to_nearest_opponents()

    assert np.array_equal(
        dtno1.property, expected_dtno1
    ), f"Expected dtno1 to be {expected_dtno1}, but got {dtno1.property}"
    assert dtno1.property.shape == (2, 1)
    assert dtno2.property.shape == (2, 1)
    assert isinstance(dtno1.property, np.ndarray)
    assert isinstance(dtno2.property, np.ndarray)
    assert dtno1.property.shape[0] == 2
    assert dtno2.property.shape[0] == 2


@pytest.mark.unit
def test_distance_model_nearest_mate_all_ones(xy_all_ones):
    """Test the DistanceModel for distance to nearest mate with all ones data."""
    dm = DistanceModel()
    dm.fit(xy_all_ones)
    dtnm = dm.distance_to_nearest_mate()

    assert np.all(np.isnan(dtnm.property)) and dtnm.property.shape == (2, 1)
    assert isinstance(dtnm.property, np.ndarray)
    assert dtnm.property.shape[0] == 2


@pytest.mark.unit
def test_distance_model_team_spread_all_ones(xy_all_ones):
    """Test the DistanceModel for team spread calculation with all ones data."""
    dm = DistanceModel()
    dm.fit(xy_all_ones)
    spread = dm.team_spread()

    assert spread.property.shape == (2, 1)
    assert np.all(spread.property == 0)
    assert isinstance(spread.property, np.ndarray)
    assert spread.property.shape[0] == 2


def test_distance_model_opponents_to_all_ones(xy_all_ones):
    """Test the DistanceModel for distance to nearest opponents."""
    # For opponent data, shift the coordinates
    opponent_data = np.array(
        [
            [
                1,
                0,
                80,
                60,
                75,
                55,
                85,
                65,
                55,
                70,
                60,
                75,
                45,
                85,
                50,
                90,
                70,
                65,
                80,
                60,
                55,
                65,
            ],
            [
                1,
                0,
                85,
                95,
                60,
                70,
                75,
                90,
                80,
                85,
                65,
                75,
                95,
                80,
                90,
                100,
                70,
                80,
                95,
                85,
                90,
                75,
            ],
        ]
    )

    expected_dtno1 = np.array([[1.0], [1.0]])

    xy_opponents = XY(opponent_data)

    dm = DistanceModel()
    dm.fit(xy_all_ones, xy_opponents)
    dtno1, dtno2 = dm.distance_to_nearest_opponents()

    assert np.array_equal(
        dtno1.property, expected_dtno1
    ), f"Expected dtno1 to be {expected_dtno1}, but got {dtno1.property}"
    assert dtno1.property.shape == (2, 1)
    assert dtno2.property.shape == (2, 1)
    assert isinstance(dtno1.property, np.ndarray)
    assert isinstance(dtno2.property, np.ndarray)
    assert dtno1.property.shape[0] == 2
    assert dtno2.property.shape[0] == 2

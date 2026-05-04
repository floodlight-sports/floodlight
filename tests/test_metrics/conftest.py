import pytest
import numpy as np

from floodlight import PlayerProperty, TeamProperty, XY


# Basic PlayerProperty fixtures
@pytest.fixture()
def player_property_simple() -> PlayerProperty:
    """Simple PlayerProperty with 6 frames, 2 players, clean data."""
    prop = PlayerProperty(
        property=np.array(
            [[10, 100], [20, 200], [30, 300], [40, 400], [50, 500], [60, 600]],
            dtype=float,
        ),
        name="values",
    )
    return prop


@pytest.fixture()
def player_property_binning() -> PlayerProperty:
    """PlayerProperty for binning with varying values across zones."""
    prop = PlayerProperty(
        property=np.array(
            [[0, 8], [1, 7], [2, 6], [3, 5], [4, 4], [5, 3]], dtype=float
        ),
        name="binning",
    )
    return prop


# PlayerProperty with NaN values
@pytest.fixture()
def player_property_with_nans() -> PlayerProperty:
    """PlayerProperty with NaN values in both players."""
    prop = PlayerProperty(
        property=np.array(
            [[10, 10], [np.nan, 20], [30, np.nan], [40, 40], [50, 50], [60, 60]],
            dtype=float,
        ),
        name="values",
    )
    return prop


@pytest.fixture()
def player_property_binning_with_nans() -> PlayerProperty:
    """PlayerProperty for binning with NaN values."""
    prop = PlayerProperty(
        property=np.array(
            [[1, 1], [np.nan, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=float
        ),
        name="binning",
    )
    return prop


# TeamProperty fixtures
@pytest.fixture()
def team_property_simple() -> TeamProperty:
    """Simple TeamProperty with 6 frames, clean data."""
    prop = TeamProperty(
        property=np.array([10, 20, 30, 40, 50, 60], dtype=float), name="values"
    )
    return prop


@pytest.fixture()
def team_property_binning() -> TeamProperty:
    """TeamProperty for binning with varying values."""
    prop = TeamProperty(
        property=np.array([0, 1, 5, 6, 7, 8], dtype=float), name="binning"
    )
    return prop


# Mismatched dimension fixtures for validation tests
@pytest.fixture()
def player_property_3_players() -> PlayerProperty:
    """PlayerProperty with 3 players instead of 2."""
    prop = PlayerProperty(
        property=np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
            dtype=float,
        ),
        name="values",
    )
    return prop


@pytest.fixture()
def player_property_4_frames() -> PlayerProperty:
    """PlayerProperty with 4 frames instead of 6."""
    prop = PlayerProperty(
        property=np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float),
        name="values",
    )
    return prop


@pytest.fixture()
def example_xy_fsim() -> XY:
    """5 frames, 4 players in a stable rectangular formation with small noise."""
    xy = XY(
        np.array(
            [
                [10, 30, 30, 30, 10, 60, 30, 60],
                [11, 31, 31, 31, 11, 61, 31, 61],
                [10, 29, 30, 29, 10, 59, 30, 59],
                [11, 30, 31, 30, 11, 60, 31, 60],
                [10, 31, 30, 31, 10, 61, 30, 61],
            ],
            dtype=float,
        ),
        framerate=10,
    )
    return xy

import pandas as pd
import pytest
import numpy as np
import datetime
import pytz


# Create example data with fixtures to test all the functions

# XY
@pytest.fixture()
def example_xy_data_pos_int() -> np.ndarray:
    positions = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    return positions


@pytest.fixture()
def example_xy_data_none() -> np.ndarray:
    positions = np.array([[None, None, None, None], [None, None, None, None]])
    return positions


@pytest.fixture()
def example_xy_data_negative() -> np.ndarray:
    positions = np.array([[-1, -2, -3, -4], [-5, -6, -7, -8]])
    return positions


@pytest.fixture()
def example_xy_data_float() -> np.ndarray:
    positions = np.array(
        [
            [1.0, 2.3333, 0.00000000000000001, 99999999999999999],
            [2.32843476297480273847, 6.0, 7.5, 8],
        ]
    )
    return positions


@pytest.fixture()
def example_xy_data_string() -> np.ndarray:
    positions = np.array([["1", "2", "3", "4"], ["5", "6", "7", "8"]])
    return positions


# Events
@pytest.fixture()
def example_events_data_minimal() -> pd.DataFrame:
    data = {
        "eID": [1, 2],
        "gameclock": [1.1, 2.2],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def example_events_data_full() -> pd.DataFrame:
    # define timestamps
    kickoff = datetime.datetime.now(tz=pytz.utc)
    deltas = [datetime.timedelta(seconds=sec) for sec in [0, 2, 2.5, 3]]
    timestamps = [kickoff + delta for delta in deltas]

    # initialize data
    data = {
        "eID": [1, 2, 3, 4],
        "gameclock": [0, 2, 2.5, 3],
        "pID": [1, 2, 3, 4],
        "outcome": [1, 0, 1, 0],
        "timestamp": timestamps,
        "minute": [0, 2, 4, 5],
        "second": [0, 12, 15, 12],
        "at_x": [1.2, -34.56, 78.9, 10.0],
        "at_y": [-10.0, 98.7, -65.4, 32.1],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def example_events_data_none() -> pd.DataFrame:
    data = {
        "eID": [None, None],
        "gameclock": [None, None],
        "pID": [None, None],
        "outcome": [None, None],
        "timestamp": [None, None],
        "minute": [None, None],
        "second": [None, None],
    }
    return pd.DataFrame(data)

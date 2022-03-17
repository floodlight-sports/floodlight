import pytest
import numpy as np
import pandas as pd

from floodlight.core.code import Code


# Sample data for easy creation of core objects
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


# Code
@pytest.fixture()
def example_code() -> Code:
    array = np.array(["A"] * 5 + ["H"] * 5)
    name = "possession"
    definitions = {"H": "Home", "A": "Away"}
    framerate = 10
    code = Code(code=array, name=name, definitions=definitions, framerate=framerate)

    return code


@pytest.fixture()
def example_code_int() -> Code:
    array = np.array([0, 1, 2, 3])
    name = "intensity"
    definitions = {0: "None", 1: "Low", 2: "Medium", 3: "High"}
    framerate = 4
    code = Code(code=array, name=name, definitions=definitions, framerate=framerate)

    return code


# Events
@pytest.fixture()
def example_events_data_minimal() -> pd.DataFrame:
    data = {
        "eID": [1, 2],
        "gameclock": [1.1, 2.2],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def example_events_data_with_outcome_and_none() -> pd.DataFrame:
    data = {
        "eID": [1, 2, 2, 4, 1],
        "gameclock": [1.1412, 2.4122, 5.213, 11.214, 21.12552],
        "outcome": [0, 1, None, 0, None],
    }
    return pd.DataFrame(data)

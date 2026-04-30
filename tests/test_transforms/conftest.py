import pytest
import numpy as np
from floodlight import XY, Code
from floodlight.core.property import TeamProperty, PlayerProperty, DyadicProperty


@pytest.fixture()
def example_sequence():
    seq = np.array(
        [np.nan, np.nan, -5.07, -2.7, -3, np.nan, np.nan, 1.53, 27.13, None, 30.06]
    )
    return seq


@pytest.fixture()
def example_sequence_empty():
    seq = np.empty(())
    return seq


@pytest.fixture()
def example_sequence_two_dimensional():
    seq = np.array([[0, 1, 2], [3, 4, 5]])
    return seq


@pytest.fixture()
def example_sequence_full():
    seq = np.array([-5.07, -2.7, 1.53, 27.13, 30.06])
    return seq


@pytest.fixture()
def example_sequence_nan():
    seq = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    return seq


@pytest.fixture()
def example_xy_filter():

    xy = XY(
        np.array(
            [
                [np.nan, -8.66, np.nan, 1],
                [np.nan, -6.29, np.nan, 2],
                [-5.07, -4.31, np.nan, 3],
                [-2.7, -1.95, np.nan, 4],
                [np.nan, -0.13, np.nan, 5],
                [np.nan, 2.31, np.nan, 6],
                [1.53, 3.74, np.nan, 7],
                [5.13, 6.53, np.nan, 8],
                [7.02, 8.07, np.nan, 9],
                [9.48, 10.53, np.nan, 8],
                [10.09, np.nan, np.nan, 7],
                [12.31, np.nan, np.nan, 6],
                [13.22, np.nan, np.nan, 5],
                [14.88, 14.88, np.nan, 4],
                [16.23, 17.05, np.nan, 3],
                [17.06, 18.37, np.nan, 2],
                [18.56, 19.27, np.nan, 1],
                [20.32, 20.46, np.nan, 2],
                [21.7, 22.61, np.nan, 3],
                [23.11, 23.54, np.nan, 4],
                [24.23, 25.25, np.nan, 5],
                [25.74, 25.95, np.nan, 6],
                [27.13, 28.06, np.nan, 7],
                [None, 29.55, np.nan, 8],
                [30.06, np.nan, np.nan, 9],
            ]
        ),
        framerate=20,
    )

    return xy


@pytest.fixture()
def example_xy_filter_short():
    xy = XY(np.array([[23.11, 23.54, np.nan], [30.06, np.nan, np.nan]]), framerate=20)
    return xy


@pytest.fixture()
def example_xy_filter_one_frame():
    xy = XY(np.array((0, 1, np.nan)), framerate=20)
    return xy


@pytest.fixture()
def example_xy_filter_empty():
    xy = XY(np.array(()), framerate=20)
    return xy


@pytest.fixture()
def example_xy_spatial():
    xy = XY(
        np.array(
            [
                [0, 0, 6, 0, 3, 6],
                [1, 1, 7, 1, 4, 7],
                [2, 2, 8, 2, 5, 8],
            ]
        ),
        framerate=10,
        direction="lr",
    )
    return xy


@pytest.fixture()
def example_xy_spatial_with_nan():
    xy = XY(
        np.array(
            [
                [0.0, 0.0, 6.0, 0.0, 3.0, 6.0],
                [np.nan, np.nan, 7.0, 1.0, 4.0, 7.0],
                [2.0, 2.0, 8.0, 2.0, 5.0, 8.0],
            ]
        ),
        framerate=10,
    )
    return xy


@pytest.fixture()
def example_xy_permutation():
    xy = XY(
        np.array(
            [
                [0.0, 0.0, 10.0, 0.0, 5.0, 10.0],
                [10.0, 0.0, 0.0, 0.0, 5.0, 10.0],
                [5.0, 10.0, 0.0, 0.0, 10.0, 0.0],
                [0.0, 0.0, 5.0, 10.0, 10.0, 0.0],
            ]
        ),
        framerate=10,
    )
    return xy


@pytest.fixture()
def example_code_temporal():
    code = Code(
        code=np.array([0, 0, 1, 1, 2, 2, 1, 1, 0, 0], dtype=int),
        name="possession",
        definitions={0: "none", 1: "home", 2: "away"},
        framerate=50,
    )
    return code


@pytest.fixture()
def example_team_property_temporal():
    prop = TeamProperty(
        property=np.arange(10, dtype=float),
        name="stretch_index",
        framerate=50,
    )
    return prop


@pytest.fixture()
def example_player_property_temporal():
    prop = PlayerProperty(
        property=np.arange(30, dtype=float).reshape(10, 3),
        name="speed",
        framerate=50,
    )
    return prop


@pytest.fixture()
def example_dyadic_property_temporal():
    # Asymmetric shape N_1=2, N_2=3 -> (10, 2, 3) guards against any accidental
    # N_1 == N_2 assumption in resample's DyadicProperty adapter.
    prop = DyadicProperty(
        property=np.arange(60, dtype=float).reshape(10, 2, 3),
        name="distance",
        framerate=50,
    )
    return prop

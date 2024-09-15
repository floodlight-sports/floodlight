import pytest
import numpy as np
from floodlight import XY


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

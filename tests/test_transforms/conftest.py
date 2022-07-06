import pytest
import numpy as np
from floodlight import XY


@pytest.fixture()
def example_sequence():
    seq = np.array(
        [np.NaN, np.NaN, -5.07, -2.7, -3, np.NaN, np.NaN, 1.53, 27.13, None, 30.06]
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
    seq = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
    return seq


@pytest.fixture()
def example_xy_filter():

    xy = XY(
        np.array(
            [
                [np.NaN, -8.66, np.NaN, 1],
                [np.NaN, -6.29, np.NaN, 2],
                [-5.07, -4.31, np.NaN, 3],
                [-2.7, -1.95, np.NaN, 4],
                [np.NaN, -0.13, np.NaN, 5],
                [np.NaN, 2.31, np.NaN, 6],
                [1.53, 3.74, np.NaN, 7],
                [5.13, 6.53, np.NaN, 8],
                [7.02, 8.07, np.NaN, 9],
                [9.48, 10.53, np.NaN, 8],
                [10.09, np.NaN, np.NaN, 7],
                [12.31, np.NaN, np.NaN, 6],
                [13.22, np.NaN, np.NaN, 5],
                [14.88, 14.88, np.NaN, 4],
                [16.23, 17.05, np.NaN, 3],
                [17.06, 18.37, np.NaN, 2],
                [18.56, 19.27, np.NaN, 1],
                [20.32, 20.46, np.NaN, 2],
                [21.7, 22.61, np.NaN, 3],
                [23.11, 23.54, np.NaN, 4],
                [24.23, 25.25, np.NaN, 5],
                [25.74, 25.95, np.NaN, 6],
                [27.13, 28.06, np.NaN, 7],
                [None, 29.55, np.NaN, 8],
                [30.06, np.NaN, np.NaN, 9],
            ]
        ),
        framerate=20,
    )

    return xy


@pytest.fixture()
def example_xy_filter_short():
    xy = XY(np.array([[23.11, 23.54, np.NaN], [30.06, np.NaN, np.NaN]]), framerate=20)
    return xy


@pytest.fixture()
def example_xy_filter_one_frame():
    xy = XY(np.array((0, 1, np.NaN)), framerate=20)
    return xy


@pytest.fixture()
def example_xy_filter_empty():
    xy = XY(np.array(()), framerate=20)
    return xy

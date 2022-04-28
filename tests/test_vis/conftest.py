import matplotlib.pyplot as plt
import numpy as np
import pytest

from floodlight.core.xy import XY


@pytest.fixture()
def example_input_plot_football_pitch() -> []:
    ax = plt.subplots()[1]
    input = [(0, 105), (0, 68), 105, 68, "m", "standard", False, ax]

    return input


@pytest.fixture()
def example_input_plot_football_pitch_axis_ticks() -> []:
    ax = plt.subplots()[1]
    input = [(0, 105), (0, 68), 105, 68, "m", "standard", True, ax]

    return input


@pytest.fixture()
def example_input_plot_handball_pitch() -> []:
    ax = plt.subplots()[1]
    input = [(0, 40), (0, 20), "m", "standard", False, ax]

    return input


@pytest.fixture()
def example_input_plot_handball_pitch_axis_ticks() -> []:
    ax = plt.subplots()[1]
    input = [(0, 40), (0, 20), "m", "standard", True, ax]

    return input


@pytest.fixture()
def example_xy_object() -> XY:
    pos = np.array(
        [
            [35, 5, 35, 63, 25, 25, 25, 50],
            [45, 10, 45, 55, 35, 20, 35, 45],
            [55, 10, 55, 55, 45, 20, 45, 45],
            [88.5, 20, 88.5, 30, 88.5, 40, 88.5, 50],
        ]
    )

    return XY(pos)

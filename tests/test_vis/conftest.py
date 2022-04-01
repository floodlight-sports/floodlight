import pytest
import matplotlib.pyplot as plt


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

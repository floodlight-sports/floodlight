import matplotlib
from matplotlib import pyplot as plt
import pytest

from floodlight.vis.utils import check_axes_given


# Test check_axes_given(func
@pytest.mark.plot
def test_check_axes_given():
    # Arrange
    @check_axes_given
    def some_function_that_requires_matplotlib_axes(ax: matplotlib.axes = None):
        if isinstance(ax, matplotlib.axes.Axes):
            return True
        else:
            return False

    # Act
    without_ax_given = some_function_that_requires_matplotlib_axes(ax=None)
    with_ax_given = some_function_that_requires_matplotlib_axes(ax=plt.subplots()[1])

    # Assert
    assert without_ax_given
    assert with_ax_given

    plt.close()

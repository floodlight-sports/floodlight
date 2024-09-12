import pytest
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt    # noqa: 402

from floodlight.vis.utils import check_axes_given   # noqa: 402


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

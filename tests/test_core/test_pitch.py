import pytest
import matplotlib.pyplot as plt
import numpy as np

from floodlight.core.pitch import Pitch


# Test def from_template(cls, template_name, **kwargs) class method
@pytest.mark.unit
def test_template_opta() -> None:
    # Arrange
    pitch = Pitch.from_template("opta")

    # Assert
    assert pitch.xlim == (0.0, 100.0)
    assert pitch.ylim == (0.0, 100.0)
    assert pitch.unit == "percent"
    assert pitch.boundaries == "fixed"


@pytest.mark.unit
def test_template_tracab() -> None:
    # Arrange
    pitch = Pitch.from_template("tracab", length=110, width=68)

    # Assert
    assert pitch.xlim == (-55, 55)
    assert pitch.ylim == (-34, 34)
    assert pitch.unit == "cm"
    assert pitch.boundaries == "flexible"


@pytest.mark.unit
def test_template_dfl() -> None:
    # Arrange
    pitch = Pitch.from_template("dfl", length=110, width=68)

    # Assert
    assert pitch.xlim == (-55, 55)
    assert pitch.ylim == (-34, 34)
    assert pitch.unit == "m"
    assert pitch.boundaries == "flexible"


@pytest.mark.unit
def test_template_statsperform() -> None:
    # Arrange
    pitch = Pitch.from_template("statsperform", length=110, width=68)

    # Assert
    assert pitch.xlim == (-55, 55)
    assert pitch.ylim == (-34, 34)
    assert pitch.unit == "m"
    assert pitch.boundaries == "flexible"


# Test def center(self) property
@pytest.mark.unit
def test_center_property() -> None:
    # Arrange
    pitch1 = Pitch(xlim=(0, 100), ylim=(0, 100), unit="m", boundaries="fixed")
    pitch2 = Pitch(xlim=(-100, 0), ylim=(-100, 100), unit="m", boundaries="fixed")
    pitch3 = Pitch(xlim=(-30, 70), ylim=(-20, 20), unit="m", boundaries="fixed")

    # Assert
    assert pitch1.center == (50.0, 50.0)
    assert pitch2.center == (-50.0, 0.0)
    assert pitch3.center == (20.0, 0.0)


# Test def plot(self color_scheme: str = "standard", show_axis_ticks: bool = False,
#               ax: plt.axes = None, **kwargs)
# Test return
@pytest.mark.unit
def test_plot_football_return_matplotlib_axes_without_given_as_argument(
    example_pitch_football,
) -> None:
    # Arrange
    axes = plt.subplots()[1]
    # Act
    ax = example_pitch_football.plot()
    # Assert
    assert type(ax) == type(axes)


@pytest.mark.unit
def test_plot_handball_pitch_return_matplotlib_axes_without_given_as_argument(
    example_pitch_handball,
) -> None:
    # Arrange
    axes = plt.subplots()[1]
    # Act
    ax = example_pitch_handball.plot()
    # Assert
    assert type(ax) == type(axes)


@pytest.mark.unit
def test_plot_football_pitch_return_matplotlib_axes_with_axes_given_as_argument(
    example_pitch_football,
) -> None:
    # Arrange
    axes = plt.subplots()[1]
    # Act
    ax = example_pitch_football.plot(ax=axes)
    # Assert
    assert ax == axes


@pytest.mark.unit
def test_plot_handball_pitch_return_matplotlib_axes_with_axes_given_as_argument(
    example_pitch_handball,
) -> None:
    # Arrange
    axes = plt.subplots()[1]
    # Act
    ax = example_pitch_handball.plot(ax=axes)
    # Assert
    assert ax == axes


# Test value error if wrong sport is given
@pytest.mark.unit
def test_plot_value_error_unvalid_sport() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 105), ylim=(0, 68), unit="m", boundaries="fixed", sport="NoSport"
    )
    # Assert
    with pytest.raises(ValueError):
        pitch.plot()


# Test value error if wrong color_scheme is given
@pytest.mark.unit
def test_plot_value_error_unvalid_color_scheme(example_pitch_football) -> None:
    # Assert
    with pytest.raises(ValueError):
        example_pitch_football.plot(color_scheme="No valid color scheme")


# Test aspect ratio for given sports and unit (and width/length)
@pytest.mark.unit
def test_plot_football_pitch_aspect_ratio_unit_m() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 105), ylim=(0, 68), unit="m", boundaries="fixed", sport="football"
    )
    # Act
    ax = pitch.plot()
    # Assert
    assert ax.get_aspect() == 1.0


@pytest.mark.unit
def test_plot_handball_pitch_aspect_ratio_unit_m() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 40), ylim=(0, 20), unit="m", boundaries="fixed", sport="handball"
    )
    # Act
    ax = pitch.plot()
    # Assert
    assert ax.get_aspect() == 1.0


@pytest.mark.unit
def test_plot_football_pitch_aspect_ratio_unit_cm() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 10500), ylim=(0, 6800), unit="cm", boundaries="fixed", sport="football"
    )
    # Act
    ax = pitch.plot()
    # Assert
    assert ax.get_aspect() == 1.0


@pytest.mark.unit
def test_plot_handball_pitch_aspect_ratio_unit_cm() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 4000), ylim=(0, 2000), unit="cm", boundaries="fixed", sport="handball"
    )
    # Act
    ax = pitch.plot()
    # Assert
    assert ax.get_aspect() == 1.0


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore: Since self.unit == 'percent'")
def test_plot_football_pitch_aspect_ratio_unit_percent_without_width_length() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 100),
        ylim=(0, 100),
        unit="percent",
        boundaries="fixed",
        sport="football",
    )
    # Act
    ax = pitch.plot()
    # Assert
    assert ax.get_aspect() == (68 / 105)


@pytest.mark.unit
def test_plot_football_pitch_aspect_ratio_unit_percent_with_width_length() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 100),
        ylim=(0, 100),
        unit="percent",
        boundaries="fixed",
        width=60,
        length=100,
        sport="football",
    )
    # Act
    ax = pitch.plot()
    # Assert
    assert ax.get_aspect() == (60 / 100)


@pytest.mark.unit
def test_plot_handball_pitch_aspect_ratio_unit_percent() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 100),
        ylim=(0, 100),
        unit="percent",
        boundaries="fixed",
        sport="handball",
    )
    # Act
    ax = pitch.plot()
    # Assert
    assert ax.get_aspect() == 0.5


# Test ticks
@pytest.mark.unit
def test_plot_football_pitch_show_axis_ticks_default(example_pitch_football) -> None:
    # Act
    ax = example_pitch_football.plot()
    # Assert
    assert ax.get_xticks() == []
    assert ax.get_yticks() == []


@pytest.mark.unit
def test_plot_football_pitch_show_axis_ticks_True(example_pitch_football) -> None:
    # Act
    ax = example_pitch_football.plot(show_axis_ticks=True)
    # Assert
    assert np.array_equal(
        np.array(ax.get_xticks()), np.array([-20, 0, 20, 40, 60, 80, 100, 120])
    )
    assert np.array_equal(
        np.array(ax.get_yticks()),
        np.array([-10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    )

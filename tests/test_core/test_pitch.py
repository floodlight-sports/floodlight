import matplotlib.axes
import matplotlib.pyplot as plt
import pytest

from floodlight.core.pitch import Pitch


# Test def from_template classmethod for all templates
@pytest.mark.unit
def test_template_dfl() -> None:
    # Arrange
    pitch = Pitch.from_template("dfl", length=110, width=68)

    # Assert
    with pytest.raises(TypeError):
        Pitch.from_template("dfl")
    assert pitch.xlim == (-55, 55)
    assert pitch.ylim == (-34, 34)
    assert pitch.unit == "m"
    assert pitch.boundaries == "flexible"


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
def test_template_statsperform_open() -> None:
    # Arrange
    pitch = Pitch.from_template("statsperform_open", length=110, width=68)

    # Assert
    with pytest.raises(TypeError):
        Pitch.from_template("statsperform_open")
    assert pitch.xlim == (-55, 55)
    assert pitch.ylim == (-34, 34)
    assert pitch.unit == "m"
    assert pitch.boundaries == "flexible"


@pytest.mark.unit
def test_template_statsperform_event() -> None:
    # Arrange
    pitch = Pitch.from_template("statsperform_event", length=110, width=68)

    # Assert
    with pytest.raises(TypeError):
        Pitch.from_template("statsperform_event")
    assert pitch.xlim == (-5500, 5500)
    assert pitch.ylim == (-3400, 3400)
    assert pitch.unit == "cm"
    assert pitch.boundaries == "flexible"


@pytest.mark.unit
def test_template_statsperform_tracking() -> None:
    # Arrange
    pitch = Pitch.from_template("statsperform_tracking", length=110, width=68)

    # Assert
    with pytest.raises(TypeError):
        Pitch.from_template("statsperform_tracking")
    assert pitch.xlim == (0, 110)
    assert pitch.ylim == (0, 68)
    assert pitch.unit == "m"
    assert pitch.boundaries == "flexible"


@pytest.mark.unit
def test_template_tracab() -> None:
    # Arrange
    pitch = Pitch.from_template("tracab", length=110, width=68)

    # Assert
    with pytest.raises(TypeError):
        Pitch.from_template("tracab")
    assert pitch.xlim == (-5500, 5500)
    assert pitch.ylim == (-3400, 3400)
    assert pitch.unit == "cm"
    assert pitch.boundaries == "flexible"


@pytest.mark.unit
def test_template_eigd() -> None:
    # Arrange
    pitch = Pitch.from_template("eigd")

    # Assert
    assert pitch.xlim == (0, 40)
    assert pitch.ylim == (0, 20)
    assert pitch.unit == "m"
    assert pitch.boundaries == "fixed"
    assert pitch.sport == "handball"


@pytest.mark.unit
def test_template_statsbomb() -> None:
    # Arrange
    pitch = Pitch.from_template("statsbomb")

    # Assert
    assert pitch.xlim == (0.0, 120.0)
    assert pitch.ylim == (0.0, 80.0)
    assert pitch.unit == "normed"
    assert pitch.boundaries == "flexible"
    assert pitch.sport == "football"


@pytest.mark.unit
def test_template_unknown_template_name() -> None:
    with pytest.raises(ValueError):
        Pitch.from_template("this is not a template name")


# Test def is_metrical(self) property
@pytest.mark.unit
def test_is_metrical_property() -> None:
    pitch1 = Pitch.from_template("opta", length=110, width=68)
    pitch2 = Pitch.from_template("tracab", length=110, width=68)

    assert pitch1.is_metrical is False
    assert pitch2.is_metrical is True


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


# Test def plot(
#   self,
#   color_scheme: str = "standard",
#   show_axis_ticks: bool = False,
#   ax: plt.axes = None,
#   **kwargs)

# Test return
@pytest.mark.plot
def test_plot_football_return_matplotlib_axes_without_given_as_argument(
    example_pitch_football,
) -> None:
    # Act
    ax = example_pitch_football.plot()
    # Assert
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


@pytest.mark.plot
def test_plot_handball_pitch_return_matplotlib_axes_without_given_as_argument(
    example_pitch_handball,
) -> None:
    # Act
    ax = example_pitch_handball.plot()
    # Assert
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


@pytest.mark.plot
def test_plot_football_pitch_return_matplotlib_axes_with_axes_given_as_argument(
    example_pitch_football,
) -> None:
    # Arrange
    axes = plt.subplots()[1]
    # Act
    ax = example_pitch_football.plot(ax=axes)
    # Assert
    assert ax == axes
    plt.close()


@pytest.mark.plot
def test_plot_handball_pitch_return_matplotlib_axes_with_axes_given_as_argument(
    example_pitch_handball,
) -> None:
    # Arrange
    axes = plt.subplots()[1]
    # Act
    ax = example_pitch_handball.plot(ax=axes)
    # Assert
    assert ax == axes
    plt.close()


# Test value error if wrong sport is given
@pytest.mark.plot
def test_plot_value_error_unvalid_sport() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 105), ylim=(0, 68), unit="m", boundaries="fixed", sport="NoSport"
    )
    # Assert
    with pytest.raises(ValueError):
        pitch.plot()
    plt.close()


# Test value error if wrong color_scheme is given
@pytest.mark.plot
def test_plot_value_error_unvalid_color_scheme(example_pitch_football) -> None:
    # Assert
    with pytest.raises(ValueError):
        example_pitch_football.plot(color_scheme="No valid color scheme")
    plt.close()


# Test aspect ratio for given sports and unit (and width/length)
@pytest.mark.plot
def test_plot_football_pitch_aspect_ratio_unit_m() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 105), ylim=(0, 68), unit="m", boundaries="fixed", sport="football"
    )
    # Act
    ax = pitch.plot()
    # Assert
    assert ax.get_aspect() == 1.0
    plt.close()


@pytest.mark.plot
def test_plot_handball_pitch_aspect_ratio_unit_m() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 40), ylim=(0, 20), unit="m", boundaries="fixed", sport="handball"
    )
    # Act
    ax = pitch.plot()
    # Assert
    assert ax.get_aspect() == 1.0
    plt.close()


@pytest.mark.plot
def test_plot_football_pitch_aspect_ratio_unit_cm() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 10500), ylim=(0, 6800), unit="cm", boundaries="fixed", sport="football"
    )
    # Act
    ax = pitch.plot()
    # Assert
    assert ax.get_aspect() == 1.0
    plt.close()


@pytest.mark.plot
def test_plot_handball_pitch_aspect_ratio_unit_cm() -> None:
    # Arrange
    pitch = Pitch(
        xlim=(0, 4000), ylim=(0, 2000), unit="cm", boundaries="fixed", sport="handball"
    )
    # Act
    ax = pitch.plot()
    # Assert
    assert ax.get_aspect() == 1.0
    plt.close()


@pytest.mark.plot
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
    plt.close()


@pytest.mark.plot
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
    plt.close()


@pytest.mark.plot
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
    plt.close()

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
def test_template_statsperform() -> None:
    # Arrange
    pitch = Pitch.from_template("statsperform", length=11000, width=6800)

    # Assert
    with pytest.raises(TypeError):
        Pitch.from_template("statsperform")
    assert pitch.xlim == (-5500, 5500)
    assert pitch.ylim == (-3400, 3400)
    assert pitch.unit == "cm"
    assert pitch.boundaries == "flexible"


@pytest.mark.unit
def test_template_tracab() -> None:
    # Arrange
    pitch = Pitch.from_template("tracab", length=110, width=68)

    # Assert
    with pytest.raises(TypeError):
        Pitch.from_template("tracab")
    assert pitch.xlim == (-55, 55)
    assert pitch.ylim == (-34, 34)
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

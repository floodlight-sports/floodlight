import pytest


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

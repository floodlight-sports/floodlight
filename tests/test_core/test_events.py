import pandas as pd
import pytest

from floodlight.core.events import Events


@pytest.mark.unit
def test_events_getter(example_events_data_minimal: pd.DataFrame) -> None:
    # Arrange
    data = Events(example_events_data_minimal)

    # Act
    eIDs = data["eID"]
    gameclocks = data["gameclock"]

    # Assert
    assert pd.Series.equals(eIDs, pd.Series([1, 2])) and pd.Series.equals(
        gameclocks, pd.Series([1.1, 2.2])
    )


@pytest.mark.unit
def test_events_setter(example_events_data_minimal: pd.DataFrame) -> None:
    # Arrange
    data = Events(example_events_data_minimal)

    # Act
    data["eID"] = ["1", "2"]
    data["gameclock"].at[1] = 3.3

    # Assert
    assert pd.Series.equals(data["eID"], pd.Series(["1", "2"])) and pd.Series.equals(
        data["gameclock"], pd.Series([1.1, 3.3])
    )


@pytest.mark.unit
def test_add_frameclock(example_events_data_minimal: pd.DataFrame) -> None:
    # Arrange
    data = Events(example_events_data_minimal)
    framerate = 25

    # Act
    data.add_frameclock(framerate)

    # Assert
    assert data["frameclock"].at[0] == 28 and data["frameclock"].at[1] == 55


@pytest.mark.unit
def test_find_values(example_events_data_with_outcome_and_none: pd.DataFrame) -> None:
    # Arrange
    data = Events(example_events_data_with_outcome_and_none)

    # Act
    outcome_one = data.find("outcome", 1)
    outcome_zero = data.find("outcome", 0)
    outcome_none = data.find("outcome", None)

    # Assert
    assert len(outcome_one) == 1 and len(outcome_zero) == 2 and len(outcome_none) == 2

import pandas as pd
import pytest
import numpy as np

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
def test_column_properties() -> None:
    # Arrange df with different columns
    columns = ["eID", "at_x", "to_x", "my_col"]
    df = pd.DataFrame(columns=columns)

    # trigger post_init
    with pytest.raises(ValueError):
        events = Events(events=df)

    # add missing column
    columns.append("gameclock")
    df = pd.DataFrame(columns=columns)
    events = Events(events=df)

    # Assert column properties
    assert events.essential == ["eID", "gameclock"]
    assert events.protected == ["at_x", "to_x"]
    assert events.custom == ["my_col"]
    assert events.essential_missing is None
    assert len(events.protected_missing) > 3


@pytest.mark.unit
def test_add_frameclock(example_events_data_minimal: pd.DataFrame) -> None:
    # Arrange
    data = Events(example_events_data_minimal)
    framerate = 25

    # Act
    data.add_frameclock(framerate)

    # Assert
    assert data["frameclock"].at[0] == 27 and data["frameclock"].at[1] == 55


@pytest.mark.unit
def test_select_single_condition(
    example_events_data_with_outcome_and_none: pd.DataFrame,
) -> None:
    # Arrange
    data = Events(example_events_data_with_outcome_and_none)

    # Act
    outcome_one = data.select([("outcome", 1)])
    outcome_zero = data.select([("outcome", 0)])
    outcome_none = data.select([("outcome", None)])

    # Assert
    assert len(outcome_one) == 1 and len(outcome_zero) == 2 and len(outcome_none) == 2


@pytest.mark.unit
def test_select_multi_condition(
    example_events_data_with_outcome_and_none: pd.DataFrame,
) -> None:
    # Arrange
    data = Events(example_events_data_with_outcome_and_none)

    # Act
    filtered_events = data.select([("eID", 1), ("outcome", None)])

    # Assert
    assert len(filtered_events) == 1


@pytest.mark.unit
def test_translation_function(
    example_events_data_xy,
    example_events_data_xy_none,
    example_events_data_minimal: pd.DataFrame,
) -> None:

    # Arrange
    data = Events(example_events_data_xy)
    data_none = Events(example_events_data_xy_none)
    data_minimal = Events(example_events_data_minimal)
    data_minimal_translated = Events(example_events_data_minimal)

    # Act + Assert
    data.translate((0, -38.6))
    assert pd.DataFrame.equals(
        data.events[["at_x", "at_y"]],
        pd.DataFrame({"at_x": [1, 3], "at_y": [-36.6, -34.6]}),
    )

    data_none.translate((3, 7))
    assert pd.DataFrame.equals(
        data_none.events[["at_x", "at_y"]],
        pd.DataFrame({"at_x": [np.NaN, np.NaN], "at_y": [np.NaN, np.NaN]}),
    )

    data_minimal_translated.translate((1, 2))
    assert data_minimal.events.equals(data_minimal_translated.events)

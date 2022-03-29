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
def test_essential_missing(
    example_events_data_minimal_missing_essential: pd.DataFrame,
) -> None:
    try:
        Events(example_events_data_minimal_missing_essential)
        assert False
    except ValueError:
        assert True


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore: Floodlight Events column")
def test_essential_invalid(
    example_events_data_minimal_invalid_essential: pd.DataFrame,
) -> None:
    # Arrange
    data = Events(example_events_data_minimal_invalid_essential)

    # Act
    invalid_essential_columns = data.essential_invalid

    # Assert
    assert invalid_essential_columns == ["gameclock"]


@pytest.mark.unit
def test_protected_missing(
    example_events_data_minimal: pd.DataFrame,
) -> None:
    # Arrange
    data = Events(example_events_data_minimal)

    # Act
    missing_protected_columns = data.protected_missing

    # Assert
    assert len(missing_protected_columns) == 14


@pytest.mark.unit
def test_protected_invalid(
    example_events_data_invalid_protected: pd.DataFrame,
) -> None:
    # Arrange
    data = Events(example_events_data_invalid_protected)

    # Act
    invalid_protected_columns = data.protected_invalid

    # Assert
    assert invalid_protected_columns == ["jID"]


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
def test_add_frameclock_with_values(example_events_data_minimal: pd.DataFrame) -> None:
    # Arrange
    data = Events(example_events_data_minimal)
    framerate = 25

    # Act
    data.add_frameclock(framerate)

    # Assert
    assert data["frameclock"].at[0] == 27 and data["frameclock"].at[1] == 55


@pytest.mark.unit
def test_add_frameclock_with_none(
    example_events_data_minimal_with_none: pd.DataFrame,
) -> None:
    # Arrange
    data = Events(example_events_data_minimal_with_none)
    framerate = 25

    # Act
    data.add_frameclock(framerate)

    # Assert
    assert data["frameclock"].at[0] == 27 and data["frameclock"].at[1] < 0


@pytest.mark.unit
def test_select_single_condition(
    example_events_data_with_outcome_none,
) -> None:
    # Arrange
    data = Events(example_events_data_with_outcome_none)

    # Act
    outcome_one = data.select([("outcome", 1)])
    outcome_zero = data.select([("outcome", 0)])
    outcome_none = data.select([("outcome", None)])

    # Assert
    assert len(outcome_one) == 1 and len(outcome_zero) == 2 and len(outcome_none) == 2


@pytest.mark.unit
def test_select_multi_condition(
    example_events_data_with_outcome_none,
) -> None:
    # Arrange
    data = Events(example_events_data_with_outcome_none)

    # Act
    filtered_events = data.select([("eID", 1), ("outcome", None)])

    # Assert
    assert len(filtered_events) == 1

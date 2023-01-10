import pandas as pd
import pytest
import numpy as np

from floodlight.core.events import Events
from floodlight.core.definitions import essential_events_columns, protected_columns


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
    assert events.essential_missing == []
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
@pytest.mark.filterwarnings("ignore: The 'gameclock' column")
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
    data = Events(example_events_data_minimal)
    missing_protected_columns = data.protected_missing
    total_num_protected = len(protected_columns)

    assert len(missing_protected_columns) == total_num_protected


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
def test_column_values_in_range(
    example_events_data_invalid_protected: pd.DataFrame,
) -> None:
    # Arrange
    data = Events(example_events_data_invalid_protected)

    # Act
    eID_in_range = data.column_values_in_range("eID", essential_events_columns)
    gameclock_in_range = data.column_values_in_range(
        "gameclock", essential_events_columns
    )
    jID_in_range = data.column_values_in_range("jID", protected_columns)

    # Assert
    assert eID_in_range
    assert gameclock_in_range
    assert not jID_in_range


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
    example_events_data_minimal_none,
) -> None:
    # Arrange
    data = Events(example_events_data_minimal_none)
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


@pytest.mark.unit
def test_scale_function(
    example_events_data_xy,
    example_events_data_xy_none,
    example_events_data_minimal: pd.DataFrame,
) -> None:

    # Arrange
    data = Events(example_events_data_xy)
    data_none = Events(example_events_data_xy_none)
    data_minimal = Events(example_events_data_minimal)
    data_minimal_scaled = Events(example_events_data_minimal)

    # Act + Assert
    data.scale(factor=2)
    assert pd.DataFrame.equals(
        data.events[["at_x", "at_y"]], pd.DataFrame({"at_x": [2, 6], "at_y": [4, 8]})
    )
    data.scale(factor=-2.9, axis="x")
    assert pd.DataFrame.equals(
        data.events[["at_x", "at_y"]],
        pd.DataFrame({"at_x": [-5.8, -17.4], "at_y": [4, 8]}),
    )
    data.scale(factor=0, axis="y")
    assert pd.DataFrame.equals(
        data.events[["at_x", "at_y"]],
        pd.DataFrame({"at_x": [-5.8, -17.4], "at_y": [0, 0]}),
    )
    with pytest.raises(ValueError):
        data.scale(factor=1, axis="z")

    data_none.scale(factor=2)
    assert pd.DataFrame.equals(
        data_none.events[["at_x", "at_y"]],
        pd.DataFrame({"at_x": [np.NaN, np.NaN], "at_y": [np.NaN, np.NaN]}),
    )

    data_minimal_scaled.scale(2)
    assert data_minimal.events.equals(data_minimal_scaled.events)


@pytest.mark.unit
def test_reflect_function(
    example_events_data_xy,
    example_events_data_xy_none,
    example_events_data_minimal: pd.DataFrame,
) -> None:

    # Arrange
    data = Events(example_events_data_xy)
    data_none = Events(example_events_data_xy_none)
    data_minimal = Events(example_events_data_minimal)
    data_minimal_reflected = Events(example_events_data_minimal)

    # Act + Assert
    data.reflect(axis="y")
    assert pd.DataFrame.equals(
        data.events[["at_x", "at_y"]], pd.DataFrame({"at_x": [-1, -3], "at_y": [2, 4]})
    )
    data.reflect(axis="x")
    assert pd.DataFrame.equals(
        data.events[["at_x", "at_y"]],
        pd.DataFrame({"at_x": [-1, -3], "at_y": [-2, -4]}),
    )
    with pytest.raises(ValueError):
        data.reflect(axis="z")

    # Act + Assert
    data_none.reflect(axis="x")
    assert pd.DataFrame.equals(
        data_none.events[["at_x", "at_y"]],
        pd.DataFrame({"at_x": [np.NaN, np.NaN], "at_y": [np.NaN, np.NaN]}),
    )
    data_none.reflect(axis="y")
    assert pd.DataFrame.equals(
        data_none.events[["at_x", "at_y"]],
        pd.DataFrame({"at_x": [np.NaN, np.NaN], "at_y": [np.NaN, np.NaN]}),
    )
    with pytest.raises(ValueError):
        data_none.reflect(axis="z")

    data_minimal_reflected.reflect("x")
    assert data_minimal.events.equals(data_minimal_reflected.events)


@pytest.mark.unit
def test_rotate(
    example_events_data_xy,
    example_events_data_xy_none,
    example_events_data_minimal: pd.DataFrame,
) -> None:

    # Arrange
    data = Events(example_events_data_xy)
    data_none = Events(example_events_data_xy_none)
    data_minimal = Events(example_events_data_minimal)
    data_minimal_rotated = Events(example_events_data_minimal)

    # Act + Assert
    data.rotate(90)
    assert pd.DataFrame.equals(
        data.events[["at_x", "at_y"]],
        pd.DataFrame({"at_x": [-2.0, -4.0], "at_y": [1.0, 3.0]}),
    )

    data.rotate(-90)
    assert pd.DataFrame.equals(
        data.events[["at_x", "at_y"]],
        pd.DataFrame({"at_x": [1.0, 3.0], "at_y": [2.0, 4.0]}),
    )

    data_none.rotate(90)
    assert pd.DataFrame.equals(
        data_none.events[["at_x", "at_y"]],
        pd.DataFrame({"at_x": [np.NaN, np.NaN], "at_y": [np.NaN, np.NaN]}),
    )

    data_minimal_rotated.rotate(0)
    assert data_minimal.events.equals(data_minimal_rotated.events)


# tests for slice with inplace=False
@pytest.mark.unit
def test_slice_new_object(
    example_events_data_minimal,
) -> None:

    # Arrange
    data_minimal = Events(example_events_data_minimal)

    # Act
    data_minimal_empty = data_minimal.slice(start=0.1, end=0.9, inplace=False)
    data_minimal_full = data_minimal.slice(inplace=False)
    data_minimal_full_arguments = data_minimal.slice(start=1.1, end=2.3, inplace=False)
    data_minimal_end_sliced = data_minimal.slice(end=2.2, inplace=False)
    data_minimal_start_sliced = data_minimal.slice(start=1.2, inplace=False)

    # Assert
    full_events = pd.DataFrame(
        {
            "eID": [1, 2],
            "gameclock": [1.1, 2.2],
        }
    )
    start_events = pd.DataFrame(
        {
            "eID": [1],
            "gameclock": [1.1],
        }
    )
    end_events = pd.DataFrame(
        {
            "eID": [2],
            "gameclock": [2.2],
        }
    )
    assert len(data_minimal_empty.events) == 0
    assert all(data_minimal_full.events == full_events)
    assert all(data_minimal_full_arguments.events == full_events)
    assert all(data_minimal_end_sliced.events == start_events)
    assert all(data_minimal_start_sliced.events == end_events)


# tests for slice with inplace=True
@pytest.mark.unit
def test_slice_inplace(
    example_events_data_minimal,
    example_events_data_frameclock,
) -> None:

    # Arrange
    data_minimal_empty = Events(example_events_data_minimal)
    data_minimal_end = Events(example_events_data_frameclock)

    # Act
    data_minimal_empty.slice(start=2.5, inplace=True)
    data_minimal_end.slice(start=0.15, inplace=True)

    # Assert
    frameclock_end_events = pd.DataFrame(
        {
            "eID": [2],
            "gameclock": [0.2],
            "frameclock": [16.7],
        }
    )
    assert len(data_minimal_empty.events) == 0
    assert all(data_minimal_end.events == frameclock_end_events)


# tests slice using the frameclock column
@pytest.mark.unit
def test_slice_frameclock(
    example_events_data_frameclock,
):
    # Arrange
    data_frameclock = Events(example_events_data_frameclock)

    # Act
    data_frameclock_sliced = data_frameclock.slice(
        end=15, slice_by="frameclock", inplace=False
    )

    # Assert
    frameclock_start_events = pd.DataFrame(
        {
            "eID": [1],
            "gameclock": [0.1],
            "frameclock": [12.4],
        }
    )
    assert all(data_frameclock_sliced.events == frameclock_start_events)


# tests slice for objects containing None
@pytest.mark.unit
def test_slice_none(
    example_events_data_minimal_none,
):
    # Arrange
    data_none = Events(example_events_data_minimal_none)

    # Act
    data_none_sliced = data_none.slice(start=0, end=10, inplace=False)
    data_none.slice(start=1.1, end=1.2, inplace=True)

    # Assert
    data_none_events = pd.DataFrame(
        {
            "eID": [None],
            "gameclock": [1.1],
        }
    )
    assert all(data_none_sliced.events == data_none_events)
    assert all(data_none.events == data_none_events)


# tests get_event_stream with and without frameclock (latter should fail)
@pytest.mark.unit
def test_get_event_stream_frameclock(
    example_events_data_minimal,
    example_events_data_frameclock,
):
    # Arrange
    data_minimal = Events(example_events_data_minimal)
    data_frameclock = Events(example_events_data_frameclock)

    # Act
    with pytest.raises(ValueError):
        data_minimal.get_event_stream()

    event_stream_frameclock = data_frameclock.get_event_stream()
    # replace all NaNs to simplify comparison
    event_stream_frameclock[np.isnan(event_stream_frameclock)] = -1

    # Assert
    assert all(
        np.equal(
            event_stream_frameclock.code,
            np.array([1, -1, -1, -1, -1, 2]),
        )
    )


# tests that get_event_stream excludes entries containing None
@pytest.mark.unit
def test_get_event_stream_frameclock_none(
    example_events_data_frameclock_none,
):
    # Arrange
    data_frameclock_none = Events(example_events_data_frameclock_none)

    # Act
    event_stream_frameclock_none = data_frameclock_none.get_event_stream()

    # Assert
    assert all(event_stream_frameclock_none == np.array(["2"]))


# tests get_event_stream for objects where frameclock values are non-ascending
@pytest.mark.unit
def test_get_event_stream_frameclock_unsorted(
    example_events_data_frameclock_unsorted,
):
    # Arrange
    data_frameclock_unsorted = Events(example_events_data_frameclock_unsorted)

    # Act
    event_stream_unsorted = data_frameclock_unsorted.get_event_stream(
        name="possession",
    )
    # replace all NaNs to simplify comparison
    event_stream_unsorted[np.isnan(event_stream_unsorted)] = -1

    # Assert
    target_values = np.array([2, -1, -1, -1, -1, 3, -1, -1, -1, -1, 1])
    assert all(np.equal(event_stream_unsorted.code, target_values))


# tests get_event_stream for different fade arguments
@pytest.mark.unit
def test_get_event_stream_fade(
    example_events_data_frameclock,
):
    # Arrange
    data_frameclock = Events(example_events_data_frameclock)

    # Act
    event_stream_none = data_frameclock.get_event_stream(fade=None)
    event_stream_zero = data_frameclock.get_event_stream(fade=0)
    event_stream_one = data_frameclock.get_event_stream(fade=1)
    event_stream_ten = data_frameclock.get_event_stream(fade=10)
    with pytest.raises(ValueError):
        data_frameclock.get_event_stream(fade=-5)

    # replace all NaNs to simplify comparison
    event_stream_none[np.isnan(event_stream_none)] = -1
    event_stream_zero[np.isnan(event_stream_zero)] = -1
    event_stream_one[np.isnan(event_stream_one)] = -1
    event_stream_ten[np.isnan(event_stream_ten)] = -1

    # Assert
    assert all(np.equal(event_stream_none.code, np.array([1, 1, 1, 1, 1, 2])))
    assert all(np.equal(event_stream_zero.code, np.array([1, -1, -1, -1, -1, 2])))
    assert all(np.equal(event_stream_one.code, np.array([1, 1, -1, -1, -1, 2])))
    assert all(np.equal(event_stream_ten.code, np.array([1, 1, 1, 1, 1, 2])))

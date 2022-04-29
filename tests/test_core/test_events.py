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


@pytest.mark.unit
def test_slice(
    example_events_data_minimal,
    example_events_data_minimal_none,
    example_events_data_frameclock,
) -> None:

    # Arrange
    data_minimal = Events(example_events_data_minimal)
    data_none = Events(example_events_data_minimal_none)
    data_frameclock = Events(example_events_data_frameclock)

    # Act
    data_minimal_full = data_minimal.slice(inplace=False)
    data_minimal_full_arguments = data_minimal.slice(start=1.1, end=2.3, inplace=False)
    data_minimal_end_sliced = data_minimal.slice(end=2.2, inplace=False)
    data_minimal_start_sliced = data_minimal.slice(start=1.2, inplace=False)
    data_none_sliced = data_none.slice(inplace=False)
    data_frameclock_sliced = data_frameclock.slice(
        end=15, use_frameclock=True, inplace=False
    )
    data_minimal.slice(start=10, inplace=True)
    data_none.slice(inplace=True)
    data_frameclock.slice(start=15, use_frameclock=True, inplace=True)

    # Assert
    assert data_minimal_full.events.shape == (2, 2)
    assert data_minimal_full_arguments.events.shape == (2, 2)
    assert data_minimal_end_sliced.events.shape == (1, 2)
    assert data_minimal_start_sliced.events.shape == (1, 2)
    assert data_none_sliced.events.shape == (1, 2)
    assert data_frameclock_sliced.events.shape == (1, 3)
    assert data_minimal.events.shape == (0, 2)
    assert data_none.events.shape == (1, 2)
    assert data_frameclock.events.shape == (1, 3)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore: Cannot estimate playing direction for an events")
@pytest.mark.filterwarnings("ignore: Need at least one goalkeeper and one shot event")
@pytest.mark.filterwarnings("ignore: Estimated playing direction is")
def test_estimate_playing_direction(
    example_events_data_minimal,
    example_events_data_xy_long,
    example_events_data_xy_none_long,
    example_football_pitch,
):
    # Arrange
    data_minimal = Events(example_events_data_minimal)
    data_xy_long = Events(example_events_data_xy_long)
    data_xy_none_long = Events(example_events_data_xy_none_long)
    links_lr = {
        0: "GoalkeeperAction",
        6: "goalkeeper",
        8: "event-KICK-goalkeeper",
        3: "undefined",
        12: "ShotOnTarget",
        125: "shotdeflected",
    }
    links_rl = {
        0: "ShotOffTarget",
        6: "leftLegShot",
        8: "right-legged-shot",
        3: "undefined",
        12: "goalkeeper_save",
        125: "Kick(GoalKeeper)",
    }

    # Act
    try:
        direction_minimal = data_minimal.estimate_playing_direction(
            pitch=example_football_pitch
        )
    except ValueError:
        direction_minimal = None

    direction_no_links = data_xy_long.estimate_playing_direction(
        pitch=example_football_pitch
    )
    direction_xy_long_lr = data_xy_long.estimate_playing_direction(
        pitch=example_football_pitch, links_eID_to_event_names=links_lr
    )
    direction_xy_long_rl = data_xy_long.estimate_playing_direction(
        pitch=example_football_pitch, links_eID_to_event_names=links_rl
    )
    direction_xy_none_long_lr = data_xy_none_long.estimate_playing_direction(
        example_football_pitch, links_eID_to_event_names=links_lr
    )
    direction_xy_none_long_rl = data_xy_none_long.estimate_playing_direction(
        example_football_pitch, links_eID_to_event_names=links_rl
    )

    # Assert
    assert direction_minimal is None
    assert direction_no_links is None
    assert direction_xy_long_lr == "lr"
    assert direction_xy_long_rl == "rl"
    assert direction_xy_none_long_lr == "lr"
    assert direction_xy_none_long_rl == "rl"


@pytest.mark.unit
def test_get_event_stream(
    example_events_data_minimal,
    example_events_data_frameclock,
    example_events_data_frameclock_none,
    example_events_data_frameclock_unsorted,
    example_events_data_frameclock_long,
):
    # Arrange
    data_minimal = Events(example_events_data_minimal)
    data_frameclock = Events(example_events_data_frameclock)
    data_frameclock_none = Events(example_events_data_frameclock_none)
    data_frameclock_unsorted = Events(example_events_data_frameclock_unsorted)
    data_frameclock_long = Events(example_events_data_frameclock_long)
    links_long = {0: "A", 2: "A", 12: "A", 1: "H", 17: "H", 23: "H"}

    # Act
    try:
        event_stream_minimal = data_minimal.get_event_stream()
    except ValueError:
        event_stream_minimal = None
    event_stream_frameclock = data_frameclock.get_event_stream(start=5, end=17)
    event_stream_frameclock_none = data_frameclock_none.get_event_stream()
    event_stream_frameclock_unsorted = data_frameclock_unsorted.get_event_stream(
        fade=3, placeholder=-1
    )
    event_stream_frameclock_long = data_frameclock_long.get_event_stream(
        start=0,
        end=500,
        fade=100,
        links_eID_to_code=links_long,
        name="possession",
        definitions={"H": "Home", "A": "Away"},
    )

    # Assert
    assert event_stream_minimal is None
    assert len(event_stream_frameclock) == 12
    assert np.nansum(event_stream_frameclock[:7]) == 0
    assert np.sum(event_stream_frameclock[8:]) == 4
    assert len(event_stream_frameclock_none) == 18
    assert np.nansum(event_stream_frameclock_none[:17]) == 0
    assert event_stream_frameclock_none[17] == 2
    assert len(event_stream_frameclock_unsorted) == 27
    assert np.sum(event_stream_frameclock_unsorted[:12]) == 12 * -1
    assert np.sum(event_stream_frameclock_unsorted[12:16]) == 4 * 2
    assert event_stream_frameclock_unsorted[16] == -1
    assert np.sum(event_stream_frameclock_unsorted[17:21]) == 4 * 3
    assert np.sum(event_stream_frameclock_unsorted[21:]) == -1 * 5 + 1 * 1
    assert len(event_stream_frameclock_long) == 500
    assert np.all(event_stream_frameclock_long[:25] == "A")
    assert np.all(event_stream_frameclock_long[25:50] == "H")
    assert np.all(event_stream_frameclock_long[50:251] == "A")
    assert np.nansum(event_stream_frameclock_long[251:375]) == 0
    assert np.all(event_stream_frameclock_long[375:425] == "A")
    assert np.all(event_stream_frameclock_long[425:] == "H")
    assert event_stream_frameclock_long.name == "possession"
    assert event_stream_frameclock_long.definitions["H"] == "Home"

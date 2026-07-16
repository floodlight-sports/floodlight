import ast
import json

import pandas as pd
import pytest

from floodlight.core.teamsheet import Teamsheet
from floodlight.io.statsbomb import read_open_event_data_json


@pytest.mark.unit
def test_statsbomb_read_events_path_not_exists(filepath_empty) -> None:

    with pytest.raises(FileNotFoundError):
        read_open_event_data_json(
            filepath_events=filepath_empty, filepath_match=filepath_empty
        )


@pytest.mark.unit
def test_statsbomb_attributes_events_to_acting_team(tmp_path) -> None:
    pass_id = "00000000-0000-4000-8000-000000000001"
    pressure_id = "00000000-0000-4000-8000-000000000002"
    interception_id = "00000000-0000-4000-8000-000000000003"
    source_events = [
        {
            "id": pass_id,
            "period": 1,
            "timestamp": "00:00:01.250",
            "minute": 0,
            "second": 1,
            "type": {"id": 30, "name": "Pass"},
            "possession": 1,
            "possession_team": {"id": 101, "name": "Team A"},
            "play_pattern": {"id": 1, "name": "Regular Play"},
            "team": {"id": 101, "name": "Team A"},
            "player": {"id": 1001, "name": "Player A"},
            "location": [30.0, 40.0],
            "pass": {
                "end_location": [45.0, 42.0],
                "outcome": {"id": 1, "name": "Complete"},
            },
        },
        {
            "id": pressure_id,
            "period": 1,
            "timestamp": "00:00:02.500",
            "minute": 0,
            "second": 2,
            "type": {"id": 17, "name": "Pressure"},
            "possession": 1,
            "possession_team": {"id": 101, "name": "Team A"},
            "play_pattern": {"id": 1, "name": "Regular Play"},
            "team": {"id": 202, "name": "Team B"},
            "player": {"id": 2001, "name": "Player B"},
            "location": [46.0, 42.5],
            "duration": 0.4,
            "under_pressure": False,
        },
        {
            "id": interception_id,
            "period": 1,
            "timestamp": "00:00:03.750",
            "minute": 0,
            "second": 3,
            "type": {"id": 10, "name": "Interception"},
            "possession": 1,
            "possession_team": {"id": 101, "name": "Team A"},
            "play_pattern": {"id": 1, "name": "Regular Play"},
            "team": {"id": 202, "name": "Team B"},
            "player": {"id": 2001, "name": "Player B"},
            "location": [44.0, 41.0],
            "interception": {"outcome": {"id": 1, "name": "Won"}},
        },
    ]
    threesixty = [
        {
            "event_uuid": pressure_id,
            "freeze_frame": [
                {
                    "teammate": True,
                    "actor": True,
                    "keeper": False,
                    "location": [46.0, 42.5],
                }
            ],
            "visible_area": [0.0, 0.0, 120.0, 0.0, 120.0, 80.0, 0.0, 80.0],
        }
    ]
    events_path = tmp_path / "900001.json"
    threesixty_path = tmp_path / "900001-360.json"
    events_path.write_text(json.dumps(source_events), encoding="utf8")
    threesixty_path.write_text(json.dumps(threesixty), encoding="utf8")
    teamsheet_home = Teamsheet(pd.DataFrame({"player": ["Player A"], "tID": [101]}))
    teamsheet_away = Teamsheet(pd.DataFrame({"player": ["Player B"], "tID": [202]}))

    events_objects, teamsheets = read_open_event_data_json(
        str(events_path),
        str(tmp_path / "unused.json"),
        str(threesixty_path),
        teamsheet_home,
        teamsheet_away,
    )
    home_events = events_objects["HT1"]["Home"].events
    away_events = events_objects["HT1"]["Away"].events

    def rows_by_source_id(events):
        rows = {}
        for _, row in events.iterrows():
            qualifier = ast.literal_eval(row["qualifier"])
            rows[qualifier["unique_identifier"]] = (row, qualifier)
        return rows

    home_rows = rows_by_source_id(home_events)
    away_rows = rows_by_source_id(away_events)

    assert pass_id in home_rows
    assert pass_id not in away_rows
    assert home_rows[pass_id][0]["tID"] == 101
    assert (
        pressure_id in away_rows
    ), "Team B's defensive event must be stored in Team B's Events object"
    assert pressure_id not in home_rows
    assert away_rows[pressure_id][0]["tID"] == 202
    assert interception_id in away_rows
    assert interception_id not in home_rows
    assert away_rows[interception_id][0]["tID"] == 202

    assert set(events_objects) == {"HT1"}
    assert set(events_objects["HT1"]) == {"Home", "Away"}
    assert teamsheets == {"Home": teamsheet_home, "Away": teamsheet_away}
    assert list(home_rows) == [pass_id]
    assert list(away_rows) == [pressure_id, interception_id]
    assert len(home_events) + len(away_events) == len(source_events) == 3
    assert json.loads(events_path.read_text(encoding="utf8")) == source_events

    pass_row, pass_qualifier = home_rows[pass_id]
    pressure_row, pressure_qualifier = away_rows[pressure_id]
    interception_row, interception_qualifier = away_rows[interception_id]
    assert [pass_row["eID"], pressure_row["eID"], interception_row["eID"]] == [
        30,
        17,
        10,
    ]
    assert [
        pass_row["event_name"],
        pressure_row["event_name"],
        interception_row["event_name"],
    ] == ["Pass", "Pressure", "Interception"]
    assert [
        pass_row["gameclock"],
        pressure_row["gameclock"],
        interception_row["gameclock"],
    ] == [1.25, 2.5, 3.75]
    assert [
        pass_row["timestamp"],
        pressure_row["timestamp"],
        interception_row["timestamp"],
    ] == ["00:00:01.250", "00:00:02.500", "00:00:03.750"]
    assert [
        (pass_row["minute"], pass_row["second"]),
        (pressure_row["minute"], pressure_row["second"]),
        (interception_row["minute"], interception_row["second"]),
    ] == [(0, 1), (0, 2), (0, 3)]
    assert [pass_row["pID"], pressure_row["pID"], interception_row["pID"]] == [
        1001,
        2001,
        2001,
    ]
    assert [pass_row["mID"], pressure_row["mID"], interception_row["mID"]] == [
        900001,
        900001,
        900001,
    ]
    assert pass_row["outcome"] == 1
    assert pd.isna(pressure_row["outcome"])
    assert interception_row["outcome"] == 1
    assert [pass_row["at_x"], pass_row["at_y"]] == [30.0, 40.0]
    assert [pass_row["to_x"], pass_row["to_y"]] == [45.0, 42.0]
    assert [pressure_row["at_x"], pressure_row["at_y"]] == [46.0, 42.5]
    assert pd.isna(pressure_row["to_x"])
    assert pd.isna(pressure_row["to_y"])
    assert "frameclock" not in home_events
    assert "frameclock" not in away_events
    for qualifier in (pass_qualifier, pressure_qualifier, interception_qualifier):
        assert qualifier["possession"] == 1
        assert qualifier["possession_team"] == {"id": 101, "name": "Team A"}
        assert qualifier["play_pattern"] == {"id": 1, "name": "Regular Play"}
    assert pass_qualifier["pass"] == source_events[0]["pass"]
    assert pressure_qualifier["duration"] == 0.4
    assert pressure_qualifier["under_pressure"] is False
    assert interception_qualifier["interception"] == source_events[2]["interception"]
    assert pressure_qualifier["360_freeze_frame"] == threesixty[0]["freeze_frame"]
    assert pressure_qualifier["360_visible_area"] == threesixty[0]["visible_area"]

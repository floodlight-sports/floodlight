import warnings
from pathlib import Path
from typing import Tuple, Union

import os
import json
import pandas as pd

from floodlight.core.events import Events


def read_open_statsbomb_event_data_json(
    filepath_events: Union[str, Path],
    filepath_match: Union[str, Path],
    filepath_threesixty: Union[str, Path] = None,
) -> Tuple[Events, Events, Events, Events]:
    """Parses a StatsPerform Match Event CSV file and extracts the event data.

    This function provides a high-level access to an events json file from the openly
    published StatsBomb open data and returns Event objects for both teams for the first
    two periods. A StatsBomb360 json file can be passed to the function to include
    information about the tracked position of (some) players at certain events to the
    ``qualifier`` column.

    Parameters
    ----------
    filepath_events: str or pathlib.Path
        Full path to json file where the Event data is saved.
    filepath_match: str or pathlib.Path
        Full path to json file where information about all matches of a competition are
         stored.
    filepath_threesixty: str or pathlib.Path, optional
        Full path to json file where the StatsBomb360 data in is saved if available. The
        information about the area of the field where player positions are tracked
        (``visible_area``) and player positions at single events (``freeze frame``) are
        stored as a string in the ``qualifier`` column.

    Returns
    -------
    data_objects: Tuple[Events, Events, Events, Events]
        Events- and Pitch-objects for both teams and both halves. The order is
        (home_ht1, home_ht2, away_ht1, away_ht2).

    Notes
    -----
    StatsBomb's open format of handling provides certain additional event attributes,
    which attach additional information to certain events. As of now, these information
    are parsed as a string in the ``qualifier`` column of the returned DataFrame and can
    be transformed to a dict of form ``{attribute: value}``.
    """

    # load json files into memory
    with open(filepath_match, "r", encoding="utf8") as f:
        matchinfo_list = json.load(f)
    with open(filepath_events, "r", encoding="utf8") as f:
        file_event_list = json.load(f)
    if filepath_threesixty is not None:
        with open(filepath_threesixty, "r", encoding="utf8") as f:
            file_threesixty_list = json.load(f)
    else:
        file_threesixty_list = None

    # 1. retrieve match info from file
    match_id = int(filepath_events.split(os.path.sep)[-1][:-5])  # from filepath
    matchinfo = None
    for info in matchinfo_list:
        if info["match_id"] == match_id:
            matchinfo = info
            break

    # 2. parse match info
    teams = ["Home", "Away"]
    tID_link = {
        int(matchinfo["home_team"]["home_team_id"]): "Home",
        int(matchinfo["away_team"]["away_team_id"]): "Away",
    }
    periods = set([event["period"] for event in file_event_list])
    segments = [f"HT{period}" for period in periods]

    # 3. parse events
    # bins TODO: to_x/y, tID, pID name or ID, outcome, type
    columns = [
        "eID",
        "gameclock",
        "pID",
        "outcome",
        "timestamp",
        "minute",
        "second",
        "at_x",
        "at_y",
        "qualifier",
    ]

    team_event_lists = {
        team: {segment: {col: [] for col in columns} for segment in segments}
        for team in teams
    }

    # loop
    for event in file_event_list:
        # get team and segment information
        period = event["period"]
        segment = "HT" + str(period)
        team = tID_link[event["possession_team"]["id"]]

        # identifier and outcome:
        eID = event["id"]
        pID = event["player"] if "player" in event else None
        if "type" in event:
            outcome = event["type"]["outcome"] if "outcome" in event["type"] else None
        else:
            outcome = None
        team_event_lists[team][segment]["eID"].append(eID)
        team_event_lists[team][segment]["pID"].append(pID)
        team_event_lists[team][segment]["outcome"].append(outcome)

        # relative time
        timestamp = event["timestamp"]
        minute = event["minute"]
        second = event["second"]
        gameclock = 60 * minute + second  # in seconds
        team_event_lists[team][segment]["timestamp"].append(timestamp)
        team_event_lists[team][segment]["minute"].append(minute)
        team_event_lists[team][segment]["second"].append(second)
        team_event_lists[team][segment]["gameclock"].append(gameclock)

        # location
        at_x = event["location"][0] if "location" in event else None
        at_y = event["location"][1] if "location" in event else None
        team_event_lists[team][segment]["at_x"].append(at_x)
        team_event_lists[team][segment]["at_y"].append(at_y)

        # qualifier
        qual_dict = {}
        for qualifier in event:
            if qualifier in [
                "id",
                "period",
                "timestamp",
                "minute",
                "second",
                "location",
            ]:
                continue

            qual_value = event[qualifier]
            qual_dict[qualifier] = qual_value
        if file_threesixty_list is not None:
            threesixty_event = [
                event for event in file_threesixty_list if event["event_uuid"] == eID
            ]
            if len(threesixty_event) == 1:
                qual_dict["360_freeze_frame"] = threesixty_event[0]["freeze_frame"]
                qual_dict["360_visible_area"] = threesixty_event[0]["visible_area"]
            elif len(threesixty_event) >= 1:
                warnings.warn(
                    f"Found ambiguous StatsBomb event ID {eID} matching to more than "
                    f"one StatsBomb360 event."
                )
        team_event_lists[team][segment]["qualifier"].append(str(qual_dict))

    # assembly
    home_ht1 = Events(
        events=pd.DataFrame(data=team_event_lists["Home"]["HT1"]),
    )
    home_ht2 = Events(
        events=pd.DataFrame(data=team_event_lists["Home"]["HT2"]),
    )
    away_ht1 = Events(
        events=pd.DataFrame(data=team_event_lists["Away"]["HT1"]),
    )
    away_ht2 = Events(
        events=pd.DataFrame(data=team_event_lists["Away"]["HT2"]),
    )

    data_objects = (home_ht1, home_ht2, away_ht1, away_ht2)

    return data_objects

import warnings
from pathlib import Path
from typing import Tuple, Union, Dict

import os
import json
import pandas as pd

from floodlight.core.events import Events
from floodlight.core.teamsheet import Teamsheet


def read_teamsheets_from_open_event_data_json(
    filepath_events: Union[str, Path],
    filepath_match: Union[str, Path],
) -> Dict[str, Teamsheet]:
    """Reads open events and match files and returns Teamsheet objects for the home
    and the away team.

    Parameters
    ----------
    filepath_events: str or pathlib.Path
        Full path to json file where the Event data is saved.
    filepath_match: str or pathlib.Path
        Full path to json file where information about all matches of a season are
        stored.

    Returns
    -------
    teamsheets: Dict[str, Teamsheet]
        Dictionary with teamsheets for the home team and the away team.
    """
    # load json file into memory
    with open(str(filepath_match), "r", encoding="utf8") as f:
        matchinfo_list = json.load(f)
    with open(str(filepath_events), "r", encoding="utf8") as f:
        file_event_list = json.load(f)

    # retrieve match info from file
    mID = int(str(filepath_events).split(os.path.sep)[-1][:-5])  # from filepath
    matchinfo = None
    for info in matchinfo_list:
        if info["match_id"] == mID:
            matchinfo = info
            break

    # raise error if match is not contained in matchinfo
    if matchinfo is None:
        raise KeyError(
            f"The match with mID {mID} was not found in the specified "
            f"File of match information ({filepath_match})."
        )

    # initialize teamsheets
    teamsheets = {
        "Home": pd.DataFrame(
            columns=["player", "position", "team_name", "jID", "pID", "tID"]
        ),
        "Away": pd.DataFrame(
            columns=["player", "position", "team_name", "jID", "pID", "tID"]
        ),
    }

    # find team data in match info
    tIDs = {
        "Home": matchinfo["home_team"]["home_team_id"],
        "Away": matchinfo["away_team"]["away_team_id"],
    }
    team_names = {
        "Home": matchinfo["home_team"]["home_team_name"],
        "Away": matchinfo["away_team"]["away_team_name"],
    }

    # parse starting eleven
    for event in file_event_list:
        if event["type"]["name"] != "Starting XI":
            continue

        # find team
        if event["team"]["id"] == tIDs["Home"]:
            team = "Home"
        elif event["team"]["id"] == tIDs["Away"]:
            team = "Away"
        else:
            team = None

        # find list of players
        players = event["tactics"]["lineup"]

        # add player data to teamsheets
        teamsheets[team]["player"] = [player["player"]["name"] for player in players]
        teamsheets[team]["pID"] = [player["player"]["id"] for player in players]
        teamsheets[team]["jID"] = [player["jersey_number"] for player in players]
        teamsheets[team]["position"] = [
            player["position"]["name"] for player in players
        ]
        teamsheets[team]["tID"] = tIDs[team]
        teamsheets[team]["team_name"] = team_names[team]

    # parse players coming in from substitutions
    for event in file_event_list:
        if event["type"]["name"] != "Substitution":
            continue

        # find team
        if event["team"]["id"] == tIDs["Home"]:
            team = "Home"
        elif event["team"]["id"] == tIDs["Away"]:
            team = "Away"
        else:
            team = None

        # append player data to teamsheet
        player_data = pd.DataFrame(
            {
                "player": [event["substitution"]["replacement"]["name"]],
                "pID": [event["substitution"]["replacement"]["id"]],
                "jID": [pd.NA],  # unfortunately not included in substitution event
                "position": [event["position"]["name"]],
                "tID": [tIDs[team]],
                "team_name": [team_names[team]],
            }
        )
        teamsheets[team] = pd.concat((teamsheets[team], player_data), ignore_index=True)

    # create teamsheet objects
    for team in team_names.keys():
        teamsheets[team] = Teamsheet(teamsheets[team])

    return teamsheets


def read_open_event_data_json(
    filepath_events: Union[str, Path],
    filepath_match: Union[str, Path],
    filepath_threesixty: Union[str, Path] = None,
    teamsheet_home: Teamsheet = None,
    teamsheet_away: Teamsheet = None,
) -> Tuple[Dict[str, Dict[str, Events]], Dict[str, Teamsheet]]:
    """Parses files for a single match from the StatsBomb open dataset and extracts the
    event data and teamsheets.

    This function provides high-level access to an events json file from the openly
    published StatsBomb open data and returns Event- and Teamsheet-objects for both
    teams for the full match. A StatsBomb360 json file can be passed to the function to
    include `StatsBomb360 data <https://statsbomb.com/articles/soccer/
    statsbomb-360-freeze-frame-viewer-a-new-release-in-statsbomb-iq/>`_ to the
    ``qualifier`` column. Requires the parsed files from the dataset to maintain their
    original names from the `official data repository
    <https://github.com/statsbomb/open-data>`_

    Parameters
    ----------
    filepath_events: str or pathlib.Path
        Full path to json file where the Event data is saved.
    filepath_match: str or pathlib.Path
        Full path to json file where information about all matches of a season are
        stored.
    filepath_threesixty: str or pathlib.Path, optional
        Full path to json file where the StatsBomb360 data in is saved if available. The
        information about the area of the field where player positions are tracked
        (``visible_area``) and player positions at single events (``freeze frame``) are
        stored as a string in the ``qualifier`` column.
    teamsheet_home: Teamsheet, optional
        Teamsheet-object for the home team used to create link dictionaries of the form
        `links[pID] = team`. If given as None (default), teamsheet is extracted from
        the events and match json files.
    teamsheet_away: Teamsheet, optional
        Teamsheet-object for the away team. If given as None (default), teamsheet is
        extracted from the events and match json files. See teamsheet_home for details.

    Returns
    -------
    data_objects: Tuple[Dict[str, Dict[str, Events]], Dict[str, Teamsheet]]
        Tuple of (nested) floodlight core objects with shape (events_objects,
        teamsheets).

        ``events_objects`` is a nested dictionary containing ``Events`` objects for
        each team and segment of the form ``events_objects[segment][team] = Events``.
        For a typical league match with two halves and teams this dictionary looks like:
        ``{'HT1': {'Home': Events, 'Away': Events}, 'HT2': {'Home': Events, 'Away':
        Events}}``.

        ``teamsheets`` is a dictionary containing ``Teamsheet`` objects for each team
        of the form ``teamsheets[team] = Teamsheet``.

    Notes
    -----
    StatsBomb's open format of handling provides certain additional event attributes,
    which attach additional information to certain events. As of now, these information
    are parsed as a string in the ``qualifier`` column of the returned DataFrame and can
    be transformed to a dict of form ``{attribute: value}``. This includes the
    information about the tracked position of (some) players and the visible area
    that is included in the StatsBomb360 data.
    """

    # load json files into memory
    with open(str(filepath_events), "r", encoding="utf8") as f:
        file_event_list = json.load(f)
    if filepath_threesixty is not None:
        with open(str(filepath_threesixty), "r", encoding="utf8") as f:
            file_threesixty_list = json.load(f)
    else:
        file_threesixty_list = None

    # create or check teamsheet objects
    if teamsheet_home is None and teamsheet_away is None:
        teamsheets = read_teamsheets_from_open_event_data_json(
            filepath_events, filepath_match
        )
        teamsheet_home = teamsheets["Home"]
        teamsheet_away = teamsheets["Away"]
    elif teamsheet_home is None:
        teamsheets = read_teamsheets_from_open_event_data_json(
            filepath_events, filepath_match
        )
        teamsheet_home = teamsheets["Home"]
    elif teamsheet_away is None:
        teamsheets = read_teamsheets_from_open_event_data_json(
            filepath_events, filepath_match
        )
        teamsheet_away = teamsheets["Away"]
    else:
        pass
        # potential check

    # create links
    links_tID_to_team = {
        teamsheet_home.teamsheet.at[0, "tID"]: "Home",
        teamsheet_away.teamsheet.at[0, "tID"]: "Away",
    }
    periods = set([event["period"] for event in file_event_list])
    segments = [f"HT{period}" for period in periods]
    mID = int(filepath_events.split(os.path.sep)[-1][:-5])  # from filepath

    # initialize event bins
    columns = [
        "eID",
        "gameclock",
        "pID",
        "tID",
        "mID",
        "outcome",
        "timestamp",
        "minute",
        "second",
        "at_x",
        "at_y",
        "to_x",
        "to_y",
        "event_name",
        "player_name",
        "team_name",
        "qualifier",
    ]

    team_event_lists = {
        team: {segment: {col: [] for col in columns} for segment in segments}
        for team in links_tID_to_team.values()
    }

    # parse events loop
    for event in file_event_list:
        # get team and segment information
        period = event["period"]
        segment = "HT" + str(period)
        team = links_tID_to_team[event["possession_team"]["id"]]

        # identifier and outcome:
        eID = event["type"]["id"]
        tID = event["team"]["id"]
        pID = event["player"]["id"] if "player" in event else None
        event_name = event["type"]["name"]
        team_name = event["team"]["name"]
        player_name = event["player"]["name"] if "player" in event else None
        outcome = None
        if "type" in event and event["type"]["name"].lower() in event:
            outcome_name = (
                event[event["type"]["name"].lower()]["outcome"]["name"]
                if "outcome" in event[event["type"]["name"].lower()]
                else "None"
            )
            if outcome_name in ["Goal", "Won", "Complete", "Success In Play"]:
                outcome = 1
            elif outcome_name in [
                "Incomplete",
                "Lost In Play",
                "Saved Off Target",
                "Off T",
                "Blocked",
            ]:
                outcome = 0
        team_event_lists[team][segment]["mID"].append(mID)
        team_event_lists[team][segment]["eID"].append(eID)
        team_event_lists[team][segment]["tID"].append(tID)
        team_event_lists[team][segment]["pID"].append(pID)
        team_event_lists[team][segment]["event_name"].append(event_name)
        team_event_lists[team][segment]["team_name"].append(team_name)
        team_event_lists[team][segment]["player_name"].append(player_name)
        team_event_lists[team][segment]["outcome"].append(outcome)

        # relative time
        timestamp = event["timestamp"]
        minute = event["minute"]
        second = event["second"]
        millisecond = int(timestamp.split(".")[1])
        gameclock = 60 * minute + second + millisecond * 0.001
        team_event_lists[team][segment]["timestamp"].append(timestamp)
        team_event_lists[team][segment]["minute"].append(minute)
        team_event_lists[team][segment]["second"].append(second)
        team_event_lists[team][segment]["gameclock"].append(gameclock)

        # location
        at_x = event["location"][0] if "location" in event else None
        at_y = event["location"][1] if "location" in event else None
        if "type" in event and event["type"]["name"].lower() in event:
            to_x = (
                event[event["type"]["name"].lower()]["end_location"][0]
                if "end_location" in event[event["type"]["name"].lower()]
                else None
            )
            to_y = (
                event[event["type"]["name"].lower()]["end_location"][1]
                if "end_location" in event[event["type"]["name"].lower()]
                else None
            )
        else:
            to_x = None
            to_y = None
        team_event_lists[team][segment]["at_x"].append(at_x)
        team_event_lists[team][segment]["at_y"].append(at_y)
        team_event_lists[team][segment]["to_x"].append(to_x)
        team_event_lists[team][segment]["to_y"].append(to_y)

        # qualifier
        qual_dict = {}
        qual_dict["unique_identifier"] = event["id"]
        for qualifier in event:
            if qualifier in [
                "team",
                "player",
                "period",
                "timestamp",
                "minute",
                "second",
                "location",
                "id",
                "type",
            ]:
                continue
            qual_value = event[qualifier]
            qual_dict[qualifier] = qual_value

        if file_threesixty_list is not None:
            threesixty_event = [
                event
                for event in file_threesixty_list
                if event["event_uuid"] == qual_dict["unique_identifier"]
            ]
            if len(threesixty_event) == 1:
                qual_dict["360_freeze_frame"] = threesixty_event[0]["freeze_frame"]
                qual_dict["360_visible_area"] = threesixty_event[0]["visible_area"]
            elif len(threesixty_event) >= 1:
                warnings.warn(
                    f"Found ambiguous StatsBomb event ID "
                    f"{qual_dict['unique_identifier']} matching to more than one "
                    f"StatsBomb360 event."
                )
        team_event_lists[team][segment]["qualifier"].append(str(qual_dict))

    # create objects
    events_objects = {}
    for segment in segments:
        events_objects[segment] = {}
        for team in ["Home", "Away"]:
            events_objects[segment][team] = Events(
                events=pd.DataFrame(data=team_event_lists[team][segment]),
            )
    teamsheets = {
        "Home": teamsheet_home,
        "Away": teamsheet_away,
    }

    # pack objects
    data_objects = (events_objects, teamsheets)

    return data_objects

from pathlib import Path
from typing import Dict, Union

import datetime
import json
import pandas as pd

from floodlight import Events
from floodlight.io.utils import get_and_convert


def read_event_data_json(
    filepath_events: Union[str, Path]
) -> Dict[str, Dict[str, Dict]]:
    """Parses the Sportradar timeline files in json format and extracts the event data.

    This function provides access to `Sport Event Timeline
    <https://developer.sportradar.com/docs/read/handball/
    Handball_v2#sport-event-timeline>`_ files from the data provider `Sportradar
    <https://sportradar.com/>`_ exported in json format and returns Event objects for
    all teams and segments of the game.

    Parameters
    ----------
    filepath_events: str or pathlib.Path
        Full path to json file where the Sport Event Timeline is saved.

    Returns
    -------
    data_objects: Dict[str: Dict[str: Events]]
        Nested dictionary with ``Events`` objects for all teams and segments. The
        returned dictionary contains one dictionary per segment, which in return contain
        one ``Events`` object per team. For a usual league match with two halves and
        two teams this dictionary looks like:
        ``{"HT1": {"Home": Events, "Away": Events}, "HT2": {Home: Events, Away:
        Events}}``

    Notes
    -----
    Sportradar provides different information depending on the respective Event type.
    This parser itemizes top-level information for each possible event type listed in
    the `Handball v2 documentation FAQ
    <https://developer.sportradar.com/docs/read/handball/
    Handball_v2#frequently-asked-questions>`_ (most recent check: 11.01.2023).

    For example, this involves individual columns for the home and away score parsed
    event type *score_change*. However, individual columns for players involved in
    Events, like *seven_m_awarded* or *shots* are not fully itemized, as they can
    contain different information depending on the situation. More complex information
    that changes per event type is instead included as dict or list of dicts in
    according column, so they can be accessed if necessary.

    In the return, the following columns contain temporal information about the event:
    ``("gameclock", "time_stamp", "minutes_gross", "seconds_gross", "minutes", "seconds"
    )``. In handball, the match-clock determines the net playing time (60 minutes) and
    diverges from the gross "real world" time passed. The "gameclock" column contains
    the gross time passed in seconds in relation to the start of the respective segment.
    The "minute_gross" and "second_gross" columns contain the "gameclock" converted to
    minutes and seconds, respectively. The columns "minutes" and "seconds" contain the
    information about the net match-clock. the column "time_stamp" contains the global
    time-stamp of the respective event in the ISO 8601 standard format.

    The column "outcome" in the return contains the "outcome" information in the raw
    event data and not information about the success {0, 1} of an event. The outcome in
    terms of success can be inferred by the ``eID``. E.g. "score_change" implies that a
    shot lead to a goal, "shot_saved" implies that a goal was not scored.
    """

    # load full json into memory
    with open(str(filepath_events)) as f:
        events = json.load(f)

    # check if timeline data exists
    if "timeline" not in events:
        raise ValueError("There appears to be no timeline data in this file.")
    else:
        timeline = events["timeline"]

    # extract match id
    mID = events["sport_event"]["id"]

    # create links from home/away to team id and team name
    teams = ["Home", "Away"]

    home_away_link = {}
    for competitor in events["statistics"]["totals"]["competitors"]:
        home_away_link.update(
            {
                competitor["qualifier"].capitalize(): (
                    competitor["id"],
                    competitor["name"],
                )
            }
        )

    # extract periods
    periods = sorted(
        set(
            [
                event["period_name"]
                for event in timeline
                if event["type"] == "period_start"
            ]
        )
    )

    # create team event dict
    columns = [
        "eID",
        "gameclock",
        "time_stamp",
        "minute_gross",
        "second_gross",
        "minute",
        "second",
        "pID",
        "player_name",
        "tID",
        "team_name",
        "mID",
        "home_score",
        "away_score",
        "scorer",
        "assists",
        "zone",
        "shot_type",
        "outcome",
        "players",
    ]

    segments = [f"HT{period[0]}" for period in periods]

    team_event_lists = {
        team: {segment: {col: [] for col in columns} for segment in segments}
        for team in teams
    }

    period = None
    # loop through event timeline
    for event in timeline:
        if period is None:
            # get first period
            if event["type"] == "period_start":
                period = event["period_name"]
                segment = f"HT{period[0]}"
                segment_start = datetime.datetime.fromisoformat(event["time"])
            else:
                # skip events before first period starts
                continue
        # get new periods
        else:
            if event["type"] == "period_start":
                period = event["period_name"]
                segment = f"HT{period[0]}"
                segment_start = datetime.datetime.fromisoformat(event["time"])

        # extract event, player and team ids and names
        eID = event["type"]

        # add all teams as competitors if no competitor is specified in event
        competitor = (
            [event["competitor"].capitalize()] if "competitor" in event else teams
        )

        tID = home_away_link[competitor[0]][0] if len(competitor) == 1 else None
        team_name = home_away_link[competitor[0]][1] if len(competitor) == 1 else None
        pID = event["player"]["id"] if "player" in event else None
        player_name = event["player"]["name"] if "player" in event else None

        # extract time codes and match-clock
        time_stamp = datetime.datetime.fromisoformat(event["time"])
        time_delta = time_stamp - segment_start
        gameclock = time_delta.seconds
        minute_gross = int(gameclock / 60)
        second_gross = int(gameclock % 60)

        if "match_clock" in event:
            match_clock = event["match_clock"]
            minute, second = [int(x) for x in match_clock.split(":")]
        else:
            minute, second = (None, None)

        # extract optional event information
        outcome = get_and_convert(event, "outcome", str)
        home_score = get_and_convert(event, "home_score", int)
        away_score = get_and_convert(event, "away_score", int)
        scorer = get_and_convert(event, "scorer", Dict)
        assists = get_and_convert(event, "assists", list)
        zone = get_and_convert(event, "zone", str)
        shot_type = get_and_convert(event, "shot_type", str)
        players = get_and_convert(event, "players", list)

        # add event to team event list
        for team in competitor:
            team_event_lists[team][segment]["eID"].append(eID)
            team_event_lists[team][segment]["gameclock"].append(gameclock)
            team_event_lists[team][segment]["time_stamp"].append(time_stamp)
            team_event_lists[team][segment]["minute_gross"].append(minute_gross)
            team_event_lists[team][segment]["second_gross"].append(second_gross)
            team_event_lists[team][segment]["minute"].append(minute)
            team_event_lists[team][segment]["second"].append(second)
            team_event_lists[team][segment]["pID"].append(pID)
            team_event_lists[team][segment]["player_name"].append(player_name)
            team_event_lists[team][segment]["tID"].append(tID)
            team_event_lists[team][segment]["team_name"].append(team_name)
            team_event_lists[team][segment]["mID"].append(mID)
            team_event_lists[team][segment]["home_score"].append(home_score)
            team_event_lists[team][segment]["away_score"].append(away_score)
            team_event_lists[team][segment]["scorer"].append(scorer)
            team_event_lists[team][segment]["assists"].append(assists)
            team_event_lists[team][segment]["zone"].append(zone)
            team_event_lists[team][segment]["shot_type"].append(shot_type)
            team_event_lists[team][segment]["outcome"].append(outcome)
            team_event_lists[team][segment]["players"].append(players)

    # flexible parser return for all segments and teams
    data_objects = {
        segment: {
            team: Events(events=pd.DataFrame(data=team_event_lists[team][segment]))
            for team in teams
        }
        for segment in segments
    }

    return data_objects

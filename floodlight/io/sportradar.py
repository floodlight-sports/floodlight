from pathlib import Path
from typing import Dict, Union

import datetime
import json
import pandas as pd

from floodlight import Events


def read_sportradar_timeline(
    filepath_events: Union[str, Path]
) -> Dict[str, Dict[str, Dict]]:
    """Parses the sport radar timeline files in json format and extracts the event data.

    This function provides access to the `Sport Event Timeline
    <https://developer.sportradar.com/docs/read/handball/
    Handball_v2#sport-event-timeline>`_ file from the dataprovider `Sportradar
    <https://sportradar.com/>`_ exported in json format and returns Event objects for
    all teams and segments of the game.

    Parameters
    ----------
    filepath_events: str or pathlib.Path
        Full path to json file where the Sport Event Timeline is saved.

    Returns
    -------
    data_objects: Dict[str: Dict[str: Events]]
        Dict with Events-objects for all teams and segments. For a usual league match
        with two halves and two teams the order is:

        {

            1st_half: {team_a: Events, team_b: Events},

            2nd_half: {team_a: Events, team_b: Events}

        }

    Notes
    -----
    Sportradar provides different information based on the respective Event type.
    We decided to parse the first "layer" of information for each possible event type
    that is listed in the `Handball v2 documentation FAQ
    <https://developer.sportradar.com/docs/read/handball/
    Handball_v2#frequently-asked-questions>`_ (most recent check: 11.01.2023). This
    involves for example individual columns for the  home and away score at the event
    type score_change but no individual columns for the players involved in Events, like
    seven_m_awarded or shots, as they can contain different information based on the
    situation. However, we parse the raw information as dict or list of dicts in the
    respective column so the user can access it if necessary.
    """

    # load full json into memory
    with open(str(filepath_events)) as f:
        events = json.load(f)
        f.close()

    # check if timeline data exists
    if "timeline" not in events:
        raise ValueError("There appears to be no timeline data in this file.")
    else:
        timeline = events["timeline"]

    # extract match id
    mID = events["sport_event"]["id"]

    # create links from home/away to team id and team name
    teams = ["home", "away"]

    tID_link = {}
    home_away_link = {}
    for competitor in events["statistics"]["totals"]["competitors"]:
        tID_link.update({competitor["id"]: competitor["qualifier"]})
        home_away_link.update(
            {competitor["qualifier"]: (competitor["id"], competitor["name"])}
        )

    # extract periods
    periods = set(
        [event["period_name"] for event in timeline if event["type"] == "period_start"]
    )

    # create team event dict
    columns = [
        "eID",
        "gameclock",
        "time_stamp",
        "minute",
        "second",
        "pID",
        "player_name",
        "tID",
        "team_name",
        "mID",
        "event_name",
        "home_score",
        "away_score",
        "scorer",
        "assists",
        "zone",
        "shot_type",
        "outcome",
        "players",
    ]

    # seems unnecessary
    segments = [f"HT{period}" for period in periods]

    team_event_lists = {
        team: {segment: {col: [] for col in columns} for segment in segments}
        for team in teams
    }

    period = None
    # loop
    for event in timeline:
        if not period:
            # get first period
            if event["type"] == "period_start":
                period = event["period_name"]
                segment = f"HT{period}"
                segment_start = datetime.datetime.fromisoformat(event["time"])
            else:
                # skip events before first period starts
                continue
        # get new periods
        else:
            if event["type"] == "period_start":
                period = event["period_name"]
                segment = f"HT{period}"
                segment_start = datetime.datetime.fromisoformat(event["time"])

        # extract event, player and team ids and names
        eID = event["id"]

        # add all teams as competitors if no competitor is specified in event
        competitor = [event["competitor"]] if "competitor" in event else teams

        tID = home_away_link[competitor[0]][0] if len(competitor) == 1 else None
        team_name = home_away_link[competitor[0]][1] if len(competitor) == 1 else None
        pID = event["player"]["id"] if "player" in event else None
        player_name = event["player"]["name"] if pID else None

        event_name = event["type"]

        # extract time codes and matchclock
        time_stamp = datetime.datetime.fromisoformat(event["time"])
        time_delta = time_stamp - segment_start
        gameclock = time_delta.seconds
        if "match_clock" in event:
            match_clock = event["match_clock"]
            minute, second = [int(x) for x in match_clock.split(":")]
        else:
            minute, second = (None, None)

        # extract optional event information
        outcome = event["outcome"] if "outcome" in event else None
        home_score = event["home_score"] if "home_score" in event else None
        away_score = event["away_score"] if "away_score" in event else None
        scorer = event["scorer"] if "scorer" in event else None
        assists = event["assists"] if "assists" in event else None
        zone = event["zone"] if "zone" in event else None
        shot_type = event["shot_type"] if "shot_type" in event else None
        players = event["players"] if "players" in event else None

        # add event to team event list
        for team in competitor:
            team_event_lists[team][segment]["eID"].append(eID)
            team_event_lists[team][segment]["gameclock"].append(gameclock)
            team_event_lists[team][segment]["time_stamp"].append(time_stamp)
            team_event_lists[team][segment]["minute"].append(minute)
            team_event_lists[team][segment]["second"].append(second)
            team_event_lists[team][segment]["pID"].append(pID)
            team_event_lists[team][segment]["player_name"].append(player_name)
            team_event_lists[team][segment]["tID"].append(tID)
            team_event_lists[team][segment]["team_name"].append(team_name)
            team_event_lists[team][segment]["mID"].append(mID)
            team_event_lists[team][segment]["event_name"].append(event_name)
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

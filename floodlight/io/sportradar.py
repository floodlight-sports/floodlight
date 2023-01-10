import datetime
import json
import pandas as pd

from floodlight import Events


def read_sportradar_timeline(filepath_events):
    """
    Reads the sport radar timeline json files.
    Parameters
    ----------
    filepath_events

    Returns
    -------

    """

    with open(filepath_events) as f:
        events = json.load(f)
        f.close()

    if "timeline" not in events:
        raise ValueError("There appears to be no timeline data in this file.")
    else:
        timeline = events["timeline"]
    mID = events["sport_event"]["id"]

    teams = ["home", "away"]

    tID_link = {}
    home_away_link = {}
    pID_link = {}
    for competitor in events["statistics"]["totals"]["competitors"]:
        tID_link.update({competitor["id"]: competitor["qualifier"]})
        home_away_link.update(
            {competitor["qualifier"]: (competitor["id"], competitor["name"])}
        )
        for player in competitor["players"]:
            pID_link.update({player["id"]: competitor["id"]})

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

    periods = set(
        [event["period_name"] for event in timeline if event["type"] == "period_start"]
    )
    segments = [f"HT{period}" for period in periods]  ### seems unnecessary

    team_event_lists = {
        team: {segment: {col: [] for col in columns} for segment in segments}
        for team in teams
    }

    period = None
    for event in timeline:
        # get period
        if event["type"] == "match_started":
            continue

        if event["type"] == "period_start":
            period = event["period_name"]
            segment = f"HT{period}"
            segment_start = datetime.datetime.fromisoformat(event["time"])

        eID = event["id"]

        competitor = [event["competitor"]] if "competitor" in event else teams

        tID = home_away_link[competitor[0]][0] if len(competitor) == 1 else None
        team_name = home_away_link[competitor[0]][1] if len(competitor) == 1 else None
        pID = event["player"]["id"] if "player" in event else None
        player_name = event["player"]["name"] if pID else None

        event_name = event["type"]

        # time
        time_stamp = datetime.datetime.fromisoformat(event["time"])
        time_delta = time_stamp - segment_start
        gameclock = time_delta.seconds
        if "match_clock" in event:
            match_clock = event["match_clock"]
            minute, second = [int(x) for x in match_clock.split(":")]
        else:
            minute, second = (None, None)

        outcome = event["outcome"] if "outcome" in event else None
        home_score = event["home_score"] if "home_score" in event else None
        away_score = event["away_score"] if "away_score" in event else None
        scorer = event["scorer"] if "scorer" in event else None
        assists = event["assists"] if "assists" in event else None
        zone = event["zone"] if "zone" in event else None
        shot_type = event["shot_type"] if "shot_type" in event else None
        players = event["players"] if "players" in event else None

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

        data_objects = {
            segment: {
                team: Events(events=pd.DataFrame(data=team_event_lists[team][segment]))
                for team in teams
            }
            for segment in segments
        }

    return data_objects

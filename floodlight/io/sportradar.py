
import datetime
import json
import pandas as pd

filepath_events = "C:\\Users\\ke6564\\Desktop\\Studium\\Data\\Handball\\HBL_Events\\season_20_21\\EventTimeline\\sport_events_23400263_timeline.json"

with open(filepath_events) as f:
    events = json.load(f)
    f.close()

timeline = events["timeline"]
mID = events["sport_event"]["id"]

teams = ["home", "away"]

tID_link = {}
home_away_link = {}
pID_link = {}
for competitor in events["statistics"]["totals"]["competitors"]:
    tID_link.update({competitor["id"]: competitor["qualifier"]})
    home_away_link.update({competitor["qualifier"]: competitor["id"]})
    for player in competitor["players"]:
        pID_link.update({player["id"]: competitor["id"]})

columns = [
        "eID",
        "gameclock",
        "timestamp",
        "minute",
        "second",
        "pID",
        "tID",
        "mID",

        "event_name",
        "player_name",
        "team_name",
        "qualifier",
    ]

periods = set([event["period_name"] for event in timeline if event["type"] == "period_start"])
segments = [f"HT{period}" for period in periods] ### seems unnecessary

team_event_lists = {
    team: {segment: {col: [] for col in columns} for segment in segments}
    for team in teams
}

period = None
for event in timeline:

    # get period
    if event["type"] == "period_start":
        period = event["period_name"]
        segment = f"HT{period}"

    eID = event["id"]

    competitor = event["competitor"] if "competitor" in event else None

    tID = home_away_link[competitor] if competitor else None
    # team_name = later
    pID = event["player"]["id"] if "player" in event else None
    player_name = event["player"]["name"] if pID else None

    event_name = event["type"]


    # time
    if event_name == "match_started":
        t_start = datetime.datetime.fromisoformat(event["time"])

    time_stamp = datetime.datetime.fromisoformat(event["time"])
    time_delta = time_stamp - t_start
    if "match_clock" in event:
        match_clock = event["match_clock"]
        minutes, seconds = [int(x) for x in match_clock.split(":")]

for event in timeline:
    if "competitor" not in event.keys():
        print(event["type"])










types = []
for event in timeline:
    if event["type"] not in types:
        types.append(event["type"])
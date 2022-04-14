import pandas as pd
from floodlight.core.events import Events

# 1st half
events_away_ht1 = pd.DataFrame()

events_away_ht1 = events_away_ht1.append(
    {
        "eID": "KickoffWhistle",
        "gameclock": 0 / 5,
        "outcome": None,
    },
    ignore_index=True,
)

events_away_ht1 = events_away_ht1.append(
    {
        "eID": "Save",
        "gameclock": 250 / 5,
        "outcome": None,
    },
    ignore_index=True,
)

events_away_ht1 = events_away_ht1.append(
    {
        "eID": "Kickoff",
        "gameclock": 700 / 5,
        "outcome": None,
    },
    ignore_index=True,
)


events_away_ht1 = events_away_ht1.append(
    {
        "eID": "Pass",
        "gameclock": 700 / 5,
        "outcome": 1,
    },
    ignore_index=True,
)


events_away_ht1 = events_away_ht1.append(
    {
        "eID": "Pass",
        "gameclock": 715 / 5,
        "outcome": 0,
    },
    ignore_index=True,
)

events_away_ht1 = events_away_ht1.append(
    {
        "eID": "FinalWhistle",
        "gameclock": 1010 / 5,
        "outcome": None,
    },
    ignore_index=True,
)

events_away_ht1 = Events(events_away_ht1)


# 2nd half
events_away_ht2 = pd.DataFrame()

events_away_ht2 = events_away_ht2.append(
    {"eID": "KickoffWhistle", "gameclock": 0 / 5},
    ignore_index=True,
)

events_away_ht2 = events_away_ht2.append(
    {"eID": "Kickoff", "gameclock": 0 / 5, "outcome": None},
    ignore_index=True,
)

events_away_ht2 = events_away_ht2.append(
    {"eID": "Pass", "gameclock": 0 / 5, "outcome": 1},
    ignore_index=True,
)

events_away_ht2 = events_away_ht2.append(
    {"eID": "Pass", "gameclock": 5 / 5, "outcome": 1},
    ignore_index=True,
)

events_away_ht2 = events_away_ht2.append(
    {"eID": "Pass", "gameclock": 5 / 5, "outcome": 1},
    ignore_index=True,
)

events_away_ht2 = events_away_ht2.append(
    {"eID": "Dribbling", "gameclock": 25 / 5, "outcome": 1},
    ignore_index=True,
)

events_away_ht2 = events_away_ht2.append(
    {"eID": "Pass", "gameclock": 40 / 5, "outcome": 1},
    ignore_index=True,
)

events_away_ht2 = events_away_ht2.append(
    {"eID": "Pass", "gameclock": 65 / 5, "outcome": 1},
    ignore_index=True,
)

events_away_ht2 = events_away_ht2.append(
    {"eID": "Pass", "gameclock": 90 / 5, "outcome": 1},
    ignore_index=True,
)

events_away_ht2 = events_away_ht2.append(
    {
        "eID": "Freekick",
        "gameclock": 300 / 5,
        "outcome": 0,
    },
    ignore_index=True,
)

events_away_ht2 = events_away_ht2.append(
    {"eID": "Penalty", "gameclock": 525 / 5, "outcome": 0},
    ignore_index=True,
)

events_away_ht2 = events_away_ht2.append(
    {"eID": "FinalWhistle", "gameclock": 740 / 5, "outcome": None},
    ignore_index=True,
)

events_away_ht2 = Events(events_away_ht2)

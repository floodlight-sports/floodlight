import numpy as np
import datetime


# fmt: off
essential_events_columns = {
    "eID": {
        "definition": "Event ID - unique number or string that identifies the event "
                      "type. The resulting system can either be provider specific, or "
                      "customary. However, a link between eID and the event "
                      "definitions/descriptions should be available",
        "dtypes": [str, int],
        "value_range": None,
    },
    "gameclock": {
        "definition": "Elapsed time relative to segment start in seconds",
        "dtypes": [float],
        "value_range": [0, np.inf]
    }
}


protected_columns = {
    "pID": {
        "definition": "Player ID - unique number or string for player identification ",
        "dtypes": [str, int],
        "value_range": None
    },
    "jID": {
        "definition": "Jersey ID - a players jersey number in a single observation",
        "dtypes": [int],
        "value_range": [0, np.inf]
    },
    "xID": {
        "definition": "Index ID - a players index in the list of all players of a team"
                      "for a given match (starts counting at 1). This is primarily used"
                      " for locating players data in XY objects, but can also be "
                      "helpful iterating or displaying all players of a team",
        "dtypes": [int],
        "value_range": [1, np.inf]
    },
    "tID": {
        "definition": "Team ID - unique number or string for team identification ",
        "dtypes": [str, int],
        "value_range": None
    },
    "mID": {
        "definition": "Match ID - unique number or string for match identification ",
        "dtypes": [str, int],
        "value_range": None
    },
    "cID": {
        "definition": "Competition ID - unique number or string for competition (e.g. "
                      "league or cup) identification",
        "dtypes": [str, int],
        "value_range": None
    },
    "frameclock": {
        "definition": "Elapsed time relative to segment start in frames given a certain"
                      "framerate.",
        "dtypes": [int],
        "value_range": [0, np.inf]
    },
    "timestamp": {
        "definition": "Datetime timestamp. Should be aware and carry a pytz timezone",
        "dtypes": [datetime.datetime],
        "value_range": None
    },
    "minute": {
        "definition": "Minute of the segment the event took place",
        "dtypes": [int],
        "value_range": [0, np.inf]
    },
    "second": {
        "definition": "Second of the minute of the segment the event took place",
        "dtypes": [int],
        "value_range": [0, np.inf]
    },
    "outcome": {
        "definition": "Result of an event as included by many data providers. "
                      "Positive/Successful is 1, Negative/Unsuccessful is 0",
        "dtypes": [int],
        "value_range": [0, 1],
    },
    "at_x": {
        "definition": "The x position (longitudinal) where the event took place or "
                      "originated from",
        "dtypes": [float],
        "value_range": None
    },
    "at_y": {
        "definition": "The y position (lateral) where the event took place or "
                      "originated from",
        "dtypes": [float],
        "value_range": None
    },
    "to_x": {
        "definition": "The x position (longitudinal) where the event ended",
        "dtypes": [float],
        "value_range": None
    },
    "to_y": {
        "definition": "The y position (lateral) where the event ended",
        "dtypes": [float],
        "value_range": None
    },
}

# fmt:on

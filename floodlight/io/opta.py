import re
from pathlib import Path
from typing import Dict, Tuple, Union

import pytz
import iso8601
import pandas as pd
from lxml import etree

from floodlight.core.events import Events
from floodlight.core.pitch import Pitch
from floodlight.io.utils import get_and_convert


def get_opta_feedtype(filepath: Union[str, Path]) -> Union[str, None]:
    """Tries to extract the feed type from Opta's XML feed.

    This function assumes that the file follows Opta's format of producing feeds.
    Thus it should have a "PRODUCTION HEADER" comment at the top of the file so that on
    line 6 it reads something like ``production module:  Opta::Feed::XML::Soccer::F24``.

    Parameters
    ----------
    filepath : Union[str, Path]
        Full path to Opta XML file.

    Returns
    -------
    feedtype: str or None
        Returns the type of the feed  as a string in case it  finds it, e.g. 'F24',
        and `None` otherwise.
    """
    with open(str(filepath), "r") as f:
        # iterate through first lines instead of loading entire file to RAM
        for i, line in enumerate(f):
            # search for production module at line 6
            if i == 6:
                production_tags = line.strip().split(":")
                if production_tags[0] == "production module":
                    feedtype = production_tags[-1]
                else:
                    feedtype = None
                break

    return feedtype


def read_event_data_xml(
    filepath: Union[str, Path]
) -> Tuple[Dict[str, Dict[str, Events]], Pitch]:
    """Parse Opta's f24 feed (containing match events) and extract event data and pitch
    information.

    This function provides a high-level access to the particular f24 feed and will
    return event objects for both teams. The number of segments is inferred from the
    data, yet data for each segment is stored in a separate object.

    Parameters
    ----------
    filepath: str or pathlib.Path
        Full path to the XML feed.

    Returns
    -------
    data_objects: Tuple[Dict[str, Dict[str, Events]], Pitch]
        Tuple of (nested) floodlight core objects with shape (events_objects,
        pitch).

        ``events_objects`` is a nested dictionary containing ``Events`` objects for
        each team and segment of the form ``events_objects[segment][team] = Events``.
        For a typical league match with two halves and teams this dictionary looks like:
        ``{'HT1': {'Home': Events, 'Away': Events}, 'HT2': {'Home': Events, 'Away':
        Events}}``.

        ``pitch`` is a ``Pitch`` object corresponding to the data.

    Notes
    -----
    Opta's format of handling event data information involves an elaborate use of so
    called qualifiers, which attach additional information to certain events. There
    also exist a number of mappings that define which qualifiers may be  attached to
    which kind of events. Parsing this information involves quite a bit of logic and is
    planned to be inclucded in further releases. As of now, qualifier information is
    parsed as a string in the `qualifier` column of the returned DataFrame and can be
    transformed to a dict of the form `{qualifier_id: value}`.
    """
    # check feed type
    if get_opta_feedtype(filepath) != "F24":
        raise ValueError(f"Not an Opta F24 feed: {filepath}")

    # load xml tree into memory
    tree = etree.parse(str(filepath))
    root = tree.getroot()

    # 1. parse match info
    matchinfo = root.xpath("Game")[0].attrib
    teams = ["Home", "Away"]
    tID_link = {
        int(matchinfo["home_team_id"]): "Home",
        int(matchinfo["away_team_id"]): "Away",
    }
    number_of_periods = len(list(filter(re.compile("period_._start").match, matchinfo)))
    segments = [f"HT{period}" for period in range(1, number_of_periods + 1)]

    # 2. parse events
    # bins
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

    event_lists = {
        team: {segment: {col: [] for col in columns} for segment in segments}
        for team in teams
    }
    directions = {team: {} for team in teams}
    dir_link = {"Left to Right": "lr", "Right to Left": "rl"}
    segment_offsets = {1: 0, 2: 45, 3: 90, 4: 105}
    kickoffs = {}

    # read kickoff events for times and playing direction
    # (NOTE: kickoff times can also be directly found in matchinfo, although the
    # explicit kickoff-event timestamps appear to be more accurate)
    for event in root.xpath("Game/Event[@type_id='32']"):
        # get team and segment information
        period = get_and_convert(event.attrib, "period_id", int)
        segment = "HT" + str(period)
        tID = get_and_convert(event.attrib, "team_id", int)
        team = tID_link[tID]
        # read kickoff times
        kickoff_timestring = get_and_convert(event.attrib, "timestamp", str)
        kickoff_datetime = iso8601.parse_date(
            kickoff_timestring, default_timezone=pytz.utc
        )
        kickoffs[segment] = kickoff_datetime
        # read playing direction
        direction_qualifier = event.xpath("Q[@qualifier_id='127']")
        if len(direction_qualifier) > 0:
            value = get_and_convert(direction_qualifier[0], "value", str)
            direction = dir_link.get(value)
        else:
            direction = None
        directions[team][segment] = direction
        # cut event from tree to prevent double parsing
        # event.getparent().remove(event)

    # loop
    for event in root.xpath("Game/Event"):
        # get team and segment information
        period = get_and_convert(event.attrib, "period_id", int)
        segment = "HT" + str(period)
        tID = get_and_convert(event.attrib, "team_id", int)
        team = tID_link[tID]
        # skip match-unrelated events
        if period not in range(1, 6):
            continue

        # identifier and outcome:
        eID = get_and_convert(event.attrib, "type_id", int)
        # skip unwanted events
        if eID in [30]:
            continue
        pID = get_and_convert(event.attrib, "player_id", int)
        outcome = get_and_convert(event.attrib, "outcome", int)
        event_lists[team][segment]["eID"].append(eID)
        event_lists[team][segment]["pID"].append(pID)
        event_lists[team][segment]["outcome"].append(outcome)

        # absolute and relative time
        event_timestring = get_and_convert(event.attrib, "timestamp", str)
        minute = get_and_convert(event.attrib, "min", int)
        # transform minute to be relative to current segment
        minute -= segment_offsets[period]
        second = get_and_convert(event.attrib, "sec", int)
        timestamp = iso8601.parse_date(event_timestring, default_timezone=pytz.utc)
        delta = timestamp - kickoffs[segment]
        gameclock = delta.total_seconds()
        # re-adjust pre-kick-off events (e.g. substitutions) to 00:00
        gameclock = max(gameclock, 0.0)
        event_lists[team][segment]["timestamp"].append(timestamp)
        event_lists[team][segment]["minute"].append(minute)
        event_lists[team][segment]["second"].append(second)
        event_lists[team][segment]["gameclock"].append(gameclock)

        # location
        at_x = get_and_convert(event.attrib, "x", float)
        at_y = get_and_convert(event.attrib, "y", float)
        event_lists[team][segment]["at_x"].append(at_x)
        event_lists[team][segment]["at_y"].append(at_y)

        # qualifier
        qual_dict = {}
        for qualifier in event.iterchildren():
            qual_id = int(qualifier.attrib["qualifier_id"])
            qual_value = qualifier.attrib.get("value")
            qual_dict[qual_id] = qual_value
        event_lists[team][segment]["qualifier"].append(str(qual_dict))

    # create objects
    events_objects = {}
    for segment in segments:
        events_objects[segment] = {}
        for team in ["Home", "Away"]:
            events_objects[segment][team] = Events(
                events=pd.DataFrame(data=event_lists[team][segment]),
                direction=directions[team][segment],
            )
    pitch = Pitch.from_template("opta", sport="football")

    # pack objects
    data_objects = (events_objects, pitch)

    return data_objects

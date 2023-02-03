import json
from pathlib import Path
from typing import Dict, Tuple, Union

import pytz
import iso8601
import numpy as np
import pandas as pd

from floodlight.core.events import Events
from floodlight.core.code import Code
from floodlight.core.pitch import Pitch
from floodlight.core.xy import XY
from floodlight.core.teamsheet import Teamsheet
from floodlight.io.utils import get_and_convert


def _get_position_precedence(position: str) -> int:
    """Get the precedence of a position abbreviation (e.g. 'GK') useful for ordering."""
    PLAYER_POSITION_PRECEDENCE = {
        "GK": 1,
        "LB": 2,
        "LCB": 3,
        "CB": 4,
        "RCB": 5,
        "RB": 6,
        "LWB": 7,
        "LDM": 8,
        "CDM": 9,
        "RDM": 10,
        "RWB": 11,
        "LM": 12,
        "LCM": 13,
        "CM": 14,
        "RCM": 15,
        "RM": 16,
        "LW": 17,
        "CAM": 18,
        "RW": 19,
        "LF": 20,
        "LCF": 21,
        "CF": 22,
        "RCF": 23,
        "RF": 24,
        "SUB": 99,
    }
    # unknown positions are inserted between known positions and substitutes
    precedence = PLAYER_POSITION_PRECEDENCE.get(position, 23)

    return precedence


def _read_metajson(
    filepath_metadata: Union[str, Path]
) -> Tuple[Dict, Dict, Dict, Pitch]:
    """Reads Second Spectrums's metadata file and extracts information about match
    metainfo, periods, playing directions, and the pitch.

    Parameters
    ----------
    filepath_metadata: str or pathlib.Path
        Full path to _meta.json file.

    Returns
    -------
    metadata: dict
        Dictionary with meta information such as framerate.
    periods: Dict[Tuple[int, int]]
        Dictionary with start and endframes of all segments, e.g.,
        ``periods[segment] = (startframe, endframe)``.
    directions: Dict[str, Dict[str, str]]
        Dictionary with playing direction information of all segments and teams,
        e.g., ``directions[segment][team] = 'lr'``
    pitch: Pitch
        Pitch object with actual pitch length and width.
    """
    # read file
    with open(str(filepath_metadata), "r") as f:
        metajson = json.load(f)

    # bin
    metadata = {}
    periods = {}
    directions = {}

    # assemble metadata
    metadata["framerate"] = metajson.get("fps", None)
    metadata["length"] = metajson.get("pitchLength", None)
    metadata["width"] = metajson.get("pitchWidth", None)
    metadata["home_tID"] = get_and_convert(metajson, "homeOptaId", int)
    metadata["away_tID"] = get_and_convert(metajson, "awayOptaId", int)

    # get period information
    for period in metajson["periods"]:
        segment = f"HT{period['number']}"
        periods[segment] = (period["startFrameIdx"], period["endFrameIdx"] + 1)

    # get playing direction
    for period in metajson["periods"]:
        segment = f"HT{period['number']}"
        directions[segment] = {}
        if period["homeAttPositive"]:
            directions[segment]["Home"] = "lr"
            directions[segment]["Away"] = "rl"
        else:
            directions[segment]["Home"] = "rl"
            directions[segment]["Away"] = "lr"

    # generate pitch object
    pitch = Pitch.from_template(
        template_name="secondspectrum",
        length=metadata["length"],
        width=metadata["width"],
        sport="football",
    )

    return metadata, periods, directions, pitch


def read_teamsheets_from_meta_json(
    filepath_metadata: Union[str, Path]
) -> Dict[str, Teamsheet]:
    """Parses the Second Spectrum meta.json-file and creates respective teamsheets for
    the home and the away team.

    Parameters
    ----------
    filepath_metadata: str or pathlib.Path
        Full path to _meta.json file.

    Returns
    -------
    teamsheets: Dict[str, Teamsheet]
        Dictionary with teamsheets for the home team and the away team.

    Notes
    -----
    The ordering of players is determined by position precedence, with the following
    precedence values assigned to Second Spectrum's player position information::

        {
            'GK': 1,
            'LB': 2,
            'LCB': 3,
            'CB': 4,
            'RCB': 5,
            'RB': 6,
            'LWB': 7,
            'LDM': 8,
            'CDM': 9,
            'RDM': 10,
            'RWB': 11,
            'LM': 12,
            'LCM': 13,
            'CM': 14,
            'RCM': 15,
            'RM': 16,
            'LW': 17,
            'CAM': 18,
            'RW': 19,
            'LF': 20,
            'LCF': 21,
            'CF': 22,
            'RCF': 23,
            'RF': 24,
            'SUB': 99
        }
    """
    # read file
    with open(str(filepath_metadata), "r") as f:
        metajson = json.load(f)

    # param
    key_map = {"Home": "homePlayers", "Away": "awayPlayers"}

    # bin
    teamsheets = {team: None for team in ["Home", "Away"]}

    # loop through teams
    for team in ["Home", "Away"]:
        # bin
        teamsheet = {
            column: [] for column in ["precedence", "player", "jID", "pID", "position"]
        }
        # query team player list
        player_list = metajson[key_map[team]]
        # add players to list
        for player in player_list:
            # query
            name = get_and_convert(player, "name", str)
            position = get_and_convert(player, "position", str)
            jID = get_and_convert(player, "number", int)
            pID = get_and_convert(player, "optaId", int)
            precedence = _get_position_precedence(position)
            # assign
            teamsheet["player"].append(name)
            teamsheet["position"].append(position)
            teamsheet["jID"].append(jID)
            teamsheet["pID"].append(pID)
            teamsheet["precedence"].append(precedence)
        # curate
        teamsheet = pd.DataFrame(teamsheet)
        teamsheet.sort_values("precedence", inplace=True)
        teamsheet.drop(["precedence"], axis=1, inplace=True)
        teamsheet.reset_index(drop=True, inplace=True)
        teamsheet = Teamsheet(teamsheet)
        teamsheets[team] = teamsheet

    return teamsheets


def read_position_data_jsonl(
    filepath_position: Union[str, Path],
    filepath_metadata: Union[str, Path],
    teamsheet_home: Teamsheet = None,
    teamsheet_away: Teamsheet = None,
) -> Tuple[
    Dict[str, Dict[str, XY]],
    Dict[str, Code],
    Dict[str, Code],
    Dict[str, Teamsheet],
    Pitch,
]:
    """Parse Second Spectrum files and extract position data, possession and ballstatus
    codes, as well as pitch information.

    Second Spectrum data is typically stored in two separate files, a .jsonl file
    containing the actual data as well as a _meta.json containing information about
    pitch size, framerate, lineups and start- and endframes of match periods. This
    function provides a high-level access to Second Spectrum data by parsing "the full
    match" given both files.

    Parameters
    ----------
    filepath_position: str or pathlib.Path
        Full path to .jsonl-file.
    filepath_metadata: str or pathlib.Path
        Full path to _meta.json file.
    teamsheet_home: Teamsheet, optional
        Teamsheet object for the home team used to create link dictionaries of the form
        `links[team][jID] = xID`. The links are used to map players to a specific xID
        in the respective XY objects. Should be supplied for custom ordering. If given
        as None (default), teamsheet is extracted from the meta.json file and xIDs are
        assigned based on the ordering determined by the
        ``read_teamsheets_from_metajson`` function (see for details).
    teamsheet_away: Teamsheet, optional
        Teamsheet object for the away team. If given as None (default), teamsheet is
        extracted from the meta.json-file. See teamsheet_home for details.

    Returns
    -------
    data_objects: Tuple[Dict[str, Dict[str, XY]], Dict[str, Code], Dict[str, Code], \
     Dict[str, Teamsheet], Pitch]
        Tuple of (nested) floodlight core objects with shape (xy_objects,
        possession_objects, ballstatus_objects, teamsheets, pitch).

        ``xy_objects`` is a nested dictionary containing ``XY`` objects for each team
        and segment of the form ``xy_objects[segment][team] = XY``. For a typical
        league match with two halves and teams this dictionary looks like:
        ``{'HT1': {'Home': XY, 'Away': XY}, 'HT2': {'Home': XY, 'Away': XY}}``.

        ``possession_objects`` is a dictionary containing ``Code`` objects with
        possession information (home or away) for each segment of the form
        ``possession_objects[segment] = Code``.

        ``ballstatus_objects`` is a dictionary containing ``Code`` objects with
        ballstatus information (dead or alive) for each segment of the form
        ``ballstatus_objects[segment] = Code``.

        ``teamsheets`` is a dictionary containing ``Teamsheet`` objects for each team
        of the form ``teamsheets[team] = Teamsheet``.

        ``pitch`` is a ``Pitch`` object corresponding to the data.
    """
    # setup
    metadata, periods, directions, pitch = _read_metajson(str(filepath_metadata))
    segments = list(periods.keys())
    teams = ["Home", "Away"]
    fps = int(metadata["framerate"])
    status_link = {True: "A", False: "D"}
    key_map = {"Home": "homePlayers", "Away": "awayPlayers"}

    # create or check teamsheet objects
    if teamsheet_home is None and teamsheet_away is None:
        teamsheets = read_teamsheets_from_meta_json(filepath_metadata)
        teamsheet_home = teamsheets["Home"]
        teamsheet_away = teamsheets["Away"]
    elif teamsheet_home is None:
        teamsheets = read_teamsheets_from_meta_json(filepath_metadata)
        teamsheet_home = teamsheets["Home"]
    elif teamsheet_away is None:
        teamsheets = read_teamsheets_from_meta_json(filepath_metadata)
        teamsheet_away = teamsheets["Away"]
    else:
        pass
        # potential check

    # create links
    if "xID" not in teamsheet_home.teamsheet.columns:
        teamsheet_home.add_xIDs()
    if "xID" not in teamsheet_away.teamsheet.columns:
        teamsheet_away.add_xIDs()
    links_jID_to_xID = {
        "Home": teamsheet_home.get_links("jID", "xID"),
        "Away": teamsheet_away.get_links("jID", "xID"),
    }

    # bins
    xydata = {
        team: {
            segment: np.full(
                [
                    periods[segment][1] - periods[segment][0],  # T frames in segment
                    len(links_jID_to_xID[team]) * 2,  # N players in team
                ],
                np.nan,
            )
            for segment in segments
        }
        for team in teams
    }
    xydata["Ball"] = {
        segment: np.full([periods[segment][1] - periods[segment][0], 2], np.nan)
        for segment in segments
    }
    codes = {
        code: {
            segment: np.full(
                [periods[segment][1] - periods[segment][0]], np.nan, dtype=object
            )
            for segment in segments
        }
        for code in ["possession", "ballstatus"]
    }

    # loop
    with open(str(filepath_position), "r") as f:
        while True:
            # get one line of file
            dataline = f.readline()
            # terminate if at end of file
            if len(dataline) == 0:
                break
            # load json
            dataline = json.loads(dataline)

            # get dataline meta information and correct for frame offset
            segment = f"HT{dataline['period']}"
            frame_abs = dataline["frameIdx"]
            frame_rel = frame_abs - periods[segment][0]

            # insert (x,y)-data into correct np.array, at correct place (t, xID)
            for team in teams:
                player_data = dataline[key_map[team]]
                for player in player_data:
                    # map jersey number to array index and infer respective columns
                    jID = player["number"]
                    x_col = (links_jID_to_xID[team][jID]) * 2
                    y_col = (links_jID_to_xID[team][jID]) * 2 + 1
                    xydata[team][segment][frame_rel, (x_col, y_col)] = player["xyz"][:2]

            # get ball data
            if "ball" in dataline and dataline["ball"].get("xyz") is not None:
                xydata["Ball"][segment][frame_rel] = dataline["ball"]["xyz"][:2]

            # get codes
            if "lastTouch" in dataline:
                possession = dataline["lastTouch"][0].upper()
            else:
                possession = None
            codes["possession"][segment][frame_rel] = possession
            if "live" in dataline:
                ballstatus = status_link[dataline["live"]]
            else:
                ballstatus = None
            codes["ballstatus"][segment][frame_rel] = ballstatus

    # create objects
    xy_objects = {}
    possession_objects = {}
    ballstatus_objects = {}
    for segment in segments:
        xy_objects[segment] = {}
        possession_objects[segment] = Code(
            code=codes["possession"][segment],
            name="possession",
            definitions={"H": "Home", "A": "Away"},
            framerate=fps,
        )
        ballstatus_objects[segment] = Code(
            code=codes["ballstatus"][segment],
            name="ballstatus",
            definitions={"D": "Dead", "A": "Alive"},
            framerate=fps,
        )
        for team in ["Home", "Away"]:
            xy_objects[segment][team] = XY(
                xy=xydata[team][segment],
                framerate=fps,
                direction=directions[segment][team],
            )
        xy_objects[segment]["Ball"] = XY(xy=xydata["Ball"][segment], framerate=fps)
    teamsheets = {
        "Home": teamsheet_home,
        "Away": teamsheet_away,
    }

    # pack objects
    data_objects = (
        xy_objects,
        possession_objects,
        ballstatus_objects,
        teamsheets,
        pitch,
    )

    return data_objects


def read_event_data_jsonl(
    filepath_insight: Union[str, Path],
    filepath_metadata: Union[str, Path],
) -> Tuple[Dict[str, Dict[str, Events]], Pitch]:
    """Parse Second Spectrums's Insight file (containing match events) and extract
    event data and pitch information.

    This function provides a high-level access to the particular Second Spectrum
    Insight file and will return event objects for both teams. The number of segments is
    inferred from the data, yet data for each segment is stored in a separate object.

    Parameters
    ----------
    filepath_insight: str or pathlib.Path
        Full path to .jsonl-file.
    filepath_metadata: str or pathlib.Path
        Full path to _meta.json file.

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
    Second Spectrum's Insight files can be seen as a union of (wrapped) Opta event feeds
    and attached Second Spectrum markings. Thus, properties of the Opta F24-feed parser
    also mostly apply to this parser. This particularly includes the handling of
    qualifiers, which are included as a string in the ``qualifier`` column of the
    returned DataFrame's. Second Spectrum markings are disregarded at this moment, but
    could be included in future releases.
    """
    # 1. assemble meta info
    metadata, periods, _, _ = _read_metajson(filepath_metadata)
    teams = ["Home", "Away"]
    tID_link = {
        metadata["home_tID"]: "Home",
        metadata["away_tID"]: "Away",
    }
    segments = periods.keys()

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

    # parse file for meta events (directions and kick-offs)
    with open(str(filepath_insight), "r") as f:
        while True:
            # get one line of file
            dataline = f.readline()
            # terminate if at end of file
            if len(dataline) == 0:
                break
            # load json
            dataline = json.loads(dataline)
            optaline = dataline["optaEvent"]
            if optaline is None:
                continue
            if optaline["typeId"] == 32:
                # get team and segment information
                period = get_and_convert(optaline, "periodId", int)
                segment = "HT" + str(period)
                tID = get_and_convert(optaline, "opContestantId", int)
                team = tID_link[tID]
                # read kickoff times
                kickoff_timestring = get_and_convert(optaline, "timeStamp", str)
                kickoff_datetime = iso8601.parse_date(
                    kickoff_timestring, default_timezone=pytz.utc
                )
                kickoffs[segment] = kickoff_datetime
                # read playing direction
                direction = None
                for qualifier in get_and_convert(optaline, "qualifier", list, []):
                    if get_and_convert(qualifier, "qualifierId", int) == 127:
                        value = get_and_convert(qualifier, "value", str)
                        direction = dir_link.get(value)
                directions[team][segment] = direction

    # parse file for match events
    with open(str(filepath_insight), "r") as f:
        while True:
            # get one line of file
            dataline = f.readline()
            # terminate if at end of file
            if len(dataline) == 0:
                break
            # load json
            dataline = json.loads(dataline)
            optaline = dataline["optaEvent"]
            # secspecline = dataline["2sMarking"]
            if optaline is None:
                continue

            # get team and segment information
            period = get_and_convert(optaline, "periodId", int)
            segment = "HT" + str(period)
            tID = get_and_convert(optaline, "opContestantId", int)
            team = tID_link[tID]

            # skip match-unrelated events
            if period not in range(1, 6):
                continue

            # identifier and outcome:
            eID = get_and_convert(optaline, "typeId", int)
            # skip unwanted events
            if eID in [30]:
                continue
            pID = get_and_convert(optaline, "opPlayerId", int)
            outcome = get_and_convert(optaline, "outcome", int)
            event_lists[team][segment]["eID"].append(eID)
            event_lists[team][segment]["pID"].append(pID)
            event_lists[team][segment]["outcome"].append(outcome)

            # absolute and relative time
            event_timestring = get_and_convert(optaline, "timeStamp", str)
            minute = get_and_convert(optaline, "timeMin", int)
            # transform minute to be relative to current segment
            minute -= segment_offsets[period]
            second = get_and_convert(optaline, "timeSec", int)
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
            at_x = get_and_convert(optaline, "x", float)
            at_y = get_and_convert(optaline, "y", float)
            event_lists[team][segment]["at_x"].append(at_x)
            event_lists[team][segment]["at_y"].append(at_y)

            # qualifier
            qual_dict = {}
            for qualifier in get_and_convert(optaline, "qualifier", list, []):
                qual_id = get_and_convert(qualifier, "qualifierId", int)
                qual_value = qualifier.get("value")
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
    pitch = Pitch.from_template(
        "opta", length=metadata["length"], width=metadata["width"], sport="football"
    )

    # pack objects
    data_objects = (events_objects, pitch)

    return data_objects

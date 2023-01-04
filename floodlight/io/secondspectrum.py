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


def create_links_from_metajson(
    filepath_metadata: Union[str, Path]
) -> Dict[str, Dict[int, int]]:
    """Parses the Second Spectrum meta.json-file for unique jIDs (jerseynumbers) and
    creates a dictionary linking jIDs to xIDs ordered by position precedence.

    Parameters
    ----------
    filepath_metadata: str or pathlib.Path
        Full path to _meta.json file.

    Returns
    -------
    links: Dict[str, Dict[int, int]]
        Link-dictionary of the form `links[team][jID] = xID`.

    Notes
    -----
    The ordering is determined by position precedence, with the following precedence
    values assigned to Second Spectrum's player position information::

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

    # bin
    links_jID_to_xID = {}

    # loop through teams
    key_map = {"Home": "homePlayers", "Away": "awayPlayers"}
    for team in ["Home", "Away"]:
        # bin (team)
        links = []
        # query team player list
        player_list = metajson[key_map[team]]
        # add players to list
        for player in player_list:
            number = player["number"]
            position = player["position"]
            precedence = _get_position_precedence(position)
            links.append((number, precedence))

        # sort link list by position precedence
        links.sort(key=lambda p: p[1])
        # create and add link dict from sorted list
        links = {player[0]: idx for idx, player in enumerate(links)}
        links_jID_to_xID[team] = links

    return links_jID_to_xID


def read_secspec_files(
    filepath_tracking: Union[str, Path],
    filepath_metadata: Union[str, Path],
):
    """Parse Second Spectrum files and extract position data, possession and ballstatus
    codes, as well as pitch information.

    Second Spectrum data is typically stored in two separate files, a .jsonl file
    containing the actual data as well as a _meta.json containing information about
    pitch size, framerate, lineups and start- and endframes of match periods. This
    function provides a high-level access to Second Spectrum data by parsing "the full
    match" given both files.

    Parameters
    ----------
    filepath_tracking: str or pathlib.Path
        Full path to .jsonl-file.
    filepath_metadata: str or pathlib.Path
        Full path to _meta.json file.

    Returns
    -------
    data_objects: Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Code, Code, Pitch]
        XY-, Code-, and Pitch-objects for both teams and both halves. The order is
        (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2,
        possession_ht1, possession_ht2, ballstatus_ht1, ballstatus_ht2, pitch)
    """
    # setup
    metadata, periods, directions, pitch = _read_metajson(str(filepath_metadata))
    links = create_links_from_metajson(str(filepath_metadata))
    segments = list(periods.keys())
    teams = list(links.keys())
    fps = int(metadata["framerate"])
    status_link = {True: "A", False: "D"}
    key_map = {"Home": "homePlayers", "Away": "awayPlayers"}

    # bins
    xydata = {
        team: {
            segment: np.full(
                [
                    periods[segment][1] - periods[segment][0],  # T frames in segment
                    len(links[team]) * 2,  # N players in team
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
    with open(str(filepath_tracking), "r") as f:
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
                    x_col = (links[team][jID]) * 2
                    y_col = (links[team][jID]) * 2 + 1
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

    # assemble core objects
    home_ht1 = XY(
        xy=xydata["Home"]["HT1"], framerate=fps, direction=directions["HT1"]["Home"]
    )
    home_ht2 = XY(
        xy=xydata["Home"]["HT2"], framerate=fps, direction=directions["HT2"]["Home"]
    )
    away_ht1 = XY(
        xy=xydata["Away"]["HT1"], framerate=fps, direction=directions["HT1"]["Away"]
    )
    away_ht2 = XY(
        xy=xydata["Away"]["HT2"], framerate=fps, direction=directions["HT2"]["Away"]
    )
    ball_ht1 = XY(xy=xydata["Ball"]["HT1"], framerate=fps)
    ball_ht2 = XY(xy=xydata["Ball"]["HT2"], framerate=fps)

    possession_ht1 = Code(
        code=codes["possession"]["HT1"],
        name="possession",
        definitions={"H": "Home", "A": "Away"},
        framerate=fps,
    )
    possession_ht2 = Code(
        code=codes["possession"]["HT2"],
        name="possession",
        definitions={"H": "Home", "A": "Away"},
        framerate=fps,
    )
    ballstatus_ht1 = Code(
        code=codes["ballstatus"]["HT1"],
        name="ballstatus",
        definitions={"D": "Dead", "A": "Alive"},
        framerate=fps,
    )
    ballstatus_ht2 = Code(
        code=codes["ballstatus"]["HT2"],
        name="ballstatus",
        definitions={"D": "Dead", "A": "Alive"},
        framerate=fps,
    )

    data_objects = (
        home_ht1,
        home_ht2,
        away_ht1,
        away_ht2,
        ball_ht1,
        ball_ht2,
        possession_ht1,
        possession_ht2,
        ballstatus_ht1,
        ballstatus_ht2,
        pitch,
    )

    return data_objects


def read_secspec_insight(
    filepath_insight: Union[str, Path],
    filepath_metadata: Union[str, Path],
) -> Tuple[Events, Events, Events, Events, Pitch]:
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
    data_objects: Tuple[Events, Events, Events, Events, Pitch]
        Events- and Pitch-objects for both teams and both halves. The order is
        (home_ht1, home_ht2, away_ht1, away_ht2, pitch)

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

        # assembly
        home_ht1 = Events(
            events=pd.DataFrame(data=event_lists["Home"]["HT1"]),
            direction=directions["Home"]["HT1"],
        )
        home_ht2 = Events(
            events=pd.DataFrame(data=event_lists["Home"]["HT2"]),
            direction=directions["Home"]["HT2"],
        )
        away_ht1 = Events(
            events=pd.DataFrame(data=event_lists["Away"]["HT1"]),
            direction=directions["Away"]["HT1"],
        )
        away_ht2 = Events(
            events=pd.DataFrame(data=event_lists["Away"]["HT2"]),
            direction=directions["Away"]["HT2"],
        )
        pitch = Pitch.from_template(
            "opta", length=metadata["length"], width=metadata["width"], sport="football"
        )

        data_objects = (home_ht1, home_ht2, away_ht1, away_ht2, pitch)

        return data_objects

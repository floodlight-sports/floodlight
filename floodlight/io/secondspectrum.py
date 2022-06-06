import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from floodlight.core.code import Code
from floodlight.core.pitch import Pitch
from floodlight.core.xy import XY


def _get_position_precedence(position: str) -> int:
    """Get the precedence of a position abbreviation (e.g. 'GK') useful for ordering."""
    PLAYER_POSITION_PRECEDENCE = {
        "GK": 1,
        "LB": 2,
        "LCB": 3,
        "CB": 4,
        "RCB": 5,
        "RB": 6,
        "LDM": 7,
        "CDM": 8,
        "RDM": 9,
        "LM": 10,
        "LCM": 11,
        "CM": 12,
        "RCM": 13,
        "RM": 14,
        "LW": 15,
        "CAM": 16,
        "RW": 17,
        "LF": 18,
        "LCF": 19,
        "CF": 20,
        "RCF": 21,
        "RF": 22,
        "SUB": 99,
    }
    # unknown positions are inserted between known positions and substitutes
    precedence = PLAYER_POSITION_PRECEDENCE.get(position, 23)

    return precedence


def _read_metajson(
    filepath_metadata: Union[str, Path]
) -> Tuple[Dict, Dict, Dict, Pitch]:
    """Reads TRACAB's metadata file and extracts information about match metainfo,
    periods, player directions, and the pitch.

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
            'LDM': 7,
            'CDM': 8,
            'RDM': 9,
            'LM': 10,
            'LCM': 11,
            'CM': 12,
            'RCM': 13,
            'RM': 14,
            'LW': 15,
            'CAM': 16,
            'RW': 17,
            'LF': 18,
            'LCF': 19,
            'CF': 20,
            'RCF': 21,
            'RF': 22,
            'UNKNOWN': 23,
            'SUB': 99,
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
    function provides a high-level access to TRACAB data by parsing "the full match"
    given both files.

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

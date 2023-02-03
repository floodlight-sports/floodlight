import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from lxml import etree

from floodlight.core.code import Code
from floodlight.core.pitch import Pitch
from floodlight.core.xy import XY
from floodlight.core.teamsheet import Teamsheet
from floodlight.io.utils import get_and_convert


def _read_metadata_from_xml(
    filepath_metadata: Union[str, Path]
) -> Tuple[Dict, Dict, Pitch]:
    """Reads TRACAB's metadata file (xml format) and extracts match meta information
    such as framerate, periods and pitch.

    Parameters
    ----------
    filepath_metadata: str or pathlib.Path
        Full path to _metadata.xml file.

    Returns
    -------
    metainfo: Dict
        Dictionary with metainformation such as framerate.
    periods: Dict
        Dictionary with start and endframes:
        `periods[segment] = (startframe, endframe)`.
    pitch: Pitch
        Pitch object with actual pitch length and width.
    """
    #  set up XML tree
    tree = etree.parse(str(filepath_metadata))
    root = tree.getroot()

    # parse XML file, extract matchinfo and period start/endframes
    metadata = {}
    periods = {}
    attributes = root.find("match").attrib

    framerate = attributes.get("iFrameRateFps")
    metadata["framerate"] = int(framerate) if framerate else None

    length = attributes.get("fPitchXSizeMeters")
    metadata["length"] = float(length) if length else None

    width = attributes.get("fPitchYSizeMeters")
    metadata["width"] = float(width) if width else None

    for elem in root.findall("match/period"):
        if elem.attrib["iEndFrame"] != "0":
            segment = "HT" + elem.attrib["iId"]
            start = int(elem.attrib["iStartFrame"])
            end = int(elem.attrib["iEndFrame"])
            periods[segment] = (start, end)

    pitch = Pitch.from_template(
        "tracab",
        length=float(metadata["length"]),
        width=float(metadata["width"]),
        sport="football",
    )

    return metadata, periods, pitch


def _read_metadata_from_json(
    filepath_metadata: Union[str, Path]
) -> Tuple[Dict, Dict, Pitch]:
    """Reads TRACAB's metadata file (json format) and extracts match meta information
    such as framerate, periods and pitch.

    Parameters
    ----------
    filepath_metadata: str or pathlib.Path
        Full path to _metadata.json file.

    Returns
    -------
    metadata: Dict
        Dictionary with metainformation such as framerate.
    periods: Dict
        Dictionary with start and endframes:
        `periods[segment] = (startframe, endframe)`.
    pitch: Pitch
        Pitch object with actual pitch length and width.
    """
    # load file
    with open(filepath_metadata, "r", encoding="utf8") as f:
        metafile = json.load(f)

    # bin
    metadata = {}
    periods = {}

    # get framerate
    metadata["framerate"] = get_and_convert(metafile, "FrameRate", int)

    # get length and with and convert from cm to m
    length = get_and_convert(metafile, "PitchLongSide", float)
    width = get_and_convert(metafile, "PitchShortSide", float)
    metadata["length"] = length / 100 if length else None
    metadata["width"] = width / 100 if width else None

    # get period start and end frames
    for i in range(1, 6):
        phase = f"Phase{i}"
        ht = f"HT{i}"
        phase_start = get_and_convert(metafile, phase + "StartFrame", int)
        phase_end = get_and_convert(metafile, phase + "EndFrame", int)
        if phase_start is None or phase_end is None:
            continue
        if phase_start == 0 or phase_end == 0:
            continue
        periods[ht] = (phase_start, phase_end)

    # create pitch
    pitch = Pitch.from_template(
        "tracab",
        length=metadata["length"],
        width=metadata["width"],
        sport="football",
    )

    return metadata, periods, pitch


def _read_dat_single_line(
    package: str,
) -> Tuple[
    int, Dict[str, Dict[str, Tuple[float, float, float]]], Dict[str, Union[str, tuple]]
]:
    """Extracts all relevant information from a single line of TRACAB's .dat file
    (i.e. one frame of data).

    Parameters
    ----------
    package: str
        One full line from TRACAB's .dat-file, equals one "package" according to the
        file-format documentation.

    Returns
    -------
    frame_number: int
        The number of current frame.
    positions: Dict[str, Dict[str, Tuple[float, float, float]]]
        Nested dictionary that stores player position information for each team and
        player. Has the form `positions[team][jID] = (x, y, speed)`.
    ball: Dict[str]
        Dictionary with ball information. Has keys 'position', 'possession' and
        'ballstatus'.
    """
    # bins
    positions = {"Home": {}, "Away": {}, "Other": {}}
    ball = {}

    # split package to chunks
    chunk1, chunk2, chunk3, _ = package.split(sep=":")

    # first chunk (frame number)
    frame_number = int(chunk1)

    # second chunk (player positions)
    targets = chunk2[:-1].split(sep=";")
    for t in targets:
        player_data = t.split(sep=",")
        # type conversions
        team, system_id, jID = map(lambda x: int(x), player_data[:3])
        x, y, speed = map(lambda x: float(x), player_data[3:])
        if team == 1:
            team = "Home"
        elif team == 0:
            team = "Away"
        else:
            team = "Other"
        # assign
        positions[team][jID] = (x, y, speed)

    # third chunk (ball data)
    ball_data = chunk3.split(sep=",")[:6]
    ball["position"] = tuple(map(lambda x: float(x), ball_data[:2]))
    ball["possession"] = ball_data[4]
    ball["ballstatus"] = ball_data[5][0]

    return frame_number, positions, ball


def _frame_in_period(
    frame_number: int, periods: Dict[str, Tuple[int, int]]
) -> Union[str, None]:
    """Checks if a given frame is within the range of start- and endframe for all
    periods and returns the name of the period the frame belongs to, or None if it
    can't find any.

    Parameters
    ----------
    frame_number: int
        Frame number to be checked.
    periods: Dict[str, Tuple[int, int]]
        Dictionary with period start- and endframes of the form
        `periods[segment] = (startframe, endframe)` as it is returned by
        :meth:`floodlight.io.tracab._read_metadata`.

    Returns
    -------
    segment: str or None
        Name of the segment the frame belongs to, or None if it does not belong to any
        of the supplied segments.
    """
    # determine current segment by iterating through all segments (i)
    segment = None
    for i in periods.keys():
        if frame_number in range(periods[i][0], periods[i][1] + 1):
            segment = i

    return segment


def _read_dat_jersey_numbers(filepath_dat: Union[str, Path]):
    """Reads entire TRACAB .dat file and extracts unique set of jIDs (jerseynumbers)
    for both teams.

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        Full path to .dat file.

    Returns
    -------
    home_jIDs: set
    away_jIDs: set
    """
    # bins
    home_jIDs = set()
    away_jIDs = set()
    # loop
    with open(str(filepath_dat), "r") as f:
        while True:
            package = f.readline()
            # terminate if at end of file
            if len(package) == 0:
                break
            # read line
            _, positions, _ = _read_dat_single_line(package)
            # Extract jersey numbers
            home_jIDs |= positions["Home"].keys()
            away_jIDs |= positions["Away"].keys()

    return home_jIDs, away_jIDs


def read_teamsheets_from_dat(filepath_dat: Union[str, Path]) -> Dict[str, Teamsheet]:
    """Parses the entire TRACAB .dat file for unique jIDs (jerseynumbers) and creates
    respective teamsheets for the home and the away team.

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        Full path to .dat file.

    Returns
    -------
    teamsheets: Dict[str, Teamsheet]
        Dictionary with teamsheets for the home team and the away team.
    """
    # bin
    teamsheets = {}

    # get jerseynumbers (jIDs)
    homejrsy, awayjrsy = _read_dat_jersey_numbers(filepath_dat)

    # loop through teams
    for team, jIDs in zip(("Home", "Away"), (homejrsy, awayjrsy)):
        jIDs = list(jIDs)
        jIDs.sort()
        player = [f"Player {i+1}" for i in range(len(jIDs))]
        teamsheet = pd.DataFrame(
            data={
                "player": player,
                "jID": jIDs,
            }
        )
        teamsheet = Teamsheet(teamsheet)
        teamsheets[team] = teamsheet

    return teamsheets


def read_teamsheets_from_meta_json(
    filepath_metadata: Union[str, Path]
) -> Dict[str, Teamsheet]:
    """Reads TRACAB's metadata file (json format) and creates respective teamsheets for
    the home and the away team.

    Parameters
    ----------
    filepath_metadata: str or pathlib.Path
        Full path to _metadata.json file.

    Returns
    -------
    teamsheets: Dict[str, Teamsheet]
        Dictionary with teamsheets for the home team and the away team.
    """
    # load file
    with open(filepath_metadata, "r", encoding="utf8") as f:
        metafile = json.load(f)

    # param
    teams = ["Home", "Away"]

    # bin
    teamsheets = {team: {var: [] for var in ["player", "pID", "jID"]} for team in teams}

    # loop through teams
    for team in teams:
        team_name = team + "Team"
        for player in metafile[team_name]["Players"]:
            first_name = get_and_convert(player, "FirstName", str, "")
            last_name = get_and_convert(player, "LastName", str, "")
            full_name = first_name + " " + last_name
            teamsheets[team]["player"].append(full_name)
            teamsheets[team]["pID"].append(get_and_convert(player, "PlayerID", int))
            teamsheets[team]["jID"].append(get_and_convert(player, "JerseyNo", int))

        teamsheets[team] = Teamsheet(pd.DataFrame(teamsheets[team]))

    return teamsheets


def read_position_data_dat(
    filepath_dat: Union[str, Path],
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
    """Parse TRACAB .dat-files (ASCII) and metadata (xml or json) and extract position
    data, possession and ballstatus codes, teamsheets as well as pitch information.

    ChyronHego's TRACAB system delivers two separate files, a .dat file containing the
    actual data as well as a metadata.xml containing information about pitch size,
    framerate and start- and endframes of match periods. This function provides a
    high-level access to TRACAB data by parsing "the full match" given both files.

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        Full path to dat-file.
    filepath_metadata: str or pathlib.Path
        Full path to metadata.xml file.
    teamsheet_home: Teamsheet, optional
        Teamsheet object for the home team used to create link dictionaries of the form
        `links[team][jID] = xID`. The links are used to map players to a specific xID
        in the respective XY objects. Should be supplied for custom ordering. If given
        as None (default), teamsheet is extracted from the .dat or .json file (see
        Notes) and xIDs are assigned to the player's jersey numbers ascendingly (dat
        case) or in order of appearance (json case).
    teamsheet_away: Teamsheet, optional
        Teamsheet object for the away team. If given as None (default), teamsheet is
        extracted from the .dat or -json file. See teamsheet_home for details.

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

    Notes
    -----
    Tracab provides metadata in two file types: xml and json. The json metadata files
    typically include player information whereas the xml files do not. The dat file
    storing tracking data (e.g. from an ASCII stream) contain only player jersey
    numbers, but no additional player information.

    This function will check whether the provided ``filepath_metadata`` points to a xml
    or json file. If it's a json, teamsheets are generated from this source. If it's a
    xml, teamsheets are generated from the dat file and players are named 'Player i'
    with i starting at 1. To identify players in this case, use the jersey numbers or
    provide custom teamsheets generated by a different parser if additional data is
    available.
    """
    # check file type of metadata
    file_extension = filepath_metadata.split(".")[-1].upper()

    # read metadata and determine logic used for teamsheet parsing
    if file_extension == "XML":
        metadata, periods, pitch = _read_metadata_from_xml(filepath_metadata)
        teamsheet_parse_func = read_teamsheets_from_dat
        teamsheet_parse_file = filepath_dat
    elif file_extension == "JSON":
        metadata, periods, pitch = _read_metadata_from_json(filepath_metadata)
        teamsheet_parse_func = read_teamsheets_from_meta_json
        teamsheet_parse_file = filepath_metadata
    else:
        raise ValueError(
            f"Expected metadata file type to be from [XML, JSON], got {file_extension}."
        )
    segments = list(periods.keys())

    # create or check teamsheet objects with select teamsheet parsing functions & file
    if teamsheet_home is None and teamsheet_away is None:
        teamsheets = teamsheet_parse_func(teamsheet_parse_file)
        teamsheet_home = teamsheets["Home"]
        teamsheet_away = teamsheets["Away"]
    elif teamsheet_home is None:
        teamsheets = teamsheet_parse_func(teamsheet_parse_file)
        teamsheet_home = teamsheets["Home"]
    elif teamsheet_away is None:
        teamsheets = teamsheet_parse_func(teamsheet_parse_file)
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

    # infer data array shapes
    number_of_home_players = max(links_jID_to_xID["Home"].values()) + 1
    number_of_away_players = max(links_jID_to_xID["Away"].values()) + 1
    number_of_frames = {}
    for segment in segments:
        start = periods[segment][0]
        end = periods[segment][1]
        number_of_frames[segment] = end - start + 1

    # bins
    xydata = {}
    xydata["Home"] = {
        segment: np.full(
            [number_of_frames[segment], number_of_home_players * 2], np.nan
        )
        for segment in segments
    }
    xydata["Away"] = {
        segment: np.full(
            [number_of_frames[segment], number_of_away_players * 2], np.nan
        )
        for segment in segments
    }
    xydata["Ball"] = {
        segment: np.full([number_of_frames[segment], 2], np.nan) for segment in segments
    }
    codes = {
        code: {segment: [] for segment in segments}
        for code in ["possession", "ballstatus"]
    }

    # loop
    with open(filepath_dat, "r") as f:
        while True:
            package = f.readline()
            # terminate if at end of file
            if len(package) == 0:
                break
            # read line to get absolute frame (in file), player positions and ball info
            frame_abs, positions, ball = _read_dat_single_line(package)

            # check if frame is in any segment
            segment = _frame_in_period(frame_abs, periods)
            if segment is None:
                # skip line if not
                continue
            else:
                # otherwise calculate relative frame (in respective segment)
                frame_rel = frame_abs - periods[segment][0]

            # insert (x,y)-data into correct np.array, at correct place (t, xID)
            for team in ["Home", "Away"]:
                for jID in positions[team].keys():
                    # map jersey number to array index and infer respective columns
                    x_col = (links_jID_to_xID[team][jID]) * 2
                    y_col = (links_jID_to_xID[team][jID]) * 2 + 1
                    xydata[team][segment][frame_rel, x_col] = positions[team][jID][0]
                    xydata[team][segment][frame_rel, y_col] = positions[team][jID][1]

            # get ball data
            xydata["Ball"][segment][
                frame_rel,
            ] = ball["position"]
            codes["possession"][segment].append(ball.get("possession", np.nan))
            codes["ballstatus"][segment].append(ball.get("ballstatus", np.nan))

    # create objects
    xy_objects = {}
    possession_objects = {}
    ballstatus_objects = {}
    for segment in segments:
        xy_objects[segment] = {}
        possession_objects[segment] = Code(
            code=np.array(codes["possession"][segment]),
            name="possession",
            definitions={"H": "Home", "A": "Away"},
            framerate=metadata["framerate"],
        )
        ballstatus_objects[segment] = Code(
            code=np.array(codes["ballstatus"][segment]),
            name="ballstatus",
            definitions={"D": "Dead", "A": "Alive"},
            framerate=metadata["framerate"],
        )
        for team in ["Home", "Away", "Ball"]:
            xy_objects[segment][team] = XY(
                xy=xydata[team][segment], framerate=metadata["framerate"]
            )
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

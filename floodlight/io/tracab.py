from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from lxml import etree

from floodlight.core.code import Code
from floodlight.core.pitch import Pitch
from floodlight.core.xy import XY


def _read_metadata(filepath_metadata: Union[str, Path]) -> Tuple[Dict, Dict, Pitch]:
    """Reads TRACAB's metadata file and extracts information about match metainfo,
    periods and the pitch.

    Parameters
    ----------
    filepath_metadata: str or pathlib.Path
        Full path to metadata.xml file.

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


def create_links_from_dat(filepath_dat: Union[str, Path]) -> Dict[str, Dict[int, int]]:
    """Parses the entire TRACAB .dat file for unique jIDs (jerseynumbers) and creates a
    dictionary linking jIDs to xIDs in ascending order.

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        Full path to .dat file.

    Returns
    -------
    links: Dict[str, Dict[int, int]]
        Link-dictionary of the form `links[team][jID] = xID`.
    """
    homejrsy, awayjrsy = _read_dat_jersey_numbers(filepath_dat)

    homejrsy = list(homejrsy)
    awayjrsy = list(awayjrsy)

    homejrsy.sort()
    awayjrsy.sort()

    links = {
        "Home": {jID: xID for xID, jID in enumerate(homejrsy)},
        "Away": {jID: xID for xID, jID in enumerate(awayjrsy)},
    }

    return links


def read_tracab_files(
    filepath_dat: Union[str, Path],
    filepath_metadata: Union[str, Path],
    links: Dict[str, Dict[int, int]] = None,
) -> Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Code, Code, Pitch]:
    """Parse TRACAB files and extract position data, possession and ballstatus codes as
    well as pitch information.

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
    links: Dict[str, Dict[int, int]], optional
        A link dictionary of the form `links[team][jID] = xID`. Player's are identified
        in TRACAB files via jID, and this dictionary is used to map them to a specific
        xID in the respective XY objects. Should be supplied if that order matters. If
        None is given (default), the links are automatically extracted from the .dat
        file at the cost of a second pass through the entire file.

    Returns
    -------
    data_objects: Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Code, Code, Pitch]
        XY-, Code-, and Pitch-objects for both teams and both halves. The order is
        (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2,
        possession_ht1, possession_ht2, ballstatus_ht1, ballstatus_ht2, pitch)
    """
    # read metadata
    metadata, periods, pitch = _read_metadata(filepath_metadata)
    segments = list(periods.keys())

    # create or check links
    if links is None:
        links = create_links_from_dat(filepath_dat)
    else:
        pass
        # potential check vs jerseys in dat file

    # infer data array shapes
    number_of_home_players = max(links["Home"].values()) + 1
    number_of_away_players = max(links["Away"].values()) + 1
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
                    x_col = (links[team][jID]) * 2
                    y_col = (links[team][jID]) * 2 + 1
                    xydata[team][segment][frame_rel, x_col] = positions[team][jID][0]
                    xydata[team][segment][frame_rel, y_col] = positions[team][jID][1]

            # get ball data
            xydata["Ball"][segment][
                frame_rel,
            ] = ball["position"]
            codes["possession"][segment].append(ball.get("possession", np.nan))
            codes["ballstatus"][segment].append(ball.get("ballstatus", np.nan))

    # create XY objects
    home_ht1 = XY(xy=xydata["Home"]["HT1"], framerate=metadata["framerate"])
    home_ht2 = XY(xy=xydata["Home"]["HT2"], framerate=metadata["framerate"])
    away_ht1 = XY(xy=xydata["Away"]["HT1"], framerate=metadata["framerate"])
    away_ht2 = XY(xy=xydata["Away"]["HT2"], framerate=metadata["framerate"])
    ball_ht1 = XY(xy=xydata["Ball"]["HT1"], framerate=metadata["framerate"])
    ball_ht2 = XY(xy=xydata["Ball"]["HT2"], framerate=metadata["framerate"])

    # create Code objects
    possession_ht1 = Code(
        code=np.array(codes["possession"]["HT1"]),
        name="possession",
        definitions={"H": "Home", "A": "Away"},
        framerate=metadata["framerate"],
    )
    possession_ht2 = Code(
        code=np.array(codes["possession"]["HT2"]),
        name="possession",
        definitions={"H": "Home", "A": "Away"},
        framerate=metadata["framerate"],
    )
    ballstatus_ht1 = Code(
        code=np.array(codes["ballstatus"]["HT1"]),
        name="ballstatus",
        definitions={"D": "Dead", "A": "Alive"},
        framerate=metadata["framerate"],
    )
    ballstatus_ht2 = Code(
        code=np.array(codes["ballstatus"]["HT2"]),
        name="ballstatus",
        definitions={"D": "Dead", "A": "Alive"},
        framerate=metadata["framerate"],
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

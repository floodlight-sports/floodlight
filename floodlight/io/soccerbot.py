from pathlib import Path
from typing import Dict, Tuple, Union

from lxml import etree
import numpy as np

from floodlight.core.xy import XY
from floodlight.core.pitch import Pitch
from floodlight.core.code import Code


def _read_mat_info(filepath_mat_info: Union[str, Path]) -> Tuple[Pitch, Dict[str, int]]:
    """Reads match_information XML file and returns the playing Pitch and kickoffs of
    each segment.

    Parameters
    ----------
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in SoccerBot format is
        saved.

    Returns
    -------
    pitch: Pitch
        Pitch object with actual pitch length and width.
    kickoffs: Dict[str, int]
        Dictionary with kickoff frames for the segment:
        `kickoffs[segment] = kickoff_frame`.

    """
    # set up XML tree
    tree = etree.parse(str(filepath_mat_info))
    root = tree.getroot()

    # parse pitch
    length = root.find("Field").get("sx")
    length = float(length) if length else None
    width = root.find("Field").get("sy")
    width = float(width) if width else None
    pitch = Pitch.from_template("soccerbot", length=length, width=width)

    # parse kick offs for all halves
    kickoffs = {}
    for segment in root.find("KickOff").attrib:
        kickoffs[segment] = int(root.find("KickOff").get(segment))

    return pitch, kickoffs


def _create_periods_from_dat(
    filepath_dat: Union[str, Path], kickoffs: Dict[str, int]
) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """Parses over position file and returns a dictionary with periods and the timedelta
    between frames

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        Path to XML File where the Position data in SoccerBot format is saved.

    Returns
    -------
    periods: Dict[str, Tuple[int, int]
        Dictionary with times for the segment:
        `periods[segment] = (starttime, endtime)`.
    timedelta: int
        Number of milliseconds between two frames.
    """
    #  set up XML tree
    tree = etree.parse(str(filepath_dat))
    root = tree.getroot()

    # parse XML file, extract framerate and periods
    end_times = {segment: kickoffs[segment] + 1 for segment in kickoffs}
    last_time = None
    timedelta = None
    for frame in root.find("Positions").iterfind("Timestamp"):
        # read time
        frame_time = int(frame.get("t"))

        # estimate framerate
        if last_time is None:
            last_time = frame_time
        elif timedelta is None:
            # convert millisecond time delta to framerate
            timedelta = frame_time - last_time
        else:
            pass

        # update end times by last frame position data frame in segment
        if len(frame.find("HomeTeam")) and len(frame.find("AwayTeam")):
            segment = None
            for seg, kickoff in sorted(kickoffs.items()):
                if frame_time >= kickoff:
                    segment = seg
            end_times[segment] = frame_time

    # update periods
    periods = {segment: (kickoffs[segment], end_times[segment]) for segment in kickoffs}

    return periods, timedelta


def create_links_from_mat_info(
    filepath_mat_info: Union[str, Path]
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[str, int]]]:
    """Parses the SoccerBot Match Information XML file for unique jIDs (jerseynumbers)
    and creates two dictionaries, one linking pIDs to jIDs and one linking jIDs to xIDs
    in ascending order.

    Parameters
    ----------
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in SoccerBot format is
        saved.

    Returns
    -------
    links: Dict[str, Dict[int, int]]
        A link dictionary of the form `links[team][jID] = xID`.
    id_to_jrsy: Dict[str, Dict[str, int]]
        A link dictionary of the form `links[team][pID] = jID`.
    """
    # set up XML tree
    tree = etree.parse(str(filepath_mat_info))
    root = tree.getroot()

    # parse XML file and extract links from pID to xID for both teams
    id_to_jrsy = {"Home": {}, "Away": {}}
    for player in root.find("Players").iterfind("Player"):
        id_to_jrsy[player.get("team")][player.get("id")] = int(player.get("shirt"))

    links = {
        "Home": {
            int(id_to_jrsy["Home"][pID]): xID + 1
            for xID, pID in enumerate(id_to_jrsy["Home"])
        },
        "Away": {
            int(id_to_jrsy["Away"][pID]): xID + 1
            for xID, pID in enumerate(id_to_jrsy["Away"])
        },
    }
    return links, id_to_jrsy


def read_dfl_files(
    filepath_dat: Union[str, Path],
    filepath_mat_info: Union[str, Path],
    links: Dict[str, Dict[int, int]] = None,
    id_to_jrsy: Dict[str, Dict[str, int]] = None,
) -> Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Code, Code, Pitch]:
    """Parse SoccerBot XML files and extract position data, possession and ballstatus
    codes as well as pitch information.

     Data in the SoccerBot format is given as two separate files, a .dat file containing
     the actual data as well as a metadata.xml containing information about pitch size
     and start- and endframes of match periods. This function provides a high-level
     access to SoccerBot data by parsing "the full match" given both files.

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        Full path to XML File where the Position data in SoccerBot format is saved.
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in SoccerBot format is
        saved.
    links: Dict, optional
        A link dictionary of the form `links[team][jID] = xID`. Player's are identified
        in the XML files via jID, and this dictionary is used to map them to a specific
        xID in the respective XY objects. Should be supplied if that order matters. If
        one of links or id_to_jrsy is given as None (default), they are automatically
        extracted from the Match Information XML file.
    id_to_jrsy: Dict, optional
        A link dictionary of the form `links[team][pID] = jID` where pID is the id
        specified in the SoccerBot Match Information file.

    Returns
    -------
        data_objects: Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Code, Code, Pitch]
        XY-, Code-, and Pitch-objects for both teams and both halves. The order is
        (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2,
    possession_ht1, possession_ht2, ballstatus_ht1, ballstatus_ht2, pitch)
    """
    # checks?

    # read metadata
    pitch, kickoffs = _read_mat_info(filepath_mat_info)
    periods, timedelta = _create_periods_from_dat(filepath_dat, kickoffs)
    segments = list(periods.keys())

    # create or check links
    if links is None:
        links, id_to_jrsy = create_links_from_mat_info(filepath_mat_info)
    else:
        pass
        # potential check

    # infer data array shapes
    number_of_home_players = max(links["Home"].values())
    number_of_away_players = max(links["Away"].values())
    number_of_frames = {}
    for segment in segments:
        number_of_frames[segment] = (
            int(round((periods[segment][1] - periods[segment][0]) / timedelta)) + 1
        )

    # bins
    xydata = {
        "Home": {
            segment: np.full(
                [number_of_frames[segment], number_of_home_players * 2], np.nan
            )
            for segment in segments
        },
        "Away": {
            segment: np.full(
                [number_of_frames[segment], number_of_away_players * 2], np.nan
            )
            for segment in segments
        },
        "Ball": {
            segment: np.full([number_of_frames[segment], 2], np.nan)
            for segment in segments
        },
    }
    codes = {
        code: {segment: [] for segment in segments}
        for code in ["possession", "ballstatus"]
    }

    # loop
    tree = etree.parse(str(filepath_dat))
    root = tree.getroot()
    for positions in root.iterfind("Positions"):
        for frame in positions.iterfind("Timestamp"):
            # skip frames without position data
            if len(frame.find("HomeTeam")) == 0 and len(frame.find("AwayTeam")) == 0:
                break

            # read time and segment
            frame_time = int(frame.get("t"))
            segment = None
            for seg in segments:
                if periods[seg][0] <= frame_time <= periods[seg][1]:
                    segment = seg
            frame_num = int((frame_time - periods[segment][0]) / timedelta)

            # teams
            for position in frame.find("HomeTeam").iterfind("Pos"):
                x_col = (links["Home"][id_to_jrsy["Home"][position.get("id")]] - 1) * 2
                y_col = (
                    links["Home"][id_to_jrsy["Home"][position.get("id")]] - 1
                ) * 2 + 1
                xydata["Home"][segment][frame_num, x_col] = float(position.get("x"))
                xydata["Home"][segment][frame_num, y_col] = float(position.get("y"))
            for position in frame.find("AwayTeam").iterfind("Pos"):
                x_col = (links["Away"][id_to_jrsy["Away"][position.get("id")]] - 1) * 2
                y_col = (
                    links["Away"][id_to_jrsy["Away"][position.get("id")]] - 1
                ) * 2 + 1
                xydata["Away"][segment][frame_num, x_col] = float(position.get("x"))
                xydata["Away"][segment][frame_num, y_col] = float(position.get("y"))

            # ball
            position = frame.find("Ball").find("Pos")
            xydata["Ball"][segment][frame_num, 0] = float(position.get("x"))
            xydata["Ball"][segment][frame_num, 1] = float(position.get("y"))
            codes["possession"][segment].append(position.get("ballPossession"))
            codes["ballstatus"][segment].append(position.get("gameState"))

    # Create XY Objects
    # estimate framerate
    framerate_est = int(1000 / timedelta)  # convert to fps
    home_ht1 = XY(xy=xydata["Home"]["firstHalf"], framerate=framerate_est)
    home_ht2 = XY(xy=xydata["Home"]["secondHalf"], framerate=framerate_est)
    away_ht1 = XY(xy=xydata["Away"]["firstHalf"], framerate=framerate_est)
    away_ht2 = XY(xy=xydata["Away"]["secondHalf"], framerate=framerate_est)
    ball_ht1 = XY(xy=xydata["Ball"]["firstHalf"], framerate=framerate_est)
    ball_ht2 = XY(xy=xydata["Ball"]["secondHalf"], framerate=framerate_est)

    # create Code objects
    possession_ht1 = Code(
        code=np.array(codes["possession"]["firstHalf"]),
        name="possession",
        definitions={"Home": "Home", "Away": "Away", "": None},
        framerate=framerate_est,
    )
    possession_ht2 = Code(
        code=np.array(codes["possession"]["firstHalf"]),
        name="possession",
        definitions={"Home": "Home", "Away": "Away", "": None},
        framerate=framerate_est,
    )
    ballstatus_ht1 = Code(
        code=np.array(codes["ballstatus"]["firstHalf"]),
        name="ballstatus",
        definitions={"dead": "Dead", "active": "Alive"},
        framerate=framerate_est,
    )
    ballstatus_ht2 = Code(
        code=np.array(codes["ballstatus"]["firstHalf"]),
        name="ballstatus",
        definitions={"dead": "Dead", "active": "Alive"},
        framerate=framerate_est,
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

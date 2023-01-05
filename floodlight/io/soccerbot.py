from pathlib import Path
from typing import Dict, Tuple, Union

from lxml import etree
import numpy as np

from floodlight.core.xy import XY
from floodlight.core.pitch import Pitch
from floodlight.core.code import Code


def _read_mat_info_xml(
    filepath_mat_info: Union[str, Path]
) -> Tuple[Pitch, Dict[str, int]]:
    """Reads Match Information XML file and returns the playing pitch and kickoffs of
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

    # parse kickoffs for all halves
    kickoffs = {}
    for segment in root.find("KickOff").attrib:
        kickoffs[segment] = int(root.find("KickOff").get(segment))

    return pitch, kickoffs


def _read_periods_and_timedelta_from_positions_xml(
    filepath_positions: Union[str, Path], kickoffs: Dict[str, int]
) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """Parses over position xml file and returns a dictionary with periods as well as a
    timedelta between frames that is later used to estimate the framerate.

    Parameters
    ----------
    filepath_positions: str or pathlib.Path
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
    tree = etree.parse(str(filepath_positions))
    root = tree.getroot()

    # parse XML file, extract framerate and periods
    end_times = {segment: kickoffs[segment] + 1 for segment in kickoffs}
    last_time = None
    timedelta = None
    for frame in root.find("Positions").iterfind("Timestamp"):
        # read time
        frame_time = int(frame.get("t"))

        # compute timedelta (used for estimating framerate)
        if last_time is None:
            last_time = frame_time
        elif timedelta is None:
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


def create_links_from_mat_info_xml(
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

    # parse XML file and extract links for both teams

    # pID to jID
    links_pID_to_jID = {"Home": {}, "Away": {}}
    for player in root.find("Players").iterfind("Player"):
        links_pID_to_jID[player.get("team")][player.get("id")] = int(
            player.get("shirt")
        )

    # jID to xID
    links_jID_to_xID = {
        "Home": {
            int(links_pID_to_jID["Home"][pID]): xID + 1
            for xID, pID in enumerate(links_pID_to_jID["Home"])
        },
        "Away": {
            int(links_pID_to_jID["Away"][pID]): xID + 1
            for xID, pID in enumerate(links_pID_to_jID["Away"])
        },
    }

    return links_jID_to_xID, links_pID_to_jID


def read_soccerbot_position_xml(
    filepath_positions: Union[str, Path],
    filepath_mat_info: Union[str, Path],
    links_jID_to_xID: Dict[str, Dict[int, int]] = None,
    links_pID_to_jID: Dict[str, Dict[str, int]] = None,
) -> Dict[str, Tuple[XY, XY, XY, Code, Code, Pitch]]:
    """Parse SoccerBot XML files and extract position data, possession and ballstatus
    codes as well as pitch information.

     Data in the SoccerBot format is given as two separate files, a position data XML
     file containing the actual data as well as a Match Information XML containing
     information about pitch size, players, teams, and start- and endframes of match
     periods. Since no information about framerate is delivered, it is estimated from
     time difference between individual frames. This function provides high-level access
     to SoccerBot data by parsing "the full match" contained in the position file.

    Parameters
    ----------
    filepath_positions: str or pathlib.Path
        Full path to XML File where the Position data in SoccerBot format is saved.
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in SoccerBot format is
        saved.
    links_jID_to_xID: Dict, optional
        A link dictionary of the form `links[team][jID] = xID`. Player's are identified
        in the XML files via jID, and this dictionary is used to map them to a specific
        xID in the respective XY objects. Should be supplied if that order matters. If
        one of links or id_to_jrsy is given as None (default), they are automatically
        extracted from the Match Information XML file.
    links_pID_to_jID: Dict, optional
        A link dictionary of the form `links[team][pID] = jID` where pID is the id
        specified in the SoccerBot Match Information file.

    Returns
    -------
    data_objects: Dict[str, Tuple[XY, XY, XY, Code, Code, Pitch]]
        XY-, Code-, and Pitch-objects for both teams for every segment included in
        the data (for a regular match "HT1" and "HT2). The order is
        (home_xy, away_xy, ball_xy, possession, ballstatus, pitch).
    """
    # read metadata
    pitch, kickoffs = _read_mat_info_xml(filepath_mat_info)

    # create or check links
    if links_jID_to_xID is None:
        links_jID_to_xID, links_pID_to_jID = create_links_from_mat_info_xml(
            filepath_mat_info
        )
    else:
        pass
        # potential check

    # create periods
    periods, timedelta = _read_periods_and_timedelta_from_positions_xml(
        filepath_positions, kickoffs
    )
    segments = list(periods.keys())

    # infer data array shapes
    number_of_home_players = max(links_jID_to_xID["Home"].values())
    number_of_away_players = max(links_jID_to_xID["Away"].values())
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

    # loop over frames containing player & ball positions for all segments
    tree = etree.parse(str(filepath_positions))
    root = tree.getroot()
    for positions in root.iterfind("Positions"):
        for frame in positions.iterfind("Timestamp"):

            # skip frames that do not contain position data
            if len(frame.find("HomeTeam")) == 0 and len(frame.find("AwayTeam")) == 0:
                break

            # read time and frame number and retrieve segment of current position
            frame_time = int(frame.get("t"))
            segment = None
            for seg in segments:
                if periods[seg][0] <= frame_time <= periods[seg][1]:
                    segment = seg
            frame_num = int((frame_time - periods[segment][0]) / timedelta)

            # insert ball position to bin
            position = frame.find("Ball").find("Pos")
            xydata["Ball"][segment][frame_num, 0] = float(position.get("x"))
            xydata["Ball"][segment][frame_num, 1] = float(position.get("y"))
            codes["possession"][segment].append(position.get("ballPossession"))
            codes["ballstatus"][segment].append(position.get("gameState"))

            # insert positions of the home team to correct place in bin
            for position in frame.find("HomeTeam").iterfind("Pos"):
                pID = position.get("id")
                jID = links_pID_to_jID["Home"][pID]
                xID = links_jID_to_xID["Home"][jID]
                x_col = (xID - 1) * 2
                y_col = (xID - 1) * 2 + 1
                xydata["Home"][segment][frame_num, x_col] = float(position.get("x"))
                xydata["Home"][segment][frame_num, y_col] = float(position.get("y"))

            # insert positions of the away team to correct place in bin
            for position in frame.find("AwayTeam").iterfind("Pos"):
                pID = position.get("id")
                jID = links_pID_to_jID["Away"][pID]
                xID = links_jID_to_xID["Away"][jID]
                x_col = (xID - 1) * 2
                y_col = (xID - 1) * 2 + 1
                xydata["Away"][segment][frame_num, x_col] = float(position.get("x"))
                xydata["Away"][segment][frame_num, y_col] = float(position.get("y"))

    # estimate framerate from the timedelta between frames and convert from f/ms to fps
    framerate_est = int(1000 / timedelta)

    # create and assemble XY-, Code-, and Pitch-objects for every segment
    data_objects = {}
    links_soccerbot_to_global_segments = {"firstHalf": "HT1", "secondHalf": "HT2"}
    for segment in segments:
        segment_data_objects = (
            XY(xy=xydata["Home"][segment], framerate=framerate_est),
            XY(xy=xydata["Away"][segment], framerate=framerate_est),
            XY(xy=xydata["Ball"][segment], framerate=framerate_est),
            Code(
                code=np.array(codes["possession"][segment]),
                name="possession",
                definitions={"Home": "Home", "Away": "Away", "": None},
                framerate=framerate_est,
            ),
            Code(
                code=np.array(codes["ballstatus"][segment]),
                name="ballstatus",
                definitions={"Home": "Home", "Away": "Away", "": None},
                framerate=framerate_est,
            ),
            pitch,
        )
        global_segment_name = links_soccerbot_to_global_segments[segment]
        data_objects[global_segment_name] = segment_data_objects

    return data_objects

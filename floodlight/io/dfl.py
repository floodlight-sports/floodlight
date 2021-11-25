from pathlib import Path
from typing import Dict, Tuple, Union
import warnings

from lxml import etree
import iso8601
import numpy as np

from floodlight.core.code import Code
from floodlight.core.pitch import Pitch
from floodlight.core.xy import XY


def _read_pitch_from_mat_info(filepath_mat_info: Union[str, Path]) -> Pitch:
    """Reads match_information XML file and returns the playing Pitch.

    Parameters
    ----------
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in DFL format is saved.

    Returns
    -------
    pitch: Pitch
        Pitch object with actual pitch length and width.
    """
    # set up XML tree
    tree = etree.parse(str(filepath_mat_info))
    root = tree.getroot()

    # parse pitch
    length = root.find("MatchInformation").find("Environment").get("PitchX")
    length = float(length) if length else None
    width = root.find("MatchInformation").find("Environment").get("PitchY")
    width = float(width) if width else None
    pitch = Pitch.from_template(
        "dfl",
        length=length,
        width=width,
        sport="football",
    )

    return pitch


def _create_periods_from_dat(
    filepath_dat: Union[str, Path]
) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """Parses over position file and returns dictionary with periods as well as an
    estimate of the framerate based on the timedelta between multiple frames.

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        Path to XML File where the Position data in DFL format is saved.

    Returns
    -------
    periods: Dict[str, Tuple[int, int]
        Dictionary with times for the segment:
        `periods[segment] = (starttime, endtime)`
    est_framerate: int
        Estimated temporal resolution of data in frames per second/Hertz.
    """
    periods = {}
    est_framerate = None

    # retrieve information from ball frame sets
    for _, frame_set in etree.iterparse(filepath_dat, tag="FrameSet"):
        if frame_set.get("TeamId") == "Ball":
            frames = [frame for frame in frame_set.iterfind("Frame")]
            periods[frame_set.get("GameSection")] = (
                int(frames[0].get("N")),
                int(frames[-1].get("N")),
            )
            delta = iso8601.parse_date(frames[1].get("T")) - iso8601.parse_date(
                frames[0].get("T")
            )
            if est_framerate is None:
                est_framerate = int(round(1 / delta.total_seconds()))
            elif est_framerate != int(round(1 / delta.total_seconds())):
                warnings.warn(
                    "Framerate estimation yielded diverging results."
                    "The originally estimated framerate of %d Hz did not "
                    "match the current estimation of %d Hz. This might be "
                    "caused by missing frame(s) in the position data."
                    "Continuing by choosing the latest estimation of %d Hz"
                    % (
                        est_framerate,
                        int(round(1 / delta.total_seconds())),
                        int(round(1 / delta.total_seconds())),
                    )
                )
                est_framerate = int(round(1 / delta.total_seconds()))

    return periods, est_framerate


def create_links_from_mat_info(
    filepath_mat_info: Union[str, Path]
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[str, int]]]:
    """Parses the DFL Match Information XML file for unique jIDs (jerseynumbers) and
    creates two dictionaries, one linking pIDs to jIDs and one linking jIDs to xIDs in
    ascending order.

    Parameters
    ----------
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in DFL format is saved

    Returns
    -------
    links_jID_to_xID: Dict[str, Dict[int, int]]
        A link dictionary of the form `links[team][jID] = xID`.
    links_pID_to_jID: Dict[str, Dict[str, int]]
        A link dictionary of the form `links[team][pID] = jID`.
    """
    # set up XML tree
    tree = etree.parse(str(filepath_mat_info))
    root = tree.getroot()

    # parse XML file and extract links from pID to xID for both teams
    links_pID_to_jID = {}
    teams = root.find("MatchInformation").find("Teams")
    home = root.find("MatchInformation").find("General").get("HomeTeamId")
    away = root.find("MatchInformation").find("General").get("AwayTeamId")

    for team in teams:
        if team.get("TeamId") == home:
            links_pID_to_jID["Home"] = {
                player.get("PersonId"): int(player.get("ShirtNumber"))
                for player in team.find("Players")
            }
        elif team.get("TeamId") == away:
            links_pID_to_jID["Away"] = {
                player.get("PersonId"): int(player.get("ShirtNumber"))
                for player in team.find("Players")
            }
        else:
            continue

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


def read_dfl_files(
    filepath_dat: Union[str, Path],
    filepath_mat_info: Union[str, Path],
    links_jID_to_xID: Dict[str, Dict[int, int]] = None,
    links_pID_to_jID: Dict[str, Dict[int, int]] = None,
) -> Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Code, Code, Pitch]:
    """Parse DFL files and extract position data, possession and ballstatus codes as
    well as pitch information.

    The official tracking system of the DFL (German Football League) delivers two
    separate XML files, one containing the actual data as well as a metadata file
    containing information about pitch size and start- and endframes of match periods.
    Since no information about framerate is delivered, it is estimated from time
    difference between individual frames. This function provides a high-level access to
    DFL data by parsing "the full match" given both files.

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        Full path to XML File where the Position data in DFL format is saved.
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in DFL format is saved.
    links_jID_to_xID: Dict, optional
        A link dictionary of the form `links[team][jID] = xID`. Player's are identified
        in the XML files via jID, and this dictionary is used to map them to a specific
        xID in the respective XY objects. Should be supplied if that order matters. If
        one of links or id_to_jrsy is given as None (default), they are automatically
        extracted from the Match Information XML file.
    links_pID_to_jID: Dict, optional
        A link dictionary of the form `links[team][pID] = jID` where pID is the PersonId
        specified in the DFL Match Information file.

    Returns
    -------
    data_objects: Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Code, Code, Pitch]
        XY-, Code-, and Pitch-objects for both teams and both halves. The order is
        (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2,
        possession_ht1, possession_ht2, ballstatus_ht1, ballstatus_ht2, pitch)

    """
    # read metadata
    pitch = _read_pitch_from_mat_info(filepath_mat_info)

    # create or check links
    if links_jID_to_xID is None or links_pID_to_jID is None:
        links_jID_to_xID, links_pID_to_jID = create_links_from_mat_info(
            filepath_mat_info
        )
    else:
        pass
        # potential check

    # create periods
    periods, framerate = _create_periods_from_dat(filepath_dat)
    segments = list(periods.keys())

    # infer data array shapes
    number_of_home_players = max(links_jID_to_xID["Home"].values())
    number_of_away_players = max(links_jID_to_xID["Away"].values())
    number_of_frames = {}
    for segment in segments:
        start = periods[segment][0]
        end = periods[segment][1]
        number_of_frames[segment] = end - start + 1

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

    # loop over frame sets containing player & ball positions for all segments
    for _, frame_set in etree.iterparse(filepath_dat, tag="FrameSet"):

        # ball
        if frame_set.get("TeamId") == "Ball":
            # (x, y) position
            segment = frame_set.get("GameSection")
            xydata["Ball"][segment][:, 0] = np.array(
                [float(frame.get("X")) for frame in frame_set.iterfind("Frame")]
            )
            xydata["Ball"][segment][:, 1] = np.array(
                [float(frame.get("Y")) for frame in frame_set.iterfind("Frame")]
            )
            # codes
            codes["ballstatus"][segment] = [
                float(frame.get("BallStatus")) for frame in frame_set.iterfind("Frame")
            ]
            codes["possession"][segment] = [
                float(frame.get("BallPossession"))
                for frame in frame_set.iterfind("Frame")
            ]

        # teams
        else:
            # find identity of frame set
            frames = [frame for frame in frame_set.iterfind("Frame")]
            segment = frame_set.get("GameSection")
            if frame_set.get("PersonId") in links_pID_to_jID["Home"]:
                team = "Home"
                jrsy = links_pID_to_jID[team][frame_set.get("PersonId")]
            elif frame_set.get("PersonId") in links_pID_to_jID["Away"]:
                team = "Away"
                jrsy = links_pID_to_jID[team][frame_set.get("PersonId")]
            else:
                pass
                # possible error or warning

            # insert (x,y) data to correct place in bin
            start = int(frames[0].get("N")) - periods[segment][0]
            end = int(frames[-1].get("N")) - periods[segment][0] + 1
            x_col = (links_jID_to_xID[team][jrsy] - 1) * 2
            y_col = (links_jID_to_xID[team][jrsy] - 1) * 2 + 1
            xydata[team][segment][start:end, x_col] = np.array(
                [float(frame.get("X")) for frame in frames]
            )
            xydata[team][segment][start:end, y_col] = np.array(
                [float(frame.get("Y")) for frame in frames]
            )

    # create XY objects
    home_ht1 = XY(xy=xydata["Home"]["firstHalf"], framerate=framerate)
    home_ht2 = XY(xy=xydata["Home"]["secondHalf"], framerate=framerate)
    away_ht1 = XY(xy=xydata["Away"]["firstHalf"], framerate=framerate)
    away_ht2 = XY(xy=xydata["Away"]["secondHalf"], framerate=framerate)
    ball_ht1 = XY(xy=xydata["Ball"]["firstHalf"], framerate=framerate)
    ball_ht2 = XY(xy=xydata["Ball"]["secondHalf"], framerate=framerate)

    # create Code objects
    possession_ht1 = Code(
        code=codes["possession"]["firstHalf"],
        name="possession",
        definitions={1: "Home", 2: "Away"},
        framerate=framerate,
    )
    possession_ht2 = Code(
        code=codes["possession"]["secondHalf"],
        name="possession",
        definitions={1: "Home", 2: "Away"},
        framerate=framerate,
    )
    ballstatus_ht1 = Code(
        code=codes["ballstatus"]["firstHalf"],
        name="ballstatus",
        definitions={0: "Dead", 1: "Alive"},
        framerate=framerate,
    )
    ballstatus_ht2 = Code(
        code=codes["ballstatus"]["secondHalf"],
        name="ballstatus",
        definitions={0: "Dead", 1: "Alive"},
        framerate=framerate,
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

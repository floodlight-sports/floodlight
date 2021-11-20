from pathlib import Path
from typing import Dict, Tuple, Union

from lxml import etree
import numpy as np
import datetime as dt

from floodlight.core.xy import XY
from floodlight.core.pitch import Pitch


def _read_mat_info(filepath_mat_info: Union[str, Path]) -> Tuple[Pitch, Dict[str, int]]:
    """Reads match_information XML file and returns the playing Pitch.

    Parameters
    ----------
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in DFL format is saved.

    Returns
    -------
    pitch: Pitch
        Pitch object with actual pitch length and width.
    kickoffs: Dict[str, int]
        Dictionary with kickoff frames for the segment:
        `kickoffs[segment] = kickoff_frame`

    """
    # set up XML tree
    tree = etree.parse(str(filepath_mat_info))
    root = tree.getroot()

    # parse pitch
    length = root.find("MatchInfoForSoccerBot").find("Field").get("sx")
    length = float(length) if length else None
    width = root.find("MatchInfoForSoccerBot").find("Field").get("sy")
    width = float(width) if width else None

    pitch = Pitch(
        length=length,
        width=width,
        xlim=(-length / 2, length / 2),
        ylim=(-width / 2, width / 2),
        unit="m",
        boundaries="flexible",
        sport="football",
    )

    # parse kick offs for all halves
    kickoffs = {}
    for segment in root.find("MatchInfoForSoccerBot").find("KickOff"):
        kickoffs[segment] = (
            root.find("MatchInfoForSoccerBot").find("KickOff").get(segment)
        )

    return pitch, kickoffs


def _read_single_frame(frame: etree.Element):
    """Extracts all relevant information from a single frame of SoccerBot Position data
    XML file

    Parameters
    ----------
    frame: etree.Element
        Element Tree XML element.

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


def create_links_from_mat_info(
    filepath_mat_info: Union[str, Path]
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[str, int]]]:
    """Parses the SoccerBot Match Information XML file for unique jIDs (jerseynumbers)
    and creates two dictionaries, one linking pIDs to jIDs and one linking jIDs to xIDs
    in ascending order.

    Parameters
    ----------
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in DFL format is saved

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
    for player in root.find("MatchInfoForSoccerBot").iterfind("Player"):
        id_to_jrsy[player.get("team")][player.get("id")] = {player.get("shirt")}

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
) -> Tuple[XY, XY, XY, XY, XY, XY, Pitch]:
    """Read a DFL format position data and match information XML

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        XML File where the Position data in DFL format is saved
    filepath_mat_info: str or pathlib.Path
        XML File where the Match Information data in DFL format is saved
    links: Dict
        Dictionary from DFL PersonId to position in the XY Object

    Returns
    -------
    List of floodlight.core.xy.XY Objects
    """
    # checks?

    # read metadata
    pitch, kickoffs = _read_mat_info(filepath_mat_info)
    segments = list(kickoffs.keys())
    periods = []
    framerate = 10

    # create or check links
    if links is None:
        links = create_links_from_mat_info(filepath_mat_info)
    else:
        pass
        # potential check

    # infer data array shapes
    number_of_home_players = max(links["Home"].values())
    number_of_away_players = max(links["Away"].values())
    number_of_frames = {}
    for segment in segments:
        number_of_frames[segment] = len(periods[segment])

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
    # codes = {
    #     code: {segment: [] for segment in segments}
    #     for code in ["possession", "ballstatus"]
    # }

    # loop over data filepath containing the positions
    for _, frame_set in etree.iterparse(filepath_dat, tag="FrameSet"):

        # assign team and columns
        if frame_set.get("TeamId") == "Ball":
            team = "Ball"
            x_col = 0
            y_col = 1
        else:
            pID = frame_set.get("PersonId")
            if pID in links["Home"]:
                team = "Home"
            elif pID in links["Away"]:
                team = "Away"
            else:
                team = None
                pass
                # possible error or warning
            x_col = (links[team][pID] - 1) * 2
            y_col = (links[team][pID] - 1) * 2 + 1

        # fill rows of xy data according to playing times of the player
        segment = frame_set.get("GameSection")
        times = [
            dt.datetime.fromisoformat(frame.get("T"))
            for frame in frame_set.iterfind("Frame")
        ]
        if len(times) == number_of_frames[segment]:  # player did play through
            xydata[team][segment][:, x_col] = np.array(
                [float(frame.get("X")) for frame in frame_set.iterfind("Frame")]
            )
            xydata[team][segment][:, y_col] = np.array(
                [float(frame.get("Y")) for frame in frame_set.iterfind("Frame")]
            )
        else:  # player did not play through
            if times[0] < periods[segment][-1] and times[-1] > periods[segment][0]:
                start = np.argmin(
                    np.abs([(t - times[0]).total_seconds() for t in periods[segment]])
                )
                end = np.argmin(
                    np.abs([(t - times[-1]).total_seconds() for t in periods[segment]])
                )
                xydata[team][segment][start : end + 1, x_col] = np.array(
                    [float(frame.get("X")) for frame in frame_set.iterfind("Frame")]
                )
                xydata[team][segment][start : end + 1, y_col] = np.array(
                    [float(frame.get("Y")) for frame in frame_set.iterfind("Frame")]
                )

    # Create XY Objects
    home_ht1 = XY(xy=xydata["Home"]["firstHalf"], framerate=framerate)
    home_ht2 = XY(xy=xydata["Home"]["secondHalf"], framerate=framerate)
    away_ht1 = XY(xy=xydata["Away"]["firstHalf"], framerate=framerate)
    away_ht2 = XY(xy=xydata["Away"]["secondHalf"], framerate=framerate)
    ball_ht1 = XY(xy=xydata["Ball"]["firstHalf"], framerate=framerate)
    ball_ht2 = XY(xy=xydata["Ball"]["secondHalf"], framerate=framerate)

    data_objects = (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2, pitch)
    return data_objects

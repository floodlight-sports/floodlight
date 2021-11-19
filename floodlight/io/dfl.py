from pathlib import Path
from typing import Dict, Tuple, Union

from lxml import etree
import datetime as dt
import numpy as np

from floodlight.core.code import Code
from floodlight.core.pitch import Pitch
from floodlight.core.xy import XY


def _read_pitch_from_mat_info(filepath_mat_info: Union[str, Path]) -> Pitch:
    """Reads match_information XML file and returns the playing Pitch.

    Parameters
    ----------
    filepath_mat_info: str or Path
        Path to XML File where the Match Information data in DFL format is saved

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
    )

    return pitch


def create_links_from_mat_info(
    filepath_mat_info: Union[str, Path]
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[str, int]]]:
    """Creates links between player_id and column in the array from matchinfo XML file

    Parameters
    ----------
    filepath_mat_info: str or Path
        Path to XML File where the Match Information data in DFL format is saved

    Returns
    -------
    links: Dict[str, Dict[int, int]]
        Dictionary mapping the jersey number to position in the team data array
    id_to_jrsy: Dict[str, Dict[str, int]]
        Dictionary mapping the DFL player id to jersey number
    """
    # set up XML tree
    tree = etree.parse(str(filepath_mat_info))
    root = tree.getroot()

    # parse XML file and extract links from pID to xID for both teams
    id_to_jrsy = {}
    teams = root.find("MatchInformation").find("Teams")
    home = root.find("MatchInformation").find("General").get("HomeTeamId")
    away = root.find("MatchInformation").find("General").get("AwayTeamId")

    for team in teams:
        if team.get("TeamId") == home:
            id_to_jrsy["Home"] = {
                player.get("PersonId"): int(player.get("ShirtNumber"))
                for player in team.find("Players")
            }
        elif team.get("TeamId") == away:
            id_to_jrsy["Away"] = {
                player.get("PersonId"): int(player.get("ShirtNumber"))
                for player in team.find("Players")
            }
        else:
            continue

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


def _create_periods_from_dat(
    filepath_dat: Union[str, Path]
) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """Parses over position file and returns dictionary with periods

    Parameters
    ----------
    filepath_dat: str or Path
        Path to XML File where the Position data in DFL format is saved

    Returns
    -------
    periods: Dict[str, Tuple[int, int]
        Dictionary with times for the segment:
        `periods[segment] = (starttime, endtime)`
    framerate: int
        Temporal resolution of data in frames per second/Hertz.
    """
    periods = {}
    framerate = None

    # retrieve information from ball frame sets
    for _, frame_set in etree.iterparse(filepath_dat, tag="FrameSet"):
        if frame_set.get("TeamId") == "Ball":
            frames = [frame for frame in frame_set.iterfind("Frame")]
            periods[frame_set.get("GameSection")] = (
                int(frames[0].get("N")),
                int(frames[-1].get("N")),
            )
            if framerate is None:
                delta = dt.datetime.fromisoformat(
                    frames[1].get("T")
                ) - dt.datetime.fromisoformat(frames[0].get("T"))
                framerate = int(round(1 / delta.total_seconds()))

    return periods, framerate


def read_dfl_files(
    filepath_dat: Union[str, Path],
    filepath_mat_info: Union[str, Path],
    links: Dict[str, Dict[int, int]] = None,
    id_to_jrsy: Dict[str, Dict[int, int]] = None,
) -> Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Code, Code, Pitch]:
    """Read a DFL format position data and match information XML.

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        XML File where the Position data in DFL format is saved.
    filepath_mat_info: str or pathlib.Path
        XML File where the Match Information data in DFL format is saved.
    links: Dict
        Dictionary mapping jersey number to position in the XY Object.
    id_to_jrsy: Dict
        Dictionary mapping DFL PersonId to jersey number.

    Returns
    -------
    List of XY Objects
    """
    # checks?

    # read metadata
    pitch = _read_pitch_from_mat_info(filepath_mat_info)

    # create or check links
    if links is None or id_to_jrsy is None:
        links, id_to_jrsy = create_links_from_mat_info(filepath_mat_info)
    else:
        pass
        # potential check

    # create periods
    periods, framerate = _create_periods_from_dat(filepath_dat)
    segments = list(periods.keys())

    # infer data array shapes
    number_of_home_players = max(links["Home"].values())
    number_of_away_players = max(links["Away"].values())
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
            if frame_set.get("PersonId") in id_to_jrsy["Home"]:
                team = "Home"
                jrsy = id_to_jrsy[team][frame_set.get("PersonId")]
            elif frame_set.get("PersonId") in id_to_jrsy["Away"]:
                team = "Away"
                jrsy = id_to_jrsy[team][frame_set.get("PersonId")]
            else:
                team = None
                jrsy = None
                pass
                # possible error or warning

            # insert (x,y) data to correct place in bin
            start = int(frames[0].get("N")) - periods[segment][0]
            end = int(frames[-1].get("N")) - periods[segment][0] + 1
            x_col = (links[team][jrsy] - 1) * 2
            y_col = (links[team][jrsy] - 1) * 2 + 1
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
    stat_ht1 = Code(
        name="ballstatus",
        code=codes["ballstatus"]["firstHalf"],
        definitions={0: "Dead", 1: "Alive"},
        framerate=framerate,
    )
    stat_ht2 = Code(
        name="ballstatus",
        code=codes["ballstatus"]["secondHalf"],
        definitions={0: "Dead", 1: "Alive"},
        framerate=framerate,
    )
    poss_ht1 = Code(
        name="possession",
        code=codes["possession"]["firstHalf"],
        definitions={1: "Home", 2: "Away"},
        framerate=framerate,
    )
    poss_ht2 = Code(
        name="ballstatus",
        code=codes["possession"]["secondHalf"],
        definitions={1: "Home", 2: "Away"},
        framerate=framerate,
    )

    data_objects = (
        home_ht1,
        home_ht2,
        away_ht1,
        away_ht2,
        ball_ht1,
        ball_ht2,
        stat_ht1,
        stat_ht2,
        poss_ht1,
        poss_ht2,
        pitch,
    )
    return data_objects

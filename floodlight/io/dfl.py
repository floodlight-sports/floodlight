from pathlib import Path
from typing import Dict, Tuple, Union

from lxml import etree
import numpy as np
import datetime as dt

from floodlight.core.xy import XY
from floodlight.core.pitch import Pitch


def _read_matchinfo(
    filepath_matchinfo: Union[str, Path]
) -> Tuple[Dict[str, float], Pitch]:
    """Reads match_information XML file and returns dictionary with meta information and
    the playing Pitch.

    Parameters
    ----------
    filepath_matchinfo: str or Path
        Full path to metadata.xml file.

    Returns
    -------
    metainfo: Dict
        Dictionary with metainformation such as framerate.
    pitch: floodlight.core.pitch.Pitch
        Pitch object with actual pitch length and width.
    """
    # set up XML tree
    tree = etree.parse(str(filepath_matchinfo))
    root = tree.getroot()

    # parse XML file and extract matchinfo
    metadata = {}

    length = root.find("MatchInformation").find("Environment").get("PitchX")
    metadata["length"] = float(length) if length else None

    width = root.find("MatchInformation").find("Environment").get("PitchY")
    metadata["width"] = float(width) if width else None

    pitch = Pitch(
        length=metadata["length"],
        width=metadata["width"],
        xlim=(-metadata["length"] / 2, metadata["length"] / 2),
        ylim=(-metadata["width"] / 2, metadata["width"] / 2),
        unit="m",
        boundaries="flexible",
        sport="football",
    )

    return metadata, pitch


def create_links_from_matchinfo(
    filepath_matchinfo: Union[str, Path]
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[str, int]]]:
    """Creates links between player_id and column in the array from matchinfo XML file

    Parameters
    ----------
    filepath_matchinfo: str or pathlib.Path
        XML File where the Match Information data in DFL format is saved

    Returns
    -------

    """
    # set up XML tree
    tree = etree.parse(str(filepath_matchinfo))
    root = tree.getroot()

    # parse XML file and extract links from pID to xID for both teams
    id_to_jrsy = {}
    teams = root.find("MatchInformation").find("Teams")
    home = root.find("MatchInformation").find("General").get("HomeTeamId")
    away = root.find("MatchInformation").find("General").get("AwayTeamId")

    for team in teams:
        if team.get("TeamId") == home:
            id_to_jrsy["Home"] = {
                player.get("PersonId"): player.get("ShirtNumber")
                for player in team.find("Players")
            }
        elif team.get("TeamId") == away:
            id_to_jrsy["Away"] = {
                player.get("PersonId"): player.get("ShirtNumber")
                for player in team.find("Players")
            }
        else:
            continue

    links = {
        "Home": {int(jID): xID + 1 for xID, jID in enumerate(id_to_jrsy["Home"])},
        "Away": {int(jID): xID + 1 for xID, jID in enumerate(id_to_jrsy["Away"])},
    }

    return links, id_to_jrsy


def _create_periods_from_dat(filepath_dat: Union[str, Path]) -> Tuple[Dict, int]:
    """Parses over position file and returns dictionary with segment timeaxis

    Parameters
    ----------
    filepath_dat: str or Path
        Full path to position.xml file.


    Returns
    -------
    periods: Dict
        Dictionary with timeaxis for the segment:
        `periods[segment] = List of dt.datetime`.
    """
    periods = {}
    framerate = None

    # parse the framesets for the ball to find the segment timeaxis
    for _, frame_set in etree.iterparse(filepath_dat, tag="FrameSet"):
        if frame_set.get("TeamId") == "Ball":
            periods[frame_set.get("GameSection")] = [
                dt.datetime.fromisoformat(frame.get("T"))
                for frame in frame_set.iterfind("Frame")
            ]
            if framerate is None:
                framerate = int(
                    1
                    / (
                        periods[frame_set.get("GameSection")][1]
                        - periods[frame_set.get("GameSection")][0]
                    ).total_seconds()
                )

    return periods, framerate


def read_dfl_files(
    filepath_dat: Union[str, Path],
    filepath_matchinfo: Union[str, Path],
    links: Dict[str, Dict[int, int]] = None,
    id_to_jrsy: Dict[str, Dict[int, int]] = None,
) -> Tuple[XY, XY, XY, XY, XY, XY, Pitch]:
    """Read a DFL format position data and match information XML.

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        XML File where the Position data in DFL format is saved.
    filepath_matchinfo: str or pathlib.Path
        XML File where the Match Information data in DFL format is saved.
    links: Dict
        Dictionary mapping jersey number to position in the XY Object.
    id_to_jrsy: Dict
        Dictionary mapping DFL PersonId to jersey number.

    Returns
    -------
    List of floodlight.core.xy.XY Objects
    """
    # checks?

    # read metadata
    metadata, pitch = _read_matchinfo(filepath_matchinfo)

    # create or check links
    if links is None or id_to_jrsy is None:
        links, id_to_jrsy = create_links_from_matchinfo(filepath_matchinfo)
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
    codes = {
        code: {segment: [] for segment in segments}
        for code in ["possession", "ballstatus"]
    }

    # loop over framesets containing player & ball positions for all segments
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

            # retrieve correct place to insert (x,y) data
            segment = frame_set.get("GameSection")
            times = [
                dt.datetime.fromisoformat(frame.get("T"))
                for frame in frame_set.iterfind("Frame")
            ]
            start = np.argmin(
                np.abs([(t - times[0]).total_seconds() for t in periods[segment]])
            )
            end = np.argmin(
                np.abs([(t - times[-1]).total_seconds() for t in periods[segment]])
            )
            x_col = (links[team][jrsy] - 1) * 2
            y_col = (links[team][jrsy] - 1) * 2 + 1

            # insert to bin
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

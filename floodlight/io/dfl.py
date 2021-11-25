from pathlib import Path
from typing import Dict, Tuple, Union
import warnings

from lxml import etree
import iso8601
import numpy as np
import pandas as pd

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
    framerate_est = None

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
            if framerate_est is None:
                framerate_est = int(round(1 / delta.total_seconds()))
            elif framerate_est != int(round(1 / delta.total_seconds())):
                warnings.warn(
                    "Framerate estimation yielded diverging results."
                    "The originally estimated framerate of %d Hz did not "
                    "match the current estimation of %d Hz. This might be "
                    "caused by missing frame(s) in the position data."
                    "Continuing by choosing the latest estimation of %d Hz"
                    % (
                        framerate_est,
                        int(round(1 / delta.total_seconds())),
                        int(round(1 / delta.total_seconds())),
                    )
                )
                framerate_est = int(round(1 / delta.total_seconds()))

    return periods, framerate_est


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


def _get_event_description(elem: etree.Element) -> Tuple[str, dict]:
    """Returns the full description of a single XML Event in the DFL format.

    Parameters
    ----------
    elem: lxml.etree.Element
        lxml.etree.Element with the Event information.

    Returns
    -------
    eID: str
        High-level description for the current event.
    attrib: dict
        Additional attributes for the current event in the form
        ´attrib[category] = label´.
    """
    # read description
    eID = elem.tag

    # read additional attributes
    attrib = {}
    for category in elem.attrib:
        attrib[category] = elem.attrib[category]

    # find nested elements
    nested_eID = None
    nested_attrib = None
    if elem.find("Play") is not None:
        nested_eID, nested_attrib = _get_event_description(elem.find("Play"))
    elif elem.find("Pass") is not None:
        nested_eID, nested_attrib = _get_event_description(elem.find("Pass"))
    elif elem.find("Cross") is not None:
        nested_eID, nested_attrib = _get_event_description(elem.find("Cross"))
    elif elem.find("ShotAtGoal") is not None:
        nested_eID, nested_attrib = _get_event_description(elem.find("ShotAtGoal"))
        nested_attrib.update(attrib)
    elif elem.find("SuccessfulShot") is not None:
        nested_eID, nested_attrib = _get_event_description(elem.find("SuccessfulShot"))
        nested_attrib.update(attrib)
    elif elem.find("SavedShot") is not None:
        nested_eID, nested_attrib = _get_event_description(elem.find("SavedShot"))
        nested_attrib.update(attrib)
    elif elem.find("BlockedShot") is not None:
        nested_eID, nested_attrib = _get_event_description(elem.find("BlockedShot"))
        nested_attrib.update(attrib)
    elif elem.find("ShotWide") is not None:
        nested_eID, nested_attrib = _get_event_description(elem.find("ShotWide"))
        nested_attrib.update(attrib)
    elif elem.find("ShotWoodWork") is not None:
        nested_eID, nested_attrib = _get_event_description(elem.find("ShotWoodWork"))
        nested_attrib.update(attrib)
    elif elem.find("OtherShot") is not None:
        nested_eID, nested_attrib = _get_event_description(elem.find("OtherShot"))

    # update
    if nested_eID is not None and nested_attrib is not None:
        eID += "_" + nested_eID
        attrib.update(nested_attrib)

    return eID, attrib


def _get_event_outcome(eID, attrib):
    """Returns the outcome of a single Event in the DFL format.

    Parameters
    ----------
    eID: str
        High-level description for the current event.
    attrib: dict
        Additional attributes for the current event in the form
        ´attrib[category] = label´.

    Returns
    -------
    outcome: int
        Outcome coded as 1 (success) or 0 (failure) of the current event.
    """
    outcome = None

    if eID == "TacklingGame":
        if attrib["WinnerRole"] == "withoutBallControl":
            outcome = 1
        elif attrib["WinnerRole"] == "withBallControl":
            outcome = 0
        else:
            pass
    elif eID == "BallClaiming":
        if "Type" in attrib:
            if attrib["Type"] in ["BallClaimed"]:
                outcome = 1
            elif attrib["Type"] in ["BallHeld"]:
                outcome = 0
            else:
                pass
    elif eID == "Play":
        if attrib["Successful"] == "true":
            outcome = 1
        elif attrib["Successful"] == "false":
            outcome = 0
        else:
            pass
    # per definition no outcome
    elif eID in [
        "OwnGoal",
        "DefensiveClearance",
        "Foul",
        "Offside",
        "Caution",
        "SendingOff",
        "Substitution",
        "KickoffWhistle",
        "FinalWhistle",
        "FairPlay",
        "RefereeBall",
        "OtherBallAction",
        "OtherPlayerAction",
    ]:
        pass
    # only defined as positive outcome -> necessary?
    elif eID in ["PreventedOwnGoal"]:
        outcome = 1
    # nested structure with outcome in sub child
    elif eID in [
        "FreeKick",
        "ThrowIn",
        "CornerKick",
        "Penalty",
        "GoalKick",
        "Kickoff",
        "ShotAtGoal",
    ]:
        pass
    else:
        warnings.warn("Unknown Event Type %s" % eID)
    return outcome


def read_events(filepath_events: Union[str, Path], framerate: int):
    """Parses the DFL Match Information XML file for unique jIDs (jerseynumbers) and
    creates two dictionaries, one linking pIDs to jIDs and one linking jIDs to xIDs in
    ascending order.

    Parameters
    ----------
    filepath_events: str or pathlib.Path
        Full path to XML File where the Event data in DFL format is saved
    framerate: int
        (Estimated) temporal resolution of data in frames per second/Hertz.
    Returns
    -------
    events: List of Event
        List of events for all segments.
    """
    # set up XML tree
    tree = etree.parse(str(filepath_events))
    root = tree.getroot()

    # find start and end of half
    kickoff_whistles = {}
    final_whistles = {}
    for whistle in root.findall("Event/KickoffWhistle"):
        kickoff_whistles[whistle.get("GameSection")] = iso8601.parse_date(
            whistle.getparent().get("EventTime")
        )
    for whistle in root.findall("Event/FinalWhistle"):
        final_whistles[whistle.get("GameSection")] = iso8601.parse_date(
            whistle.getparent().get("EventTime")
        )

    # define periods
    segments = list(kickoff_whistles.keys())
    periods = {}
    for segment in segments:
        periods[segment] = (kickoff_whistles[segment], final_whistles[segment])

    # set up bins
    events = {segment: pd.DataFrame() for segment in segments}

    # loop over events
    for elem in root.findall("Event"):
        # initialize
        event = {}

        # check for structure that is an element Event with a single child
        if len(elem) != 1:
            warnings.warn(
                "Access of an Element with multiple children! This can cause"
                "unprecise Event descriptions and outcomes."
            )

        # time information: timestamp (absolute) and gameclock (relative)
        event["timestamp"] = iso8601.parse_date(elem.get("EventTime"))
        event["gameclock"] = np.nan
        segment = None
        for seg in segments:
            if periods[seg][0] <= event["timestamp"] <= periods[seg][1]:
                event["gameclock"] = (
                    event["timestamp"] - periods[segment][0]
                ).total_seconds()
                segment = seg

        # description and outcome
        for child in elem:
            event["eID"], attrib = _get_event_description(child)
            event.update(attrib)
            event["outcome"] = _get_event_outcome(child, attrib)

        events[segment].append(event)


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
    periods, framerate_est = _create_periods_from_dat(filepath_dat)
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
    home_ht1 = XY(xy=xydata["Home"]["firstHalf"], framerate=framerate_est)
    home_ht2 = XY(xy=xydata["Home"]["secondHalf"], framerate=framerate_est)
    away_ht1 = XY(xy=xydata["Away"]["firstHalf"], framerate=framerate_est)
    away_ht2 = XY(xy=xydata["Away"]["secondHalf"], framerate=framerate_est)
    ball_ht1 = XY(xy=xydata["Ball"]["firstHalf"], framerate=framerate_est)
    ball_ht2 = XY(xy=xydata["Ball"]["secondHalf"], framerate=framerate_est)

    # create Code objects
    possession_ht1 = Code(
        code=codes["possession"]["firstHalf"],
        name="possession",
        definitions={1: "Home", 2: "Away"},
        framerate=framerate_est,
    )
    possession_ht2 = Code(
        code=codes["possession"]["secondHalf"],
        name="possession",
        definitions={1: "Home", 2: "Away"},
        framerate=framerate_est,
    )
    ballstatus_ht1 = Code(
        code=codes["ballstatus"]["firstHalf"],
        name="ballstatus",
        definitions={0: "Dead", 1: "Alive"},
        framerate=framerate_est,
    )
    ballstatus_ht2 = Code(
        code=codes["ballstatus"]["secondHalf"],
        name="ballstatus",
        definitions={0: "Dead", 1: "Alive"},
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

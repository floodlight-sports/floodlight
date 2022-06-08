from pathlib import Path
from typing import Dict, Tuple, Union
import warnings

from lxml import etree
import iso8601
import numpy as np
import pandas as pd

from floodlight.core.code import Code
from floodlight.core.events import Events
from floodlight.core.pitch import Pitch
from floodlight.core.xy import XY


def _create_periods_from_dat(
    filepath_positions: Union[str, Path]
) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """Parses over position file and returns dictionary with periods as well as an
    estimate of the framerate based on the timedelta between multiple frames.

    Parameters
    ----------
    filepath_positions: str or pathlib.Path
        Path to XML File where the Position data in DFL format is saved.

    Returns
    -------
    periods: Dict[str, Tuple[int, int]
        Dictionary with times for the segment:
        ``periods[segment] = (starttime, endtime)``
    est_framerate: int
        Estimated temporal resolution of data in frames per second/Hertz.
    """
    periods = {}
    framerate_est = None

    # retrieve information from ball frame sets
    for _, frame_set in etree.iterparse(filepath_positions, tag="FrameSet"):
        if frame_set.get("TeamId").lower() == "ball":
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
                    f"Framerate estimation yielded diverging results."
                    f"The originally estimated framerate of {framerate_est} Hz did not "
                    f"match the current estimation of "
                    f"{int(round(1 / delta.total_seconds()))} Hz. This might be "
                    f"caused by missing frame(s) in the position data."
                    f"Continuing by choosing the latest estimation of "
                    f"{int(round(1 / delta.total_seconds()))} Hz"
                )
                framerate_est = int(round(1 / delta.total_seconds()))

        frame_set.clear()

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
    if "AwayTeamId" in root.find("MatchInformation").find("General").attrib:
        away = root.find("MatchInformation").find("General").get("AwayTeamId")
    elif "GuestTeamId" in root.find("MatchInformation").find("General").attrib:
        away = root.find("MatchInformation").find("General").get("GuestTeamId")
    else:
        away = None

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
            int(links_pID_to_jID["Home"][pID]): xID
            for xID, pID in enumerate(links_pID_to_jID["Home"])
        },
        "Away": {
            int(links_pID_to_jID["Away"][pID]): xID
            for xID, pID in enumerate(links_pID_to_jID["Away"])
        },
    }

    return links_jID_to_xID, links_pID_to_jID


def _get_event_description(
    elem: etree.Element,
) -> Tuple[str, Dict[str, Union[str, int]]]:
    """Returns the full description of a single XML Event in the DFL format.

    Parameters
    ----------
    elem: lxml.etree.Element
        lxml.etree.Element with the Event information.

    Returns
    -------
    eID: str
        High-level description for the current event.
    attrib: Dict
        Additional attributes for the current event in the form
        ``attrib[category] = label``.
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

    # update nested elements
    if nested_eID is not None and nested_attrib is not None:
        eID += "_" + nested_eID
        attrib.update(nested_attrib)

    return eID, attrib


def _get_event_outcome(eID, attrib) -> int:
    """Returns the outcome of a single Event in the DFL format.

    Parameters
    ----------
    eID: str
        High-level description for the current event.
    attrib: Dict
        Additional attributes for the current event in the form
        ``attrib[category] = label``.

    Returns
    -------
    outcome: int
        Outcome coded as 1 (success) or 0 (failure) of the current event or np.nan in
        case no outcome is defined.
    """
    outcome = np.nan

    # well-defined outcome
    if "TacklingGame" in eID:
        if attrib["WinnerRole"] == "withoutBallControl":
            outcome = 1
        elif attrib["WinnerRole"] == "withBallControl":
            outcome = 0
    elif "BallClaiming" in eID:
        if "Type" in attrib:
            if attrib["Type"] in ["BallClaimed"]:
                outcome = 1
            elif attrib["Type"] in ["BallHeld"]:
                outcome = 0
    elif "Play" in eID:
        if "Successful" in attrib:
            if attrib["Successful"] == "true":
                outcome = 1
            elif attrib["Successful"] == "false":
                outcome = 0
    elif "ShotAtGoal" in eID:
        if "SuccessfulShot" in eID:
            outcome = 1
        else:
            outcome = 0
    # per definition no outcome
    elif eID in [
        "OwnGoal",
        "PreventedOwnGoal",
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
    # missing sub child in event
    elif eID in [
        "FreeKick",
        "ThrowIn",
        "CornerKick",
        "Penalty",
        "GoalKick",
        "Kickoff",
    ]:
        pass

    return outcome


def _get_event_team_and_player(eID, attrib) -> Tuple[str, str]:
    """Returns the player and team of a single Event in the DFL format.

    Parameters
    ----------
    eID: str
        High-level description for the current event.
    attrib: Dict
        Additional attributes for the current event in the form
        ``attrib[category] = label``.

    Returns
    -------
    team: str
        DFL Team ID of the Event
    player: str
        DFL Player ID of the Event
    """
    # team
    team = None
    if "Team" in attrib:
        team = attrib["Team"]
    elif eID == "TacklingGame" and "WinnerTeam" in attrib and "LoserTeam" in attrib:
        if attrib["WinnerRole"] == "withBallControl":
            team = attrib["WinnerTeam"]
        elif attrib["WinnerRole"] == "withoutBallControl":
            team = attrib["LoserTeam"]
    elif "TeamFouler" in attrib:
        team = attrib["TeamFouler"]

    # player
    player = None
    if "Player" in attrib:
        player = attrib["Player"]
    elif eID == "TacklingGame" and "Winner" in attrib and "Loser" in attrib:
        if attrib["WinnerRole"] == "withBallControl":
            player = attrib["Winner"]
        elif attrib["WinnerRole"] == "withoutBallControl":
            player = attrib["Loser"]
    elif "Fouler" in attrib:
        player = attrib["Fouler"]

    return team, player


def read_pitch_from_mat_info_xml(filepath_mat_info: Union[str, Path]) -> Pitch:
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


def read_event_data_xml(
    filepath_events: Union[str, Path]
) -> Tuple[Events, Events, Events, Events]:
    """Parses a DFL Match Event XML file and extracts the event data.

    This function provides a high-level access to the particular DFL Match Event feed
    and returns Event objects for both teams. The number of segments is inferred from
    the data, yet data for each segment is stored in a separate object.

    Parameters
    ----------
    filepath_events: str or pathlib.Path
        Full path to XML File where the Event data in DFL format is saved.

    Returns
    -------
    data_objects: Tuple[Events, Events, Events, Events]
        Events- and Pitch-objects for both teams and both halves.

    Notes
    -----
    The DFL format of handling event data information involves an elaborate use of
    certain event attributes, which attach additional information to certain events.
    There also exist detailed definitions for these attributes. Parsing this information
    involves quite a bit of logic and is planned to be included in further releases. As
    of now, qualifier information is parsed as a string in the `qualifier` column of the
    returned DataFrame and might be transformed to a dict of the form:
    `{attribute: value}`.
    """
    # set up XML tree
    tree = etree.parse(str(filepath_events))
    root = tree.getroot()

    # find start of halves
    start_times = {}
    start_events = root.findall("Event/KickoffWhistle")
    # look at different encodings as the data format changed over time
    if not bool(start_events):  # if no KickoffWhistle is in data search for Kickoff
        start_events = root.findall("Event/Kickoff")
    if not bool(start_events):  # if no Kickoff is in data search for KickOff
        start_events = root.findall("Event/KickOff")
    for event in start_events:
        if event.get("GameSection") is not None:
            start_times[event.get("GameSection")] = iso8601.parse_date(
                event.getparent().get("EventTime")
            )

    # find end of halves
    end_times = {}
    end_events = root.findall("Event/FinalWhistle")
    for event in end_events:
        if event.get("GameSection") is not None:
            end_times[event.get("GameSection")] = iso8601.parse_date(
                event.getparent().get("EventTime")
            )

    # initialize periods
    segments = list(start_times.keys())
    periods = {}
    for segment in segments:
        periods[segment] = (start_times[segment], end_times[segment])

    # set up bins
    team_events = {segment: {} for segment in segments}

    # loop over events
    for elem in root.findall("Event"):
        # initialize
        event = {}

        # check for structure that is an element Event with a single child
        if len(elem) != 1:
            warnings.warn(
                "An XML Event has multiple children. This likely causes imprecise "
                "Event descriptions and outcomes."
            )

        # absolute time information (timestamp)
        event["timestamp"] = iso8601.parse_date(elem.get("EventTime"))
        event["gameclock"] = np.nan

        # segment in which event took place
        segment = None
        for seg in segments:
            if periods[seg][0] <= event["timestamp"] <= periods[seg][1]:
                segment = seg
        # assign to closest start point if not within any segments
        if segment is None:
            seg_ind = np.argmin(
                [np.abs(event["timestamp"] - periods[seg][0]) for seg in segments]
            )
            segment = segments[int(seg_ind)]

        # relative time information (gameclock)
        event["gameclock"] = (event["timestamp"] - periods[segment][0]).total_seconds()
        event["minute"] = np.floor(event["gameclock"] / 60)
        event["second"] = np.floor(event["gameclock"] - event["minute"] * 60)

        # description, outcome, team, and player
        child = next(iter(elem))
        eID, attrib = _get_event_description(child)
        outcome = _get_event_outcome(eID, attrib)
        tID, pID = _get_event_team_and_player(eID, attrib)
        event["eID"] = eID
        event["qualifier"] = attrib
        event["outcome"] = outcome
        event["tID"] = tID
        event["pID"] = pID

        # insert to bin
        if tID not in team_events[segment]:
            team_events[segment][tID] = []
        if event["eID"] == "Substitution":  # split for the special case substitution
            # in-sub
            event["eID"] = "InSubstitution"
            event["pID"] = event["qualifier"]["PlayerIn"]
            team_events[segment][tID].append(event)
            # out-sub
            event["eID"] = "OutSubstitution"
            event["pID"] = event["qualifier"]["PlayerOut"]
            team_events[segment][tID].append(event)
        else:
            team_events[segment][tID].append(event)

    # postprocessing
    team_dfs = {segment: {} for segment in segments}
    for segment in segments:

        # teams
        teams = [tID for tID in team_events[segment] if tID is not None]

        # loop over teams
        for tID in teams:

            # assign events with tID None to both teams
            team_events[segment][tID] += team_events[segment][None]

            # transform to data DataFrame
            team_dfs[segment][tID] = pd.DataFrame(team_events[segment][tID])

            # columns to standard order
            team_dfs[segment][tID] = team_dfs[segment][tID][
                [
                    "eID",
                    "gameclock",
                    "tID",
                    "pID",
                    "outcome",
                    "timestamp",
                    "minute",
                    "second",
                    "qualifier",
                ]
            ]
            team_dfs[segment][tID] = team_dfs[segment][tID].sort_values("gameclock")
            team_dfs[segment][tID] = team_dfs[segment][tID].reset_index(drop=True)

    # check for teams
    team1 = list(team_dfs[segments[0]].keys())[0]
    team2 = list(team_dfs[segments[0]].keys())[1]
    if not np.all([team1 in team_dfs[segment].keys() for segment in segments]):
        KeyError(
            f"Found tID {team1} of the first segment missing in at least one "
            f"other segment!"
        )
    if not np.all([team2 in team_dfs[segment].keys() for segment in segments]):
        KeyError(
            f"Found tID {team2} of the first segment missing in at least one "
            f"other segment!"
        )

    # assembly
    events_team1_ht1 = Events(
        events=team_dfs[segments[0]][team1],
    )
    events_team1_ht2 = Events(
        events=team_dfs[segments[1]][team1],
    )
    events_team2_ht1 = Events(
        events=team_dfs[segments[0]][team2],
    )
    events_team2_ht2 = Events(
        events=team_dfs[segments[1]][team2],
    )
    data_objects = (
        events_team1_ht1,
        events_team1_ht2,
        events_team2_ht1,
        events_team2_ht2,
    )

    return data_objects


def read_position_data_xml(
    filepath_positions: Union[str, Path],
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
    filepath_positions: str or pathlib.Path
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
    pitch = read_pitch_from_mat_info_xml(filepath_mat_info)

    # create or check links
    if links_jID_to_xID is None or links_pID_to_jID is None:
        links_jID_to_xID, links_pID_to_jID = create_links_from_mat_info(
            filepath_mat_info
        )
    else:
        pass
        # potential check

    # create periods
    periods, framerate_est = _create_periods_from_dat(filepath_positions)
    segments = list(periods.keys())

    # infer data array shapes
    number_of_home_players = max(links_jID_to_xID["Home"].values()) + 1
    number_of_away_players = max(links_jID_to_xID["Away"].values()) + 1
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
    for _, frame_set in etree.iterparse(filepath_positions, tag="FrameSet"):

        # ball
        if frame_set.get("TeamId").lower() == "ball":
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
                continue
                # possible error or warning

            # insert (x,y) data to correct place in bin
            start = int(frames[0].get("N")) - periods[segment][0]
            end = int(frames[-1].get("N")) - periods[segment][0] + 1
            x_col = (links_jID_to_xID[team][jrsy]) * 2
            y_col = (links_jID_to_xID[team][jrsy]) * 2 + 1
            xydata[team][segment][start:end, x_col] = np.array(
                [float(frame.get("X")) for frame in frames]
            )
            xydata[team][segment][start:end, y_col] = np.array(
                [float(frame.get("Y")) for frame in frames]
            )

        frame_set.clear()

    # create XY objects
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
        definitions={1: "Home", 2: "Away"},
        framerate=framerate_est,
    )
    possession_ht2 = Code(
        code=np.array(codes["possession"]["secondHalf"]),
        name="possession",
        definitions={1: "Home", 2: "Away"},
        framerate=framerate_est,
    )
    ballstatus_ht1 = Code(
        code=np.array(codes["ballstatus"]["firstHalf"]),
        name="ballstatus",
        definitions={0: "Dead", 1: "Alive"},
        framerate=framerate_est,
    )
    ballstatus_ht2 = Code(
        code=np.array(codes["ballstatus"]["secondHalf"]),
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

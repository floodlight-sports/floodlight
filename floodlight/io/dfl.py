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
from floodlight.core.teamsheet import Teamsheet
from floodlight.io.utils import get_and_convert


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


def read_teamsheets_from_mat_info_xml(filepath_mat_info) -> Dict[str, Teamsheet]:
    """Reads match_information XML file and returns two teamsheet objects for the home
    and the away team.

    Parameters
    ----------
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in DFL format is saved.

    Returns
    -------
    teamsheets: Dict[str, Teamsheet]
        Dictionary with teamsheets for the home team and the away team.
    """
    # set up XML tree
    tree = etree.parse(str(filepath_mat_info))
    root = tree.getroot()

    # initialize teamsheets
    teamsheets = {
        "Home": pd.DataFrame(
            columns=["player", "position", "team", "jID", "pID", "tID"]
        ),
        "Away": pd.DataFrame(
            columns=["player", "position", "team", "jID", "pID", "tID"]
        ),
    }

    # find team ids
    team_informations = root.find("MatchInformation").find("Teams")
    home_id = root.find("MatchInformation").find("General").get("HomeTeamId")
    if "AwayTeamId" in root.find("MatchInformation").find("General").attrib:
        away_id = root.find("MatchInformation").find("General").get("AwayTeamId")
    elif "GuestTeamId" in root.find("MatchInformation").find("General").attrib:
        away_id = root.find("MatchInformation").find("General").get("GuestTeamId")
    else:
        away_id = None

    # parse player information
    for team_info in team_informations:
        if team_info.get("TeamId") == home_id:
            team = "Home"
        elif team_info.get("TeamId") == away_id:
            team = "Away"
        else:
            team = None

        # skip referees sometimes referred to as a team in new data formats
        if team not in ["Home", "Away"]:
            continue

        # create list of players
        players = team_info.find("Players")

        # create teamsheets
        teamsheets[team]["player"] = [
            get_and_convert(player, "Shortname", str) for player in players
        ]
        teamsheets[team]["pID"] = [
            get_and_convert(player, "PersonId", str) for player in players
        ]
        teamsheets[team]["jID"] = [
            get_and_convert(player, "ShirtNumber", int) for player in players
        ]
        teamsheets[team]["position"] = [
            get_and_convert(player, "PlayingPosition", str) for player in players
        ]
        teamsheets[team]["tID"] = team_info.get("TeamId")
        teamsheets[team]["team"] = team_info.get("TeamName")

    # create teamsheet objects
    for team in teamsheets:
        teamsheets[team] = Teamsheet(teamsheets[team])

    return teamsheets


def read_event_data_xml(
    filepath_events: Union[str, Path],
    filepath_mat_info: Union[str, Path],
    teamsheet_home: Teamsheet = None,
    teamsheet_away: Teamsheet = None,
) -> Tuple[Dict[str, Dict[str, Events]], Dict[str, Teamsheet], Pitch]:
    """Parses a DFL Match Event XML file and extracts the event data as well as
    teamsheets.

    The structure of the official tracking system of the DFL (German Football League)
    contains two separate xml files, one containing the actual data as well as a
    metadata file containing information about teams, pitch size, and start- and
    endframes of match periods. This function provides high-level access to DFL data by
    parsing "the full match" and returning Events-objects parsed from the event data
    xml-file as well as Teamsheet-objects parsed from the metadata xml-file. The number
    of segments is inferred from the data, yet data for each segment is stored in a
    separate object.

    Parameters
    ----------
    filepath_events: str or pathlib.Path
        Full path to XML File where the Event data in DFL format is saved.
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in DFL format is saved.
    teamsheet_home: Teamsheet, optional
        Teamsheet-object for the home team used to assign the tIDs of the teams to the
        "Home" and "Away" position. If given as None (default), teamsheet is extracted
        from the Match Information XML file.
    teamsheet_away: Teamsheet, optional
        Teamsheet-object for the away team. If given as None (default), teamsheet is
        extracted from the Match Information XML file. See teamsheet_home for details.

    Returns
    -------
    data_objects: Tuple[Dict[str, Dict[str, Events]], Dict[str, Teamsheet], Pitch]
        Tuple of (nested) floodlight core objects with shape (events_objects,
        teamsheets, pitch).

        ``events_objects`` is a nested dictionary containing ``Events`` objects for
        each team and segment of the form ``events_objects[segment][team] = Events``.
        For a typical league match with two halves and teams this dictionary looks like:
        ``{
        'firstHalf': {'Home': Events, 'Away': Events},
        'secondHalf': {'Home': Events,'Away': Events}
        }``.

        ``teamsheets`` is a dictionary containing ``Teamsheet`` objects for each team
        of the form ``teamsheets[team] = Teamsheet``.

        ``pitch`` is a ``Pitch`` object corresponding to the data.

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

    # read metadata
    pitch = read_pitch_from_mat_info_xml(filepath_mat_info)

    # create or check teamsheet objects
    if teamsheet_home is None and teamsheet_away is None:
        teamsheets = read_teamsheets_from_mat_info_xml(filepath_mat_info)
        teamsheet_home = teamsheets["Home"]
        teamsheet_away = teamsheets["Away"]
    elif teamsheet_home is None:
        teamsheets = read_teamsheets_from_mat_info_xml(filepath_mat_info)
        teamsheet_home = teamsheets["Home"]
    elif teamsheet_away is None:
        teamsheets = read_teamsheets_from_mat_info_xml(filepath_mat_info)
        teamsheet_away = teamsheets["Away"]
    else:
        pass
        # potential check

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

    # link team1 and team2 to home and away
    home_tID = teamsheet_home.teamsheet.at[0, "tID"]
    away_tID = teamsheet_away.teamsheet.at[0, "tID"]
    links_team_to_role = {
        "Home": home_tID,
        "Away": away_tID,
    }

    # check if home and away tIDs occur in event data
    if team1 != home_tID and team2 != home_tID:
        raise AttributeError(
            f"Neither tID of teams in the event data ({team1} and {team2}) "
            f"matches the tID of the home team from the "
            f"teamsheet_home ({home_tID})!"
        )
    if team1 != away_tID and team2 != away_tID:
        raise AttributeError(
            f"Neither tID of teams in the event data ({team1} and {team2}) "
            f"matches the tID of the away team from the "
            f"teamsheet_away ({away_tID})!"
        )

    # create objects
    events_objects = {}
    for segment in segments:
        events_objects[segment] = {}
        for team in ["Home", "Away"]:
            events_objects[segment][team] = Events(
                events=team_dfs[segment][links_team_to_role[team]],
            )
    teamsheets = {
        "Home": teamsheet_home,
        "Away": teamsheet_away,
    }

    # pack objects
    data_objects = (events_objects, teamsheets, pitch)

    return data_objects


def read_position_data_xml(
    filepath_positions: Union[str, Path],
    filepath_mat_info: Union[str, Path],
    teamsheet_home: Teamsheet = None,
    teamsheet_away: Teamsheet = None,
) -> Tuple[
    Dict[str, Dict[str, XY]],
    Dict[str, Code],
    Dict[str, Code],
    Dict[str, Teamsheet],
    Pitch,
]:
    """Parse DFL files and extract position data, possession and ballstatus codes as
    well as pitch information and teamsheets.

    The structure of the official tracking system of the DFL (German Football League)
    contains two separate xml files, one containing the actual data as well as a
    metadata file containing information about teams, pitch size, and start- and
    endframes of match periods. However, since no information about framerate is
    contained in the metadata, the framerate is estimated from the time difference
    between individual frames. This function provides high-level access to DFL data by
    parsing "the full match" and returning XY- and Code-objects parsed from the position
    data xml-file as well as Pitch- and Teamsheet-objects parsed from the metadata
    xml-file.

    Parameters
    ----------
    filepath_positions: str or pathlib.Path
        Full path to XML File where the Position data in DFL format is saved.
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in DFL format is saved.
    teamsheet_home: Teamsheet, optional
        Teamsheet-object for the home team used to create link dictionaries of the form
        `links[team][jID] = xID` and  `links[team][pID] = jID`. The links are used to
        map players to a specific xID in the respective XY objects. Should be supplied
        for custom ordering. If given as None (default), teamsheet is extracted from the
        Match Information XML file and its xIDs are assigned in order of appearance.
    teamsheet_away: Teamsheet, optional
        Teamsheet-object for the away team. If given as None (default), teamsheet is
        extracted from the Match Information XML file. See teamsheet_home for details.

    Returns
    -------
    data_objects: Tuple[Dict[str, Dict[str, XY]], Dict[str, Code], Dict[str, Code], \
     Dict[str, Teamsheet], Pitch]
        Tuple of (nested) floodlight core objects with shape (xy_objects,
        possession_objects, ballstatus_objects, teamsheets, pitch).

        ``xy_objects`` is a nested dictionary containing ``XY`` objects for each team
        and segment of the form ``xy_objects[segment][team] = XY``. For a typical
        league match with two halves and teams this dictionary looks like:
        ``{'firstHalf': {'Home': XY, 'Away': XY}, 'secondHalf': {'Home': XY, 'Away':
        XY}}``.

        ``possession_objects`` is a dictionary containing ``Code`` objects with
        possession information (home or away) for each segment of the form
        ``possession_objects[segment] = Code``.

        ``ballstatus_objects`` is a dictionary containing ``Code`` objects with
        ballstatus information (dead or alive) for each segment of the form
        ``ballstatus_objects[segment] = Code``.

        ``teamsheets`` is a dictionary containing ``Teamsheet`` objects for each team
        of the form ``teamsheets[team] = Teamsheet``.

        ``pitch`` is a ``Pitch`` object corresponding to the data.
    """
    # read metadata
    pitch = read_pitch_from_mat_info_xml(filepath_mat_info)

    # create or check teamsheet objects
    if teamsheet_home is None and teamsheet_away is None:
        teamsheets = read_teamsheets_from_mat_info_xml(filepath_mat_info)
        teamsheet_home = teamsheets["Home"]
        teamsheet_away = teamsheets["Away"]
    elif teamsheet_home is None:
        teamsheets = read_teamsheets_from_mat_info_xml(filepath_mat_info)
        teamsheet_home = teamsheets["Home"]
    elif teamsheet_away is None:
        teamsheets = read_teamsheets_from_mat_info_xml(filepath_mat_info)
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
    links_pID_to_jID = {
        "Home": teamsheet_home.get_links("pID", "jID"),
        "Away": teamsheet_away.get_links("pID", "jID"),
    }

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

    # create objects
    xy_objects = {}
    possession_objects = {}
    ballstatus_objects = {}
    for segment in segments:
        xy_objects[segment] = {}
        possession_objects[segment] = Code(
            code=np.array(codes["possession"][segment]),
            name="possession",
            definitions={1: "Home", 2: "Away"},
            framerate=framerate_est,
        )
        ballstatus_objects[segment] = Code(
            code=np.array(codes["ballstatus"][segment]),
            name="ballstatus",
            definitions={0: "Dead", 1: "Alive"},
            framerate=framerate_est,
        )
        for team in ["Home", "Away", "Ball"]:
            xy_objects[segment][team] = XY(
                xy=xydata[team][segment],
                framerate=framerate_est,
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

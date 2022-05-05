import os.path
import warnings
from typing import Any, Dict, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request
from lxml import etree

from floodlight.core.code import Code
from floodlight.core.events import Events
from floodlight.core.pitch import Pitch
from floodlight.core.xy import XY
from settings import DATA_DIR

# ----------------------------- StatsPerform Open Format -------------------------------


def _create_metadata_from_open_csv_df(
    csv_df: pd.DataFrame,
) -> Tuple[Dict[int, tuple], Pitch]:
    """Creates meta information from the CSV file as parsed by pd.read_csv().

    Parameters
    ----------
    csv_df: DataFrame
        Containing all data from the positions CSV file as DataFrame.

    Returns
    -------
    periods: Dict[int, int]
        Dictionary with start and endframes:
            ``periods[segment] = (startframe, endframe)``.
    pitch: Pitch
        Playing Pitch object.
    """

    # create pitch
    pi_len = csv_df["pitch_dimension_long_side"].values[0]
    pi_wid = csv_df["pitch_dimension_short_side"].values[0]
    pitch = Pitch.from_template(
        "statsperform_open",
        length=pi_len,
        width=pi_wid,
        sport="football",
    )

    # create periods for segments, coded as jumps in the frame sequence
    periods = {}
    frame_values = csv_df["frame_count"].unique()

    seg_idx = np.where(np.diff(frame_values, prepend=frame_values[0]) > 1)
    seg_idx = np.insert(seg_idx, 0, 0)
    seg_idx = np.append(seg_idx, len(frame_values))
    for segment in range(len(seg_idx) - 1):
        start = int(frame_values[seg_idx[segment]])
        end = int(frame_values[seg_idx[segment + 1] - 1])
        periods[segment] = (start, end)

    return periods, pitch


def _create_links_from_open_csv_df(
    csv_df: pd.DataFrame, team_ids: Dict[str, float]
) -> Dict[str, Dict[int, int]]:
    """Checks the entire parsed StatsPerform CSV file for unique jIDs (jerseynumbers)
    and creates a dictionary linking jIDs to xIDs in ascending order.

    Parameters
    ----------
    csv_df: pd.DataFrame
        Containing the parsed DataFrame from the positions CSV file
    team_ids: Dict[str, float]
        Dictionary that stores the StatsPerform team id of the Home and Away team

    Returns
    -------
    links: Dict[str, Dict[int, int]]
        A link dictionary of the form ``links[team][jID] = xID``.
    """
    links = {}
    for team in team_ids:
        links[team] = {
            jID: xID
            for xID, jID in enumerate(
                csv_df[csv_df["team_id"] == team_ids[team]]["jersey_no"].unique()
            )
        }
    return links


def _read_open_event_csv_single_line(
    line: str,
) -> Tuple[Dict, str, str]:
    """Extracts all relevant information from a single line of StatsPerform's Event CSV
    file (i.e. one single event in the data).

    Parameters
    ----------
    line: str
        One full line from StatsPerform's Event CSV file.

    Returns
    -------
    event: Dict
        Dictionary with relevant event information in the form:
        ``event[attribute] = value``.
    """
    event = {}
    attrib = line.split(sep=",")

    # description
    event["eID"] = attrib[5].replace(" ", "")

    # relative time
    event["gameclock"] = float(attrib[4])
    event["frameclock"] = float(attrib[2])

    # segment, player and team
    segment = attrib[3]
    team = attrib[9]
    event["tID"] = team
    event["pID"] = attrib[8]

    # outcome
    event["outcome"] = np.nan
    if "Won" in attrib[5].split(" "):
        event["outcome"] = 1
    elif "Lost" in attrib[5].split(" "):
        event["outcome"] = 0

    # minute and second of game
    event["minute"] = np.floor(event["gameclock"] / 60)
    event["second"] = np.floor(event["gameclock"] - event["minute"] * 60)

    # additional information (qualifier)
    event["qualifier"] = {
        "event_id": attrib[1],
        "event_type_id": attrib[6],
        "sequencenumber": attrib[7],
        "jersey_no": attrib[10],
        "is_pass": attrib[11],
        "is_cross": attrib[12],
        "is_corner": attrib[13],
        "is_free_kick": attrib[14],
        "is_goal_kick": attrib[15],
        "passtypeid": attrib[16],
        "wintypeid": attrib[17],
        "savetypeid": attrib[18],
        "possessionnumber": attrib[19],
    }

    return event, team, segment


def create_links_from_open_tracking_csv(
    filepath_tracking: Union[str, Path]
) -> Dict[str, Dict[int, int]]:
    """Parses the entire StatsPerform CSV file for unique jIDs (jerseynumbers) and team
    Ids and creates a dictionary linking jIDs to xIDs in ascending order.

    Parameters
    ----------
    filepath_tracking: str or pathlib.Path
        CSV file where the position data in StatsPerform format is saved.

    Returns
    -------
    links: Dict[str, Dict[str, int]]
        A link dictionary of the form ``links[team][jID] = xID``.
    """
    # read dat-file into pd.DataFrame
    dat_df = pd.read_csv(str(filepath_tracking))

    # initialize team and ball ids
    team_ids = {"Home": 1.0, "Away": 2.0}
    ball_id = 4

    # check for additional tIDs
    for ID in dat_df["team_id"].unique():
        if not (ID in team_ids.values() or ID == ball_id):
            warnings.warn(f"Team ID {ID} did not match any of the standard IDs!")

    return _create_links_from_open_csv_df(dat_df, team_ids)


def read_open_event_data_csv(
    filepath_events: Union[str, Path],
) -> Tuple[Events, Events, Events, Events]:
    """Parses an open StatsPerform Match Event CSV file and extracts the event data.

    This function provides a high-level access to the particular openly published
    StatsPerform Match Events CSV file (e.g. for the Pro Forum '22) and returns Event
    objects for both teams.

    Parameters
    ----------
    filepath_events: str or pathlib.Path
        Full path to XML File where the Event data in StatsPerform CSV format is
        saved
    Returns
    -------
    data_objects: Tuple[Events, Events, Events, Events]
        Events-objects for both teams and both halves.

    Notes
    -----
    StatsPerform's open format of handling provides certain additional event attributes,
    which attach additional information to certain events. As of now, these information
    are parsed as a string in the ``qualifier`` column of the returned DataFrame and can
    be transformed to a dict of form ``{attribute: value}``.
    """
    # initialize bin and variables
    events = {}
    teams = ["1.0", "2.0"]
    segments = ["1", "2"]
    for team in teams:
        events[team] = {segment: pd.DataFrame() for segment in segments}

    # parse event data
    with open(str(filepath_events), "r") as f:
        while True:
            line = f.readline()

            # terminate if at end of file
            if len(line) == 0:
                break

            # skip the head
            if line.split(sep=",")[3] == "current_phase":
                continue

            # read single line
            event, team, segment = _read_open_event_csv_single_line(line)

            # insert to bin
            if team:
                events[team][segment] = events[team][segment].append(
                    event, ignore_index=True
                )
            else:  # if no clear assignment possible, insert to bins for both teams
                for team in teams:
                    events[team][segment] = events[team][segment].append(
                        event, ignore_index=True
                    )

    # assembly
    t1_ht1 = Events(
        events=events["1.0"]["1"],
    )
    t1_ht2 = Events(
        events=events["1.0"]["2"],
    )
    t2_ht1 = Events(
        events=events["2.0"]["1"],
    )
    t2_ht2 = Events(
        events=events["2.0"]["2"],
    )
    data_objects = (t1_ht1, t1_ht2, t2_ht1, t2_ht2)

    return data_objects


def read_open_tracking_data_csv(
    filepath_tracking: Union[str, Path],
    links: Dict[str, Dict[int, int]] = None,
) -> Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Pitch]:
    """Parses an open StatsPerform CSV file and extract position data and possession
    codes as well as pitch information.

     Openly published StatsPerform position data (e.g. for the Pro Forum '22) is stored
     in a CSV file containing all position data (for both halves) as well as information
     about players, the pitch, and the ball possession. This function provides a
     high-level access to StatsPerform data by parsing the CSV file.

    Parameters
    ----------
    filepath_tracking: str or pathlib.Path
        Full path to the CSV file.
    links: Dict[str, Dict[int, int]], optional
        A link dictionary of the form ``links[team][jID] = xID``. Player's are
        identified in StatsPerform files via jID, and this dictionary is used to map
        them to a specific xID in the respective XY objects. Should be supplied if that
        order matters. If None is given (default), the links are automatically extracted
        from the CSV file at the cost of a second pass through the entire file.

    Returns
    -------
    data_objects: Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Pitch]
        XY-, Code-, and Pitch-objects for both teams and both halves. The order is
        (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2,
        possession_ht1, possession_ht2, pitch)
    """
    # parse the CSV file into pd.DataFrame
    dat_df = pd.read_csv(str(filepath_tracking))

    # initialize team and ball ids
    team_ids = {"Home": 1.0, "Away": 2.0}
    ball_id = 4

    # check for additional tIDs
    for ID in dat_df["team_id"].unique():
        if not (ID in team_ids.values() or ID == ball_id):
            warnings.warn(f"Team ID {ID} did not match any of the standard IDs!")

    # create or check links
    if links is None:
        links = _create_links_from_open_csv_df(dat_df, team_ids)
    else:
        pass
        # potential check vs jerseys in dat file

    # create periods and pitch
    periods, pitch = _create_metadata_from_open_csv_df(dat_df)
    segments = list(periods.keys())

    # infer data shapes
    number_of_players = {team: len(links[team]) for team in links}
    number_of_frames = {}
    for segment in segments:
        start = periods[segment][0]
        end = periods[segment][1]
        number_of_frames[segment] = end - start + 1

    # bins
    codes = {"possession": {segment: [] for segment in segments}}
    xydata = {
        "Home": {
            segment: np.full(
                [
                    number_of_frames[segment],
                    number_of_players[list(links.keys())[0]] * 2,
                ],
                np.nan,
            )
            for segment in periods
        },
        "Away": {
            segment: np.full(
                [
                    number_of_frames[segment],
                    number_of_players[list(links.keys())[1]] * 2,
                ],
                np.nan,
            )
            for segment in periods
        },
        "Ball": {
            segment: np.full([number_of_frames[segment], 2], np.nan)
            for segment in periods
        },
    }

    # loop
    for segment in segments:

        # teams
        for team in team_ids:
            team_df = dat_df[dat_df["team_id"] == team_ids[team]]
            for pID in team_df["player_id"].unique():
                # extract player information
                pl_df = team_df[team_df["player_id"] == pID]
                frames = pl_df["frame_count"].values
                x_position = pl_df["pos_x"].values
                y_position = pl_df["pos_y"].values

                # compute appearance of player in segment
                appearance = np.array(
                    [
                        (periods[segment][0] <= frame <= periods[segment][-1])
                        for frame in frames
                    ]
                )
                # check for players that did not play in segment
                if not np.sum(appearance):
                    continue

                # insert player position to bin array
                jrsy = pl_df["jersey_no"].values[0]
                x_col = (links[team][jrsy] - 1) * 2
                y_col = (links[team][jrsy] - 1) * 2 + 1
                start = frames[appearance][0] - periods[segment][0]
                end = frames[appearance][-1] - periods[segment][0] + 1
                xydata[team][segment][start:end, x_col] = x_position[appearance]
                xydata[team][segment][start:end, y_col] = y_position[appearance]

        # ball
        ball_df = dat_df[dat_df["team_id"] == 4]
        frames = ball_df["frame_count"].values
        appearance = np.array(
            [(periods[segment][0] <= frame <= periods[segment][-1]) for frame in frames]
        )
        xydata["Ball"][segment][:, 0] = ball_df["pos_x"].values[appearance]
        xydata["Ball"][segment][:, 1] = ball_df["pos_x"].values[appearance]

        # update codes
        codes["possession"][segment] = ball_df["possession"].values[appearance]

    # create XY objects
    home_ht1 = XY(xy=xydata["Home"][0], framerate=10)
    home_ht2 = XY(xy=xydata["Home"][1], framerate=10)
    away_ht1 = XY(xy=xydata["Away"][0], framerate=10)
    away_ht2 = XY(xy=xydata["Away"][1], framerate=10)
    ball_ht1 = XY(xy=xydata["Ball"][0], framerate=10)
    ball_ht2 = XY(xy=xydata["Ball"][1], framerate=10)

    # create Code objects
    poss_ht1 = Code(
        name="possession",
        code=codes["possession"][0],
        definitions=dict([(team_id, team) for team, team_id in team_ids.items()]),
        framerate=10,
    )
    poss_ht2 = Code(
        name="possession",
        code=codes["possession"][1],
        definitions=dict([(team_id, team) for team, team_id in team_ids.items()]),
        framerate=10,
    )

    data_objects = (
        home_ht1,
        home_ht2,
        away_ht1,
        away_ht2,
        ball_ht1,
        ball_ht2,
        poss_ht1,
        poss_ht2,
        pitch,
    )
    return data_objects


# ----------------------------- StatsPerform Format ---------------------------


def _get_and_convert(dic: dict, key: Any, value_type: type, default: Any = None) -> Any:
    """Performs dictionary get and type conversion simultaneously.

    Parameters
    ----------
    dic: dict
        Dictionary to be queried.
    key: Any
        Key to be looked up.
    value_type: type
        Desired output type the value should be cast into.
    default: Any, optional
        Return value if key is not in dic, defaults to None.

    Returns
    -------
    value: value_type
        Returns the value for key if key is in dic, else default. Tries type conversion
        to `type(value) = value_type`. If type conversion fails, e.g. by trying to force
        something like `float(None)` due to a missing dic entry, value is returned in
        its original data type.
    """
    value = dic.get(key, default)
    try:
        value = value_type(value)
    except TypeError:
        pass
    except ValueError:
        pass

    return value


def _read_full_tracking_data_txt_single_line(
    package: str,
) -> Tuple[
    int,
    int,
    Dict[str, Dict[str, Tuple[float, float, float]]],
    Dict[str, Union[str, tuple]],
]:
    """Extracts all relevant information from a single line of StatsPerform's .txt file
    (i.e. one frame of data).

    Parameters
    ----------
    package: str
        One full line from StatsPerform's .txt-file, equals one "package" according
        to the file-format documentation.

    Returns
    -------
    gameclock: int
        The gameclock of the current segment in milliseconds.
    segment: int
        The segment ID.
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

    # read chunks
    chunks = package.split(":")
    time_chunk = chunks[0]
    player_chunks = chunks[1].split(";")

    ball_chunk = None
    if len(chunks) > 2:  # check if ball information exist in chunk
        ball_chunk = chunks[2]

    # time chunk
    # systemclock = time_chunk.split(";")[0]
    # possible check or synchronization step
    timeinfo = time_chunk.split(";")[1].split(",")
    gameclock = int(timeinfo[0])
    segment = int(timeinfo[1])
    # ballstatus = timeinfo[2].split(":")[0] == '0'  # '0' seems to be always the case?

    # player chunks
    for player_chunk in player_chunks:

        # skip final entry of chunk
        if not player_chunk or player_chunk == "\n":
            continue

        # read team
        chunk_data = player_chunk.split(",")
        if chunk_data[0] in ["0", "3"]:
            team = "Home"
        elif chunk_data[0] in ["1", "4"]:
            team = "Away"
        else:
            team = "Other"

        # read IDs
        # pID = chunk_data[1]
        jID = chunk_data[2]

        # read positions
        x, y = map(lambda x: float(x), chunk_data[3:])

        # assign
        positions[team][jID] = (x, y)

    # ball chunk
    if ball_chunk is not None:
        x, y, z = map(lambda x: float(x), ball_chunk.split(";")[0].split(","))
        # ball["position"] = (x, y, z)  # z-coordinate is not yet supported
        ball["position"] = (x, y)

    return gameclock, segment, positions, ball


def _read_time_information_from_tracking_data_txt(
    file_location_txt: Union[str, Path],
    estimate_framerate: bool = None,
) -> Tuple[Dict, int]:
    """Reads StatsPerform's tracking .txt file and extracts information about the first
    and final frame of periods. If specified, the framerate is estimated from the
    gameclock difference between the samples.

    Parameters
    ----------
    file_location_txt: str or pathlib.Path
        Full path to the txt file containing the tracking data.
    estimate_framerate: bool
        Whether the framerate should estimated. Returns None if set to False.
    Returns
    -------
    periods: Dict
        Dictionary with start and endframes:
        `periods[segment] = [startframe, endframe]`.
    framerate_est: int or None
        Estimated temporal resolution of data in frames per second/Hertz if specified
        or None otherwise.
    """

    # bins
    startframes = {}
    endframes = {}
    framerate_est = None

    # read txt file from disk
    file_txt = open(file_location_txt, "r")

    # loop
    last_gameclock = None
    last_segment = None
    for package in file_txt.readlines():

        # read gameclock and segment
        gameclock, segment, _, _ = _read_full_tracking_data_txt_single_line(package)

        # update periods
        if segment not in startframes:
            startframes[segment] = gameclock
            if last_gameclock is not None:
                endframes[last_segment] = last_gameclock

        # estimate framerate if desired
        if estimate_framerate and last_gameclock is not None:
            delta = np.absolute(gameclock - last_gameclock)  # in milliseconds
            if framerate_est is None:
                framerate_est = int(1000 / delta)
            elif framerate_est != int(1000 / delta) and last_segment == segment:
                warnings.warn(
                    f"Framerate estimation yielded diverging results."
                    f"The originally estimated framerate of {framerate_est} Hz did not "
                    f"match the current estimation of {int(1000 / delta)} Hz. This "
                    f"might be caused by missing frame(s) in the position data. "
                    f"Continuing by choosing the latest estimation of "
                    f"{int(1000 / delta)} Hz"
                )
                framerate_est = int(1000 / delta)

        # update variables
        last_gameclock = gameclock
        last_segment = segment

    # update end of final segment
    endframes[last_segment] = last_gameclock

    # assembly
    periods = {
        segment: (startframes[segment], endframes[segment]) for segment in startframes
    }

    # close file
    file_txt.close()

    return periods, framerate_est


def _read_jersey_numbers_from_tracking_data_txt(
    file_location_txt: Union[str, Path],
) -> Tuple[set, set]:
    """Reads StatsPerform's tracking .txt file and extracts unique set of jIDs
    (jerseynumbers) for both teams.

    Parameters
    ----------
    file_location_txt: str or pathlib.Path
        Full path to the txt file containing the tracking data.

    Returns
    -------
    home_jIDs: set
    away_jIDs: set
    """

    # bins
    home_jIDs = set()
    away_jIDs = set()

    # read txt file from disk
    file_txt = open(file_location_txt, "r")

    # loop
    for package in file_txt.readlines():

        # read line
        _, _, positions, _ = _read_full_tracking_data_txt_single_line(package)

        # extract jersey numbers
        home_jIDs |= set(positions["Home"].keys())
        away_jIDs |= set(positions["Away"].keys())

    # close file
    file_txt.close()

    return home_jIDs, away_jIDs


def create_links_from_statsperform_tracking_data_txt(
    filepath_txt: Union[str, Path],
) -> Dict[str, Dict[int, int]]:
    """Parses the entire StatsPerform .txt file for unique jIDs (jerseynumbers) and
    creates a dictionary linking jIDs to xIDs in ascending order.

    Parameters
    ----------
    filepath_txt: str or pathlib.Path
        Full path to the txt file containing the tracking data.

    Returns
    -------
    links: Dict[str, Dict[int, int]]
        Link-dictionary of the form `links[team][jID] = xID`.
    """
    homejrsy, awayjrsy = _read_jersey_numbers_from_tracking_data_txt(filepath_txt)

    homejrsy = list(homejrsy)
    awayjrsy = list(awayjrsy)

    homejrsy.sort()
    awayjrsy.sort()

    links = {
        "Home": {jID: xID for xID, jID in enumerate(homejrsy)},
        "Away": {jID: xID for xID, jID in enumerate(awayjrsy)},
    }

    return links


def read_statsperform_event_data_xml(
    filepath_xml: Union[str, Path],
) -> Tuple[Events, Events, Events, Events, Pitch]:
    """Parses a StatsPerform .xml file and extracts event data and pitch information.

    This function provides a high-level access to the StatsPerform Match Events xml file
    and returns Events objects for both teams.

    Parameters
    ----------
    filepath_xml: str
        Full path to the xml file containing the event data.

    Returns
    -------
    data_objects: Tuple[Events, Events, Events, Events, Pitch]
        Events-objects for both teams and both halves and pitch information. The order
        is (home_ht1, home_ht2, away_ht1, away_ht2, pitch).
    """
    # load xml tree into memory
    tree = etree.parse(str(filepath_xml))
    root = tree.getroot()

    # create bins, read segments, and assign teams
    columns = [
        "eID",
        "gameclock",
        "pID",
        "jID",
        "minute",
        "second",
        "at_x",
        "at_y",
        "to_x",
        "to_y",
        "qualifier",
    ]
    segments = [
        f"HT{_get_and_convert(period.attrib, 'IdHalf', str)}"
        for period in root.findall("Events/EventsHalf")
    ]
    teams = ["Home", "Away"]

    # bins
    event_lists = {
        team: {segment: {col: [] for col in columns} for segment in segments}
        for team in teams
    }

    # create links
    links_pID_to_tID = {}
    links_pID_to_jID = {}
    links_pID_to_name = {}
    for teamsheet in root.findall("MatchSheet/Team"):
        # skip referees
        if teamsheet.attrib["Type"] == "Referees":
            continue
        # read team
        team = teamsheet.attrib["Type"][:-4]  # cut 'Team' of e.g. 'HomeTeam'
        # assign player ids to team
        for actor in teamsheet.findall("Actor"):
            if actor.attrib["Occupation"] != "Player":  # coaches etc.
                continue
            links_pID_to_tID[_get_and_convert(actor, "IdActor", int)] = team
            links_pID_to_jID[
                _get_and_convert(actor, "IdActor", int)
            ] = _get_and_convert(actor, "JerseyNumber", int)
            links_pID_to_name[
                _get_and_convert(actor, "IdActor", int)
            ] = _get_and_convert(actor, "NickName", str)

    # loop over events
    for half in root.findall("Events/EventsHalf"):
        # get segment information
        period = _get_and_convert(half.attrib, "IdHalf", str)
        segment = "HT" + str(period)
        for event in half.findall("Event"):
            # read pID
            pID = _get_and_convert(event.attrib, "IdActor1", int)

            # assign team
            team = _get_and_convert(links_pID_to_tID, pID, str)

            # create list of either a single team or both teams if no clear assignment
            if team == "None":
                teams_assigned = teams  # add to both teams
            else:
                teams_assigned = [team]  # only add to one team

            # identifier
            eID = _get_and_convert(event.attrib, "EventName", str)
            jID = _get_and_convert(links_pID_to_jID, pID, int)
            for team in teams_assigned:
                event_lists[team][segment]["eID"].append(eID)
                event_lists[team][segment]["pID"].append(pID)
                event_lists[team][segment]["jID"].append(jID)

            # relative time
            gameclock = _get_and_convert(event.attrib, "Time", int)
            minute = np.floor(gameclock / 60)
            second = np.floor(gameclock - minute * 60)
            for team in teams_assigned:
                event_lists[team][segment]["gameclock"].append(gameclock)
                event_lists[team][segment]["minute"].append(minute)
                event_lists[team][segment]["second"].append(second)

            # location
            at_x = _get_and_convert(event.attrib, "LocationX", float)
            at_y = _get_and_convert(event.attrib, "LocationY", float)
            to_x = _get_and_convert(event.attrib, "TargetX", float)
            to_y = _get_and_convert(event.attrib, "TargetY", float)
            for team in teams_assigned:
                event_lists[team][segment]["at_x"].append(at_x)
                event_lists[team][segment]["at_y"].append(at_y)
                event_lists[team][segment]["to_x"].append(to_x)
                event_lists[team][segment]["to_y"].append(to_y)

            # qualifier
            qual_dict = {}
            for qual_id in event.attrib:
                qual_value = event.attrib.get(qual_id)
                qual_dict[qual_id] = qual_value
            for team in teams_assigned:
                event_lists[team][segment]["qualifier"].append(str(qual_dict))

    # create pitch
    length = _get_and_convert(root.attrib, "FieldLength", int)
    width = _get_and_convert(root.attrib, "FieldWidth", int)
    pitch = Pitch.from_template(
        "statsperform_internal",
        length=length,
        width=width,
        sport="football",
    )

    # assembly
    home_ht1 = Events(
        events=pd.DataFrame(data=event_lists["Home"]["HT1"]),
    )
    home_ht2 = Events(
        events=pd.DataFrame(data=event_lists["Home"]["HT2"]),
    )
    away_ht1 = Events(
        events=pd.DataFrame(data=event_lists["Away"]["HT1"]),
    )
    away_ht2 = Events(
        events=pd.DataFrame(data=event_lists["Away"]["HT2"]),
    )

    data_objects = (home_ht1, home_ht2, away_ht1, away_ht2, pitch)

    return data_objects


def read_statsperform_tracking_data_txt(
    filepath_txt: Union[str, Path],
    links: Dict[str, Dict[int, int]] = None,
) -> Tuple[XY, XY, XY, XY, XY, XY]:
    """Parses a StatsPerform .txt file and extracts position data.

     Internal StatsPerform position data is stored as a .txt file containing all
     position data (for both halves) as well as information about the pitch. This
     function provides a high-level access to StatsPerform data by parsing the txt file.

    Parameters
    ----------
    filepath_txt: str or pathlib.Path
        Full path to the txt file containing the tracking data.
    links: Dict[str, Dict[int, int]], optional
        A link dictionary of the form ``links[team][jID] = xID``. Player's are
        identified in StatsPerform files via jID, and this dictionary is used to map
        them to a specific xID in the respective XY objects. Should be supplied if that
        order matters. If None is given (default), the links are automatically extracted
        from the CSV file at the cost of a second pass through the entire file.

    Returns
    -------
    data_objects: Tuple[XY, XY, XY, XY, XY, XY]
        XY-objects for both teams and both halves. The order is (home_ht1, home_ht2,
        away_ht1, away_ht2, ball_ht1, ball_ht2).
    """

    # read framerate from txt description if stored as f"..._{framerate}FPS..."
    framerate = None
    for file_description in filepath_txt.split("_"):
        if "FPS" in file_description:
            framerate = int(file_description.split("FPS")[0])

    # parse txt file for periods and estimate framerate if not contained in filepath
    periods, framerate_est = _read_time_information_from_tracking_data_txt(
        filepath_txt, framerate is None
    )
    segments = list(periods.keys())
    framerate = framerate_est if framerate is None else framerate

    # create or check links
    if links is None:
        links = create_links_from_statsperform_tracking_data_txt(filepath_txt)
    else:
        pass
        # potential check vs jerseys in txt file

    # infer data array shapes
    number_of_home_players = max(links["Home"].values())
    number_of_away_players = max(links["Away"].values())
    number_of_frames = {}
    for segment in segments:
        number_of_frames[segment] = (
            int((periods[segment][1] - periods[segment][0]) / framerate) + 1
        )

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

    # read txt file from disk
    with open(filepath_txt, "r") as f:
        tracking_data_lines = f.readlines()

    # loop
    for package in tracking_data_lines:

        # read line to get gameclock, player positions and ball info
        (
            gameclock,
            segment,
            positions,
            ball,
        ) = _read_full_tracking_data_txt_single_line(package)

        # check if frame is in any segment
        if segment is None:
            # skip line if not
            continue
        else:
            # otherwise calculate relative frame (in respective segment)
            frame_rel = int((gameclock - periods[segment][0]) / framerate)

        # insert (x,y)-data into np.array
        for team in ["Home", "Away"]:
            for jID in positions[team].keys():
                # map jersey number to array index and infer respective columns
                x_col = (links[team][jID] - 1) * 2
                y_col = (links[team][jID] - 1) * 2 + 1
                xydata[team][segment][frame_rel, x_col] = positions[team][jID][0]
                xydata[team][segment][frame_rel, y_col] = positions[team][jID][1]

        # get ball data
        xydata["Ball"][segment][frame_rel] = ball.get("position", np.nan)

    # create XY objects
    home_ht1 = XY(xy=xydata["Home"][1], framerate=framerate)
    home_ht2 = XY(xy=xydata["Home"][2], framerate=framerate)
    away_ht1 = XY(xy=xydata["Away"][1], framerate=framerate)
    away_ht2 = XY(xy=xydata["Away"][2], framerate=framerate)
    ball_ht1 = XY(xy=xydata["Ball"][1], framerate=framerate)
    ball_ht2 = XY(xy=xydata["Ball"][2], framerate=framerate)

    data_objects = (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2)

    return data_objects


def read_statsperform_event_data_url(
    url_events: str,
) -> Tuple[Events, Events, Events, Events, Pitch]:
    """Reads a URL from the StatsPerform API containing an events csv file and extracts
     the event data.

    Parameters
    ----------
    url_events: str
        URL to the xml file containing the event data.

    Returns
    -------
    data_objects: Tuple[Events, Events, Events, Events, Pitch]
        Events-objects for both teams and both halves and pitch information. The order
        is (home_ht1, home_ht2, away_ht1, away_ht2, pitch).
    """
    data_dir = os.path.join(DATA_DIR, "statsperform")
    temp_file = "events_temp.xml"
    download_event_data_url_to_xml_file(
        url_events=url_events, target_filename=temp_file
    )
    data_objects = read_statsperform_event_data_xml(
        filepath_xml=os.path.join(data_dir, temp_file)
    )
    os.remove(os.path.join(data_dir, temp_file))
    return data_objects


def read_statsperform_tracking_data_url(
    url_tracking: str,
    links: Dict[str, Dict[int, int]] = None,
) -> Tuple[XY, XY, XY, XY, XY, XY]:
    """Reads a URL from the StatsPerform API containing a txt file with tracking data
    and extracts position data and pitch information.

    The tracking from the URL is downloaded into a temporary file stored in the
    repository's internal root ``.data``-folder that is removed afterwards.

    Parameters
    ----------
    url_tracking: str or pathlib.Path
        URL to the txt file containing the tracking data.
    links: Dict[str, Dict[int, int]], optional
        A link dictionary of the form ``links[team][jID] = xID``. Player's are
        identified in StatsPerform files via jID, and this dictionary is used to map
        them to a specific xID in the respective XY objects. Should be supplied if that
        order matters. If None is given (default), the links are automatically extracted
        from the CSV file at the cost of a second pass through the entire file.

    Returns
    -------
    data_objects: Tuple[XY, XY, XY, XY, XY, XY]
        XY- and Pitch-objects for both teams and both halves. The order is
        (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2, pitch)
    """
    data_dir = os.path.join(DATA_DIR, "statsperform")
    temp_file = "tracking_temp.txt"
    download_tracking_data_url_to_txt_file(
        url_tracking=url_tracking, target_filename=temp_file
    )
    data_objects = read_statsperform_tracking_data_txt(
        filepath_txt=os.path.join(data_dir, temp_file),
        links=links,
    )
    os.remove(os.path.join(data_dir, temp_file))
    return data_objects


def download_event_data_url_to_xml_file(
    url_events: str, target_filename: Union[str, Path] = None
):
    """Downloads a tracking data xml file stored at the given URL into a sub folder
    ``statsperform`` within the repository's root ``.data``-folder.

    Parameters
    ----------
    url_events: str
        URL to the xml file containing the event data.
    target_filename: str or pathlib.Path
        The target filename. If None (default), the name is chosen according to URL
        after the final '/'.
    """
    # setup paths
    data_dir = os.path.join(DATA_DIR, "statsperform")
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    if target_filename is None:
        target_filename = url_events.split("/")[-1]
    if target_filename[-4:].lower() != ".xml":
        target_filename += ".xml"

    # write url stream to file
    with urllib.request.urlopen(url_events) as url_stream:
        with open(os.path.join(data_dir, target_filename), "wb") as file_xml:
            while True:
                line = url_stream.read()
                if len(line) == 0:
                    break
                file_xml.write(line)


def download_tracking_data_url_to_txt_file(
    url_tracking: str, target_filename: Union[str, Path] = None
):
    """Downloads a tracking data txt file stored at the given URL into a sub folder
    ``statsperform`` within the repository's root ``.data``-folder.

    Parameters
    ----------
    url_tracking: str
        URL to the txt file containing the tracking data.
    target_filename: str or pathlib.Path
        The target filename. If None (default), the name is chosen according to URL
        after the final '/'.
    """
    # setup paths
    data_dir = os.path.join(DATA_DIR, "statsperform")
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    if target_filename is None:
        target_filename = url_tracking.split("/")[-1]
    if target_filename[-4:].lower() != ".txt":
        target_filename += ".txt"

    # write url stream to file
    with urllib.request.urlopen(url_tracking) as url_stream:
        with open(os.path.join(data_dir, target_filename), "wb") as file_txt:
            while True:
                line = url_stream.read()
                if len(line) == 0:
                    break
                file_txt.write(line)

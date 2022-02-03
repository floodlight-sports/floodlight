import warnings
from typing import Any, Dict, Tuple, Union, List
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request
from lxml import etree


from floodlight.core.code import Code
from floodlight.core.events import Events
from floodlight.core.pitch import Pitch
from floodlight.core.xy import XY


# ----------------------------- StatsPerform Open Format -------------------------------


def _create_metadata_from_csv_df(
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


def _create_links_from_csv_df(
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
            jID: xID + 1
            for xID, jID in enumerate(
                csv_df[csv_df["team_id"] == team_ids[team]]["jersey_no"].unique()
            )
        }
    return links


def _read_event_single_line(
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


def create_links_from_tracking_file(
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

    return _create_links_from_csv_df(dat_df, team_ids)


def read_open_statsperform_event_data_csv(
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
            event, team, segment = _read_event_single_line(line)

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


def read_open_statsperform_tracking_data_csv(
    filepath_tracking: Union[str, Path],
    links: Dict[str, Dict[int, int]] = None,
) -> Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Pitch]:
    """Parses an open StatsPerform CSV file and extract position data and possession
    codes as well as pitch information.

     Openly published StatsPerform position data (e.g. for the Pro Forum '22) is stored
     in a CSV file containing all position data (for both halves) as well as information
     about players, the pitch size, and the ball possession. This function provides a
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
        links = _create_links_from_csv_df(dat_df, team_ids)
    else:
        pass
        # potential check vs jerseys in dat file

    # create periods and pitch
    periods, pitch = _create_metadata_from_csv_df(dat_df)
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


# ----------------------------- StatsPerform Closed Format -----------------------------


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


def _read_gameclocks_and_periods_from_tracking_data(
    tracking_data_lines: List[str],
) -> Tuple[Dict, Dict]:
    """Reads StatsPerform's tracking .txt file from the url and extracts gameclocks and
    periods.

    Parameters
    ----------
    tracking_data_lines: List of str
        All lines from the tracking data txt file.

    Returns
    -------
    gameclocks: Dict
        Dictionary containing a sorted list of all gameclock values within a segment.
    periods: Dict
        Dictionary with start and endframes:
        `periods[segment] = (startframe, endframe)`.
    """

    # bins
    gameclocks = {}
    periods = {}

    # parse the .txt file from the given url
    for package in tracking_data_lines:

        # read line tracking_data_lines[-1]
        gameclock, segment, _, _ = _read_txt_single_line(package)

        # update gameclocks
        if segment not in gameclocks:
            gameclocks[segment] = set()
        gameclocks[segment].add(gameclock)

        # update periods
        if segment not in periods:
            periods[segment] = [gameclock, gameclock]
        else:
            if gameclock <= periods[segment][0]:
                periods[segment][0] = gameclock
            elif gameclock >= periods[segment][1]:
                periods[segment][1] = gameclock

    # sort gameclocks in ascending order
    for segment in gameclocks:
        gameclocks[segment] = sorted(gameclocks[segment])

    return gameclocks, periods


def _read_txt_single_line(
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
    chunks = package.split(";")

    # first chunk (system clock)
    # possible check or synchronization step

    # second chunk (time info)
    timeinfo = chunks[1].split(",")
    gameclock = int(timeinfo[0])
    segment = int(timeinfo[1])
    # ballstatus = timeinfo[2].split(":")[0] == '0'  # 0 seems to be always the case

    # remaining chunks (player ball positions)
    for pos_chunk in chunks[3:]:

        # skip final entry of chunk
        if pos_chunk == "\n":
            continue

        # check if ball or player
        if pos_chunk[0] == ":":  # ball data starts with ':
            x, y, z = map(lambda x: float(x), pos_chunk.split(":")[1].split(","))
            # ball["position"] = (x, y, z)  # z-coordinate is not yet supported
            ball["position"] = (x, y)
        else:  # player

            # read team
            chunk_data = pos_chunk.split(",")
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

    return gameclock, segment, positions, ball


def _read_jersey_numbers_from_tracking_lines(
    tracking_data_lines: List[str],
) -> Tuple[set, set]:
    """Reads StatsPerform's tracking .txt file from the url and extracts unique set of
    jIDs (jerseynumbers) for both teams.


    Parameters
    ----------
    tracking_data_lines: List of str
        All lines from the tracking data txt file.

    Returns
    -------
    home_jIDs: set
    away_jIDs: set
    """

    # bins
    home_jIDs = set()
    away_jIDs = set()

    # parse the .txt file from the given url
    for package in tracking_data_lines:

        # read line
        _, _, positions, _ = _read_txt_single_line(package)

        # extract jersey numbers
        home_jIDs |= positions["Home"].keys()
        away_jIDs |= positions["Away"].keys()

    return home_jIDs, away_jIDs


def create_links_from_tracking_url(
    tracking_data_lines: List[str],
) -> Dict[str, Dict[int, int]]:
    """Parses the entire StatsPerform .txt file for unique jIDs (jerseynumbers) and
    creates a dictionary linking jIDs to xIDs in ascending order.

    Parameters
    ----------
    tracking_data_lines: List of str
        All lines from the tracking data txt file.

    Returns
    -------
    links: Dict[str, Dict[int, int]]
        Link-dictionary of the form `links[team][jID] = xID`.
    """
    homejrsy, awayjrsy = _read_jersey_numbers_from_tracking_lines(tracking_data_lines)

    homejrsy = list(homejrsy)
    awayjrsy = list(awayjrsy)

    homejrsy.sort()
    awayjrsy.sort()

    links = {
        "Home": {jID: xID + 1 for xID, jID in enumerate(homejrsy)},
        "Away": {jID: xID + 1 for xID, jID in enumerate(awayjrsy)},
    }

    return links


def read_statsperform_event_data_xml(
    url_events: Union[str, Path],
) -> Tuple[Events, Events, Events, Events, Pitch]:
    """Parses a StatsPerform .xml file stored at a given URL address for Events.

    This function provides a high-level access to the internal StatsPerform Match Events
    XML file that is stored at a (secret) URL and returns Events objects for both teams.

    Parameters
    ----------
    url_events: str
        Full url to the csv file with the event data.

    Returns
    -------
    data_objects: Tuple[Events, Events, Events, Events]
        Events-objects for both teams and both halves.
    """
    # load xml tree from url into memory
    tree = etree.parse(url_events)
    root = tree.getroot()

    # read segments and assign teams
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
            links_pID_to_tID[_get_and_convert(actor, "IdActor1", int)] = team
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
            pID = _get_and_convert(event.attrib, "IdActor1", str)

            # assign team
            team = _get_and_convert(links_pID_to_tID, pID, str)

            # create list of either a single team or both teams if no clear assignment
            if team == "None":
                teams_assigned = teams  # add to both teams
            else:
                teams_assigned = [team]  # only add to one team

            # identifier
            eID = _get_and_convert(event.attrib, "EventName", str)
            pID = _get_and_convert(event.attrib, "IdActor1", int)
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


def read_statsperform_tracking_data_url(
    url_tracking: Union[str, Path],
    links: Dict[str, Dict[int, int]] = None,
) -> Tuple[XY, XY, XY, XY, XY, XY]:
    """Parses a StatsPerform .txt file stored at a given URL address and extracts
    position data and activeness codes.

     Internal StatsPerform position data is stored as a .txt file containing all
     position data (for both halves) as well as information about players and the
     activeness of the game at a (secret) URL location (StatsEdgeViewer). This function
     provides a high-level access to StatsPerform data by parsing the txt file.

    Parameters
    ----------
    url_tracking: str
        Full url to the txt file with the tracking data.
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
    # parse url and extract lines
    response = urllib.request.urlopen(url_tracking)
    response_lines = response.readlines()
    tracking_data_lines = []
    for line in response_lines:
        tracking_data_lines.append(str(line.decode("utf-8")))

    # read framerate from url address
    framerate = None
    metadata = {}
    for url_info_chunk in url_tracking.split("_"):
        if "FPS" in url_info_chunk:
            framerate = url_info_chunk.split("FPS")[0]
    metadata["framerate"] = int(framerate) if framerate else None

    # parse tracking data lines for metadata, gameclocks and periods
    gameclocks, periods = _read_gameclocks_and_periods_from_tracking_data(
        tracking_data_lines
    )
    segments = list(periods.keys())

    # create or check links
    if links is None:
        links = create_links_from_tracking_url(tracking_data_lines)
    else:
        pass
        # potential check vs jerseys in dat file

    # infer data array shapes
    number_of_home_players = max(links["Home"].values())
    number_of_away_players = max(links["Away"].values())
    number_of_frames = {}
    for segment in segments:
        number_of_frames[segment] = len(gameclocks[segment])

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
    codes = {code: {segment: [] for segment in segments} for code in ["ballstatus"]}

    # loop
    for package in tracking_data_lines:

        # read line to get gameclock, player positions and ball info
        gameclock, segment, positions, ball = _read_txt_single_line(package)

        # check if frame is in any segment
        if segment is None:
            # skip line if not
            continue
        else:
            # otherwise calculate relative frame (in respective segment)
            frame_rel = gameclocks[segment].index(gameclock)

        # insert (x,y)-data into correct np.arary, at correct place (t, xID)
        for team in ["Home", "Away"]:
            for jID in positions[team].keys():
                # map jersey number to array index and infer respective columns
                x_col = (links[team][jID] - 1) * 2
                y_col = (links[team][jID] - 1) * 2 + 1
                xydata[team][segment][frame_rel, x_col] = positions[team][jID][0]
                xydata[team][segment][frame_rel, y_col] = positions[team][jID][1]

        # get ball data
        xydata["Ball"][segment][
            frame_rel,
        ] = ball.get("position", np.nan)
        codes["ballstatus"][segment].append(ball.get("ballstatus", np.nan))

    # create XY objects
    home_ht1 = XY(xy=xydata["Home"][1], framerate=metadata["framerate"])
    home_ht2 = XY(xy=xydata["Home"][2], framerate=metadata["framerate"])
    away_ht1 = XY(xy=xydata["Away"][1], framerate=metadata["framerate"])
    away_ht2 = XY(xy=xydata["Away"][2], framerate=metadata["framerate"])
    ball_ht1 = XY(xy=xydata["Ball"][1], framerate=metadata["framerate"])
    ball_ht2 = XY(xy=xydata["Ball"][2], framerate=metadata["framerate"])

    data_objects = (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2)

    return data_objects

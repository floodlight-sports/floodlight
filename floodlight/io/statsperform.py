import warnings
from typing import Dict, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd


from floodlight.core.code import Code
from floodlight.core.events import Events
from floodlight.core.pitch import Pitch
from floodlight.core.xy import XY


def _create_metadata_from_dat_df(
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
            `periods[segment] = (startframe, endframe)`.
    pitch: Pitch
        Playing Pitch object.
    """

    # create pitch
    pi_len = csv_df["pitch_dimension_long_side"].values[0]
    pi_wid = csv_df["pitch_dimension_short_side"].values[0]
    pitch = Pitch.from_template(
        "statsperform",
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


def _create_links_from_dat_df(
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
        A link dictionary of the form `links[team][jID] = xID`.
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
) -> Tuple[Dict[str, Union[str, int]], str, str]:
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
        `event[attribute] = value`.
    """
    event = {}
    attrib = line.split(sep=",")

    # absolute times
    event["frameclock"] = attrib[2]
    event["gameclock"] = attrib[4]

    # description and outcome
    event["eID"] = attrib[5].replace(" ", "")
    event["outcome"] = np.nan
    if "Won" in attrib[5].split(" "):
        event["outcome"] = 1
    elif "Lost" in attrib[5].split(" "):
        event["outcome"] = 0

    # segment, player and team
    segment = attrib[3]
    event["pID"] = attrib[8]
    team = attrib[9]

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


def create_links_from_dat(filepath_dat: Union[str, Path]) -> Dict[str, Dict[int, int]]:
    """Parses the entire StatsPerform CSV file for unique jIDs (jerseynumbers) and team
    Ids and creates a dictionary linking jIDs to xIDs in ascending order.

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        CSV file where the position data in StatsPerform format is saved.

    Returns
    -------
    links: Dict[str, Dict[str, int]]
        A link dictionary of the form `links[team][jID] = xID`.
    """
    # read dat-file into pd.DataFrame
    dat_df = pd.read_csv(str(filepath_dat))

    # initialize team and ball ids
    team_ids = {"Home": 1.0, "Away": 2.0}
    ball_id = 4

    # check for additional tIDs
    for ID in dat_df["team_id"].unique():
        if not (ID in team_ids.values() or ID == ball_id):
            warnings.warn("Team ID %.1f did not match any of the standard IDs!" % ID)

    return _create_links_from_dat_df(dat_df, team_ids)


def read_open_statsperform_event_file(
    filepath_events: Union[str, Path],
) -> Tuple[Events, Events, Events, Events]:
    """Parses a StatsPerform Match Event CSV file and extracts the event data.

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
    are parsed as a string in the `qualifier` column of the returned DataFrame and can
    be transformed to a dict of form `{attribute: value}`.
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

            event, team, segment = _read_event_single_line(line)

            # skip the head
            if segment == "current_phase":
                continue

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


def read_open_statsperform_position_file(
    filepath_dat: Union[str, Path],
    links: Dict[str, Dict[int, int]] = None,
) -> Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Pitch]:
    """Parse a StatsPerform CSV file and extract position data and possession codes as
     well as pitch information.

     Openly published StatsPerform position data (e.g. for the Pro Forum '22) is stored
     in a CSV file containing all position data (for both halves) as well as information
     about players, the pitch size, and the ball possession. This function provides a
     high-level access to StatsPerform data by parsing the CSV file.

    Parameters
    ----------
    filepath_dat: str or pathlib.Path
        Full path to the CSV file.
    links: Dict[str, Dict[int, int]], optional
        A link dictionary of the form `links[team][jID] = xID`. Player's are identified
        in StatsPerform files via jID, and this dictionary is used to map them to a
        specific xID in the respective XY objects. Should be supplied if that order
        matters. If None is given (default), the links are automatically extracted from
        the CSV file at the cost of a second pass through the entire file.

    Returns
    -------
    data_objects: Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Pitch]
        XY-, Code-, and Pitch-objects for both teams and both halves. The order is
        (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2,
        possession_ht1, possession_ht2, pitch)
    """
    # parse the CSV file into pd.DataFrame
    dat_df = pd.read_csv(str(filepath_dat))

    # initialize team and ball ids
    team_ids = {"Home": 1.0, "Away": 2.0}
    ball_id = 4

    # check for additional tIDs
    for ID in dat_df["team_id"].unique():
        if not (ID in team_ids.values() or ID == ball_id):
            warnings.warn("Team ID %.1f did not match any of the standard IDs!" % ID)

    # create or check links
    if links is None:
        links = _create_links_from_dat_df(dat_df, team_ids)
    else:
        pass
        # potential check vs jerseys in dat file

    # create periods and pitch
    periods, pitch = _create_metadata_from_dat_df(dat_df)
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

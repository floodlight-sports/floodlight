from typing import Dict, Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np

from floodlight.core.xy import XY
from floodlight.core.pitch import Pitch
from floodlight.core.code import Code


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


def create_links_from_csv(filepath_csv: Union[str, Path]) -> Dict[str, Dict[int, int]]:
    """Parses the entire StatsPerform CSV file for unique jIDs (jerseynumbers) and team
    Ids and creates a dictionary linking jIDs to xIDs in ascending order.

    Parameters
    ----------
    filepath_csv: str or pathlib.Path
        CSV file where the position data in StatsPerform format is saved.

    Returns
    -------
    links: Dict[str, Dict[str, int]]
        A link dictionary of the form `links[team][jID] = xID`.
    """
    # read dat-file into pd.DataFrame
    dat_df = pd.read_csv(str(filepath_csv))

    # initialize team and ball ids
    team_ids = {"Home": 1.0, "Away": 2.0}
    ball_id = 4
    # check
    assert [ID in team_ids or ID == ball_id for ID in dat_df["team_id"].unique()]

    return _create_links_from_csv_df(dat_df, team_ids)


def read_positions(
    filepath_csv: Union[str, Path],
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
    filepath_csv: str or pathlib.Path
        Full path to the CSV file.
    links: Dict[str, Dict[int, int]], optional
        A link dictionary of the form `links[team][jID] = xID`. Player's are identified
        in StatsPerform files via jID, and this dictionary is used to map them to a
        specificxID in the respective XY objects. Should be supplied if that order
        matters. If None is given (default), the links are automatically extracted from
        the .dat file at the cost of a second pass through the entire file.

    Returns
    -------
    data_objects: Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Pitch]
        XY-, Code-, and Pitch-objects for both teams and both halves. The order is
        (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2,
        possession_ht1, possession_ht2, pitch)
    """
    # parse the CSV file into pd.DataFrame
    dat_df = pd.read_csv(str(filepath_csv))

    # initialize team and ball ids
    team_ids = {"Home": 1.0, "Away": 2.0}
    ball_id = 4
    # check
    assert [ID in team_ids or ID == ball_id for ID in dat_df["team_id"].unique()]

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

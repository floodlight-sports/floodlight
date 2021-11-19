from typing import Dict, Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np

from floodlight.core.xy import XY
from floodlight.core.pitch import Pitch
from floodlight.core.code import Code


def _create_metadata_from_dat_df(
    dat_df: pd.DataFrame,
) -> Tuple[int, Dict[int, tuple], Pitch]:
    """Creates meta information from the ball position DataFrame

    Parameters
    ----------
    dat_df: pd.DataFrame
        Containing all data from the positions.csv file as parsed by pd.read_csv()

    Returns
    -------
    framerate: int
        Framerate in frames per second/Hertz
    periods: Dict[int, int]
        Dictionary with start and endframes:
            `periods[segment] = (startframe, endframe)`.
    pitch: floodlight.core.pitch.Pitch
        Playing Pitch object
    """
    # extract ball information used for all computations
    # ball_df = dat_df[dat_df["team_id"] == 4]

    # create pitch
    pi_len = dat_df["pitch_dimension_long_side"].values[0]
    pi_wid = dat_df["pitch_dimension_short_side"].values[0]
    pitch = Pitch.from_template(
        "statsperform",
        length=pi_len,
        width=pi_wid,
        sport="football",
    )

    # compute framerate
    time_values = dat_df["timeelapsed"].unique()
    framerate = 1 / (time_values[1] - time_values[0])

    # create periods for segments, coded as jumps in the frame sequence
    periods = {}
    frame_values = dat_df["frame_count"].unique()

    seg_idx = np.where(np.diff(frame_values, prepend=frame_values[0]) > 1)
    seg_idx = np.insert(seg_idx, 0, 0)
    seg_idx = np.append(seg_idx, len(frame_values))
    for segment in range(len(seg_idx) - 1):
        start = int(frame_values[seg_idx[segment]])
        end = int(frame_values[seg_idx[segment + 1] - 1] + 1)
        periods[segment] = (start, end)

    return framerate, periods, pitch


def _create_links_from_dat_df(
    dat_df: pd.DataFrame, team_ids: Dict[str, float]
) -> Dict[str, Dict[int, int]]:
    """Creates links between player_id and column in the array from parsed dat-file

    Parameters
    ----------
    dat_df: pd.DataFrame
        Containing all data from the positions.csv file as parsed by pd.read_csv()


    Returns
    -------
    links: Dict[str, Dict[int, int]]
        Dictionary with frame axis for the given segment of the form
            `periods[segment] = (startframe, endframe)`.
    """
    links = {}
    for team in team_ids:
        links[team] = {
            jID: xID + 1
            for xID, jID in enumerate(
                dat_df[dat_df["team_id"] == team_ids[team]]["jersey_no"].unique()
            )
        }
    return links


def create_links_from_dat(filepath_dat: Union[str, Path]) -> Dict[str, Dict[int, int]]:
    """Creates links between player_id and column in the array from dat-file.

    Parameters
    ----------
    filepath_dat: str or Path
        Full path to dat-file.


    Returns
    -------
    links: Dict[str, Dict[str, int]]
        Dictionary with frame axis for the given segment of the form
            `periods[segment] = np.arange(startframe, endframe + 1)`.
    """
    # read dat-file into pd.DataFrame
    dat_df = pd.read_csv(str(filepath_dat))

    # create links
    links = {}
    for team_id in dat_df["team_id"].unique():
        if team_id == 4:  # code for the ball
            continue
        elif team_id == 1:
            team = "Home"
        elif team_id == 2:
            team = "Away"
        else:
            team = None
            # possible error or warning

        links[team] = {
            jID: xID + 1
            for xID, jID in enumerate(
                dat_df[dat_df["team_id"] == team_id]["jersey_no"].unique()
            )
        }
    return links


def read_positions(
    filepath: Union[str, Path]
) -> Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Pitch]:
    """Parse StatsPerform dat file for (x,y)-data.

    Read a position data CSV file that is given in the StatsPerform format into a
    Tuple of six floodlight.core.xy.XY Objects comprising a Tuple of six
    floodlight.core.xy.XY objects comprising respectively the position data for the both
    halftimes of the game for the home team, away team, and the ball.

    Parameters
    ----------
    filepath: str or pathlib.Path
        XML File where the Position data in DFL format is saved.

    Returns
    -------
    data_objects: Tuple[XY, XY, XY, XY, XY, XY, Code, Code, Pitch]
        Tuple of XY objects in the order: home team halftime 1, home team halftime 2,
        away team halftime 1, away team halftime 2, ball halftime 1, ball halftime 2,
        possession code for halftime 1, possession code for halftime 2, playing pitch.
    """

    # parse the CSV file into pd.DataFrame
    dat_df = pd.read_csv(str(filepath))

    # initialize team and ball ids
    team_ids = {"Home": 1.0, "Away": 2.0}
    ball_id = 4
    # check
    assert [ID in team_ids or ID == ball_id for ID in dat_df["team_id"].unique()]

    # create links, metadata and pitch
    links = _create_links_from_dat_df(dat_df, team_ids)
    framerate, periods, pitch = _create_metadata_from_dat_df(dat_df)
    segments = list(periods.keys())

    # infer data shapes
    number_of_players = {team: len(links[team]) for team in links}
    number_of_frames = {}
    for segment in segments:
        start = periods[segment][0]
        end = periods[segment][1]
        number_of_frames[segment] = end - start

    # bins
    codes = {}
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
        codes[segment] = Code(
            code=ball_df["possession"].values[appearance],
            name="possession",
            definitions=dict([(team_id, team) for team, team_id in team_ids.items()]),
        )

    # create XY Objects
    home_ht1 = XY(xy=xydata["Home"][0], framerate=framerate)
    home_ht2 = XY(xy=xydata["Home"][1], framerate=framerate)
    away_ht1 = XY(xy=xydata["Away"][0], framerate=framerate)
    away_ht2 = XY(xy=xydata["Away"][1], framerate=framerate)
    ball_ht1 = XY(xy=xydata["Ball"][0], framerate=framerate)
    ball_ht2 = XY(xy=xydata["Ball"][1], framerate=framerate)
    poss_ht1 = codes[0]
    poss_ht2 = codes[1]

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

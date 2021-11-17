from typing import Dict, Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np

from floodlight.core.xy import XY
from floodlight.core.pitch import Pitch


def _create_metadata_from_dat_df(dat_df: pd.DataFrame) -> Tuple[int, Dict, Pitch]:
    """Creates metainformation from the ball position DataFrame

    Parameters
    ----------
    dat_df: pd.DataFrame
        Containing only information about the ball

    Returns
    -------

    """
    # extract ball information
    ball_df = dat_df[dat_df["team_id"] == 4]

    # create pitch
    pi_len = ball_df["pitch_dimension_long_side"].values[0]
    pi_wid = ball_df["pitch_dimension_short_side"].values[0]
    pitch = Pitch.from_template(
        length=pi_len,
        width=pi_wid,
        sport="football",
    )

    # compute framerate
    framerate = 1 / (ball_df["timeelapsed"].values[1] - ball_df["timeelapsed"].values[0])

    # create period time axis for segments
    seg_idx = np.where(
        np.diff(
            ball_df["frame_count"].values, prepend=ball_df["frame_count"].values[0]
        )
        > 1
    )
    seg_idx = np.insert(seg_idx, 0, 0)
    seg_idx = np.append(seg_idx, len(ball_df) - 1)
    periods = {
        segment:
            np.arange(
                ball_df["frame_count"].values[seg_idx[segment]],
                ball_df["frame_count"].values[seg_idx[segment + 1] - 1] + 1,
            )
        for segment in range(len(seg_idx) - 1)
    }
    return framerate, periods, pitch


def create_links_from_dat_df(dat_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """"""
    links = {}
    for team_id in dat_df["team_id"].unique():
        if team_id == 4:  # code for the ball
            continue
        links[team_id] = {
            pID: xID
            for xID, pID in enumerate(
                dat_df[dat_df["team_id"] == team_id]["player_id"].unique()
            )
        }
    return links


def read_positions(filepath: Union[str, Path]) -> Tuple[XY, XY, XY, XY, XY, XY, Pitch]:
    """Read a StatsPerform format position data CSV File

    Read a position data CSV file that is given in the Statsperformm format into a
    List of four floodlight.core.xy.XY Objects comprising the positions for the first
    and the second half of the home and the away team

    Parameters
    ----------
    filepath: str or pathlib.Path
        XML File where the Position data in DFL format is saved

    Returns
    -------
    List of floodlight.core.xy.XY Objects
    """

    # parse the CSV file into pd.DataFrame
    dat_df = pd.read_csv(str(filepath))

    links = create_links_from_dat_df(dat_df)

    framerate, periods, pitch = _create_metadata_from_dat_df(dat_df)

    # infer data shapes
    number_of_frames = {
        segment: len(periods[segment])
        for segment in periods
    }
    number_of_players = {team: len(links[team]) for team in links}

    # bins
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
    for segment in periods:
        for team_id in dat_df["team_id"].unique():
            if team_id == 4:
                team = "Ball"
            elif team_id == list(links.keys())[0]:
                team = "Home"
            elif team_id == list(links.keys())[1]:
                team = "Away"
            else:
                team = None
                pass
                # possible warning or error message

            # write player/ball position data to bins
            team_pos = dat_df[dat_df["team_id"] == team_id]
            for pID in team_pos['player_id'].unique():
                pl_pos = team_pos[team_pos["player_id"] == pID]
                pl_axis = np.array(
                    (pl_pos["frame_count"].values >= periods[segment][0])
                    * (pl_pos["frame_count"].values <= periods[segment][-1])
                )
                # assign positions at the correct frames
                if pl_pos["frame_count"].values[pl_axis].shape[0] > 0:
                    if np.sum(pl_axis) == number_of_frames[segment]:  # player played through
                        if team == "Ball":
                            xydata[team][segment][:, 0] = pl_pos["pos_x"].values[pl_axis]
                            xydata[team][segment][:, 1] = pl_pos["pos_y"].values[pl_axis]
                        else:
                            xydata[team][segment][:, 2 * links[team_id][pID]] = pl_pos["pos_x"].values[pl_axis]
                            xydata[team][segment][:, 2 * links[team_id][pID] + 1] = pl_pos["pos_y"].values[pl_axis]
                    else:  # player did not play through
                        init = np.where(
                            periods[segment] == pl_pos["frame_count"].values[pl_axis][0]
                        )[0][0]
                        end = (
                                np.where(
                                    periods[segment] == pl_pos["frame_count"].values[pl_axis][-1]
                                )[0][0]
                                + 1
                        )
                        xydata[team][segment][init:end, 2 * links[team_id][pID]] = pl_pos["pos_x"].values[
                            pl_axis
                        ]
                        xydata[team][segment][init:end, 2 * links[team_id][pID] + 1] = pl_pos["pos_y"].values[
                            pl_axis
                        ]

    # Create XY Objects
    home_ht1 = XY(xy=xydata["Home"][0], framerate=framerate)
    home_ht2 = XY(xy=xydata["Home"][1], framerate=framerate)
    away_ht1 = XY(xy=xydata["Away"][0], framerate=framerate)
    away_ht2 = XY(xy=xydata["Away"][1], framerate=framerate)
    ball_ht1 = XY(xy=xydata["Ball"][0], framerate=framerate)
    ball_ht2 = XY(xy=xydata["Ball"][1], framerate=framerate)

    data_objects = (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2, pitch)
    return data_objects

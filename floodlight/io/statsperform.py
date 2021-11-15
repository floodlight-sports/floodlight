import pandas as pd
import numpy as np
from pathlib import Path

from floodlight.core.xy import XY
from floodlight.core.pitch import Pitch


def read_positions(filepath: str or Path):
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

    # initialize variables
    all_xy = []

    # parse the CSV file into pd.DataFrame
    all_pos = pd.read_csv(str(filepath))

    # extract ball position
    ball_pos = all_pos[all_pos["team_id"] == 4]

    # create Pitch object
    pi_len = all_pos["pitch_dimension_long_side"].values[0]
    pi_wid = ball_pos["pitch_dimension_short_side"].values[0]
    pitch = Pitch(
        xlim=(-pi_len / 2, pi_len / 2),
        ylim=(-pi_wid / 2, pi_wid / 2),
        unit="m",
        boundaries="flexible!",
        length=pi_len,
        width=pi_wid,
        sport="soccer",
    )

    # compute frame rate
    fps = 1 / (ball_pos["timeelapsed"].values[1] - ball_pos["timeelapsed"].values[0])

    # search for jumps in the frame sequence into find the segments (e.g. halves)
    seg_idx = np.where(
        np.diff(
            ball_pos["frame_count"].values, prepend=ball_pos["frame_count"].values[0]
        )
        > 1
    )
    seg_idx = np.insert(seg_idx, 0, 0)
    seg_idx = np.append(seg_idx, len(ball_pos) - 1)

    # create team specific XY objects for all segments
    for seg in range(len(seg_idx) - 1):

        # define frame axis for current segment
        frame_axis = np.arange(
            ball_pos["frame_count"].values[seg_idx[seg]],
            ball_pos["frame_count"].values[seg_idx[seg + 1] - 1] + 1,
        )

        # define ball x/y positions and fill with values from the DataFrame
        ball_xy = np.zeros((len(frame_axis), 2))
        ball_xy.fill(np.nan)
        ball_xy[:, 0] = ball_pos["pos_x"].values[seg_idx[seg] : seg_idx[seg + 1]]
        ball_xy[:, 1] = ball_pos["pos_y"].values[seg_idx[seg] : seg_idx[seg + 1]]

        # iterate over teams to create the team specific XY objects
        for team_id in all_pos["team_id"].unique():

            # skip for the team id of the ball
            if team_id == 4:
                continue

            # initialize np.arrays with positions and define indices of players
            team_pos = all_pos[all_pos["team_id"] == team_id]
            links = {pID: xID for xID, pID in enumerate(team_pos["player_id"].unique())}
            team_xy = np.zeros((len(frame_axis), 2 + 2 * len(links)))
            team_xy.fill(np.nan)

            # iterate over players in team
            for pID in links:
                pl_pos = team_pos[team_pos["player_id"] == pID]
                pl_axis = np.array(
                    (pl_pos["frame_count"].values >= frame_axis[0])
                    * (pl_pos["frame_count"].values <= frame_axis[-1])
                )

                # if player played in current segment assign positions at correct frames
                if pl_pos["frame_count"].values[pl_axis].shape[0] > 0:
                    if pl_axis.shape[0] == len(frame_axis):  # player played through
                        team_xy[:, 2 * links[pID]] = pl_pos["pos_x"].values[pl_axis]
                        team_xy[:, 2 * links[pID] + 1] = pl_pos["pos_y"].values[pl_axis]
                    else:  # player did not play through
                        init = np.where(
                            frame_axis == pl_pos["frame_count"].values[pl_axis][0]
                        )[0][0]
                        end = (
                            np.where(
                                frame_axis == pl_pos["frame_count"].values[pl_axis][-1]
                            )[0][0]
                            + 1
                        )
                        team_xy[init:end, 2 * links[pID]] = pl_pos["pos_x"].values[
                            pl_axis
                        ]
                        team_xy[init:end, 2 * links[pID] + 1] = pl_pos["pos_y"].values[
                            pl_axis
                        ]

            # create team specific xy object
            all_xy.append(XY(np.hstack((ball_xy, team_xy)), framerate=fps))

    data_objects = (all_xy[0], all_xy[1], all_xy[2], all_xy[3], pitch)
    return data_objects

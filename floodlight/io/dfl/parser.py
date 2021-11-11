from pathlib import Path
import numpy as np
import datetime as dt
from lxml import etree


def read_positions(filepath: str or Path):
    """
    Read a position data XML file into floodlight.core.xy.XY Objects

    Parameters
    ----------
    filepath: str or pathlib.Path
        XML File where the Position data in DFL format is saved

    Returns
    -------
    List of floodlight.core.xy.XY Objects
    """

    # initialize variables
    sub_tol = 30  # tolerance fir a possible substitution; in [s]
    all_xy = []  # List with all XY Objects
    player_xy = {}  # player positions at each observation
    player_t = {}  # player timestamps at each observation
    ball_xy = {}  # ball positions at each observation

    # loop over all FrameSets containing the positions of the
    # respective objects/players
    for _, frame_set in etree.iterparse(filepath, tag="FrameSet"):

        # read TeamId describing the affiliation with a team
        team_id = frame_set.get("TeamId")

        # get current obs_id
        obs_id = frame_set.get("GameSection")

        # initialize player position/timestamp container for observation
        if obs_id not in player_xy:
            player_xy[obs_id] = {}
            player_t[obs_id] = {}
        if team_id not in player_xy[obs_id]:
            player_xy[obs_id][team_id] = []
            player_t[obs_id][team_id] = []

        # create temporary Lists with x/y positions and timestamps
        frame_set_x_pos = []
        frame_set_y_pos = []
        frame_set_times = []

        # loop over all Frames in current observation
        for frame in frame_set.iterfind("Frame"):

            # append x/y position and timestamp at current frame
            frame_set_x_pos.append(frame.get("X"))
            frame_set_y_pos.append(frame.get("Y"))
            frame_set_times.append(dt.datetime.fromisoformat(frame.get("T")))

            # release memory
            frame.clear()

        # assign x/y positions and timestamps
        if team_id == "Ball":
            ball_xy[obs_id] = np.array([frame_set_x_pos, frame_set_y_pos]).T
        else:
            # x/y positions
            player_xy[obs_id][team_id].append(
                np.array([frame_set_x_pos, frame_set_y_pos]).T
            )
            # times
            player_t[obs_id][team_id].append(frame_set_times)

        # release memory
        frame_set.clear()

    # account for substitutions
    # TODO: in_sub->out_sub->in_sub
    for obs_id in player_t:
        for team_id in player_t[obs_id]:
            for out_pl_num in range(len(player_t[obs_id][team_id])):

                # check for players that did not play through
                if len(player_xy[obs_id][team_id][out_pl_num]) < len(ball_xy[obs_id]):

                    # read end time of retired player
                    # TODO: compare with ball end time
                    end_time = player_t[obs_id][team_id][out_pl_num][-1]

                    # compare end time with start time of other players
                    time_deltas = [
                        (
                            end_time - player_t[obs_id][team_id][in_pl_num][0]
                        ).total_seconds()
                        if in_pl_num != out_pl_num
                        else np.nan
                        for in_pl_num in range(len(player_t[obs_id][team_id]))
                    ]

                    # append x/y position of substituting player if time delta
                    # within a given tolerance
                    if np.nanmin(time_deltas) <= sub_tol:
                        player_xy[obs_id][team_id][out_pl_num] = np.vstack(
                            (
                                player_xy[obs_id][team_id][out_pl_num],
                                player_xy[obs_id][team_id][
                                    int(np.nanargmin(time_deltas))
                                ],
                            )
                        )
                        del player_xy[obs_id][team_id][int(np.nanargmin(time_deltas))]

                    # append zeros in case no substituting player is within the
                    # tolerance (red card or injury)
                    else:
                        player_xy[obs_id][team_id][out_pl_num] = np.append(
                            player_xy[obs_id][team_id][out_pl_num],
                            np.zeros(
                                (
                                    len(ball_xy[obs_id])
                                    - len(player_xy[obs_id][team_id][out_pl_num]),
                                    2,
                                )
                            ),
                        )

    # summarize player and ball positions to single np.array
    for obs_id in player_xy:
        for team_id in player_xy[obs_id]:

            # initialize temporary np.array
            n_rows = len(ball_xy[obs_id])
            n_cols = 2 + 2 * len(player_xy[obs_id])
            obs_xy = np.zeros(shape=(n_rows, n_cols))

            # write ball position to array
            obs_xy[:, 0:2] = ball_xy[obs_id]

            # write player positions to array
            for pl_num in range(len(player_xy[obs_id][team_id])):
                obs_xy[:, 2 + 2 * pl_num : 2 + 2 * pl_num + 1] = player_xy[obs_id][
                    team_id
                ][pl_num]

            # append to global object
            all_xy.append(obs_xy)


# read_positions("D:\\dfl_data\\PositionalData\\BL\\DFL-MAT-0002UK.xml")

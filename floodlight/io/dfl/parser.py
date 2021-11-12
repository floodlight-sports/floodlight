from pathlib import Path
import numpy as np
import datetime as dt
from lxml import etree

from floodlight.core.xy import XY


def read_positions(filepath: str or Path):
    """Read a DFL format position data XML

    Returns a List of floodlight.core.xy.XY Objects
    TODO: Extensive description

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
    in_subs = {}  # player numbers of all in subs
    all_xy = []  # List with all XY Objects
    pl_xy = {}  # player positions at each segment
    pl_time = {}  # player timestamps at each segment
    ball_xy = {}  # ball positions at each segment
    ball_time = {}  # ball timestamps at each segment

    # loop over all FrameSets containing the positions of the
    # respective objects/players
    for _, frame_set in etree.iterparse(filepath, tag="FrameSet"):

        # read TeamId describing the affiliation with a team
        team_id = frame_set.get("TeamId")

        # get GameSection describing the current segment
        sgm_id = frame_set.get("GameSection")

        # create temporary Lists with x/y positions and timestamps
        frame_set_x_pos = []
        frame_set_y_pos = []
        frame_set_times = []

        # append x/y positions and timestamps for all Frames in current segment
        for frame in frame_set.iterfind("Frame"):

            # append x/y position and timestamp at current frame
            frame_set_x_pos.append(frame.get("X"))
            frame_set_y_pos.append(frame.get("Y"))
            frame_set_times.append(dt.datetime.fromisoformat(frame.get("T")))

            # release memory
            frame.clear()

        # assign x/y positions and timestamps
        if team_id == "Ball":
            ball_xy[sgm_id] = np.array([frame_set_x_pos, frame_set_y_pos]).T
            ball_time[sgm_id] = frame_set_times

        else:  # player

            # initialize player position and timestamp containers for segment
            if sgm_id not in pl_xy:
                pl_xy[sgm_id] = {}
                pl_time[sgm_id] = {}
            if team_id not in pl_xy[sgm_id]:
                pl_xy[sgm_id][team_id] = []
                pl_time[sgm_id][team_id] = []

            # x/y positions
            pl_xy[sgm_id][team_id].append(
                np.array([frame_set_x_pos, frame_set_y_pos]).T
            )
            # timestamps
            pl_time[sgm_id][team_id].append(frame_set_times)

        # release memory
        frame_set.clear()

    # account for substitutions
    for sgm_id in pl_time:
        for team_id in pl_time[sgm_id]:

            # initialize in-player number containers
            if sgm_id not in in_subs:
                in_subs[sgm_id] = {}
            if team_id not in in_subs[sgm_id]:
                in_subs[sgm_id][team_id] = []

            # perform substitution loop for every player
            for out_pl_num in range(len(pl_time[sgm_id][team_id])):

                # check if player did start the segment but did not play through
                if (
                    len(pl_xy[sgm_id][team_id][out_pl_num]) < len(ball_xy[sgm_id])
                    and pl_time[sgm_id][team_id][out_pl_num][0] <= ball_time[sgm_id][0]
                ):

                    # read end time of out-player
                    end_time = pl_time[sgm_id][team_id][out_pl_num][-1]

                    # substitution loop to search for possible in-player replacements
                    for _ in range(len(pl_time[sgm_id][team_id])):

                        # check if the out-player retired before the end of segment
                        if end_time >= ball_time[sgm_id][-1]:
                            break

                        # compare end time of out-player to to start times of in-players
                        time_deltas = [
                            (
                                np.abs(
                                    end_time - pl_time[sgm_id][team_id][in_pl_num][0]
                                ).total_seconds()
                            )
                            if in_pl_num != out_pl_num
                            else np.nan
                            for in_pl_num in range(len(pl_time[sgm_id][team_id]))
                        ]

                        # perform sub if the minimum time delta is within tolerance
                        if np.nanmin(time_deltas) <= sub_tol:

                            # get the number of the chosen in-player
                            in_pl_num = int(np.nanargmin(time_deltas))

                            # append x/y position of in-player to out-player
                            pl_xy[sgm_id][team_id][out_pl_num] = np.vstack(
                                (
                                    pl_xy[sgm_id][team_id][out_pl_num],
                                    pl_xy[sgm_id][team_id][in_pl_num],
                                )
                            )

                            # store number of in-player to remove from containers later
                            in_subs[sgm_id][team_id].append(in_pl_num)

                            # assign new end time as end time of in-player
                            end_time = pl_time[sgm_id][team_id][in_pl_num][-1]

                        # if no in-player is found (red card, injury) append zeros
                        else:
                            pl_xy[sgm_id][team_id][out_pl_num] = np.append(
                                pl_xy[sgm_id][team_id][out_pl_num],
                                np.zeros(
                                    (
                                        len(ball_xy[sgm_id])
                                        - len(pl_xy[sgm_id][team_id][out_pl_num]),
                                        2,
                                    )
                                ),
                                axis=0,
                            )

                            # assign new end time as ball end time
                            end_time = ball_time[sgm_id][-1]

                        # TODO
                        # if pl_xy[sgm_id][team_id][out_pl_num].shape[0] == 70429:
                        #    debug = 0

            # remove in-player positions from container if substitution was performed
            for in_pl_num in in_subs[sgm_id][team_id]:
                pl_xy[sgm_id][team_id][in_pl_num] = None
            pl_xy[sgm_id][team_id] = [
                xy for xy in pl_xy[sgm_id][team_id] if xy is not None
            ]

    # summarize player and ball positions to single np.array
    for sgm_id in pl_xy:
        for team_id in pl_xy[sgm_id]:

            # initialize temporary np.array
            n_rows = len(ball_xy[sgm_id])
            n_cols = 2 + 2 * len(pl_xy[sgm_id][team_id])
            obs_xy = np.zeros(shape=(n_rows, n_cols))

            # write ball position to array
            obs_xy[:, 0:2] = ball_xy[sgm_id]

            # write player positions to array
            for pl_num in range(len(pl_xy[sgm_id][team_id])):
                obs_xy[:, 2 + 2 * pl_num : 2 + 2 * pl_num + 2] = pl_xy[sgm_id][team_id][
                    pl_num
                ]

            # compute frame rate [in Hz]
            fps = int(1 / (ball_time[sgm_id][1] - ball_time[sgm_id][0]).total_seconds())

            # append to global object
            all_xy.append(XY(obs_xy, framerate=fps))

    return all_xy

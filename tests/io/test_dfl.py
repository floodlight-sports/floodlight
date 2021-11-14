import os
import numpy as np
from pathlib import Path

from floodlight.io.dfl.parser import read_positions


def test_read_positions_for_all_files(data_loc: str or Path):

    # iterate over files in data location
    for file in os.listdir(data_loc):

        # perform test battery
        print("Testing File @ " + file)

        # read positions to XY
        file = os.path.join(data_loc, file)
        match_xy = read_positions(file)

        # check that each segment (i.e. half) exists twice (for both home and away team)
        for obs_xy in match_xy:
            assert (
                np.sum([obs_xy.xy.shape == obs_2_xy.xy.shape for obs_2_xy in match_xy])
                >= 2
            )

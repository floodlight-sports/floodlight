import os
import numpy as np
from pathlib import Path

from floodlight.io.dfl.parser import read_positions


def test_read_positions_for_all_files(data_loc: str or Path):

    # iterate over files in data location
    # DEBUG
    for file in os.listdir(data_loc)[5:]:

        # perform test battery
        print("Testing File @ " + file)

        # read positions to XY
        file = os.path.join(data_loc, file)
        match_xy = read_positions(file)

        # check that each segment exists for two teams
        for obs_xy in match_xy:
            assert (
                np.sum([obs_xy.xy.shape == obs_2_xy.xy.shape for obs_2_xy in match_xy])
                >= 2
            )


test_read_positions_for_all_files("D:\\dfl_data\\PositionalData\\BL")

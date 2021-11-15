import os
from pathlib import Path

from floodlight.io.dfl.parser import read_positions
from floodlight.core.xy import XY


def test_read_positions_for_all_files(data_loc: str or Path):

    # iterate over files in data location
    for file in os.listdir(data_loc):

        # perform test battery
        print("Testing File @ " + file)

        # read positions to XY
        file = os.path.join(data_loc, file)
        match_xy = read_positions(file)

        # assert that match_xy is a List of XY objects
        assert isinstance(match_xy, list)
        for xy in match_xy:
            assert isinstance(xy, XY)

from pathlib import Path
from typing import Union

import os

from floodlight.core.code import Code
from floodlight.core.pitch import Pitch
from floodlight.io.dfl import read_dfl_files
from floodlight.core.xy import XY


def test_read_positions_for_all_files(
    loc_dat: Union[str, Path] = None, loc_metadata: Union[str, Path] = None
):

    if loc_dat is not None and loc_metadata is not None:
        # iterate over files in data location
        for file in os.listdir(loc_dat):

            # perform test battery
            print("Testing File @ " + file)

            # read positions to XY
            match_xy = read_dfl_files(
                os.path.join(loc_dat, file), os.path.join(loc_metadata, file)
            )

            # check that each segment exists for two teams
            for i, obj in enumerate(match_xy):
                if i < 6:
                    assert isinstance(obj, XY)
                elif i < 10:
                    assert isinstance(obj, Code)
                else:
                    assert isinstance(obj, Pitch)

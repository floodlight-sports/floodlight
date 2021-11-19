import os
from typing import Union
from pathlib import Path

from floodlight.io.statsperform import read_positions
from floodlight.core.code import Code
from floodlight.core.pitch import Pitch
from floodlight.core.xy import XY


def test_read_positions_for_all_files(data_loc: Union[str, Path] = None):

    if data_loc is not None:
        for file in os.listdir(str(data_loc)):
            data = read_positions(os.path.join(data_loc, file))

            # perform test battery
            assert isinstance(data, tuple)

            for i, obj in enumerate(data):
                if i < 6:
                    assert isinstance(obj, XY)
                elif i < 8:
                    assert isinstance(obj, Code)
                else:
                    assert isinstance(obj, Pitch)

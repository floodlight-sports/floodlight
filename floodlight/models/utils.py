import numpy as np

from floodlight import XY


def _exclude_x_ids(xy: XY, xIDs_to_exclude: list) -> np.ndarray:
    """Returns an array of booleans to denote which xIDs should be excluded.

    Parameters
    ----------
    xy: XY
        Player spatiotemporal data from which xIDs should be excluded.
    xIDs_to_exclude: list
        A list of xIDs to be excluded.

    Returns
    -------
    include: np.ndarray
        A numpy array with Boolean values, where the xIDs to be excluded are marked as
        False.
    """

    # boolean for column inclusion, initialize to True for all columns
    include = np.full((xy.N * 2), True)

    # exclude columns according to xIDs_to_exclude
    if xIDs_to_exclude:
        for xID in xIDs_to_exclude:
            if xID not in range(0, xy.N):
                raise ValueError(
                    f"Expected entries of exclude_xIDs to be in range 0 to {xy.N}, "
                    f"got {xID}."
                )
            exclude_start = xID * 2
            exclude_end = exclude_start + 2
            include[exclude_start:exclude_end] = False

    return include

import warnings

import numpy as np

from floodlight import XY


def subtract_centroid(xy: XY, exclude_xIDs: list = None) -> XY:
    """Subtracts the per-frame team centroid from all player positions.

    For each frame, the centroid is computed as the arithmetic mean of all
    included player positions (using nanmean). The centroid is then subtracted
    from every player's coordinates, yielding positions relative to the team
    center.

    Parameters
    ----------
    xy: XY
        Spatiotemporal data with shape (T, 2*N).
    exclude_xIDs: list, optional
        A list of player indices (xIDs) to exclude from the centroid
        computation. Excluded players' coordinates are still translated by the
        centroid of the remaining players. This can be useful to exclude
        goalkeepers from computation.

    Returns
    -------
    xy_centered: XY
        New XY object with centroid-subtracted coordinates. Same shape,
        framerate, and direction as input.

    Notes
    -----
    This transform is commonly used as a preprocessing step for formation
    analysis, where the relative spacing of players is more informative than
    their absolute positions on the pitch.
    """
    if exclude_xIDs is None:
        exclude_xIDs = []

    # build inclusion mask for centroid computation
    include = np.full(xy.N * 2, True)
    for xID in exclude_xIDs:
        if xID not in range(0, xy.N):
            raise ValueError(
                f"Expected entries of exclude_xIDs to be in range 0 to "
                f"{xy.N - 1}, got {xID}."
            )
        include[xID * 2 : xID * 2 + 2] = False

    T = len(xy)
    xy_data = xy.xy.copy().astype(float)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # compute per-frame centroid from included players only
        centroids = np.nanmean(xy_data[:, include].reshape(T, -1, 2), axis=1)  # (T, 2)

    # subtract centroid from all players (including excluded ones)
    xy_centered = xy_data.reshape(T, -1, 2) - centroids[:, np.newaxis, :]
    xy_centered = xy_centered.reshape(T, -1)

    return XY(xy=xy_centered, framerate=xy.framerate, direction=xy.direction)


def min_max_normalize(positions: np.ndarray) -> np.ndarray:
    """Min-max normalizes an (N, 2) formation array to [0, 1] per axis.

    Each axis (x, y) is independently scaled so that its minimum maps to 0
    and its maximum maps to 1. This makes formations with different
    compactness comparable.

    Parameters
    ----------
    positions: np.ndarray
        Formation positions of shape (N, 2).

    Returns
    -------
    normalized: np.ndarray
        Normalized positions of shape (N, 2) with values in [0, 1].

    Notes
    -----
    If all positions share the same value along one axis (zero range), that
    axis is left at 0.0 rather than producing a division-by-zero error.
    """
    min_vals = np.nanmin(positions, axis=0)
    max_vals = np.nanmax(positions, axis=0)
    ranges = max_vals - min_vals

    # handle zero range (all players on same line along one axis)
    ranges[ranges == 0] = 1.0

    return (positions - min_vals) / ranges

import warnings

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from floodlight import XY


def _assign_roles_once(
    xy_data: np.ndarray, reference: np.ndarray, T: int, N: int
) -> np.ndarray:
    """Single pass of Hungarian role assignment across all frames.

    Parameters
    ----------
    xy_data: np.ndarray
        Raw data array of shape (T, 2*N).
    reference: np.ndarray
        Reference positions of shape (N, 2).
    T: int
        Number of frames.
    N: int
        Number of players.

    Returns
    -------
    assigned_data: np.ndarray
        Role-assigned data array of shape (T, 2*N).
    """
    assigned_data = np.full_like(xy_data, np.nan, dtype=float)

    for t in range(T):
        frame = xy_data[t].reshape(-1, 2).astype(float)  # (N, 2)

        # identify valid (non-NaN) players
        valid_mask = ~np.isnan(frame[:, 0])
        if not np.any(valid_mask):
            continue

        valid_positions = frame[valid_mask]  # (n_valid, 2)

        # filter reference to non-NaN role slots
        ref_mask = ~np.isnan(reference[:, 0])
        valid_ref = reference[ref_mask]  # (n_ref, 2)
        ref_indices = np.where(ref_mask)[0]

        # rectangular cost matrix: valid players vs valid role slots
        cost_matrix = cdist(valid_positions, valid_ref)  # (n_valid, n_ref)
        row, col = linear_sum_assignment(cost_matrix)

        # map valid players into their assigned role slots
        for r, c in zip(row, col):
            role_idx = ref_indices[c]
            assigned_data[t, role_idx * 2 : role_idx * 2 + 2] = valid_positions[r]

    return assigned_data


def assign_roles(xy: XY, reference: np.ndarray = None, n_iter: int = 1) -> XY:
    """Assigns consistent roles to players across frames using the Hungarian
    algorithm. [1]_

    For each frame, player positions are matched to reference positions by
    minimizing the total Euclidean distance. The resulting XY object has columns
    reordered so that column *i* consistently represents role *i* across all
    frames.

    Parameters
    ----------
    xy: XY
        Spatiotemporal data with shape (T, 2*N).
    reference: np.ndarray, optional
        Reference positions of shape (N, 2) used as the assignment target. If
        None (default), the mean position of each player column across all frames is
        used.
    n_iter: int, optional
        Number of assignment iterations. After each iteration the reference is
        recomputed as the column-wise mean of the assigned data. More iterations
        can improve convergence for longer sequences. Defaults to 1, which is
        sufficient for short sequences as proposed in [1]_.

    Returns
    -------
    xy_assigned: XY
        New XY object with columns reordered per frame so that column *i*
        consistently represents role *i*. Same shape, framerate, and direction
        as input.

    Notes
    -----
    Players with NaN positions in a given frame are excluded from the
    assignment. Their corresponding role slots in the output are filled with
    NaN. Reference positions that are NaN are likewise excluded from the
    cost matrix, so the assignment operates only on valid data.

    References
    ----------
    .. [1] `Bialkowski, A., Lucey, P., Carr, P., Yue, Y., Sridharan, S., &
        Matthews, I. (2014). Identifying team style in soccer using formations
        learned from spatiotemporal tracking data. In IEEE International
        Conference on Data Mining Workshop (pp. 9-14).
        <https://doi.org/10.1109/ICDMW.2014.167>`_
    """
    T = len(xy)
    N = xy.N

    # compute reference from mean positions if not provided
    if reference is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            reference = np.nanmean(xy.xy, axis=0).reshape(-1, 2)  # (N, 2)

    if reference.shape != (N, 2):
        raise ValueError(
            f"Expected reference of shape ({N}, 2), got {reference.shape}."
        )

    assigned_data = _assign_roles_once(xy.xy, reference, T, N)

    # iterative refinement: recompute reference from assigned data
    for _ in range(n_iter - 1):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            reference = np.nanmean(assigned_data.reshape(T, -1, 2), axis=0)  # (N, 2)
        assigned_data = _assign_roles_once(assigned_data, reference, T, N)

    xy_assigned = XY(xy=assigned_data, framerate=xy.framerate, direction=xy.direction)

    return xy_assigned

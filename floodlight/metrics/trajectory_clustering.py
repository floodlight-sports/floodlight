import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from floodlight import XY
from floodlight.transforms.spatial import subtract_centroid, min_max_normalize
from floodlight.transforms.permutation import assign_roles


def _fsim_score(query: np.ndarray, template: np.ndarray, delta: float) -> float:
    """Compute FSIM between two normalized (N, 2) formation arrays."""
    # squared Euclidean distance matrix
    dist_sq = cdist(query, template, metric="sqeuclidean")
    # similarity matrix clamped to [0, 1]
    similarity = np.maximum(1.0 - dist_sq / delta, 0.0)
    # optimal assignment maximizing total similarity
    row, col = linear_sum_assignment(-similarity)

    return float(similarity[row, col].mean())


def formation_similarity(
    xy: XY,
    template: np.ndarray,
    exclude_xIDs: list = None,
    role_assignment: bool = True,
    n_iter: int = 1,
    delta: float = 1 / 3,
) -> float:
    """Computes formation similarity (FSIM) between observed player positions
    and a formation template via template matching. [1]_

    The algorithm classifies a team's formation by comparing role-resolved
    average positions against an idealized formation template.

    Parameters
    ----------
    xy: XY
        Spatiotemporal tracking data for one team, shape (T, 2*N).
    template: np.ndarray
        Template position array of shape (N_outfield, 2), where N_outfield is
        the number of players after excluding ``exclude_xIDs``.
    exclude_xIDs: list, optional
        A list of player indices (xIDs) to exclude from the analysis. Excluded
        players are removed from centroid computation and from the formation
        comparison. This is typically used to exclude goalkeepers.
    role_assignment: bool, optional
        If True (default), role assignment via the Hungarian algorithm is
        applied before averaging. Set to False to skip this step, e.g. when
        the input data has already been role-assigned externally.
    n_iter: int, optional
        Number of role assignment iterations. Only used when
        ``role_assignment`` is True. Defaults to 1.
    delta: float, optional
        Similarity decay parameter controlling how quickly similarity drops
        with distance between role positions. Motivated by a coarse 3x3
        pitch partitioning where roles one zone apart should have near-zero
        similarity. Defaults to 1/3 according to [1]_.

    Returns
    -------
    fsim: float
        Formation similarity score in [0, 1].

    Notes
    -----
    The pipeline proceeds as follows:

    1. Per-frame centroid subtraction to obtain center-relative positions.
    2. Role assignment (optional, default=True) [2]_ via Hungarian algorithm using mean
       positions as reference (one iteration per default, as proposed in [1]_).
    3. Averaging role-resolved positions across frames to obtain a single
       formation snapshot of shape (N, 2).
    4. Independent min-max normalization of query and template to [0, 1]
       per axis, ensuring scale and compactness invariance.
    5. Computation of an N x N similarity matrix using
       :math:`m_{i,j} = \\max(1 - \\|\\tilde{r}_i - \\tilde{r}_j\\|^2 / \\delta,
       \\; 0)`, optimal one-to-one assignment via the Hungarian algorithm,
       and averaging matched similarities to obtain the FSIM score.

    References
    ----------
    .. [1] `Müller-Budack, E., Theiner, J., Rigoll, G., & Ewerth, R. (2019).
        Does 4-4-2 exist? An analytics approach to understand and classify
        football team formations in single match situations. In Proceedings of
        the 2nd International Workshop on Multimedia Content Analysis in Sports
        (pp. 25-33). <https://doi.org/10.1145/3347318.3355527>`_
    .. [2] `Bialkowski, A., Lucey, P., Carr, P., Yue, Y., Sridharan, S., &
        Matthews, I. (2014). Identifying team style in soccer using formations
        learned from spatiotemporal tracking data. In IEEE International
        Conference on Data Mining Workshop (pp. 9-14).
        <https://doi.org/10.1109/ICDMW.2014.167>`_
    """
    if exclude_xIDs is None:
        exclude_xIDs = []

    # step 1: subtract per-frame centroid (excluding specified players)
    xy_centered = subtract_centroid(xy, exclude_xIDs=exclude_xIDs)

    # remove excluded players from the data before role assignment
    if exclude_xIDs:
        keep = np.full(xy.N * 2, True)
        for xID in exclude_xIDs:
            keep[xID * 2 : xID * 2 + 2] = False
        xy_centered = XY(
            xy=xy_centered.xy[:, keep],
            framerate=xy_centered.framerate,
            direction=xy_centered.direction,
        )

    N = xy_centered.N

    # validate template
    template = np.asarray(template, dtype=float)
    if template.shape != (N, 2):
        raise ValueError(f"Template has shape {template.shape}, expected ({N}, 2).")

    # step 2: role assignment via Hungarian algorithm
    if role_assignment:
        xy_centered = assign_roles(xy_centered, n_iter=n_iter)

    # step 3: average role-resolved positions across frames
    T = len(xy_centered)
    mean_formation = np.nanmean(xy_centered.xy.reshape(T, -1, 2), axis=0)  # (N, 2)

    # step 4: normalize query and template
    query = min_max_normalize(mean_formation)
    norm_template = min_max_normalize(template)

    # step 5: compute FSIM
    fsim = _fsim_score(query, norm_template, delta)

    return fsim

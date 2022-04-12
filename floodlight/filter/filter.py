from typing import List, Dict

import numpy as np

from floodlight import XY


def get_sequences_without_nans(data_column: np.ndarray) -> np.ndarray:
    """Gets indices of sequences without NaNs from a sequence of data.

    Parameters
    ----------
    data_column: np.ndarray
        One-dimensional array of data with alternating sequences of Numbers and NaNs

    Returns
    -------
    sequences: np.ndarray
        Two-dimensional array with nx2 entrys of n sequences without NaNs and n[0] being
        the first frame of the sequences and n[1] being the last frame (excluded) of the
        sequence.
    """

    # check if even or odd sequences contain NaNs
    if np.isnan(data_column[0]):
        first_sequence = 1
    else:
        first_sequence = 0

    # indices where nans and numbers are next to each other
    change_points = np.where(
        np.diff(np.isnan(data_column), prepend=np.nan, append=np.nan)
    )[0]
    # sequences without nans
    sequences = np.array(
        [
            (change_points[i], change_points[i + 1])
            for i in range(len(change_points) - 1)
        ]
    )[first_sequence::2]

    return sequences


def get_sequences_from_xy(
    xy: XY, team_links: Dict[str, int]
) -> List[Dict[str, np.ndarray]]:
    """Gets sequences without NaNs from a xy-Objects.

    Parameters
    ----------
    xy: XY
        List of XY-objects from any floodlight parser.
    team_links: Dict[str, int]
        Link-dictionary of the form ``links[identifier-ID] = xID``.
    Returns
    -------
    sequence_list: Dict[str, np.ndarray]
        List with dictionaries for each player with key identifier-ID and value nd.array
        of dimensions n x 2. First entry corresponds to the first frame in a sequence
        without NaNs, second entry corresponds to the last frame (excluded) in a
        sequence without NaNs.
    """

    nan_dict = {}
    for player in team_links:
        # players x-index in xy-Object
        pos_idx = team_links[player] * 2

        data_col = xy[:, pos_idx]
        sequences = get_sequences_without_nans(data_col)
        nan_dict.update({player: sequences})

    return nan_dict

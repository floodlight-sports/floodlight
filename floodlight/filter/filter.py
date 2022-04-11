from typing import List, Dict

import numpy as np

from floodlight import XY


def get_sequences_without_nans(
    xy_list: List[XY], links: Dict[str, Dict[str, int]]
) -> List[Dict[str, np.ndarray]]:
    """Gets sequences without NaNs from a list of xy-Objects.

    Parameters
    ----------
    xy_list: List[XY]
        List of XY-objects from any floodlight parser.
    links: Dict[str, Dict[str, int]]
        Link-dictionary of the form ``links[group][identifier-ID] = xID``.
    Returns
    -------
    sequence_list: List[Dict[str, np.ndarray]]
        List with dictionaries for each player with key identifier-ID and value nd.array
        of dimensions n x 2. First entry corresponds to the first frame in a sequence
        without NaNs, second entry corresponds to the last frame (excluded) in a
        sequence without NaNs.
    """

    sequence_list = []
    for i, team in enumerate(links):
        nan_dict = {}
        for player in links[team]:
            # players x-index in xy-Object
            pos_idx = links[team][player] * 2

            # check if even or odd sequences contain NaNs
            if np.isnan(xy_list[i][0, pos_idx]):
                first_sequence = 1
            else:
                first_sequence = 0

            # indices where nans and numbers are next to each other
            change_points = np.where(
                np.diff(np.isnan(xy_list[i][:, pos_idx]), prepend=np.nan, append=np.nan)
            )[0]
            # sequences without nans
            sequences = np.array(
                [
                    (change_points[i], change_points[i + 1])
                    for i in range(len(change_points) - 1)
                ]
            )[first_sequence::2]

            nan_dict.update({player: sequences})

        sequence_list.append(nan_dict)
    return sequence_list

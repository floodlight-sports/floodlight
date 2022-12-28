from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np


@dataclass
class Code:
    """Fragment of continuous signal encoding one game state. Core class of floodlight.

    Parameters
    ----------
    code: np.ndarray
        One-dimensional array with codes describing a sequence of play.
    name: str
        Name of encoded game state (e.g. 'possession').
    definitions: dict, optional
        Dictionary of the form {token: definition} where each code category is defined
        or explained.
    framerate: int, optional
        Temporal resolution of data in frames per second/Hertz.

    Attributes
    ----------
    token: list
        A list of all tokens used in game code, in ascending order.

    """

    code: np.ndarray
    name: str
    definitions: dict = None
    framerate: int = None

    def __str__(self):
        return f"Floodlight Code object encoding '{self.name}'"

    def __len__(self):
        return len(self.code)

    def __getitem__(self, key):
        return self.code[key]

    def __setitem__(self, key, value):
        self.code[key] = value

    def __eq__(self, other):
        return self.code == other

    def __ne__(self, other):
        return self.code != other

    def __gt__(self, other):
        return self.code > other

    def __lt__(self, other):
        return self.code < other

    def __ge__(self, other):
        return self.code >= other

    def __le__(self, other):
        return self.code <= other

    @property
    def token(self) -> list:
        """A list of all tokens used in game code, in ascending order."""
        token = list(np.unique(self.code))
        token.sort()

        return token

    def find_sequences(
        self, return_type: str = "dict"
    ) -> Union[Dict[Any, tuple], List[tuple]]:
        """Finds all sequences of consecutive appearances for each token and returns
        their start and end indices.

        Parameters
        ----------
        return_type: {'dict', 'list'}, default='dict'
            Specifies type of the returned sequences object.


        Returns
        -------
        sequences: Union[Dict[Any, tuple], List[tuple]]
            If ``return_type`` is 'dict', returns a dictionary of the form
            ``{token: [(sequence_start_idx, sequence_end_idx)]}``.
            If ``return_type`` is 'list', returns a list of the form
            ``[(sequence_start_idx, sequence_end_idx, token)]`` ordered by the
            respective sequence start indices.


        Examples
        --------
        >>> import numpy as np
        >>> from floodlight import Code

        >>> code = Code(code=np.array([1, 1, 2, 1, 1, 3, 1, 1]), name="intensity")
        >>> code.find_sequences()
        {1: [(0, 2), (3, 5), (6, 8)], 2: [(2, 3)], 3: [(5, 6)]}

        >>> code = Code(code=np.array(['A', 'A', 'H', 'H', 'H', 'H', 'A', 'A', 'A']),
        ...                           name="possession")
        >>> code.find_sequences(return_type="list")
        [(0, 2, 'A'), (2, 6, 'H'), (6, 9, 'A')]
        """
        if return_type not in ["dict", "list"]:
            raise ValueError(
                f"Expected return_type to be one of ['list', 'dict'], got {return_type}"
            )

        # get all token for token-wise query
        all_token = self.token
        # get change points for each token
        # NOTE: as np.diff can't be called on non-numerical token, a token-wise approach
        # is necessary, where self == token produces a boolean array
        change_points = {
            token: np.where(np.diff(self == token, prepend=np.nan, append=np.nan))[0]
            for token in all_token
        }
        # determine token-wise ranges of respective sequence lengths
        ranges = {token: range(len(change_points[token]) - 1) for token in all_token}
        # create token-wise dictionary of sequence start- and end-points
        sequences = {
            token: [
                (
                    change_points[token][i],
                    change_points[token][i + 1],
                )
                for i in ranges[token]
                if self[change_points[token][i]] == token
            ]
            for token in all_token
        }

        # combine all sequences in list and sort if desired
        if return_type == "list":
            sequences_list = []
            for token in all_token:
                sequences_list.extend(
                    [(sequence[0], sequence[1], token) for sequence in sequences[token]]
                )
            sequences = sorted(sequences_list)

        return sequences

    def slice(
        self, startframe: int = None, endframe: int = None, inplace: bool = False
    ):
        """Return copy of object with sliced code. Mimics numpy's array slicing.

        Parameters
        ----------
        startframe : int, optional
            Start of slice. Defaults to beginning of segment.
        endframe : int, optional
            End of slice (endframe is excluded). Defaults to end of segment.
        inplace: bool, optional
            If set to ``False`` (default), a new object is returned, otherwise the
            operation is performed in place on the called object.

        Returns
        -------
        code_sliced: Union[Code, None]
        """
        sliced_data = self.code[startframe:endframe].copy()
        code_sliced = None

        if inplace:
            self.code = sliced_data
        else:
            code_sliced = Code(
                code=sliced_data,
                name=deepcopy(self.name),
                definitions=deepcopy(self.definitions),
                framerate=deepcopy(self.framerate),
            )

        return code_sliced

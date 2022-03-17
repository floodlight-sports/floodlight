from copy import deepcopy
from dataclasses import dataclass

import numpy as np


@dataclass
class Code:
    """Fragment of continuous signal encoding one game state. Core class of floodlight.

    Attributes
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
        token = list(np.unique(self.code))
        token.sort()

        return token

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

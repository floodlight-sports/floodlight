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

    @property
    def token(self) -> list:
        token = list(np.unique(self.code))
        token.sort()

        return token

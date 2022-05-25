from copy import deepcopy
from dataclasses import dataclass

import numpy as np


@dataclass
class BaseProperty:

    property: np.ndarray
    name: str
    framerate: int = None

    def __str__(self):
        return f"Floodlight {self.__class__.__name__} object encoding '{self.name}'"

    def __len__(self):
        return len(self.property)

    def __getitem__(self, key):
        return self.property[key]

    def __setitem__(self, key, value):
        self.property[key] = value

    @classmethod
    def _slice_new(cls, sliced_property, name, framerate):
        sliced_copy = cls(
            property=sliced_property,
            name=name,
            framerate=framerate,
        )

        return sliced_copy

    def slice(
        self, startframe: int = None, endframe: int = None, inplace: bool = False
    ):
        """Return copy of object with sliced property. Mimics numpy's array slicing.

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
        property_sliced: Union[cls, None]
        """
        sliced_data = self.property[startframe:endframe].copy()
        sliced_object = None

        if inplace:
            self.property = sliced_data
        else:
            sliced_object = self._slice_new(
                sliced_property=sliced_data,
                name=deepcopy(self.name),
                framerate=deepcopy(self.framerate),
            )

        return sliced_object


@dataclass
class TeamProperty(BaseProperty):
    """Fragment of one continuous team property. Core class of floodlight.

    Parameters
    ----------
    property: np.ndarray
        A 1-dimensional array of properties of shape (T), where T is the number of
        total frames.
    name: str
        Name of the property (e.g. 'stretch_index').
    framerate: int, optional
        Temporal resolution of data in frames per second/Hertz.
    """


@dataclass
class PlayerProperty(BaseProperty):
    """Fragment of one continuous property per player. Core class of floodlight.

    Parameters
    ----------
    property: np.ndarray
        A 2-dimensional array of properties of shape (T, N), where T is the number of
        total frames and N is the number of players.
    name: str
        Name of the property (e.g. 'speed').
    framerate: int, optional
        Temporal resolution of data in frames per second/Hertz.
    """


@dataclass
class DyadicProperty(BaseProperty):
    """Fragment of one continuous property per player dyad. Core class of floodlight.

    Parameters
    ----------
    property: np.ndarray
        A 3-dimensional array of properties of shape (T, N_1, N_2), where T is the
        number of total frames and {N_1, N_2} are the number of players between dyads
        are formed. For example, the item at (1, 2, 3) encodes the relation from player
        with xID=2 to player with xID=3 at frame 1. Note that players could be in the
        same team (intra-team relations, in this case N_1 = N_2) or opposing teams
        (inter-team relations).
    name: str
        Name of the property (e.g. 'distance').
    framerate: int, optional
        Temporal resolution of data in frames per second/Hertz.
    """

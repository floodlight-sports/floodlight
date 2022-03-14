from copy import deepcopy
from dataclasses import dataclass

import numpy as np


@dataclass
class PlayerProperty:
    """Fragment of one continuous property per player. Core class of floodlight.

    Attributes
    ----------
    property: np.ndarray
        A 2-dimensional array of properties of shape (T, N), where T is the number of
        time steps and N is the number of players.
    name: str
        Name of the property (e.g. 'speed').
    framerate: int, optional
        Temporal resolution of data in frames per second/Hertz.
    """

    property: np.ndarray
    name: str
    framerate: int = None

    def __str__(self):
        return f"Floodlight PlayerProperty object encoding '{self.name}'"

    def __len__(self):
        return len(self.property)

    def __getitem__(self, key):
        return self.property[key]

    def __setitem__(self, key, value):
        self.property[key] = value

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
        property_sliced: Union[PlayerProperty, None]
        """
        sliced_data = self.property[startframe:endframe].copy()
        property_sliced = None

        if inplace:
            self.property = sliced_data
        else:
            property_sliced = PlayerProperty(
                property=sliced_data,
                name=deepcopy(self.name),
                framerate=deepcopy(self.framerate),
            )

        return property_sliced

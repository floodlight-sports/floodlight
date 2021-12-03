from dataclasses import dataclass

import pandas as pd

from floodlight.core.definitions import essential_events_columns, protected_columns


@dataclass
class Events:
    """Event data fragment. Core class of floodlight.

    Event data is stored in `pandas` ``DataFrame``, where each row stores one event
    with its different properties organized in columns. You may put whatever
    information you like in these columns. Yet, the columns `"eID"` and `"gameclock"`
    are mandatory to identify and time-locate events. Some special column names are
    reserved for properties that follow conventions. These may be necessary and their
    existence is checked for running particular analyses.

    Attributes
    ----------
    events: pd.DataFrame
        DataFrame containing rows of events and columns of respective event properties.
    direction: str, optional
        Playing direction of players in data fragment, should be either
        'lr' (left-to-right) or 'rl' (right-to-left).
    essential: list
        List of essential columns available for stored events.
    protected: list
        List of protected columns available for stored events.
    custom: list
        List of custom (i.e. non-essential and non-protected) columns available for
        stored events.
    """

    events: pd.DataFrame
    direction: str = None

    def __str__(self):
        return f"Floodlight Events object of shape {self.events.shape}"

    def __len__(self):
        return len(self.events)

    @property
    def essential(self):
        essential = list(set(self.events.columns) & set(essential_events_columns))

        return essential

    @property
    def protected(self):
        protected = list(set(self.events.columns) & set(protected_columns))

        return protected

    @property
    def custom(self):
        custom = list(
            set(self.events.columns)
            - (set(essential_events_columns) | set(protected_columns))
        )

        return custom

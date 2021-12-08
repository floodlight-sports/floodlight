from dataclasses import dataclass
from typing import Dict, Any
import warnings

import numpy as np
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

    def __post_init__(self):
        # check for missing essential columns
        missing_columns = self.missing_essential_columns()
        if missing_columns is not None:
            raise ValueError(
                f"Floodlight Events object is missing the essential "
                f"column(s) {missing_columns}!"
            )

        # warn if value ranges are violated
        incorrect_columns = self.incorrect_value_ranges()
        if incorrect_columns is not None:
            for col in incorrect_columns:
                warnings.warn(
                    f"Floodlight Events column {col} violating the defined value range!"
                    f" See floodlight.core.definitions for details."
                )

    def __str__(self):
        return f"Floodlight Events object of shape {self.events.shape}"

    def __len__(self):
        return len(self.events)

    def __getitem__(self, key):
        return self.events[key]

    def __setitem__(self, key, value):
        self.events[key] = value

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

    def missing_essential_columns(self):
        """Returns columns from essential_events_column that are missing in the inner
        events DataFrame().

        Returns
        -------
        missing_columns: List or None
            List of missing columns or None if no columns are missing.
        """
        missing_columns = [
            col for col in essential_events_columns if col not in self.essential
        ]

        if not missing_columns:
            missing_columns = None

        return missing_columns

    def incorrect_value_ranges(self):
        """Returns columns from the inner events DataFrame with value ranges violating
        the specifications from floodlight.core.definitions.
        """

        incorrect_columns = []

        # essential columns
        incorrect_columns += [
            col
            for col in self.essential
            if not self.check_single_column_value_range(col, essential_events_columns)
        ]

        # protected columns
        incorrect_columns += [
            col
            for col in self.protected
            if not self.check_single_column_value_range(col, protected_columns)
        ]

        if not incorrect_columns:
            incorrect_columns = None

        return incorrect_columns

    def check_single_column_value_range(
        self, col: str, definitions: Dict[str, Dict]
    ) -> bool:
        """Perform the checks for value range for a single column of the inner event
        DataFrame using the specifications from floodlight.core.definitions.

        Parameters
        ----------
        col: str
            Column of the inner event DataFrame to be checked
        definitions: Dict
            Dictionary for each column name to be checked. Each entry is a Dictionary
            containing definitions for allowed value ranges of the respective
            column. Information about value_range is given as a List of the form
            [min_value, max_value, extra_value] where the extra_value indicates an
            additionally allowed value that does not fit within the defined value range.

        Returns
        -------
        bool
            True if the checks for value range pass and False otherwise
        """
        # check if value range is defined
        if definitions[col]["value_range"] is None:
            return True

        # skip extra allowed values
        check_col = self.events[col]
        if len(definitions[col]["value_range"]) == 3:
            if definitions[col]["value_range"][2] is None:
                check_col = check_col.dropna()
            else:
                check_col = check_col[check_col != definitions[col]["value_range"][2]]

        # check value range for remaining values
        if not (definitions[col]["value_range"][0] <= check_col).all():
            return False
        if not (check_col <= definitions[col]["value_range"][1]).all():
            return False

        # all checks passed
        return True

    def add_frameclock(self, framerate: int):
        """Add the column "frameclock", computed as the rounded multiplication of
        gameclock and framerate, to the inner events DataFrame.

        Parameters
        ----------
        framerate: int
            Temporal resolution of data in frames per second/Hertz.
        """
        frameclock = np.full((len(self.events)), -1, dtype=int)
        frameclock[:] = int(self.events["gameclock"].values * framerate)
        self.events["frameclock"] = frameclock

    def find(self, col: str, value: Any) -> pd.Series:
        """Return all elements with the given value from the specified column of the
        inner events DataFrame.

        Parameters
        ----------
        col: str
            Column of the inner event DataFrame to be searched
        value: Any
            Value used to filter the entries from the specified column

        Returns
        -------
        pd.Series
            Series with all entries from the specified column that match the given
            value, is empty if no entry with the given value is found in the column.
        """
        if value is None:
            return self.events[col][self.events[col].isna()]
        else:
            return self.events[self.events[col] == value][col]

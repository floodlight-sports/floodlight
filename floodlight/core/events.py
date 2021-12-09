from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
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
    missing_essential_columns: list or None
        List of missing columns or None if no columns are missing.
    incorrect_value_range_columns: list or None
        List of columns that violate the definitions or None if all columns match
        the definitions.
    """

    events: pd.DataFrame
    direction: str = None

    def __post_init__(self):
        # check for missing essential columns
        missing_columns = self.essential_missing
        if missing_columns is not None:
            raise ValueError(
                f"Floodlight Events object is missing the essential "
                f"column(s) {missing_columns}!"
            )

        # warn if value ranges are violated
        incorrect_columns = self.essential_invalid
        if incorrect_columns is not None:
            for col in incorrect_columns:
                warnings.warn(
                    f"Floodlight Events column {col} does not match the defined value"
                    f"range (from floodlight.core.definitions). You can pursue at this "
                    f"point, however, be aware that this may lead to unexpected "
                    f"behavior in the future."
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

    @property
    def essential_missing(self):
        missing_columns = [
            col for col in essential_events_columns if col not in self.essential
        ]

        if not missing_columns:
            return None
        else:
            return missing_columns

    @property
    def essential_invalid(self):
        incorrect_columns = [
            col
            for col in self.essential
            if not self.column_values_in_range(col, essential_events_columns)
        ]

        if not incorrect_columns:
            return None
        else:
            return incorrect_columns

    @property
    def protected_missing(self):
        missing_columns = [
            col for col in protected_columns if col not in self.protected
        ]

        if not missing_columns:
            missing_columns = None

        return missing_columns

    @property
    def protected_invalid(self):
        incorrect_columns = [
            col
            for col in self.protected
            if not self.column_values_in_range(col, protected_columns)
        ]
        if not incorrect_columns:
            incorrect_columns = None

        return incorrect_columns

    def column_values_in_range(self, col: str, definitions: Dict[str, Dict]) -> bool:
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
            [min_value, max_value].

        Returns
        -------
        bool
            True if the checks for value range pass and False otherwise
        """
        # skip if value range is not defined
        if definitions[col]["value_range"] is None:
            return True

        # skip values that are None or NaN
        check_col = self.events[col].dropna()

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
        frameclock[:] = np.floor(self.events["gameclock"].values * framerate)
        self.events["frameclock"] = frameclock

    def find(
        self, conditions: Tuple[str, Any] or List[Tuple[str, Any]]
    ) -> pd.DataFrame:
        """Returns a DataFrame containing all entries from the inner events DataFrame
         that satisfy all given conditions.

        Parameters
        ----------
        conditions: Tuple or List of Tuples
            Conditions used to filter the columns. Conditions need to have the form
            (column, value) where column specifies the column of the inner events
            DataFrame and value specifies the desired value of the column entries.
        Returns
        -------
        filtered_events: pd.DataFrame
            A copy of the inner events DataFrame where all entries fulfill all criteria
            specified in conditions. The DataFrame can be empty if no entry with the
            given value is found in the column.
        """
        filtered_events = self.events.copy()

        for condition in conditions:
            if condition[1] is None:
                filtered_events = filtered_events[filtered_events[condition[0]].isna()]
            else:
                filtered_events = filtered_events[
                    filtered_events[condition[0]] == condition[1]
                ]

        return filtered_events

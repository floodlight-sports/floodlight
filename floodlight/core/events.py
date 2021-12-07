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
        self.check()

    def __str__(self):
        return f"Floodlight Events object of shape {self.events.shape}"

    def __len__(self):
        return len(self.events)

    def __getitem__(self, key):
        return self.events[key]

    def __setitem__(self, key, value):
        self.events[key] = value
        self.check()

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

    def check(self):
        """Checks an Event object and throws an error if any mandatory column misses

        It is checked if the Event object contains the mandatory columns (defined in
        floodlight.core.definitions) and an error is thrown if this check fails.
        Additionally, all essential and all protected columns from the Event object are
        checked for dtypes and value ranges from the definitions, however, if these
        checks fail there is only a warning at runtime.
        """
        # check for essential columns and throw error if check fails
        if not self.check_for_essential_cols():
            raise TypeError(
                "Found irregular Events object missing at least one essential column!"
            )

        # check dtypes and value range of essential and protected columns
        self.check_essential_cols()
        self.check_protected_cols()

    def check_for_essential_cols(self) -> bool:
        """Checks if the mandatory essential columns exist in the inner events DataFrame

        Returns
        -------
        check: bool
            True if all essential columns are in self.events and False otherwise
        """
        check = True
        for col in essential_events_columns:
            if col not in self.essential:
                check = False
                warnings.warn(
                    f"Events Object is missing the essential event column {col}!"
                )
        return check

    def check_essential_cols(self) -> bool:
        """Perform the checks against the definitions (from floodlight.core.definitions)
        for all essential columns in the inner events DataFrame

        Returns
        -------
        check: bool
            True if the checks for all essential columns pass and False otherwise
        """
        check = True
        for col in self.essential:
            check *= self.definition_check(col, essential_events_columns)

        return check

    def check_protected_cols(self) -> bool:
        """Perform the checks against the definitions (from
        floodlight.core.definitions) for all protected columns in the inner events
        DataFrame

        Returns
        -------
        check: bool
            True if the checks for all protected columns pass and False otherwise
        """
        check = True
        for col in self.protected:
            check *= self.definition_check(col, protected_columns)

        return check

    def definition_check(self, col: str, definitions: Dict[str, Dict]) -> bool:
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
        check: bool
            True if the checks for value range pass and False otherwise
        """
        check = True

        # check if value range is defined
        if definitions[col]["value_range"] is None:
            return check

        # skip extra allowed values
        check_col = self.events[col]
        if len(definitions[col]["value_range"]) == 3:
            if definitions[col]["value_range"][2] is None:
                check_col = check_col.dropna()
            else:
                check_col = check_col[check_col != definitions[col]["value_range"][2]]

        # check value range for remaining values
        if not (definitions[col]["value_range"][0] <= check_col).all():
            check = False
            warnings.warn(
                f"Events column {col} has at least one value smaller than "
                f"{definitions[col]['value_range'][0]}!"
            )
        if not (check_col <= definitions[col]["value_range"][1]).all():
            check = False
            warnings.warn(
                f"Events column {col} has at least one value larger than "
                f"{definitions[col]['value_range'][1]}!"
            )

        return check

    def add_frameclock(self, framerate: int):
        """Add the column frameclock, computed as the rounded multiplication of
        gameclock and framerate to the inner events DataFrame.

        Parameters
        ----------
        framerate: int
            Temporal resolution of data in frames per second/Hertz.
        """
        frameclock = np.full((len(self.events)), -1, dtype=int)
        frameclock[:] = np.round(self.events["gameclock"].values * framerate)
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

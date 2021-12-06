import warnings
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

    def __post_init__(self):

        # check for missing essential columns
        assert (
            self.check_for_essential_cols()
        ), "Could not create Events object, missing essential columns!"

        # check for ranges and dtypes
        assert self.check_essential_cols(), (
            "Could not create Events object, at least one essential column does not "
            "satisfy the general conditions!"
        )
        assert self.check_protected_cols(), (
            "Could not create Events object, at least one protected column does not "
            "satisfy the general conditions!"
        )

    def __str__(self):
        return f"Floodlight Events object of shape {self.events.shape}"

    def __len__(self):
        return len(self.events)

    def __getitem__(self, key):
        return self.events[key]

    def __setitem__(self, key, value):
        self.events[key] = value
        print("Checks")
        self.check_essential_cols()
        self.check_protected_cols()

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

    def check_for_essential_cols(self):
        check = True

        if "eID" not in self.essential:
            check = False
            warnings.warn("Events Object is missing the essential event column eID!")
        if "gameclock" not in self.essential:
            check = False
            warnings.warn(
                "Events Object is missing the essential event column gameclock!"
            )

        return check

    def check_essential_cols(self):
        check = True
        for col in self.essential:
            check *= self.definition_check(col, essential_events_columns)

        return check

    def check_protected_cols(self):
        check = True
        for col in self.protected:
            check *= self.definition_check(col, protected_columns)

        return check

    def definition_check(self, col, definitions):
        check = True

        # check dtypes
        if self.events.dtypes[col] not in definitions[col]["dtypes"]:
            check = False
            warnings.warn(
                f"Events column {col} having illegal dtype {self.events.dtypes[col]}, "
                f"should be {definitions[col]['dtypes']}!"
            )

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
            warnings.warn(
                f"Events column {col} has at least one value smaller than "
                f"{definitions[col]['value_range'][0]}!"
            )
        if not (check_col <= definitions[col]["value_range"][1]).all():
            warnings.warn(
                f"Events column {col} has at least one value larger than "
                f"{definitions[col]['value_range'][1]}!"
            )

        return check

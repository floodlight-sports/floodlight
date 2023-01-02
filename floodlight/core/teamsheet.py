from dataclasses import dataclass
from typing import Dict
import warnings

import pandas as pd

from floodlight.core.definitions import essential_teamsheet_columns, protected_columns


@dataclass
class Teamsheet:
    """Teamsheet storing player information. Core class of floodlight.

    Teamsheet data is stored in `pandas` ``DataFrame``, where each row stores one player
    with their different properties organized in columns. You may put whatever
    information you like in these columns. Yet, the column `"player"` is mandatory
    to identify a player. Some special column names are reserved for properties that
    follow conventions. These may be necessary and their existence is checked for
    running particular analyses.

    Parameters
    ----------
    teamsheet: pd.DataFrame
        DataFrame containing rows of players and columns of respective properties.

    Attributes
    ----------
    essential: list
        List of essential columns available for stored players.
    protected: list
        List of protected columns available for stored players.
    custom: list
        List of custom (i.e. non-essential and non-protected) columns available for
        stored events.
    essential_missing: list
        List of missing essential columns.
    essential_invalid: list
        List of essential columns that violate the definitions.
    protected_missing: list
        List of missing protected columns.
    protected_invalid: list
        List of protected columns that violate the definitions.
    """

    teamsheet: pd.DataFrame

    def __post_init__(self):
        # check for missing essential columns
        missing_columns = self.essential_missing
        if missing_columns:
            raise ValueError(
                f"Floodlight Teamsheet object is missing the essential "
                f"column(s) {missing_columns}!"
            )

        # warn if value ranges are violated
        incorrect_columns = self.essential_invalid
        if incorrect_columns:
            for col in incorrect_columns:
                warnings.warn(
                    f"Floodlight Teamsheet column {col} does not match the defined "
                    f"value range (from floodlight.core.definitions). You can pursue at"
                    f" this point, however, be aware that this may lead to unexpected "
                    f"behavior in the future."
                )

    def __str__(self):
        return f"Floodlight Events object of shape {self.teamsheet.shape}"

    def __len__(self):
        return len(self.teamsheet)

    def __getitem__(self, key):
        return self.teamsheet[key]

    def __setitem__(self, key, value):
        self.teamsheet[key] = value

    @property
    def essential(self):
        essential = [
            col for col in self.teamsheet.columns if col in essential_teamsheet_columns
        ]

        return essential

    @property
    def protected(self):
        protected = [col for col in self.teamsheet.columns if col in protected_columns]
        return protected

    @property
    def custom(self):
        custom = [
            col
            for col in self.teamsheet.columns
            if col not in essential_teamsheet_columns and col not in protected_columns
        ]

        return custom

    @property
    def essential_missing(self):
        missing_columns = [
            col for col in essential_teamsheet_columns if col not in self.essential
        ]

        return missing_columns

    @property
    def essential_invalid(self):
        invalid_columns = [
            col
            for col in self.essential
            if not self.column_values_in_range(col, essential_teamsheet_columns)
        ]

        return invalid_columns

    @property
    def protected_missing(self):
        missing_columns = [
            col for col in protected_columns if col not in self.protected
        ]

        return missing_columns

    @property
    def protected_invalid(self):
        invalid_columns = [
            col
            for col in self.protected
            if not self.column_values_in_range(col, protected_columns)
        ]

        return invalid_columns

    def column_values_in_range(self, col: str, definitions: Dict[str, Dict]) -> bool:
        """Check if values for a single column of the inner teamsheet DataFrame are in
        correct range using using the specifications from floodlight.core.definitions.

        Parameters
        ----------
        col: str
            Column name of the inner teamsheet DataFrame to be checked
        definitions: Dict
            Dictionary (from floodlight.core.definitions) containing specifications for
            the columns to be checked.

            The definitions need to contain an entry for the column to be checked and
            this entry needs to contain information about the value range in the form:
            ``definitions[col][value_range] = (min, max)``.

        Returns
        -------
        bool
            True if the checks for value range pass and False otherwise

        Notes
        -----
        Non-integer results of this computation will always be rounded to the next
        smaller integer.
        """
        # skip if value range is not defined
        if definitions[col]["value_range"] is None:
            return True

        # skip values that are None or NaN
        col_nan_free = self.teamsheet[col].dropna()

        # retrieve value range from definitions
        min_val, max_val = definitions[col]["value_range"]

        # check value range for remaining values
        if not (min_val <= col_nan_free).all() & (col_nan_free <= max_val).all():
            return False

        # all checks passed
        return True

    def get_links(self, keys: str, values: str) -> dict:
        """Creates dictionary of links between two columns of the teamsheet as specified
        by `keys` and `values`.

        Parameters
        ----------
        keys : str
            Column of teamsheet used as keys in links dictionary.
        values : str
            Column of teamsheet used as values in links dictionary.

        Returns
        -------
        links : dict
            Dictionary of links between columns specified by `keys` and `values`
            argument.
        """
        # checks
        if keys not in self.teamsheet.columns:
            raise ValueError(f"No '{keys}' column in teamsheet.")
        if values not in self.teamsheet.columns:
            raise ValueError(f"No '{values}' column in teamsheet.")
        if not self.teamsheet[keys].is_unique:
            raise ValueError(
                f"Cannot construct dictionary with unambiguous assignments"
                f" as '{keys}' column has duplicate entries."
            )
        # bin
        links = {}

        # loop through players
        for idx in self.teamsheet.index:
            # add key-value pairs to links dict:
            links[self.teamsheet.at[idx, keys]] = self.teamsheet.at[idx, values]

        return links

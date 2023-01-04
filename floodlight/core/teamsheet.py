from dataclasses import dataclass
from typing import Dict
from lxml import etree
import warnings

import pandas as pd

from floodlight.core.definitions import essential_teamsheet_columns, protected_columns


@dataclass
class Teamsheet:
    """Teamsheet storing player information. Core class of floodlight.

    Teamsheet data is stored in a `pandas` ``DataFrame``, where each row stores one
    player with their different properties organized in columns. Columns may contain
    any relevant information. A `"player"` column is required for instantiation
    to identify a player, and some particular column names are protected (see Notes).

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
        stored players.
    essential_missing: list
        List of missing essential columns.
    essential_invalid: list
        List of essential columns that violate the definitions.
    protected_missing: list
        List of missing protected columns.
    protected_invalid: list
        List of protected columns that violate the definitions.

    Notes
    -----
    Teamsheet data, particularly information available for players, may vary across
    data providers. To accommodate all data flavours, any column name or data type is
    permissible. However, one `essential` column is required (`"player"`). Other column
    names are `protected`. Using these names assumes that data stored in these columns
    follows conventions in terms of data types and value ranges. These are required for
    methods working with protected columns to assure correct calculations. Definitions
    for `essential` and `protected` columns can be found in
    :ref:`floodlight.core.definitions <definitions target>`.

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
                    f"The '{col}' column does not match the defined value range (from "
                    f"floodlight.core.definitions). This may lead to unexpected "
                    f"behavior of methods using this column."
                )

    def __str__(self):
        return f"Floodlight Teamsheet object of shape {self.teamsheet.shape}"

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

    @classmethod
    def from_file(
        cls, filepath_data: str, provider_name: str,
    ):
        """
        Creates a teamsheet object from a provider specific match information file.

        Parameters
        ----------
        filepath_data:  str
            Full path to file where the information is stored.
        provider_name: str
            The name of the template the teamsheet should follow. Currently supported
            are {'dfl'}.

        Returns
        -------
        teamsheet: Teamsheet
            A class instance of the given provider format.
        """
        if provider_name == "dfl":
            return cls.__parse_from_dfl_mat_info(filepath_data)
        else:
            return None

    def column_values_in_range(self, col: str, definitions: Dict[str, Dict]) -> bool:
        """Check if values for a single column of the inner teamsheet DataFrame are in
        correct range using the specifications from
        :ref:`floodlight.core.definitions <definitions target>`.

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

    @classmethod
    def __parse_from_dfl_mat_info(cls, filepath_mat_info):
        """Reads Teamsheet data from a xml file in DFL format and returns a
        teamsheet object.

        Parameters
        ----------
        filepath_mat_info: str or pathlib.Path
            Full path to XML File where the Match Information data in DFL format is saved

        Returns
        -------
        teamsheet: Teamsheet
        """
        # set up XML tree
        tree = etree.parse(str(filepath_mat_info))
        root = tree.getroot()

        # set up meta information
        teams = root.find("MatchInformation").find("Teams")
        home_id = root.find("MatchInformation").find("General").get("HomeTeamId")
        if "AwayTeamId" in root.find("MatchInformation").find("General").attrib:
            away_id = root.find("MatchInformation").find("General").get("AwayTeamId")
        elif "GuestTeamId" in root.find("MatchInformation").find("General").attrib:
            away_id = root.find("MatchInformation").find("General").get("GuestTeamId")
        else:
            away_id = None

        # intialize matchsheets
        home_teamsheet = pd.DataFrame(
            columns=["player", "pID", "jID", "position", "tID", "team_name"]
        )
        away_teamsheet = pd.DataFrame()

        # parse information
        for team in teams:
            if team.get("TeamId") == home_id:
                home_teamsheet["player"] = [
                    player.get("Shortname") for player in team.find("Players")
                ]
                home_teamsheet["pID"] = [
                    player.get("PersonId") for player in team.find("Players")
                ]
                home_teamsheet["jID"] = [
                    int(player.get("ShirtNumber")) for player in team.find("Players")
                ]
                home_teamsheet["position"] = [
                    player.get("PlayingPosition") for player in team.find("Players")
                ]
                home_teamsheet["tID"] = [home_id for _ in team.find("Players")]
                home_teamsheet["team_name"] = [
                    team.get("TeamName") for _ in team.find("Players")
                ]
            elif team.get("TeamId") == away_id:
                away_teamsheet["player"] = [
                    player.get("Shortname") for player in team.find("Players")
                ]
                away_teamsheet["pID"] = [
                    player.get("PersonId") for player in team.find("Players")
                ]
                away_teamsheet["jID"] = [
                    int(player.get("ShirtNumber")) for player in team.find("Players")
                ]
                away_teamsheet["position"] = [
                    player.get("PlayingPosition") for player in team.find("Players")
                ]
                away_teamsheet["tID"] = [away_id for _ in team.find("Players")]
                away_teamsheet["team_name"] = [
                    team.get("TeamName") for _ in team.find("Players")
                ]
            else:
                continue

        return cls(home_teamsheet), cls(away_teamsheet)

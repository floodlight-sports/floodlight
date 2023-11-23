import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

from floodlight.core.xy import XY
from floodlight.core.xyz import XYZ


def get_column_names_from_csv(
    filepath_data: Union[str, Path], delimiter: Optional[str] = ","
) -> List[str]:
    """Reads first line of a Kinexon.csv-file and extracts the column names.

    Parameters
    ----------
    filepath_data: str or pathlib.Path
        Full path to Kinexon.csv-file.
    delimiter : str, optional
        Delimiter used in the csv file. Defaults to ",".

    Returns
    -------
    columns: List[str]
        List with every column name of the .csv-file.
    """

    with open(str(filepath_data), encoding="utf-8") as f:
        columns = f.readline().split(delimiter)

    return columns


def _get_column_links(
    filepath_data: Union[str, Path], delimiter: Optional[str] = ","
) -> Union[None, Dict[str, int]]:
    """Creates a dictionary with the relevant recorded columns and their
    corresponding column index in the Kinexon.csv-file.

    Parameters
    ----------
    filepath_data: str or pathlib.Path
        Full path to Kinexon.csv-file.
    delimiter : str, optional
        Delimiter used in the csv file. Defaults to ",".

    Returns
    -------
    column_links: Dict[str, int]
        Dictionary with column index for relevant recorded columns.
        'column_links[column] = index'
        The following columns are currently considered relevant:
              floodlight id: 'column name in Kinexon.csv-file
            - time: 'ts in ms'
            - sensor_id: 'sensor id'
            - mapped_id: 'mapped id'
            - name: 'full name'
            - group_id: 'group id'
            - x_coord: 'x in m'
            - y_coord: 'y in m'
            - z_coord: 'z in m'
    """

    recorded_columns = get_column_names_from_csv(str(filepath_data), delimiter)

    # relevant columns
    mapping = {
        "ts in ms": "time",
        "sensor id": "sensor_id",
        "mapped id": "mapped_id",
        "league id": "league_id",
        "full name": "name",
        "number": "number",
        "group id": "group_id",
        "group name": "group_name",
        "x in m": "x_coord",
        "y in m": "y_coord",
        "z in m": "z_coord",
    }

    necessary_columns = ["time", "x_coord", "y_coord"]

    column_links = {}
    # loop
    for key in mapping:
        # create links
        if key in recorded_columns:
            column_links.update({mapping[key]: recorded_columns.index(key)})

    # check if necessary columns are available
    if not all(columns in column_links for columns in necessary_columns):
        warnings.warn(
            "Data file lacks critical information! "
            "No timestamp or coordinates found."
        )
        return None

    return column_links


def _get_group_id(df: pd.DataFrame, column_links: Dict[str, int]) -> pd.Series:
    """
    Fast calculation of the group identifier based on the 'group_name'
    and 'group_id' columns.


    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the Kinexon data including 'group_name'
        and 'group_id' columns.
    column_links: Dict[str, int]
        Dictionary with column index for relevant recorded columns.
        'column_links[column] = index'
        The following columns are currently considered relevant:
              floodlight id: 'column name in Kinexon.csv-file
            - time: 'ts in ms'
            - sensor_id: 'sensor id'
            - mapped_id: 'mapped id'
            - league_id: 'league id'
            - name: 'full name'
            - group_id: 'group id'
            - x_coord: 'x in m'
            - y_coord: 'y in m'
            - z_coord: 'z in m'

    Returns
    -------
    pd.Series
        The group identifier series.

    """
    # Fill missing values with '0' to avoid NaNs
    df["group_name"] = df["group_name"].fillna("0")
    df["group_id"] = df["group_id"].fillna("0")

    # Create a mask to identify rows where 'group_name' is not '0'
    mask = df["group_name"] != "0"

    # Assign values from 'group_name' to 'group_identifier' where mask is True
    df.loc[mask, "group_identifier"] = df.loc[mask, "group_name"]

    # Assign values from 'group_id' to 'group_identifier' where mask is False
    df.loc[~mask, "group_identifier"] = df.loc[~mask, "group_id"]

    # Return the 'group_identifier' series
    return df["group_identifier"]


def get_meta_data(
    filepath_data: Union[str, Path], delimiter: str = ","
) -> Tuple[Dict[str, Dict[str, List[str]]], int, int, int]:
    """
    Reads Kinexon's position data file and extracts meta-data about groups, sensors,
    length, and framerate using optimized methods for better performance.

    Parameters
    ----------
    filepath_data: str or pathlib.Path
        Full path to Kinexon.csv-file.
    delimiter : str, optional
        Delimiter used in the csv file. Defaults to ",".

    Returns
    -------
    pID_dict: Dict[str, Dict[str, List[str]]],
        Nested dictionary that stores information about the pIDs from every player-
        identifying column in every group.
        'pID_dict[group_identifier][identifying_column] = [pID1, pID2, ..., pIDn]'
        When recording and exporting Kinexon data, the pID can be stored
        in different columns. Player-identifying columns are "sensor_id", "mapped_id",
        and "full_name". If the respective column is in the recorded data, its pIDs are
        listed in pID_dict.
        As with pID, group ids can be stored in different columns. Group-identifying
        columns are "group_name" and "group_id". If both are available, group_name will
        be favored over group_id as the group_identifier.
    number_of_frames: int
        Number of frames from the first to the last recorded frame.
    framerate: int
        Estimated framerate in frames per second. Estimated from the smallest difference
        between two consecutive frames.
    t_null: int
        Timestamp of the first recorded frame
    """
    # Load data into pandas dataframe
    df = pd.read_csv(filepath_data, delimiter=delimiter)

    # Create a dictionary linking columns to their index
    column_links = _get_column_links(str(filepath_data), delimiter)
    sensor_identifier = {"league_id", "name", "number", "sensor_id", "mapped_id"}
    column_links_set = set(column_links)
    recorded_sensor_identifier = list(column_links_set & sensor_identifier)
    sensor_links = {
        key: index
        for (key, index) in column_links.items()
        if key in recorded_sensor_identifier
    }
    group_identifier_set = {"group_id", "group_name"}
    recorded_group_identifier = list(column_links_set & group_identifier_set)

    # Assume df_columns are the current column names
    df_columns = df.columns

    # Create a dictionary with old column names and new names from column_links
    rename_dict = {df_columns[index]: name for name, index in column_links.items()}

    # Rename columns
    df.rename(columns=rename_dict, inplace=True)

    # Initialize dictionary for pIDs
    pID_dict = defaultdict(lambda: defaultdict(set))

    # Determine group id
    df["group_id"] = _get_group_id(df, recorded_group_identifier)

    # Add sensor identifiers to pID_dict
    for identifier in sensor_links:
        grouped = df.groupby("group_id")[identifier].unique()
        for group_id, pIDs in grouped.items():
            pID_dict[group_id][identifier] = set(pIDs)

    # Convert pID_dict to a normal dict with lists instead of sets
    pID_dict = {
        group: {id: list(pIDs) for id, pIDs in identifiers.items()}
        for group, identifiers in pID_dict.items()
    }

    # Get list of unique sorted timestamps
    timestamps = df["time"].unique()
    timestamps.sort()

    # Estimate framerate
    minimum_time_step = np.min(np.diff(timestamps))
    framerate = 1000 / minimum_time_step

    # Warn if framerate is non-integer
    if not framerate.is_integer():
        warnings.warn(
            f"Non-integer frame rate: Minimum time step of"
            f"{minimum_time_step} detected. Framerate is round to {int(framerate)}."
        )

    # Round framerate to integer
    framerate = int(framerate)

    # Calculate number of frames and timestamp of first recorded frame
    number_of_frames = int((timestamps[-1] - timestamps[0]) / (1000 / framerate))
    t_null = timestamps[0]

    return pID_dict, number_of_frames, framerate, t_null


def _get_available_sensor_identifier(pID_dict: Dict[str, Dict[str, List[str]]]) -> str:
    """Returns an available sensor identifier that has been recorded. Will favor
    "name" over "mapped_id" over "sensor_id" over "number".

    Parameters
    ----------
    pID_dict: Dict[str, Dict[str, List[str]]],
        Nested dictionary that stores information about the pIDs from every player-
        identifying column in every group.
        'pID_dict[group][identifying_column] = [pID1, pID2, ..., pIDn]'
        When recording and exporting Kinexon data, the pID can be stored
        in different columns. Player-identifying columns are "sensor_id", "mapped_id",
        and "full_name". If the respective column is in the recorded data, its pIDs are
        listed in pID_dict.

    Returns
    -------
    identifier: str
        One sensor identifier that has been recorded.
    """

    player_identifiers = ["league_id", "name", "mapped_id", "sensor_id", "number"]
    available_identifier = [
        idt for idt in player_identifiers if idt in list(pID_dict.values())[0]
    ]
    identifier = available_identifier[0]

    return identifier


def create_links_from_meta_data(
    pID_dict: Dict[str, Dict[str, List[str]]], identifier: str = None
) -> Dict[str, Dict[str, int]]:
    """Creates a dictionary from the pID_dict linking the identifier to the xID.

    Parameters
    ----------
    pID_dict: Dict[str, Dict[str, List[str]]]
        Nested dictionary that stores information about the pIDs from every player-
        identifying column in every group. The format is
        ``pID_dict[group][identifying_column] = [pID1, pID2, ..., pIDn]``.
        When recording and exporting Kinexon data, the pID can be stored in different
        columns. Player-identifying columns are ``"sensor_id"``, ``"mapped_id"``, and
        ``"full_name"``. If the respective column is in the recorded data, its pIDs are
        listed in ``pID_dict``.
    identifier: str, optional
        Column-name of personal identifier in Kinexon.csv-file, defaults to None.
        Can be one of: ``"sensor_id"``, ``"mapped_id"``, ``"name"``.

        When recording and exporting Kinexon data, the pID can be stored
        in different columns. Player-identifying columns are ``"sensor_id"``,
        ``"mapped_id"``, and ``"full_name"``. If specified to one of the above, keys in
        links will be the pIDs in that column. If not specified, it will use one of the
        columns, favoring ``"name"`` over ``"mapped_id"`` over ``"sensor_id"``.

    Returns
    -------
    links: Dict[str, Dict[str, int]]
        Link-dictionary of the form ``links[group][identifier-ID] = xID``.
    """

    if identifier is None:
        # available sensor identifier
        identifier = _get_available_sensor_identifier(pID_dict)

    links = {}
    for group_id in pID_dict:
        links.update(
            {
                group_id: {
                    ID: xID for xID, ID in enumerate(pID_dict[group_id][identifier])
                }
            }
        )

    return links


def read_position_data_csv(
    filepath_data: Union[str, Path], delimiter: Optional[str] = ","
) -> List[XYZ]:
    """
    Parses a Kinexon csv file and extracts position data.
    Kinexon's local positioning system delivers one .csv file containing the position
    data. This function provides a high-level access to Kinexon data by parsing "the
    full file" given the path to the file.

    This function uses pandas for efficient data processing,
    and minimizes loops where possible.

    Parameters
    ----------
    filepath_data: str or pathlib.Path
        Full path to Kinexon.csv-file.
    delimiter : str, optional
        Delimiter used in the csv file. Defaults to ",".

    Returns
    -------
    positions: List[XYZ]
        List of XYZ-objects for the whole game, one per group. The order of groups is
        ascending according to their group_id. If no groups are specified in the file,
        all data gets assigned to a dummy group "0". The order inside the groups is
        ascending according to their appearance in the data.
    """

    # Load data into pandas dataframe.
    # # This replaces the initial file read and line-by-line parsing
    df = pd.read_csv(filepath_data, delimiter=delimiter)

    # Get metadata using the optimized function
    pID_dict, number_of_frames, framerate, t_null = get_meta_data(
        filepath_data, delimiter
    )

    # Get links dictionary, used later for mapping sensor id to column index
    links = create_links_from_meta_data(pID_dict)

    # Get the links for the columns in the file
    column_links = _get_column_links(str(filepath_data), delimiter)

    # Define possible sensor identifiers and get the ones recorded in the file
    column_links_set = set(column_links)

    # Define possible group identifiers and get the ones recorded in the file
    group_identifier_set = {"group_id", "group_name"}
    recorded_group_identifier = list(column_links_set & group_identifier_set)

    # The current column names of the dataframe
    df_columns = df.columns

    # Create a dictionary with old column names and new names from column_links
    rename_dict = {df_columns[index]: name for name, index in column_links.items()}

    # Rename columns to match the links
    df.rename(columns=rename_dict, inplace=True)

    # Get available sensor identifier
    identifier = _get_available_sensor_identifier(pID_dict)

    # Create np.array for data, replacing the initial array creation
    number_of_sensors = {group: len(links[group]) for group in links}
    xydata = {
        group: np.full([number_of_frames + 1, number_of_sensors[group] * 3], np.nan)
        for group in links
    }

    # Calculate group_id for each row using the optimized function
    df["group_id"] = _get_group_id(df, recorded_group_identifier)

    # Flatten the links dictionary to match the sensors with their column indices
    flat_links = {k: v for d in links.values() for k, v in d.items()}
    df["x_col"] = df[identifier].map(flat_links) * 3
    df["y_col"] = df["x_col"] + 1
    df["z_col"] = df["x_col"] + 2  # Add z_col calculation

    # Calculate row index for each row
    df["row"] = ((df["time"] - t_null) / (1000 / framerate)).astype(int)

    # Insert (x, y)-data into column and row of respective array,
    # replacing the original for-loop.
    for group_id, group_data in df.groupby("group_id"):
        valid_x = group_data["x_coord"].notna()
        valid_y = group_data["y_coord"].notna()
        valid_z = group_data["z_coord"].notna()


        xydata[group_id][
            group_data.loc[valid_x, "row"], group_data.loc[valid_x, "x_col"]
        ] = group_data.loc[valid_x, "x_coord"]
        xydata[group_id][
            group_data.loc[valid_y, "row"], group_data.loc[valid_y, "y_col"]
        ] = group_data.loc[valid_y, "y_coord"]
        xydata[group_id][
            group_data.loc[valid_z, "row"], group_data.loc[valid_z, "z_col"]
        ] = group_data.loc[valid_z, "z_coord"]  # Insert Z-coordinate data

    # Convert to XY objects
    data_objects = [XYZ(xyz=xydata[group_id], framerate=framerate) for group_id in xydata]

    return data_objects

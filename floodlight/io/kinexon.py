import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Union

import numpy as np

from floodlight.core.xy import XY


def get_column_names_from_csv(filepath_data: Union[str, Path]) -> List[str]:
    """Reads first line of a Kinexon.csv-file and extracts the column names.

    Parameters
    ----------
    filepath_data: str or pathlib.Path
        Full path to Kinexon.csv-file.

    Returns
    -------
    columns: List[str]
        List with every column name of the .csv-file.
    """

    with open(str(filepath_data), encoding="utf-8") as f:
        columns = f.readline().split(",")

    return columns


def _get_column_links(filepath_data: Union[str, Path]) -> Union[None, Dict[str, int]]:
    """Creates a dictionary with the relevant recorded columns and their
    corresponding column index in the Kinexon.csv-file.

    Parameters
    ----------
    filepath_data: str or pathlib.Path
        Full path to Kinexon.csv-file.

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
    """

    recorded_columns = get_column_names_from_csv(str(filepath_data))

    # relevant columns
    mapping = {
        "ts in ms": "time",
        "sensor id": "sensor_id",
        "mapped id": "mapped_id",
        "full name": "name",
        "number": "number",
        "group id": "group_id",
        "group name": "group_name",
        "x in m": "x_coord",
        "y in m": "y_coord",
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


def _get_group_id(
    recorded_group_identifier: List[str],
    column_links: Dict[str, int],
    single_line: List[str],
) -> Union[str, None]:
    """Returns the group_name or group_id if it was recorded or "0" if not.
    Favors the group_name over the group_id.

    Parameters
    ----------
    recorded_group_identifier: List[str]
        List of all recorded group identifiers. Group identifiers are "group_id" or
        "group_name".
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
    single_line: List[str]
        Single line of a Kinexon.csv-file that has been split at the respective
        delimiter, eg. ",".

    Returns
    -------
    group_id: str
        The respective group id in that line or "0" if there is no group id.
    """

    # check for group identifier
    has_groups = len(recorded_group_identifier) > 0

    if has_groups:
        # extract group identifier
        if "group_name" in recorded_group_identifier:
            group_identifier = "group_name"
        elif "group_id" in recorded_group_identifier:
            group_identifier = "group_id"
        else:
            warnings.warn("Data has groups but no group identifier!")
            return None

        group_id = single_line[column_links[group_identifier]]

    # no groups
    else:
        group_id = "0"

    return group_id


def get_meta_data(
    filepath_data: Union[str, Path]
) -> Tuple[Dict[str, Dict[str, List[str]]], int, int, int]:
    """Reads Kinexon's position data file and extracts meta-data about groups, sensors,
    length and framerate.

    Parameters
    ----------
    filepath_data: str or pathlib.Path
        Full path to Kinexon.csv-file.

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

    column_links = _get_column_links(str(filepath_data))
    sensor_identifier = {"name", "number", "sensor_id", "mapped_id"}
    column_links_set = set(column_links)
    recorded_sensor_identifier = list(column_links_set & sensor_identifier)
    sensor_links = {
        key: index
        for (key, index) in column_links.items()
        if key in recorded_sensor_identifier
    }
    group_identifier_set = {"group_id", "group_name"}
    recorded_group_identifier = list(column_links_set & group_identifier_set)

    # dict for pIDs
    pID_dict = {}
    # list for timestamps
    t = []
    # check for group identifier
    has_groups = len(recorded_group_identifier) > 0
    if not has_groups:
        warnings.warn("Since no group exist in data, dummy group '0' is created!")

    # loop
    with open(str(filepath_data), "r", encoding="utf-8") as f:
        # skip the header of the file
        _ = f.readline()
        while True:
            line_string = f.readline()
            # terminate if at end of file
            if len(line_string) == 0:
                break
            # split str
            line = line_string.split(",")
            # extract frames timestamp
            t.append(int(line[column_links["time"]]))
            # extract group_id
            group_id = _get_group_id(recorded_group_identifier, column_links, line)
            # create group dict in pID_dict
            if group_id not in pID_dict:
                pID_dict.update({group_id: {}})
            # create links
            for identifier in sensor_links:
                # extract identifier
                if identifier not in pID_dict[group_id]:
                    pID_dict[group_id].update({identifier: []})
                # extract ids
                if line[column_links[identifier]] not in pID_dict[group_id][identifier]:
                    pID_dict[group_id][identifier].append(
                        line[column_links[identifier]]
                    )

    # sort dict
    pID_dict = dict(sorted(pID_dict.items()))

    # estimate framerate
    timestamps = list(set(t))
    timestamps.sort()
    timestamps = np.array(timestamps)
    minimum_time_step = np.min(np.diff(timestamps))
    # timestamps are in milliseconds. Magic number 1000 is needed for conversion to
    # seconds.
    framerate = 1000 / minimum_time_step

    # non-integer framerate
    if not framerate.is_integer():
        warnings.warn(
            f"Non-integer frame rate: Minimum time step of "
            f"{minimum_time_step} detected. Framerate is round to "
            f"{int(framerate)}."
        )

    framerate = int(framerate)

    # 1000 again needed to account for millisecond to second conversion.
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

    player_identifiers = ["name", "mapped_id", "sensor_id", "number"]
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


def read_kinexon_file(filepath_data: Union[str, Path]) -> List[XY]:
    """Parse Kinexon files and extract position data.

    Kinexon's local positioning system delivers one .csv file containing the position
    data. This function provides a high-level access to Kinexon data by parsing "the
    full file" given the path to the file.

    Parameters
    ----------
    filepath_data: str or pathlib.Path
        Full path to Kinexon.csv-file.

    Returns
    -------
    positions: List[XY]
        List of XY-objects for the whole game, one per group. The order of groups is
        ascending according to their group_id. If no groups are specified in the file,
        all data gets assigned to a dummy group "0". The order inside the groups is
        ascending according to their appearance in the data.
    """

    # get metadata
    pID_dict, number_of_frames, framerate, t_null = get_meta_data(filepath_data)

    # get links
    links = create_links_from_meta_data(pID_dict)
    # get column-links
    column_links = _get_column_links(filepath_data)
    column_links_set = set(column_links)
    group_identifier_set = {"group_id", "group_name"}
    recorded_group_identifier = list(column_links_set & group_identifier_set)

    # available sensor identifier
    identifier = _get_available_sensor_identifier(pID_dict)

    # create np.array for data
    number_of_sensors = {}
    xydata = {}
    for group in links:
        number_of_sensors.update({group: len(links[group])})
        xydata.update(
            {
                group: np.full(
                    [number_of_frames + 1, number_of_sensors[group] * 2], np.nan
                )
            }
        )

    # loop
    with open(str(filepath_data), "r", encoding="utf-8") as f:
        # skip the header of the file
        _ = f.readline()
        while True:
            line_string = f.readline()
            # terminate if at end of file
            if len(line_string) == 0:
                break
            # split str
            line = line_string.split(",")
            timestamp = int(line[column_links["time"]])
            # set group
            group_id = _get_group_id(recorded_group_identifier, column_links, line)
            # set column
            x_col = links[group_id][line[column_links[identifier]]] * 2
            y_col = x_col + 1
            # set row
            row = int((timestamp - t_null) / (1000 / framerate))
            # set (x, y)-data
            x_coordinate = line[column_links["x_coord"]]
            y_coordinate = line[column_links["y_coord"]]

            # insert (x, y)-data into column and row of respective array
            if x_coordinate != "":
                xydata[group_id][row, x_col] = x_coordinate

            if y_coordinate != "":
                xydata[group_id][row, y_col] = y_coordinate

    data_objects = []
    for group_id in xydata:
        data_objects.append(XY(xy=xydata[group_id], framerate=framerate))

    return data_objects

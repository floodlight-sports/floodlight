from pathlib import Path
from typing import List, Dict, Tuple, Union

import numpy as np

from floodlight.core.xy import XY


def get_column_names_from_csv(filepath_data: Union[str, Path]) -> List[str]:
    """Reads first line of a Kinexon and extracts the column names.

    Parameters
    ----------
    filepath_data: str or pathlib.Path
        Full path to Kinexon.csv-file.

    Returns
    -------
    parameters: List[str]
        List with every column name of the .csv-file.
    """

    with open(filepath_data) as f:
        parameters = f.readline().split(",")

    return parameters


def _get_parameter_links(
    filepath_data: Union[str, Path]
) -> Union[None, Dict[str, int]]:
    """Creates a dictionary with the relevant recorded parameters and their
    corresponding column index in the Kinexon.csv-file.

    Parameters
    ----------
    filepath_data: str of pathlib.Path
        Full path to Kinexon.csv-file.


    Returns
    -------
    parameter_links: Dict[str, int]
        Dictionary with column index for relevant recorded parameters.
        'parameter_links[parameter] = index'
        The following parameters are currently considered relevant:
              floodligth id: 'column name in Kinexon.csv-file
            - time: 'ts in ms'
            - sensor_id: 'sensor id'
            - mapped_id: 'mapped id'
            - name: 'full name'
            - group_id: 'group id'
            - x_coord: 'x in m'
            - y_coord: 'y in m'

    """

    recorded_parameters = get_column_names_from_csv(str(filepath_data))

    # relevant columns
    mapping = {
        "time": "ts in ms",
        "sensor_id": "sensor id",
        "mapped_id": "mapped id",
        "name": "full name",
        "group_id": "group id",
        "x_coord": "x in m",
        "y_coord": "y in m",
    }

    necessary_columns = ["time", "x_coord", "y_coord"]

    parameter_links = {}
    # loop
    for key in mapping:
        # create links
        if mapping[key] in recorded_parameters:
            parameter_links.update({key: recorded_parameters.index(mapping[key])})

    # insert group_id if its missing
    if "group_id" not in parameter_links:
        parameter_links.update({"group_id": None})

    # check if necessary columns are available
    if not all(params in parameter_links for params in necessary_columns):
        print(
            "Data file lacks critical information! "
            "No timestamp or coordinates found."
        )
        return None

    return parameter_links


def get_meta_data(
    filepath_data: Union[str, Path]
) -> Tuple[Dict[str, Dict[str, List[str]]], int, int, int]:
    """Reads Kinexon's position data file and extracts meta-data about groups, sensors,
    length and framerate

    Parameters
    ----------
    filepath_data: str of pathlib.Path
        Full path to Kinexon.csv-file.

    Returns
    -------
    team_infos: Dict[str, Dict[str, List[str]]],
        Nested dictionary that stores information about the ids from every identifier
        in every group. Identifier are sensor_id, mapped_id, name. If the respective
        id-type is not in the recorded parameters, it is not listed in team_infos.
        'team_info[group][identifier] = [id1, id2, ..., idn]'
    number_of_frames: int
        Number of frames from the first to the last recorded frame.
    frame_rate: int
        Estimated framerate in frames per second. Estimated from the smallest difference
        between two consecutive frames.
    t_null: int
        Timestamp of the first recorded frame
    """

    parameter_links = _get_parameter_links(str(filepath_data))

    # sensor-identifier
    sensor_links = parameter_links.copy()
    for key in ["time", "group_id", "x_coord", "y_coord"]:
        del sensor_links[key]

    team_infos = {}
    t = []
    # check for group id
    if parameter_links["group_id"] is not None:
        # loop
        with open(str(filepath_data), "r") as f:
            while True:
                line_string = f.readline()
                # terminate if at end of file
                if len(line_string) == 0:
                    break
                line = line_string.split(",")
                # skip the header of the file
                if line[parameter_links["time"]].isdecimal():
                    # extract group id
                    t.append(int(line[parameter_links["time"]]))
                    if line[parameter_links["group_id"]] not in team_infos:
                        team_infos.update({line[parameter_links["group_id"]]: {}})
                    # create links
                    for identifier in sensor_links:
                        # extract identifier
                        if (
                            identifier
                            not in team_infos[line[parameter_links["group_id"]]]
                        ):
                            team_infos[line[parameter_links["group_id"]]].update(
                                {identifier: []}
                            )
                        # extract ids
                        if (
                            line[parameter_links[identifier]]
                            not in team_infos[line[parameter_links["group_id"]]][
                                identifier
                            ]
                        ):
                            team_infos[line[parameter_links["group_id"]]][
                                identifier
                            ].append(line[parameter_links[identifier]])
    # no group id in data
    else:
        print("Since no groups exist in data, artificial group '0' is created!")
        with open(str(filepath_data), "r") as f:
            while True:
                line_string = f.readline()
                # terminate if at end of file
                if len(line_string) == 0:
                    break
                line = line_string.split(",")
                # skip the header of the file
                if line[parameter_links["time"]].isdecimal():
                    t.append(int(line[parameter_links["time"]]))
                    if "0" not in team_infos:
                        team_infos.update({"0": {}})
                    # create links
                    for identifier in sensor_links:
                        # extract identifier
                        if identifier not in team_infos["0"]:
                            team_infos["0"].update({identifier: []})
                        # extract ids
                        if (
                            line[parameter_links[identifier]]
                            not in team_infos["0"][identifier]
                        ):
                            team_infos["0"][identifier].append(
                                line[parameter_links[identifier]]
                            )

    # sort dict
    team_infos = dict(sorted(team_infos.items()))

    # estimate framerate
    timestamps = list(set(t))
    timestamps.sort()
    timestamps = np.array(timestamps)
    minimum_time_step = np.min(np.diff(timestamps))
    frame_rate = int(1000 / minimum_time_step)

    number_of_frames = int((timestamps[-1] - timestamps[0]) / (1000 / frame_rate))
    t_null = timestamps[0]

    return team_infos, number_of_frames, frame_rate, t_null


def create_links_from_meta_data(
    team_infos: Dict[str, Dict[str, List[str]]], identifier: str = None
) -> Dict[str, Dict[str, int]]:
    """Creates a dictionary from the team_info dict linking the identifier to the xID.

    Parameters
    ----------
    team_infos: Dict[str, Dict[str, List[str]]],
        Nested dictionary that stores information about the ids from every identifier
        in every group. Identifier are sensor_id, mapped_id, name. If the respective
        id-type is not in the recorded parameters, it is not listed in team_infos.
        'team_info[group][identifier] = [id1, id2, ..., idn]'
    identifier: str
        Column-name of personal identifier in Kinexon.csv-file.
        Can be one of:
            - 'sensor_id'
            - 'mapped_id'
            - 'name'

    Returns
    -------
    links: Dict[str, Dict[str, int]]
        Link-dictionary of the form `links[group][identifier-ID] = xID`.
    """

    for player_id in ["name", "mapped_id", "sensor_id"]:
        if identifier is None:
            if player_id in list(team_infos.values())[0]:
                identifier = player_id
                break

    links = {}
    for gID in team_infos:
        links.update(
            {gID: {ID: xID for xID, ID in enumerate(team_infos[gID][identifier])}}
        )

    return links


def read_kinexon_file(filepath_data: Union[str, Path], identifier: str = None) -> XY:
    """Parse Kinexon files and extract position data.

    Kinexon's local positioning system delivers one .csv file containing the position
    data. This function provides a high-level access to Kinexon data by parsing "the
    full file" given the path to the file.

    Parameters
    ----------
    filepath_data: str of pathlib.Path
        Full path to Kinexon.csv-file.
    identifier: str
        Column-name of personal identifier in Kinexon.csv-file.
        Can be one of:
            - 'sensor_id'
            - 'mapped_id'
            - 'name'

    Returns
    -------
    positions: XY
        XY-Object for the whole game. The order of groups is ascending according to
        their group id. The order inside the groups is ascending according to their
        appearance in the data.
    """

    # get metadata
    team_infos, number_of_frames, frame_rate, t_null = get_meta_data(filepath_data)

    # get links
    links = create_links_from_meta_data(team_infos, identifier)
    # get parameter-links
    parameter_links = _get_parameter_links(filepath_data)

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
    with open(str(filepath_data), "r") as f:
        while True:
            line_string = f.readline()
            # terminate if at end of file
            if len(line_string) == 0:
                break
            line = line_string.split(",")
            # skip the header of the file
            if line[parameter_links["time"]].isdecimal():
                timestamp = int(line[parameter_links["time"]])
                # set group
                group = line[parameter_links["group_id"]]
                # set column
                x_col = links[group][line[parameter_links["name"]]] * 2
                y_col = x_col + 1
                # set row
                row = int((timestamp - t_null) / (1000 / frame_rate))

                # set (x, y)-data
                x_coordinate = line[parameter_links["x_coord"]]
                y_coordinate = line[parameter_links["y_coord"]]

                # insert (x, y)-data into column and row of respective array
                if x_coordinate != "":
                    xydata[group][row, x_col] = x_coordinate

                if y_coordinate != "":
                    xydata[group][row, y_col] = y_coordinate

    # join group arrays
    xy = None
    for group in xydata:
        if xy is None:
            xy = xydata[group]
        else:
            xy = np.concatenate((xy, xydata[group]), axis=1)

    # create XY-object
    positions = XY(xy=xy)

    return positions


#
# filepath_data = "C:\\Users\\ke6564\\Desktop\\Studium\\Promotion\\Data\\SSG_3v3.csv"
# filepath_data = (
#     "C:\\Users\\ke6564\\Desktop\\Studium\\Promotion\\"
#     "Data\\Handball\\HBL_Positions\\edit\\Game_1.csv"
# )
# xy = read_kinexon_file(filepath_data)

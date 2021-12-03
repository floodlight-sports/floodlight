from pathlib import Path
from typing import List, Dict, Tuple, Union

import numpy as np

from floodlight.core.xy import XY


def get_column_names_from_csv(filepath_data: Union[str, Path]) -> List[str]:
    """

    Parameters
    ----------
    filepath_data

    Returns
    -------

    """

    with open(filepath_data) as f:
        parameters = f.readline().split(",")

    return parameters


def _get_parameter_links(filepath_data: Union[str, Path]) -> Dict[str, int]:
    """

    Parameters
    ----------
    filepath_data:

    Returns
    -------

    """

    recorded_parameters = get_column_names_from_csv(str(filepath_data))

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
    for key in mapping:
        if mapping[key] in recorded_parameters:
            parameter_links.update({key: recorded_parameters.index(mapping[key])})

    if "group_id" not in parameter_links:
        parameter_links.update({"group_id": None})

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
    """

    Parameters
    ----------
    filepath_data

    Returns
    -------

    """

    parameter_links = _get_parameter_links(filepath_data)

    sensor_links = parameter_links.copy()
    for key in ["time", "group_id", "x_coord", "y_coord"]:
        del sensor_links[key]

    team_infos = {}
    t = []

    if parameter_links["group_id"] is not None:
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
                    if line[parameter_links["group_id"]] not in team_infos:
                        team_infos.update({line[parameter_links["group_id"]]: {}})

                    for param in sensor_links:
                        if param not in team_infos[line[parameter_links["group_id"]]]:
                            team_infos[line[parameter_links["group_id"]]].update(
                                {param: []}
                            )
                        if (
                            line[parameter_links[param]]
                            not in team_infos[line[parameter_links["group_id"]]][param]
                        ):
                            team_infos[line[parameter_links["group_id"]]][param].append(
                                line[parameter_links[param]]
                            )
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

                    for param in sensor_links:
                        if param not in team_infos["0"]:
                            team_infos["0"].update({param: []})
                        if line[parameter_links[param]] not in team_infos["0"][param]:
                            team_infos["0"][param].append(line[parameter_links[param]])

    team_infos = dict(sorted(team_infos.items()))

    timestamps = list(set(t))
    timestamps.sort()
    timestamps = np.array(timestamps)
    minimum_time_step = np.min(np.diff(timestamps))
    frame_rate = int(1000 / minimum_time_step)
    number_of_frames = int((timestamps[-1] - timestamps[0]) / (1000 / frame_rate))
    t_null = timestamps[0]

    return team_infos, number_of_frames, frame_rate, t_null


def create_links_from_meta_data(
    team_infos: Dict[str, Dict[str, List[str]]], identificator: str = None
) -> Dict[str, Dict[str, int]]:
    """

    Parameters
    ----------
    team_infos
    identificator

    Returns
    -------

    """

    for player_id in ["name", "mapped_id", "sensor_id"]:
        if identificator is None:
            if player_id in list(team_infos.values())[0]:
                identificator = player_id
                break

    links = {}
    for gID in team_infos:
        links.update(
            {gID: {ID: xID for xID, ID in enumerate(team_infos[gID][identificator])}}
        )

    return links


def read_kinexon_file(
    filepath_data: Union[str, Path], identificator: str = None
) -> Tuple[Dict[str, Dict[str, List[str]]], XY]:
    """

    Parameters
    ----------
    filepath_data
    mapped_id

    Returns
    -------

    """

    team_infos, number_of_frames, frame_rate, t_null = get_meta_data(filepath_data)

    links = create_links_from_meta_data(team_infos, identificator)
    parameter_links = _get_parameter_links(filepath_data)

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
                group = line[parameter_links["group_id"]]
                x_col = (
                    links[group][line[parameter_links["name"]]] * 2
                )  # build generic for ID used in create links
                y_col = x_col + 1
                row = int((timestamp - t_null) / (1000 / frame_rate))

                x_coordinate = line[parameter_links["x_coord"]]
                y_coordinate = line[parameter_links["y_coord"]]

                if x_coordinate != "":
                    xydata[group][row, x_col] = x_coordinate

                if y_coordinate != "":
                    xydata[group][row, y_col] = y_coordinate

    xy = None
    for group in xydata:
        if xy is None:
            xy = xydata[group]
        else:
            xy = np.concatenate((xy, xydata[group]), axis=1)

    positions = XY(xy=xy)

    return team_infos, positions


filepath_data = "C:\\Users\\ke6564\\Desktop\\Studium\\Promotion\\Data\\SSG_3v3.csv"
filepath_data = (
    "C:\\Users\\ke6564\\Desktop\\Studium\\Promotion\\"
    "Data\\Handball\\HBL_Positions\\edit\\Game_1.csv"
)
team_infos, xy = read_kinexon_file(filepath_data)

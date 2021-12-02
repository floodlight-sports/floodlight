from pathlib import Path
from typing import List, Dict, Tuple, Union

import csv
import numpy as np


from floodlight.core.xy import XY


def _csv_to_dict_generator(filename_path: Union[str, Path]) -> Dict[str, str]:
    """
    Wraps incoming .csv file into a generator to loop over.

    Parameters
    ----------
    filename_path: str
        path to .csv file.

    Returns
    -------
    row: dict
        Yields one row of the .csv file with each iteration.
    """
    with open(filename_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def get_recorded_parameters(filename_path: Union[str, Path]) -> List[str]:
    """

    Parameters
    ----------
    filename_path

    Returns
    -------

    """

    with open(filename_path) as f:
        parameters = f.readline().split(",")

    return parameters


def get_parameter_links(filename_path: Union[str, Path]) -> Dict[str, int]:
    """

    Parameters
    ----------
    filename_path

    Returns
    -------

    """

    recorded_parameters = get_recorded_parameters(str(filename_path))

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
    filename_path: Union[str, Path]
) -> Tuple[Dict[str, Dict[str, List[str]]], int, int, int]:
    """

    Parameters
    ----------
    filename_path

    Returns
    -------

    """

    parameter_links = get_parameter_links(filename_path)

    sensor_links = parameter_links.copy()
    for key in ["time", "group_id", "x_coord", "y_coord"]:
        del sensor_links[key]

    team_infos = {}
    t = []

    if parameter_links["group_id"] is not None:
        with open(str(filename_path), "r") as f:
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
        with open(str(filename_path), "r") as f:
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
    team_infos: Dict[str, Dict[str, List[str]]], mapped_id: str = "name"
) -> Dict[str, Dict[str, int]]:
    """

    Parameters
    ----------
    team_infos
    mapped_id

    Returns
    -------

    """

    links = {}
    for gID in team_infos:
        links.update(
            {gID: {ID: xID for xID, ID in enumerate(team_infos[gID][mapped_id])}}
        )

    return links


def read_kinexon_file(
    filename_path: Union[str, Path], mapped_id: str = None
) -> Tuple[Dict[str, Dict[str, List[str]]], XY]:
    """

    Parameters
    ----------
    filename_path
    mapped_id

    Returns
    -------

    """

    team_infos, number_of_frames, frame_rate, t_null = get_meta_data(filename_path)

    links = create_links_from_meta_data(team_infos, mapped_id="name")
    parameter_links = get_parameter_links(filename_path)

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

    with open(str(filename_path), "r") as f:
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


# filename_path = "C:\\Users\\ke6564\\Desktop\\Studium\\Promotion\\Data\\SSG_3v3.csv"
# team_infos, xy = read_kinexon_file(filename_path)

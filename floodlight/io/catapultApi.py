import requests
import numpy as np
from floodlight.core.xy import XY
from typing import List, Dict, Tuple, Union
from pathlib import Path
import warnings
import csv
import io

## Custom Errors
class APIRequestError(Exception):
    """Custom error class for API request failures."""
    def __init__(self, message):
        super().__init__(message)

## Utils
def _getAllplayerFormActivity_response(baseUrl: str, api_token: str, activity_id: Union[str, int]) -> requests.Response:
    """
    Fetches data for all players involved in a specific activity.

    Args:
        baseUrl (str): The base URL for the API requests.
        api_token (str): The API token for authentication.
        activity_id (str or int): The unique identifier for the activity to fetch players from.

    Returns:
        requests.Response: The response object containing player data.

    Raises:
        APIRequestError: If the API request fails with a status code other than 200.
    """
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_token}",
        "nulls": "0"
    }
    endpoint = f"activities/{activity_id}/athletes/"
    response = requests.get(baseUrl + endpoint, headers=headers)
    if response.status_code == 200:
        return response
    else:
        raise APIRequestError(f"API request failed with status code {response.status_code}")

def getResponse_data_json_list(baseUrl: str, api_token: str, activity_id: Union[str, int]) -> List[Dict]:
    """
    Retrieves sensor data for each player involved in a specific activity.

    Args:
        baseUrl (str): The base URL for the API requests.
        api_token (str): The API token for authentication.
        activity_id (str or int): The unique identifier for the activity.

    Returns:
        List[Dict]: A list of JSON data responses for each player, containing sensor data.

    Raises:
        APIRequestError: If any API request fails with a status code other than 200.
    """
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_token}",
        "nulls": "0"
    }
    allPlayersFromActivity_response = _getAllplayerFormActivity_response(baseUrl, api_token, activity_id)
    allPlayersFromActivity_JSON = allPlayersFromActivity_response.json()
    playerIDsInActivity_list = list(set(i["id"] for i in allPlayersFromActivity_JSON))
    response_data_json_list = []

    for i in playerIDsInActivity_list:
        url = f'activities/{activity_id}/athletes/{i}/sensor'
        response_data_json_i = requests.get(baseUrl + url, headers=headers)
        if response_data_json_i.status_code == 200:
            response_data_json_list.append(response_data_json_i.json())
        else:
            raise APIRequestError(f"API request failed with status code {response_data_json_i.status_code}")

    return response_data_json_list

def json_data_to_csv(json_data_lists: List[Dict]) -> str:
    """
    Converts a list of JSON data into a CSV formatted string.

    Args:
        json_data_lists (List[Dict]): A list of JSON data lists, each containing sensor data for an athlete.

    Returns:
        str: A string containing the CSV-formatted data.
    """
    csv_data = []
    output = io.StringIO()

    for json_data_list in json_data_lists:
        for entry in json_data_list:
            base_info = {
                "athlete_id": entry["athlete_id"],
                "device_id": entry["device_id"],
                "stream_type": entry["stream_type"],
                "player_id": entry["player_id"],
                "name": f"{entry['athlete_first_name']} {entry['athlete_last_name']}",
                "jersey": entry["jersey"],
                "team_id": entry["team_id"],
                "team_name": entry["team_name"]
            }
            for data_entry in entry["data"]:
                row = {
                    "ts": data_entry["ts"],
                    "cs": data_entry["cs"],
                    "lat": data_entry.get("lat", ""),
                    "long": data_entry.get("long", ""),
                    "o": data_entry.get("o", ""),
                    "v": data_entry.get("v", ""),
                    "a": data_entry.get("a", ""),
                    "hr": data_entry.get("hr", ""),
                    "pl": data_entry.get("pl", ""),
                    "mp": data_entry.get("mp", ""),
                    "sl": data_entry.get("sl", ""),
                    "x": data_entry["x"],
                    "y": data_entry["y"]
                }
                row.update(base_info)
                csv_data.append(row)

    header = [
        "athlete_id", "device_id", "stream_type", "player_id", 
        "name", "jersey", "team_id", "team_name", "ts","cs", 
        "lat", "long", "o", "v", "a", "hr", "pl", "mp", "sl", "x", "y"
    ]

    writer = csv.DictWriter(output, fieldnames=header)
    writer.writeheader()
    for row in csv_data:
        writer.writerow(row)
    csv_content_in_memory = output.getvalue()
    output.close()

    return csv_content_in_memory


def get_column_names_from_csv(csv_content: str) -> List[str]:
    """
    Extracts column names from a CSV content string.

    Args:
        csv_content (str): A string containing CSV-formatted data.

    Returns:
        List[str]: A list of column names extracted from the CSV data.
    """
    csv_file = io.StringIO(csv_content)
    columns = csv_file.readline().strip().split(",")
    return columns


def _get_column_links(csv_content: str) -> Union[None, Dict[str, int]]:
    """
    Maps column names from the CSV content to internal names and indexes.

    Args:
        csv_content (str): A string containing CSV-formatted data.

    Returns:
        Union[None, Dict[str, int]]: A dictionary mapping internal names to column indexes, 
                                     or None if critical columns are missing.
    """
    recorded_columns = get_column_names_from_csv(csv_content)
    mapping = {
        "ts": "time",
        "device id": "sensor_id",
        "mapped id": "mapped_id",
        "name": "name",
        "jersey": "number",
        "team_id": "group_id",
        "team_name": "group_name",
        "x": "x_coord",
        "y": "y_coord",
    }

    necessary_columns = ["time", "x_coord", "y_coord"]
    column_links = {mapping[key]: recorded_columns.index(key) for key in mapping if key in recorded_columns}

    if not all(column in column_links for column in necessary_columns):
        warnings.warn("Data file lacks critical information! No timestamp or coordinates found.")
        return None

    return column_links


def _get_group_id(
    recorded_group_identifier: List[str],
    column_links: Dict[str, int],
    single_line: List[str],
) -> Union[str, None]:
    """
    Extracts the group ID from a single line of CSV data.

    Args:
        recorded_group_identifier (List[str]): List of group identifiers available in the data.
        column_links (Dict[str, int]): A dictionary mapping internal column names to CSV column indexes.
        single_line (List[str]): A list of values representing a single line of CSV data.

    Returns:
        Union[str, None]: The group ID, or '0' if no group identifier is recorded.
    """
    if recorded_group_identifier:
        group_identifier = recorded_group_identifier[0]
        group_id = single_line[column_links[group_identifier]]
    else:
        group_id = "0"

    return group_id


def get_meta_data(csv_content: str) -> Tuple[Dict[str, Dict[str, List[str]]], int, int, int]:
    """
    Extracts metadata from the CSV content, including player IDs, frame rate, and number of frames.

    Args:
        csv_content (str): A string containing CSV-formatted data.

    Returns:
        Tuple[Dict[str, Dict[str, List[str]]], int, int, int]: A tuple containing:
            - A dictionary of player IDs by group.
            - The number of frames.
            - The frame rate.
            - The initial timestamp (t_null).
    """
    column_links = _get_column_links(csv_content)
    if not column_links:
        return {}, 0, 0, 0

    sensor_identifier = {"name", "number", "sensor_id", "mapped_id"}
    recorded_sensor_identifier = list(set(column_links) & sensor_identifier)
    sensor_links = {key: index for key, index in column_links.items() if key in recorded_sensor_identifier}
    group_identifier_set = {"group_id", "group_name"}
    recorded_group_identifier = list(set(column_links) & group_identifier_set)

    pID_dict = {}
    t = []

    csv_file = io.StringIO(csv_content)
    csv_file.readline()  # Skip the header

    while True:
        line_string = csv_file.readline()
        if not line_string:
            break

        line = line_string.strip().split(",")
        t.append(int(line[column_links["time"]]))
        group_id = _get_group_id(recorded_group_identifier, column_links, line)

        if group_id not in pID_dict:
            pID_dict[group_id] = {}

        for identifier in sensor_links:
            if identifier not in pID_dict[group_id]:
                pID_dict[group_id][identifier] = []
            if line[column_links[identifier]] not in pID_dict[group_id][identifier]:
                pID_dict[group_id][identifier].append(line[column_links[identifier]])

    timestamps = sorted(set(t))
    minimum_time_step = np.min(np.diff(timestamps))
    # timestamps are in milliseconds. Magic number 1000 is needed for conversion to
    # seconds.
    framerate = 1000 / minimum_time_step

    if not framerate.is_integer():
        warnings.warn(f"Non-integer frame rate: {framerate}. Rounding to nearest integer.")
        framerate = int(framerate)

    number_of_frames = int((timestamps[-1] - timestamps[0]) / (1000 / framerate))
    t_null = timestamps[0]

    return pID_dict, number_of_frames, framerate, t_null


def _get_available_sensor_identifier(pID_dict: Dict[str, Dict[str, List[str]]]) -> str:
    """
    Determines an available sensor identifier from the metadata.

    Args:
        pID_dict (Dict[str, Dict[str, List[str]]]): Metadata dictionary containing player IDs by group.

    Returns:
        str: The available sensor identifier.
    """
    player_identifiers = ["name", "mapped_id", "sensor_id", "number"]
    available_identifier = [idt for idt in player_identifiers if idt in list(pID_dict.values())[0]]
    return available_identifier[0]


def create_links_from_meta_data(pID_dict: Dict[str, Dict[str, List[str]]], identifier: str = None) -> Dict[str, Dict[str, int]]:
    """
    Creates a mapping of player IDs to numerical indices based on the metadata.

    Args:
        pID_dict (Dict[str, Dict[str, List[str]]]): Metadata dictionary containing player IDs by group.
        identifier (str, optional): The sensor identifier to use for mapping. Defaults to the available identifier.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary mapping group IDs to player ID indices.
    """
    if identifier is None:
        identifier = _get_available_sensor_identifier(pID_dict)

    links = {}
    for group_id in pID_dict:
        links[group_id] = {ID: xID for xID, ID in enumerate(pID_dict[group_id][identifier])}

    return links


def read_position_data_csv(csv_content: str) -> List[XY]:
    """
    Reads position data from CSV content and converts it to a list of XY objects.

    Args:
        csv_content (str): A string containing CSV-formatted position data.

    Returns:
        List[XY]: A list of XY objects representing the position data for each group.
    """
    pID_dict, number_of_frames, framerate, t_null = get_meta_data(csv_content)
    if not pID_dict:
        return []

    links = create_links_from_meta_data(pID_dict)
    column_links = _get_column_links(csv_content)
    group_identifier_set = {"group_id", "group_name"}
    recorded_group_identifier = list(set(column_links) & group_identifier_set)

    identifier = _get_available_sensor_identifier(pID_dict)

    number_of_sensors = {}
    xydata = {}
    for group in links:
        number_of_sensors[group] = len(links[group])
        xydata[group] = np.full([number_of_frames + 1, number_of_sensors[group] * 2], np.nan)

    csv_file = io.StringIO(csv_content)
    csv_file.readline()  # Skip the header

    while True:
        line_string = csv_file.readline()
        if not line_string:
            break
        
        line = line_string.strip().split(",")
        t = int(line[column_links["time"]])
        frame_idx = round((t - t_null) * framerate / 1000)

        group_id = _get_group_id(recorded_group_identifier, column_links, line)
        if group_id not in xydata:
            continue

        for idx, sensor_id in enumerate(links[group_id]):
            x = line[column_links["x_coord"]]
            y = line[column_links["y_coord"]]

            try:
                x = float(x)
                y = float(y)
            except ValueError:
                continue

            sensor_idx = links[group_id][sensor_id]
            xydata[group_id][frame_idx, sensor_idx * 2] = x
            xydata[group_id][frame_idx, sensor_idx * 2 + 1] = y

    return [XY(xydata[group], framerate=framerate) for group in xydata]

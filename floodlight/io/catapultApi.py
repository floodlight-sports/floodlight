import requests
import numpy as np
from floodlight.core.xy import XY
from typing import List, Dict, Tuple, Union, Any
from pathlib import Path
import warnings
import csv
import io
import json
import os

# for testing: so that data has not to be downloaded every run

def saveResponse(response: requests.Response, endpoint: str) -> None:
    """
    Saves the response from an API request to a JSON file.

    Args:
        response (requests.Response): The response object from the API request.
        endpoint (str): The endpoint to determine the directory for saving the file.

    Returns:
        None
    """
    if response.status_code == 200:
        response_data = response.json()[0]
        os.makedirs(endpoint, exist_ok=True)
        with open(os.path.join(endpoint, "response.json"), "w") as json_file:
            json.dump(response_data, json_file, indent=4)
            print(f"API response successfully saved in '{os.path.join(endpoint, 'response.json')}'")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def load_responses(endpoint: str) -> List[Dict]:
    """
    Loads and returns a list of JSON data responses from the specified directory.

    Args:
        endpoint (str): The directory containing the JSON files.

    Returns:
        List[Dict]: A list of JSON data responses.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    response_data_json_list = []
    
    # Check if the endpoint directory exists
    if os.path.exists(endpoint):
        # Walk through all directories and files in the given directory
        for root, dirs, files in os.walk(endpoint):
            for filename in files:
                if filename.endswith(".json"):
                    file_path = os.path.join(root, filename)
                    
                    # Load the JSON file and append the data to the list
                    with open(file_path, "r") as json_file:
                        data = json.load(json_file)
                        response_data_json_list.append(data)
    else:
        raise FileNotFoundError(f"Directory {endpoint} does not exist.")
    
    return response_data_json_list

## for testing end 

# Custom Errors
class APIRequestError(Exception):
    """
    Custom error class for API request failures.
    """
    def __init__(self, message: str):
        super().__init__(message)

def _getAllplayerFormActivity_response(baseUrl: str, api_token: str, activity_id: str) -> requests.Response:
    """
    Fetches data for all players involved in a specific activity.

    Args:
        baseUrl (str): The base URL for the API requests.
        api_token (str): The API token for authentication.
        activity_id (str): The unique identifier for the activity to fetch players from.

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

def getResponse_data_json_list(baseUrl: str, api_token: str, activity_id: str) -> List[Dict]:
    """
    Retrieves sensor data for each player involved in a specific activity.

    Args:
        baseUrl (str): The base URL for the API requests.
        api_token (str): The API token for authentication.
        activity_id (str): The unique identifier for the activity.

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
            response_data_json_list.append(response_data_json_i.json()[0])
            saveResponse(response_data_json_i, url)
        else:
            raise APIRequestError(f"API request failed with status code {response_data_json_i.status_code}")

    return response_data_json_list

def get_key_names_from_json(response_data_json_list: List[Dict]) -> set:
    """
    Extracts unique column names from a list of JSON data responses.

    Args:
        response_data_json_list (List[Dict]): List of JSON data responses.

    Returns:
        set: A set of unique column names.
    """
    recorded_columns = set()

    for data_i in response_data_json_list:
        recorded_columns.update(data_i.keys())
        if "data" in data_i and isinstance(data_i["data"], list) and data_i["data"]:
            recorded_columns.update(data_i["data"][0].keys())

    return recorded_columns

def _get_key_links(response_data_json_list: List[Dict]) -> Union[Dict[str, int], None]:
    """
    Generates a mapping of key names to their indices in the JSON data responses.

    Args:
        response_data_json_list (List[Dict]): List of JSON data responses.

    Returns:
        Union[Dict[str, int], None]: A dictionary mapping key names to their indices or None if critical keys are missing.
    """
    for data_i in response_data_json_list:
        data_i['name'] = f"{data_i['athlete_first_name']} {data_i['athlete_last_name']}"

    key_names = list(get_key_names_from_json(response_data_json_list))

    mapping = {
        "ts": "time",
        "device id": "sensor_id",
        "athlete_id": "mapped_id",
        "athlete_first_name": "athlete_first_name",
        "athlete_last_name": "athlete_last_name",
        "stream_type": "stream_type",
        "name": "name",
        "jersey": "number",
        "team_id": "group_id",
        "team_name": "group_name",
        "x": "x_coord",
        "y": "y_coord",
        "cs": "cs",
        "lat": "lat",
        "long": "long",
        "o": "o",
        "v": "v",
        "a": "a",
        "hr": "hr",
        "pl": "pl",
        "mp": "mp",
        "sl": "sl",
    }

    necessary_keys = ["time", "x_coord", "y_coord"]
    key_links = {}
    for key in mapping:
        if key in key_names:
            key_links.update({mapping[key]: key_names.index(key)})

    if not all(columns in key_links for columns in necessary_keys):
        warnings.warn("Data file lacks critical information! No timestamp or coordinates found.")
        return None

    return key_links

def get_meta_data(response_data_json_list: List[Dict]) -> Tuple[Dict[str, Dict[str, List[str]]], int, int, int]:
    """
    Extracts metadata from the JSON data responses.

    Args:
        response_data_json_list (List[Dict]): List of JSON data responses.

    Returns:
        Tuple[Dict[str, Dict[str, List[str]]], int, int, int]: 
            - Dictionary mapping group IDs to player identifiers.
            - Number of frames.
            - Framerate.
            - Null timestamp.
    """
    framerate = 10

    key_links = _get_key_links(response_data_json_list) # should i do this with get_key_names_from_json() and a mapping,
                                                        # since the key_links are not beeing used as such 
                                                        # (beacause of how the json file looks #{key1, key2, key3 ... data[key8,key9,...]}) 
    sensor_identifier = {"name", "number", "sensor_id", "mapped_id"}
    key_links_set = set(key_links)

    recorded_sensor_identifier = list(key_links_set & sensor_identifier)
    sensor_links = {
        key: index
        for (key, index) in key_links.items()
        if key in recorded_sensor_identifier
    }
    group_identifier_set = {"group_id", "group_name"}
    recorded_group_identifier = list(key_links_set & group_identifier_set)
    pID_dict = {}
    t = []
    has_groups = len(recorded_group_identifier) > 0
    if not has_groups:
        warnings.warn("Since no group exist in data, dummy group '0' is created!")

    reversed_group_mapping = {
        "group_id": "team_id",
        "group_name": "team_name",
    }

    reversed_sensor_mapping = {
        "sensor_id": "device id",
        "mapped_id": "athlete_id",
        "name": "name",
        "number": "jersey",
    }

    for data_i in response_data_json_list:
        for entry in data_i["data"]:
            t.append(int(entry["ts"]) * 100 + int(entry["cs"]))

        group_identifier = recorded_group_identifier[0]
        group_id = data_i[reversed_group_mapping[group_identifier]]# not doing it with key_liks beacause of how json looks like: 
                                                                        #{key1, key2, key3 ... data[key8,key9,...]}

        if group_id not in pID_dict:
            pID_dict.update({group_id: {}})
        for identifier in sensor_links:
            if identifier not in pID_dict[group_id]:
                pID_dict[group_id].update({identifier: []})
            if data_i[reversed_sensor_mapping[identifier]] not in pID_dict[group_id][identifier]:
                pID_dict[group_id][identifier].append(data_i[reversed_sensor_mapping[identifier]])

    pID_dict = dict(sorted(pID_dict.items()))

    timestamps_cs = list(set(t))
    timestamps_cs.sort()

    number_of_frames = int((timestamps_cs[-1] - timestamps_cs[0]) / (100 / framerate))
    t_null = timestamps_cs[0]

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

def create_links_from_meta_data(
    pID_dict: Dict[str, Dict[str, List[str]]], identifier: str = None
) -> Dict[str, Dict[str, int]]:
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

def read_position_data_from_json(response_data_json_list: List[Dict], coord_format: str = "xy") -> List[XY]:
    """
    Reads position data from JSON data responses and returns a list of XY objects.

    Args:
        response_data_json_list (List[Dict]): List of JSON data responses.
        coord_format (str, optional): The coordinate format ["xy", "latlong"]. Defaults to "xy".

    Returns:
        List[XY]: A list of XY objects containing position data.
    """
    pID_dict, number_of_frames, framerate, t_null = get_meta_data(response_data_json_list)

    links = create_links_from_meta_data(pID_dict)

    key_links = _get_key_links(response_data_json_list)
    key_links_set = set(key_links)
    group_identifier_set = {"group_id", "group_name"}
    recorded_group_identifier = list(key_links_set & group_identifier_set)
    identifier = _get_available_sensor_identifier(pID_dict)

    reversed_group_mapping = {
        "group_id": "team_id",
        "group_name": "team_name",
    }
    number_of_sensors = {}
    xydata = {}

    for group in links:
        number_of_sensors.update({group: len(links[group])})
        xydata.update({group: np.full([number_of_frames + 1, number_of_sensors[group] * 2], np.nan)})

    for data_i in response_data_json_list:
        for entry in data_i["data"]:
            if coord_format == "xy":
                x_coordinate = entry["x"] / 10  # convert mm to cm
                y_coordinate = entry["y"] / 10  # convert mm to cm
            elif coord_format == "latlong":
                x_coordinate = entry["lat"]
                y_coordinate = entry["long"]
            else:
                raise ValueError(f"Expected coordinate format to be ['xy', 'latlong'], but got {coord_format} instead.")

            group_identifier = recorded_group_identifier[0]
            group_id = data_i[reversed_group_mapping[group_identifier]] # not doing it with key_liks beacause of how json looks like: 
                                                                        #{key1, key2, key3 ... data[key8,key9,...]}

            x_col = links[group_id][data_i[identifier]] * 2
            y_col = x_col + 1

            row = int((int(entry["ts"]) * 100 + int(entry["cs"]) - t_null) / (100 / framerate))

            if x_coordinate != "":
                xydata[group_id][row, x_col] = x_coordinate

            if y_coordinate != "":
                xydata[group_id][row, y_col] = y_coordinate

    data_objects = []

    for group_id in xydata:
        data_objects.append(XY(xy=xydata[group_id], framerate=framerate))

    return data_objects

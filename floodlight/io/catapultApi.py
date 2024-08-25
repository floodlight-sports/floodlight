import requests
import numpy as np
from floodlight.core.xy import XY
from typing import List, Dict, Tuple, Union, Any
from pathlib import Path
import warnings
import json
import os
from collections import Counter


class APIRequestError(Exception):
    """
    Custom error class for API request failures.

    Parameters
    ----------
    message : str
        Error message to be displayed.
    """
    def __init__(self, message: str):
        super().__init__(message)

def get_mappings() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Returns dictionaries containing mappings between various key names.

    This function provides three mappings:
    1. A general mapping of key names to their corresponding new names.
    2. A reversed mapping for group-related keys.
    3. A reversed mapping for sensor-related keys.

    Returns
    -------
    mapping : dict
        A dictionary that maps original key names to their corresponding new key names.
        The keys represent the original key names, and the values represent the new key names.
    
    reversed_group_mapping : dict
        A dictionary that maps new key names for group-related fields back to their original key names.
        The keys represent the new key names, and the values represent the original key names.
    
    reversed_sensor_mapping : dict
        A dictionary that maps new key names for sensor-related fields back to their original key names.
        The keys represent the new key names, and the values represent the original key names.
    """
    mapping = {
        "ts": "time",
        "device id": "sensor_id",
        "athlete_id": "mapped_id",
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
    }

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
    return mapping, reversed_group_mapping, reversed_sensor_mapping

def write_response(response: requests.Response, endpoint: str) -> None:
    """
    Saves the response from an API request to a JSON file.

    Parameters
    ----------
    response : requests.Response
        The response object from the API request.
    endpoint : str
        The endpoint to determine the directory for saving the file.

    Returns
    -------
    None
    """
    response_ok = 200
    if response.status_code == response_ok:
        response_data = response.json()[0]
        os.makedirs(endpoint, exist_ok=True)
        with open(os.path.join(endpoint, "response.json"), "w") as json_file:
            json.dump(response_data, json_file, indent=4)
            print(f"API response successfully saved in '{os.path.join(endpoint, 'response.json')}'")
    else:
        raise ValueError(f"Expected status code 200, but got {response.status_code}. Response text: {response.text}")

def read_response(endpoint: str) -> List[Dict]:
    """
    Loads and returns a list of response_data from the specified directory.

    Parameters
    ----------
    endpoint : str
        The directory containing the JSON files.

    Returns
    -------
    List[Dict]
        A list of response_data.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.

    Note
    ----
    Ensure that this directory only contains the JSON files from the specific request in question, as unrelated .json files may cause issues.
    """
    response_data = []

    if os.path.exists(endpoint):
        json_files = [os.path.join(root, f) for root, _, files in os.walk(endpoint) for f in files if f.endswith(".json")]
        for file_path in json_files:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                response_data.append(data)
    
    return response_data


def get_response(base_url: str, api_token: str, activity_id: str) -> Tuple[List[Dict], str]:
    """
    Fetches data for all players involved in a specific activity.

    Parameters
    ----------
    base_url : str
        The base URL for the API requests.
        Ensure you use the URL corresponding to the API for which you have credentials.
        For example, to access the European API, use "https://connect-eu.catapultsports.com/api/v6/".

    api_token : str
        The API token for authentication.
    activity_id : str
        The unique identifier for the activity to fetch players from. 
        Refer to the Catapult API documentation for details on how to obtain this ID:
            https://docs.connect.catapultsports.com/reference/introduction.

    Returns
    -------
    Tuple[List[Dict], str]
        - List of dictionaries containing player data.
        - Endpoint string used for the API request.

    Raises
    ------
    APIRequestError
        If the API request fails with a status code other than 200.
    """
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_token}",
        "nulls": "0"
    }
    endpoint = f"activities/{activity_id}/athletes/"
    response = requests.get(base_url + endpoint, headers=headers)
    response_ok =200
    if response.status_code == response_ok:
        response_data = response.json()
        
        return response_data, endpoint
    else:
        raise APIRequestError(f"API request failed with status code {response.status_code}")

def get_response_data_dict_list(base_url: str, api_token: str, activity_id: str, response: List[Dict]) -> List[Dict]:
    """
    Retrieves sensor data for each player involved in a specific activity.

    Parameters
    ----------
    base_url : str
        The base URL for the API requests.
        Ensure you use the URL corresponding to the API for which you have credentials.
        For example, to access the European API, use "https://connect-eu.catapultsports.com/api/v6/".
    api_token : str
        The API token for authentication.
    activity_id : str
        The unique identifier for the activity.
    response : List[Dict]
        The response containing player data.

    Returns
    -------
    List[Dict]
        A list of response_data for each player, containing sensor data.

    Raises
    ------
    APIRequestError
        If any API request fails with a status code other than 200.
    """
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_token}",
        "nulls": "0"
    }

    player_ids = sorted(set(i["id"] for i in response))
    response_data_dict_list = []

    response_ok =200
    for i in player_ids:
        url = f'activities/{activity_id}/athletes/{i}/sensor'
        player_data = requests.get(base_url + url, headers=headers)
        
        if player_data.status_code == response_ok:
            response_data_dict_list.append(player_data.json()[0])
        else:
            raise APIRequestError(f"API request failed with status code {player_data.status_code}")

    return response_data_dict_list

def get_key_names_from_dict_list(response_data_dict_list: List[Dict]) -> set:
    """
    Extracts unique key names from a list of response_data.

    Parameters
    ----------
    response_data_dict_list : List[Dict]
        List of response_data.

    Returns
    -------
    set
        A set of unique key names.
    """
    
    mapping, _, _ = get_mappings()

    recorded_keys = set()

    for athlete_entry in response_data_dict_list:
        recorded_keys.update(key for key in athlete_entry.keys() if key in mapping)
        if len(athlete_entry["data"]) != 0:
            recorded_keys.update(key for key in athlete_entry["data"][0].keys() if key in mapping)

    return recorded_keys

def get_meta_data(response_data_dict_list: List[Dict], coord_format: str = "xy") -> Tuple[Dict[str, Dict[str, List[str]]], int, int, int]:
    """
    Extracts metadata from the response_data.

    Parameters
    ----------
    response_data_dict_list : List[Dict]
        List of response_data.

    Returns
    -------
    Tuple[Dict[str, Dict[str, List[str]]], int, int, int]
        - Dictionary mapping group IDs to player identifiers.
        - Number of frames.
        - Framerate.
        - Null timestamp.
    """

    mapping, reversed_group_mapping, reversed_sensor_mapping = get_mappings()
    
    key_names = list(get_key_names_from_dict_list(response_data_dict_list))
    key_names_mapped = [mapping[i] for i in key_names]
    sensor_identifier = {"name", "number", "sensor_id", "mapped_id"}
    key_names_mapped_set = set(key_names_mapped)

    recorded_sensor_identifier = list(key_names_mapped_set & sensor_identifier)
    sensor_links = {
        key: index
        for (index,key) in enumerate(key_names_mapped) #.items()
        if key in recorded_sensor_identifier
    }
    group_identifier_set = {"group_id", "group_name"}
    recorded_group_identifier = list(key_names_mapped_set & group_identifier_set)
    pID_dict = {}
    t = []
    has_groups = len(recorded_group_identifier) > 0
    if not has_groups:
        warnings.warn("Since no group exist in data, dummy group '0' is created!")

    max_count = 0
    for athlete_entry in response_data_dict_list:
        athlete_t = [int(entry["ts"]) for entry in athlete_entry["data"]]
        t.extend([int(entry["ts"]) * 100 + int(entry["cs"]) for entry in athlete_entry["data"]]) # timestamps are in centiseconds. Magic number 100 is needed for conversion to centiseconds
    
        timestamps_athlete_count = Counter(athlete_t)
    
        if timestamps_athlete_count:
            max_count_athlete = max(timestamps_athlete_count.values())
    
        group_identifier = recorded_group_identifier[0]
        group_id = athlete_entry[reversed_group_mapping[group_identifier]]
    
        if group_id not in pID_dict:
            pID_dict[group_id] = {}
    
        for identifier in sensor_links:
            if identifier not in pID_dict[group_id]:
                pID_dict[group_id][identifier] = []
            if athlete_entry[reversed_sensor_mapping[identifier]] not in pID_dict[group_id][identifier]:
                pID_dict[group_id][identifier].append(athlete_entry[reversed_sensor_mapping[identifier]])
    
    framerate = max(max_count, max_count_athlete)
    pID_dict = dict(sorted(pID_dict.items()))

    timestamps_cs = list(set(t))
    timestamps_cs.sort()

    number_of_frames = int((timestamps_cs[-1] - timestamps_cs[0]) / (100 / framerate)) # timestamps are in centiseconds. Magic number 100 is needed for conversion to
    t_null = timestamps_cs[0]

    return pID_dict, number_of_frames, framerate, t_null

def _get_available_sensor_identifier(pID_dict: Dict[str, Dict[str, List[str]]]) -> str:
    """
    Determines an available sensor identifier from the metadata.

    Parameters
    ----------
    pID_dict : Dict[str, Dict[str, List[str]]]
        Metadata dictionary containing player IDs by group.

    Returns
    -------
    str
        The available sensor identifier.
    """
    player_identifiers = ["name", "mapped_id", "sensor_id", "number"]
    available_identifier = [idt for idt in player_identifiers if idt in list(pID_dict.values())[0]]
    return available_identifier[0]

def create_links_from_meta_data(
    pID_dict: Dict[str, Dict[str, List[str]]], identifier: str = None
) -> Dict[str, Dict[str, int]]:
    """
    Creates a mapping of player IDs to numerical indices based on the metadata.

    Parameters
    ----------
    pID_dict : Dict[str, Dict[str, List[str]]]
        Metadata dictionary containing player IDs by group.
    identifier : str, optional
        The sensor identifier to use for mapping. Defaults to the available identifier.

    Returns
    -------
    Dict[str, Dict[str, int]]
        A dictionary mapping group IDs to player ID indices.
    """
    if identifier is None:
        identifier = _get_available_sensor_identifier(pID_dict)

    links = {}
    for group_id in pID_dict:
        links[group_id] = {ID: xID for xID, ID in enumerate(pID_dict[group_id][identifier])}

    return links

def read_position_data_from_dict_list(response_data_dict_list: List[Dict], coord_format: str = "xy") -> List[XY]:
    """
    Reads position data from responses data and returns a list of XY objects.

    Parameters
    ----------
    response_data_dict_list : List[Dict]
        List of response_data.
    coord_format : str, optional
        The coordinate format ["xy", "latlong"]. Defaults to "xy".

    Returns
    -------
    List[XY]
        A list of XY objects containing position data.
    
    Raises
    ------
    ValueError
        If an invalid coordinate format is specified.
    """
    mapping, reversed_group_mapping, reversed_sensor_mapping = get_mappings()

    pID_dict, number_of_frames, framerate, t_null = get_meta_data(response_data_dict_list)

    links = create_links_from_meta_data(pID_dict)

    key_names = get_key_names_from_dict_list(response_data_dict_list)
    key_names_mapped = [mapping[i] for i in list(key_names)]
    key_links_set = set(key_names_mapped)
    group_identifier_set = {"group_id", "group_name"}
    recorded_group_identifier = list(key_links_set & group_identifier_set)
    identifier = _get_available_sensor_identifier(pID_dict)

    number_of_sensors = {}
    xydata = {}

    for group in links:
        number_of_sensors.update({group: len(links[group])})
        xydata.update({group: np.full([number_of_frames + 1, number_of_sensors[group] * 2], np.nan)})

    for athlete_entry in response_data_dict_list:
        for entry in athlete_entry["data"]:
            if coord_format == "xy":
                x_coordinate = entry["x"] 
                y_coordinate = entry["y"] 
            elif coord_format == "latlong":
                x_coordinate = entry["lat"]
                y_coordinate = entry["long"]
            else:
                raise ValueError(f"Expected coordinate format to be ['xy', 'latlong'], but got {coord_format} instead.")

            group_identifier = recorded_group_identifier[0]
            group_id = athlete_entry[reversed_group_mapping[group_identifier]]

            x_col = links[group_id][athlete_entry[reversed_sensor_mapping[identifier]]] * 2
            y_col = x_col + 1

            row = int((int(entry["ts"]) * 100 + int(entry["cs"]) - t_null) / (100 / framerate)) # timestamps are in centiseconds. Magic number 100 is needed for conversion to

            if x_coordinate != "":
                xydata[group_id][row, x_col] = x_coordinate

            if y_coordinate != "":
                xydata[group_id][row, y_col] = y_coordinate

    data_objects = []

    for group_id in xydata:
        data_objects.append(XY(xy=xydata[group_id], framerate=framerate))

    return data_objects

def read_position_data_from_activity(base_url: str, api_token: str, activity_id: str, save: bool = False, pathToRespone: str = None, coord_format: str = "xy") -> List[XY]:
    """
    Reads position data for a specific activity either from a saved response or by making an API request.

    Parameters
    ----------
    base_url : str
        The base URL for the API requests.
        Ensure you use the URL corresponding to the API for which you have credentials.
        For example, to access the European API, use "https://connect-eu.catapultsports.com/api/v6/".
    api_token : str
        The API token for authentication.
    activity_id : str
        The unique identifier for the activity.
    save : bool, optional
        Whether to save the response to a file. Defaults to False.
    pathToRespone : str, optional
        The path to the saved response directory. If provided, data will be read from this path instead of making an API request.
    coord_format : str, optional
        The format of the coordinates to be returned (e.g., "xy"). Defaults to "xy".

    Returns
    -------
    List[XY]
        A list of XY objects containing position data.

    Conditional Behavior
    --------------------
    1. If `pathToRespone` is provided:
       - The function reads the response data from the specified file path, skipping any API requests.

    2. If `pathToRespone` is not provided:
       - The function makes an API request to fetch the data.
       - If `save` is set to `True`, the fetched response is saved to a file.
       - If `save` is `False`, the response data is processed directly without saving.
    
    """
    if pathToRespone: 
        response_data_dict_list = read_response(pathToRespone)
    else:
        response, endpoint = get_response(base_url, api_token, activity_id)
        if save:
            write_response(response, endpoint)
        else:
            response_data_dict_list = get_response_data_dict_list(base_url, api_token, activity_id, response)
    
    data_objects = read_position_data_from_dict_list(response_data_dict_list, coord_format)

    return data_objects

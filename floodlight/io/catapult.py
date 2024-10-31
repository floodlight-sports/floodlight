import os
import json
import warnings
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
import requests
import pickle

from floodlight.core.xy import XY
from floodlight.utils.exceptions import APIRequestError


def _get_mappings() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Returns dictionaries containing mappings between various key names.

    1. **Mapping**: A mapping of key names to their corresponding new names.
    2. **Reversed Group Mapping**: A reversed mapping for group-related keys.
    3. **Reversed Sensor Mapping**: A reversed mapping for sensor-related keys.

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


def dump_list_of_dicts(data: List[Dict], file_path: str, file_format: str) -> None:
    """
    Serializes a list of dictionaries into a file in the specified format (pickle or JSON).

    Parameters
    ----------
    data : List[Dict]
        The list of dictionaries to be serialized.
    file_path : str
        The file path where the file will be saved.
    file_format : str
        The format to save the data in. Options are 'pickle' or 'json'.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the specified file_format is not 'pickle' or 'json'.
    """

    # Check if the file format is 'pickle'
    if file_format == "pickle":
        # Open the file in binary write mode and serialize data using pickle
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
            print(f"Data successfully serialized to '{file_path}' as pickle.")

    # Check if the file format is 'json'
    elif file_format == "json":
        # Open the file in text write mode and serialize data using JSON with indentation for readability
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
            print(f"Data successfully serialized to '{file_path}' as JSON.")

    # Raise an error if an unsupported file format is provided
    else:
        raise ValueError("Invalid file format specified. Use 'pickle' or 'json'.")


def load_list_of_dicts(file_path: str) -> List[Dict]:
    """
    Deserializes a list of dictionaries from a file. The file format is determined by its extension
    (.pkl for pickle files and .json for JSON files).

    Parameters
    ----------
    file_path : str
        The file path of the file to be loaded.

    Returns
    -------
    List[Dict]
        The deserialized list of dictionaries.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is not recognized.
    """

    # Check if the file has a .pkl extension, indicating it's a pickle file
    if file_path.endswith(".pkl"):
        try:
            # Open the file in binary read mode and deserialize using pickle
            with open(file_path, "rb") as file:
                data = pickle.load(file)
                print(f"Data successfully loaded from '{file_path}' as pickle.")
                return data
        # Handle case where the file is not found
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' does not exist.")
            return []
        # Handle any pickle-specific errors
        except pickle.PickleError as e:
            print(f"Error: An error occurred while loading the pickle file: {e}")
            return []

    # Check if the file has a .json extension, indicating it's a JSON file
    elif file_path.endswith(".json"):
        try:
            # Open the file in text read mode and deserialize using JSON
            with open(file_path, "r") as file:
                data = json.load(file)
                print(f"Data successfully loaded from '{file_path}' as JSON.")
                return data
        # Handle case where the file is not found
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' does not exist.")
            return []
        # Handle any JSON-specific
        except json.JSONDecodeError as e:
            print(f"Error: An error occurred while loading the JSON file: {e}")
            return []

    # Raise an error if the file extension is not recognized
    else:
        raise ValueError(
            "Unsupported file format. Use '.pkl' for pickle files or '.json' for JSON files."
        )


def get_activity_players_info(
    base_url: str, api_token: str, activity_id: str
) -> Tuple[List[Dict], str]:
    """
    Fetches player data for all players involved in a specific activity.

    Note
    ----
    The player data does not contain position data and looks like the following example:

        {
            "id": "4a9b398a-3d6b-45be-8b69-3f83b3781093",
            "first_name": "Example",
            "last_name": "Player",
            "gender": "Unspecified",
            "jersey": "99y",
            "nickname": "",
            "height": 160,
            "weight": 77,
            "date_of_birth": 74184800,
            "velocity_max": 5,
            "acceleration_max": 0,
            "heart_rate_max": 200,
            "player_load_max": 500,
            "image": "",
            "icon": "circle",
            "stroke_colour": "",
            "fill_colour": "",
            "trail_colour_start": "",
            "trail_colour_end": "",
            "is_synced": 0,
            "is_deleted": 0,
            "created_at": "2010-12-08 11:15:16",
            "modified_at": "2011-10-18 15:15:00",
            "activity_athlete_id": "e8801f56-fdd2-4f82-a581-2a92c734e126",
            "date_of_birth_date": "1999-06-08",
            "tag_list": [
                "Full"
            ],
            "tags": [
                {
                    "id": "dda44591-5120-11ec-b007-06765be6a661",
                    "tag_type_id": "dd974945-5120-11ec-b007-06765be6a661",
                    "name": "Full",
                    "tag_type_name": "Participation",
                    "tag_name": "Full"
                }
            ],
            "position": "CDM",
            "position_id": "a1331be5-d2ad-11e4-b293-22000afc007c",
            "position_name": "Central Defensive Midfielder"
        }

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
        Refer to the Catapult API documentation for details on how to obtain this ID.

    Returns
    -------
    Tuple[List[Dict], str]
        - List of dictionaries containing player data of each player participating in the specified activity.
        - Endpoint string used for the API request.

    Raises
    ------
    APIRequestError
        If the API request fails with a status code other than 200.
    """

    # Set the headers with the required authorization and JSON format
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_token}",
        "nulls": "0",
    }

    # Create the endpoint URL for fetching players associated with the given activity_id
    endpoint = f"activities/{activity_id}/athletes/"
    # Make the GET request to the API
    activity_players_info_response = requests.get(base_url + endpoint, headers=headers)
    response_ok = 200  # Define the success status code

    # Check if the request was successful (status code 200)
    if activity_players_info_response.status_code != response_ok:
        raise APIRequestError(
            message="API request failed",
            status_code=activity_players_info_response.status_code,
            endpoint=endpoint,
            response_text=activity_players_info_response.text
        )    
        
    # Parse the JSON response to get player data
    activity_players_info = activity_players_info_response.json()
    return activity_players_info, endpoint

def get_players_sensor_data_dict_list(
    base_url: str, api_token: str, activity_id: str, activity_players_info: List[Dict]
) -> List[Dict]:
    """
    Retrieves sensor data for each player involved in a specific activity.

    Example
    -------
    An example of the `players_sensor_data_dict_list` looks like:

    {
        "athlete_id": "33ffcf-8b1b-46d6-876c-e8d9bf254a7a",
        "device_id": 50576,
        "stream_type": "gps",
        "player_id": "",
        "athlete_first_name": "Example",
        "athlete_last_name": "Player",
        "jersey": "99",
        "team_id": "33054b55-9900-11e3-b9b6-22000af8166b",
        "team_name": "Example Team",
        "data": [
            {
                "ts": 1697703120,
                "cs": 10,
                "lat": 0,
                "long": 0,
                "o": 0,
                "v": 0,
                "a": 0,
                "hr": 0,
                "pl": 0.01,
                "mp": 0,
                "sl": 0.11,
                "x": 0,
                "y": 0
            },
            {
                "ts": 1697703120,
                "cs": 20,
                "lat": 0,
                ...
            }
        ]
    }

    Parameters
    ----------
    base_url : str
        The base URL for the API requests. Ensure you use the URL corresponding to the API for which you have credentials.
        For example, to access the European API, use "https://connect-eu.catapultsports.com/api/v6/".
    api_token : str
        The API token for authentication.
    activity_id : str
        The unique identifier for the activity.
    activity_players_info : List[Dict]
        The response containing player data.

    Returns
    -------
    List[Dict]
        A list of dictionaries, each containing sensor data for a player.

    Raises
    ------
    APIRequestError
        If any API request fails with a status code other than 200.
    """

    # Setting up headers for the API request
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_token}",
        "nulls": "0",
    }

    # Extracting unique player IDs from the provided activity_players_info, ensuring the list is sorted
    player_ids = sorted(set(i["id"] for i in activity_players_info))
    players_sensor_data_dict_list = []

    response_ok = 200  # Expected HTTP status code for a successful request
    # Loop through each player's ID to fetch their sensor data
    for i in player_ids:
        # Construct the API endpoint for fetching sensor data of the current player
        endpoint = f"activities/{activity_id}/athletes/{i}/sensor"
        player_data_response = requests.get(base_url + endpoint, headers=headers)
        # Check if the request was successful
    
        if player_data_response.status_code != response_ok:
            raise APIRequestError(
                message="API request failed",
                status_code=player_data_response.status_code,
                endpoint=endpoint,
                response_text=player_data_response.text
            )    
        try:
            player_data = player_data_response.json()
            # Check if the response is not empty and contains the expected structure
            if player_data and isinstance(player_data, list) and len(player_data) > 0:
                players_sensor_data_dict_list.append(player_data[0])
            else:
                print(
                    f"Warning: No valid sensor data found for player in activity {activity_id}. Response: {player_data}"
                )
        except (IndexError, KeyError) as e:
            print(f"Error: Issue processing sensor data for activity {activity_id}. Exception: {e}")

    return players_sensor_data_dict_list


def _get_key_names_from_dict_list(players_sensor_data_dict_list: List[Dict]) -> set:
    """
    Extracts unique key names from a list of dictionaries containing sensor data.

    Parameters
    ----------
    players_sensor_data_dict_list : List[Dict]
        A list of dictionaries, each containing sensor data for a player.
        Each dictionary represents a player's sensor data and typically includes
        metadata and a list of data entries, where each data entry is a dictionary
        with various sensor measurements.

    Returns
    -------
    set
        A set of unique key names found in the dictionaries within the list.
        This includes keys from both the main dictionary and the nested data entries.
    """
    mapping, _, _ = _get_mappings()

    recorded_keys = set()

    for athlete_entry in players_sensor_data_dict_list:
        # Update with keys from the main athlete entry dictionary if they are in the mapping
        recorded_keys.update(key for key in athlete_entry.keys() if key in mapping)

        # Update with keys from the nested 'data' entries if they are in the mapping
        if athlete_entry["data"]:
            recorded_keys.update(
                key for key in athlete_entry["data"][0].keys() if key in mapping
            )
    return recorded_keys


def get_meta_data(
    players_sensor_data_dict_list: List[Dict],
) -> Tuple[Dict[str, Dict[str, List[str]]], int, int, int]:
    """
    Extracts metadata from the list of player sensor data dictionaries.

    Parameters
    ----------
    players_sensor_data_dict_list : List[Dict]
        A list of dictionaries, each containing sensor data for a player.
        Each dictionary represents a player's sensor data and typically includes
        metadata and a list of data entries, where each data entry is a dictionary
        with various sensor measurements.

    Returns
    -------
    Tuple[Dict[str, Dict[str, List[str]]], int, int, int]
        - Dictionary mapping group IDs to player identifiers.
        - Number of frames.
        - Framerate.
        - Null timestamp.
    """
    # Get mappings for sensor and group identifiers
    mapping, reversed_group_mapping, reversed_sensor_mapping = _get_mappings()

    # Extract key names from the list of dictionaries and map them using the provided mappings
    key_names = list(_get_key_names_from_dict_list(players_sensor_data_dict_list))
    key_names_mapped = [mapping[i] for i in key_names]

    # Define sets for sensor and group identifiers to check against
    sensor_identifier = {"name", "number", "sensor_id", "mapped_id"}
    key_names_mapped_set = set(key_names_mapped)

    # Determine which keys in the data correspond to sensor identifiers
    recorded_sensor_identifier = list(key_names_mapped_set & sensor_identifier)
    sensor_links = {
        key: index
        for index, key in enumerate(key_names_mapped)
        if key in recorded_sensor_identifier
    }

    # Determine which keys in the data correspond to group identifiers
    group_identifier_set = {"group_id", "group_name"}
    recorded_group_identifier = list(key_names_mapped_set & group_identifier_set)

    # Initialize dictionaries to store player IDs and timestamps
    pID_dict = {}
    t = []
    max_count_athlete = 0
    # Check if there are any group identifiers present in the data
    has_groups = len(recorded_group_identifier) > 0
    if not has_groups:
        warnings.warn("Since no group exists in data, dummy group '0' is created!")

    # Track the maximum count of timestamps observed for calculating framerate
    max_count = 0
    for athlete_entry in players_sensor_data_dict_list:
        # Extract and convert timestamps from each data entry to centiseconds
        athlete_t = []
        for entry in athlete_entry["data"]:
            try:
                ts = int(entry["ts"])
                cs = (
                    int(entry["cs"]) if entry["cs"] is not None else 0
                )  # Default to 0 if cs is None
                athlete_t.append(ts)
                t.append(ts * 100 + cs)
            except TypeError:
                print(f"Skipping entry due to TypeError: {entry}")

        # Count occurrences of each timestamp
        timestamps_athlete_count = Counter(athlete_t)

        if timestamps_athlete_count:
            max_count_athlete = max(timestamps_athlete_count.values())

        # Extract group ID from the current athlete entry
        group_identifier = recorded_group_identifier[0]
        group_id = athlete_entry[reversed_group_mapping[group_identifier]]
        if group_id not in pID_dict:
            pID_dict[group_id] = {}

        # Update the player ID dictionary with sensor identifiers and their corresponding values
        for identifier in sensor_links:
            if identifier not in pID_dict[group_id]:
                pID_dict[group_id][identifier] = []
            if (
                athlete_entry[reversed_sensor_mapping[identifier]]
                not in pID_dict[group_id][identifier]
            ):
                pID_dict[group_id][identifier].append(
                    athlete_entry[reversed_sensor_mapping[identifier]]
                )

    # Determine the framerate from the maximum count of timestamps
    framerate = max(max_count, max_count_athlete)

    # Sort and process the collected timestamps
    # Filter out keys that are None or not integers/strings (or other types you expect)
    valid_pID_dict = {
        k: v for k, v in pID_dict.items() if k is not None and isinstance(k, (int, str))
    }

    # Sort the filtered dictionary
    sorted_pID_dict = dict(sorted(valid_pID_dict.items()))

    # Update the original dictionary
    pID_dict = sorted_pID_dict
    timestamps_cs = list(set(t))
    timestamps_cs.sort()

    if not timestamps_cs:
        raise ValueError(
            "The timestamps_cs list is empty. Cannot calculate the number of frames."
        )
    # Calculate the number of frames based on the duration of the timestamps
    # The magic number 100 is used to convert centiseconds to seconds.
    number_of_frames = int((timestamps_cs[-1] - timestamps_cs[0]) / (100 / framerate))
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
        The available sensor identifier. This function checks for the presence of known sensor identifiers
        ("name", "mapped_id", "sensor_id", "number") and returns the first one found in the metadata.
    """
    player_identifiers = ["name", "mapped_id", "sensor_id", "number"]
    # Check if the pID_dict is empty
    if not pID_dict.values():
        return {}  # Return empty dict if the pID_dict is empty

    available_identifier = [
        idt for idt in player_identifiers if idt in list(pID_dict.values())[0]
    ]

    return available_identifier[0]


def _create_links_from_meta_data(
    pID_dict: Dict[str, Dict[str, List[str]]], identifier: str = None
) -> Dict[str, Dict[str, int]]:
    """
    Creates a mapping of player IDs to numerical indices based on the metadata.

    Parameters
    ----------
    pID_dict : Dict[str, Dict[str, List[str]]]
        Metadata dictionary containing player IDs by group.
    identifier : str, optional
        The sensor identifier to use for mapping. Defaults to the available identifier if not provided.

    Returns
    -------
    Dict[str, Dict[str, int]]
        A dictionary mapping group IDs to player ID indices. Each group ID maps to a dictionary where player IDs
        are associated with their respective numerical indices.
    """
    if identifier is None:
        identifier = _get_available_sensor_identifier(pID_dict)

    links = {}
    for group_id in pID_dict:
        links[group_id] = {
            ID: xID for xID, ID in enumerate(pID_dict[group_id][identifier])
        }
    return links


def read_position_data_from_dict_list(
    players_sensor_data_dict_list: List[Dict], coord_format: str = "xy"
) -> List[XY]:
    """
    Reads position data from players_sensor_data_dict_list and returns a list of XY objects.

    Parameters
    ----------
    players_sensor_data_dict_list : List[Dict]
        A list of dictionaries, each containing sensor data for a player.
        Each dictionary represents a player's sensor data and typically includes
        metadata and a list of data entries, where each data entry is a dictionary
        with various sensor measurements.

    coord_format : str, optional
        The coordinate format to use for interpreting the position data. This determines how the coordinate values
        are extracted from the data:
        - "xy": Coordinates will be in the fields "x" and "y" of each entry.
        - "latlong": Coordinates will be in the fields "lat" (latitude) and "long" (longitude) of each entry.
        Defaults to "xy".

    Returns
    -------
    List[XY]
        A list of XY objects containing position data.

    Raises
    ------
    ValueError
        If an invalid coordinate format is specified.
    """
    mapping, reversed_group_mapping, reversed_sensor_mapping = _get_mappings()

    # Extract metadata from the sensor data
    pID_dict, number_of_frames, framerate, t_null = get_meta_data(
        players_sensor_data_dict_list
    )

    # Create a mapping of player IDs to numerical indices
    links = _create_links_from_meta_data(pID_dict)

    # Get the key names used in the sensor data
    key_names = _get_key_names_from_dict_list(players_sensor_data_dict_list)
    key_names_mapped = [mapping[i] for i in list(key_names)]
    key_links_set = set(key_names_mapped)
    group_identifier_set = {"group_id", "group_name"}
    recorded_group_identifier = list(key_links_set & group_identifier_set)

    # Determine the sensor identifier to use
    identifier = _get_available_sensor_identifier(pID_dict)

    number_of_sensors = {}
    xydata = {}

    # Initialize data structures for storing position data
    for group in links:
        number_of_sensors.update({group: len(links[group])})
        xydata.update(
            {
                group: np.full(
                    [number_of_frames + 1, number_of_sensors[group] * 2], np.nan
                )
            }
        )

    # Process each athlete's data
    for athlete_entry in players_sensor_data_dict_list:
        for entry in athlete_entry["data"]:
            # Extract coordinates based on the specified format
            if coord_format == "xy":
                x_coordinate = entry["x"]
                y_coordinate = entry["y"]
            elif coord_format == "latlong":
                x_coordinate = entry["lat"]
                y_coordinate = entry["long"]
            else:
                raise ValueError(
                    f"Expected coordinate format to be ['xy', 'latlong'], but got {coord_format} instead."
                )

            # Get the group ID and calculate the column indices for x and y coordinates
            group_identifier = recorded_group_identifier[0]
            group_id = athlete_entry[reversed_group_mapping[group_identifier]]

            x_col = (
                links[group_id][athlete_entry[reversed_sensor_mapping[identifier]]] * 2
            )
            y_col = x_col + 1

            # Calculate the row index based on the timestamp
            # Timestamps are in centiseconds. The magic value 100 is used to convert centiseconds to seconds.
            row = int(
                (int(entry["ts"]) * 100 + int(entry["cs"]) - t_null) / (100 / framerate)
            )

            # Store the coordinates in the appropriate position in the data array
            if x_coordinate != "":
                xydata[group_id][row, x_col] = x_coordinate

            if y_coordinate != "":
                xydata[group_id][row, y_col] = y_coordinate

    # Convert the position data to a list of XY objects
    data_objects = []
    for group_id in xydata:
        data_objects.append(XY(xy=xydata[group_id], framerate=framerate))
    return data_objects


def read_position_data_from_activity(
    base_url: str,
    api_token: str,
    activity_id: str,
    save: bool = False,
    save_format: str = "pkl",
    path_to_players_sensor_data_dict_list: str = None,
    coord_format: str = "xy",
) -> List[XY]:
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
        Whether to save the players_sensor_data_dict_list to a file. Defaults to False.
    save_format : str, optional
        The format to use when saving the response data. It can be either "json" or "pickle".
        - "json": Saves the data in JSON format.
        - "pickle": Saves the data in pickle format.
        Defaults to "pickle".
    path_to_players_sensor_data_dict_list : str, optional
        The path to the saved players_sensor_data_dict_list file. If provided, data will be read from this path instead of making an API request.
    coord_format : str, optional
        The coordinate format to use for interpreting the position data. This determines how the coordinate values
        are extracted from the data:
        - "xy": Coordinates will be in the fields "x" and "y" of each entry.
        - "latlong": Coordinates will be in the fields "lat" (latitude) and "long" (longitude) of each entry.
        Defaults to "xy".

    Returns
    -------
    List[XY]
        A list of XY objects containing position data.

    Conditional Behavior
    --------------------
    1. If `path_to_players_sensor_data_dict_list` is provided:
       - The function reads the response data from the specified file path, skipping any API requests.

    2. If `path_to_players_sensor_data_dict_list` is not provided:
       - The function makes an API request to fetch the data.
       - If `save` is set to `True`, the fetched response is saved to files in both pickle and JSON formats.
       - If `save` is `False`, the response data is processed directly without saving.

    Notes
    -----
    - Ensure that the `path_to_players_sensor_data_dict_list` points to a valid file path containing previously saved response data.
    - The `coord_format` parameter allows flexibility in how position data is interpreted based on the available fields in the data.
    - The function `read_position_data_from_dict_list` converts the raw data dictionaries into `XY` objects.

    Example
    -------
    To fetch and process position data from an API, you might use:

    >>> base_url = "https://connect-eu.catapultsports.com/api/v6/"
    >>> api_token = "your_api_token_here"
    >>> activity_id = "your_activity_id_here"
    >>> data = read_position_data_from_activity(base_url, api_token, activity_id, save=True, coord_format="xy")

    In this example:
    - Data is fetched from the API.
    - The response is saved in both pickle and JSON formats.
    - Position data is processed into a list of `XY` objects using the "xy" coordinate format.

    Alternatively, to read data from a file:

    >>> path_to_file = "path/to/saved/players_sensor_data_dict_list.json"
    >>> data = read_position_data_from_activity(base_url, api_token, activity_id, path_to_players_sensor_data_dict_list=path_to_file, coord_format="latlong")

    In this case:
    - Data is read from the specified file path.
    - Position data is processed using the "latlong" coordinate format.
    """

    # Check if a path to a saved response is provided
    if path_to_players_sensor_data_dict_list:
        # Read data from a specified file path
        players_sensor_data_dict_list = load_list_of_dicts(
            path_to_players_sensor_data_dict_list
        )
    else:
        # Fetch player information for the activity from the API
        activity_players_info, endpoint = get_activity_players_info(
            base_url, api_token, activity_id
        )

        # Fetch sensor data for the players involved in the activity
        players_sensor_data_dict_list = get_players_sensor_data_dict_list(
            base_url, api_token, activity_id, activity_players_info
        )

        if save:
            if save_format == "pickle":
                dump_list_of_dicts(
                    players_sensor_data_dict_list,
                    endpoint + "/players_sensor_data_dict_list.pkl",
                    file_format="pickle",
                )
            elif save_format == "json":
                dump_list_of_dicts(
                    players_sensor_data_dict_list,
                    endpoint + "/players_sensor_data_dict_list.json",
                    file_format="json",
                )

    # Convert the raw data dictionaries into XY objects
    data_objects = read_position_data_from_dict_list(
        players_sensor_data_dict_list, coord_format
    )

    return data_objects

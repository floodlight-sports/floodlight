import pytest
import pickle
import json
import numpy as np
import requests

from floodlight.core.xy import XY
from floodlight.io.catapult import (
    dump_list_of_dicts,
    load_list_of_dicts,
    get_meta_data,
    _get_key_names_from_dict_list,
    _get_available_sensor_identifier,
    _create_links_from_meta_data,
    read_position_data_from_dict_list,
    _get_mappings,
)


# Test if API is live and reachable.
@pytest.mark.unit
def test_api_is_live():
    base_url = "https://connect-eu.catapultsports.com/api/v6/"
    try:
        response = requests.get(base_url)
        assert (
            response.status_code is not None
        ), "No response from API - it may be down or unreachable."
    except requests.exceptions.RequestException:
        pytest.fail("API is not live or unreachable.")


# Test dumping data as pickle format
@pytest.mark.unit
def test_dump_list_of_dicts_pickle(tmp_path) -> None:
    # Sample data
    data = [{"key1": "value1", "key2": "value2"}, {"key3": "value3"}]
    pickle_file = tmp_path / "test_data.pkl"
    dump_list_of_dicts(data, str(pickle_file), "pickle")

    # Check that the file was created
    assert pickle_file.exists(), "Pickle file was not created."

    # Deserialize and verify the contents
    with open(pickle_file, "rb") as f:
        loaded_data = pickle.load(f)
    assert loaded_data == data, "Pickle file data does not match the original."


# Test dumping data as JSON format
@pytest.mark.unit
def test_dump_list_of_dicts_json(tmp_path) -> None:
    # Sample data
    data = [{"key1": "value1", "key2": "value2"}, {"key3": "value3"}]
    json_file = tmp_path / "test_data.json"
    dump_list_of_dicts(data, str(json_file), "json")

    # Check that the file was created
    assert json_file.exists(), "JSON file was not created."

    # Deserialize and verify the contents
    with open(json_file, "r") as f:
        loaded_data = json.load(f)
    assert loaded_data == data, "JSON file data does not match the original."


# Test loading data from a pickle file
@pytest.mark.unit
def test_load_list_of_dicts_pickle(tmp_path) -> None:
    # Sample data
    data = [{"key1": "value1", "key2": "value2"}, {"key3": "value3"}]
    pickle_file = tmp_path / "test_data.pkl"

    with open(pickle_file, "wb") as f:
        pickle.dump(data, f)
    loaded_data = load_list_of_dicts(str(pickle_file))

    # Verify that the loaded data matches the original
    assert loaded_data == data, "Loaded pickle data does not match the original."


# Test loading data from a JSON file
@pytest.mark.unit
def test_load_list_of_dicts_json(tmp_path) -> None:
    # Sample data
    data = [{"key1": "value1", "key2": "value2"}, {"key3": "value3"}]
    json_file = tmp_path / "test_data.json"
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    loaded_data = load_list_of_dicts(str(json_file))

    # Verify that the loaded data matches the original
    assert loaded_data == data, "Loaded JSON data does not match the original."


# Test loading from a non-existent file
@pytest.mark.unit
def test_load_list_of_dicts_file_not_found() -> None:
    non_existent_file = "non_existent_file.json"

    # Check if loading from a non-existent file returns an empty list
    loaded_data = load_list_of_dicts(non_existent_file)
    assert (
        loaded_data == []
    ), "Loading from a non-existent file did not return an empty list."


# Test if correct key_names_extraction
@pytest.mark.unit
def test_get_key_names_from_dict_list(players_sensor_data_dict_list):
    expected_keys = {
        "lat",
        "long",
        "stream_type",
        "cs",
        "athlete_id",
        "team_name",
        "ts",
        "x",
        "y",
    }
    result = _get_key_names_from_dict_list(players_sensor_data_dict_list)

    # Check if the result contains the expected keys
    assert result == expected_keys


@pytest.mark.unit
def test_get_key_names_from_dict_list_empty():
    # Test with an empty list
    result = _get_key_names_from_dict_list([])

    # Ensure it returns an empty set
    assert result == set()


# Test to check behavior with a typical data input for consistency and accuracy
def test_get_meta_data(players_sensor_data_dict_list):
    pID_dict, number_of_frames, framerate, t_null = get_meta_data(
        players_sensor_data_dict_list
    )

    # Expected output
    expected_pID_dict = {
        "test": {"mapped_id": ["4a9b398a-3d6b-45be-8b69-3f83b3781093"]},
        "test2": {"mapped_id": ["4a9b398a-3d6b-45be-8b69-3f83b3781093"]},
    }

    # Assertions
    assert (
        pID_dict == expected_pID_dict
    ), "Player ID dictionary does not match expected output."


# Test to ensure ValueError is raised when timestamps are None
def test_get_meta_data_value_error_on_none_timestamps(
    players_sensor_data_dict_list_none,
):
    with pytest.raises(
        ValueError,
        match="The timestamps_cs list is empty. Cannot calculate the number of frames.",
    ):
        get_meta_data(players_sensor_data_dict_list_none)


# Test to verify output when data contains only zero values
def test_get_meta_data_zeros(players_sensor_data_dict_list_zeros):
    pID_dict, number_of_frames, framerate, t_null = get_meta_data(
        players_sensor_data_dict_list_zeros
    )

    # Expected output
    expected_pID_dict = {0: {"mapped_id": [0]}}
    assert (
        pID_dict == expected_pID_dict
    ), "Player ID dictionary does not match expected output."


@pytest.mark.unit
def test_get_available_sensor_identifier():
    # Test case with multiple identifiers present
    pID_dict_1 = {
        "group1": {
            "name": ["Player A"],
            "mapped_id": ["ID1"],
            "sensor_id": ["SENSOR123"],
            "number": ["99"],
        },
        "group2": {"name": ["Player B"], "mapped_id": ["ID2"]},
    }
    assert (
        _get_available_sensor_identifier(pID_dict_1) == "name"
    )  # "name" is the first identifier in the list

    # Test case with only one identifier present
    pID_dict_2 = {"group1": {"mapped_id": ["ID1"]}}
    assert (
        _get_available_sensor_identifier(pID_dict_2) == "mapped_id"
    )  # "mapped_id" is the only identifier

    # Test case with no identifiers present
    pID_dict_3 = {"group1": {}}
    with pytest.raises(
        IndexError
    ):  # Expect an IndexError when trying to access the first element of an empty list
        _get_available_sensor_identifier(pID_dict_3)

    # Test case with no identifiers in the first group
    pID_dict_4 = {"group1": {"other_id": ["ID1"]}}
    with pytest.raises(IndexError):  # Expect an IndexError for no available identifiers
        _get_available_sensor_identifier(pID_dict_4)


@pytest.mark.unit
def test_create_links_from_meta_data_with_default_identifier():
    pID_dict = {
        "group1": {
            "name": ["Player 1", "Player 2"],
            "mapped_id": ["id1", "id2"],
            "sensor_id": ["sensor1", "sensor2"],
            "number": ["99", "88"],
        },
        "group2": {
            "name": ["Player 3", "Player 4"],
            "mapped_id": ["id3", "id4"],
            "sensor_id": ["sensor3", "sensor4"],
            "number": ["77", "66"],
        },
    }

    expected_links = {
        "group1": {"Player 1": 0, "Player 2": 1},
        "group2": {"Player 3": 0, "Player 4": 1},
    }

    result = _create_links_from_meta_data(pID_dict)
    assert result == expected_links


@pytest.mark.unit
def test_create_links_from_meta_data_with_custom_identifier():
    pID_dict = {
        "group1": {
            "name": ["Player 1", "Player 2"],
            "mapped_id": ["id1", "id2"],
            "sensor_id": ["sensor1", "sensor2"],
            "number": ["99", "88"],
        }
    }

    expected_links = {
        "group1": {"sensor1": 0, "sensor2": 1},
    }

    result = _create_links_from_meta_data(pID_dict, identifier="sensor_id")
    assert result == expected_links


@pytest.mark.unit
def test_create_links_from_meta_data_empty_dict():
    pID_dict = {}
    expected_links = {}

    result = _create_links_from_meta_data(pID_dict)
    assert result == expected_links


@pytest.mark.unit
def test_create_links_from_meta_data_with_no_valid_identifier():
    pID_dict = {
        "group1": {
            "name": [],
            "mapped_id": [],
            "sensor_id": [],
            "number": [],
        }
    }

    expected_links = {"group1": {}}

    result = _create_links_from_meta_data(pID_dict)
    assert result == expected_links


@pytest.mark.unit
def test_read_position_data_xy(players_sensor_data_dict_list):
    # Retrieve actual mappings
    mapping, reversed_group_mapping, reversed_sensor_mapping = _get_mappings()

    # Execute the function
    result = read_position_data_from_dict_list(
        players_sensor_data_dict_list, coord_format="xy"
    )

    # Validate the result
    assert len(result) == 2  # Expecting two group of XY data
    assert isinstance(result[0], XY)  # Result should be of type XY
    assert np.array_equal(
        result[0].xy, np.array([[0.0, 0.0]])
    )  # Validate extracted coordinates
    assert np.array_equal(result[0].xy, np.array([[0.0, 0.0]]))


@pytest.mark.unit
def test_read_position_data_latlong(players_sensor_data_dict_list):
    # Retrieve actual mappings
    mapping, reversed_group_mapping, reversed_sensor_mapping = _get_mappings()

    result = read_position_data_from_dict_list(
        players_sensor_data_dict_list, coord_format="latlong"
    )

    # Validate the result
    assert len(result) == 2  # Expecting 2 group of XY data
    assert isinstance(result[0], XY)  # Result should be of type XY
    assert np.array_equal(result[0].xy, np.array([[5.0, 10.0]]))  # Validate extracted
    assert np.array_equal(result[1].xy, np.array([[15.0, 20.0]]))


@pytest.mark.unit
def test_read_position_data_invalid_format(players_sensor_data_dict_list):
    # Retrieve actual mappings
    mapping, reversed_group_mapping, reversed_sensor_mapping = _get_mappings()

    # Test for ValueError with invalid coordinate format
    with pytest.raises(ValueError) as excinfo:
        read_position_data_from_dict_list(
            players_sensor_data_dict_list, coord_format="invalid_format"
        )
    assert (
        "Expected coordinate format to be ['xy', 'latlong'], \
            but got invalid_format instead."
        in str(excinfo.value)
    )

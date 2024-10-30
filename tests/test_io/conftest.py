import pytest
import numpy as np


# EIGD mock data (first 5 frames of first 8 players of team A
# from match_name="48dcd3", segment="00-06-00" in original EIGD format (as in h5 files)
@pytest.fixture()
def eigd_sample_data_h5_shape() -> np.ndarray:
    data = np.array(
        [
            [
                [30.416, 15.463],
                [32.848, 10.999],
                [34.93, 2.903],
                [35.271, 16.948],
                [33.492, 6.2],
                [32.273, 9.547],
                [37.354, 9.504],
                [np.nan, np.nan],
            ],
            [
                [30.416, 15.463],
                [32.848, 10.999],
                [34.93, 2.903],
                [35.271, 16.948],
                [33.492, 6.2],
                [32.273, 9.547],
                [37.354, 9.504],
                [np.nan, np.nan],
            ],
            [
                [30.416, 15.463],
                [32.848, 10.999],
                [34.93, 2.903],
                [35.271, 16.948],
                [33.492, 6.2],
                [32.273, 9.547],
                [37.354, 9.504],
                [np.nan, np.nan],
            ],
            [
                [30.416, 15.463],
                [32.848, 10.999],
                [34.93, 2.903],
                [35.271, 16.948],
                [33.492, 6.2],
                [32.273, 9.547],
                [37.354, 9.504],
                [np.nan, np.nan],
            ],
            [
                [30.416, 15.463],
                [32.848, 10.999],
                [34.93, 2.903],
                [35.271, 16.948],
                [33.492, 6.2],
                [32.273, 9.547],
                [37.354, 9.504],
                [np.nan, np.nan],
            ],
        ]
    )

    return data


# EIGD mock data (first 5 frames of first 8 players of team A
# from match_name="48dcd3", segment="00-06-00" in (transformed) floodlight format
@pytest.fixture()
def eigd_sample_data_floodlight_shape() -> np.ndarray:
    data = np.array(
        [
            [
                30.416,
                15.463,
                32.848,
                10.999,
                34.93,
                2.903,
                35.271,
                16.948,
                33.492,
                6.2,
                32.273,
                9.547,
                37.354,
                9.504,
                np.nan,
                np.nan,
            ],
            [
                30.416,
                15.463,
                32.848,
                10.999,
                34.93,
                2.903,
                35.271,
                16.948,
                33.492,
                6.2,
                32.273,
                9.547,
                37.354,
                9.504,
                np.nan,
                np.nan,
            ],
            [
                30.416,
                15.463,
                32.848,
                10.999,
                34.93,
                2.903,
                35.271,
                16.948,
                33.492,
                6.2,
                32.273,
                9.547,
                37.354,
                9.504,
                np.nan,
                np.nan,
            ],
            [
                30.416,
                15.463,
                32.848,
                10.999,
                34.93,
                2.903,
                35.271,
                16.948,
                33.492,
                6.2,
                32.273,
                9.547,
                37.354,
                9.504,
                np.nan,
                np.nan,
            ],
            [
                30.416,
                15.463,
                32.848,
                10.999,
                34.93,
                2.903,
                35.271,
                16.948,
                33.492,
                6.2,
                32.273,
                9.547,
                37.354,
                9.504,
                np.nan,
                np.nan,
            ],
        ]
    )

    return data


@pytest.fixture()
def filepath_empty() -> str:
    path = ".data\\EMPTY"
    return path


# Sample response from catapult api with only zeros
@pytest.fixture
def activity_players_info_zeros():
    return [
        {
            "id": "0",
            "first_name": "0",
            "last_name": "0",
            "gender": "0",
            "jersey": "0",
            "nickname": "0",
            "height": 0,
            "weight": 0,
            "date_of_birth": 0,
            "velocity_max": 0,
            "acceleration_max": 0,
            "heart_rate_max": 0,
            "player_load_max": 0,
            "image": "",
            "icon": "",
            "stroke_colour": "",
            "fill_colour": "",
            "trail_colour_start": "",
            "trail_colour_end": "",
            "is_synced": 0,
            "is_deleted": 0,
            "created_at": "0",
            "modified_at": "0",
            "activity_athlete_id": "0",
            "date_of_birth_date": "0",
            "tag_list": ["Full"],
            "tags": [
                {
                    "id": "0",
                    "tag_type_id": "0",
                    "name": "0",
                    "tag_type_name": "0",
                    "tag_name": "0",
                }
            ],
            "position": "0",
            "position_id": "0",
            "position_name": "0",
        }
    ]


# Sample response from catapult api with only None
@pytest.fixture
def activity_players_info_none():
    return [
        {
            "id": None,
            "first_name": None,
            "last_name": None,
            "gender": None,
            "jersey": None,
            "nickname": None,
            "height": None,
            "weight": None,
            "date_of_birth": None,
            "velocity_max": None,
            "acceleration_max": None,
            "heart_rate_max": None,
            "player_load_max": None,
            "image": None,
            "icon": None,
            "stroke_colour": None,
            "fill_colour": None,
            "trail_colour_start": None,
            "trail_colour_end": None,
            "is_synced": None,
            "is_deleted": None,
            "created_at": None,
            "modified_at": None,
            "activity_athlete_id": None,
            "date_of_birth_date": None,
            "tag_list": ["Full"],
            "tags": [
                {
                    "id": None,
                    "tag_type_id": None,
                    "name": None,
                    "tag_type_name": None,
                    "tag_name": None,
                }
            ],
            "position": None,
            "position_id": None,
            "position_name": None,
        }
    ]


# Sample sensor response from catapult api
@pytest.fixture
def players_sensor_data_dict_list():
    return [
        {
            "team_name": "test",
            "athlete_id": "4a9b398a-3d6b-45be-8b69-3f83b3781093",
            "device_id": 50576,
            "stream_type": "gps",
            "data": [
                {
                    "ts": 1697703120,
                    "cs": 1,
                    "lat": 5,
                    "long": 10,
                    "v": 5,
                    "a": 0,
                    "hr": 200,
                    "pl": None,
                    "x": 0,
                    "y": 0,
                }
            ],
        },
        {
            "team_name": "test2",
            "athlete_id": "4a9b398a-3d6b-45be-8b69-3f83b3781093",
            "device_id": 50576,
            "stream_type": "gps",
            "data": [
                {
                    "ts": 1697703120,
                    "cs": 1,
                    "lat": 15,
                    "long": 20,
                    "v": 5,
                    "a": 0,
                    "hr": 200,
                    "pl": None,
                    "x": 0,
                    "y": 0,
                }
            ],
        },
    ]


# All none sensor response from catapult api with
@pytest.fixture
def players_sensor_data_dict_list_none():
    return [
        {
            "team_name": None,
            "athlete_id": None,
            "device_id": None,
            "stream_type": None,
            "data": [
                {
                    "ts": None,
                    "cs": None,
                    "lat": None,
                    "long": None,
                    "v": None,
                    "a": None,
                    "hr": None,
                    "pl": None,
                }
            ],
        }
    ]


# All zero sensor response from catapult api with
@pytest.fixture
def players_sensor_data_dict_list_zeros():
    return [
        {
            "team_name": 0,
            "athlete_id": 0,
            "device_id": 0,
            "stream_type": 0,
            "data": [
                {
                    "ts": 0,
                    "cs": 0,
                    "lat": 0,
                    "long": 0,
                    "v": 0,
                    "a": 0,
                    "hr": 0,
                    "pl": 0,
                }
            ],
        }
    ]

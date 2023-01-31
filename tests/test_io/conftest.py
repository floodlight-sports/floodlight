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

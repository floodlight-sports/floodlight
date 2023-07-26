# Import necessary modules and functions
import pytest
from pathlib import Path
from floodlight.io.kinexon import (
    get_column_names_from_csv,
    _get_column_links,
    get_meta_data,
    create_links_from_meta_data,
    read_position_data_csv,
)
from floodlight.core.xy import XY
import numpy as np

# Define the path to the test data file
test_data_path = Path(".data/kinexon_dataset/anonymized_data.csv")


# Test the function that retrieves column names from the CSV file
@pytest.mark.unit
def test_get_column_names_from_csv() -> None:
    columns = get_column_names_from_csv(test_data_path)
    assert len(columns) == 11
    assert "ts in ms" in columns
    assert "x in m" in columns
    assert "y in m" in columns


# Test the function that creates a dictionary with column links from the CSV file
@pytest.mark.unit
def test__get_column_links() -> None:
    column_links = _get_column_links(test_data_path)
    assert len(column_links) == 10
    assert "time" in column_links
    assert "x_coord" in column_links
    assert "y_coord" in column_links


# Test the function that retrieves meta-data from the CSV file
@pytest.mark.unit
def test_get_meta_data() -> None:
    pID_dict, number_of_frames, framerate, t_null = get_meta_data(test_data_path)
    assert isinstance(pID_dict, dict)
    assert isinstance(number_of_frames, int)
    assert isinstance(framerate, int)
    assert isinstance(t_null, np.int64)


# Test the function that creates links from the retrieved meta-data
@pytest.mark.unit
def test_create_links_from_meta_data() -> None:
    pID_dict, _, _, _ = get_meta_data(test_data_path)
    links = create_links_from_meta_data(pID_dict)
    assert isinstance(links, dict)
    assert len(links) == len(pID_dict)


# Test the function that reads position data from the CSV
# file and converts it to XY objects
@pytest.mark.unit
def test_read_position_data_csv() -> None:
    positions = read_position_data_csv(test_data_path)
    assert isinstance(positions, list)
    assert all(isinstance(position, XY) for position in positions)

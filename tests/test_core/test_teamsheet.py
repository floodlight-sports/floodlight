from copy import deepcopy

import pytest
import pandas as pd

from floodlight.core.teamsheet import Teamsheet
from floodlight.core.definitions import protected_columns


@pytest.mark.unit
def test_teamsheet_getter(example_teamsheet_data: pd.DataFrame) -> None:
    data = Teamsheet(example_teamsheet_data)
    pIDs = data["pID"]
    assert pd.Series.equals(pIDs, pd.Series([1, 2, 3]))


@pytest.mark.unit
def test_teamsheet_setter(example_teamsheet_data: pd.DataFrame) -> None:
    data = Teamsheet(example_teamsheet_data)
    data["pID"] = [2, 3, 4]
    assert pd.Series.equals(data["pID"], pd.Series([2, 3, 4]))


@pytest.mark.unit
def test_column_properties() -> None:
    # Arrange df with different columns
    columns = ["name", "pID"]
    df = pd.DataFrame(columns=columns)

    # trigger post_init
    with pytest.raises(ValueError):
        teamsheet = Teamsheet(teamsheet=df)

    # add missing column
    columns.append("player")
    df = pd.DataFrame(columns=columns)
    teamsheet = Teamsheet(teamsheet=df)

    # Assert column properties
    assert teamsheet.essential == ["player"]
    assert teamsheet.protected == ["pID"]
    assert teamsheet.custom == ["name"]
    assert teamsheet.essential_missing == []
    assert len(teamsheet.protected_missing) > 3


@pytest.mark.unit
def test_protected_missing(
    example_teamsheet_data: pd.DataFrame,
) -> None:
    data = Teamsheet(example_teamsheet_data)
    missing_protected_columns = data.protected_missing
    total_num_protected = len(protected_columns)

    assert len(missing_protected_columns) == total_num_protected - 2


@pytest.mark.unit
def test_protected_invalid(
    example_teamsheet_data: pd.DataFrame,
) -> None:
    data = Teamsheet(example_teamsheet_data)
    data.teamsheet.at[1, "jID"] = -1
    invalid_protected_columns = data.protected_invalid
    assert invalid_protected_columns == ["jID"]


@pytest.mark.unit
def test_column_values_in_range(
    example_teamsheet_data: pd.DataFrame,
) -> None:
    # Arrange
    data = Teamsheet(example_teamsheet_data)
    jID_in_range = data.column_values_in_range("jID", protected_columns)

    assert jID_in_range


@pytest.mark.unit
def test_get_links(example_teamsheet_data) -> None:
    data = Teamsheet(example_teamsheet_data)

    # trigger checks
    with pytest.raises(ValueError):
        data.get_links("xID", "pID")
    with pytest.raises(ValueError):
        data.get_links("pID", "xID")
    with pytest.raises(ValueError):
        data.get_links("position", "pID")

    links = data.get_links("pID", "jID")
    assert links == {1: 1, 2: 13, 3: 99}


@pytest.mark.unit
def test_add_xIDs(example_teamsheet_data) -> None:
    data = Teamsheet(example_teamsheet_data)
    data.add_xIDs()
    assert all(data.teamsheet["xID"].values == [0, 1, 2])


@pytest.mark.unit
def test_add_xIDs_overwrite(example_teamsheet_data) -> None:
    example_teamsheet_data["xID"] = [2, 0, 1]
    original_data = Teamsheet(example_teamsheet_data)
    new_data = Teamsheet(deepcopy(example_teamsheet_data))
    new_data.add_xIDs()
    assert all(original_data.teamsheet["xID"].values == [2, 0, 1])
    assert all(new_data.teamsheet["xID"].values == [0, 1, 2])

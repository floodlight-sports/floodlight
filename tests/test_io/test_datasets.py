import pytest
import numpy as np

from floodlight.io.datasets import EIGDDataset
from floodlight.io.datasets import StatsBombOpenDataset
from floodlight import Events
from floodlight.core.teamsheet import Teamsheet


# Test _transform staticmethod from EIGDDataset
@pytest.mark.unit
def test_eigd_transform(
    eigd_sample_data_h5_shape, eigd_sample_data_floodlight_shape
) -> None:
    # transform data in raw format
    data_transformed = EIGDDataset._transform(eigd_sample_data_h5_shape)

    assert np.array_equal(
        data_transformed, eigd_sample_data_floodlight_shape, equal_nan=True
    )


# Test get method from StatsBombDataset
@pytest.mark.unit
def test_statsbomb_get() -> None:

    dataset = StatsBombOpenDataset()
    events, teamsheets = dataset.get(
        "Champions League",
        "2004/2005",
        "AC Milan vs. Liverpool",
    )
    assert isinstance(events["HT1"]["Home"], Events)
    assert isinstance(events["HT4"]["Away"], Events)
    assert isinstance(teamsheets["Home"], Teamsheet)
    assert isinstance(teamsheets["Away"], Teamsheet)


# Test get_teamsheet method from StatsBombDataset
@pytest.mark.unit
def test_statsbomb_get_teamsheet() -> None:

    dataset = StatsBombOpenDataset()
    teamsheets = dataset.get_teamsheets(
        "Champions League",
        "2004/2005",
        "AC Milan vs. Liverpool",
    )
    assert isinstance(teamsheets["Home"], Teamsheet)
    assert isinstance(teamsheets["Away"], Teamsheet)
    assert teamsheets["Home"].teamsheet.at[0, "team_name"] == "AC Milan"
    assert teamsheets["Away"].teamsheet.at[0, "team_name"] == "Liverpool"
    assert len(teamsheets["Home"].teamsheet) == 14
    assert len(teamsheets["Away"].teamsheet) == 14
    assert len(teamsheets["Home"].teamsheet.columns) == 6
    assert len(teamsheets["Away"].teamsheet.columns) == 6


# Test passing custom home_teamsheet to get method
@pytest.mark.unit
def test_statsbomb_get_pass_custom_home_teamsheet() -> None:

    # get teamsheets
    dataset = StatsBombOpenDataset()
    teamsheets = dataset.get_teamsheets(
        "Champions League",
        "2004/2005",
        "AC Milan vs. Liverpool",
    )

    # customize home teamsheet
    teamsheets["Home"].teamsheet.at[0, "player"] = "Dida"  # custom entry
    teamsheets["Home"].teamsheet.at[0, "pID"] = 999999  # custom entry
    teamsheets["Home"]["custom_col"] = 99  # custom column passed to function
    teamsheets["Away"]["my_col"] = 99  # custom column but not passed to function

    # call get function with custom teamsheet
    events, teamsheets = dataset.get(
        "Champions League",
        "2004/2005",
        "AC Milan vs. Liverpool",
        teamsheet_home=teamsheets["Home"],
    )

    assert teamsheets["Home"].teamsheet.at[0, "player"] == "Dida"
    assert teamsheets["Home"].teamsheet.at[0, "pID"] == 999999
    assert "custom_col" in teamsheets["Home"].teamsheet.columns
    assert "my_col" not in teamsheets["Away"].teamsheet.columns


# Test passing custom away_teamsheet to get method
@pytest.mark.unit
def test_statsbomb_get_pass_custom_away_teamsheet() -> None:
    # get teamsheets
    dataset = StatsBombOpenDataset()
    teamsheets = dataset.get_teamsheets(
        "Champions League",
        "2004/2005",
        "AC Milan vs. Liverpool",
    )

    # customize home teamsheet
    teamsheets["Home"].teamsheet.at[0, "player"] = "Dida"  # custom entry but not passed
    teamsheets["Home"].teamsheet.at[0, "pID"] = 999999  # custom entry but not passed
    teamsheets["Home"]["custom_col"] = 99  # custom column but not passed to function
    teamsheets["Away"]["my_col"] = 99  # custom column passed to function

    # call get function with custom teamsheet
    events, teamsheets = dataset.get(
        "Champions League",
        "2004/2005",
        "AC Milan vs. Liverpool",
        teamsheet_away=teamsheets["Away"],
    )

    assert teamsheets["Home"].teamsheet.at[0, "player"] != "Dida"
    assert teamsheets["Home"].teamsheet.at[0, "pID"] != 999999
    assert "custom_col" not in teamsheets["Home"].teamsheet.columns
    assert "my_col" in teamsheets["Away"].teamsheet.columns


# Test passing custom away_teamsheet to get method
@pytest.mark.unit
def test_statsbomb_get_pass_custom_teamsheets() -> None:
    # get teamsheets
    dataset = StatsBombOpenDataset()
    teamsheets = dataset.get_teamsheets(
        "Champions League",
        "2004/2005",
        "AC Milan vs. Liverpool",
    )

    # customize home teamsheet
    teamsheets["Home"].teamsheet.at[0, "player"] = "Dida"  # custom entry
    teamsheets["Home"].teamsheet.at[0, "pID"] = 999999  # custom entry
    teamsheets["Home"]["custom_col"] = 99  # custom column but not passed to function
    teamsheets["Away"]["my_col"] = 99  # custom column passed to function

    # call get function with custom teamsheet
    events, teamsheets = dataset.get(
        "Champions League",
        "2004/2005",
        "AC Milan vs. Liverpool",
        teamsheet_home=teamsheets["Home"],
        teamsheet_away=teamsheets["Away"],
    )

    assert teamsheets["Home"].teamsheet.at[0, "player"] == "Dida"
    assert teamsheets["Home"].teamsheet.at[0, "pID"] == 999999
    assert "custom_col" in teamsheets["Home"].teamsheet.columns
    assert "my_col" in teamsheets["Away"].teamsheet.columns

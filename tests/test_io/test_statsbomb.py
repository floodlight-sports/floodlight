import pytest

from floodlight.io.statsbomb import read_open_statsbomb_event_data_json


# Test _transform staticmethod from EIGDDataset
@pytest.mark.unit
def test_statsbomb_read_events_path_not_exists(
        filepath_empty
) -> None:

    with pytest.raises(FileNotFoundError):
        read_open_statsbomb_event_data_json(filepath_events=filepath_empty,
                                            filepath_match=filepath_empty)
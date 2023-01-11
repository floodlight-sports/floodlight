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
def test_statsbomb_get(
) -> None:

    dataset = StatsBombOpenDataset()
    data = dataset.get(
        "Champions League",
        "2004/2005",
        "AC Milan vs. Liverpool",
    )
    assert isinstance(data[0], Events)
    assert isinstance(data[1], Events)
    assert isinstance(data[2], Events)
    assert isinstance(data[3], Events)
    assert isinstance(data[4], Teamsheet)
    assert isinstance(data[5], Teamsheet)

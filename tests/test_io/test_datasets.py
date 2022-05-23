import pytest
import numpy as np

from floodlight.io.datasets import EIGDDataset


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

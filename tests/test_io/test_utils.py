import pytest

from floodlight.io.utils import get_and_convert


# Test get_and_convert function
@pytest.mark.unit
def test_get_and_convert() -> None:
    sample_dict = {"foo": "1"}

    # get
    assert get_and_convert(sample_dict, "foo", int) == 1
    # convert
    assert type(get_and_convert(sample_dict, "foo", int)) is int
    # missing entry
    assert get_and_convert(sample_dict, "bar", int) is None
    # fallback if failed type conversion
    assert get_and_convert(sample_dict, "foo", dict) == "1"
    # custom default with failed conversion
    assert get_and_convert(sample_dict, "bar", int, "default") == "default"

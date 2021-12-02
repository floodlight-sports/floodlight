import pytest


from floodlight.core.code import Code


# Test def token(self) property
@pytest.mark.unit
def test_token(example_code: Code) -> None:
    # Act
    token = example_code.token

    # Assert
    assert token == ["A", "H"]

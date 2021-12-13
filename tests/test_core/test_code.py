import pytest


from floodlight.core.code import Code


# Test def token property
@pytest.mark.unit
def test_token(example_code: Code) -> None:
    # Act
    token = example_code.token

    # Assert
    assert token == ["A", "H"]


# Test def slice(startframe, endframe, inplace) method
@pytest.mark.unit
def test_slice(example_code: Code) -> None:
    # copy
    code = example_code
    code_deep_copy = code.slice()
    assert code is not code_deep_copy
    assert code.code is not code_deep_copy.code
    assert (code.code == code_deep_copy.code).all()

    # slicing
    code_short = code.slice(endframe=3)
    code_mid = code.slice(startframe=4, endframe=6)
    code_none = code.slice(startframe=1, endframe=1)
    assert (code_short.code == ["A", "A", "A"]).all()
    assert (code_mid.code == ["A", "H"]).all()
    assert code_none.code.size == 0
    assert code.slice()

    # inplace
    code.slice(startframe=8, inplace=True)
    assert (code.code == ["H", "H"]).all()

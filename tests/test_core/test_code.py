import pytest


from floodlight.core.code import Code


# Test dunder methods
@pytest.mark.unit
def test_dunder_methods(example_code: Code, example_code_int: Code) -> None:
    example_code[1] = "R"
    code_eq_R = [False, True, False, False, False, False, False, False, False, False]
    code_ne_R = [True, False, True, True, True, True, True, True, True, True]

    # __len__
    assert len(example_code) == 10
    # __getitem__ and __setitem__
    assert example_code[0] == "A"
    assert example_code[1] == "R"
    # __eq__ and __ne__
    assert ((example_code == "R") == code_eq_R).all()
    assert ((example_code != "R") == code_ne_R).all()
    # __gt__, __lt__, __ge__, and __le__
    assert ((example_code_int < 1) == [True, False, False, False]).all()
    assert ((example_code_int > 2) == [False, False, False, True]).all()
    assert ((example_code_int <= 1) == [True, True, False, False]).all()
    assert ((example_code_int >= 2) == [False, False, True, True]).all()


# Test def token property
@pytest.mark.unit
def test_token(example_code: Code) -> None:
    # Act
    token = example_code.token

    # Assert
    assert token == ["A", "H"]


# Test def find_sequences(return_type) method
@pytest.mark.unit
def test_find_sequences(
    example_code: Code, example_code_int: Code, example_code_empty
) -> None:
    # literal token
    assert example_code.find_sequences() == {"A": [(0, 5)], "H": [(5, 10)]}
    assert example_code.find_sequences(return_type="list") == [
        (0, 5, "A"),
        (5, 10, "H"),
    ]
    # numeric token and single occurrences
    assert example_code_int.find_sequences() == {
        0: [(0, 1)],
        1: [(1, 2)],
        2: [(2, 3)],
        3: [(3, 4)],
    }
    assert example_code_int.find_sequences(return_type="list") == [
        (0, 1, 0),
        (1, 2, 1),
        (2, 3, 2),
        (3, 4, 3),
    ]
    # empty code
    assert example_code_empty.find_sequences() == {}
    assert example_code_empty.find_sequences(return_type="list") == []


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

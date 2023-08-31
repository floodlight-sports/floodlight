import pytest


from floodlight.core.definitions import essential_events_columns, protected_columns


# Test for column specifications - tests only dictionary syntax, so that new added
# columns dont' break existing methods
@pytest.mark.unit
def test_essential_events_columns() -> None:
    # Arrange
    for key in essential_events_columns:
        # Act + Assert
        column = essential_events_columns[key]
        assert type(column["definition"]) is str
        assert type(column["dtypes"]) is list

        if column["value_range"] is not None:
            assert type(column["value_range"]) is list
            assert len(column["value_range"]) == 2
            min_val, max_val = column["value_range"]
            assert min_val <= max_val


@pytest.mark.unit
def test_protected_columns() -> None:
    # Arrange
    for key in protected_columns:
        # Act + Assert
        column = protected_columns[key]
        assert type(column["definition"]) is str
        assert type(column["dtypes"]) is list

        if column["value_range"] is not None:
            assert type(column["value_range"]) is list
            assert len(column["value_range"]) == 2
            min_val, max_val = column["value_range"]
            assert min_val <= max_val

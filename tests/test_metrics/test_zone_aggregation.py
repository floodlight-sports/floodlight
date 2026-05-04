import pytest
import numpy as np

from floodlight import PlayerProperty
from floodlight.metrics.zone_aggregation import aggregate_property_by_zones


# Test basic sum aggregation with PlayerProperty
@pytest.mark.unit
def test_aggregate_property_by_zones_basic_sum(
    player_property_simple, player_property_binning
):
    # Arrange
    zones = [(0, 3), (3, 6), (6, 9)]
    zone_names = ["Low", "Medium", "High"]

    # Act
    result = aggregate_property_by_zones(
        player_property_simple,
        player_property_binning,
        zones,
        zone_names,
        aggregation="sum",
    )

    # Assert
    assert result.shape == (2, 3)
    assert list(result.columns) == ["Low", "Medium", "High"]
    assert np.array_equal(result.iloc[0].values, [60.0, 150.0, 0.0])
    assert np.array_equal(result.iloc[1].values, [0.0, 1500.0, 600.0])


# Test count aggregation returns correct frame counts
@pytest.mark.unit
def test_aggregate_property_by_zones_count(
    player_property_simple, player_property_binning
):
    # Arrange
    zones = [(0, 4), (4, 8)]

    # Act
    result = aggregate_property_by_zones(
        player_property_simple, player_property_binning, zones, aggregation="count"
    )

    # Assert
    assert result.shape == (2, 2)
    assert np.array_equal(result.iloc[0].values, [4.0, 2.0])
    assert np.array_equal(result.iloc[1].values, [1.0, 4.0])


# Test aggregation with TeamProperty inputs
@pytest.mark.unit
def test_aggregate_property_by_zones_team_property(
    team_property_simple, team_property_binning
):
    # Arrange
    zones = [(0, 2), (5, 9)]
    zone_names = ["Low", "High"]

    # Act
    result = aggregate_property_by_zones(
        team_property_simple,
        team_property_binning,
        zones,
        zone_names,
        aggregation="sum",
    )

    # Assert
    assert result.shape == (1, 2)
    assert list(result.columns) == ["Low", "High"]
    assert np.array_equal(result.iloc[0].values, [30.0, 180.0])


# Test PlayerProperty aggregated by TeamProperty (broadcasting)
@pytest.mark.unit
def test_aggregate_property_by_zones_player_by_team(
    player_property_simple, team_property_binning
):
    # Arrange
    zones = [(0, 2), (5, 9)]

    # Act
    result = aggregate_property_by_zones(
        player_property_simple, team_property_binning, zones, aggregation="sum"
    )

    # Assert
    assert result.shape == (2, 2)
    assert np.array_equal(result.iloc[0].values, [30.0, 180.0])
    assert np.array_equal(result.iloc[1].values, [300.0, 1800.0])


# Test that NaN values in either property are excluded from aggregation
@pytest.mark.unit
def test_aggregate_property_by_zones_nan_handling(
    player_property_with_nans, player_property_binning_with_nans
):
    # Arrange
    zones = [(0, 7)]

    # Act
    result_sum = aggregate_property_by_zones(
        player_property_with_nans,
        player_property_binning_with_nans,
        zones,
        aggregation="sum",
    )
    result_count = aggregate_property_by_zones(
        player_property_with_nans,
        player_property_binning_with_nans,
        zones,
        aggregation="count",
    )

    # Assert
    assert np.array_equal(result_sum.iloc[0].values, [190.0])
    assert np.array_equal(result_sum.iloc[1].values, [180.0])
    assert np.array_equal(result_count.iloc[0].values, [5.0])
    assert np.array_equal(result_count.iloc[1].values, [5.0])


# Test that zone boundaries use [min, max) (inclusive min, exclusive max)
@pytest.mark.unit
def test_aggregate_property_by_zones_boundary_handling():
    # Arrange
    binning = PlayerProperty(
        property=np.array([[0], [2], [4], [6]], dtype=float), name="binning"
    )
    values = PlayerProperty(
        property=np.array([[1], [1], [1], [1]], dtype=float), name="values"
    )
    zones = [(0, 2), (2, 4), (4, 6), (6, 8)]

    # Act
    result = aggregate_property_by_zones(values, binning, zones, aggregation="count")

    # Assert
    assert np.array_equal(result.iloc[0].values, [1.0, 1.0, 1.0, 1.0])


# Test that empty zones return 0 for sum/count and NaN for mean
@pytest.mark.unit
def test_aggregate_property_by_zones_empty_zones(
    player_property_simple, player_property_binning
):
    # Arrange
    zones = [(0, 5), (10, 20)]

    # Act
    result_sum = aggregate_property_by_zones(
        player_property_simple, player_property_binning, zones, aggregation="sum"
    )
    result_count = aggregate_property_by_zones(
        player_property_simple, player_property_binning, zones, aggregation="count"
    )
    result_mean = aggregate_property_by_zones(
        player_property_simple, player_property_binning, zones, aggregation="mean"
    )

    # Assert
    assert result_sum.iloc[0, 1] == 0.0
    assert result_sum.iloc[1, 1] == 0.0
    assert result_count.iloc[0, 1] == 0.0
    assert result_count.iloc[1, 1] == 0.0
    assert np.isnan(result_mean.iloc[0, 1])
    assert np.isnan(result_mean.iloc[1, 1])


# Test that TeamProperty by PlayerProperty raises ValueError
@pytest.mark.unit
def test_aggregate_property_by_zones_invalid_team_by_player(
    team_property_simple, player_property_binning
):
    # Arrange
    zones = [(0, 5)]

    # Act & Assert
    with pytest.raises(
        ValueError, match="Cannot aggregate TeamProperty by PlayerProperty"
    ):
        aggregate_property_by_zones(
            team_property_simple, player_property_binning, zones, aggregation="sum"
        )


# Test that mismatched dimensions raise ValueError
@pytest.mark.unit
def test_aggregate_property_by_zones_mismatched_dimensions(
    player_property_simple, player_property_3_players, player_property_4_frames
):
    # Arrange
    zones = [(0, 5)]

    # Act & Assert - mismatched player dimension
    with pytest.raises(ValueError, match="Player dimensions must match"):
        aggregate_property_by_zones(
            player_property_simple, player_property_3_players, zones, aggregation="sum"
        )

    # Act & Assert - mismatched time dimension
    with pytest.raises(ValueError, match="Time dimensions must match"):
        aggregate_property_by_zones(
            player_property_simple, player_property_4_frames, zones, aggregation="sum"
        )


# Test default zone name generation and single zone handling
@pytest.mark.unit
def test_aggregate_property_by_zones_default_names_and_single_zone():
    # Arrange
    values = PlayerProperty(property=np.array([[10]], dtype=float), name="values")
    binning = PlayerProperty(property=np.array([[1]], dtype=float), name="binning")
    zones = [(0, 5)]

    # Act
    result = aggregate_property_by_zones(values, binning, zones, aggregation="sum")

    # Assert
    assert list(result.columns) == ["0 to 5"]
    assert result.shape == (1, 1)
    assert np.array_equal(result.iloc[0].values, [10.0])


# Test that all aggregation functions run correctly
@pytest.mark.unit
def test_aggregate_property_by_zones_all_aggregations(
    player_property_simple, player_property_binning
):
    # Arrange
    zones = [(0, 10)]

    # Act & Assert
    for agg in ["sum", "count", "mean", "min", "max"]:
        result = aggregate_property_by_zones(
            player_property_simple, player_property_binning, zones, aggregation=agg
        )

        assert result.shape == (2, 1)
        assert not result.isnull().all().all()

        if agg == "min":
            assert result.iloc[0, 0] == 10.0
        elif agg == "max":
            assert result.iloc[0, 0] == 60.0
        elif agg == "mean":
            expected_mean = np.mean([10, 20, 30, 40, 50, 60])
            assert result.iloc[0, 0] == expected_mean

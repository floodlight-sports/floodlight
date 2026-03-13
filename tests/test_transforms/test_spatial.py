import pytest
import numpy as np

from floodlight import XY
from floodlight.transforms.spatial import subtract_centroid, min_max_normalize


@pytest.mark.unit
def test_subtract_centroid(example_xy_spatial: XY) -> None:
    # Arrange
    xy = example_xy_spatial

    # Act
    xy_centered = subtract_centroid(xy)

    # Assert
    # centroid frame 0: (3, 2), frame 1: (4, 3), frame 2: (5, 4)
    expected = np.array(
        [
            [-3, -2, 3, -2, 0, 4],
            [-3, -2, 3, -2, 0, 4],
            [-3, -2, 3, -2, 0, 4],
        ],
        dtype=float,
    )
    assert np.array_equal(xy_centered.xy, expected)
    assert xy_centered.framerate == 10
    assert xy_centered.direction == "lr"


@pytest.mark.unit
def test_subtract_centroid_exclude(example_xy_spatial: XY) -> None:
    # Arrange
    xy = example_xy_spatial

    # Act — exclude player 0, centroid from players 1 and 2 only
    xy_centered = subtract_centroid(xy, exclude_xIDs=[0])

    # Assert
    # centroid frame 0 from players 1,2: ((6+3)/2, (0+6)/2) = (4.5, 3)
    # all 3 players translated by (-4.5, -3)
    expected_frame0 = np.array([-4.5, -3, 1.5, -3, -1.5, 3])
    assert np.array_equal(xy_centered.xy[0], expected_frame0)
    # shape preserved: all 3 players still present
    assert xy_centered.N == 3


@pytest.mark.unit
def test_subtract_centroid_exclude_invalid_xid(example_xy_spatial: XY) -> None:
    # Arrange
    xy = example_xy_spatial

    # Act & Assert
    with pytest.raises(ValueError, match="Expected entries of exclude_xIDs"):
        subtract_centroid(xy, exclude_xIDs=[5])


@pytest.mark.unit
def test_subtract_centroid_with_nan(example_xy_spatial_with_nan: XY) -> None:
    # Arrange
    xy = example_xy_spatial_with_nan

    # Act
    xy_centered = subtract_centroid(xy)

    # Assert
    # frame 1: player 0 is NaN, centroid from all 3 (NaN ignored by nanmean)
    # centroid frame 1: ((nan+7+4)/2, (nan+1+7)/2) = (5.5, 4)
    assert np.isnan(xy_centered.xy[1, 0])
    assert np.isnan(xy_centered.xy[1, 1])
    assert np.array_equal(xy_centered.xy[1, 2:], np.array([1.5, -3.0, -1.5, 3.0]))


@pytest.mark.unit
def test_min_max_normalize() -> None:
    # Arrange
    positions = np.array([[0, 0], [10, 5], [5, 10]])

    # Act
    normalized = min_max_normalize(positions)

    # Assert
    expected = np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]])
    assert np.array_equal(normalized, expected)


@pytest.mark.unit
def test_min_max_normalize_zero_range() -> None:
    # Arrange — all players on same x-line
    positions = np.array([[5, 0], [5, 5], [5, 10]])

    # Act
    normalized = min_max_normalize(positions)

    # Assert — x-axis has zero range, stays at 0.0
    expected = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]])
    assert np.array_equal(normalized, expected)


@pytest.mark.unit
def test_subtract_centroid_immutability(example_xy_spatial: XY) -> None:
    # Arrange
    xy = example_xy_spatial
    original_data = xy.xy.copy()

    # Act
    subtract_centroid(xy)

    # Assert — input unchanged
    assert np.array_equal(xy.xy, original_data)

import numpy as np
import pytest

from floodlight import XY, Code
from floodlight.core.property import PlayerProperty, DyadicProperty, TeamProperty
from floodlight.transforms.temporal import resample


# --- Core path: identity, downsample, upsample ---


@pytest.mark.unit
def test_resample_identity(example_xy_spatial: XY) -> None:
    # Arrange
    xy = example_xy_spatial  # framerate=10, direction="lr", T=3

    # Act
    xy_out = resample(xy, 10, interp_method="spline")

    # Assert — bit-equal payload, metadata preserved, fresh object
    assert isinstance(xy_out, XY)
    assert np.array_equal(xy_out.xy, xy.xy)
    assert xy_out.framerate == 10
    assert xy_out.direction == "lr"
    assert xy_out is not xy
    assert xy_out.xy is not xy.xy


@pytest.mark.unit
def test_resample_downsample_default(example_xy_permutation: XY) -> None:
    # Arrange
    xy = example_xy_permutation

    # Act
    xy_out = resample(xy, 5)

    # Assert
    expected = np.array(
        [
            [0.0, 0.0, 10.0, 0.0, 5.0, 10.0],
            [5.0, 10.0, 0.0, 0.0, 10.0, 0.0],
        ]
    )
    assert np.array_equal(xy_out.xy, expected)
    assert xy_out.framerate == 5
    assert isinstance(xy_out.framerate, int)


@pytest.mark.unit
def test_resample_upsample_linear(example_xy_permutation: XY) -> None:
    # Arrange
    xy = example_xy_permutation

    # Act
    xy_out = resample(xy, 20, interp_method="linear")

    # Assert
    expected = np.array(
        [
            [0.0, 0.0, 10.0, 0.0, 5.0, 10.0],
            [5.0, 0.0, 5.0, 0.0, 5.0, 10.0],
            [10.0, 0.0, 0.0, 0.0, 5.0, 10.0],
            [7.5, 5.0, 0.0, 0.0, 7.5, 5.0],
            [5.0, 10.0, 0.0, 0.0, 10.0, 0.0],
            [2.5, 5.0, 2.5, 5.0, 10.0, 0.0],
            [0.0, 0.0, 5.0, 10.0, 10.0, 0.0],
        ]
    )
    assert np.array_equal(np.round(xy_out.xy, 4), expected)
    assert xy_out.framerate == 20
    assert isinstance(xy_out.framerate, int)


@pytest.mark.unit
def test_resample_upsample_polynomial(example_xy_permutation: XY) -> None:
    # Arrange
    xy = example_xy_permutation

    # Act
    xy_out = resample(xy, 20, interp_method="polynomial")

    # Assert
    expected = np.array(
        [
            [0.0, 0.0, 10.0, 0.0, 5.0, 10.0],
            [7.8125, -3.125, 3.4375, 0.625, 3.75, 12.5],
            [10.0, 0.0, 0.0, 0.0, 5.0, 10.0],
            [8.4375, 5.625, -0.9375, -0.625, 7.5, 5.0],
            [5.0, 10.0, 0.0, 0.0, 10.0, 0.0],
            [1.5625, 9.375, 2.1875, 3.125, 11.25, -2.5],
            [0.0, 0.0, 5.0, 10.0, 10.0, 0.0],
        ]
    )
    assert np.array_equal(np.round(xy_out.xy, 4), expected)
    assert xy_out.framerate == 20


@pytest.mark.unit
def test_resample_upsample_spline(example_xy_permutation: XY) -> None:
    # Arrange
    xy = example_xy_permutation

    # Act
    xy_out = resample(xy, 20, interp_method="spline")

    # Assert
    expected = np.array(
        [
            [0.0, 0.0, 10.0, 0.0, 5.0, 10.0],
            [7.8125, -3.125, 3.4375, 0.625, 3.75, 12.5],
            [10.0, 0.0, 0.0, 0.0, 5.0, 10.0],
            [8.4375, 5.625, -0.9375, -0.625, 7.5, 5.0],
            [5.0, 10.0, 0.0, 0.0, 10.0, 0.0],
            [1.5625, 9.375, 2.1875, 3.125, 11.25, -2.5],
            [0.0, 0.0, 5.0, 10.0, 10.0, 0.0],
        ]
    )
    assert np.array_equal(np.round(xy_out.xy, 4), expected)
    assert xy_out.framerate == 20


@pytest.mark.unit
def test_resample_upsample_nearest(example_xy_permutation: XY) -> None:
    # Arrange
    xy = example_xy_permutation

    # Act
    xy_out = resample(xy, 20, interp_method="nearest")

    # Assert
    expected = np.array(
        [
            [0.0, 0.0, 10.0, 0.0, 5.0, 10.0],
            [0.0, 0.0, 10.0, 0.0, 5.0, 10.0],
            [10.0, 0.0, 0.0, 0.0, 5.0, 10.0],
            [10.0, 0.0, 0.0, 0.0, 5.0, 10.0],
            [5.0, 10.0, 0.0, 0.0, 10.0, 0.0],
            [5.0, 10.0, 0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 5.0, 10.0, 10.0, 0.0],
        ]
    )
    assert np.array_equal(xy_out.xy, expected)
    assert xy_out.framerate == 20


@pytest.mark.unit
def test_resample_tie_break_30_to_20() -> None:
    # Arrange
    xy = XY(
        np.array([[k, k] for k in range(7)], dtype=float),
        framerate=30,
    )

    # Act
    xy_out = resample(xy, 20)

    # Assert
    assert xy_out.xy.shape == (5, 2)
    assert np.array_equal(xy_out.xy[:, 0], np.array([0, 2, 3, 5, 6], dtype=float))
    assert xy_out.framerate == 20


# --- Type dispatch ---


@pytest.mark.unit
def test_resample_dispatch_code(example_code_temporal: Code) -> None:
    # Arrange
    code = example_code_temporal

    # Act
    code_out = resample(code, 25)

    # Assert
    assert isinstance(code_out, Code)
    assert code_out.code.shape == (5,)
    assert code_out.framerate == 25
    assert code_out.name == "possession"
    assert code_out.definitions == {0: "none", 1: "home", 2: "away"}


@pytest.mark.unit
def test_resample_dispatch_team_property(
    example_team_property_temporal: TeamProperty,
) -> None:
    # Arrange
    prop = example_team_property_temporal

    # Act
    prop_out = resample(prop, 25)

    # Assert
    assert prop_out.property.shape == (5,)
    assert prop_out.framerate == 25
    assert prop_out.name == "stretch_index"
    assert np.array_equal(prop_out.property, prop.property[::2])


@pytest.mark.unit
def test_resample_dispatch_player_property(
    example_player_property_temporal: PlayerProperty,
) -> None:
    # Arrange
    prop = example_player_property_temporal

    # Act
    prop_out = resample(prop, 25)

    # Assert
    assert isinstance(prop_out, PlayerProperty)
    assert prop_out.property.shape == (5, 3)
    assert prop_out.framerate == 25
    assert prop_out.name == "speed"


@pytest.mark.unit
def test_resample_dispatch_dyadic_property(
    example_dyadic_property_temporal: DyadicProperty,
) -> None:
    # Arrange
    prop = example_dyadic_property_temporal

    # Act
    prop_out = resample(prop, 25)

    # Assert
    assert isinstance(prop_out, DyadicProperty)
    assert prop_out.property.shape == (5, 2, 3)
    assert prop_out.framerate == 25
    assert prop_out.name == "distance"


# --- Side-effect / invariant tests ---


@pytest.mark.unit
def test_resample_nan_passthrough(example_xy_filter: XY) -> None:
    # Arrange
    xy = example_xy_filter

    # Act
    xy_out = resample(xy, 10, interp_method="linear")

    # Assert
    assert np.all(np.isnan(xy_out.xy[:, 2].astype(float)))


# --- Edge cases and errors ---


@pytest.mark.unit
def test_resample_empty(example_xy_filter_empty: XY) -> None:
    # Arrange
    xy = example_xy_filter_empty

    # Act
    xy_out = resample(xy, 10, interp_method="spline")

    # Assert
    assert isinstance(xy_out, XY)
    assert xy_out.xy.shape == (0,)
    assert xy_out.framerate == 10


@pytest.mark.unit
def test_resample_spline_insufficient_samples_raises(
    example_xy_filter_short: XY,
) -> None:
    # Arrange
    xy = example_xy_filter_short

    # Act & Assert
    with pytest.raises(
        ValueError,
        match=r"Method 'spline' requires at least 4 source samples, got 2\.",
    ):
        resample(xy, 40, interp_method="spline")


@pytest.mark.unit
def test_resample_framerate_none_raises() -> None:
    # Arrange
    xy = XY(np.arange(20, dtype=float).reshape(10, 2))

    # Act & Assert
    with pytest.raises(
        ValueError, match=r"Expected obj\.framerate to be set, got None\."
    ):
        resample(xy, 25)


@pytest.mark.unit
def test_resample_target_zero_raises(example_xy_spatial: XY) -> None:
    # Arrange
    xy = example_xy_spatial

    # Act & Assert
    with pytest.raises(
        ValueError, match=r"Expected target_framerate to be a positive integer"
    ):
        resample(xy, 0)


@pytest.mark.unit
def test_resample_unsupported_type_raises() -> None:
    # Arrange
    not_a_core_object = np.array([1.0, 2.0, 3.0])

    # Act & Assert
    with pytest.raises(ValueError, match=r"Expected obj to be one of"):
        resample(not_a_core_object, 25)

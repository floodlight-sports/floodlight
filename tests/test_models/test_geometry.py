import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from floodlight.models.geometry import (
    CentroidModel,
    ConvexHullModel,
    NearestMateModel,
    NearestOpponentModel,
)


# Test fit function of CentroidModel with different xIDs excluded
@pytest.mark.unit
def test_centroid_model_fit(example_xy_object_geometry) -> None:
    # Arrange
    xy = example_xy_object_geometry

    # Act
    model = CentroidModel()
    model.fit(xy)
    centroid1 = model._centroid_
    model.fit(xy, exclude_xIDs=[0])
    centroid2 = model._centroid_
    model.fit(xy, exclude_xIDs=[0, 1])
    centroid3 = model._centroid_

    # Assert
    assert np.array_equal(centroid1, np.array(((1.5, -1), (1.25, 0.5))))
    assert np.array_equal(centroid2, np.array(((2, -2), (1, 0.5))))
    assert np.array_equal(
        centroid3,
        np.array(((np.nan, -2), (1, 1))),
        equal_nan=True,
    )


# Test centroid function of CentroidModel
@pytest.mark.unit
def test_centroid(example_xy_object_geometry) -> None:
    # Arrange
    xy = example_xy_object_geometry

    # Act
    model = CentroidModel()
    model.fit(xy)
    centroid = model.centroid()

    # Assert
    assert np.array_equal(centroid, np.array(((1.5, -1), (1.25, 0.5))))


# Test centroid_distance function of CentroidModel
@pytest.mark.unit
def test_centroid_distance(example_xy_object_geometry) -> None:
    # Arrange
    xy = example_xy_object_geometry

    # Act
    model = CentroidModel()
    model.fit(xy)
    distance = model.centroid_distance(xy)

    # Assert
    assert np.array_equal(
        np.round(distance, 3),
        np.array(((2.062, 1.118, np.nan), (np.nan, np.nan, 0.559))),
        equal_nan=True,
    )


# Test stretch_index function of CentroidModel
@pytest.mark.unit
def test_stretch_index(example_xy_object_geometry) -> None:
    # Arrange
    xy = example_xy_object_geometry
    xy.framerate = 20

    # Act
    model = CentroidModel()
    model.fit(xy)
    stretch_index1 = model.stretch_index(xy)
    stretch_index2 = model.stretch_index(xy, axis="x")
    stretch_index3 = model.stretch_index(xy, axis="y")

    # Assert
    assert np.array_equal(np.round(stretch_index1, 3), np.array((1.59, 0.559)))
    assert np.array_equal(np.round(stretch_index2, 3), np.array((0.5, 0.25)))
    assert np.array_equal(np.round(stretch_index3, 3), np.array((1.333, 0.5)))
    assert stretch_index1.framerate == 20


# Test fit function of NearestMateModel
@pytest.mark.unit
def test_nearest_mate_model_fit(example_xy_object_geometry):
    # Arrange
    xy = example_xy_object_geometry

    # Act
    model = NearestMateModel()
    model.fit(xy)

    # Assert
    assert model._pairwise_distances_ is not None
    assert isinstance(model._pairwise_distances_, np.ndarray)
    assert model._pairwise_distances_.shape == (2, 3, 3)


# Test distance_to_nearest_mate function of NearestMateModel
@pytest.mark.unit
def test_nearest_mate_model_distance_to_nearest_mate(example_xy_object_geometry):
    # Arrange
    xy = example_xy_object_geometry

    # Act
    model = NearestMateModel()
    model.fit(xy)
    dtnm = model.distance_to_nearest_mate()

    # Assert
    expected = np.array(((np.sqrt(10), np.sqrt(10), np.nan), (np.nan, np.nan, np.nan)))
    assert np.array_equal(dtnm.property, expected, equal_nan=True)


# Test team_spread function of NearestMateModel
@pytest.mark.unit
def test_nearest_mate_model_team_spread(example_xy_object_geometry):
    # Arrange
    xy = example_xy_object_geometry

    # Act
    model = NearestMateModel()
    model.fit(xy)
    spread = model.team_spread()

    # Assert
    expected = np.array((np.sqrt(10), np.nan))
    assert np.array_equal(spread.property, expected, equal_nan=True)


# Test framerate propagation for NearestMateModel
@pytest.mark.unit
def test_nearest_mate_model_with_framerate(example_xy_object_geometry):
    # Arrange
    xy = example_xy_object_geometry
    xy.framerate = 25

    # Act
    model = NearestMateModel()
    model.fit(xy)
    dtnm = model.distance_to_nearest_mate()
    spread = model.team_spread()

    # Assert
    assert dtnm.framerate == 25
    assert spread.framerate == 25


# Test NearestMateModel with horizontal NaN slice
@pytest.mark.unit
def test_nearest_mate_model_horizontal_nan_slice(
    example_xy_object_geometry_horizontal_nan,
):
    # Arrange
    xy = example_xy_object_geometry_horizontal_nan

    # Act
    model = NearestMateModel()
    model.fit(xy)
    dtnm = model.distance_to_nearest_mate()
    spread = model.team_spread()

    # Assert
    expected_dtnm = np.array(
        (
            (np.sqrt(8), np.sqrt(10), np.sqrt(8)),
            (np.nan, np.nan, np.nan),
            (np.sqrt(6.25), np.sqrt(6.25), np.sqrt(8)),
        )
    )
    assert np.array_equal(dtnm.property, expected_dtnm, equal_nan=True)
    assert np.isnan(spread.property[1])


# Test NearestMateModel with single player
@pytest.mark.unit
def test_nearest_mate_model_single_player(example_xy_object_single_player):
    # Arrange
    xy = example_xy_object_single_player

    # Act
    model = NearestMateModel()
    model.fit(xy)
    dtnm = model.distance_to_nearest_mate()
    spread = model.team_spread()

    # Assert
    expected_dtnm = np.array(((np.nan,), (np.nan,), (np.nan,)))
    expected_spread = np.array((np.nan, np.nan, np.nan))
    assert np.array_equal(dtnm.property, expected_dtnm, equal_nan=True)
    assert np.array_equal(spread.property, expected_spread, equal_nan=True)


# Test fit function of NearestOpponentModel
@pytest.mark.unit
def test_nearest_opponent_model_fit(example_xy_objects_space_control):
    # Arrange
    xy1, xy2 = example_xy_objects_space_control

    # Act
    model = NearestOpponentModel()
    model.fit(xy1, xy2)

    # Assert
    assert model._pairwise_distances_ is not None
    assert isinstance(model._pairwise_distances_, np.ndarray)
    assert model._pairwise_distances_.shape == (2, 3, 3)


# Test distance_to_nearest_opponent function of NearestOpponentModel
@pytest.mark.unit
def test_nearest_opponent_model_distance_to_nearest_opponent(
    example_xy_objects_space_control,
):
    # Arrange
    xy1, xy2 = example_xy_objects_space_control

    # Act
    model = NearestOpponentModel()
    model.fit(xy1, xy2)
    dtno1, dtno2 = model.distance_to_nearest_opponent()

    # Assert
    expected_dtno1 = np.array(
        ((30.0, 0.0, 10.0), (np.sqrt(1417), np.sqrt(146), np.sqrt(500)))
    )
    assert np.array_equal(dtno1.property, expected_dtno1)


# Test framerate propagation for NearestOpponentModel
@pytest.mark.unit
def test_nearest_opponent_model_with_framerate(example_xy_objects_space_control):
    # Arrange
    xy1, xy2 = example_xy_objects_space_control

    # Act
    model = NearestOpponentModel()
    model.fit(xy1, xy2)
    dtno1, dtno2 = model.distance_to_nearest_opponent()

    # Assert
    assert dtno1.framerate == 20
    assert dtno2.framerate == 20


# Test NearestOpponentModel with NaN propagation
@pytest.mark.unit
def test_nearest_opponent_model_with_nan_propagation(example_xy_objects_space_control):
    # Arrange
    xy1, xy2 = example_xy_objects_space_control

    # Act
    model = NearestOpponentModel()
    model.fit(xy1, xy2)
    dtno1, dtno2 = model.distance_to_nearest_opponent()

    # Assert
    expected_dtno2 = np.array(((30.0, 0.0, np.sqrt(116)), (31.0, np.nan, np.sqrt(146))))
    assert np.array_equal(dtno2.property, expected_dtno2, equal_nan=True)


# Test NearestOpponentModel with horizontal NaN slice
@pytest.mark.unit
def test_nearest_opponent_model_horizontal_nan_slice(
    example_xy_objects_horizontal_nan,
):
    # Arrange
    xy1, xy2 = example_xy_objects_horizontal_nan

    # Act
    model = NearestOpponentModel()
    model.fit(xy1, xy2)
    dtno1, dtno2 = model.distance_to_nearest_opponent()

    # Assert
    expected_dtno1 = np.array(
        (
            (100.0, np.sqrt(8200), np.sqrt(6800)),
            (np.nan, np.nan, np.nan),
            (100.0, np.sqrt(8200), np.sqrt(6800)),
        )
    )
    expected_dtno2 = np.array(
        (
            (np.sqrt(6800), np.sqrt(8200), 100.0),
            (np.nan, np.nan, np.nan),
            (np.sqrt(6800), np.sqrt(8200), 100.0),
        )
    )
    assert np.array_equal(dtno1.property, expected_dtno1, equal_nan=True)
    assert np.array_equal(dtno2.property, expected_dtno2, equal_nan=True)


# Test NearestOpponentModel with single players
@pytest.mark.unit
def test_nearest_opponent_model_single_players(example_xy_objects_single_players):
    # Arrange
    xy1, xy2 = example_xy_objects_single_players

    # Act
    model = NearestOpponentModel()
    model.fit(xy1, xy2)
    dtno1, dtno2 = model.distance_to_nearest_opponent()

    # Assert
    expected_dtno1 = np.array(((100.0,), (100.0,), (100.0,)))
    expected_dtno2 = np.array(((100.0,), (100.0,), (100.0,)))
    assert np.array_equal(dtno1.property, expected_dtno1)
    assert np.array_equal(dtno2.property, expected_dtno2)


# Test ConvexHullModel fit and area calculation with known geometry
@pytest.mark.unit
def test_convex_hull_model_fit_and_area(example_xy_object_known_geometry):
    # Arrange
    xy = example_xy_object_known_geometry

    # Act
    model = ConvexHullModel()
    model.fit(xy)
    area = model.convex_hull_area()

    # Assert
    assert model._convex_hulls_ is not None
    assert len(model._convex_hulls_) == 1
    assert area.property.shape == (1,)
    assert np.allclose(
        area.property,
        np.array(
            [
                100.0,
            ]
        ),
    )
    assert area.name == "convex_hull_area"


# Test ConvexHullModel with multiple XY objects and framerate propagation
@pytest.mark.unit
def test_convex_hull_model_multiple_xy_and_framerate(example_xy_objects_space_control):
    # Arrange
    xy1, xy2 = example_xy_objects_space_control

    # Act
    model = ConvexHullModel()
    model.fit([xy1, xy2])
    area = model.convex_hull_area()

    # Assert
    assert area.framerate == 20
    assert np.array_equal(area.property, np.array((600.0, 682.0)))


# Test ConvexHullModel player exclusion functionality
@pytest.mark.unit
def test_convex_hull_model_exclusion(example_xy_object_geometry_horizontal_nan):
    # Arrange
    xy = example_xy_object_geometry_horizontal_nan

    # Act
    model1 = ConvexHullModel()
    model1.fit(xy)
    area1 = model1.convex_hull_area()

    model2 = ConvexHullModel()
    model2.fit(xy, exclude_xIDs=[[0]])
    area2 = model2.convex_hull_area()

    # Assert
    assert np.allclose(area1.property, np.array((4.0, np.nan, 0.5)), equal_nan=True)
    assert np.array_equal(
        area2.property, np.array((np.nan, np.nan, np.nan)), equal_nan=True
    )
    assert area1.property.shape == area2.property.shape


# Test ConvexHullModel NaN handling (partial NaNs, horizontal NaN, all NaN)
@pytest.mark.unit
def test_convex_hull_model_nan_handling(
    example_xy_object_geometry,
    example_xy_object_geometry_horizontal_nan,
    example_xy_object_all_nan,
):
    # Arrange & Act
    # Test 1: Partial NaNs (example_xy_object_geometry has NaN values)
    model1 = ConvexHullModel()
    model1.fit(example_xy_object_geometry)
    area1 = model1.convex_hull_area()

    # Test 2: Horizontal NaN slice (frame 1 is all NaN)
    model2 = ConvexHullModel()
    model2.fit(example_xy_object_geometry_horizontal_nan)
    area2 = model2.convex_hull_area()

    # Test 3: All NaN
    model3 = ConvexHullModel()
    model3.fit(example_xy_object_all_nan)
    area3 = model3.convex_hull_area()

    # Assert
    assert np.isnan(area1.property[1])
    assert np.isnan(area2.property[1])

    assert not np.isnan(area2.property[0])
    assert not np.isnan(area2.property[2])

    assert np.all(np.isnan(area3.property))


# Test ConvexHullModel with insufficient points (single, two, collinear)
@pytest.mark.unit
def test_convex_hull_model_insufficient_points(
    example_xy_object_single_player, example_xy_object_collinear
):
    # Arrange & Act
    # Single player (< 3 points)
    model1 = ConvexHullModel()
    model1.fit(example_xy_object_single_player)
    area1 = model1.convex_hull_area()

    # Collinear points (scipy raises QhullError)
    model2 = ConvexHullModel()
    model2.fit(example_xy_object_collinear)
    area2 = model2.convex_hull_area()

    # Assert
    assert np.all(np.isnan(area1.property))
    assert np.isnan(area2.property[0])


# Test ConvexHullModel validation errors
@pytest.mark.unit
def test_convex_hull_model_validation_errors():
    # Arrange
    from floodlight import XY

    xy1 = XY(np.array(((0, 0, 10, 10),)))
    xy2 = XY(np.array(((0, 0, 10, 10), (5, 5, 15, 15))))

    # Act & Assert
    # Mismatched lengths
    model = ConvexHullModel()
    with pytest.raises(ValueError, match="All XY objects must have same"):
        model.fit([xy1, xy2])

    # Invalid xID
    model = ConvexHullModel()
    with pytest.raises(ValueError, match="out of range"):
        model.fit(xy1, exclude_xIDs=[[5]])


# Test ConvexHullModel plot method
@pytest.mark.plot
def test_convex_hull_model_plot(example_xy_object_known_geometry):
    # Arrange
    xy = example_xy_object_known_geometry
    model = ConvexHullModel()
    model.fit(xy)

    # Act
    fig, ax = plt.subplots()
    model.plot(t=0, ax=ax)

    # Assert
    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.lines) > 0  # Should have plotted the hull boundary
    assert len(ax.patches) > 0  # Should have filled area
    plt.close()

    # Test plot without fill
    fig2, ax2 = plt.subplots()
    model.plot(t=0, ax=ax2, fill=False)
    assert isinstance(ax2, matplotlib.axes.Axes)
    assert len(ax2.lines) > 0  # Should have boundary
    # No filled polygon when fill=False
    plt.close()

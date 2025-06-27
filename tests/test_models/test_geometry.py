import pytest
import numpy as np
from scipy.spatial import ConvexHull

from floodlight.models.geometry import CentroidModel
from floodlight.models.geometry import ConvexHullModel


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


# Test fit function of ConvexHullModel with different xIDs excluded
@pytest.mark.unit
def test_convex_hull_model_fit(example_xy_object_geometry_convex_hull) -> None:
    # Arrange
    xy = example_xy_object_geometry_convex_hull
    c_hull = ConvexHull(xy.xy[0].reshape(-1, 2))  # create ConvexHull object from scipy

    # Act
    model = ConvexHullModel()
    model.fit(xy)
    convex_hull = model.convex_hulls

    # Assert
    assert type(convex_hull[0]) == type(c_hull)


@pytest.mark.unit
def test_convex_hull_model_fit_with_exclude_ids(
    example_xy_object_geometry_convex_hull,
) -> None:
    # Arrange
    xy = example_xy_object_geometry_convex_hull
    exclude_x_id = [4]

    # Act
    model = ConvexHullModel()
    model.fit(xy, exclude_x_id)
    chull = model.convex_hulls

    # Assert
    assert np.array_equal(
        chull[0].points, np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    )
    assert np.array_equal(chull[1].points, np.array([[0, 0], [10, 0], [5, 10], [2, 2]]))
    assert np.array_equal(chull[2].points, np.array([[0, 5], [5, 10], [10, 5], [5, 0]]))


@pytest.mark.unit
def test_convex_hull_model_area_convex_hull(
    example_xy_object_geometry_convex_hull,
) -> None:
    # Arrange
    xy = example_xy_object_geometry_convex_hull

    # Act
    model = ConvexHullModel()
    model.fit(xy)
    areas = model.area_convex_hull()

    # Assert
    assert np.allclose(areas.property, np.array([100, 50, 50]))

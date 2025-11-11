import pytest
import numpy as np
from floodlight.models.spatialMetricsModel import SpatialMetricsModel
from floodlight.core.property import TeamProperty


@pytest.mark.unit
def test_convex_hull_area(xy_data_valid1):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_valid1)
    convex_hull_result = smm.convex_hull_area()

    assert isinstance(convex_hull_result, TeamProperty)
    assert not np.isnan(convex_hull_result.property).all()


@pytest.mark.unit
def test_convex_hull_area_not_enough_points(xy_data_insufficient_points):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_insufficient_points)
    convex_hull_result = smm.convex_hull_area()

    assert np.isnan(convex_hull_result.property).all()


@pytest.mark.unit
def test_convex_hull_area_not_enough_players(xy_data_insufficient_players):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_insufficient_players)
    convex_hull_result = smm.convex_hull_area()

    assert np.isnan(convex_hull_result.property).all()


@pytest.mark.unit
def test_convex_hull_area_all_nan(xy_data_all_nan):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_all_nan)
    convex_hull_result = smm.convex_hull_area()

    assert np.isnan(convex_hull_result.property).all()


@pytest.mark.unit
def test_convex_hull_area_all_zeros(xy_data_all_zeros):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_all_zeros)
    convex_hull_result = smm.convex_hull_area()

    assert np.isnan(convex_hull_result.property).all()


@pytest.mark.unit
def test_convex_hull_area_all_ones(xy_data_all_ones):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_all_ones)
    convex_hull_result = smm.convex_hull_area()

    assert np.isnan(convex_hull_result.property).all()


@pytest.mark.unit
def test_convex_hull_area_one_nan_frame(xy_data_one_nan_frame):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_one_nan_frame)
    convex_hull_result = smm.convex_hull_area()

    assert isinstance(convex_hull_result, TeamProperty)
    assert not np.isnan(convex_hull_result.property).all()


@pytest.mark.unit
def test_effective_playing_space(xy_data_valid1, xy_data_valid2):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_valid1, xy_data_valid2)
    eps_result = smm.effective_playing_space()

    assert isinstance(eps_result, TeamProperty)
    assert not np.isnan(eps_result.property).all()


@pytest.mark.unit
def test_effective_playing_space_to_all_nan(xy_data_valid1, xy_data_all_nan):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_valid1, xy_data_all_nan)
    eps_result = smm.effective_playing_space()

    assert isinstance(eps_result, TeamProperty)
    assert np.isnan(eps_result.property).all()


@pytest.mark.unit
def test_effective_playing_space_to_all_zero(xy_data_valid1, xy_data_all_zeros):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_valid1, xy_data_all_zeros)
    eps_result = smm.effective_playing_space()

    assert isinstance(eps_result, TeamProperty)
    assert not np.isnan(eps_result.property).all()


@pytest.mark.unit
def test_effective_playing_space_to_all_ones(xy_data_valid1, xy_data_all_ones):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_valid1, xy_data_all_ones)
    eps_result = smm.effective_playing_space()

    assert isinstance(eps_result, TeamProperty)
    assert not np.isnan(eps_result.property).all()


@pytest.mark.unit
def test_effective_playing_space_to_incomplete_points(
    xy_data_valid1, xy_data_insufficient_points
):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_valid1, xy_data_insufficient_points)
    eps_result = smm.effective_playing_space()

    assert isinstance(eps_result, TeamProperty)
    assert not np.isnan(eps_result.property).all()


@pytest.mark.unit
def test_effective_playing_space_to_incomplete_players(
    xy_data_valid1, xy_data_insufficient_players
):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_valid1, xy_data_insufficient_players)
    eps_result = smm.effective_playing_space()

    assert isinstance(eps_result, TeamProperty)
    assert not np.isnan(eps_result.property).all()


@pytest.mark.unit
def test_effective_playing_space_to_one_nan_frame(
    xy_data_valid1, xy_data_one_nan_frame
):
    smm = SpatialMetricsModel()
    smm.fit(xy_data_valid1, xy_data_one_nan_frame)
    eps_result = smm.effective_playing_space()

    assert isinstance(eps_result, TeamProperty)
    assert not np.isnan(eps_result.property).all()

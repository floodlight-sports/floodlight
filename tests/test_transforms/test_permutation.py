import pytest
import numpy as np

from floodlight import XY
from floodlight.transforms.permutation import assign_roles


@pytest.mark.unit
def test_assign_roles_identity(example_xy_spatial: XY) -> None:
    # Arrange — players already in consistent positions across frames
    xy = example_xy_spatial

    # Act
    xy_assigned = assign_roles(xy)

    # Assert — output should match input (no swaps needed)
    assert xy_assigned.xy.shape == xy.xy.shape
    assert np.array_equal(xy_assigned.xy, xy.xy.astype(float))
    assert xy_assigned.framerate == 10


@pytest.mark.unit
def test_assign_roles_swap(example_xy_permutation: XY) -> None:
    # Arrange — players swap positions between frames
    xy = example_xy_permutation

    # Act
    xy_assigned = assign_roles(xy)

    # Assert — after assignment, each role column should be consistent
    # Role slots should have the same position across all frames
    T = len(xy_assigned)
    roles = xy_assigned.xy.reshape(T, -1, 2)
    for role_idx in range(3):
        role_positions = roles[:, role_idx, :]
        # all frames should have the same position for this role
        assert np.allclose(role_positions, role_positions[0])


@pytest.mark.unit
def test_assign_roles_with_nan() -> None:
    # Arrange — player 0 missing in frame 1
    xy = XY(
        np.array(
            [
                [0.0, 0.0, 10.0, 0.0, 5.0, 10.0],
                [np.nan, np.nan, 10.0, 0.0, 5.0, 10.0],
                [0.0, 0.0, 10.0, 0.0, 5.0, 10.0],
            ]
        ),
        framerate=10,
    )

    # Act
    xy_assigned = assign_roles(xy)

    # Assert — shape preserved, frame 1 has one NaN role slot
    assert xy_assigned.xy.shape == (3, 6)
    # frames 0 and 2 should have no NaN
    assert not np.any(np.isnan(xy_assigned.xy[0]))
    assert not np.any(np.isnan(xy_assigned.xy[2]))
    # frame 1 should have exactly 2 NaN values (one player missing)
    assert np.sum(np.isnan(xy_assigned.xy[1])) == 2


@pytest.mark.unit
def test_assign_roles_custom_reference(example_xy_permutation: XY) -> None:
    # Arrange
    xy = example_xy_permutation
    reference = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])

    # Act
    xy_assigned = assign_roles(xy, reference=reference)

    # Assert — role 0 should always be near (0,0), etc.
    T = len(xy_assigned)
    roles = xy_assigned.xy.reshape(T, -1, 2)
    for t in range(T):
        assert np.allclose(roles[t, 0], [0.0, 0.0])
        assert np.allclose(roles[t, 1], [10.0, 0.0])
        assert np.allclose(roles[t, 2], [5.0, 10.0])


@pytest.mark.unit
def test_assign_roles_invalid_reference(example_xy_spatial: XY) -> None:
    # Arrange
    xy = example_xy_spatial
    wrong_reference = np.array([[0.0, 0.0], [1.0, 1.0]])  # 2 instead of 3

    # Act & Assert
    with pytest.raises(ValueError, match="Expected reference of shape"):
        assign_roles(xy, reference=wrong_reference)


@pytest.mark.unit
def test_assign_roles_n_iter(example_xy_permutation: XY) -> None:
    # Arrange
    xy = example_xy_permutation

    # Act
    xy_iter1 = assign_roles(xy, n_iter=1)
    xy_iter2 = assign_roles(xy, n_iter=2)

    # Assert — both produce valid output with correct shape
    assert xy_iter1.xy.shape == xy.xy.shape
    assert xy_iter2.xy.shape == xy.xy.shape
    # n_iter=2 should also produce consistent roles
    T = len(xy_iter2)
    roles = xy_iter2.xy.reshape(T, -1, 2)
    for role_idx in range(3):
        role_positions = roles[:, role_idx, :]
        assert np.allclose(role_positions, role_positions[0])

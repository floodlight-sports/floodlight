import pytest
import numpy as np

from floodlight import XY
from floodlight.metrics.trajectory_clustering import _fsim_score, formation_similarity


@pytest.mark.unit
def test_fsim_score_identical() -> None:
    # Arrange — identical query and template
    formation = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

    # Act
    score = _fsim_score(formation, formation, delta=1 / 3)

    # Assert
    assert score == 1.0


@pytest.mark.unit
def test_fsim_score_distant() -> None:
    # Arrange — very different formations
    query = np.array([[0.0, 0.0], [0.1, 0.0], [0.05, 0.1]])
    template = np.array([[0.9, 0.9], [1.0, 0.9], [0.95, 1.0]])

    # Act
    score = _fsim_score(query, template, delta=1 / 3)

    # Assert
    assert score == 0.0


@pytest.mark.unit
def test_formation_similarity_perfect_match(example_xy_fsim: XY) -> None:
    # Arrange — template matches the formation in the XY data
    # The fixture has a rectangular formation at roughly (10,30), (30,30),
    # (10,60), (30,60). After centering and normalizing, a matching template
    # should yield a high score.
    template = np.array([[10, 30], [30, 30], [10, 60], [30, 60]], dtype=float)

    # Act
    score = formation_similarity(example_xy_fsim, template)

    # Assert
    assert score == 1.0


@pytest.mark.unit
def test_formation_similarity_exclude_xids(example_xy_fsim: XY) -> None:
    # Arrange — exclude player 0, template for remaining 3 players
    template_3 = np.array([[30, 30], [10, 60], [30, 60]], dtype=float)

    # Act
    score = formation_similarity(example_xy_fsim, template_3, exclude_xIDs=[0])

    # Assert
    assert score == 1.0


@pytest.mark.unit
def test_formation_similarity_template_shape_error(example_xy_fsim: XY) -> None:
    # Arrange — wrong template shape (3 instead of 4 players)
    wrong_template = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)

    # Act & Assert
    with pytest.raises(ValueError, match="Template has shape"):
        formation_similarity(example_xy_fsim, wrong_template)


@pytest.mark.unit
def test_formation_similarity_no_role_assignment(example_xy_fsim: XY) -> None:
    # Arrange
    template = np.array([[10, 30], [30, 30], [10, 60], [30, 60]], dtype=float)

    # Act
    score = formation_similarity(example_xy_fsim, template, role_assignment=False)

    # Assert
    assert score == 1.0

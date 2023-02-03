import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from floodlight.core.pitch import Pitch
from floodlight.models.space import DiscreteVoronoiModel


# tests for DiscreteVoronoiModel (dvm)
@pytest.mark.unit
def test_dvm_constructor(example_pitch_dfl) -> None:
    # create sample pitch
    pitch = example_pitch_dfl

    # trigger constructor checks
    with pytest.raises(ValueError):
        model = DiscreteVoronoiModel(pitch, mesh="foo")
    with pytest.raises(ValueError):
        model = DiscreteVoronoiModel(pitch, xpoints=9)
    with pytest.raises(ValueError):
        model = DiscreteVoronoiModel(pitch, xpoints=1001)

    # check correct executing of post_init
    model = DiscreteVoronoiModel(pitch, xpoints=10)
    assert not model.is_fitted
    assert model._meshx_ is not None
    assert model._meshy_ is not None


# test mesh generation method with different mesh types
@pytest.mark.unit
@pytest.mark.filterwarnings("ignore: Model initialized with non-metrical pitch.")
def test_generate_mesh_square() -> None:
    xpoints = 10
    pitch_opta = Pitch.from_template("opta", sport="football", length=109, width=69)
    pitch_statsbomb = Pitch.from_template(
        "statsbomb", sport="football", length=109, width=69
    )
    pitch_dfl = Pitch.from_template("dfl", sport="football", length=109, width=69)
    pitch_tracab = Pitch.from_template("tracab", sport="football", length=109, width=69)
    pitch_statsperform = Pitch.from_template(
        "statsperform_tracking", sport="football", length=109, width=69
    )

    # Opta
    model = DiscreteVoronoiModel(pitch_opta, mesh="square", xpoints=xpoints)
    assert np.allclose(
        np.array(
            [[5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0]]
            * model._meshx_.shape[0]
        ),
        model._meshx_,
    )
    assert np.allclose(
        np.array(
            [[95.0, 85.0, 75.0, 65.0, 55.0, 45.0, 35.0, 25.0, 15.0, 5.0]]
            * model._meshy_.shape[1]
        ).transpose(),
        model._meshy_,
    )
    # StatsBomb
    model = DiscreteVoronoiModel(pitch_statsbomb, mesh="square", xpoints=xpoints)
    assert np.allclose(
        np.array(
            [[6.0, 18.0, 30.0, 42.0, 54.0, 66.0, 78.0, 90.0, 102.0, 114.0]]
            * model._meshx_.shape[0]
        ),
        model._meshx_,
    )
    assert np.allclose(
        np.array(
            [
                [
                    74.28571429,
                    62.85714286,
                    51.42857143,
                    40.0,
                    28.57142857,
                    17.14285714,
                    5.71428571,
                ]
            ]
            * model._meshy_.shape[1]
        ).transpose(),
        model._meshy_,
    )
    # DFL
    model = DiscreteVoronoiModel(pitch_dfl, mesh="square", xpoints=xpoints)
    assert np.allclose(
        np.array(
            [[-49.05, -38.15, -27.25, -16.35, -5.45, 5.45, 16.35, 27.25, 38.15, 49.05]]
            * model._meshx_.shape[0]
        ),
        model._meshx_,
    )
    assert np.allclose(
        np.array(
            [[28.75, 17.25, 5.75, -5.75, -17.25, -28.75]] * model._meshy_.shape[1]
        ).transpose(),
        model._meshy_,
    )
    # Tracab
    model = DiscreteVoronoiModel(pitch_tracab, mesh="square", xpoints=xpoints)
    assert np.allclose(
        np.array(
            [
                [
                    -4905.0,
                    -3815.0,
                    -2725.0,
                    -1635.0,
                    -545.0,
                    545.0,
                    1635.0,
                    2725.0,
                    3815.0,
                    4905.0,
                ]
            ]
            * model._meshx_.shape[0]
        ),
        model._meshx_,
    )
    assert np.allclose(
        np.array(
            [[2875.0, 1725.0, 575.0, -575.0, -1725.0, -2875.0]] * model._meshy_.shape[1]
        ).transpose(),
        model._meshy_,
    )
    # Statsperform
    model = DiscreteVoronoiModel(pitch_statsperform, mesh="square", xpoints=xpoints)
    assert np.allclose(
        np.array(
            [[5.45, 16.35, 27.25, 38.15, 49.05, 59.95, 70.85, 81.75, 92.65, 103.55]]
            * model._meshx_.shape[0]
        ),
        model._meshx_,
    )
    assert np.allclose(
        np.array(
            [[63.25, 51.75, 40.25, 28.75, 17.25, 5.75]] * model._meshy_.shape[1]
        ).transpose(),
        model._meshy_,
    )


# test mesh generation method with different mesh types
@pytest.mark.unit
@pytest.mark.filterwarnings("ignore: Model initialized with non-metrical pitch.")
def test_generate_mesh_hex() -> None:
    xpoints = 10
    pitch_opta = Pitch.from_template("opta", sport="football", length=109, width=69)
    pitch_statsbomb = Pitch.from_template(
        "statsbomb", sport="football", length=109, width=69
    )
    pitch_dfl = Pitch.from_template("dfl", sport="football", length=109, width=69)
    pitch_tracab = Pitch.from_template("tracab", sport="football", length=109, width=69)
    pitch_statsperform = Pitch.from_template(
        "statsperform_tracking", sport="football", length=109, width=69
    )

    # Opta
    model = DiscreteVoronoiModel(pitch_opta, mesh="hexagonal", xpoints=xpoints)
    assert np.allclose(
        np.array(
            [
                [
                    0.0,
                    10.52631579,
                    21.05263158,
                    31.57894737,
                    42.10526316,
                    52.63157895,
                    63.15789474,
                    73.68421053,
                    84.21052632,
                    94.73684211,
                ],
                [
                    5.26315789,
                    15.78947368,
                    26.31578947,
                    36.84210526,
                    47.36842105,
                    57.89473684,
                    68.42105263,
                    78.94736842,
                    89.47368421,
                    100.0,
                ],
            ]
            * int(model._meshx_.shape[0] / 2)
        ),
        model._meshx_,
    )
    assert np.allclose(
        np.array(
            [
                [
                    100.0,
                    90.90909091,
                    81.81818182,
                    72.72727273,
                    63.63636364,
                    54.54545455,
                    45.45454545,
                    36.36363636,
                    27.27272727,
                    18.18181818,
                    9.09090909,
                    0.0,
                ]
            ]
            * model._meshy_.shape[1]
        ).transpose(),
        model._meshy_,
    )
    # StatsBomb
    model = DiscreteVoronoiModel(pitch_statsbomb, mesh="hexagonal", xpoints=xpoints)
    assert np.allclose(
        np.array(
            [
                [
                    0.0,
                    12.63157895,
                    25.26315789,
                    37.89473684,
                    50.52631579,
                    63.15789474,
                    75.78947368,
                    88.42105263,
                    101.05263158,
                    113.68421053,
                ],
                [
                    6.31578947,
                    18.94736842,
                    31.57894737,
                    44.21052632,
                    56.84210526,
                    69.47368421,
                    82.10526316,
                    94.73684211,
                    107.36842105,
                    120.0,
                ],
            ]
            * int(model._meshx_.shape[0] / 2)
        ),
        model._meshx_,
    )
    assert np.allclose(
        np.array(
            [
                [
                    80.0,
                    68.57142857,
                    57.14285714,
                    45.71428571,
                    34.28571429,
                    22.85714286,
                    11.42857143,
                    0.0,
                ]
            ]
            * model._meshy_.shape[1]
        ).transpose(),
        model._meshy_,
    )
    # DFL
    model = DiscreteVoronoiModel(pitch_dfl, mesh="hexagonal", xpoints=xpoints)
    assert np.allclose(
        np.array(
            [
                [
                    -54.5,
                    -43.02631579,
                    -31.55263158,
                    -20.07894737,
                    -8.60526316,
                    2.86842105,
                    14.34210526,
                    25.81578947,
                    37.28947368,
                    48.76315789,
                ],
                [
                    -48.76315789,
                    -37.28947368,
                    -25.81578947,
                    -14.34210526,
                    -2.86842105,
                    8.60526316,
                    20.07894737,
                    31.55263158,
                    43.02631579,
                    54.5,
                ],
            ]
            * int(model._meshx_.shape[0] / 2)
        ),
        model._meshx_,
    )
    assert np.allclose(
        np.array(
            [
                [
                    34.5,
                    24.64285714,
                    14.78571429,
                    4.92857143,
                    -4.92857143,
                    -14.78571429,
                    -24.64285714,
                    -34.5,
                ]
            ]
            * model._meshy_.shape[1]
        ).transpose(),
        model._meshy_,
    )
    # Tracab
    model = DiscreteVoronoiModel(pitch_tracab, mesh="hexagonal", xpoints=xpoints)
    assert np.allclose(
        np.array(
            [
                [
                    -5450.0,
                    -4302.63157895,
                    -3155.26315789,
                    -2007.89473684,
                    -860.52631579,
                    286.84210526,
                    1434.21052632,
                    2581.57894737,
                    3728.94736842,
                    4876.31578947,
                ],
                [
                    -4876.31578947,
                    -3728.94736842,
                    -2581.57894737,
                    -1434.21052632,
                    -286.84210526,
                    860.52631579,
                    2007.89473684,
                    3155.26315789,
                    4302.63157895,
                    5450.0,
                ],
            ]
            * int(model._meshx_.shape[0] / 2)
        ),
        model._meshx_,
    )
    assert np.allclose(
        np.array(
            [
                [
                    3450.0,
                    2464.28571429,
                    1478.57142857,
                    492.85714286,
                    -492.85714286,
                    -1478.57142857,
                    -2464.28571429,
                    -3450.0,
                ]
            ]
            * model._meshy_.shape[1]
        ).transpose(),
        model._meshy_,
    )
    # Statsperform
    model = DiscreteVoronoiModel(pitch_statsperform, mesh="hexagonal", xpoints=xpoints)
    assert np.allclose(
        np.array(
            [
                [
                    0.0,
                    11.47368421,
                    22.94736842,
                    34.42105263,
                    45.89473684,
                    57.36842105,
                    68.84210526,
                    80.31578947,
                    91.78947368,
                    103.26315789,
                ],
                [
                    5.73684211,
                    17.21052632,
                    28.68421053,
                    40.15789474,
                    51.63157895,
                    63.10526316,
                    74.57894737,
                    86.05263158,
                    97.52631579,
                    109.0,
                ],
            ]
            * int(model._meshx_.shape[0] / 2)
        ),
        model._meshx_,
    )
    assert np.allclose(
        np.array(
            [
                [
                    69.0,
                    59.14285714,
                    49.28571429,
                    39.42857143,
                    29.57142857,
                    19.71428571,
                    9.85714286,
                    0.0,
                ]
            ]
            * model._meshy_.shape[1]
        ).transpose(),
        model._meshy_,
    )


# test calculation of controls with euclidean distance
@pytest.mark.unit
def test_calc_cell_controls_euclidean_square(
    example_xy_objects_space_control, example_pitch_dfl
) -> None:
    xy1, xy2 = example_xy_objects_space_control
    pitch = example_pitch_dfl
    model_square = DiscreteVoronoiModel(pitch, mesh="square", xpoints=10)
    model_square.fit(xy1, xy2)

    assert np.array_equal(
        np.array(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 5.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0],
            ]
        ),
        model_square._cell_controls_[0],
    )
    assert np.array_equal(
        np.array(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 5.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0],
            ]
        ),
        model_square._cell_controls_[1],
    )


@pytest.mark.unit
def test_calc_cell_controls_euclidean_hex(
    example_xy_objects_space_control, example_pitch_dfl
) -> None:
    xy1, xy2 = example_xy_objects_space_control
    pitch = example_pitch_dfl
    model_hex = DiscreteVoronoiModel(pitch, mesh="hexagonal", xpoints=10)
    model_hex.fit(xy1, xy2)

    assert np.array_equal(
        np.array(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 5.0, 3.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0],
            ]
        ),
        model_hex._cell_controls_[0],
    )
    assert np.array_equal(
        np.array(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 5.0, 3.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0],
            ]
        ),
        model_hex._cell_controls_[1],
    )


# test calculation of player areas
@pytest.mark.unit
def test_player_controls_square(
    example_xy_objects_space_control, example_pitch_dfl
) -> None:
    xy1, xy2 = example_xy_objects_space_control
    pitch = example_pitch_dfl
    model_square = DiscreteVoronoiModel(pitch, mesh="square", xpoints=10)
    model_square.fit(xy1, xy2)

    areas1, areas2 = model_square.player_controls()

    assert np.array_equal(
        areas1.property, np.array([[34.0, 4.0, 16.0], [30.0, 8.0, 16.0]])
    )
    assert np.array_equal(
        areas2.property, np.array([[30.0, 0.0, 16.0], [30.0, 0.0, 16.0]])
    )
    assert np.allclose(
        np.sum(areas1, axis=1) + np.sum(areas2, axis=1),
        np.array([100.0, 100.0]),
        atol=0.5,
    )


@pytest.mark.unit
def test_player_controls_hex(
    example_xy_objects_space_control, example_pitch_dfl
) -> None:
    xy1, xy2 = example_xy_objects_space_control
    pitch = example_pitch_dfl
    model_hex = DiscreteVoronoiModel(pitch, mesh="hexagonal", xpoints=10)
    model_hex.fit(xy1, xy2)

    areas1, areas2 = model_hex.player_controls()

    assert np.array_equal(
        areas1.property, np.array([[33.33, 8.33, 11.67], [33.33, 6.67, 15.0]])
    )
    assert np.array_equal(
        areas2.property, np.array([[33.33, 0.0, 13.33], [31.67, 0.0, 13.33]])
    )
    assert np.allclose(
        np.sum(areas1, axis=1) + np.sum(areas2, axis=1),
        np.array([100.0, 100.0]),
        atol=0.5,
    )


# test calculation of team areas
@pytest.mark.unit
def test_team_controls_square(
    example_xy_objects_space_control, example_pitch_dfl
) -> None:
    xy1, xy2 = example_xy_objects_space_control
    pitch = example_pitch_dfl
    model_square = DiscreteVoronoiModel(pitch, mesh="square", xpoints=10)
    model_square.fit(xy1, xy2)

    areas1, areas2 = model_square.team_controls()

    assert np.array_equal(areas1.property, np.array([[54.0], [54.0]]))
    assert np.array_equal(areas2.property, np.array([[46.0], [46.0]]))
    assert np.allclose(
        np.sum(areas1, axis=1) + np.sum(areas2, axis=1),
        np.array([100.0, 100.0]),
        atol=0.5,
    )


@pytest.mark.unit
def test_team_controls_hex(example_xy_objects_space_control, example_pitch_dfl) -> None:
    xy1, xy2 = example_xy_objects_space_control
    pitch = example_pitch_dfl
    model_hex = DiscreteVoronoiModel(pitch, mesh="hexagonal", xpoints=10)
    model_hex.fit(xy1, xy2)

    areas1, areas2 = model_hex.team_controls()

    assert np.array_equal(areas1.property, np.array([[53.33], [55.0]]))
    assert np.array_equal(areas2.property, np.array([[46.67], [45.0]]))
    assert np.allclose(
        np.sum(areas1, axis=1) + np.sum(areas2, axis=1),
        np.array([100.0, 100.0]),
        atol=0.5,
    )


# test plotting
@pytest.mark.plot
def test_plot_square(example_xy_objects_space_control, example_pitch_dfl) -> None:
    # get data
    xy1, xy2 = example_xy_objects_space_control
    pitch = example_pitch_dfl
    model_square = DiscreteVoronoiModel(pitch, mesh="square", xpoints=10)
    model_square.fit(xy1, xy2)

    # create plot
    fig, ax = plt.subplots()
    pitch.plot(ax=ax)
    # plot with all variables and kwarg
    model_square.plot(t=1, team_colors=("red", "blue"), ax=ax, ec="black")

    # assert plot generation
    assert isinstance(ax, matplotlib.axes.Axes)

    # assert rectangle generation
    plotted_rectangles = 0
    for patch in plt.gca().patches:
        if isinstance(patch, matplotlib.patches.Rectangle):
            plotted_rectangles += 1
    assert plotted_rectangles == 50

    plt.close()


@pytest.mark.plot
def test_plot_hex(example_xy_objects_space_control, example_pitch_dfl) -> None:
    # get data
    xy1, xy2 = example_xy_objects_space_control
    pitch = example_pitch_dfl
    model_hex = DiscreteVoronoiModel(pitch, mesh="hexagonal", xpoints=10)
    model_hex.fit(xy1, xy2)

    # create plot
    fig, ax = plt.subplots()
    pitch.plot(ax=ax)
    # plot with all variables and kwarg
    model_hex.plot(t=1, team_colors=("red", "blue"), ax=ax, ec="black")

    # assert plot generation
    assert isinstance(ax, matplotlib.axes.Axes)

    # assert recangle generation
    plotted_polygons = 0
    for patch in plt.gca().patches:
        if isinstance(patch, matplotlib.patches.RegularPolygon):
            plotted_polygons += 1
    assert plotted_polygons == 60

    plt.close()


@pytest.mark.plot
def test_plot_mesh(example_xy_objects_space_control, example_pitch_dfl) -> None:
    # get data
    xy1, xy2 = example_xy_objects_space_control
    pitch = example_pitch_dfl
    model_hex = DiscreteVoronoiModel(pitch, mesh="hexagonal", xpoints=10)
    model_hex.fit(xy1, xy2)

    # create plot
    fig, ax = plt.subplots()
    pitch.plot(ax=ax)
    model_hex.plot_mesh(ax=ax)

    # assert plot generation
    assert isinstance(ax, matplotlib.axes.Axes)

    plt.close()

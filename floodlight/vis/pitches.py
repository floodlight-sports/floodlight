from typing import Tuple

import matplotlib
from matplotlib.patches import Arc, Wedge, Rectangle

from floodlight.utils.types import Numeric


def plot_handball_pitch(
    xlim: Tuple[Numeric, Numeric],
    ylim: Tuple[Numeric, Numeric],
    unit: str,
    color_scheme: str,
    show_axis_ticks: bool,
    ax: matplotlib.axes,
    **kwargs,
) -> matplotlib.axes:
    """Plots a handball pitch on a given matplotlib.axes.

    Parameters
    ----------
    xlim: Tuple[Numeric, Numeric]
        Limits of pitch boundaries in longitudinal direction. This tuple has the form
        (x_min, x_max) and delimits the length of the pitch (not of any actual data)
        within the coordinate system.
    ylim: Tuple[Numeric, Numeric]
        Limits of pitch boundaries in lateral direction. This tuple has the form
        (y_min, y_max) and delimits the width of the pitch (not of any actual data)
        within the coordinate system.
    unit: str
        The unit in which data is measured along axes. Possible types are
        {'m', 'cm', 'percent'}.
    color_scheme: str
        Color scheme of the plot. One of {'standard', 'bw'}.
    show_axis_ticks: bool
        If set to True, the axis ticks are visible.
    ax: matplotlib.axes
        Axes from matplotlib library on which the handball field is plotted.
    kwargs:
        Optional keyworded arguments {'linewidth', 'zorder', 'scalex', 'scaley'}
        which can be used for the plot functions from matplotlib. The kwargs are
        only passed to all the plot functions of matplotlib.

    Returns
    -------
    axes : matplotlib.axes
        Axes from matplotlib library on which a handball pitch is plotted.

    Notes
    -----
    The kwargs are only passed to the plot functions of matplotlib. To customize the
    plots have a look at `matplotlib
    <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.
    For example in order to modify the linewidth pass a float to the keyworded
    argument 'linewidth'. The same principle applies to other kwargs like 'zorder',
    'scalex' and 'scaley'.

    .. _handball-pitch-label:

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from floodlight.vis.pitches import plot_handball_pitch
    >>> # create matplotlib.axes
    >>> ax = plt.subplots()[1]

    >>> # plot handball pitch
    >>> plot_handball_pitch(xlim=(0,40), ylim=(0,20), unit='m', color_scheme='standard',
    >>>                     show_axis_ticks=False, ax=ax)
    >>> plt.show()

    .. image:: ../../_img/pitch_handball_example.png

    """

    # kwargs which are used to configure the plot with default values 1 and 0.
    # all the other kwargs will be just passed to all the plot functions.
    linewidth = kwargs.pop("linewidth", 1)
    zorder = kwargs.pop("zorder", 0)

    # customizing visualization
    if color_scheme == "bw":
        ax.set_facecolor("white")  # color of the pitch
        color_contour_lines = "black"
        color_goal_area_fill = "white"
        color_goal_posts = "black"
    else:
        ax.set_facecolor("skyblue")  # color of the pitch
        color_contour_lines = "white"
        color_goal_area_fill = "khaki"
        color_goal_posts = "white"

    # Positions and ranges on the field which are used for plotting all lines and
    # arcs.
    # All the positions and ranges are scaled based on percentages of the x-range
    # and y-range.

    # key positions for the handball pitch
    xmin, xmax = xlim
    ymin, ymax = ylim
    x_range = xmax - xmin
    y_range = ymax - ymin
    x_half = (xmax + xmin) / 2
    y_half = (ymax + ymin) / 2

    # Positions (x,y) on the handball pitch
    pos_left_side_right_goal_post = (xmin, ymin + y_range * 0.425)
    pos_left_side_left_goal_post = (xmin, ymin + y_range * 0.575)
    pos_right_side_left_goal_post = (xmax, ymin + y_range * 0.425)
    pos_right_side_right_goal_post = (xmax, ymin + y_range * 0.575)
    pos_start_right_side_rect_goal_area = (
        xmax - x_range * 0.15,
        ymin + y_range * 0.425,
    )

    # Ranges on the handball pitch
    y_height_free_throw_arc = y_range * 0.9
    x_width_free_throw_arc = x_range * 0.45
    y_height_goal_area_arc = y_range * 0.6
    x_width_goal_area_arc = x_range * 0.3
    radius_goal_area_arc = x_range * 0.15
    width_goal = y_range * 0.15
    x_height_goal = x_range * 0.025

    # Y positions on the handball pitch
    y_range_center_to_post = width_goal / 2
    y_pos_goal_lower_post = y_half - y_range_center_to_post
    y_pos_goal_upper_post = y_half + y_range_center_to_post
    y_pos_lower_goal_posts = ymin + y_range * 0.425
    y_pos_upper_goal_posts = ymin + y_range * 0.575
    lower_y_pos_4m_line = ymin + (y_range * 0.49625)
    upper_y_pos_4m_line = ymin + (y_range * 0.50375)
    lower_y_pos_7m_line = ymin + (y_range * 0.475)
    upper_y_pos_7m_line = ymin + (y_range * 0.525)

    # X positions on the handball pitch
    x_pos_left_side_goal_area_line_edge = xmin + (x_range * 0.15)
    x_pos_right_side_goal_area_line_edge = xmax - (x_range * 0.15)
    x_pos_left_side_free_throw_line_edge = xmin + (x_range * 0.225)
    x_pos_right_side_free_throw_line_edge = xmax - (x_range * 0.225)
    x_pos_left_side_4m_line = xmin + (x_range * 0.1)
    x_pos_right_side_4m_line = xmax - (x_range * 0.1)
    x_pos_left_side_7m_line = xmin + (x_range * 0.175)
    x_pos_right_side_7m_line = xmax - (x_range * 0.175)

    # angle for the free throw arc changes when unit is 'percent'
    angle = 10 if unit == "percent" else 0

    # margins of the plot
    x_margin = x_range * 0.025
    y_margin = y_range * 0.05

    # set up the boundaries of ax
    ax.set_xlim([xmin - (x_height_goal + x_margin), xmax + (x_height_goal + x_margin)])
    ax.set_ylim([ymin - y_margin, ymax + y_margin])

    # paint handball pitch with all properties
    # main boundaries
    ax.plot(
        [xmin, xmin],
        [ymin, ymax],
        color=color_contour_lines,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    ax.plot(
        [xmax, xmax],
        [ymin, ymax],
        color=color_contour_lines,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    ax.plot(
        [xmin, xmax],
        [ymin, ymin],
        color=color_contour_lines,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    ax.plot(
        [xmin, xmax],
        [ymax, ymax],
        color=color_contour_lines,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )

    # midline
    ax.plot(
        [x_half, x_half],
        [ymin, ymax],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
        **kwargs,
    )

    # free-throw lines
    # lower left
    ax.add_patch(
        Arc(
            pos_left_side_right_goal_post,
            width=x_width_free_throw_arc,
            height=y_height_free_throw_arc,
            angle=0,
            theta1=290 - angle,
            theta2=360,
            linestyle="dashed",
            linewidth=linewidth,
            color=color_contour_lines,
            zorder=zorder,
            **kwargs,
        )
    )
    # upper left
    ax.add_patch(
        Arc(
            pos_left_side_left_goal_post,
            width=x_width_free_throw_arc,
            height=y_height_free_throw_arc,
            angle=0,
            theta1=0,
            theta2=70 + angle,
            linestyle="dashed",
            linewidth=linewidth,
            color=color_contour_lines,
            zorder=zorder,
            **kwargs,
        )
    )
    # lower right
    ax.add_patch(
        Arc(
            pos_right_side_left_goal_post,
            width=x_width_free_throw_arc,
            height=y_height_free_throw_arc,
            angle=0,
            theta1=180,
            theta2=250 + angle,
            linestyle="dashed",
            linewidth=linewidth,
            color=color_contour_lines,
            zorder=zorder,
            **kwargs,
        )
    )
    # upper right
    ax.add_patch(
        Arc(
            pos_right_side_right_goal_post,
            width=x_width_free_throw_arc,
            height=y_height_free_throw_arc,
            angle=0,
            theta1=110 - angle,
            theta2=180,
            linestyle="dashed",
            linewidth=linewidth,
            color=color_contour_lines,
            zorder=zorder,
            **kwargs,
        )
    )
    # goal area lines
    # lower left
    # Filling the goal area
    if unit != "percent":  # Since wedges can't be scaled
        ax.add_patch(
            Wedge(
                pos_left_side_right_goal_post,
                r=radius_goal_area_arc,
                theta1=270,
                theta2=360,
                linewidth=linewidth,
                color=color_goal_area_fill,
                zorder=zorder,
                **kwargs,
            )
        )
    # goal area line
    ax.add_patch(
        Arc(
            pos_left_side_right_goal_post,
            height=y_height_goal_area_arc,
            width=x_width_goal_area_arc,
            angle=0,
            theta1=270,
            theta2=360,
            linewidth=linewidth,
            color=color_contour_lines,
            zorder=zorder,
            **kwargs,
        )
    )
    # upper left
    # filling the goal area
    if unit != "percent":  # Since wedges can't be scaled
        ax.add_patch(
            Wedge(
                pos_left_side_left_goal_post,
                r=radius_goal_area_arc,
                theta1=0,
                theta2=90,
                linewidth=linewidth,
                color=color_goal_area_fill,
                zorder=zorder,
                **kwargs,
            )
        )
    # goal area line
    ax.add_patch(
        Arc(
            pos_left_side_left_goal_post,
            height=y_height_goal_area_arc,
            width=x_width_goal_area_arc,
            angle=0,
            theta1=0,
            theta2=90,
            linewidth=linewidth,
            color=color_contour_lines,
            zorder=zorder,
            **kwargs,
        )
    )
    # lower right
    # filling the goal area
    if unit != "percent":  # Since wedges can't be scaled
        ax.add_patch(
            Wedge(
                pos_right_side_left_goal_post,
                r=radius_goal_area_arc,
                theta1=180,
                theta2=270,
                linewidth=linewidth,
                color=color_goal_area_fill,
                zorder=zorder,
                **kwargs,
            )
        )
    # goal area line
    ax.add_patch(
        Arc(
            pos_right_side_left_goal_post,
            height=y_height_goal_area_arc,
            width=x_width_goal_area_arc,
            angle=0,
            theta1=180,
            theta2=270,
            linewidth=linewidth,
            color=color_contour_lines,
            zorder=zorder,
            **kwargs,
        )
    )
    # upper right
    # filling the goal area
    if unit != "percent":  # Since wedges can't be scaled
        ax.add_patch(
            Wedge(
                pos_right_side_right_goal_post,
                r=radius_goal_area_arc,
                theta1=90,
                theta2=180,
                linewidth=linewidth,
                color=color_goal_area_fill,
                zorder=zorder,
                **kwargs,
            )
        )
    # goal area line
    ax.add_patch(
        Arc(
            pos_right_side_right_goal_post,
            height=y_height_goal_area_arc,
            width=x_width_goal_area_arc,
            angle=0,
            theta1=90,
            theta2=180,
            linewidth=linewidth,
            color=color_contour_lines,
            zorder=zorder,
            **kwargs,
        )
    )
    # mid left
    # filling the goal area
    if unit != "percent":  # Since wedges can't be scaled the rectangles
        ax.add_patch(  # are also not used to fill the goal area
            Rectangle(
                pos_left_side_right_goal_post,
                width=radius_goal_area_arc,
                height=width_goal,
                color=color_goal_area_fill,
                zorder=zorder,
                **kwargs,
            )
        )
    # mid right
    # filling the goal area
    if unit != "percent":  # Since wedges can't be scaled the rectangles
        ax.add_patch(  # are also not used to fill the goal area
            Rectangle(
                pos_start_right_side_rect_goal_area,
                width=radius_goal_area_arc,
                height=width_goal,
                color=color_goal_area_fill,
                zorder=zorder,
                **kwargs,
            )
        )

    # vertical goal area lines
    ax.plot(
        [x_pos_left_side_goal_area_line_edge, x_pos_left_side_goal_area_line_edge],
        [y_pos_lower_goal_posts, y_pos_upper_goal_posts],
        color=color_contour_lines,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    ax.plot(
        [
            x_pos_right_side_goal_area_line_edge,
            x_pos_right_side_goal_area_line_edge,
        ],
        [y_pos_lower_goal_posts, y_pos_upper_goal_posts],
        color=color_contour_lines,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    # vertical free throw lines
    ax.plot(
        [
            x_pos_left_side_free_throw_line_edge,
            x_pos_left_side_free_throw_line_edge,
        ],
        [y_pos_lower_goal_posts, y_pos_upper_goal_posts],
        color=color_contour_lines,
        linewidth=linewidth,
        linestyle="dashed",
        zorder=zorder,
        **kwargs,
    )
    ax.plot(
        [
            x_pos_right_side_free_throw_line_edge,
            x_pos_right_side_free_throw_line_edge,
        ],
        [y_pos_lower_goal_posts, y_pos_upper_goal_posts],
        color=color_contour_lines,
        linewidth=linewidth,
        linestyle="dashed",
        zorder=zorder,
        **kwargs,
    )

    # 4 m lines
    ax.plot(
        [x_pos_left_side_4m_line, x_pos_left_side_4m_line],
        [lower_y_pos_4m_line, upper_y_pos_4m_line],
        color=color_contour_lines,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    ax.plot(
        [x_pos_right_side_4m_line, x_pos_right_side_4m_line],
        [lower_y_pos_4m_line, upper_y_pos_4m_line],
        color=color_contour_lines,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )

    # 7 m lines
    ax.plot(
        [x_pos_left_side_7m_line, x_pos_left_side_7m_line],
        [lower_y_pos_7m_line, upper_y_pos_7m_line],
        color=color_contour_lines,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    ax.plot(
        [x_pos_right_side_7m_line, x_pos_right_side_7m_line],
        [lower_y_pos_7m_line, upper_y_pos_7m_line],
        color=color_contour_lines,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )

    # goals
    # left goal
    ax.plot(
        [xmin - x_height_goal, xmin - x_height_goal],
        [y_pos_goal_lower_post, y_pos_goal_upper_post],
        color=color_goal_posts,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    ax.plot(
        [xmin - x_height_goal, xmin],
        [y_pos_goal_lower_post, y_pos_goal_lower_post],
        color=color_goal_posts,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    ax.plot(
        [xmin - x_height_goal, xmin],
        [y_pos_goal_upper_post, y_pos_goal_upper_post],
        color=color_goal_posts,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    # right goal
    ax.plot(
        [xmax + x_height_goal, xmax + x_height_goal],
        [y_pos_goal_lower_post, y_pos_goal_upper_post],
        color=color_goal_posts,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    ax.plot(
        [xmax + x_height_goal, xmax],
        [y_pos_goal_lower_post, y_pos_goal_lower_post],
        color=color_goal_posts,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    ax.plot(
        [xmax + x_height_goal, xmax],
        [y_pos_goal_upper_post, y_pos_goal_upper_post],
        color=color_goal_posts,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    # baseline
    ax.plot(
        [xmax, xmax],
        [ymin, ymax],
        color=color_contour_lines,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )
    # baseline
    ax.plot(
        [xmin, xmin],
        [ymin, ymax],
        color=color_contour_lines,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )

    # remove labels and ticks
    if not show_axis_ticks:
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

    return ax


def plot_football_pitch(
    xlim: Tuple[Numeric, Numeric],
    ylim: Tuple[Numeric, Numeric],
    length: Numeric,
    width: Numeric,
    unit: str,
    color_scheme: str,
    show_axis_ticks: bool,
    ax: matplotlib.axes,
    **kwargs,
) -> matplotlib.axes:
    """Plots a football pitch on a given matplotlib.axes.

    Parameters
    ----------
    xlim: Tuple[Numeric, Numeric]
        Limits of pitch boundaries in longitudinal direction. This tuple has the form
        (x_min, x_max) and delimits the length of the pitch (not of any actual data)
        within the coordinate system.
    ylim: Tuple[Numeric, Numeric]
        Limits of pitch boundaries in lateral direction. This tuple has the form
        (y_min, y_max) and delimits the width of the pitch (not of any actual data)
        within the coordinate system.
    length: Numeric
        Length of the actual pitch in `unit`.
    width: Numeric, optional
        Width of the actual pitch in `unit`.
    unit: str
        The unit in which data is measured along axes. Possible types are
        {'m', 'cm', 'percent'}.
    color_scheme: str
        Color scheme of the plot. One of {'standard', 'bw'}.
    show_axis_ticks: bool
        If set to True, the axis ticks are visible.
    ax: matplotlib.axes
        Axes from matplotlib library on which the football field is plotted.
    kwargs:
        Optional keyworded arguments {'linewidth', 'zorder', 'scalex', 'scaley'}
        which can be used for the plot functions from matplotlib. The kwargs are
        only passed to all the plot functions of matplotlib.

    Returns
    -------
    axes : matplotlib.axes
        Axes from matplotlib library on which a football pitch is plotted.

    Notes
    -----
    The kwargs are only passed to the plot functions of matplotlib. To customize the
    plots have a look at `matplotlib
    <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.
    For example in order to modify the linewidth pass a float to the keyworded
    argument 'linewidth'. The same principle applies to other kwargs like 'zorder',
    'scalex' and 'scaley'.

    .. _football-pitch-label:

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from floodlight.vis.pitches import plot_football_pitch
    >>> # create matplotlib.axes
    >>> ax = plt.subplots()[1]

    >>> # plot handball pitch
    >>> plot_football_pitch(xlim=(0,108), ylim=(0,68), length=108, width=68, unit='m',
    >>>                     color_scheme='standard', show_axis_ticks=False, ax=ax)
    >>> plt.show()

    .. image:: ../../_img/pitch_football_example.png

    """
    # kwargs which are used to configure the plot with default values 1 and 0.
    # all the other kwargs will be just passed to all the plot functions.
    linewidth = kwargs.pop("linewidth", 1)
    zorder = kwargs.pop("zorder", 0)

    # customizing visualization
    if color_scheme == "bw":
        ax.set_facecolor("white")  # color of the pitch
        color_contour_lines = "black"
    else:
        ax.set_facecolor("green")  # color of the pitch
        color_contour_lines = "white"

    # Since football pitch sizes can vary while the elements on the pitch
    # (i.e penalty area) have the same size, the actual elements on the pitch can't
    # be scaled based on the x- or y-range.
    # Therefore a norm factor for the x and y direction is needed for every element
    # on the pitch that has a fixed size.
    # The norm factor is specified based on the given unit.
    # If the unit is 'm' or 'cm' the ratio between x and y is set to 1
    # (see in the Pitch.plot() method ax.set_aspect(1)) and the norm factos are set
    # to 1 ('m') or 100 ('cm'). But if the unit is 'percent' the ratio between
    # width/length is set to ax.set_aspect(width/length).
    # That means if an element like the goal area, is drawn, it get's rescaled based
    # on the ratio of width and length.

    # norm_factor for all elements on the pitch that are scaled in the x direction
    norm_factor_x = (
        1
        if unit == "m"
        else 100
        if unit == "cm"
        else 100 / length
        if length
        else 100 / 105
    )
    # norm_factor for all elements on the pitch that are scaled in the y direction
    norm_factor_y = (
        1
        if unit == "m"
        else 100
        if unit == "cm"
        else 100 / width
        if width
        else 100 / 68
    )

    # All the positions and ranges of certain elements on the pitch
    # (i.e the penalty area) have fixed sizes and are scaled based on the
    # x and y norm factors.

    # key positions for the football pitch
    xmin, xmax = xlim
    ymin, ymax = ylim
    x_half = (xmax + xmin) / 2
    y_half = (ymax + ymin) / 2
    x_radius_points = 0.25 * norm_factor_x * 2
    y_radius_points = 0.25 * norm_factor_y * 2
    x_goal_height = 2.44 * norm_factor_x
    y_goal_width = 7.32 * norm_factor_y
    y_range_center_to_post = y_goal_width / 2

    # goal area
    x_pos_left_side_goal_area_line = xmin + 5.5 * norm_factor_x
    x_pos_right_side_goal_area_line = xmax - 5.5 * norm_factor_x
    y_pos_lower_goal_area = y_half - 9.16 * norm_factor_y
    y_pos_upper_goal_area = y_half + 9.16 * norm_factor_y

    # goal
    y_pos_goal_lower_post = y_half - y_range_center_to_post
    y_pos_goal_upper_post = y_half + y_range_center_to_post

    # penalty area
    x_pos_left_side_penalty_area_line = xmin + 16.5 * norm_factor_x
    y_pos_penalty_area_lower_line = y_half - 20.16 * norm_factor_y
    y_pos_penalty_area_upper_line = y_half + 20.16 * norm_factor_y
    x_pos_right_side_penalty_area_line = xmax - 16.5 * norm_factor_x
    x_pos_left_side_center_penalty_arc = xmin + 11 * norm_factor_x
    x_pos_right_side_center_penalty_arc = xmax - 11 * norm_factor_x
    x_pos_left_side_penalty_point = xmin + 11 * norm_factor_x
    x_pos_right_side_penalty_point = xmax - 11 * norm_factor_x

    # center circle
    x_radius_center_circle = 9.15 * norm_factor_x * 2
    y_radius_center_circle = 9.15 * norm_factor_y * 2

    # angle for the penalty area arc changes when unit is 'percent'
    angle = 10 if unit == "percent" else 0

    # field boundaries
    ax.plot(
        [xmin, xmax],
        [ymin, ymin],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmin, xmax],
        [ymax, ymax],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmin, xmin],
        [ymin, ymax],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmax, xmax],
        [ymin, ymax],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [x_half, x_half],
        [ymin, ymax],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )

    # goal area
    # goal area left
    ax.plot(
        [x_pos_left_side_goal_area_line, x_pos_left_side_goal_area_line],
        [y_pos_lower_goal_area, y_pos_upper_goal_area],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmin, x_pos_left_side_goal_area_line],
        [y_pos_lower_goal_area, y_pos_lower_goal_area],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmin, x_pos_left_side_goal_area_line],
        [y_pos_upper_goal_area, y_pos_upper_goal_area],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    # goal area right
    ax.plot(
        [x_pos_right_side_goal_area_line, x_pos_right_side_goal_area_line],
        [y_pos_lower_goal_area, y_pos_upper_goal_area],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmax, x_pos_right_side_goal_area_line],
        [y_pos_lower_goal_area, y_pos_lower_goal_area],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmax, x_pos_right_side_goal_area_line],
        [y_pos_upper_goal_area, y_pos_upper_goal_area],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )

    # penalty area
    # penalty area left
    ax.plot(
        [x_pos_left_side_penalty_area_line, x_pos_left_side_penalty_area_line],
        [y_pos_penalty_area_lower_line, y_pos_penalty_area_upper_line],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmin, x_pos_left_side_penalty_area_line],
        [y_pos_penalty_area_lower_line, y_pos_penalty_area_lower_line],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmin, x_pos_left_side_penalty_area_line],
        [y_pos_penalty_area_upper_line, y_pos_penalty_area_upper_line],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    # penalty area right
    ax.plot(
        [
            x_pos_right_side_penalty_area_line,
            x_pos_right_side_penalty_area_line,
        ],
        [y_pos_penalty_area_lower_line, y_pos_penalty_area_upper_line],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmax, x_pos_right_side_penalty_area_line],
        [y_pos_penalty_area_lower_line, y_pos_penalty_area_lower_line],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmax, x_pos_right_side_penalty_area_line],
        [y_pos_penalty_area_upper_line, y_pos_penalty_area_upper_line],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    # penalty point
    # left
    ax.add_patch(
        Arc(
            (x_pos_left_side_penalty_point, y_half),
            width=x_radius_points,
            height=y_radius_points,
            angle=0,
            theta1=0,
            theta2=360,
            color=color_contour_lines,
            zorder=zorder,
            linewidth=linewidth,
        )
    )
    # right
    ax.add_patch(
        Arc(
            (x_pos_right_side_penalty_point, y_half),
            width=x_radius_points,
            height=y_radius_points,
            angle=0,
            theta1=0,
            theta2=360,
            color=color_contour_lines,
            zorder=zorder,
            linewidth=linewidth,
        )
    )
    # penalty area circle left
    ax.add_patch(
        Arc(
            (x_pos_left_side_center_penalty_arc, y_half),
            width=x_radius_center_circle,
            height=y_radius_center_circle,
            angle=0,
            theta1=307.5 - angle,
            theta2=52.5 + angle,
            color=color_contour_lines,
            zorder=zorder,
            linewidth=linewidth,
        )
    )
    # penalty area circle right
    ax.add_patch(
        Arc(
            (x_pos_right_side_center_penalty_arc, y_half),
            width=x_radius_center_circle,
            height=y_radius_center_circle,
            angle=0,
            theta1=127.5 - angle,
            theta2=232.5 + angle,
            color=color_contour_lines,
            zorder=zorder,
            linewidth=linewidth,
        )
    )
    # goal
    # left goal
    ax.plot(
        [xmin - x_goal_height, xmin - x_goal_height],
        [y_pos_goal_lower_post, y_pos_goal_upper_post],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmin - x_goal_height, xmin],
        [y_pos_goal_upper_post, y_pos_goal_upper_post],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmin - x_goal_height, xmin],
        [y_pos_goal_lower_post, y_pos_goal_lower_post],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    # right goal
    ax.plot(
        [xmax + x_goal_height, xmax + x_goal_height],
        [y_pos_goal_lower_post, y_pos_goal_upper_post],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmax + x_goal_height, xmax],
        [y_pos_goal_upper_post, y_pos_goal_upper_post],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )
    ax.plot(
        [xmax + x_goal_height, xmax],
        [y_pos_goal_lower_post, y_pos_goal_lower_post],
        color=color_contour_lines,
        zorder=zorder,
        linewidth=linewidth,
    )

    # center circle
    ax.add_patch(
        Arc(
            (x_half, y_half),
            width=x_radius_center_circle,
            height=y_radius_center_circle,
            angle=0,
            theta1=0,
            theta2=360,
            color=color_contour_lines,
            zorder=zorder,
            linewidth=linewidth,
        )
    )
    # center point
    ax.add_patch(
        Arc(
            (x_half, y_half),
            width=x_radius_points,
            height=y_radius_points,
            angle=0,
            theta1=0,
            theta2=360,
            color=color_contour_lines,
            zorder=zorder,
            linewidth=linewidth,
        )
    )

    # remove labels and ticks
    if not show_axis_ticks:
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

    return ax

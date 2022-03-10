import os
import warnings
from dataclasses import dataclass
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Wedge, Rectangle

from floodlight.utils.types import Numeric


@dataclass
class Pitch:
    """
    Pitch and coordinate system specifications. Core class of floodlight.

    Attributes
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
        {'m', 'cm', 'percent'}, whereas 'percent' should be used to denote standardized
        pitches where data is scaled along the axes independent of the actual pitch
        size. In this case, you should specify the `length` and `width` attributes.
    boundaries: str
        One of {'fixed', 'flexible'}. Here, 'fixed' denotes coordinate systems that
        limit axes to the respective xlim and ylim. On the contrary, 'flexible'
        coordinate systems do not explicitly specify a limit. Instead, the limit is
        implicitly set by the actual pitch length and width.
    length: Numeric, optional
        Length of the actual pitch in `unit`.
    width: Numeric, optional
        Width of the actual pitch in `unit`.
    sport: str, optional
        Sport for which the pitch is used. This is used to automatically generate lines
        and markings.
    """

    xlim: Tuple[Numeric, Numeric]
    ylim: Tuple[Numeric, Numeric]
    unit: str
    boundaries: str
    length: Numeric = None
    width: Numeric = None
    sport: str = None

    def __str__(self):
        return (
            f"Floodlight Pitch object with axes x = {self.xlim} / y = {self.ylim} "
            f"({self.boundaries}) in [{self.unit}]"
        )

    @classmethod
    def from_template(cls, template_name: str, **kwargs):
        """
        Creates a Pitch object representing common data provider formats.

        Parameters
        ----------
        template_name: str
            The name of the template the pitch should follow. Currently supported are
            {'dfl', 'opta', 'statsperform', 'tracab'}.
        kwargs:
            You may pass optional arguments (`length`, `width`, `sport`} used for class
            instantiation. For some data providers, additional kwargs are needed to
            represent their format correctly. For example, pass the `length` and `width`
            argument to create a Pitch object in the 'chyronhego_international' format.

        Returns
        -------
        pitch: Pitch
            A class instance of the given provider format.
        """
        if template_name == "dfl":
            if "length" not in kwargs or "width" not in kwargs:
                raise TypeError(
                    "For an exact DFL (German Football League) "
                    "Pitch object, `length` and `width` of the pitch need"
                    "to be passed as keyworded arguments"
                )
            x_half = round(kwargs["length"] / 2, 3)
            y_half = round(kwargs["width"] / 2, 3)
            return cls(
                xlim=(-x_half, x_half),
                ylim=(-y_half, y_half),
                unit="m",
                boundaries="flexible",
                length=kwargs.get("length"),
                width=kwargs.get("width"),
                sport=kwargs.get("sport"),
            )
        elif template_name == "opta":
            return cls(
                xlim=(0.0, 100.0),
                ylim=(0.0, 100.0),
                unit="percent",
                boundaries="fixed",
                length=kwargs.get("length"),
                width=kwargs.get("width"),
                sport=kwargs.get("sport"),
            )
        elif template_name == "statsperform":
            if "length" not in kwargs or "width" not in kwargs:
                raise TypeError(
                    "For an exact StatsPerform Pitch object, "
                    "`length` and `width` of the pitch need "
                    "to be passed as keyworded arguments"
                )
            x_half = round(kwargs["length"] / 2, 3)
            y_half = round(kwargs["width"] / 2, 3)
            return cls(
                xlim=(-x_half, x_half),
                ylim=(-y_half, y_half),
                unit="m",
                boundaries="flexible",
                length=kwargs.get("length"),
                width=kwargs.get("width"),
                sport=kwargs.get("sport"),
            )
        elif template_name == "tracab":
            if "length" not in kwargs or "width" not in kwargs:
                raise TypeError(
                    "For an exact TRACAB (ChyronHego, international format) "
                    "Pitch object, `length` and `width` of the pitch need "
                    "to be passed as keyworded arguments"
                )
            x_half = round(kwargs["length"] / 2, 3)
            y_half = round(kwargs["width"] / 2, 3)
            return cls(
                xlim=(-x_half, x_half),
                ylim=(-y_half, y_half),
                unit="cm",
                boundaries="flexible",
                length=kwargs.get("length"),
                width=kwargs.get("width"),
                sport=kwargs.get("sport"),
            )
        else:
            raise ValueError(f"Unsupported template name '{template_name}'")

    @property
    def center(self):
        center = (
            round((self.xlim[0] + self.xlim[1]) / 2, 3),
            round((self.ylim[0] + self.ylim[1]) / 2, 3),
        )
        return center

    def plot(
        self,
        sport: str = None,
        color_scheme: str = None,
        save: bool = False,
        path: str = str(os.path.dirname(os.path.realpath(__file__))),
        show_plot: bool = True,
        show_axis: bool = False,
        ax: plt.axes = None,
        **kwargs,
    ) -> plt.axes:
        """Returns a plot for a given sport (i.e handball, football).

        Parameters
        ----------
        sport: str, optional
            Sport for which the pitch is created. If the sport is already declared in
            the pitch object the sport must not be passed as an argument.
        color_scheme: str, optional
            Color scheme of the plot. One of {'standard', 'bw'}. If not given
            'standard' is the defaulte color scheme.
        save: bool, optional
            You may want to save the graph. If the graphic should be saved 'save' must
            be True. If not given as an argument the pitch is not going to be saved.
        path: str, optional
            Path to which the plot should be saved if 'save' is True. If not given as an
            an argument the default value is the current working directory.
        show_plot: bool, True
            You may want to see the plot. The plot is shown per default. If you don't
            want to see the plot the 'show_plot' variable must be set to False.
        show_axis: bool, optional
            You may want to show the axis. If 'show_axis' is True the axis are going to
            be visible. If not given as an argument the axis are not going to be
            visible.
        ax: plt.axes, optional
            Axes from matplotlib library on which the playing field is plotted. If not
            given as an argument a plt.axes object with standard configurations
            of matplotlib will be created. In order to modify for instance the figsize
            an plt.axes object must be created and passed as an argument.
        kwargs:
            You may pass optional arguments (`linewidth`, `zorder`, 'scalex','scaley'}
            which can be used for the plot functions from matplotlib. The kwargs
            only getting passed to all the plot functions.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            An axes to which all elements of the sport specific pitch are added.
        """
        # list of existing color_schemes and sports
        color_schemes = ["bw", "standard", None]
        sports = ["football", "handball"]
        sport = sport or self.sport

        # check if valide sport was chosen
        if not sport:
            raise ValueError(
                "To visualize a pitch the sport is needed. "
                "For instance pitch.plot(sport = 'handball')"
            )
        if sport not in sports:
            raise ValueError("Choose a valid sport: " + f"{sports}")

        # check if a valide color scheme was chosen
        if color_scheme not in color_schemes:
            raise ValueError(
                "No valid color scheme description. Choose one of: "
                + f"{color_schemes}"
            )

        # check wether an axes to plot is given or if a new axes element
        # has to be created
        ax = ax or plt.subplots()[1]

        # set ratio between x and y values of the plot to ensure that the ratio between
        # length and width is correct regardless of the figsize.
        if self.unit != "percent":
            ax.set_aspect(1)
        # set ratio if unit is percent and sport is football
        elif self.unit == "percent" and (self.sport or sport) == "football":
            if self.length and self.width:
                ax.set_aspect(self.width / self.length)
            # set ratio to standard pitch size of 68/105
            else:
                ax.set_aspect(68 / 105)  # standard ratio of length and width
                warnings.warn(
                    "Since the unit is 'percent' and no information on the actual pitch"
                    " size in terms of 'length' and 'width' is provided the pitch is "
                    "set to default values length: 105 and width: 68"
                )
        # set ratio if unit is percent and sport is handball
        elif self.unit == "percent" and (self.sport or sport) == "handball":
            ax.set_aspect(0.5)

        # create axes with handball pitch
        if sport == "handball":
            return self._plot_handball_pitch(
                color_scheme=color_scheme,
                save=save,
                path=path,
                show_plot=show_plot,
                show_axis=show_axis,
                ax=ax,
                **kwargs,
            )

        # create axes with football pitch
        if sport == "football":
            return self._plot_football_pitch(
                color_scheme=color_scheme,
                save=save,
                path=path,
                show_plot=show_plot,
                show_axis=show_axis,
                ax=ax,
                **kwargs,
            )

    def _plot_handball_pitch(
        self,
        color_scheme: str,
        save: bool,
        path: str,
        show_plot: bool,
        show_axis: bool,
        ax: plt.axes,
        **kwargs,
    ):

        """Plots a handball pitch on a given axes.

        Parameters
        ----------
        color_scheme: str
             Color scheme of the plot. One of {'standard', 'bw'}.If not given 'standard'
              is the default color scheme.
        save: bool
            You may want to save the graph. If the graphic should be saved 'save'
            must be True. If not given as an argument the pitch is not going to be
            saved. Plots are going to be stored as png in the same directory layer as
            the repository.
        path: str
            Path to which the plot should be saved if 'save' is True. If not given as an
            an argument the default value is the current working directory.
        show_plot: bool, True
            You may want to see the plot. The plot is shown per default. If you don't
            want to see the plot the 'show_plot' variable must be set to False.
        show_axis: bool, optional
            You may want to show the axis. If 'show_axis' is True the axis are going to
            be visible. If not given as an argument the axis are not going to be
            visible.
        ax: plt.axes
            Axes from matplotlib library on which the handball field is plotted.
        kwargs:
            You may pass optional arguments (`linewidth`, `zorder`, 'scalex','scaley'}
            which can be used for the plot functions from matplotlib. The kwargs
            only getting passed to all the plot functions.

        Returns
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            An axes to which all elements of the sport specific pitch are added.
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
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
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
        angle = 10 if self.unit == "percent" else 0

        # Margins of the plot
        x_margin = x_range * 0.025
        y_margin = y_range * 0.05

        # set up the boundaries of ax
        ax.set_xlim(
            [xmin - (x_height_goal + x_margin), xmax + (x_height_goal + x_margin)]
        )
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
        #

        # goal area lines
        # lower left
        # Filling the goal area
        if self.unit != "percent":  # Since wedges can't be scaled
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
        # line
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
        if self.unit != "percent":  # Since wedges can't be scaled
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
        # line
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
        if self.unit != "percent":  # Since wedges can't be scaled
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
        # line
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
        if self.unit != "percent":  # Since wedges can't be scaled
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
        # line
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
        if self.unit != "percent":  # Since wedges can't be scaled the rectangles
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
        if self.unit != "percent":  # Since wedges can't be scaled the rectangles
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
        # vetical free throw lines
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
        if not show_axis:
            ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

        if save:
            plt.savefig(path + "_handball_pitch.png")

        if show_plot:
            plt.show()

        return ax

    def _plot_football_pitch(
        self,
        color_scheme: str,
        save: bool,
        path: str,
        show_plot: bool,
        show_axis: bool,
        ax: plt.axes,
        **kwargs,
    ):
        """Plots a football pitch on a given axes.

        Parameters
        ----------
        color_scheme: str
             Color scheme of the plot. One of {'standard', 'bw'}. If not given
             'standard' is the defaulte color scheme.
        save: bool, optional
            You may want to save the graph. If the graphic should be saved 'save' must
            be True. If not given as an argument the pitch is not going to be saved.
            Plots are going to be stored as png on the same directory layer as the
            repository
        path: str
            Path to which the plot should be saved if 'save' is True. If not given as an
            an argument the default value is the current working directory.
        show_plot: bool, True
            You may want to see the plot. The plot is shown per default. If you don't
            want to see the plot the 'show_plot' variable must be set to False.
        show_axis: bool, optional
            You may want to show the axis. If 'show_axis' is True the axis are going to
             be visible. If not given as an argument the axis are not going to be
             visible.
        ax: plt.axes
            Axes from matplotlib library on which the football field is plotted.
        kwargs:
            You may pass optional arguments (`linewidth`, `zorder`, 'scalex','scaley'}
            which can be used for the plot functions from matplotlib. The kwargs
            only getting passed to all the plot functions.

        Returns
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            An axes to which all elements of the sport specific pitch are added.
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
        # on the pitch that has a fixeds size.
        # The norm factor is specified based on the given unit.
        # If the unit is 'm' or 'cm' the ratio between x and y is set to 1
        # (see in the Pitch.plot() method ax.set_aspect(1)). But if the unit is
        # 'percent' the ratio between width/length is set to ax.set_aspect(width/length)
        # If an element like the goal area, which reaches 5.5 m into the x direction of
        # the field, is drawn now, the fixed size of 5.5 m gets rescaled.

        # norm_factor for all elements on the pitch that are scaled in the x direction
        norm_factor_x = (
            1
            if self.unit == "m"
            else 100
            if self.unit == "cm"
            else 100 / self.length
            if self.length
            else 100 / 105
        )
        # norm_factor for all elements on the pitch that are scaled in the y direction
        norm_factor_y = (
            1
            if self.unit == "m"
            else 100
            if self.unit == "cm"
            else 100 / self.width
            if self.width
            else 100 / 68
        )

        # All the positions and ranges of certain elements on the pitch
        # (i.e the penalty area) have fixed sizes and are scaled based on the
        # x and y norm factors.

        # key positions for the football pitch
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
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
        angle = 10 if self.unit == "percent" else 0

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
        if not show_axis:
            ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

        if save:
            plt.savefig(path + "/football_pitch.png")

        if show_plot:
            plt.show()

        return ax

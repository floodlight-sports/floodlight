import os
from dataclasses import dataclass
from typing import Tuple
import warnings

import matplotlib
import matplotlib.pyplot as plt  # add to poetry environment?
from matplotlib.patches import Arc, Wedge, Rectangle  # add to poetry environment?

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
        ax: plt.axes = None,
        **kwargs,
    ) -> plt.axes:
        """Returns a plot for a given sport (i.e handball, soccer/football)

        Parameters
        ----------
        sport: str, optional
            Sport for which the pitch is used. This is used to automatically generate
             lines and markings. If sport is already declared in the pitch object the
             sport must not be passed as an argument.

        color_scheme: str, optional
            Color scheme of the plot. One of {'normal', 'black_white'}. If not given '
            normal' is the defaulte color scheme.

        save: bool, optional
            You may want to save the graph. If the graphic should be saved 'save' must
            be True. If not given as an argument the pitch is not going to be saved.

        kwargs:
            You may pass optional arguments (`linewidth`, `figsize`, 'scalex', 'scaley',
            'x_size', 'y_size'} which can be used for the plot with matplotlib. For
            example, pass the `linewidth` and `figsize` argument to modify the
            visualization.

        ax: plt.axes, optional
            Axes from matplotlib library on which the playing field is plotted.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            An subplot to which all elements of the sport specific pitch are added.
        """

        color_schemes = ["black_white", "normal", None]
        sports = ["football", "soccer", "handball"]
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

        # initialize essential parameters for plotting
        scalex = kwargs.get("scalex", False)
        scaley = kwargs.get("scaley", False)

        # check wether an axes to plot is given or if a new axes element
        # has to be created
        ax = ax or plt.subplots(**kwargs)[1]
        ax.set_aspect(1)  # ensures that the ratio between length and width is correct
        # regardless of the figsize

        # create axes with handball pitch
        if sport == "handball":
            return self._plot_handball_pitch(
                scalex=scalex,
                scaley=scaley,
                color_scheme=color_scheme,
                save=save,
                ax=ax,
                **kwargs,
            )

        # create axes with soccer/football pitch
        if sport == "soccer" or "football":
            return self._plot_soccer_pitch(
                scalex=scalex,
                scaley=scaley,
                color_scheme=color_scheme,
                save=save,
                ax=ax,
                **kwargs,
            )

    def _plot_handball_pitch(
        self,
        scalex: bool,
        scaley: bool,
        color_scheme: str,
        save: bool,
        ax: plt.axes,
        **kwargs,
    ):

        """Plots a handball pitch on a given axes

        Parameters
        ----------
        scalex: bool
            Determines if the view limits are adapted to the
            data limits of the x values.
        scaley: bool
            Determines if the view limits are adapted to the
            data limits of the y values.
        color_scheme: str
             Color scheme of the plot. One of {'normal', 'black_white'}.
             If not given 'normal' is the defaulte color scheme.
        save: bool
            You may want to save the graph. If the graphic should be saved 'save'
            must be True. If not given as an argument the pitch is not going to be
            saved. Plots are going to be stored as .png on the same directory layer as
            the repository
        kwargs:
            You may pass optional arguments (`linewidth`} which can be used for the plot
            with matplotlib.
        ax: plt.axes
            Axes from matplotlib library on which the handball field is plotted.

        Returns
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            An axes to which all elements of the sport specific pitch are added.
        """

        # customizing visualization
        if color_scheme == "black_white":
            ax.set_facecolor("white")
            outline = "black"
            innerline = "black"
            fill = "white"
            goal = "black"
        else:
            ax.set_facecolor("skyblue")
            outline = "white"
            innerline = "white"
            fill = "khaki"
            goal = "white"

        # key data for the handball pitch
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim

        x_range = xmax - xmin
        y_range = ymax - ymin
        x_half = (xmax + xmin) / 2
        y_half = (ymax + ymin) / 2

        linewidth = kwargs.get("linewidth", 1)

        # points on the field which are used for plotting all lines and arcs
        left_side_right_goal_post = (xmin, ymin + y_range * 0.425)
        left_side_left_goal_post = (xmin, ymin + y_range * 0.575)
        right_side_left_goal_post = (xmax, ymin + y_range * 0.425)
        right_side_right_goal_post = (xmax, ymin + y_range * 0.575)
        width_hight_free_throw_line = y_range * 0.9
        width_hight_goal_area_line = y_range * 0.6
        radius_goal_area_line_arc = y_range * 0.3
        goal_width = y_range * 0.15
        goal_height = y_range * 0.1
        y_center_to_post = goal_width / 2
        y_value_goal_lower_post = y_half - y_center_to_post
        y_value_goal_upper_post = y_half + y_center_to_post
        y_pos_lower_goal_posts = ymin + y_range * 0.425
        y_pos_upper_goal_posts = ymin + y_range * 0.575
        left_side_x_pos_goal_area_line_edge = xmin + (x_range * 0.15)
        right_side_x_pos_goal_area_line_edge = xmax - (x_range * 0.15)
        left_side_x_pos_free_throw_line_edge = xmin + (x_range * 0.225)
        right_side_x_pos_free_throw_line_edge = xmax - (x_range * 0.225)
        left_side_4m_line_x_pos = xmin + (x_range * 0.1)
        lower_y_pos_4m_line = ymin + (y_range * 0.49625)
        upper_y_pos_4m_line = ymin + (y_range * 0.50375)
        left_side_7m_line_x_pos = xmin + (x_range * 0.175)
        right_side_7m_line_x_pos = xmax - (x_range * 0.175)
        lower_y_pos_7m_line = ymin + (y_range * 0.475)
        upper_y_pos_7m_line = ymin + (y_range * 0.525)
        x_margin = x_range / 40
        y_margin = y_range / 20

        # set up the boundaries of ax
        ax.set_xlim(
            [
                xmin - (goal_height + x_margin),
                xmax + (goal_height + x_margin),
            ]
        )
        ax.set_ylim([ymin - y_margin, ymax + y_margin])

        # paint handball pitch with all properties
        # main boundaries
        ax.plot(
            [xmin, xmin],
            [ymin, ymax],
            color=outline,
            scalex=False,
            scaley=False,
            linewidth=linewidth,
            zorder=0,
        )
        ax.plot(
            [xmax, xmax],
            [ymin, ymax],
            color=outline,
            scalex=False,
            scaley=False,
            linewidth=linewidth,
            zorder=0,
        )
        ax.plot(
            [xmin, xmax],
            [ymin, ymin],
            color=outline,
            scalex=False,
            scaley=False,
            linewidth=linewidth,
            zorder=0,
        )
        ax.plot(
            [xmin, xmax],
            [ymax, ymax],
            color=outline,
            scalex=False,
            scaley=False,
            linewidth=linewidth,
            zorder=0,
        )

        # midline
        ax.plot(
            [x_half, x_half],
            [ymin, ymax],
            color=outline,
            scalex=False,
            scaley=False,
            zorder=0,
            linewidth=linewidth,
        )

        # free-throw lines
        # lower left
        ax.add_patch(
            Arc(
                left_side_right_goal_post,
                width=width_hight_free_throw_line,
                height=width_hight_free_throw_line,
                angle=0,
                theta1=290,
                theta2=360,
                linestyle="dashed",
                linewidth=linewidth,
                color=innerline,
                zorder=0,
            )
        )
        # upper left
        ax.add_patch(
            Arc(
                left_side_left_goal_post,
                width=width_hight_free_throw_line,
                height=width_hight_free_throw_line,
                angle=0,
                theta1=0,
                theta2=70,
                linestyle="dashed",
                linewidth=linewidth,
                color=innerline,
                zorder=0,
            )
        )
        # lower right
        ax.add_patch(
            Arc(
                right_side_left_goal_post,
                width=width_hight_free_throw_line,
                height=width_hight_free_throw_line,
                angle=0,
                theta1=180,
                theta2=250,
                linestyle="dashed",
                linewidth=linewidth,
                color=innerline,
                zorder=0,
            )
        )
        # upper right
        ax.add_patch(
            Arc(
                right_side_right_goal_post,
                width=width_hight_free_throw_line,
                height=width_hight_free_throw_line,
                angle=0,
                theta1=110,
                theta2=180,
                linestyle="dashed",
                linewidth=linewidth,
                color=innerline,
                zorder=0,
            )
        )

        # goal area lines
        # lower left
        ax.add_patch(
            Wedge(
                left_side_right_goal_post,
                r=radius_goal_area_line_arc,
                theta1=270,
                theta2=360,
                linewidth=linewidth,
                color=fill,
                zorder=0,
            )
        )
        ax.add_patch(
            Arc(
                left_side_right_goal_post,
                width=width_hight_goal_area_line,
                height=width_hight_goal_area_line,
                angle=0,
                theta1=270,
                theta2=360,
                linewidth=linewidth,
                color=outline,
                zorder=0,
            )
        )
        # upper left
        ax.add_patch(
            Wedge(
                left_side_left_goal_post,
                r=radius_goal_area_line_arc,
                theta1=0,
                theta2=90,
                linewidth=linewidth,
                color=fill,
                zorder=0,
            )
        )
        ax.add_patch(
            Arc(
                left_side_left_goal_post,
                width_hight_goal_area_line,
                width_hight_goal_area_line,
                angle=0,
                theta1=0,
                theta2=90,
                linewidth=linewidth,
                color=outline,
                zorder=0,
            )
        )
        # lower right
        ax.add_patch(
            Wedge(
                right_side_left_goal_post,
                r=radius_goal_area_line_arc,
                theta1=180,
                theta2=270,
                linewidth=linewidth,
                color=fill,
                zorder=0,
            )
        )
        ax.add_patch(
            Arc(
                right_side_left_goal_post,
                width_hight_goal_area_line,
                width_hight_goal_area_line,
                angle=0,
                theta1=180,
                theta2=270,
                linewidth=linewidth,
                color=outline,
                zorder=0,
            )
        )
        # upper right
        ax.add_patch(
            Wedge(
                right_side_right_goal_post,
                r=radius_goal_area_line_arc,
                theta1=90,
                theta2=180,
                linewidth=linewidth,
                color=fill,
                zorder=0,
            )
        )
        ax.add_patch(
            Arc(
                right_side_right_goal_post,
                width_hight_goal_area_line,
                width_hight_goal_area_line,
                angle=0,
                theta1=90,
                theta2=180,
                linewidth=linewidth,
                color=outline,
                zorder=0,
            )
        )
        # mid left
        ax.add_patch(
            Rectangle(
                left_side_right_goal_post,
                width=radius_goal_area_line_arc,
                height=goal_width,
                color=fill,
                zorder=0,
            )
        )
        # mid right
        ax.add_patch(
            Rectangle(
                (xmax - radius_goal_area_line_arc, y_pos_lower_goal_posts),
                width=radius_goal_area_line_arc,
                height=goal_width,
                color=fill,
                zorder=0,
            )
        )

        # vertical goal area lines
        ax.plot(
            [left_side_x_pos_goal_area_line_edge, left_side_x_pos_goal_area_line_edge],
            [y_pos_lower_goal_posts, y_pos_upper_goal_posts],
            color=outline,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )
        ax.plot(
            [
                right_side_x_pos_goal_area_line_edge,
                right_side_x_pos_goal_area_line_edge,
            ],
            [y_pos_lower_goal_posts, y_pos_upper_goal_posts],
            color=outline,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )

        ax.plot(
            [
                left_side_x_pos_free_throw_line_edge,
                left_side_x_pos_free_throw_line_edge,
            ],
            [y_pos_lower_goal_posts, y_pos_upper_goal_posts],
            color=innerline,
            linewidth=linewidth,
            linestyle="dashed",
            scalex=False,
            scaley=False,
            zorder=0,
        )
        ax.plot(
            [
                right_side_x_pos_free_throw_line_edge,
                right_side_x_pos_free_throw_line_edge,
            ],
            [y_pos_lower_goal_posts, y_pos_upper_goal_posts],
            color=innerline,
            linewidth=linewidth,
            linestyle="dashed",
            scalex=False,
            scaley=False,
            zorder=0,
        )

        # 4 m lines
        ax.plot(
            [left_side_4m_line_x_pos, left_side_4m_line_x_pos],
            [lower_y_pos_4m_line, upper_y_pos_4m_line],
            color=outline,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )
        ax.plot(
            [xmax - (x_range * 0.1), xmax - (x_range * 0.1)],
            [lower_y_pos_4m_line, upper_y_pos_4m_line],
            color=outline,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )

        # 7 m lines
        ax.plot(
            [left_side_7m_line_x_pos, left_side_7m_line_x_pos],
            [lower_y_pos_7m_line, upper_y_pos_7m_line],
            color=innerline,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )
        ax.plot(
            [right_side_7m_line_x_pos, right_side_7m_line_x_pos],
            [lower_y_pos_7m_line, upper_y_pos_7m_line],
            color=innerline,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )

        # goals
        # left goal
        ax.plot(
            [xmin - goal_height, xmin - goal_height],
            [y_value_goal_lower_post, y_value_goal_upper_post],
            color=goal,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )
        ax.plot(
            [xmin - goal_height, xmin],
            [y_value_goal_lower_post, y_value_goal_lower_post],
            color=goal,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )
        ax.plot(
            [xmin - goal_height, xmin],
            [y_value_goal_upper_post, y_value_goal_upper_post],
            color=goal,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )
        # right goal
        ax.plot(
            [xmax + goal_height, xmax + goal_height],
            [y_value_goal_lower_post, y_value_goal_upper_post],
            color=goal,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )
        ax.plot(
            [xmax + goal_height, xmax],
            [y_value_goal_lower_post, y_value_goal_lower_post],
            color=goal,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )
        ax.plot(
            [xmax + goal_height, xmax],
            [y_value_goal_upper_post, y_value_goal_upper_post],
            color=goal,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )
        # baseline
        ax.plot(
            [xmax, xmax],
            [ymin, ymax],
            color=innerline,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )
        # baseline
        ax.plot(
            [xmin, xmin],
            [ymin, ymax],
            color=innerline,
            linewidth=linewidth,
            scalex=False,
            scaley=False,
            zorder=0,
        )

        # remove labels and ticks
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if save:
            path = os.path.abspath(
                os.path.join(
                    os.path.abspath(
                        os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), os.pardir
                        )
                    ),
                    os.pardir,
                )
            )
            plt.savefig(str(path) + "_handball_pitch.png")

        return ax

    def _plot_soccer_pitch(
        self,
        scalex: bool,
        scaley: bool,
        color_scheme: str,
        save: bool,
        ax: plt.axes,
        **kwargs,
    ):
        """Plots a soccer/football pitch on a given axes

        Parameters
        ----------
        scalex: bool
            Determines if the view limits are adapted to the data limits of the
            x values.
        scaley: bool
            Determines if the view limits are adapted to the data limits of the
            y values.
        color_scheme: str
             Color scheme of the plot. One of {'normal', 'black_white'}. If not given
             'normal' is the defaulte color scheme.
        save: bool, optional
            You may want to save the graph. If the graphic should be saved 'save' must
            be True. If not given as an argument the pitch is not going to be saved.
            Plots are going to be stored as .png on the same directory layer as the
            repository
        ax: plt.axes
            Axes from matplotlib library on which the soccer/football field is plotted.
        kwargs:
            You may pass optional arguments (`linewidth`} which can be used for the
            plot with matplotlib.

        Returns
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            An axes to which all elements of the sport specific pitch are added.
        """

        # customizing visualization
        if color_scheme == "black_white":
            ax.set_facecolor("white")
            line = "black"
        else:
            ax.set_facecolor("green")
            line = "white"

        # key data for football/soccer pitch. If the unit is "percent" the values
        # are either adjusted based on length
        # and width (if available) or set to default values (length = 105, width = 68)
        xmin = self.xlim[0] if self.unit != "percent" else 0
        xmax = (
            self.xlim[1]
            if self.unit != "percent"
            else self.length
            if self.length
            else 105
        )
        ymin = self.ylim[0] if self.unit != "percent" else 0
        ymax = (
            self.ylim[1] if self.unit != "percent" else self.width if self.width else 68
        )

        if self.unit == "percent" and not self.length and not self.width:
            warnings.warn(
                "Since the unit is 'percent' and no information on the actual pitch "
                "size in terms of 'length' and 'width' is provided the pitch is set to "
                "default values length: 105 and width: 68"
            )

        # factor to determine the actual size of the pitch and adjust all the
        # elements (i.e goal area)
        norm_factor = 1 if self.unit == "m" else 100 if self.unit == "cm" else 1

        # key data for the soccer/football pitch
        x_half = (xmax + xmin) / 2
        y_half = (ymax + ymin) / 2
        linewidth = kwargs.get("linewidth", 1)
        radius_points = 0.25 * norm_factor
        goal_height = 2.44 * norm_factor
        goal_width = 7.32 * norm_factor
        y_center_to_post = goal_width / 2

        # goal area
        left_side_x_value_goal_area_line = xmin + 5.5 * norm_factor
        right_side_x_value_goal_area_line = xmax - 5.5 * norm_factor
        y_value_goal_area_lower_post = y_half - 9.16 * norm_factor
        y_value_goal_area_upper_post = y_half + 9.16 * norm_factor

        # goal
        y_value_goal_lower_post = y_half - y_center_to_post
        y_value_goal_upper_post = y_half + y_center_to_post

        # penalty area
        left_side_x_value_penalty_area_line = xmin + 16.5 * norm_factor
        y_value_penalty_area_lower_line = y_half - 20.16 * norm_factor
        y_value_penalty_area_upper_line = y_half + 20.16 * norm_factor
        right_side_x_value_penalty_area_line = xmax - 16.5 * norm_factor
        left_side_penalty_arc = xmin + 11 * norm_factor
        width_penalty_arc = 20.15 * norm_factor
        right_side_penalty_arc = xmax - 11 * norm_factor
        left_penalty_point = xmin + 11 * norm_factor
        right_penalty_point = xmax - 11 * norm_factor

        # center circle
        radius_center_circle = 9.15 * norm_factor

        # field boundaries
        ax.plot([xmin, xmax], [ymin, ymin], color=line, zorder=0, linewidth=linewidth)
        ax.plot([xmin, xmax], [ymax, ymax], color=line, zorder=0, linewidth=linewidth)
        ax.plot([xmin, xmin], [ymin, ymax], color=line, zorder=0, linewidth=linewidth)
        ax.plot([xmax, xmax], [ymin, ymax], color=line, zorder=0, linewidth=linewidth)
        ax.plot(
            [x_half, x_half], [ymin, ymax], color=line, zorder=0, linewidth=linewidth
        )

        # goal area
        # goal area left
        ax.plot(
            [left_side_x_value_goal_area_line, left_side_x_value_goal_area_line],
            [y_value_goal_area_lower_post, y_value_goal_area_upper_post],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        ax.plot(
            [xmin, left_side_x_value_goal_area_line],
            [y_value_goal_area_lower_post, y_value_goal_area_lower_post],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        ax.plot(
            [xmin, left_side_x_value_goal_area_line],
            [y_value_goal_area_upper_post, y_value_goal_area_upper_post],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        # goal area right
        ax.plot(
            [right_side_x_value_goal_area_line, right_side_x_value_goal_area_line],
            [y_value_goal_area_lower_post, y_value_goal_area_upper_post],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        ax.plot(
            [xmax, right_side_x_value_goal_area_line],
            [y_value_goal_area_lower_post, y_value_goal_area_lower_post],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        ax.plot(
            [xmax, right_side_x_value_goal_area_line],
            [y_value_goal_area_upper_post, y_value_goal_area_upper_post],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )

        # penalty area
        # penalty area left
        ax.plot(
            [left_side_x_value_penalty_area_line, left_side_x_value_penalty_area_line],
            [y_value_penalty_area_lower_line, y_value_penalty_area_upper_line],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        ax.plot(
            [xmin, left_side_x_value_penalty_area_line],
            [y_value_penalty_area_lower_line, y_value_penalty_area_lower_line],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        ax.plot(
            [xmin, left_side_x_value_penalty_area_line],
            [y_value_penalty_area_upper_line, y_value_penalty_area_upper_line],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        # penalty area right
        ax.plot(
            [
                right_side_x_value_penalty_area_line,
                right_side_x_value_penalty_area_line,
            ],
            [y_value_penalty_area_lower_line, y_value_penalty_area_upper_line],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        ax.plot(
            [xmax, right_side_x_value_penalty_area_line],
            [y_value_penalty_area_lower_line, y_value_penalty_area_lower_line],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        ax.plot(
            [xmax, right_side_x_value_penalty_area_line],
            [y_value_penalty_area_upper_line, y_value_penalty_area_upper_line],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        # penalty point
        ax.add_patch(
            plt.Circle(
                (left_penalty_point, y_half),
                radius_points,
                color=line,
                zorder=0,
                linewidth=linewidth,
            )
        )
        ax.add_patch(
            plt.Circle(
                (right_penalty_point, y_half),
                radius_points,
                color=line,
                zorder=0,
                linewidth=linewidth,
            )
        )
        # penalty area circle left
        ax.add_patch(
            Arc(
                (left_side_penalty_arc, y_half),
                width=width_penalty_arc,
                height=width_penalty_arc,
                angle=302.5,
                theta1=0,
                theta2=115,
                color=line,
                zorder=0,
                linewidth=linewidth,
            )
        )
        # penalty area circle right
        ax.add_patch(
            Arc(
                (right_side_penalty_arc, y_half),
                width=width_penalty_arc,
                height=width_penalty_arc,
                angle=122.5,
                theta1=0,
                theta2=115,
                color=line,
                zorder=0,
                linewidth=linewidth,
            )
        )

        # goal
        # left goal
        ax.plot(
            [xmin - goal_height, xmin - goal_height],
            [y_value_goal_lower_post, y_value_goal_upper_post],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        ax.plot(
            [xmin - goal_height, xmin],
            [y_value_goal_upper_post, y_value_goal_upper_post],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        ax.plot(
            [xmin - goal_height, xmin],
            [y_value_goal_lower_post, y_value_goal_lower_post],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        # right goal
        ax.plot(
            [xmax + goal_height, xmax + goal_height],
            [y_value_goal_lower_post, y_value_goal_upper_post],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        ax.plot(
            [xmax + goal_height, xmax],
            [y_value_goal_upper_post, y_value_goal_upper_post],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )
        ax.plot(
            [xmax + goal_height, xmax],
            [y_value_goal_lower_post, y_value_goal_lower_post],
            color=line,
            zorder=0,
            linewidth=linewidth,
        )

        # center circle
        ax.add_patch(
            plt.Circle(
                (x_half, y_half),
                radius_center_circle,
                edgecolor=line,
                fill=False,
                zorder=0,
                linewidth=linewidth,
            )
        )
        ax.add_patch(
            plt.Circle(
                (x_half, y_half),
                radius_points,
                color=line,
                zorder=0,
                linewidth=linewidth,
            )
        )

        # remove labels and ticks
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if save:
            path = os.path.abspath(
                os.path.join(
                    os.path.abspath(
                        os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), os.pardir
                        )
                    ),
                    os.pardir,
                )
            )
            plt.savefig(str(path) + "_soccer_pitch.png")

        return ax

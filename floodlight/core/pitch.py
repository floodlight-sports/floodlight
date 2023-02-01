import warnings
from dataclasses import dataclass
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt

from floodlight.vis.pitches import plot_handball_pitch, plot_football_pitch
from floodlight.utils.types import Numeric


@dataclass
class Pitch:
    """
    Pitch and coordinate system specifications. Core class of floodlight.

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
    unit: {'m', 'cm', 'percent', 'normed'}
        The unit in which data is measured along axes. The values 'percent' and 'normed'
        can be used to denote standardized pitches where data is scaled along the axes
        independent of the actual pitch size. In this case, 'percent' refers to a
        scaling onto the range (0, 100), and 'normed' to all other scalings.
        To get non-distored calculations from these unit-systems, the `length` and
        `width` attributes need to be specified.
    boundaries: str
        One of {'fixed', 'flexible'}. Here, 'fixed' denotes coordinate systems that
        limit axes to the respective xlim and ylim. On the contrary, 'flexible'
        coordinate systems do not explicitly specify a limit. Instead, the limit is
        implicitly set by the actual pitch length and width.
    length: Numeric, optional
        Actual pitch length in *m*.
    width: Numeric, optional
        Actual pitch width in *m*.
    sport: str, optional
        Sport for which the pitch is used. This is used to automatically generate lines
        and markings.

    Attributes
    ----------
    center: tuple
        Returns coordinates of the pitch center.
    is_metrical: bool
        Returns True if the object's unit is metrical, False otherwise.
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
            {'dfl', 'eigd', 'opta', 'statsbomb', 'secondspectrum', 'statsperform_event',
            'statsperform_tracking', 'statsperform_open', 'tracab'}.
        kwargs:
            You may pass optional arguments (`length`, `width`, `sport`) used for class
            instantiation. For some data providers, additional kwargs are needed to
            represent their format correctly. For example, pass the `length` and `width`
            argument to create a Pitch object in the 'tracab' format.

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
        elif template_name == "statsperform_open":
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
        elif template_name == "secondspectrum":
            if "length" not in kwargs or "width" not in kwargs:
                raise TypeError(
                    "For an exact Second Spectrum Pitch object, "
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
        elif template_name == "statsperform_event":
            if "length" not in kwargs or "width" not in kwargs:
                raise TypeError(
                    "For an exact StatsPerform Pitch object, "
                    "`length` and `width` of the pitch need "
                    "to be passed as keyworded arguments"
                )
            x_half = round((kwargs["length"] * 100) / 2, 3)
            y_half = round((kwargs["width"] * 100) / 2, 3)
            return cls(
                xlim=(-x_half, x_half),
                ylim=(-y_half, y_half),
                unit="cm",
                boundaries="flexible",
                length=kwargs.get("length"),
                width=kwargs.get("width"),
                sport=kwargs.get("sport"),
            )
        elif template_name == "statsperform_tracking":
            if "length" not in kwargs or "width" not in kwargs:
                raise TypeError(
                    "For an exact StatsPerform Pitch object, "
                    "`length` and `width` of the pitch need "
                    "to be passed as keyworded arguments"
                )
            return cls(
                xlim=(0, kwargs["length"]),
                ylim=(0, kwargs["width"]),
                unit="m",
                boundaries="flexible",
                length=kwargs.get("length"),
                width=kwargs.get("width"),
            )
        elif template_name == "tracab":
            if "length" not in kwargs or "width" not in kwargs:
                raise TypeError(
                    "For an exact TRACAB (ChyronHego, international format) "
                    "Pitch object, `length` and `width` of the pitch need "
                    "to be passed as keyworded arguments"
                )
            x_half = round((kwargs["length"] * 100) / 2, 3)
            y_half = round((kwargs["width"] * 100) / 2, 3)
            return cls(
                xlim=(-x_half, x_half),
                ylim=(-y_half, y_half),
                unit="cm",
                boundaries="flexible",
                length=kwargs.get("length"),
                width=kwargs.get("width"),
                sport=kwargs.get("sport"),
            )
        elif template_name == "eigd":
            return cls(
                xlim=(0, 40),
                ylim=(0, 20),
                unit="m",
                boundaries="fixed",
                length=40,
                width=20,
                sport="handball",
            )
        elif template_name == "statsbomb":
            return cls(
                xlim=(0.0, 120.0),
                ylim=(0.0, 80.0),
                unit="normed",
                boundaries="flexible",
                sport="football",
            )
        else:
            raise ValueError(f"Unsupported template name '{template_name}'")

    @property
    def is_metrical(self) -> bool:
        is_metrical = False
        if self.unit in ["m", "cm"]:
            is_metrical = True
        return is_metrical

    @property
    def center(self) -> tuple:
        center = (
            round((self.xlim[0] + self.xlim[1]) / 2, 3),
            round((self.ylim[0] + self.ylim[1]) / 2, 3),
        )
        return center

    def plot(
        self,
        color_scheme: str = "standard",
        show_axis_ticks: bool = False,
        ax: matplotlib.axes = None,
        **kwargs,
    ) -> matplotlib.axes:
        """Plots a pitch on a matplotlib.axes for a given sport.

        Parameters
        ----------
        color_scheme: str, optional
            Color scheme of the plot. One of {'standard', 'bw'}. Defaults to 'standard'.
        show_axis_ticks: bool, optional
            If set to True, the axis ticks are visible. Defaults to False.
        ax: matplotlib.axes, optional
            Axes from matplotlib library on which the playing field is plotted. If ax is
            None, a default-sized matplotlib.axes object is created.
        kwargs:
            Optional keyworded arguments {'linewidth', 'zorder', 'scalex', 'scaley'}
            which can be used for the plot functions from matplotlib. The kwargs are
            only passed to all the plot functions of matplotlib.

        Returns
        -------
        axes: matplotlib.axes
            Axes from matplotlib library on which the specified pitch is plotted.

        Notes
        -----
        The kwargs are only passed to the plot functions of matplotlib. To customize the
        plots have a look at `matplotlib
        <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.
        For example in order to modify the linewidth pass a float to the keyworded
        argument 'linewidth'. The same principle applies to other kwargs like 'zorder',
        'scalex' and 'scaley'.

        Examples
        --------
        - :ref:`Handball pitch <handball-pitch-label>`
        - :ref:`Football pitch <football-pitch-label>`

        """
        # list of existing color_schemes and sports
        color_schemes = ["bw", "standard"]
        sports = ["football", "handball"]
        sport = self.sport

        # check if valide sport was chosen
        if sport not in sports or sport is None:
            raise ValueError(
                f"Expected self.sport to be from {sports}, got {self.sport}"
            )

        # check if a valide color scheme was chosen
        if color_scheme not in color_schemes:
            raise ValueError(
                f"Expected color_scheme to be from {color_schemes}, got {color_scheme}"
            )

        # check wether an axes to plot is given or if a new axes element has to be
        # created
        ax = ax or plt.subplots()[1]

        # set ratio between x and y values of the plot to ensure that the ratio between
        # length and width is correct regardless of the figsize.
        default_length = 105
        default_width = 68

        if self.unit != "percent":
            ax.set_aspect(1)
        # set ratio if unit is percent and sport is football
        elif self.unit == "percent" and sport == "football":
            if self.length and self.width:
                ax.set_aspect(self.width / self.length)
            # set ratio to standard pitch size of 68/105
            else:  # standard ratio of length and width
                ax.set_aspect(default_width / default_length)
                warnings.warn(
                    "Since self.unit == 'percent' but self.length and self.width are "
                    f"None the pitch is set to default values length: {default_length} "
                    f"and width: {default_width}"
                )
        # set ratio if unit is percent and sport is handball
        elif self.unit == "percent" and sport == "handball":
            ax.set_aspect(0.5)

        # create matplotlib.axes with handball pitch
        if sport == "handball":
            return plot_handball_pitch(
                self.xlim,
                self.ylim,
                self.unit,
                color_scheme=color_scheme,
                show_axis_ticks=show_axis_ticks,
                ax=ax,
                **kwargs,
            )

        # create matplotlib.axes with football pitch
        if sport == "football":
            return plot_football_pitch(
                self.xlim,
                self.ylim,
                self.length,
                self.width,
                self.unit,
                color_scheme=color_scheme,
                show_axis_ticks=show_axis_ticks,
                ax=ax,
                **kwargs,
            )

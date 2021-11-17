from dataclasses import dataclass
from typing import Tuple

from floodlight.utils.typing import Numeric


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
            f"Floodlight Pitch object of size x = {self.xlim} / y = {self.ylim} "
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
            {'opta', 'chyronhego_international'}.
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
        if template_name == "opta":
            return cls(
                xlim=(0.0, 100.0),
                ylim=(0.0, 100.0),
                unit="percent",
                boundaries="fixed",
                length=kwargs.get("length"),
                width=kwargs.get("width"),
                sport=kwargs.get("sport"),
            )
        elif template_name == "chyronhego_international":
            if "length" not in kwargs or "width" not in kwargs:
                raise TypeError(
                    "For an exact ChyronHego (international format) "
                    "Pitch object, `length` and `width` of the pitch need "
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
        elif template_name == "statsperform":
            if "length" not in kwargs or "width" not in kwargs:
                raise TypeError(
                    "For an exact StatsPerform Pitch object, "
                    "length` and `width` of the pitch need "
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
        else:
            raise ValueError(f"Unsupported template name '{template_name}'")

    @property
    def center(self):
        center = (
            round((self.xlim[0] + self.xlim[1]) / 2, 3),
            round((self.ylim[0] + self.ylim[1]) / 2, 3),
        )
        return center

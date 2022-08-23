from .core.code import Code
from .core.events import Events
from .core.pitch import Pitch
from .core.xy import XY
from .core.property import DyadicProperty, PlayerProperty, TeamProperty

__all__ = [
    "__version__",
    "__doc__",
    "Code",
    "DyadicProperty",
    "Events",
    "Pitch",
    "PlayerProperty",
    "TeamProperty",
    "XY",
]

__version__ = "0.3.3"

__doc__ = """
A high-level, data-driven sports analytics framework
====================================================

**floodlight** is a Python package for streamlined analysis of sports data. It is
designed with a clear focus on scientific computing and built upon popular libraries
such as *numpy* or *pandas*.

Load, integrate, and process tracking and event data, codes and other match-related
information from major data providers. This package provides a set of  standardized
data objects to structure and handle sports data, together with a suite of common
processing operations such as transforms or data manipulation methods.

All implementations run completely provider- and sports-independent, while maintaining
a maximum of flexibility to incorporate as many data flavours as possible. A high-level
interface allows easy access to all standard routines, so that you can stop worrying
about data wrangling and start focussing on the analysis instead!
"""

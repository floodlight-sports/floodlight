from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union
import warnings

import numpy as np
import pandas as pd

from floodlight.utils.types import Numeric
from floodlight.core.definitions import essential_events_columns, protected_columns
from floodlight.core.code import Code


@dataclass
class Events:
    """Event data fragment. Core class of floodlight.

    Event data is stored in a `pandas` ``DataFrame``, where each row stores one
    event with its different properties organized in columns. Columns may contain
    any relevant information. An `"eID"` (event ID) and `"gameclock"` column is
    required for instantiation, to identify and time-locate events. Some particular
    column names are protected (see Notes).

    Parameters
    ----------
    events: pd.DataFrame
        DataFrame containing rows of events and columns of respective event properties.
    direction: str, optional
        Playing direction of players in data fragment, should be either
        'lr' (left-to-right) or 'rl' (right-to-left).

    Attributes
    ----------
    essential: list
        List of essential columns available for stored events.
    protected: list
        List of protected columns available for stored events.
    custom: list
        List of custom (i.e. non-essential and non-protected) columns available for
        stored events.
    essential_missing: list
        List of missing essential columns.
    essential_invalid: list
        List of essential columns that violate the definitions.
    protected_missing: list
        List of missing protected columns.
    protected_invalid: list
        List of protected columns that violate the definitions.

    Notes
    -----
    Event data, particularly information available for each event, may vary across
    data providers. To accommodate all data flavours, any column name or data type is
    permissible. However, two `essential` column are required (`"eID"` and
    `"gameclock`). Other column names are `protected`. Using these names assumes that
    data stored in these columns follows conventions in terms of data types and value
    ranges. These are required for methods working with protected columns to assure
    correct calculations. Definitions for `essential` and `protected` columns can be
    found in
    :ref:`floodlight.core.definitions <definitions target>`.

    """

    events: pd.DataFrame
    direction: str = None

    def __post_init__(self):
        # check for missing essential columns
        missing_columns = self.essential_missing
        if missing_columns:
            raise ValueError(
                f"Floodlight Events object is missing the essential "
                f"column(s) {missing_columns}!"
            )

        # warn if value ranges are violated
        incorrect_columns = self.essential_invalid
        if incorrect_columns:
            for col in incorrect_columns:
                warnings.warn(
                    f"The '{col}' column does not match the defined value range (from "
                    f"floodlight.core.definitions). This may lead to unexpected "
                    f"behavior of methods using this column."
                )

    def __str__(self):
        return f"Floodlight Events object of shape {self.events.shape}"

    def __len__(self):
        return len(self.events)

    def __getitem__(self, key):
        return self.events[key]

    def __setitem__(self, key, value):
        self.events[key] = value

    @property
    def essential(self):
        essential = [
            col for col in self.events.columns if col in essential_events_columns
        ]

        return essential

    @property
    def protected(self):
        protected = [col for col in self.events.columns if col in protected_columns]
        return protected

    @property
    def custom(self):
        custom = [
            col
            for col in self.events.columns
            if col not in essential_events_columns and col not in protected_columns
        ]

        return custom

    @property
    def essential_missing(self):
        missing_columns = [
            col for col in essential_events_columns if col not in self.essential
        ]

        return missing_columns

    @property
    def essential_invalid(self):
        invalid_columns = [
            col
            for col in self.essential
            if not self.column_values_in_range(col, essential_events_columns)
        ]

        return invalid_columns

    @property
    def protected_missing(self):
        missing_columns = [
            col for col in protected_columns if col not in self.protected
        ]

        return missing_columns

    @property
    def protected_invalid(self):
        invalid_columns = [
            col
            for col in self.protected
            if not self.column_values_in_range(col, protected_columns)
        ]

        return invalid_columns

    def column_values_in_range(self, col: str, definitions: Dict[str, Dict]) -> bool:
        """Check if values for a single column of the inner event DataFrame are in
        correct range using the specifications from
        :ref:`floodlight.core.definitions <definitions target>`.

        Parameters
        ----------
        col: str
            Column name of the inner events DataFrame to be checked
        definitions: Dict
            Dictionary (from floodlight.core.definitions) containing specifications for
            the columns to be checked.

            The definitions need to contain an entry for the column to be checked and
            this entry needs to contain information about the value range in the form:
            ``definitions[col][value_range] = (min, max)``.

        Returns
        -------
        bool
            True if the checks for value range pass and False otherwise

        Notes
        -----
        Non-integer results of this computation will always be rounded to the next
        smaller integer.
        """
        # skip if value range is not defined
        if definitions[col]["value_range"] is None:
            return True

        # skip values that are None or NaN
        col_nan_free = self.events[col].dropna()

        # retrieve value range from definitions
        min_val, max_val = definitions[col]["value_range"]

        # check value range for remaining values
        if not (min_val <= col_nan_free).all() & (col_nan_free <= max_val).all():
            return False

        # all checks passed
        return True

    def add_frameclock(self, framerate: int):
        """Add the column "frameclock", computed as the rounded multiplication of
        gameclock and framerate, to the inner events DataFrame.

        Parameters
        ----------
        framerate: int
            Temporal resolution of data in frames per second/Hertz.
        """
        frameclock = np.full((len(self.events)), -1, dtype=int)
        frameclock[:] = np.floor(self.events["gameclock"].values * framerate)
        self.events["frameclock"] = frameclock

    def select(
        self, conditions: Tuple[str, Any] or List[Tuple[str, Any]]
    ) -> pd.DataFrame:
        """Returns a DataFrame containing all entries from the inner events DataFrame
         that satisfy all given conditions.

        Parameters
        ----------
        conditions: Tuple or List of Tuples
            A single or a list of conditions used for filtering. Each condition should
            follow the form ``(column, value)``. If ``value`` is given as a variable
            (can also be None), it is used to filter for an exact value. If given as a
            tuple ``value = (min, max)`` that specifies a minimum and maximum value, it
            is filtered for a value range.

            For example, to filter all events that have the ``eID`` of ``"Pass"`` and
            that happened within the first 1000 seconds of the segment, conditions
            should look like:
            ``conditions = [("eID", "Pass"), ("gameclock", (0, 1000))]``

        Returns
        -------
        filtered_events: pd.DataFrame
            A view of the inner events DataFrame with rows fulfilling all criteria
            specified in conditions. The DataFrame can be empty if no row fulfills all
            specified criteria.
        """
        filtered_events = self.events

        # convert single non-list condition to list
        if not isinstance(conditions, list):
            conditions = [conditions]

        # loop through and filter by conditions
        for column, value in conditions:

            # if the value is None filter for all entries with None, NaN or NA
            if value is None:
                filtered_events = filtered_events[filtered_events[column].isna()]

            # check if a single value or a value range is given
            else:

                # value range: filter by minimum and maximum value
                if isinstance(value, (list, tuple)):
                    min_val, max_val = value
                    filtered_events = filtered_events[
                        filtered_events[column] >= min_val
                    ]
                    filtered_events = filtered_events[
                        filtered_events[column] <= max_val
                    ]

                # single value: filter by that value
                else:
                    filtered_events = filtered_events[filtered_events[column] == value]

        return filtered_events

    def translate(self, shift: Tuple[Numeric, Numeric]):
        """Translates data by shift vector.

        Parameters
        ----------
        shift : list or array-like
            Shift vector of form v = (x, y). Any iterable data type with two numeric
            entries is accepted.
        """
        if "at_x" in self.protected and self.events["at_x"].dtype in [
            "int64",
            "float64",
        ]:
            self.events["at_x"] = self.events["at_x"].map(lambda x: x + shift[0])

        if "at_y" in self.protected and self.events["at_y"].dtype in [
            "int64",
            "float64",
        ]:
            self.events["at_y"] = self.events["at_y"].map(lambda x: x + shift[1])

        if "to_x" in self.protected and self.events["to_x"].dtype in [
            "int64",
            "float64",
        ]:
            self.events["to_x"] = self.events["to_x"].map(lambda x: x + shift[0])

        if "to_y" in self.protected and self.events["to_y"].dtype in [
            "int64",
            "float64",
        ]:
            self.events["to_y"] = self.events["to_y"].map(lambda x: x + shift[1])

    def scale(self, factor: float, axis: str = None):
        """Scales data by a given factor and optionally selected axis.

        Parameters
        ----------
        factor : float
            Scaling factor.
        axis : {None, 'x', 'y'}, optional
            Name of scaling axis. If set to 'x' data is scaled on x-axis, if set to 'y'
            data is scaled on y-axis. If None, data is scaled in both directions
            (default).
        """

        if axis not in ["x", "y", None]:
            raise ValueError(f"Expected axis to be one of ('x', 'y', None), got {axis}")

        if axis is None or axis == "x":
            if "at_x" in self.protected and self.events["at_x"].dtype in [
                "int64",
                "float64",
            ]:
                self.events["at_x"] = self.events["at_x"].map(lambda x: x * factor)
            if "to_x" in self.protected and self.events["to_x"].dtype in [
                "int64",
                "float64",
            ]:
                self.events["at_x"] = self.events["at_x"].map(lambda x: x * factor)

        if axis is None or axis == "y":
            if "at_y" in self.protected and self.events["at_y"].dtype in [
                "int64",
                "float64",
            ]:
                self.events["at_y"] = self.events["at_y"].map(lambda x: x * factor)
            if "to_y" in self.protected and self.events["to_y"].dtype in [
                "int64",
                "float64",
            ]:
                self.events["to_y"] = self.events["to_y"].map(lambda x: x * factor)

    def reflect(self, axis: str):
        """Reflects data on given `axis`.

        Parameters
        ----------
        axis : {'x', 'y'}
            Name of reflection axis. If set to "x", data is reflected on x-axis,
            if set to "y", data is reflected on y-axis.
        """
        if axis == "x":
            self.scale(factor=-1, axis="y")
        elif axis == "y":
            self.scale(factor=-1, axis="x")
        else:
            raise ValueError(f"Expected axis to be one of ('x', 'y'), got {axis}")

    def rotate(self, alpha: float):
        """Rotates data on given angle 'alpha' around the origin.

        Parameters
        ----------
        alpha: float
            Rotation angle in degrees. Alpha must be between -360 and 360.
            If positive alpha, data is rotated in counter clockwise direction.
            If negative, data is rotated in clockwise direction around the origin.
        """
        if not (-360 <= alpha <= 360):
            raise ValueError(
                f"Expected alpha to be from -360 to 360, got {alpha} instead"
            )

        phi = np.radians(alpha)
        cos = np.cos(phi)
        sin = np.sin(phi)

        # construct rotation matrix
        r = np.array([[cos, -sin], [sin, cos]]).transpose()

        if "at_x" in self.protected and self.events["at_x"].dtype in [
            "int64",
            "float64",
        ]:
            if "at_y" in self.protected and self.events["at_y"].dtype in [
                "int64",
                "float64",
            ]:
                self.events[["at_x", "at_y"]] = pd.DataFrame(
                    np.round(np.dot(self.events[["at_x", "at_y"]], r), 3)
                )

        if "to_x" in self.protected and self.events["to_x"].dtype in [
            "int64",
            "float64",
        ]:
            if "to_y" in self.protected and self.events["to_y"].dtype in [
                "int64",
                "float64",
            ]:
                self.events[["to_x", "to_y"]] = pd.DataFrame(
                    np.round(np.dot(self.events[["to_x", "to_y"]], r), 3)
                )

    def slice(
        self,
        start: float = None,
        end: float = None,
        slice_by="gameclock",
        inplace: bool = False,
    ):
        """Return copy of object with events sliced in a time interval.

        Intended columns for using this function are ``gameclock`` (total seconds) or
        ``frameclock``. However, also allows slicing by any other column that manifests
        a  temporal relation between events (e.g. ``minute``). Excludes all entries
        without a valid entry in the specified column (e.g. None).

        Parameters
        ----------
        start : float, optional
            Start frame or second of slice. Defaults to beginning of segment.
        end : float, optional
            End frame or second of slice (endframe is excluded). Defaults to last event
            of segment (including).
        slice_by: {'gameclock', 'frameclock'}, optional
            Column used to slice the events. Defaults to ``gameclock``.
        inplace: bool, optional
            If set to ``False`` (default), a new object is returned, otherwise the
            operation is performed in place on the called object.

        Returns
        -------
        events_sliced: Union[Event, None]
        """
        if slice_by not in self.events:
            ValueError(f"Events object does not contain column {slice_by}!")
        if start is None:
            start = 0
        if end is None:
            end = np.nanmax(self.events[slice_by].values) + 1

        sliced_data = self.events[self.events[slice_by] >= start].copy()
        sliced_data = sliced_data[sliced_data[slice_by] < end]
        sliced_data.reset_index(drop=True, inplace=True)

        events_sliced = None
        if inplace:
            self.events = sliced_data
        else:
            events_sliced = Events(
                events=sliced_data,
                direction=deepcopy(self.direction),
            )

        return events_sliced

    def get_event_stream(
        self,
        fade: Union[int, None] = 0,
        **kwargs,
    ) -> Code:
        """Generates a Code object containing the eIDs of all events at the
        respective frame and optionally subsequent frames as defined by the fade
        argument.

        This function translates the object's DataFrame of temporally irregular events
        to a continuous frame-wise representation. This can be especially helpful to
        connect event data with spatiotemporal data, e.g., for filtering the latter
        based on the former. Events overwrite preceding event's fade, and unfilled
        values are set to np.nan.

        Notes
        ------
        Requires the DataFrame to contain the protected ``frameclock`` column.

        Parameters
        ----------
        fade: int, optional
            Number of additional frames for which the Code object should stay at a
            value after the event occurred. The value is overwritten if another event
            occurs within the fade duration. If chosen to zero, the value is maintained
            only for a single frame. If chosen to None, the value is maintained until
            either the next event or until the end of the sequence. Defaults to 0.
        kwargs:
            Keyword arguments of the Code object ("name", "definitions", "framerate")
            that are passed down to instantiate the returned event_stream.

        Returns
        -------
        event_stream: Code
            Generated continuous event stream describing the designated game state.
        """
        if "frameclock" in self.protected_missing:
            raise ValueError(
                "Cannot create event stream from Events object missing "
                "the protected column 'frameclock'. Consider calling "
                "add_frameclock to the Events object first!"
            )
        if fade is not None and fade < 0:
            raise ValueError(
                f"Expected fade to be a positive integer or None, got {fade} instead."
            )

        tmp_events = self.events.sort_values("frameclock")
        start = int(np.round(np.nanmin(tmp_events["frameclock"].values)))
        end = int(np.round(np.nanmax(tmp_events["frameclock"].values))) + 1

        code = np.full((end - start,), np.nan, dtype=object)
        for i in tmp_events.index:
            if pd.isna(tmp_events.at[i, "frameclock"]):
                continue
            frame = int(np.round(tmp_events.at[i, "frameclock"]))
            if fade is None:
                code[frame - start :] = tmp_events.at[i, "eID"]
            else:
                code[frame - start : frame - start + fade + 1] = tmp_events.at[i, "eID"]

        event_stream = Code(
            code=code,
            name=kwargs.get("name"),
            definitions=kwargs.get("definitions"),
            framerate=kwargs.get("framerate"),
        )

        return event_stream

from dataclasses import dataclass
from typing import ClassVar, Dict, List

import pandas as pd


@dataclass
class Events:
    """"""

    events: pd.DataFrame
    direction: str = None

    # ToDo: outsource to file and complete
    _special_columns: ClassVar[Dict[str, List[str]]] = {
        "essential_columns": ["eID", "gameclock"],
        "protected_columns": [
            "pID",
            "tID",
            "mID",
            "outcome",
            "timestamp",
            "minute",
            "second",
        ],
    }

    def __str__(self):
        pass

    def __len__(self):
        pass

    @property
    def essential_columns(self):
        essential_columns = list(
            set(self.events.columns) & set(self._special_columns["essential_columns"])
        )

        return essential_columns

    @property
    def protected_columns(self):
        protected_columns = list(
            set(self.events.columns) & set(self._special_columns["protected_columns"])
        )

        return protected_columns

    @property
    def custom_columns(self):
        custom_columns = list(
            set(self.events.columns)
            - (
                set(self._special_columns["essential_columns"])
                | set(self._special_columns["protected_columns"])
            )
        )

        return custom_columns

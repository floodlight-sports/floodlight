from dataclasses import dataclass

import pandas as pd


@dataclass
class Events:
    """"""

    events: pd.DataFrame
    direction: str = None

    def __str__(self):
        pass

    def __len__(self):
        pass

import os
import numpy as np
import pandas as pd

from floodlight.core.xy import XY
from floodlight.core.events import Events
from floodlight.core.code import Code
from floodlight.core.pitch import Pitch
from typing import Tuple


def load_data() -> Tuple[
    XY,
    XY,
    XY,
    XY,
    XY,
    XY,
    Events,
    Events,
    Events,
    Events,
    Code,
    Code,
    Code,
    Code,
    Pitch,
]:
    """
    Loads the sample data and converts it into appropriate floodlight objects.

    Returns
    -------
    data_objects: Tuple[XY, XY, XY, XY, XY, XY, Events, Events, Events, Events,
    Code, Code, Code, Code, Pitch]
        Sample XY-, Code-, Event-, and Pitch-objects for both teams and both halves. The
        order is (xy_home_ht1, xy_home_ht2, xy_away_ht1, xy_away_ht2, xy_ball_ht1,
        xy_ball_ht2, events_home_ht1, events_home_ht2, events_away_ht1, events_away_ht2,
        possession_ht1, possession_ht2, ballstatus_ht1, ballstatus_ht2, pitch)

    """

    # directory of the current script where the sample data is stored
    __sample_data_location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )

    #: XY-coordinates for the home team playing from left to right in the first half.
    xy_home_ht1 = XY(
        xy=np.load(os.path.join(__sample_data_location__, "xy_home_ht1.npy")),
        framerate=5,
        direction="lr",
    )

    #: XY-coordinates for the home team playing from right to left in the second half.
    xy_home_ht2 = XY(
        xy=np.load(os.path.join(__sample_data_location__, "xy_home_ht2.npy")),
        framerate=5,
        direction="rl",
    )

    #: XY-coordinates for the away team playing from right to left in the first half.
    xy_away_ht1 = XY(
        xy=np.load(os.path.join(__sample_data_location__, "xy_away_ht1.npy")),
        framerate=5,
        direction="rl",
    )

    #: XY-coordinates for the away team playing from left to right in the second half.
    xy_away_ht2 = XY(
        xy=np.load(os.path.join(__sample_data_location__, "xy_away_ht2.npy")),
        framerate=5,
        direction="rl",
    )

    #: XY-coordinates for the ball in the first half
    xy_ball_ht1 = XY(
        xy=np.load(os.path.join(__sample_data_location__, "xy_ball_ht1.npy")),
        framerate=5,
        direction="rl",
    )

    #: XY-coordinates for the ball in the second half
    xy_ball_ht2 = XY(
        xy=np.load(os.path.join(__sample_data_location__, "xy_ball_ht2.npy")),
        framerate=5,
        direction="rl",
    )

    #: Events assigned to home team in first half.
    events_home_ht1 = Events(
        events=pd.read_csv(
            os.path.join(__sample_data_location__, "events_home_ht1.csv")
        )
    )

    #: Events assigned to home team in second half.
    events_home_ht2 = Events(
        events=pd.read_csv(
            os.path.join(__sample_data_location__, "events_home_ht2.csv")
        )
    )

    #: Events assigned to away team in first half.
    events_away_ht1 = Events(
        events=pd.read_csv(
            os.path.join(__sample_data_location__, "events_away_ht1.csv")
        )
    )

    #: Events assigned to away team in second half.
    events_away_ht2 = Events(
        events=pd.read_csv(
            os.path.join(__sample_data_location__, "events_away_ht2.csv")
        )
    )

    #: Code indicating dead or alive ball in the first half.
    ballstatus_ht1 = Code(
        code=np.load(os.path.join(__sample_data_location__, "ballstatus_ht1.npy")),
        name="ballstatus",
        definitions={0: "Dead", 1: "Alive"},
        framerate=5,
    )

    #: Code indicating dead or alive ball in the second half.
    ballstatus_ht2 = Code(
        code=np.load(os.path.join(__sample_data_location__, "ballstatus_ht2.npy")),
        name="ballstatus",
        definitions={0: "Dead", 1: "Alive"},
        framerate=5,
    )

    #: Code indicating home or away possession in the first half.
    possession_ht1 = Code(
        code=np.load(os.path.join(__sample_data_location__, "possession_ht1.npy")),
        name="possession",
        definitions={1: "Home", 2: "Away"},
        framerate=5,
    )

    #: Code indicating home or away possession in the second half.
    possession_ht2 = Code(
        code=np.load(os.path.join(__sample_data_location__, "possession_ht1.npy")),
        name="possession",
        definitions={1: "Home", 2: "Away"},
        framerate=5,
    )

    #: Sample football pitch of size 105m x 68m with coordinate origin in the center.
    pitch = Pitch(
        xlim=(-52.5, 52.5),
        ylim=(-34, 34),
        unit="m",
        boundaries="flexible",
        length=105,
        width=68,
        sport="football",
    )

    # assemble
    data_objects = (
        xy_home_ht1,
        xy_home_ht2,
        xy_away_ht1,
        xy_away_ht2,
        xy_ball_ht1,
        xy_ball_ht2,
        events_home_ht1,
        events_home_ht2,
        events_away_ht1,
        events_away_ht2,
        possession_ht1,
        possession_ht2,
        ballstatus_ht1,
        ballstatus_ht2,
        pitch,
    )

    return data_objects

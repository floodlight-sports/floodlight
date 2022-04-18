=========================
floodlight.io.sample_data
=========================

Collection of different sample data objects for a sample football match.
The match consists of two teams (home and away) and is played for two exemplary halves.
All data is captured at an arbitrary framerate of 5 fps. It is be obtained via:

.. code-block:: python

    from floodlight.io.sample_data.loader import load_data

    xy_home_ht1, xy_home_ht2, xy_away_ht1, xy_away_ht2, xy_ball_ht1, xy_ball_ht2,
    events_home_ht1, events_home_ht2, events_away_ht1, events_away_ht2,
    possession_ht1, possession_ht2, ballstatus_ht1, ballstatus_ht2, pitch = load_data()

.. automodule:: floodlight.io.sample_data.loader
    :members:

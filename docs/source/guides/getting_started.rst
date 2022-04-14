===============
Getting Started
===============

Here's everything you need to know to quickly get set up and start using **floodlight**!


Installation
============

The package can be installed via pip

.. code-block:: bash

    pip install floodlight

and then imported into your local Python environment

.. code-block:: python

    import floodlight


Loading Data
============

Obviously you need to get some data to get going.

Provider Data
-------------

If you have data files saved in a specific provider format, you may use on of our parsers. Parsing works differently depending on your file types and preferences. But in essence, we have one submodule per data provider, and one function per file type. Additionally, there is typically one function that reads "the full file(s)" for one observation - e.g. when you have a data containing position data and an attached metadata file.

Let's look at a quick example loading Tracab position and Opta event data:

.. code-block:: python

    from floodlight.io.tracab import read_tracab_files
    from floodlight.io.opta import read_f24

    filepath_dat = <filepath_to_tracab_dat_file>
    filepath_meta = <filepath_to_tracab_metadata_file>
    filepath_f24 = <filepath_to_opta_f24_feed>

    (
        xy_home_ht1,
        xy_home_ht2,
        xy_away_ht1,
        xy_away_ht2,
        xy_ball_ht1,
        xy_ball_ht2,
        possession_ht1,
        possession_ht2,
        ballstatus_ht1,
        ballstatus_ht2,
        pitch_xy,
    ) = read_tracab_files(filepath_dat, filepath_meta)

    (
        events_home_ht1,
        events_home_ht2,
        events_away_ht1,
        events_away_ht2,
        pitch_events
    ) = read_f24(filepath_f24)


Sample Data
------------

As provider data is usually proprietary, you might find yourself without any data files at hand. In this case, you can either insert data by hand, or use our sample data. We provide a small set of synthetic match data for instruction purposes and to play around with. Load them by running:

.. code-block:: python

    from floodlight.io.sample_data.home_xy import xy_home_ht1, xy_home_ht2
    from floodlight.io.sample_data.away_xy import xy_away_ht1, xy_away_ht2
    from floodlight.io.sample_data.ball_xy import xy_ball_ht1, xy_ball_ht2
    from floodlight.io.sample_data.home_events import events_home_ht1, events_home_ht2
    from floodlight.io.sample_data.away_events import events_away_ht1, events_away_ht2
    from floodlight.io.sample_data.codes import possession_ht1, possession_ht2
    from floodlight.io.sample_data.codes import ballstatus_ht1, ballstatus_ht2
    from floodlight.io.sample_data.pitch import pitch

Note that the sample data is already projected to the same pitch, so there are no separate objects for tracking data and events.

Object Manipulation
===================

Now that we have some objects loaded, let's manipulate them. Below are just a few examples, for all methods check out the :doc:`core <../modules/core/core>` module reference.

.. code-block:: python

    # rotate position data 180 degrees (counter-clockwise)
    xy_home_ht1.rotate(180)
    # show only x coordinates
    print(xy_home_ht1.x)
    # show points of 3rd player (xID=3)
    xy_home_ht1.player(3)
    # slice position data to first 100 frames
    xy_home_ht1.slice(startframe=0, endframe=100, inplace=True)

    # print coordinates of pitch middle
    print(pitch.center)

    # add "frameclock" column to events object
    events_away_ht1.add_frameclock(5)
    # show all "Pass" events within first 800 frames
    events_away_ht1.select(conditions=[("eID", "Pass"), ("frameclock", (0, 800))])

    # check what's stored in code object
    print(possession_ht1.definitions)
    # slice ball possession code to first 10 frames
    possession_ht1.slice(startframe=0, endframe=10, inplace=True)

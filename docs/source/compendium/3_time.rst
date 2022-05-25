============================
Time - Framerates and Clocks
============================

Sports data naturally capture phenomena unfolding through time. That's not too big of a problem per se, but a few challenges arise managing the time dimension of the data. Plus, a lack of naming conventions leads to sometimes confusing references to parts of sports play (e.g. some call it half times, some periods). So we've decided to keep naming of things consistent. Let's go through this top-down:


Observation-level
=================

Given all the data for a single observation (such as a match or practice session), we need to deal with the different parts of this event (such as halftimes or different exercises within the session). We call these different parts *segments*.

A segment can be any part of an observation, but it should be a distinct part of an observation in terms of its temporal dimension. Examples are half times (e.g. in football), thirds (e.g. in hockey), quarters (e.g. in basketball), drills (e.g. in a practice session), overtime (e.g. after draws in regular time), or runs (e.g. in an experiment). That really depends on the observation.

The key advantage of this separation is an sports-independent handling of the different parts of an observation. Additionally, we leverage these segments to separate the portions of data. Each data-level object is a fragment of data for exactly one segment, and treated independent of the other segments. Only at the observation level are segments tied together. Here, they are identified by the ``segments`` attribute that lists all segments and which can be used for indexing.

In terms of indexing, there are a few naming conventions we use throughout the code. As an example, for matches that are organized in half times, we use ``segments = ["HT1", "HT2"]``. This list is extended with ``["HT3", "HT4"]`` if we go into overtime.

Data-level
==========

Let's move on to the data-level handling of time. There's a major distinction to be made here between *frame-based* objects such as :doc:`XY <../modules/core/xy>`  and :doc:`Code <../modules/core/code>` and *list-based* objects such as :doc:`Events <../modules/core/events>`.

Frame-based objects represent consistent signals that run through an entire segment, such as player positions. Each frame refers to all data at one particular point in time. All these objects are based on *numpy*\'s ``ndarray``\s, where **the first dimenion always encodes the time dimension**\.

List-based objects are collections of events that can happen at any time (and any place). They are based on *pandas*\' ``DataFrame``\s.

Both categories are treated differently in the time dimension (see below), but there is one thing that unites them: **All time references are always relative to the respective segment** (with timestamps being the exception to the rule).


List-based
----------

As events occur irregularly throughout a segment, each event needs to carry information as to when it took place. There are multiple ways to do it, for example by timestamps or the time on the scoreboard (e.g. '35:12'). Accordingly, you will find ``{"timestamp", "minute", "second"}`` in the list of protected columns. However, these are merely added for convenience, the single time-identifier we rely on is the ``gameclock``\, which measures the elapsed time since the segment started in seconds.

The reason for this choice is that time on the scoreboard is problematic due to stoppage time. In some sports such as football, the referee deliberatly adds a couple of minutes to the playing time. Quite commonly, this is denoted as something like '45:00+01:32', but *it does not have to be*, sometimes it's just '46:32'. That's not an ideal format for implemenation, and we refuse to adapt the entire code to deal with this. Irrespective of first half extra time, the second half will start again at '45:00', leading to potentially ambiguous timestamps such as '46:32'.

Timestamps are problematic due to their absolute nature. They're great for parsing, or link and sync data-level objects. But they're really not helpful for finding out when an event happend during the game unless you know the timestamp of the segment start. Unfortunately, that's not always the case, as some data providers do not provide timestamps. Plus, the vast majority of data manipulation algorithms rely on relative rather than absolute timings, so we wan't to avoid the hassle of computing timedeltas all over the place.

.. TIP::
    We use the built-in ``datetime`` objects to handle timestamps, but only if they are *aware*, i.e. timezone information is provided. Some providers use local time-zones when coding these, others always use UTC. So *unaware* timezones can quickly become a problem. For timezone handling, we use the *pytz* package.


Frame-based
-----------

Frame-based objects depend on a ``framerate``\, which is an attribute in the respective classes. The frame rate denotes the number of frames per second, and typically ranges from one up to a hundred for tracking data. It is important to know for every analysis that is time-sensitive. Each frame thus has a frame number, which can be used for indexing.

We decided that the frame number suffices as the *only* reference for time in frame-based objects. There are no timestamps or other time information attached to single frames. Instead, the timestamp of a frame is *implicitly* encoded by its index position in the array. For example, tracking data stored in a :doc:`XY <../modules/core/xy>` object is encoded as a big array where rows are frames, and columns are player's *x*\- and *y*\-coordinates.

.. TIP::
    You can rely on (and need to take care of) frame-based objects spanning the entire segment at a given frame rate. If your segment is 10 seconds long, and you have a frame rate of 25, there should be a total of 250 frames. Missing data such as skipped frames during data acquisition or player's missing half the segment due to substitutions are instead replaced by *numpy*'s ``np.nan``\s.

You might argue that this is not very Pythonic, but we found that it leads to much leaner objects and let's us use the full power of *numpy*, such as indexing or slicing! It's rather straightforward in this format to manipulate xy or code data at once, and fast when vectorizing manipulations over the whole time dimension.


Handling
========

Of course we try to support any usage of whatever time-information-identification that you prefer. For many purposes, its easier to use the scoreboard clock (e.g. for printing out stuff) or timestamps (e.g. for linking stuff). Internally, however, we rely on the ``gameclock`` as much as possible. This is due to the robustness reasons given above. But the real deal is the case of joint manipulation of frame-based and list-based objects!

Within a segment (e.g. relative to its start), list-based objects can be time-identified with the ``gameclock``, and frame-based objects with index positions (plus a framerate). To join the former with the latter, let's introduce another clock, the ``frameclock``\. Whereas the ``gameclock`` measures elapsed times in seconds, the ``frameclock`` measures elapsed time in frames for a given frame rate. So it's really just ``frameclock = int(gameclock * framerate)``\, but its helpful link to get all objects on the same page (or clock, if you like).

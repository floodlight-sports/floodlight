==============================
Data - Structuring Information
==============================

The core idea behind this package is to provide streamlined data structures that can hold the various information contained in sports data. This is realized by a set of Python objects, each of which is handling one - *and just one* - type of information. But before we dig deeper into implementation details we need to take a close look at what we're dealing with: the data.

Typically, sports data from across major providers differs drastically in format and shape. By now you've probably heard about three typical sources of data: tracking or position data, event data, and video. Each of these three data sources has its own history, and their appearances have changed over the years. So what does this mean for all those files on your machine?


Provider Data
=============

**Event data** has its root in so called hand annotations: back in the days analysts came up with annotation schemes to manually track actions over the course of the game on their notepads. This task is now primarily in the hands of companies that supply sports organizations with packages of entire leagues being annotated - match by match, and still mostly by hand.

There exist virtually no conventions and best practices in this domain and consequently, each provider has its own scheme in which they record the games. These schemes differ dramatically in terms of scope, depth, accuracy, and the used definitions for game actions. As the conceptualizations differ, so do implementations and data formats. And as fine-grained descriptions of match play become complex rather quickly, the logic used when storing this data in XML- or JSON-files is usually fundamentally different across providers.

**Video** is perhaps the most important data source in the everyday life of a professional organization. We won't touch this topic right now, as video is not yet supported in our package. We plan to change that in the future, though.

**Tracking data** has not been around for long, but it came here to stay. It's just fascinating seeing those dots move around the screen, and modelling possibilities seem endless. One must acknowledge though that not all tracking data are created equal. There's GPS, multi camera tracking or (local) sensor-based tracking. And its acquisition has a strong influence on the eventual data. It can be represented as latitudes and longitudes or coordinates in a reference system, frame rates changing from one all the way to a hundred Hertz, additional data such as accelerations or impacts mixed in, and then, there's the analysts final enemy: missing and inaccurate data.

But wait, there's even more! If you've got your hands on recent tracking data, you've might not only see player positions in the corresponding files but also frame-by-frame **contextual tags** regarding ball possession or whether the play is interrupted. And of course, professional sports teams still have their own analysts who cover every match **manually coding** phases of interesting play for effective post-game video analysis. Plus, as of recently, advances in deep learning have produced first results on generating player actions from video or tracking data, or tracking data from moving, single-camera setups. The list goes on...


The Challenge
=============

By now you might be able to imagine all those little obstacles one can run into. And if you've ever worked with this data, you most probably have.  And that's all concerning just a *single* source of data. The real trouble starts now: mixing up or attempting to bring things together. Event- and tracking data are still awfully out of sync (at least that started to change). Different frame rates or coordinate systems quickly cause headaches.

.. DANGER::
    You can have a tough time when working with event data and switching from *provider A* to *provider B*. It's becoming worse when working with two data sources, one from *provider A*, one from *provider B* which turn out to be...not exactly compatible. And at this point, it turns into a plain nightmare if you start replacing any part of your pipeline with that new cool stuff from *provider C*.

Oh, and so far we've only talked about data for single matches. What we haven't yet talked about is all that match meta-information: lineups, scorelines, standings, location, weather, and so on. Or, how do you even link information across different matches? There are of course extensive databases with elaborate ID systems that can identify every player, club or referee, including their personal and career information. Unfortunately, they obviously change across providers. By the way, what happens if you don't want to analyze last saturday's match, but the tracking data from today's training session containing six little drills, partly in parallel, taking place on different areas on the pitch?

Let's get serious again. Sports data analysis is awesome, but it can become quickly complicated and rather tedious on the implementation level. All this complication leads to - in line with good ol' data analysis tradition - massive overhead time needed for data parsing, pre-processing and wrangling. Furthermore, the formal incompatibility of different data sources is a noticeable hindrance on unfolding the data's full potential. There's a good reason why hardly any applications or scientific publications exist that combine two of the aforementioned data sources (with a few exceptions).


Core Objects
============

We aim to tackle this challenge with this package. The basic idea is rather simple: To formalize the logic behind team sport data and break down inherent complexity into standalone data structures by *abstraction* and *generalization*. The resulting objects should be independent from any data provider or source. They should be clear and intuitive to use and allow a clean interface to data loading and processing. That way, any data processing is - in typical object-oriented fashion - attached to the data objects. And effectively decoupled from any provider specifics.

To realize this idea, we've tried to break down all that information you can extract from team sports data and come up with our own systemization. In essence, we introduce core data objects on three levels:

1. **Data level objects** store raw data such as player positions, events, or the used coordinate system. These are essentially independent data fragments that in itself do not carry any further information of where they come from. They are pure data structures with methods concerned with data manipulation: spatial or temporal transforms, clipping and slicing, modifications, visualizations, and so on.

2. **Observation level objects** are concerned with bundling and enriching data level objects into "meaningful" data structures. Each observation, such as a match or training drill, can contain a number of data level objects for each *segment* (such as half times) and *team* (such as the home and away team). An observation-level object contains all these data-level objects and further incorporates objects regarding match or player information.

3. **Analysis level objects** contain analysis-related objects such as performance metrics or high-level models of match play.

The exact schema is further discussed in the :doc:`core <../modules/core/core>` submodule reference. On the following pages we also discuss a range of topics that are directly linked to the creation and handling of these core data structures, such as handling spatial and temporal data, identities, and so on. This should hopefully further clarify our implemented conceptualization.


But... why?
===========

At this point you might be rightfully asking yourself: Why? Why do we need another package that introduces its own data structures and ways of dealing with certain problems? And, to be honest, what's the purpose of trying to integrate all these different files and fit them into a single framework? Especially since there already exist packages that aim to solve certain parts of that pipeline?

The answer is, while we love those packages out there, that we did not find a solution that did fit our needs. Available packages are either tightly connected to a certain data format, or solve *one* particular problem. Ultimately, this means that each of these isolated solutions has their own interface. And this still leaves us with the core problem discussed on this page: connecting all those, partly incompatible, interfaces.

We felt that as long as there is no underlying, high-level framework, each and every use case again and again needs its own implementation. At last, we found ourselves refactoring the same code - and there are certain data processing or plotting routines that are required in *almost every* project - over and over again just to fit the particular data structures we we're dealing with at that time.

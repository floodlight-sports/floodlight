==============================
Data - Structuring Information
==============================

The core idea behind this package is to provide streamlined data structures that can hold the various information contained in sports data. This is realized by a set of Python classes, each of which is handling exactly one type of information. But before we dig deeper into implementation details we start with a closer look at what we're dealing with: the data.

We found that sports data from across major providers typically differs drastically in format and shape. In general, there are three main data types that can be identified: tracking or position data, event data, and video. Each of these three data sources has its own history, and their appearances have changed over the years. Let's have a quick look at each one of them.


Provider Data
=============

**Event data** has its root in so called hand annotations: back in the days analysts came up with annotation schemes to manually track actions over the course of the game on their notepads. This task is now primarily in the hands of companies that supply sports organizations with packages of entire leagues being annotated - match by match, and still mostly by hand.

As far as we know, there hardly exist any conventions in this domain and consequently, each provider has its own scheme in which they record the games. These schemes differ dramatically in terms of scope, depth, accuracy, and the used definitions for game actions. As the conceptualizations differ, so do implementations and data formats. And as fine-grained descriptions of match play become complex rather quickly, the logic used when storing this data in XML- or JSON-files is usually fundamentally different across providers.

**Video** is perhaps the most important data source in the everyday life of a professional match analyst. We won't touch this topic right now, as video is not yet supported in our package.

**Tracking data** has not been around for long, but is gaining popularity quickly. It's just fascinating seeing those dots move around the screen, and modelling possibilities seem endless. Not all "tracking data" are the same though, and their final shape differs in terms of the data acquisition technique. Roughly speaking, the most prominent sources are GPS sensors, multi camera tracking or (local) sensor-based tracking. The acquisition has a strong influence on the eventual data in terms of quality, spatial and temporal resolution, as well as the actual data format. It can be represented as latitudes and longitudes or coordinates in a reference system, frame rates change from one up to a hundred Hertz, sometimes additional body sensor data such as accelerations or impacts are mixed in, and then, there's the big issue of missing and inaccurate data.

Although these are the three main data sources, sometimes there's even more data collected for a match! Looking at recent tracking data, they often come with frame-by-frame **contextual tags** regarding ball possession or ball status, i.e., information whether the play is underway or interrupted. Additionally, professional sports teams still have their own analysts who cover every match **manually coding** phases of interesting play for effective post-game video analysis. Plus, as of recently, advances in deep learning have produced first results on generating player actions from video or tracking data, or tracking data from moving, single-camera setups.


The Challenge
=============

The sheer amount of data, it's different types, formats and heterogeneity are a natural cause of complexity for the analysis process. If you've ever worked with any one these data sources, you most probably have encountered some challenges during processing. The task becomes even more complex when integrating multiple of these data sources. Event- and tracking data are still typically out of sync due to timing errors in event data acquisition. Differing frame rates or coordinate systems are another hurdle to take in multimodal analyses.

Another major challenge are analyses that contain more than a single match. In this case, match meta-information come into play: lineups, scorelines, standings, location, weather, and so on. Linking information across different matches requires keeping track of player and or team IDs, substitutions, and so on. Depending on the provider, there exist extensive databases with elaborate ID systems that can identify every player, club or referee, including their personal and career information. Unfortunately, they (naturally) change across providers.

Last but not least, the events under observation (we collectively call these *observations*, such as matches, training drills or study experiments) can also differ in shape. Even a typical match can sometimes have overtime periods, or penalty shootouts, which changes requirements to the deployed algorithms. But especially data from training sessions or experimental setups can vary strongly in terms of duration, pitch dimensions, or the number of involved players.

To sum up, sports data analysis is awesome, but it can become quickly complicated and rather tedious on the implementation level looking at these challenges. All this complication ultimately leads to massive overhead effort needed for data parsing, pre-processing and wrangling. Furthermore, the formal incompatibility of different data sources is a noticeable hindrance on unfolding the data's full potential. There's a good reason why, to our knowledge, hardly any applications or scientific publications exist that combine two of the aforementioned data sources (with a few exceptions).


Core Objects
============

As stated before, the aim of this package is to tackle some of these challenges. The starting point is to formalize the logic behind team sport data and systematically break down inherent complexity into stand-alone data structures by abstraction and generalization. Most importantly, the desired data classes should be independent from any data provider or source. They should also be performant, clear and intuitive to use and allow a clean interface to data loading and processing. That way, any data processing is attached to the data objects and effectively decoupled from any provider specifics.

To realize this idea, we've attempted to break down all that information you can extract from team sports data and come up with systemization that translates smoothly to a object-oriented implementation. Generally speaking, we introduce core data objects on three levels:

1. **Data level objects** store raw data such as player positions, events, or the used coordinate system. These are essentially independent data fragments that in itself do not carry any further information of where they come from. Instead, they are pure data structures with methods concerned with data manipulation: spatial or temporal transforms, clipping and slicing, modifications, visualizations, and so on. Each fragment (and thus object instance) only stores data for one *team* (such as the home and away team) and temporal *segment* (such as a halftime).

2. **Observation level objects** are concerned with bundling and enriching data level objects into "meta" data structures. Each observation, such as a match or training drill, can contain a number of data level objects for each segment and team. An observation-level object contains all respective data-level objects and further incorporates objects regarding match or player information.

3. **Analysis level objects** contain analysis-related objects such as performance metrics or high-level models of match play.

On the following pages we discuss a range of topics that are directly linked to the creation and handling of these core data structures, such as handling spatial and temporal data, identities, and so on.


But... why?
===========

Before we proceed, a quick personal note on the necessity of this package. At this point you might be rightfully asking yourself: Why do we need another package that introduces its own data structures and ways of dealing with certain problems? And what's the purpose of trying to integrate all different data sources and fit them into a single framework? Especially since there already exist packages that aim to solve certain parts of that pipeline?

Our answer is - although we love those packages out there - that we did not find a solution that did fit our needs. Available packages are either tightly connected to a certain data format/provider, adapt to the subtleties of a
particular sport, or solve *one* particular problem. Ultimately, this means that each of these isolated solutions has their own interface. And this still left us with the core problem discussed on this page: connecting all those, partly incompatible, interfaces.

We felt that as long as there is no underlying, high-level framework, each and every use case again and again needs its own implementation. At last, we found ourselves refactoring the same code - and there are certain data processing or plotting routines that are required in *almost every* project - over and over again just to fit the particular data structures we we're dealing with at that time.

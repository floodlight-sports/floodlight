[version-image]: https://img.shields.io/pypi/v/floodlight?color=006666
[version-url]: https://pypi.org/project/floodlight/
[python-image]: https://img.shields.io/pypi/pyversions/floodlight?color=006666
[python-url]: https://pypi.org/project/floodlight/
[docs-image]: https://readthedocs.org/projects/floodlight/badge/?version=latest
[docs-url]: https://floodlight.readthedocs.io/en/latest/?badge=latest
[tutorial-url]: https://floodlight.readthedocs.io/en/latest/guides/getting_started.html
[build-image]: https://github.com/floodlight-sports/floodlight/actions/workflows/build.yaml/badge.svg
[build-url]: https://github.com/floodlight-sports/floodlight/actions/workflows/build.yaml
[lint-image]: https://github.com/floodlight-sports/floodlight/actions/workflows/linting.yaml/badge.svg
[lint-url]: https://github.com/floodlight-sports/floodlight/actions/workflows/linting.yaml
[black-image]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-url]: https://github.com/psf/black
[contrib-image]: https://img.shields.io/badge/contributions-welcome-006666
[contrib-url]: https://github.com/floodlight-sports/floodlight/blob/main/CONTRIBUTING.md
[arxiv-image]: https://img.shields.io/badge/arXiv-2206.02562-b31b1b.svg
[arxiv-url]: https://arxiv.org/abs/2206.02562
[joss-image]: https://joss.theoj.org/papers/10.21105/joss.04588/status.svg
[joss-url]: https://doi.org/10.21105/joss.04588
[codecov-image]: https://codecov.io/gh/floodlight-sports/floodlight/branch/develop/graph/badge.svg?token=RLY582UBC6
[codecov-url]: https://codecov.io/gh/floodlight-sports/floodlight


# floodlight

[![Latest Version][version-image]][version-url]
[![Python Version][python-image]][python-url]
[![Documentation Status][docs-image]][docs-url]
[![Build Status][build-image]][build-url]
[![Linting Status][lint-image]][lint-url]
[![Codecov][codecov-image]][codecov-url]
[![DOI][joss-image]][joss-url]

## A high-level, data-driven sports analytics framework

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

----------------------------------------------------------------------------------------

* [Quick Demo](#quick-demo)
* [Features](#features)
* [Installation](#installation)
* [Documentation](#documentation)
* [How to contribute](#contributing)

----------------------------------------------------------------------------------------

### Quick Demo

**floodlight** simplifies sports data loading, processing and advanced performance
analyses. Check out the example below, where querying a public data sample, filtering
the data and computing the expended metabolic work of the active home team players is
done in a few lines of code:

```
>>> from floodlight.io.datasets import EIGDDataset
>>> from floodlight.transforms.filter import butterworth_lowpass
>>> from floodlight.models.kinetics import MetabolicPowerModel

>>> dataset = EIGDDataset()
>>> home_team_data, away_team_data, ball_data = dataset.get()

>>> home_team_data = butterworth_lowpass(home_team_data)

>>> model = MetabolicPowerModel()
>>> model.fit(home_team_data)
>>> metabolic_power = model.cumulative_metabolic_power()

>>> print(metabolic_power[-1, 0:7])

[1669.18781115 1536.22481121 1461.03243489 1488.61249785  773.09264071
 1645.01702421  746.94057676]
```

To find out more, see the full set of features below or get started quickly with
[one of our many tutorials][tutorial-url] from the official documentation!


### Features

We provide core data structures for team sports data, parsing functionality for major
data providers, access points to public data sets, data filtering, plotting routines and
many computational models from the literature. The feature set is constantly expanding,
and if you want to add more just open an issue!

#### Data-level Objects

- Tracking data
- Event data
- Pitch information
- Teamsheets with player information (*new*)
- Codes such as ball possession information
- Properties such as distances or advanced computations

#### Parser

- **Tracab/ChyronHego**: Tracking data, Teamsheets, Codes
- **DFL/STS**: Tracking data, Event data, Teamsheets, Codes
- **Kinexon**: Tracking data
- **Opta**: Event data (F24 feeds)
- **Second Spectrum**: Tracking data, Event data (*new*)
- **Sportradar**: Event data (*new*)
- **StatsPerform**: Tracking data, Event data (with URL access)
- **StatsBomb**: Event data

#### Datasets

- EIGD-H (Handball tracking data)
- StatsBomb OpenData (Football event data)

#### Manipulation and Plotting

- Spatial transformations for all data structures
- Lowpass-filter tracking data
- Slicing, selection and sequencing methods
- Plot pitches, player positions and model overlays

#### Models and Metrics

- Approximate Entropy
- Centroids
- Distances, Velocities & Accelerations
- Metabolic Power & Equivalent Distances
- Voronoi Space Control (*new*)

### Installation

The package can be installed easily via pip:

```
pip install floodlight
```


### Documentation

You can find all documentation [here][docs-url].



### Contributing

[![Contributions][contrib-image]][contrib-url]
[![Code style: black][black-image]][black-url]


Check out [Contributing.md][contrib-url] for a quick rundown of what you need to
know to get started. We also provide an extended, beginner-friendly guide on how to
start contributing in our documentation.



### Citing

If you've used *floodlight* in your scientific work, please cite the [corresponding paper][joss-url].

```
@article{Raabe2022,
    doi = {10.21105/joss.04588},
    url = {https://doi.org/10.21105/joss.04588},
    year = {2022},
    publisher = {The Open Journal},
    volume = {7},
    number = {76},
    pages = {4588},
    author = {Dominik Raabe and Henrik Biermann and Manuel Bassek and Martin Wohlan and Rumena Komitova
              and Robert Rein and Tobias Kuppens Groot and Daniel Memmert},
    title = {floodlight - A high-level, data-driven sports analytics framework},
    journal = {Journal of Open Source Software}
}
```



### Why

Why do we need another package that introduces its own data structures and ways of dealing with certain problems?
And what's the purpose of trying to integrate all different data sources and fit them into a single framework?
Especially since there already exist packages that aim to solve certain parts of that pipeline?

Our answer is - although we love those packages out there - that we did not find a solution that did fit our needs.
Available packages are either tightly connected to a certain data format/provider, adapt to the subtleties of a
particular sport, or solve *one* particular problem. This still left us with the essential problem of adapting to
different interfaces.

We felt that as long as there is no underlying, high-level framework, each and every use case again and again needs its
own implementation. At last, we found ourselves refactoring the same code - and there are certain data processing or
plotting routines that are required in *almost every* project - over and over again just to fit the particular data
structures we're dealing with at that time.


### About

This project has been kindly supported by the [Institute of Exercise Training and Sport
Informatics](https://www.dshs-koeln.de/en/institut-fuer-trainingswissenschaft-und-sportinformatik/) at the German Sport
University Cologne under supervision of Prof. Daniel Memmert.



### Related Projects

- [matplotsoccer](https://github.com/TomDecroos/matplotsoccer)
- [kloppy](https://github.com/PySport/kloppy)
- [codeball](https://github.com/metrica-sports/codeball)

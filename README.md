[version-image]: https://img.shields.io/pypi/v/floodlight?color=006666
[version-url]: https://pypi.org/project/floodlight/
[python-image]: https://img.shields.io/pypi/pyversions/floodlight?color=006666
[python-url]: https://pypi.org/project/floodlight/
[docs-image]: https://readthedocs.org/projects/floodlight/badge/?version=latest
[docs-url]: https://floodlight.readthedocs.io/en/latest/?badge=latest
[build-image]: https://github.com/floodlight-sports/floodlight/actions/workflows/build.yaml/badge.svg
[build-url]: https://github.com/floodlight-sports/floodlight/actions/workflows/build.yaml
[lint-image]: https://github.com/floodlight-sports/floodlight/actions/workflows/linting.yaml/badge.svg
[lint-url]: https://github.com/floodlight-sports/floodlight/actions/workflows/linting.yaml
[black-image]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-url]: https://github.com/psf/black
[contrib-image]: https://img.shields.io/badge/contributions-welcome-006666
[contrib-url]: https://github.com/floodlight-sports/floodlight/blob/main/CONTRIBUTING.md

# floodlight
[![Latest Version][version-image]][version-url]
[![Python Version][python-image]][python-url]
[![Documentation Status][docs-image]][docs-url]
[![Build Status][build-image]][build-url]
[![Linting Status][lint-image]][lint-url]
[![Contributions][contrib-image]][contrib-url]
[![Code style: black][black-image]][black-url]


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

* [Features](#features)
* [Installation](#installation)
* [Documentation](#documentation)
* [How to contribute](#contributing)

----------------------------------------------------------------------------------------


### Features

This project is still under development, and we hope to expand the set
of features in the future. At this point, we provide core data structures,
parsing functionality for major data providers, access to public data sets, data
filtering, basic plotting routines and computational models.

#### Data-level Objects

- Tracking data
- Event data
- Pitch information
- Codes such as ball possession information
- Properties such as distances or advanced computations

#### Parser

- ChyronHego (Tracking data, Codes)
- DFL (Tracking data, Event data, Codes)
- Kinexon (Tracking data)
- Opta (Event data - F24 feeds)
- Second Spectrum (Tracking data)
- StatsPerform (Tracking data, Event data - also directly from URLs)
- StatsBomb (Event data)

#### Datasets

- EIGD-H (Handball tracking data)
- StatsBomb OpenData (Football event data)

#### Manipulation and Plotting

- Spatial transformations for all data structures
- Lowpass-filter tracking data
- Slicing and selection methods
- Plot pitches and tracking data

#### Models and Metrics

- Centroids
- Distances, Velocities, Accelerations
- Metabolic Power and Equivalent Distances
- Approximate Entropy

### Installation

The package can be installed easily via pip:

```
pip install floodlight
```


### Documentation

You can find all documentation [here][docs-url].



### Contributing

Check out [Contributing.md][contrib-url] for a quick rundown of what you need to
know to get started. We also provide an extended, beginner-friendly guide on how to
start contributing in our documentation.



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

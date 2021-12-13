[version-image]: https://img.shields.io/badge/status-beta-006666
[version-url]: https://img.shields.io/badge/status-beta-006666
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
[![PyPI][version-image]][version-url]
[![Documentation Status][docs-image]][docs-url]
[![Build Status][build-image]][build-url]
[![Linting Status][lint-image]][lint-url]
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


### Features

This project is still in its early childhood, and we hope to quickly expand the set
of features in the future. At this point, we've implemented core data structures and
parsing functionality for major data providers.

#### Data objects

- Data-level objects to store
  - Tracking data
  - Event data
  - Pitch information
  - Codes such as ball possession information

#### Parser

- ChyronHego
  - Tracking data
  - Codes
- DFL
  - Tracking data
  - Codes
  - Event data
- Kinexon
  - Tracking data
- Opta
  - Event data (f24 feeds)
- Stats Perform
  - Tracking data
  - Event data


### Installation

The package can be installed easily via pip:

```
pip install floodlight
```


### Contributing [![Contributions][contrib-image]][contrib-url]

Check out [Contributing.md][contrib-url] for a quick rundown of what you need to
know to get started. We also provide an extended, beginner-friendly guide on how to
start contributing in our documentation.


### Documentation

You can find all documentation [here][docs-url].


### Why

Why do we need another package that introduces its own data structures and ways of dealing with certain problems? And,
to be honest, what's the purpose of trying to integrate all these different files and fit them into a single framework?
Especially since there already exist packages that aim to solve certain parts of that pipeline?

The answer is, while we love those packages out there, that we did not find a solution that did fit our needs.
Available packages are either tightly connected to a certain data format/provider, adapt to the subtleties of a
particular sport, or only solve *one* particular problem. This still left us with the essential problem of adapting to
all those different interfaces.

We felt that as long as there is no underlying, high-level framework, each and every use case again and again needs its
own implementation. At last, we found ourselves refactoring the same code - and there are certain processing or
plotting routines that are required in *almost every* project - over and over again, just to fit the particular data
structures we were dealing with at that time.


### Related Projects

- [matplotsoccer](https://github.com/TomDecroos/matplotsoccer)
- [kloppy](https://github.com/PySport/kloppy)
- [codeball](https://github.com/metrica-sports/codeball)

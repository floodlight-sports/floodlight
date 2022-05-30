**floodlight** Documentation
============================

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


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Guides

    Getting started <guides/getting_started>
    Contributing <guides/contrib_manual>
    Tutorial: Data Analysis <guides/tutorial_analysis>
    Tutorial: Match Sheets <guides/tutorial_matchsheets>


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Compendium

    Intro <compendium/0_compendium>
    Data <compendium/1_data>
    Design <compendium/2_design>
    Time <compendium/3_time>
    Space <compendium/4_space>
    Identifier <compendium/5_identifier>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Module Reference

   modules/core/core
   modules/io/io
   modules/metrics/metrics
   modules/models/models
   modules/transforms/transforms
   modules/utils/utils
   modules/vis/vis

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

---
title: 'floodlight - A high-level, data-driven sports analytics framework'
tags:
  - Python
  - sport science
  - performance analysis
  - exercise physiology
  - training science
  - collective movement behavior
  - sports analytics
authors:
  - name: Dominik Raabe^[Corresponding author]
    orcid: 0000-0001-7264-4575
    affiliation: 1
  - name: Henrik Biermann
    orcid: 0000-0001-5660-9876
    affiliation: 1
  - name: Manuel Bassek
    orcid: 0000-0002-9394-913X
    affiliation: 1
  - name: Martin Wohlan
    orcid: 0000-0002-7403-7338
    affiliation: 1
  - name: Rumena Komitova
    orcid: 0000-0003-1300-9330
    affiliation: 1
  - name: Robert Rein
    orcid: 0000-0001-7059-5319
    affiliation: 1
  - name: Tobias Kuppens Groot
    orcid: 0000-0003-1323-1628
    affiliation: 2
  - name: Daniel Memmert
    orcid:  0000-0002-3406-9175
    affiliation: 1
affiliations:
 - name: Institute of Exercise Training and Sport Informatics, German Sport University Cologne, Germany
   index: 1
 - name: Independent Researcher, Netherlands
   index: 2
date: 07 June 2022
bibliography: paper.bib
---

# Summary

The increase of available data has had a positive impact on the entire sports domain and especially sport science [@Morgulev2018]. Two major data sources of relevance in this domain are spatiotemporal tracking data of athlete positions as well as manually annotated match event data [@Stein2017; @Memmert2018]. These two data types are regularly collected by professional sport organizations in different team invasion games such as football, basketball, or handball [@Memmert2021]. These data sources open up a whole range of new analysis possibilities across multiple (sub)disciplines in the field, including match and performance analysis, exercise physiology, training science, or collective movement behavior analysis. As an example, player tracking data has been used extensively to analyze physical [@Castellano2014] as well as tactical [@Rein2016] performance in football.

The *floodlight* Python package provides a framework to support and automate team sport data analysis. *floodlight* is constructed to process spatiotemporal tracking data, event data, and other game meta-information to support scientific performance analyses. *floodlight* was designed to provide a general yet flexible approach to performance analysis, while simultaneously providing a user-friendly high-level interface for users with basic programming skills. The package includes routines for most aspects of the data analysis process, including dedicated data classes, file parsing functionality, public dataset APIs, pre-processing routines, common data models and several standard analysis algorithms previously used in the literature, as well as basic visualization functionality.

The following features are implemented:

**Data-level Objects**: ``XY`` (tracking data), ``Events`` (event data), ``Code`` (meta information), ``Pitch`` (pitch layout), ``PlayerProperty`` (player information per frame), ``TeamProperty`` (team information per frame),``DyadicProperty`` (player interaction information per frame).

**Data parser**: For files from ChyronHego (tracking data, codes), DFL (tracking data, codes), Kinexon (tracking data), Opta (event data), Second Spectrum (tracking data), StatsPerform (tracking data, event data) StatsBomb (event data).

**Datasets**: EIGD-H, StatsBomb OpenData

**Processing**: Spatial transforms (tracking and event data), Butterworth and Savitzky-Golay lowpass filter (tracking data), data slicing (all temporal objects), selection and sequencing (event data, codes).

**Visualization**: Pitches (football and handball), player positions, player trajectories.

**Data models**: Distances, distance covered, velocities, accelerations, centroids, centroid distances, stretch index, metabolic power, equivalent distance, approximate entropy.

**Documentation**: Module reference, extended contributing guide, team sports data analysis compendium, tutorials (getting started, analyzing data, preparing match sheets).

Central to the package is a set of generalized, provider- and sports-independent core data structures based on *numpy* [@Harris2020] and *pandas* [@McKinney2010]. Each of these data structures are dedicated to one specific type of sports data, including spatiotemporal tracking data, event data, game codes (meta information such as ball possession information), pitch information regarding the embedding of data and playing surfaces into Cartesian coordinate systems, as well as team and player properties (such as frame-wise velocity or acceleration values). The data structures are designed with a focus on scientific computing, i.e., optimized for accessible and intuitive data manipulations as well as sensitive to performance by utilizing *numpy*'s view-, vectorization- and indexing techniques.

![Positions of football players (left) and trajectories of handball players (right) from real-world match data as visualized with *floodlight*.\label{fig:sample}](plotting_sample.png)

The core data classes allow internal storage and processing of sports data whilst decoupling from any format-specific requirements. Consequently, *floodlight* is built around these objects, comprising several elementary modules of the data processing pipeline. For data loading, the package provides parsing submodules with functions that dissect and map data from specific provider formats to core data structures (including providers such as Kinexon, Tracab, Stats Perform, StatsBomb, Second Spectrum, Opta, or DFL), which eliminates problems caused by the many, strongly varying data formats in use. Data loaders and mappers for available public datasets such as the EIGD-H dataset [@Biermann2021] are additionally included. In terms of data processing, the package provides dedicated manipulation functionality such as spatial transformations helpful for spatial data synchronization or signal filters based on *scipy* [@Virtanen2020]. For data inspection, basic visualization functionality based on the *matplotlib* package [@Hunter2007] is included (see \autoref{fig:sample}).

The actual data analysis part is realized by a submodule providing several data models. These models provide a toolbox of domain-specific data analysis procedures from different subdomains such as exercise physiology, e.g., the metabolic power model [@diPrampero2018], dynamical system approaches, e.g., approximate entropy [@Pincus1991], or collective tactical behavior, e.g., centroid-based measures [@Sampaio2012; @Bourbousson2010]. All models follow the same syntax inspired by the *scikit-learn* package [@Buitinck2013], where upon instantiation, a central fitting method is called with core data structures. Subsequently, required computations can be queried with additional class methods. This allows a consistent syntax and collection of similar measures into cohesive data models while limiting the repetition of basic calculations and allowing simple future extensions.

The following code sample illustrates how *floodlight* reduces a typical performance analysis pipeline to just a few lines of code. In the example, one sample of data is queried from the public EIGD-H dataset, filtered, and the cumulative metabolic work of the home team is calculated for the entire segment of data:

```
from floodlight.io.datasets import EIGDDataset
from floodlight.transforms.filter import butterworth_lowpass
from floodlight.models.kinetics import MetabolicPowerModel

dataset = EIGDDataset()
home_team_data, away_team_data, ball_data = dataset.get()

home_team_data = butterworth_lowpass(home_team_data)

model = MetabolicPowerModel()
model.fit(home_team_data)
metabolic_power = model.cumulative_metabolic_power()
```

These lines of code create a new core object named ``metabolic_power``, which contains an array of shape (*T* x *N*) storing (cumulated) metabolic power values (in joule per kilogram) for each of the *T* time points and *N* players. For example, we can print the total metabolic work of the seven active players:

```
>>> print(metabolic_power[-1, 0:7])

[1669.18781115 1536.22481121 1461.03243489 1488.61249785  773.09264071
 1645.01702421  746.94057676]
```

# Statement of need

Despite the increase in volume, the technical requirements for team sport data analysis have constantly remained high. This can be partially attributed to the complexity and heterogeneity of the data itself [@Stein2017; @Memmert2018], but also to multiple practical and theoretical challenges. These include the necessity of complex file parsing procedures for provider-specific data formats, low compatibility across data providers, or differing standards for spatial or temporal resolution of data, often requiring specialized pre-processing routines. Meeting these challenges typically requires massive and customized overhead programming in sports data analysis projects. At the same time, there hardly exist any general, proprietary or open source, software alternatives which can be used out of the box for scientific purposes. Existing software is either commercially driven (i.e., proprietary, limited to a specific data provider or focused on industrial applications), or task-specific (i.e., limited to a certain data source, data format, sport or subtask) which leaves the problem of adapting code to multiple different APIs within the analysis process.

These current constraints resulted in a situation where a typical analysis workflow requires the (re)implementation of each processing pipeline module in its entirety with respect to the specific project's needs. For sport scientists who typically lack programming skills (which are usually not part of their formal training) this can become an insurmountable hurdle. As a consequence, advanced team sports data analyses remain inaccessible for large parts of the sport scientific community which poses a significant hindrance for future progress. Accordingly, the *floodlight* package was designed to specifically address this problem and significantly ease advanced analyses of sports data. *floodlight* automates standard data processing routines and provides a high-level interface accessible to users with just basic programming skills. The *floodlight* documentation contains several tutorials as well as an extensive compendium discussing the technical aspects of team sports data analysis to ensure easy access and understanding of the routines and their design choices. The tutorials increase the beginner-friendliness of *floodlight* and allow its usage in educational settings, e.g., for team sport data analytics courses.

Another hurdle faced by sports scientists relates to the current lack of collaboration and code sharing practices within the field. At present, sharing proposed data models or algorithms for analyses is the exception rather than the rule. In parts, the lack of sharing often stems from the proprietary nature of the raw data, but is further exacerbated by lack of data format gold standards. More generally, disciplines that employ team sports data analysis have reported a culture that contains very little replications and works incorporating previous findings [@Herold2019], low applicability of research by practitioners [@Bishop2008; @Mackenzie2013; @Herold2019] and limited interdisciplinary approaches between computer and sport scientists [@Rein2016; @Goes2021]. A major milestone in the process of meeting these challenges is to find feasible ways of sharing data and algorithms [@Rein2016]. The *floodlight* package can be seen as a first step in this direction with a toolbox-approach collecting common data manipulation and processing techniques.

*floodlight* will therefore be equally useful for sports scientists as well as computer scientists, working in academia or applied settings. The package will therefore serve to bring these users groups together and foster future interdisciplinary collaborations. Ideally, this will also promote further open source contributions that share advanced data processing algorithms in the domain and enable future work incorporating previous findings.


# Acknowledgements

This project has received funding from the German Federal Ministry of Education and Research (BMBF) to the last author under grant number 01IS20021A.

# References

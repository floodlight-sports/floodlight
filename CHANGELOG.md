# Changelog

## 0.4.0 (2023-02-03)

#### New Features

* add discretized voronoi model ([#124](https://github.com/floodlight-sports/floodlight/issues/124))
* add parser for metadata and teamsheets in json format ([#113](https://github.com/floodlight-sports/floodlight/issues/113))
* integrate teamsheet logic to tracab parser ([#112](https://github.com/floodlight-sports/floodlight/issues/112))
* integrate teamsheet logic to secondspectrum parsers ([#111](https://github.com/floodlight-sports/floodlight/issues/111))
* integrate teamsheet logic to statsperform parsers ([#108](https://github.com/floodlight-sports/floodlight/issues/108))
* integrate teamsheet logic to statsbomb parser  ([#109](https://github.com/floodlight-sports/floodlight/issues/109))
* integrate teamsheet logic to dfl parser ([#106](https://github.com/floodlight-sports/floodlight/issues/106))
* add sportradar parser ([#110](https://github.com/floodlight-sports/floodlight/issues/110))
* add teamsheets core object ([#103](https://github.com/floodlight-sports/floodlight/issues/103))
* add second spectrum insights parser ([#100](https://github.com/floodlight-sports/floodlight/issues/100))
#### Fixes

* io refactor fixes ([#117](https://github.com/floodlight-sports/floodlight/issues/117))
* set matplotlib backend env in workflows
* replace pandas iterrows ([#96](https://github.com/floodlight-sports/floodlight/issues/96))
* missing https in hyperlink leading to an unknown page when clicking on it in github ([#98](https://github.com/floodlight-sports/floodlight/issues/98))
* MetabolicPower framerate bug ([#93](https://github.com/floodlight-sports/floodlight/issues/93))
* UTF-8 encoding Kinexon parser
* UFT-8 encoding Kinexon parser
* minimum signal length ([#81](https://github.com/floodlight-sports/floodlight/issues/81))
* statsperform pitch templates ([#75](https://github.com/floodlight-sports/floodlight/issues/75))
* adapt dfl parser to different format versions and fix statsperform gameclock unit ([#76](https://github.com/floodlight-sports/floodlight/issues/76))
* pass difference argument to axis-specific differentiation and fix prepend ([#77](https://github.com/floodlight-sports/floodlight/issues/77))
#### Refactorings

* flexible dictionary returns for parser ([#116](https://github.com/floodlight-sports/floodlight/issues/116))
* streamline IO function names ([#115](https://github.com/floodlight-sports/floodlight/issues/115))
#### Docs

* fix butterworth cutoff freq doc ([#94](https://github.com/floodlight-sports/floodlight/issues/94))
* update tutorial matchsheets ([#95](https://github.com/floodlight-sports/floodlight/issues/95))
* update changelog
* add information on setup.py
* update datasets description
* update readme
* update changelog
* add paper reference
#### Others

* bump version
* add dependency pytest-cov
* add dependency coverage
* bump version
* update location of flake8 which was moved from gitlab to github ([#99](https://github.com/floodlight-sports/floodlight/issues/99))
* add coverage reports generation and codecov upload
* update README
* update readme
* update module init
* update columns in tests

Full set of changes: [`0.3.1...0.4.0`](https://github.com/floodlight-sports/floodlight/compare/0.3.1...0.4.0)

## 0.3.1 (2022-06-06)

#### New Features

* add second spectrum parser
* add approximate entropy ([#67](https://github.com/floodlight-sports/floodlight/issues/67))
* add second spectrum parser
* add approximate entropy ([#67](https://github.com/floodlight-sports/floodlight/issues/67))
#### Fixes

* adapt test to changes
* broken settings path
* iterating stepsize
* include settings to module
* adapt test to changes
* broken settings path
* iterating stepsize
* include settings to module
#### Docs

* update changelog
* match sheet tutorial ([#71](https://github.com/floodlight-sports/floodlight/issues/71))
* overhaul contributing guide
* overhaul compendium
* extend getting started section
* add metrics and entropy module
* add data analysis tutorial
* remove redundancies
* move plotting examples from wrapper to function ([#69](https://github.com/floodlight-sports/floodlight/issues/69))
* clean class docstrings
* update changelog
* match sheet tutorial ([#71](https://github.com/floodlight-sports/floodlight/issues/71))
* overhaul contributing guide
* overhaul compendium
* extend getting started section
* add metrics and entropy module
* add data analysis tutorial
* remove redundancies
* move plotting examples from wrapper to function ([#69](https://github.com/floodlight-sports/floodlight/issues/69))
* clean class docstrings
#### Others

* bump version
* bump version
* fix typos
* update readme
* fix typos
* update readme

Full set of changes: [`0.3.0...0.3.1`](https://github.com/floodlight-sports/floodlight/compare/0.3.0...0.3.1)

## 0.3.0 (2022-05-23)

#### New Features

* add butterworth and savgol lowpass filter ([#51](https://github.com/floodlight-sports/floodlight/issues/51))
* add data transformation for eigd dataset
* add statsperform parser - standard format ([#41](https://github.com/floodlight-sports/floodlight/issues/41))
* add statsbomb open data parser and dataset ([#59](https://github.com/floodlight-sports/floodlight/issues/59))
* adds xy plotting methods with tests and docs ([#61](https://github.com/floodlight-sports/floodlight/issues/61))
* add pitch.plot method for football and handball ([#44](https://github.com/floodlight-sports/floodlight/issues/44))
* add vis module
* add slice and get_event_stream methods ([#60](https://github.com/floodlight-sports/floodlight/issues/60))
* add metabolic power model ([#48](https://github.com/floodlight-sports/floodlight/issues/48))
* add require_fit decorator
* add kinematics module - distance, velocity, acceleration models ([#45](https://github.com/floodlight-sports/floodlight/issues/45))
* create io eigd dataset ([#55](https://github.com/floodlight-sports/floodlight/issues/55))
* add is_fitted property
* add centroid-based model and measures
* add base-, team-, and dyadic properties
* add basic base model
* transformation methods for events object ([#47](https://github.com/floodlight-sports/floodlight/issues/47))
* add N property
* add is_metrical property
* add method for finding sequences
* add comparison dunder methods
* add models module
* add playerproperty class
#### Fixes

* broken link
* xy translate dtype-handling
* xy scale dtype-handling
* xy rotation dtype- and nan-handling
* add wraps to decorators to prevent shadowing
* broken link
* dfl events parser warnings, statsperform xID ([#66](https://github.com/floodlight-sports/floodlight/issues/66))
* minor fixes ([#65](https://github.com/floodlight-sports/floodlight/issues/65))
* xID indexing for dfl parser
* xID indexing for tracab parser
* extracting of zip with temorary file ([#58](https://github.com/floodlight-sports/floodlight/issues/58))
* typo
* typo
* dfl parser xml element access and memory release ([#52](https://github.com/floodlight-sports/floodlight/issues/52))
#### Refactorings

* refactor get and convert as general function
* sample_data.py -> ToyDataset() ([#57](https://github.com/floodlight-sports/floodlight/issues/57))
* change axis arguments from {0, 1} to {'x', 'y'}
* colum checks return empty lists instead of None
#### Docs

* update changelog
* add core module reference page
* add transforms module reference page
* add models module referene page
* add vis submodule
* add io module reference page
* correct xID indexing documentation
* include datasets in docs
* add property module reference
#### Others

* release 0.3.0
* bump version
* add dependency matplotlib
* add dependency scipy
* add dependency h5py
* release 0.2.1
* remove python 3.7 from actions matrix
* update readme
* update templates
* explicitly loop through players
* docstrings and errors
* add tests for dataset templates
* add test for eigd transform
* add tests for geometry model
* add test for require fit decorator
* add tests for base model
* column_in_range method ([#53](https://github.com/floodlight-sports/floodlight/issues/53))
* update tests for events core class ([#46](https://github.com/floodlight-sports/floodlight/issues/46))
* add test for N property
* add tests for is_metrical property
* add tests for find_sequences method
* add tests for dunder methods

Full set of changes: [`0.2.1...0.3.0`](https://github.com/floodlight-sports/floodlight/compare/0.2.1...0.3.0)

## 0.2.1 (2022-02-02)

#### Fixes

* change dtype of parsed codes to np.arrays
* change minute column to be relative to segment
* readjust negative gameclocks
* update project python version to match latest on runner
#### Docs

* update changelog
#### Others

* bump version
* release 0.2.0
* update readme

Full set of changes: [`0.2.0...0.2.1`](https://github.com/floodlight-sports/floodlight/compare/0.2.0...0.2.1)

## 0.2.0 (2021-12-13)

#### New Features

* add missing tests ([#31](https://github.com/floodlight-sports/floodlight/issues/31))
* add sample data ([#29](https://github.com/floodlight-sports/floodlight/issues/29))
* add method for xy rotation ([#25](https://github.com/floodlight-sports/floodlight/issues/25))
* general core functionality ([#24](https://github.com/floodlight-sports/floodlight/issues/24))
* basic events functionality ([#23](https://github.com/floodlight-sports/floodlight/issues/23))
* add kinexon parser ([#18](https://github.com/floodlight-sports/floodlight/issues/18))
* add several parsers ([#11](https://github.com/floodlight-sports/floodlight/issues/11))
* add event class skeleton
* add basic pitch object
* add dedicated typing module
* add xy core class skeleton
* add gitignore
#### Fixes

* block rotation matrix construction
* typo
* typo
#### Refactorings

* add skeleton
#### Docs

* add changelog
* update readme
* add project vision
* automatic version generation
* add dummy css to prevent warning
* remove misleading example
* add getting started chapter
* update index
* remove previous next buttons
* update sphinx configuration
* add compendium and extended contributing guide ([#17](https://github.com/floodlight-sports/floodlight/issues/17))
* add pitch documentation
* add minimal working docs
#### Others

* release 0.2.0
* bump version
* complete project config
* update dependencies
* add readthedocs config
* add to __all__
* remove old workflow
* update workflows ([#34](https://github.com/floodlight-sports/floodlight/issues/34))
* update dependencies
* add sphinx dependencies for docs
* add pandas
* integrate poetry
* add pre-commit hooks
* add github actions workflow
* add changelog generator ([#19](https://github.com/floodlight-sports/floodlight/issues/19))
* add documentation template ([#22](https://github.com/floodlight-sports/floodlight/issues/22))
* add bug template ([#21](https://github.com/floodlight-sports/floodlight/issues/21))
* add feature request template ([#20](https://github.com/floodlight-sports/floodlight/issues/20))
* linting
* add events tests ([#26](https://github.com/floodlight-sports/floodlight/issues/26))
* unit test skeleton and first example ([#10](https://github.com/floodlight-sports/floodlight/issues/10))
* add testing with pytest

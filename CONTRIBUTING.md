Contributing to floodlight
==========================

Thank you for considering to help with this package! We warmly welcome all kinds of contributions that fit our scope, irrespective of size.

Please also feel encouraged to open an issue (or PR) if you found a bug, have general feedback or if you are unsure whether your new feature would be a good addition.

If all or some of this is new to you, and you would prefer a detailed step-by-step explanation of how to contribute, check out our extended contributing manual in the [documentation](https://floodlight.readthedocs.io).


Development
-----------

We use *poetry* for development and dependency management, which is based on the static `pyproject.toml` file and replaces `setup.py`. With *poetry*, setting up your local dev environment is rather straightforward:

1. Fork and Clone this repository
2. Install poetry if you have not done so yet
3. Use poetry to create a virtualenv with all necessary dependencies in the correct versions. To do so, simply run:

```
poetry install
```

Once you've finished your contribution, please send a Pull Request to merge either into `develop` (if you have an independent contribution) or a related feature branch. We would like to kindly encourage you to include appropriate unit tests and documentation into your contribution.


Standards
---------

We follow a few standards to ensure code quality:

- [PEP8](https://www.python.org/dev/peps/pep-0008/) is a must, with only very few exceptions:
  - Line length can be up to 88 characters
  - Variable names can include uppercase literals for IDs
- Please also keep the [Zen of Python](https://www.python.org/dev/peps/pep-0020/) in mind regarding code style.
- We make extensive use of [typing](https://docs.python.org/3/library/typing.html), and we encourage you to include type hints as much as possible.
- [Docstrings](https://www.python.org/dev/peps/pep-0257/) are formatted in [numpy-style](https://numpydoc.readthedocs.io/en/latest/format.html) (as in this [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)) to enable smooth creation of the API reference.
- Commit messages should follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)


Continuous Integration
----------------------

We use GitHub Actions for continuous integration, and every PR is checked with *flake8*, *black* and the entire test suite.

To check locally if your PR will successfully pass all workflows, we set up a number of pre-commit hooks. To install them, run

```
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg
```

After installation, all hooks will be automatically called for each commit. You may also call the hooks without committing by running

```
poetry run pre-commit run --all-files
```


Testing
-------

All tests and necessary mock data is stored in the `docs/` folder. We use *pytest* for testing, and you can run the entire test suite with

```
poetry run pytest
```

Note that tests are not included as a pre-commit hook to avoid long wait times. You should run them separately if you want to ensure that your contribution does not break any tests.

We make use of fixtures for mock data, and also marks to group tests according to their purpose. For example, to run only unit tests (and avoid time-expensive integration tests), you can execute

```
poetry run pytest -m unit
```


Docs
----

All documentation is stored in the `docs/` folder and are based on the *sphinx* package. There is a dedicated `README.md` in this folder with instructions on how to build the docs.

Vision
------

The list of possible features for this package is endless, so we aim to keep things together. In the beginning, our focus really is on:

* parsing functionality
* observation-level core data structures
* core object methods and handling
* design and interface optimization

The next step (in the future) would be to include a set of plotting methods, and analyses from published articles.

Please also make sure you've checked out design principles from the documentation to understand our perspective on this package.

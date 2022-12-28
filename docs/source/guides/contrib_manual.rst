============================
How To Contribute (Extended)
============================

Do you consider contributing to our project? That's great! We welcome all kinds of contributions - whether you discovered a bug, want to support the project by helping other users, code new features or just give some general feedback.

.. TIP::

   If you have some experience contributing to open source projects, check out our Contributing file on GitHub to get started. This guide is intended for users that would like to contribute to our project but have less experience or would prefer a more in-depth explanation of all the steps necessary to do so.

We're firm believers of open source and want to create an inclusive environment around our project where everybody is welcome to join! In this spirit, you don't need a lot of experience in Python to contribute, and we try hard to provide all necessary information to anybody not familiar with all this DevOps stuff. This guide is an extended manual that aims to cover all topics needed to get you started developing in our project! The covered topics include

* Installing the repository in *development-mode*
* Setting up your dev-environment
* Conventions and best-practices we use
* Necessary steps for a successful pull request (PR)
* Some explanations for why this is so much more complicated than scripting.


Preliminaries
=============

There are a few pre-requisites for using this guide:

* An account on `GitHub <https://github.com/>`_ where we host all the source code.
* Both `python 3.8/3.9/3.10` and `git` installed on your machine. If that's not yet the case, install python from the `official page <https://www.python.org/downloads/>`_ (or check out `pyenv <https://github.com/pyenv/pyenv>`_) and `git <https://git-scm.com/>`_.

.. NOTE::

    This guide was designed and tested on Windows. If you use a different OS, the basics still apply, although the specific steps during setup might differ. We try to include sources for all these steps so that you can check for yourself if there are OS-dependent changes on how to proceed.


Developing
==========

Let's start and install our repository in *dev*-mode for developing. This installation differs from *production*-mode. The latter describes the ready-to-use version of the package as you would install it e.g. from PyPI. *Dev*-mode, on the contrary, refers to the direct copy of the repository as you would find it on GitHub, including tools used for developing, testing, quality assurance and all (public) branches where new features are developed.

This mode has more dependencies which change regularly as we develop the next release, i.e., a snapshot of a publishable version of the code. Dependency management (and packaging) used to be inconvenient in Python, yet it is important that every contributor works on the same environment when collaborating on code. Luckily, there's a tool called *poetry* which simplifies this a lot. *poetry* works with the static ``pyproject.toml`` file containing project metadata, and replaces the usage of ``setup.py`` files. Thus, the first step is to install poetry!


Poetry requires a system-wide installation that's different on Windows and MacOS. The full installation instructions can be found on `the official page <https://python-poetry.org/docs/master/#installation>`_. The quick four-step-version goes:

1. Open Windows PowerShell
2. Copy, paste and execute

.. code-block:: shell

    (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py -UseBasicParsing).Content | python -


3. The installer will tell you the path of the installed executable. Add the path to your system ``PATH`` if that's not done automatically.

4. Re-open PowerShell and execute

.. code-block:: shell

    poetry --version

to check if the installation was successful

With poetry and git, getting the right environment is now rather easy. First, you need to get a copy of the original repository. To do so, follow these steps:

1. Go to the `repository page <https://github.com/floodlight-sports/floodlight>`_
2. Hit the **Fork** Button on the top right of the page. This will create a personal blueprint in your GitHub account. Compared to the base repository, you have the permission to manage this repository in whatever way you like
3. Clone the repository to your local machine as usual.

You've now got your own "version" of the original repository on your machine. The last step is to install all the necessary dependencies. Go the repo's directory and just run

.. code-block:: shell

    poetry install

(e.g. in git bash)! Poetry will then create a ``virtualenv`` and install all necessary dependencies. This is basically everything you need to start contributing. However, we follow a number of standards to ensure code quality. Make sure you know and follow these conventions so that your code fits nicely into the existing codebase!


Standards
=========

1. Codestyle

    * `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ and the `Zen of Python <https://www.python.org/dev/peps/pep-0008/>`_
    * `Typing <https://docs.python.org/3/library/typing.html>`_
    * `Docstrings <https://www.python.org/dev/peps/pep-0257/>`_ in `numpy-style <https://numpydoc.readthedocs.io/en/latest/format.html>`_ (as in this `example <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_)

2. DevOps

    * Structured commit messages with `Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/>`_
    * The `git-flow <https://nvie.com/posts/a-successful-git-branching-model/>`_ branching model
    * Semantic Versioning `SemVer <https://semver.org/>`_ for versioning


Workflows
=========

Local Workflows
---------------

Most of these conventions are enforced through the contributing workflow
(fork - clone - edit - pull request) as well as automatically with GitHub Actions used for
continuous integration purposes. However, you may want to ensure a local dev environment that
actively facilitates these conventions. There are a number of tools you can use to do so:

Pre-Commit Hooks
~~~~~~~~~~~~~~~~

You can install pre-defined pre-commit hooks by running:

.. code-block:: shell

    poetry run pre-commit install
    poetry run pre-commit install --hook-type commit-msg

These hooks will automatically get activated whenever you commit any code, and check for code style
(via black and flake8) as well as commit message structure. You can also activate each of these tools
manually by running the following commands (see the respective docs for full intros):

.. NOTE::

    You need to start every command with `poetry run` if executables are not in your `PATH`.

* re-format all code with black: `black`.
* run linter: `flake8`.
* dummy check all pre-commit hooks: `pre-commit run --all-files`.
* update hooks: `pre-commit autoupdate`.
* check a commit message: `cz check -m "my commit message"`.

IDEs
~~~~

Additionally, if you use an IDE like PyCharm, you can set up your favorite tool to help you right
during coding. For example:

1. Add new Interpreter and point to python.exe in poetry-created env
2. `Integrate black <https://black.readthedocs.io/en/stable/integrations/editors.html>`_ (you could do the same with flake8)
3. Configure Inspections -> PEP8 checking
4. Setting > Tools > Python Integrated Tools: Set default tester and docstring format


Global Workflows
----------------

Once you have made your fork and clone of the original repository, there are three copies that are of interest:

* the original repository, hereafter called `base` or `upstream`
* your fork that's stored on GitHub (`origin`)
* the local clone on your machine (`local`)

Up to this point, you're set up so that you can develop on `local`. The remaining question is: once you've done some work and coded that cool new feature, how do you get your changes into `base`? The standard way for contributing to an open source repository without having direct write access is to develop locally, then merge globally. In a nutshell, you want to keep your `local` up to date with `base`, develop a new feature on `local`, and request to merge it into `base` once you're finished. The long story goes like this:

Remember that we follow a (slim-fit) version of the git-flow model, which gives the `main` and `develop` branch a special role. These are reserved for stable snapshots of the code (`main`) as well as (potentially unstable) checkpoints during development of a new version (`develop`). There's two implications here:

* You want to keep your local copies of these two branches up to date with the original ones to avoid merge conflicts due to missed updates
* You shouldn't work on these branches directly but use feature- or hotfix-branches for your work that branch from and merge into `develop`

If you add your own feature branch, there's now three repositories and three branches flying around. This might be puzzling at first sight, maybe take a moment and try to sort these out. On second sight, however, the GitHub-workflow and git-flow model are great teamplayers. There's a one-way road opening up that goes like this:

``base:main`` /``base:develop`` > updates > ``local:main``/``local:develop`` > branches > ``local:my_feat_branch``

That's pretty much half of the cycle that starts at `base` and ends at your local feature branch. The other half goes in a different direction as you're lacking write access to push your changes up the road where the original code came from. Here, you need to take a little detour over `origin` - your GitHub copy of `base`:

``local:my_feat_branch`` > pushes > ``origin:my_feat_branch`` > merge > ``base:develop``

Again, you would need write access to `base` to perform the merge in the last step by yourself. Instead, the final step of contributing your code is handled by GitHubs **Pull Request (PR)**. Essentally, you use GitHub to explain/present your work, show that it passes all the workflows triggered by GitHub Actions and ask the maintainer to merge your changes.

So much of the theory, let's see how one can perform all these steps in practice:

1. It's important to keep your `local` up to date with `base`, so that your contribution integrates smoothly with the current version instead of relying on code that's a few commits behind. To this end, you may add `base` as an additional remote location so that from now on you can pull new commits directly from there:

    .. code-block:: bash

        git remote add upstream https://github.com/floodlight-sports/floodlight
        git fetch upstream

2. As you never push to `origin:develop` or `origin:main` anyways, you can let them track `base:develop` and `base:main` instead. For `develop` that's done by:

    .. code-block:: bash

       git checkout develop
       git branch -u upstream/develop

Same goes for `main`.

3. Don't use `main` or `develop` for your development directly, rather keep them in sync with the equivalent branches in `base` by hitting

    .. code-block:: bash

        git pull

on the respective branch.

4. For your new feature, create a new branch from the latest version of the code:

    .. code-block:: bash

       git checkout develop
       git checkout -b my_feat_branch

5. Code and commit on this branch as you would normally do.

6. Once you're finished, make sure you haven't missed any updates on `base` while you were coding:

    .. code-block:: bash

       git checkout develop
       git pull
       git checkout my_feat_branch
       git rebase develop

7. Push the changes to your GitHub fork:

    .. code-block:: bash

       git push -u origin

8. Go to the `repository page <https://github.com/floodlight-sports/floodlight>`_ and do a PR. Make sure you ask to merge your changes from `origin:my_feat_branch` into `base:develop`.


Testing
=======

Next, let's talk about testing. This project's is a big fan of test-driven development and maintains an extensive test suite. If you want to contribute a new feature, thorough tests are expected to be included in your addition. This section discusses everything you need to know to write good tests!

Why testing code?
-----------------

* The programmer has to focus on the requirements before writing code.
* Ensures and improves the quality of your code (number of bugs will be reduced).
* Can be viewed as a sort of code documentation.
* Notice whether changes in one place might break the code in another place.

.. _General rules:

Rules of Thumb
--------------

.. TIP::

    * Test files follow a certain naming convention: ``test_<module_Name>.py``
    * Test methods follow the same convention:

    .. code-block:: python

        def test_method_name():
            # some testing code

    * Tests should be **easy to understand**.
    * Tests should only test a **tiny bit of functionality**.
    * Tests should run alone and **independent**.
    * Tests should **run fast**.
    * Tests should be **run frequently** (at least before and after every coding session).
    * The test methods should have long and **descriptive names**.
    * Every unit test should follow the **Arrange-Act-Assert model** (see below).

Tests types
-----------

Generally tests can be structured based on the complexity of code that they are testing.

**Unit tests** make sure that on the lowest layer classes and functions behave as they should.

**Integration tests** combine multiple modules, classes or methods to test if they are all working together.

**System tests** operate on the highest layer and test whether completely integrated systems fulfill the specified requirements.

Testing layout
--------------

To ensure that the structure of the testing suite remains clear the tests are stored in a separate ``/test`` folder. The structure below this folder is then simply a mirror image of the actual folder structure with the difference that the various modules have a ``test_*.py`` in front of their normal file name. Here is a shortened example of the described structure::


    floodlight/
        core/
            events.py
            pitch.py
            xy.py
        utils/
    tests/
        test_core/
            test_events.py
            test_pitch.py
            test_xy.py
        test_utils/

Arrange-Act-Assert model
------------------------

Every unit test should follow the Arrange-Act-Assert model.
    #. Arrange (set up) the input or conditions for the test
    #. Act by calling a method
    #. Assert whether some end condition is true

To clarify this structure here is a very simple example:

.. code-block:: python

    # function to test
    def square(number):
	    return number*number

    # test function
    def test_square_zero()
	    #Arrange
	    number = 0

	    #Act
	    result = square(number)

	    #Assert
	    assert result == 0, "assert message that will be shown if the assert statement is false"


The Pytest Framework
--------------------
The pytest framework provides a feature-rich, plugin-based ecosystem that helps to easily write small as well as readable tests and it can also scale to support complex functional testing. To make sure that you can use the full functionality of pytest this section provides you some conventions and commands that are useful. If you want to get more into the whole framework you can find further information `here <https://docs.pytest.org/en/6.2.x/contents.html#toc>`__.
As described in the :ref:`general rules <General rules>` pytest follows a strict naming convention for files (``test_*.py``) and methods (``def test_*()``).

.. _How to execute pytest:

How to execute pytest
---------------------
As part of the continuous integration pipeline build into the floodlight repository all the tests are going to be executed when making the pull request. Irrespective of this, tests should be carried out internally on a regular basis.
In order to test files, classes or methods in the current directory and subdirectories there are some helpful `commands <https://docs.pytest.org/en/6.2.x/usage.html#calling-pytest-through-python-m-pytest>`_ to execute from the terminal:

.. code-block:: shell

    $ pytest # to run all tests

.. code-block:: shell

    $ pytest <directory>/ # to run all tests in the <directory> directory

.. code-block:: shell

    $ pytest <filename>.py # to run tests in the <filename> file

.. code-block:: shell

    $ pytest -m <name> # to run all tests with the @pytest.mark.<name> decorator (see below)
    $ pytest -m "not <name>" # to run all tests which do not have the @pytest.mark.<name> decorator (see below)

.. code-block:: shell

    $ pytest -k "<string1> and not <string2>" # to run all tests which contain the <string1> and not the <string2> expression

.. code-block:: shell

    $ pytest <filename>.py::<methode_name> # to run a specific test (<method_name>) within a module (<filename>)

In order to understand the test report provided by pytest in detail this `link <https://docs.pytest.org/en/latest/how-to/output.html>`__ is recommended.

Fixtures
--------

Most of the tests depend on some sort of input. With `fixtures <https://docs.pytest.org/en/6.2.x/fixture.html>`_ pytest provides a feature with which data, test doubles or some system state can be created. Fixtures are reusable and can be used for multiple tests. In order to create a fixture you have to build a function that returns the data or system state that is needed for your testing. To do that just decorate this function with ``@pytest.fixture``. The function name can now get passed to a testing method as an argument. As the number of fixtures increases with the project, it makes sense to put them into a structure to keep track of them. Pytest provides a solution to keep everything structured (:ref:`Where to create fixtures? <Where to create fixtures?>`). You can basically store fixtures in the same files where you use them. However, it is also possible to store them in a separated ``conftest.py`` file on which every testing file in the same layer or in a subdirectory has access without any import. The following example should clarify how fixtures work:

.. code-block:: python

    ''' tests.test_core.conftest '''
    import pytest
    import numpy as np

    # creation of the fixture
    @pytest.fixture()
    def example_xy_data_pos_int() -> np.ndarray:
        positions = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        return positions

    ''' tests.test_core.test_xy '''
    import pytest
    import numpy as np

    from floodlight.core.xy import XY
    # testing a function with the fixture being passed as an argument
    def test_x_pos_int(example_xy_data_pos_int: np.ndarray) -> None:
        # Arrange
        data = XY(example_xy_data_pos_int)

        # Act
        x_position = data.x

        # Assert
        assert np.array_equal(x_position, np.array([[1, 3], [5, 7]]))

Fixtures are a quite powerful tool since they are modular and can also request other fixtures. In a nutshell they can be understood as minimal examples of e.g. data-level objects such as XY, Events, or Code. But compared to the normal objects, they are much clearer and are still able to test the full functionality of the methods. Of course, they look different depending on the method tested.

When to create fixtures?
~~~~~~~~~~~~~~~~~~~~~~~~
In case you are writing multiple tests that all make use of the same underlying test data, then it can be advantageous to create a fixture. Otherwise it is common to arrange the data inside your testing function.

.. _Where to create fixtures?:

Where to create fixtures?
~~~~~~~~~~~~~~~~~~~~~~~~~
With the pytest framework there are different possibilities where the fixtures can be implemented. Creating fixtures in different locations only serves to clarify the test environment, especially when working collaboratively in a team. The following options are common solutions:

    #. Inside the testing files.
    #. Inside a ``conftest.py`` file.
    #. Inside an extra file which is then integrated into the ``conftest.py`` file as a plugin.

The ``conftest.py`` file just follows a naming convention of pytest and enables to share fixtures across multiple files. The fixtures implemented inside the ``conftest.py`` file can be accessed from testing files laying in the same folder layer or in a subdirectory without any import. For more detailed information (especially on option 3.) have a look on this `link <https://docs.pytest.org/en/6.2.x/fixture.html>`_.

Marks
-----

Marks can be used to categorize your tests. To do so you need to decorate the method with ``@pytest.mark.<mark_name>``. When executing the ``pytest -m <mark_name>`` command (see :ref:`how to execute pytest <How to execute pytest>`) only methods decorated with ``@pytest.mark.<mark_name>`` will be selected for the testing. This can be advantageous if you have tests that are slower because they are for example accessing a database but you want to quickly run your test suite.

.. code-block:: python

    @pytest.mark.<mark_name>
    def test_x_pos_int(example_xy_data_pos_int: np.ndarray) -> None:
        # Arrange
        data = XY(example_xy_data_pos_int)

        # Act
        x_position = data.x

        # Assert
        assert np.array_equal(x_position, np.array([[1, 3], [5, 7]]))

Pytest comes with a few marks out of the box which can bee seen `here <https://docs.pytest.org/en/6.2.x/mark.html#>`_. To create your own customized mark you have add the following plugin to the ``pyproject.toml`` file:

.. code-block::

    [tool.pytest.ini_options]
    markers = [
        "<mark_name1>: description",
        "<mark_name2: description"
    ]

Testing workflow
----------------

A helpful testing workflow could look something like this:

    #. Before starting the coding session :ref:`run pytest <How to execute pytest>` in your terminal to see if everything works or you get some errors which have to be fixed.
    #. After or before writing a class or method write the according tests and fixtures to keep your test suite always up to date.
    #. After finishing your coding session :ref:`run pytest <How to execute pytest>` again.
    #. If you have to interrupt your work, write a test that points to were you ended the last time.

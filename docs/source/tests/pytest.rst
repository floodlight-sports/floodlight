============================
About The Pytest Framework
============================
The pytest framework provides a feature-rich, plugin-based ecosystem that helps to easily write small as well as readable tests and it can also scale to support complex functional testing. To make sure that you can use the full functionality of pytest this section provides you some conventions and commands that are useful. If you want to get more into the whole framework you can find further information `here <https://docs.pytest.org/en/6.2.x/contents.html#toc>`__.
As described in the :doc:`previous section </tests/testing>` pytest follows a strict naming convention for files (``test_*.py``) and methods (``def test_*()``).

How to execute pytest
=====================
As part of the `continuous integration <https://www.atlassian.com/continuous-delivery/continuous-integration>`_ pipeline build into the floodlight repository all the tests are going to be executed when making the pull request. Irrespective of this, tests should be carried out internally on a regular basis.
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

# Should I exclude this examples and just reference the part of the documentation of pytest which contains all these commands

In order to understand the test report provided by pytest in detail this `link <https://docs.pytest.org/en/latest/how-to/output.html>`__ is recommended.


Useful features
===============

Pytest provides multiple features that are very useful to simplify the testing procedure and keep your tests organized and structured.

Fixtures
---------
Most of the tests depend on some sort of input. With `fixtures <https://docs.pytest.org/en/6.2.x/fixture.html>`_ pytest provides a feature with which data, test doubles or some system state can be created. Fixtures are reusable and can be used for multiple tests. In order to create a fixture you have to build a function that returns the data or system state that is needed for your testing. To do that just decorate this function with ``@pytest.fixture``. The function name can now get passed to a testing method as an argument. Here you can see an example of how fixtures can be implemented:

.. code-block:: python

    ''' tests.test_core.test_xy '''
    import pytest
    import numpy as np

    from floodlight.core.xy import XY

    # creation of the fixture
    @pytest.fixture()
    def example_xy_data_pos_int() -> np.ndarray:
        positions = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        return positions

    # testing a function with the fixture being passed as an argument
    def test_x_pos_int(example_xy_data_pos_int: np.ndarray) -> None:
        # Arrange
        data = XY(example_xy_data_pos_int)

        # Act
        x_position = data.x

        # Assert
        assert np.array_equal(x_position, np.array([[1, 3], [5, 7]]))

Fixtures are a quite powerful tool since they are modular and can also request other fixtures.

When to create fixtures?
~~~~~~~~~~~~~~~~~~~~~~~~
In case you are writing multiple tests that all make use of the same underlying test data, then it can be advantageous to create a fixture. Otherwise it is common to arrange the data inside your testing function.

Where to create fixtures?
~~~~~~~~~~~~~~~~~~~~~~~~~
With the pytest framework there are different possibilities where the fixtures can be implemented. Creating fixtures in different locations only serves to clarify the test environment, especially when working collaboratively in a team. The following options are common solutions:

    #. Inside the testing files.
    #. Inside a ``conftest.py`` file.
    #. Inside an extra file which is then integrated into the ``conftest.py`` file as a plugin.

The ``conftest.py`` file just follows a naming convention of pytest and enables to share fixtures across multiple files. The fixtures implemented inside the ``conftest.py`` file can be accessed from testing files laying in the same folder layer or in a subdirectory without any import. For more detailed information (especially on option 3.) have a look on this `link <https://docs.pytest.org/en/6.2.x/fixture.html>`_.

Marks
------
Marks can be used to categorize your tests. To do so you need to decorate the method with ``@pytest.mark.<mark_name>``. When executing the ``pytest -m <mark_name>`` command (:doc:`see here </tests/testing>`) only methods decorated with ``@pytest.mark.<mark_name>`` will be selected for the testing. This can be advantageous if you have tests that are slower because they are for example accessing a database but you want to quickly run your test suite.

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
        "<mark_name12: description"
    ]

Testing workflow
================
    #. Before starting the coding session :doc:`run pytest </tests/testing>` in your terminal to see if everything works or you get some errors which have to be fixed.
    #. After or before writing a class or method write the according tests and fixtures to keep your test suite always up to date.
    #. After finishing your coding session :doc:`run pytest </tests/testing>` again.
    #. If you have not finished your task write a test that points to were you ended the last time.

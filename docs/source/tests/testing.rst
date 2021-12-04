============================
About Testing In General
============================


Why testing code?
=================

* The programmer has to focus on the requirements before writing code.
* Ensures and improves the quality of your code (number of bugs will be reduced).
* Can be viewed as a sort of code documentation.
* Notice whether changes in one place might break the code in another place.

General rules
=============

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
* After or before writing a class or method write the according tests (keep your test suite always **up to date**).
* You should write broken tests when you have to interrupt your work. When coming back you will have a pointer to where you have finished the last time.
* The test methods should have long and **descriptive names**.
* Every unit test should follow the Arrange-Act-Assert model (see below).

Tests types
===========
Generally tests can be structured based on the complexity of code that they are testing.

Unit test
---------
Unit tests make sure that on the lowest layer classes and functions behave as they should.

Integration test
----------------
Integration tests combine multiple modules, classes or methods to test if they are all working together.

System test
-----------
System tests operate on the highest layer and test whether completely integrated systems fulfill the specified requirements.

Testing layout
==============
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
========================

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

from functools import wraps

import matplotlib.pyplot as plt


def check_axes_given(func):
    """Decorator function that checks if a matplotlib.axes is given as an argument.
    Creates one if not.

    Parameters
    ----------
    func:
        Function object that needs a matplotlib.axes as an argument. If ax == None
        an axes is created an passed to the given function object as a keyworded
        argument.

    Returns
    -------
    func:
        Function with matplotlib.axes as additional argument if not specified.
        Otherwise the function is returned as it is.
    """

    @wraps(func)
    def add_ax(*args, **kwargs):  # actual wrapper function that gets args and kwargs
        # from the funtion that was passed
        # If matplotlib.axes is not given (ax == None) an axes is created.
        if not kwargs.get("ax"):
            kwargs.pop("ax")  # Remove ax from kwargs
            ax = plt.subplots()[1]  # Create matplotlib.axes
            return func(*args, ax=ax, **kwargs)  # return function with axes

        # If matplotlib.axes is given nothing changes and the function is returned with
        # the given *args and **kwargs
        return func(*args, **kwargs)

    return add_ax

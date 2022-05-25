import warnings
from functools import wraps

from floodlight import Pitch


class BaseModel:
    """Base class for all models.

    Parameters
    ----------
    pitch: Pitch, optional
        Some models require pitch information, so the corresponding Pitch object is
        handled during initialization.
    """

    def __init__(self, pitch: Pitch = None):

        if pitch is not None:
            self.check_pitch(pitch)
            self._pitch = pitch

    def __str__(self):
        return f"Floodlight {self.__class__.__name__}"

    @property
    def is_fitted(self) -> bool:
        """Returns ``True`` if all model parameters (those with a trailing underscore)
        are fitted (i.e. not None), and ``False`` otherwise."""
        fitted = all(
            [
                vars(self)[v] is not None
                for v in vars(self)
                if (v.endswith("_") and not v.startswith("__"))
            ]
        )

        return fitted

    @staticmethod
    def check_pitch(pitch: Pitch):
        """
        Performs a series of checks on a Pitch object and raises warnings if the pitch
        configuration may affect computation results.

        Parameters
        ----------
        pitch: Pitch
            Pitch object the checks are performed on.
        """
        # check if metrical system
        if not pitch.is_metrical:
            warnings.warn(
                "Model initialized with non-metrical pitch. Results may be distorted, "
                "use at your own risk.",
                category=RuntimeWarning,
            )


def requires_fit(func):
    """Decorator function for Model-based class-methods that require a previous call to
    that model's fit()-method. Raises a ValueError if fit() has not been called yet."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        model = args[0]
        if not model.is_fitted:
            raise ValueError(
                f"Not all model parameters have been calculated yet. Try "
                f"running {model.__class__.__name__}.fit() before calling "
                f"this method"
            )
        return func(*args, **kwargs)

    return wrapper

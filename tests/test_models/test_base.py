import warnings

import pytest

from floodlight import Pitch
from floodlight.models.base import BaseModel, requires_fit


# Test BaseModel dunder-methods
@pytest.mark.unit
def test_base_model_dunders() -> None:
    # Arrange
    pitch = Pitch((0, 40), (0, 20), unit="m", boundaries="fixed")
    model = BaseModel(pitch)

    # Assert
    assert hasattr(model, "_pitch")
    assert model.__str__() == "Floodlight BaseModel"


# Test is_fitted @property
@pytest.mark.unit
def test_is_fitted_property() -> None:
    # Arrange
    model1 = BaseModel()
    model2 = BaseModel()
    model3 = BaseModel()

    # Act
    # Model 1 is fitted
    model1._model_parameter1_ = 0
    model1.model_parameter2_ = 1
    model1.__not_a_model_parameter__ = 2
    model1._not_a_model_parameter = 3
    model1.not_a_model_parameter = None
    # Model 2 is not fully fitted
    model2._model_parameter1_ = None
    model2.model_parameter2_ = 0
    model2.__not_a_model_parameter__ = 1
    model2._not_a_model_parameter = 2
    model2.not_a_model_parameter = None
    # Model 3 has no model parameters
    model3.__not_a_model_parameter__ = 1
    model3._not_a_model_parameter = 2
    model3.not_a_model_parameter = None

    # Assert
    assert model1.is_fitted
    assert not model2.is_fitted
    assert model3.is_fitted


# Test check_pitch @staticmethod
@pytest.mark.unit
def test_check_pitch_staticmethod() -> None:
    # Arrange
    pitch1 = Pitch((0, 40), (0, 20), unit="m", boundaries="fixed")
    pitch2 = Pitch((0, 40), (0, 20), unit="percent", boundaries="fixed")

    # Assert
    # raise NO warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BaseModel.check_pitch(pitch1)
    # raise warning
    with pytest.warns(RuntimeWarning):
        BaseModel.check_pitch(pitch2)


# Test requires_fit decorator
@pytest.mark.unit
def test_requires_fit_decorator_function() -> None:
    # Arrange
    class MockModel(BaseModel):
        def __init__(self):
            super().__init__()
            self._fitted_parameter_ = None

        def fit(self, arg):
            self._fitted_parameter_ = arg

        @requires_fit
        def get_fitted_parameter(self):
            return self._fitted_parameter_

    # Act
    model1 = MockModel()
    model2 = MockModel()
    arg = 1
    model2.fit(arg)

    # Assert
    with pytest.raises(ValueError):
        model1.get_fitted_parameter()
    assert model2.get_fitted_parameter() == arg

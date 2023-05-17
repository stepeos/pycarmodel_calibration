""" module to handle models"""

from abc import ABC, abstractmethod
from typing import Type

from calibration_tool.fileaccess.parameter import Parameters, ModelEnum

class TrafficFollowingModel(ABC):
    """class to handle traffic following model"""

    def __init__(self, model_parameters: Type[Parameters]):
        self._model_type = None
        self._model_parameters = model_parameters
        self._params = self._model_parameters.get_parameters()

    @staticmethod
    def get_model_from_parameters(params: Type[Parameters]):
        """create model instance from Parameters set"""
        model_type = params.get_model_type()
        classname = model_type.name
        classname = classname[0].upper() + classname[1:].lower()
        class_ = globals()[classname]
        model = class_(params)
        return model

    @abstractmethod
    def initialize(self, data_generator):
        """initialize the model, required for e.g. training purposes"""
        return None

    @abstractmethod
    def step(self, current_input):
        """simulation time step"""
        return None

    def get_model_type(self) -> ModelEnum:
        """method to get the model type"""
        return self._model_type

class Idm(TrafficFollowingModel):
    """idm discrete implementation"""

    def __init__(self, model_params):
        super().__init__(model_params)
        self._model_type = ModelEnum.IDM

    def step(self, current_input):
        pass

    def initialize(self, data_generator):
        return

class Iidm(TrafficFollowingModel):
    """iidm discrete implementation"""

    def __init__(self, model_params):
        super().__init__(model_params)
        self._model_type = ModelEnum.IIDM

    def step(self, current_input):
        pass

    def initialize(self, data_generator):
        return

class Hdm(TrafficFollowingModel):
    """Hdm discrete implementation"""

    def __init__(self, model_params):
        super().__init__(model_params)
        self._model_type = ModelEnum.HDM

    def step(self, current_input):
        pass

    def initialize(self, data_generator):
        return

class Eidm(TrafficFollowingModel):
    """eidm discrete implementation"""

    def __init__(self, model_params):
        super().__init__(model_params)
        self._model_type = ModelEnum.EIDM

    def step(self, current_input):
        pass

    def initialize(self, data_generator):
        return

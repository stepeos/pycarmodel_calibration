"""module to handle config files like parameter file"""

from enum import Enum
import numpy as np
from scipy.optimize import LinearConstraint

from calibration_tool.fileaccess.configs import JSON


class ModelEnum(Enum):
    """type of parameterset"""
    IDM = 0
    IIDM = 1
    HDM = 2
    EIDM = 3

class Parameters(JSON):
    """class to handle paramter config set"""
    def __init__(self, filename, model: ModelEnum):
        super().__init__(filename)
        self._model = model
        self.set_value(["carFollowModel"], self._model.name)

    def get_model_type(self):
        """returns the model type"""
        return self._model

    def set_parameters(self, parameter_set):
        """
        parameters_set must be {"accel": value,...}
        """
        for name, value in parameter_set.items():
            self.set_value([name], value)

    def get_parameters(self):
        """get paremeters as list of tuples"""
        params = ()
        for key, value in self._values.items():
            if key == "carFollowModel":
                continue
            params += ((key, value),)
        return params

    def items(self):
        """like dict.items()"""
        for key, value in self._values.items():
            yield key, value

    @staticmethod
    def get_defaults_dict(min_gap: float = None,
                          taccmax: float = None,
                          m_beg: float = None,
                          m_flat: float = None,
                          speed_factor: float = None,
                          startup_delay: float = None) -> dict:
        """get default paramter set"""
        assert (taccmax is None or isinstance(taccmax, float))
        assert (min_gap is None or isinstance(min_gap, float))
        assert (m_beg is None or isinstance(m_beg, float))
        assert (m_flat is None or isinstance(m_flat, float))
        assert (speed_factor is None or isinstance(speed_factor, float))
        assert (startup_delay is None or isinstance(startup_delay, float))
        params = {
            "speedFactor": speed_factor or 1.2,
            "minGap": min_gap or 0.5,
            "accel": 2.6,
            "decel": 4.5,
            "emergencyDecel": 15,
            "startupDelay": startup_delay or 0.01,
            "tau": 0.24,
            "delta": 4,
            "stepping": 0.25,
            "tpreview": 4,
            "tPersDrive": 3,
            "tPersEstimate": 10,
            "treaction": 0.5,
            "ccoolness": 0.99,
            "sigmaleader": 0,
            "sigmagap": 0,
            "sigmaerror": 0,
            "jerkmax": 3,
            "epsilonacc": 1,
            "taccmax": taccmax or 1.2,
            "Mflatness": m_flat or 2,
            "Mbegin": m_beg or 0.7
        }
        for key, value in params.items():
            bounds = Parameters.get_bounds_from_keys([key], 0.04)
            if len(bounds) == 0:
                continue
            value = max(value, bounds[0][0])
            value = min(value, bounds[0][1])
            value = max(value, bounds[0][0] + 0.0001)
            value = min(value, bounds[0][1] - 0.0001)
            params[key] = value
        return params

    @staticmethod
    def load_from_json(filename):
        """create a model parameter set from json file"""
        param_set_file = JSON(filename)
        param_set_file.load_values()
        model_type = param_set_file.get_value("carFollowModel")
        classname = model_type.lower()
        classname = classname[0].upper() + classname[1:] + "Parameters"
        class_ = globals()[classname]
        model = class_(filename, ModelEnum[model_type.upper()])
        param_set = param_set_file.get_values()
        model.set_values(param_set)
        return model

    @classmethod
    def get_bounds_from_keys(cls, keys: list, step_size: float):
        """returns Bounds for a list of keys"""
        assert step_size < 1
        param_bounds = {
            "speedFactor": [0.8, 1.2],
            "minGap": [0.5, 4],
            "accel": [0.2, 4],
            "decel": [0.5, 5],
            "emergencyDecel": [0.5, 20],
            "startupDelay": [0, 2],
            "tau": [0.5, 1.5],
            "delta": [1,5],
            "stepping": [0.0001, 1],
            "tpreview": [1, 10],
            "tPersDrive": [1, 6],
            "tPersEstimate": [1, 20],
            "treaction": [0.2, 0.9],
            "ccoolness": [0, 1],
            "sigmaleader": [0, 1],
            "sigmagap": [0, 1],
            "sigmaerror": [0, 1],
            "jerkmax": [1, 5],
            "epsilonacc": [0.1, 3],
            "taccmax": [0.5, 5],
            "Mflatness": [1, 5],
            "Mbegin": [0.1, 1.5]
            # "maxvehpreview": [0, ],
            # "vehdynamics": [0, ],
        }
        bounds = ()
        for key in keys:
            boundary = param_bounds.get(str(key))
            if boundary is not None:
                bounds += ((boundary[0], boundary[1]),)
        return bounds


class IdmParameters(Parameters):
    """parameter configuration set for the IDM-model"""

    @staticmethod
    # pylint: disable=C0103
    def create_idm_parameter_set(filename, **params):
        """
        :param params:      dict with all params as key value
        """
        new_set = IdmParameters(filename, ModelEnum.IDM)
        params["carFollowModel"] = "IDM"
        new_set.set_parameters(params)
        return new_set


class EidmParameters(Parameters):
    """parameter configuration set for the EIDM-Model"""

    @staticmethod
    # pylint: disable=C0103
    def create_eidm_parameter_set(filename, **params):
        """
        :param params:      {"param1": (value, unit), ...}
        see
        https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#car-following_model_parameters
        """
        params["carFollowModel"] = "EIDM"
        new_set = EidmParameters(filename, ModelEnum.EIDM)
        new_set.set_parameters(params)
        return new_set

    @staticmethod
    def get_constraints():
        """returns constraints used for optimization"""
        keys = ["minGap", "accel", "decel", "emergencyDecel", "tau",
                "delta", "stepping", "tpreview", "tPersDrive",
                "tPersEstimate", "treaction", "ccoolness",
                "jerkmax", "epsilonacc", "taccmax", "Mflatness", "Mbegin"]

        arr = np.zeros(len(Parameters.get_bounds_from_keys(keys, 0.04)))
        arr[2] = 1
        arr[3] = -1
        # decel > emergencedecel
        # of tau > simulation step_size
        cons = LinearConstraint(
            arr,
            -np.inf, 0)
        return cons

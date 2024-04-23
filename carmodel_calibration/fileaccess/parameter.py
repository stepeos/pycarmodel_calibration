"""module to handle config files like parameter file"""

from enum import Enum
import numpy as np
from scipy.optimize import LinearConstraint

from carmodel_calibration.fileaccess.configs import JSON


class ModelEnum(Enum):
    """type of parameterset"""
    IDM = 0
    IIDM = 1
    HDM = 2
    EIDM = 3
    KRAUSS = 4

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
            "speedFactor": speed_factor or 1.0,
            "minGap": min_gap or 0.5,
            "accel": 2.60, # 10.00
            "desAccel1": 3.5, # 4m/s | 3.0, # 5m/s
            "desAccel2": 2.6, # 9m/s | 2.5, # 12m/s
            "desAccel3": 1.6, # 14m/s | 1.5, # 20m/s
            "desAccel4": 1.1, # 22m/s | 0.9, # 30m/s
            "desAccel5": 0.8, # 32m/s | 0.6, # 40m/s
            "desAccel6": 0.5, # 45m/s | 0.4, # 50m/s
            "decel": 2.5, # 2.5 may be better, was previously 4.5
            "emergencyDecel": 15,
            "startupDelay": startup_delay or 0.0,
            "tau": 1.00,
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
            "actionStepLength": 0.0,
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
            "speedFactor": [0.8, 1.3],
            "minGap": [0.5, 4],
            "accel": [0.2, 4],
            "desAccel1": [1.0, 6.0],
            "desAccel2": [0.6, 5.0],
            "desAccel3": [0.4, 4.0],
            "desAccel4": [0.2, 3.0],
            "desAccel5": [0.1, 2.5],
            "desAccel6": [0.1, 2.0],
            "decel": [0.5, 5],
            "emergencyDecel": [0.5, 20],
            "startupDelay": [0, 2],
            "tau": [0.5, 1.5], # max 2.5 may be better
            "delta": [1,5], # max 6 may be better
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
            "actionStepLength": [0.0, 1.0],
            "taccmax": [0.5, 3],
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


class ModelParameters(Parameters):
    """parameter configuration set for the CF-Model"""

    @staticmethod
    # pylint: disable=C0103
    def create_parameter_set(filename, model, **params):
        """
        :param params:      {"param1": (value, unit), ...}
        see
        https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#car-following_model_parameters
        """
        
        if model == "idm":
            params["carFollowModel"] = "IDM"
            new_set = ModelParameters(filename, ModelEnum.IDM)
        elif model == "eidm":
            params["carFollowModel"] = "EIDM"
            new_set = ModelParameters(filename, ModelEnum.EIDM)
        elif model == "krauss":
            params["carFollowModel"] = "Krauss"
            new_set = ModelParameters(filename, ModelEnum.KRAUSS)
        else:
            raise RuntimeError("Model " + model + " not known. Aborting process")
        
        new_set.set_parameters(params)
        return new_set

    @staticmethod
    def get_constraints():
        """returns constraints used for optimization"""
        keys = ["minGap", "accel", "desAccel1", "desAccel2", "desAccel3", "desAccel4",
                "desAccel5", "desAccel6", "decel", "emergencyDecel", "tau",
                "delta", "stepping", "tpreview", "tPersDrive",
                "tPersEstimate", "treaction", "ccoolness", "jerkmax",
                "epsilonacc", "actionStepLength", "taccmax", "Mflatness", "Mbegin"]

        arr = np.zeros(len(Parameters.get_bounds_from_keys(keys, 0.04)))
        arr[2] = 1
        arr[3] = -1
        # decel > emergencedecel
        # of tau > simulation step_size
        cons = LinearConstraint(
            arr,
            -np.inf, 0)
        return cons

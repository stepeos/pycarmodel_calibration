"""module to test fileaccess stuff"""

from pathlib import Path
from unittest import TestCase
import json

from calibration_tool.fileaccess.parameter import (Parameters, IdmParameters, 
    ModelEnum)
from tests.testhelpers import flatten, chdir

test_params = {
    "carFollowModel": "idm",
    "param_1": {
        "value": 1000,
        "unit": "s"
    },
    "param_2": {
        "value": 10,
        "unit": "m/s"
    }
}
TMP_WORK_DIR = str(Path("tests/resources/fileaccess").absolute())

class TestParams(TestCase):
    """test the JSON class"""

    @chdir(TMP_WORK_DIR)
    def setUp(self):
        self.test_file = Path("IDM.log")
        with open(str(self.test_file), 'w', encoding="utf-8") as file:
            json.dump(test_params, file)
        self.test_instance1 = IdmParameters("IDM.log", ModelEnum.IDM)
        self.test_instance2 = IdmParameters("EIDM.log", ModelEnum.IDM)

    @chdir(TMP_WORK_DIR)
    def tearDown(self) -> None:
        if self.test_file.exists():
            self.test_file.unlink()
        file = Path("EDIM.log")
        if file.exists():
            file.unlink()

    @chdir(TMP_WORK_DIR)
    def test_set_parameters(self):
        """method to test setting the modelparametrs"""
        self.test_instance2.set_parameters(test_params)
        result = self.test_instance2.get_parameters()
        mock_test_params = test_params.copy()
        del mock_test_params["carFollowModel"]
        target = ()
        for key, value in mock_test_params.items():
            target += ((key, value,),)
        assert result == target

    @chdir(TMP_WORK_DIR)
    def test_load_from_json(self):
        """method to test creating paramsets from config file"""
        test_instance = Parameters.load_from_json(str(self.test_file))
        assert isinstance(test_instance, IdmParameters)
        assert test_instance.get_model_type() == ModelEnum.IDM

    @chdir(TMP_WORK_DIR)
    def test_check_json_loaded_params(self):
        """ test the parameters for correctness"""
        test_instance = Parameters.load_from_json(str(self.test_file))
        mock_test_params = test_params.copy()
        del mock_test_params["carFollowModel"]
        target = ()
        for key, value in mock_test_params.items():
            target += ((key, value,),)
        assert test_instance.get_parameters() == target

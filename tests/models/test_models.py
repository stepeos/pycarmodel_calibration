"""testclass to test models"""

from unittest import TestCase
import glob
import shutil
import os
from pathlib import Path
from mock import MagicMock

from calibration_tool.models.models import TrafficFollowingModel, Idm
from calibration_tool.fileaccess.parameter import ModelEnum
from tests.testclass_helper import TmpDir


class TestTrafficFollowingModel(TestCase, TmpDir):
    """class to test TraffiFollowingModel"""

    def setUp(self):
        self.prepare_tmpdir()
        self.base_dir = None
        self.data_in = Path("data_in")
        self.data_in.mkdir()

    def tearDown(self):
        self.clean_tmpdir()

    def test_models(self):
        """method to test the models"""
        for model_type in ModelEnum:
            param = MagicMock()
            param.get_model_type.return_value = model_type
            test_instance = TrafficFollowingModel.get_model_from_parameters(param)
            assert test_instance.get_model_type() is not None
            assert model_type == test_instance.get_model_type()

    def test_idm_step(self):
        """method to test the idm model stepping"""
        for model_type in ModelEnum:
            param = MagicMock()
            param.get_model_type.return_value = model_type
            test_instance = TrafficFollowingModel.get_model_from_parameters(param)
            test_instance.step(MagicMock())

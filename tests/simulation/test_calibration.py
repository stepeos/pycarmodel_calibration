"""module to test simulations-handling"""
# pylint: disable=W0212
from unittest import TestCase

from pathlib import Path
import shutil
from scipy.optimize._optimize import OptimizeResult

from mock import MagicMock, patch
import numpy as np

from calibration_tool.control_program.calibration_handling import (
    CalibrationHandler)

def _mock_optimization_factory(result, mode="scipy"):
    def _optimization_mock_scipy(*_, **kwargs):
        callback = kwargs["callback"]
        print("differential_evolution step 1: f(x)= 1")
        callback(np.zeros(100), )
        return result
    def _optimization_mock_pygad(*_, **kwargs):
        callback = kwargs["on_fitness"]
        print("f(x)= 1")
        pygad_instance = MagicMock()
        pygad_instance.best_solutions = np.zeros(100)
        pygad_instance.last_generation_fitness = np.zeros(100) - 1
        pygad_instance.population = np.zeros((100, 100))
        return_mock = MagicMock()
        return_mock.return_value = (np.zeros(100), 1, None)
        pygad_instance.best_solution = return_mock
        callback(pygad_instance, np.zeros(100)+1)
        return pygad_instance
    if mode == "scipy":
        return _optimization_mock_scipy
    elif mode == "pygad":
        return _optimization_mock_pygad

class TestCalibrationHandler(TestCase):
    """
    Test class for calibration handler
    """
    def setUp(self):
        self.test_resources = (Path(__file__).parents[1]
                               / "resources/calibration2")
        self.directory = Path(".tmp/test_directory")
        self.input_data = Path(".tmp/test_input_data")
        if self.input_data.exists():
            shutil.rmtree(self.input_data)
        shutil.copytree(str(self.test_resources), str(self.input_data))
        self.models = ["model1", "model2"]
        self.optimization = "differential_evolution"
        self.max_iter = 2
        self.param_keys = ["param1", "param2"]
        self.num_workers = 50
        self.project_path = Path(".tmp/test_project_path")
        self.tmp_dir = Path(".tmp") / "calibration_test_dir"

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_constructor_with_param_keys(self):
        """test the constructor"""
        cali = CalibrationHandler(self.directory, self.input_data,
                           self.models,
                           optimization=self.optimization,
                           max_iter=self.max_iter,
                           param_keys=self.param_keys,
                           num_workers=self.num_workers,
                           project_path=self.project_path)

    @patch("calibration_tool.simulation.calibration_handling.op")
    @patch("calibration_tool.simulation.calibration_handling.SumoInterface")
    def test_calibration_direct(self, _, scipy_op):
        """running calibration with direct optimization"""
        scipy_op.direct.return_value = OptimizeResult(
            x = np.zeros((100)),
            success = True,
            status = 0,
            message = "test_message",
            fun = 1
        )
        scipy_op._optimize.OptimizeResult = OptimizeResult
        calibration = CalibrationHandler(self.directory,
                                         self.input_data,
                                         self.models,
                                         optimization="direct",
                                         max_iter=self.max_iter,
                                         project_path=self.project_path)
        result = calibration.run_calibration()
        self.assertTrue(len(result)>0)
        self.assertTrue(all(result["weightedError"]==1))

    @patch("calibration_tool.simulation.calibration_handling.op")
    @patch("calibration_tool.simulation.calibration_handling.SumoInterface")
    def test_calibration_differential_evolution(self, _, scipy_op):
        """running calibration with differential evolution optimization"""
        result = OptimizeResult(
            x = np.zeros((100)),
            success = True,
            status = 0,
            message = "test_message",
            fun = 1
        )
        scipy_op.differential_evolution = _mock_optimization_factory(result)
        scipy_op._optimize.OptimizeResult = OptimizeResult
        calibration = CalibrationHandler(self.directory,
                                         self.input_data,
                                         self.models,
                                         optimization="differential_evolution",
                                         max_iter=self.max_iter,
                                         project_path=self.project_path)
        result = calibration.run_calibration()
        self.assertTrue(len(result)>0)
        self.assertTrue(all(result["weightedError"]==1))

    @patch("calibration_tool.simulation.calibration_handling.op")
    @patch("calibration_tool.simulation.calibration_handling.pygad")
    @patch("calibration_tool.simulation.calibration_handling.SumoInterface")
    def test_calibration_genetic_algorithm(self, _, pygad, scipy_op):
        """running calibration with genetic aglorithm optimization"""
        result = OptimizeResult(
            x = np.zeros((100)),
            success = True,
            status = 0,
            message = "test_message",
            fun = 1
        )
        pygad.GA = _mock_optimization_factory(result, "pygad")
        scipy_op._optimize.OptimizeResult = OptimizeResult
        calibration = CalibrationHandler(self.directory,
                                         self.input_data,
                                         self.models,
                                         optimization="genetic_algorithm",
                                         max_iter=self.max_iter,
                                         project_path=self.project_path)
        result = calibration.run_calibration()
        self.assertTrue(len(result)>0)
        self.assertTrue(all(result["weightedError"]==1))

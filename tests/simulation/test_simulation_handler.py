"""module to test simulations-handling"""
# pylint: disable=W0212
from unittest import TestCase

from pathlib import Path
import shutil
from scipy.optimize._optimize import OptimizeResult

from mock import MagicMock, patch
import numpy as np

from carmodel_calibration.control_program.calibration_handling import SimulationHandler
from carmodel_calibration.control_program.simulation_handler import get_weighted_errors
from carmodel_calibration.fileaccess.parameter import Parameters, EidmParameters
from carmodel_calibration.optimization import measure_of_performance_factory


class TestSimulationandler(TestCase):
    """
    Test class for simulation handler
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

    def test_prepare_selection(self):
        model = "eidm"
        sim_handler = SimulationHandler(directory=self.directory, input_data=self.input_data,
                          param_keys=self.param_keys,
                          num_workers=self.num_workers, model=model,
                          project_path=self.project_path)
        sim_handler.prepare_selection(False)
        eidms = [EidmParameters.create_eidm_parameter_set(
            ".tmp/test_set.json",
            **EidmParameters.get_defaults_dict())]
        weighted_errors = []
        for _, selection in sim_handler.selection_data.iterrows():
            identification = (selection["leader"], selection["follower"], selection["recordingId"])
            objective_function = measure_of_performance_factory(
                identification)
            weighted_error = get_weighted_errors(
                param_sets=eidms,
                data_path=self.input_data,
                identification=identification,
                objective_function=objective_function,
                project_path=self.project_path)[0]
            weighted_errors.append(weighted_error)
        self.assertTrue(len(weighted_errors) == 2)
        # self.assertTrue(np.isclose(weighted_errors[0], 0.07563424360969646,
        #                            rtol=0.01))
        self.assertTrue(np.isclose(weighted_errors[1], 0.07563424360969646,
                                   rtol=0.01))

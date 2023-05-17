"""test the optimization file"""
from pathlib import Path

from unittest.mock import patch
import unittest
import shutil
import pandas as pd

from calibration_tool.sumo.simulation_module import SumoInterface
from calibration_tool.sumo.sumo_project import SumoProject
from calibration_tool.optimization import (
    measure_of_performance_factory, TargetCollector)
from calibration_tool.fileaccess.parameter import Parameters

def get_simulation_result():
    """test helper for getting simulation results"""
    sumo_project_path = Path(
            "tests/resources/.tmp/test_sumo_project_path")
    leader_follower_path = Path("tests/resources/calibration2")
    SumoProject.create_sumo(sumo_project_path, 2)
    selection = pd.read_csv(
        leader_follower_path / "01_trajectories_selection.csv",
        index_col=0)
    selection = selection.reset_index()
    identification = tuple(selection.loc[
        0, ["leader", "follower", "recordingId"]])
    sumo_interface = SumoInterface(sumo_project_path, leader_follower_path)
    simulation_results = sumo_interface.run_simulation(identification)
    del sumo_interface
    return simulation_results, identification

class TestOptimizationTargetCollector(unittest.TestCase):
    """tests the target collector of the optimization"""
    def setUp(self):
        self.simulation_result, self.identification = get_simulation_result()
        self.ground_truth = self.simulation_result[1]["ground_truth"]
        self.prediction = self.simulation_result[1]["prediction"]

    def tearDown(self) -> None:
        temp_path = Path("tests/resources/.tmp")
        if temp_path.exists():
            shutil.rmtree(temp_path)

    def test_get_results(self):
        """test getting the simulation results"""
        result = TargetCollector.get_results(self.simulation_result,
                                    self.identification, 0)
        self.assertTrue(len(result) > 0)


class TestOptimizationFactory(unittest.TestCase):
    """test class for optimization"""
    def setUp(self):
        simulation_result, self.identification = get_simulation_result()
        self.ground_truth = simulation_result[1]["ground_truth"]
        self.prediction = simulation_result[1]["prediction"]

    def tearDown(self) -> None:
        temp_path = Path("tests/resources/.tmp")
        if temp_path.exists():
            shutil.rmtree(temp_path)

    def test_measure_of_performance_factory_sop_pair(self):
        """test the factory with sop distance and pair object"""
        objectives = ["distance"]
        weights = None
        gof = "rmse"
        handle = measure_of_performance_factory(self.identification,
                                       objectives=objectives,
                                       weights=weights,
                                       gof=gof)
        weighted_error = handle(self.ground_truth, self.prediction)
        self.assertTrue(weighted_error > 0)

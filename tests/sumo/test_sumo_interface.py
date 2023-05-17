# pylint: disable=W0212
# pylint: disable=C0115
# pylint: disable=C0116

from pathlib import Path

from mock import patch, MagicMock
import unittest
import shutil
import pandas as pd

from calibration_tool.sumo.simulation_module import SumoInterface
from calibration_tool.sumo.sumo_project import SumoProject
from calibration_tool.exceptions import FolderNotFound, MissingConfiguration
from calibration_tool.fileaccess.parameter import Parameters


class TestSumoInterface(unittest.TestCase):
    def setUp(self):
        self.sumo_project_path = (
            Path(__file__).parents[1]
            / "resources/.tmp/test_sumo_project_path")
        shutil.rmtree(self.sumo_project_path, ignore_errors=True)
        self.sumo_project_path.mkdir(parents=True)
        self.leader_follower_path = (
            Path(__file__).parents[1]
            / "resources/calibration2")
        SumoProject.create_sumo(self.sumo_project_path, 2)
        self.sumo_interface = None

    def tearDown(self) -> None:
        if self.sumo_interface:
            del self.sumo_interface
        temp_path = Path(__file__).parents[1] / "resources/.tmp"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_sumo_project_path_not_found(self):
        sumo_project_path = (
            Path(__file__).parents[1]
            / "resources/.tmp/test_sumo_project_path_not_found")
        leader_follower_path = (
            Path(__file__).parents[1]
            / "resources/.tmp/test_leader_follower_path")
        with self.assertRaises(FileNotFoundError):
            self.sumo_interface = SumoInterface(
                sumo_project_path, leader_follower_path)

    def test_sumo_project_path_is_file(self):
        sumo_project_path = (
            Path(__file__).parents[1]
            / "resources/.tmp/test_sumo_project_path.file")
        sumo_project_path.touch()
        leader_follower_path = (
            Path(__file__).parents[1]
            / "resources/.tmp/test_leader_follower_path")
        with self.assertRaises(FileExistsError):
            self.sumo_interface = SumoInterface(
                sumo_project_path, leader_follower_path)

    @patch.object(Path, "exists", return_value = False)
    def test_network_file_not_found(self, _):
        with self.assertRaises(FileNotFoundError):
            self.sumo_interface = SumoInterface(self.sumo_project_path,
                                                self.leader_follower_path)
            self.sumo_interface._get_network_info()

    @patch.object(Path, "glob", return_value=[])
    def test_config_not_found(self, _):
        with self.assertRaises(MissingConfiguration):
            self.sumo_interface = SumoInterface(self.sumo_project_path,
                                                self.leader_follower_path)
            self.sumo_interface.start_simulation_module()

    @patch("calibration_tool.sumo.sumo_interface.traci")
    def test_serialized(self, traci_mock):
        traci_mock = MagicMock()
        self.sumo_interface = SumoInterface(self.sumo_project_path,
                                            self.leader_follower_path)
        self.sumo_interface.serialized()

    def test_run_simulation(self):
        try:
            selection = pd.read_csv(self.leader_follower_path / "01_trajectories_selection.csv",
                index_col=0)
        except FileNotFoundError:
            file_path= str((self.leader_follower_path
                           / "01_trajectories_selection.csv").resolve())
            raise FileNotFoundError(f" NOT FOUND AT THIS {file_path}")
        selection = selection.reset_index()
        identification = tuple(selection.loc[
            0, ["leader", "follower", "recordingId"]])
        self.sumo_interface = SumoInterface(self.sumo_project_path,
                                            self.leader_follower_path)
        self.sumo_interface.run_simulation(identification)

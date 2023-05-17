import sys
import os
from pathlib import Path

from mock import MagicMock, patch
import unittest

from calibration_tool.cmd import main
from calibration_tool.exceptions import MissingRequirements
from calibration_tool.control_program.calibration_handling import (
    CalibrationHandler)

class TestMain(unittest.TestCase):

    def setUp(self) -> None:
        self.resources = Path(__file__).parent / "resources/calibration2"
        self.data = Path(__file__).parent / "resources/.tmp"

    @patch.object(os, "environ")
    def test_sumo_home(self, environ_mock):
        environ_mock = {"SUMO_HOME" : "/path/to/sumo"}
        self.assertRaises(MissingRequirements, main)

    # @patch.object(sys, "exit")
    # def test_valid_sumo_home(self, sys_mock):
    #     os.environ["SUMO_HOME"] = "/usr/share/sumo"
    #     main()
    #     sys_mock.assert_called_once()

    def test_action_create_reports(self):
        sys.argv = ["script_name", "--action=create_reports", "/path/to/data", "/path/to/results"]
        self.assertEqual(main(), None)

    def test_action_calibrate_invalid_path(self):
        sys.argv = ["script_name", "--action=calibrate", "/path/to/data",
                    str(self.resources),
                    "differential_evolution",
                    "--max-iter=1",
                    "--num-workers=4"]
        self.assertRaises(FileNotFoundError, main)

    @patch("calibration_tool.cmd.CalibrationHandler")
    def test_action_calibrate_de(self, handler_mock):
        sys.argv = ["script_name", "--action=calibrate",
                    str(self.resources),
                    str(self.data),
                    "differential_evolution",
                    "--max-iter=1",
                    "--num-workers=20"]
        handler_mock_instance = MagicMock()
        handler_mock.return_value = handler_mock_instance
        main()
        handler_mock_instance.run_calibration_cli.assert_called_once()

    @patch("calibration_tool.cmd.CalibrationHandler")
    def test_action_calibrate_ga(self, handler_mock):
        sys.argv = ["script_name", "--action=calibrate",
                    str(self.resources),
                    str(self.data),
                    "genetic_algorithm",
                    "--max-iter=1",
                    "--num-workers=20"]
        handler_mock_instance = MagicMock()
        handler_mock.return_value = handler_mock_instance
        main()
        handler_mock_instance.run_calibration_cli.assert_called_once()

    @patch.object(sys, "exit")
    @patch("calibration_tool.cmd.CalibrationHandler")
    def test_action_calibrate_direct(self, handler_mock, sys_mock):
        sys.argv = ["script_name", "--action=calibrate",
                    str(self.resources),
                    str(self.data),
                    "direct",
                    "--max-iter=1",
                    "--num-workers=20"]
        handler_mock_instance = MagicMock()
        handler_mock.return_value = handler_mock_instance
        main()
        sys_mock.assert_called_once()
        sys.argv = ["script_name", "--action=calibrate",
                    str(self.resources),
                    str(self.data),
                    "direct",
                    "--max-iter=1"]
        handler_mock_instance = MagicMock()
        handler_mock.return_value = handler_mock_instance
        main()
        handler_mock_instance.run_calibration_cli.assert_called_once()

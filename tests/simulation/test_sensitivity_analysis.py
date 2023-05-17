"""module to test sensitivity analysis"""
# pylint: disable=W0212
from unittest import TestCase

from pathlib import Path
import shutil

from mock import MagicMock, patch
import numpy as np
from multiprocessing.pool import ThreadPool

from calibration_tool.control_program.sensitivity_analysis import (
    SensitivityAnalysisHandler)

def imap_mock(_, __, samples):
    return np.zeros(samples.shape[0]) + 1

class TestSensitivityAnalysisHandler(TestCase):
    """
    Test class for sensitivity analysis handler
    """
    def setUp(self):
        self.test_resources = (Path(__file__).parents[1]
                               / "resources/calibration2")
        self.directory = Path(".tmp/test_directory")
        self.input_data = Path(".tmp/test_input_data")
        if self.input_data.exists():
            shutil.rmtree(self.input_data)
        shutil.copytree(str(self.test_resources), str(self.input_data))
        self.optimization = "differential_evolution"
        self.max_iter = 2
        self.param_keys = ["speedFactor", "minGap", "accel", "tau", "delta",
                           "taccmax", "Mflatness", "Mbegin"]
        self.param_keys = ["param1", "param2"]
        self.num_workers = 25
        self.project_path = Path(".tmp/test_project_path")
        self.tmp_dir = Path(".tmp") / "sa_test_dir"

    @patch.object(ThreadPool, "imap", imap_mock)
    def test_run_analysis(self):
        """test run_analysis"""
        analysis_sample = None
        weights = None
        num_samples = 1500
        model = "eidm"
        gof = "rmse"
        mop = ["distance"]
        sa_handler = SensitivityAnalysisHandler(
            self.directory,
            self.input_data,
            self.num_workers,
            self.project_path,
            param_keys=self.param_keys,
            analysis_sample=analysis_sample,
            weights=weights,
            num_samples=num_samples,
            model=model,
            gof=gof,
            mop=mop,
            seed=2023,
            force_recalculation=False)
        
        sa_handler.run_analysis()
        file_exists = (self.directory
                       / "sensitivity_analysis_fast.pdf").exists()
        self.assertTrue(file_exists)

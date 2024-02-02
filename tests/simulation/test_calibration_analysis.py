import unittest
import matplotlib
import pandas as pd
from pathlib import Path
import shutil
import numpy as np

from carmodel_calibration.sumo.simulation_module import SumoInterface
# from carmodel_calibration.optimization import factory_wrapper
from carmodel_calibration.control_program.calibration_analysis import (
    _simulate_single, _plot_single, create_calibration_analysis,
    _get_unique_calibrations)
from carmodel_calibration.sumo.sumo_project import SumoProject

def create_results():
    results = {"iteration": {0: 1, 1: 2},
               "weightedError":{0: 0.132937, 1: 0.132937},
               "speedFactor": {0: 0.8217808090358688, 1: 1.038954574497346},
               "minGap": {0: 1.353100953174422, 1: 2.708045749528696},
               "accel": {0: 3.0823502318809544, 1: 2.662156645819453},
               "decel": {0: 4.5616312982242, 1: 4.573514718883563},
               "emergencyDecel": {0: 15, 1: 15},
               "startupDelay": {0: 0.0699564512680886, 1: 0.8776108511027431},
               "tau": {0: 0.1832671950098249, 1: 0.2363952269619716},
               "delta": {0: 2.076480982493067, 1: 2.608825692535976},
               "stepping": {0: 0.25, 1: 0.25},
               "tpreview": {0: 4, 1: 4},
               "tPersDrive": {0: 3, 1: 3},
               "tPersEstimate": {0: 10, 1: 10},
               "treaction": {0: 0.7682655992934183, 1: 0.6559243120343751},
               "ccoolness": {0: 0.99, 1: 0.99},
               "sigmaleader": {0: 0.0001, 1: 0.0001},
               "sigmagap": {0: 0.0001, 1: 0.0001},
               "sigmaerror": {0: 0.0001, 1: 0.0001},
               "jerkmax": {0: 3, 1: 3},
               "epsilonacc": {0: 1, 1: 1},
               "taccmax": {0: 4.780869167970448, 1: 2.795878001463554},
               "Mflatness": {0: 2.3889355010530373, 1: 3.216744253805101},
               "Mbegin": {0: 0.7048598464016551, 1: 0.4073078318556883},
               "leader": {0: "t_0_leader", 1: "t_0_leader"},
               "follower": {0: "t_0", 1: "t_0"},
               "recordingId": {0: 1, 1: 1},
               "algorithm": {0: "differential_evolution",
                             1: "differential_evolution"},
               "objectives": {0: "distance", 1: "distance"},
               "weights": {0: np.nan, 1: np.nan},
               "gof": {0: "rmse", 1: "rmse"}}
    results_frame = pd.DataFrame.from_dict(results)
    return results_frame

class CreateCalibrationAnalysisTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_resources = Path(__file__).parents[1] / "resources/"
        self.output_path = self.test_resources / ".tmp/analysis"
        if self.output_path.exists():
            shutil.rmtree(self.output_path)
        self.output_path.mkdir(parents=True)
        self.data_directory = self.test_resources / "calibration2"
        results = create_results()
        results.to_csv(self.output_path
                       / "calibration_results_differential_evolution.csv")

    def tearDown(self) -> None:
        shutil.rmtree(self.output_path)

    def test_it_creates_pdfs(self):
        create_calibration_analysis(self.output_path, self.data_directory)
        result_file = (self.output_path
                       / 'calibration_results_differential_evolution_plots.pdf'
        )
        self.assertTrue(result_file.exists())
    
    def test_it_correctly_generates_calibration_plot(self):
        create_calibration_analysis(self.output_path, self.data_directory)
        # Add a test for the figures in the generated PDF
        # Here you can read the pdf file and use matplotlib functions to check if the plots are correct

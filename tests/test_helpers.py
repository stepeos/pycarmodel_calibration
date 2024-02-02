"""test helpers"""
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from carmodel_calibration.helpers import _estimate_parameters

np.random.seed(0)


class TestHelpers(unittest.TestCase):
    """testclass to test the helpers"""
    def setUp(self):
        self.identification = ["t_0_leader", "t_0", 1]
        self.data_chunk = pd.read_csv(Path(__file__).parent
                                      / "resources/data_chunk.csv")
        self.meta_data = pd.read_csv(Path(__file__).parent
                                      / "resources/meta_data.csv")

    def test_estimate_parameters(self):
        """method to test calculating the min_gap mbg mfl taccmax"""
        result = _estimate_parameters(self.identification,
                                           self.data_chunk,
                                           self.meta_data)
        expected_result = np.array((5.1,
                                    1.57,
                                    0.6400492351369179,
                                    2.8228514979648875,
                                    1.0475991619206706,
                                    0))
        rtol = 0.1
        self.assertTrue(all(np.isclose(result, expected_result, rtol=rtol)))

    def test_estimate_parameters_no_follower(self):
        """method to test calculating the min_gap mbg mfl taccmax"""
        self.identification[1] = np.nan
        result = _estimate_parameters(self.identification,
                                           self.data_chunk,
                                           self.meta_data)
        expected_result = np.array((None,
                                    2.37,
                                    None,
                                    None,
                                    1.0475991619206706,
                                    None))
        rtol = 0.1
        self.assertTrue(np.isclose(result[1], expected_result[1], rtol=rtol))
        self.assertTrue(np.isclose(result[4], expected_result[4], rtol=rtol))
        self.assertTrue(result[0] is None)
        self.assertTrue(result[2] is None)
        self.assertTrue(result[3] is None)
        self.assertTrue(result[5] is None)

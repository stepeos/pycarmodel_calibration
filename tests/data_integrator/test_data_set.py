"""module to test DataSet class"""


import json
from unittest import TestCase
import pandas as pd
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from mock import patch

from calibration_tool.data_integration.data_set import DataSet
from tests.testhelpers import chdir

TEST_DATA = {"col1": range(10,0, -1),
             "col2": range(10),
             "recordingId": range(10),
             "trackId": range(10)}

TEST_CONFIG = '''\
{
    "data_file_regex": "^\\\\d*\\\\_tracks\\\\.*",
    "skip_corrupt": false,
    "input_format": {},
    "output_format": {},
    "df_config": {
        "col1": {"new_name": "col2"},
        "col2": {"new_name": "col1"}
        },
    "df_kwargs": {}
}\
'''
TEST_DF = pd.DataFrame(data=TEST_DATA)


TMP_WORK_DIR = str(Path("tests/resources/fileaccess").absolute())

def show_mock(*_, **__):
    plt.close()

class TestDataSet(TestCase):
    """test DataSet class"""

    def setUp(self):
        if Path(TMP_WORK_DIR).exists():
            shutil.rmtree(TMP_WORK_DIR)
        data_set_path = Path(Path(__file__).parents[1]
                             / "resources/calibration2")
        shutil.copytree(
            data_set_path,
            TMP_WORK_DIR)

        # self.test_file = Path("123_tracks.log")
        # self.test_config = Path("data_config.json")
        # self.test_path = Path("321_tracks.log")
        # json_data = json.loads(TEST_CONFIG)
        # with open(self.test_config, 'w', encoding='utf-8') as file:
        #     json.dump(json_data, file)
        # TEST_DF.to_csv(str(self.test_file))
        # if self.test_path.exists():
        #     self.test_path.rmdir()

    def tearDown(self):
        if Path(TMP_WORK_DIR).exists():
            shutil.rmtree(TMP_WORK_DIR)

    @chdir(TMP_WORK_DIR)
    def test_convert_working(self):
        """method to test converting a dataframe without corrupt data"""
        test_instance = DataSet(TMP_WORK_DIR)
        datas = []
        for data, *_ in test_instance.get_next_dataframes():
            keys = ['heading', 'class', 'lane', 'acc', 'frame',
                    'distanceIntersectionCrossing', 'time',
                    'intersectionCrossing', 'speed', 'trackId', 'xCenter',
                    'yCenter']
            assert len(set(data.columns) & set(keys)) == len(keys)
            datas.append(data)
        assert len(datas) == 1

    @chdir(TMP_WORK_DIR)
    def test_get_filename_by_id(self):
        """method to test converting a dataframe without corrupt data"""
        test_instance = DataSet(TMP_WORK_DIR)
        result = Path(test_instance.get_filename_by_id(1))
        assert "01_trajectories.csv" == result.name

    @chdir(TMP_WORK_DIR)
    def test_get_dataframes_by_id(self):
        """method to test converting a dataframe without corrupt data"""
        test_instance = DataSet(TMP_WORK_DIR)
        test_instance.get_dataframes_by_id("01")

    @chdir(TMP_WORK_DIR)
    def test_get_dataframes_by_id_not_centered(self):
        """method to test converting a dataframe with not centered data"""
        with open(Path(TMP_WORK_DIR) / "data_config.json", "r", encoding="utf-8") as file:
            config_file = json.load(file)
        config_file.update({"xy_is_not_center": True})
        with open(Path(TMP_WORK_DIR) / "data_config.json", "w", encoding="utf-8") as file:
            json.dump(config_file, file)
        test_instance = DataSet(TMP_WORK_DIR)
        test_instance.get_dataframes_by_id("01")
        
    @chdir(TMP_WORK_DIR)
    @patch.object(plt, "show", show_mock)
    def test_interactive_plot(self):
        """method to test converting a dataframe with not centered data"""
        test_instance = DataSet(TMP_WORK_DIR)
        test_instance.interactive_plot(1)

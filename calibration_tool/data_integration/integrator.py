"""
Module to integrate ipnut-data into common data structure
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)


class Integrator(ABC):
    """"
    class to ingerate data
    """

    def __init__(self, directory):
        self._input_files = None
        self._directory = Path(directory)
        self._regex = None
        self._config_file = None
        self._skip_corrupt = False
        self._in_format = None
        self._out_format = None
        self._converted = None
        self._df_config = None
        self._df_kwargs = None
        self._metafile_replace = None
        self._intersection_points_lonlat = None
        self._intersection_points_xy = None
        self._file_specific_options = None
        self._xy_is_not_center = None
        self._lanes = None

    def _prepare_reading(self, data_file: Path) -> bool:
        assert isinstance(data_file, Path)
        if not data_file.exists():
            raise FileNotFoundError
        if not data_file.is_file():
            _LOGGER.error("path %s is directory not file", str(data_file))
            raise FileNotFoundError(
                f"path {str(data_file)} leads to directory")

    def _read_dataset_config(self):
        """method to read dataset config"""
        self._regex = self._config_file.get_value("data_file_regex") or\
             "^.*\\.json"
        self._skip_corrupt = \
            self._config_file.get_value("skip_corrupt") or False
        if self._config_file.get_value("xy_is_not_center"):
            self._xy_is_not_center = True
        else:
            self._xy_is_not_center = False
        self._intersection_points_xy = \
            self._config_file.get_value("intersection_points_xy") or []
        self._intersection_points_lonlat = \
            self._config_file.get_value("intersection_points_lonlat") or []
        self._file_specific_options = \
            self._config_file.get_value("file_specific_options") or []
        self._intersection_points_lonlat = \
            np.array(self._intersection_points_lonlat)
        self._lanes = self._config_file.get_value("lanes")
        self._intersection_points_xy = np.array(self._intersection_points_xy)
        self._in_format = self._config_file.get_value("input_format") or {}
        self._out_format = self._config_file.get_value("output_format") or {}
        self._df_config = self._config_file.get_value("df_config") or {}
        self._df_kwargs = self._config_file.get_value("df_kwargs") or {}
        self._metafile_replace = \
            self._config_file.get_value("metafile_replace") or ["", ""]

    def get_file_specific_options(self, recording_id) -> dict:
        """gets the file specific options"""
        general = self._config_file.get_values()
        for options in self._file_specific_options:
            if int(float(options["recordingId"])) == recording_id:
                general.update(options)
                return general
        return general

    @abstractmethod
    def convert(self) -> pd.DataFrame:
        """method to convert to our databse format"""
        return None

    @abstractmethod
    def prepare_reading(self):
        """prepares reading of the files"""
        return None

    @abstractmethod
    def get_next_dataframes(self):
        """generator that gets the next dataframes from the dataset directory"""
        yield

    @abstractmethod
    def get_lane_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        returns df with lanes as trajectory data and trackId as their name
        """
        return None

    @abstractmethod
    def get_filename_by_id(self, identification: int) -> str:
        """returns a data frame identified by an id"""
        return None

    @abstractmethod
    def get_meta_data_by_id(self, identification: int):
        """returns the meta data for a specific recording"""
        return None

"""module to handle simulatinos"""
import logging
import pickle
from multiprocessing import Manager
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import pandas as pd

from calibration_tool.data_integration.case_identification import (
    car_at_redlight, following_cars)
from calibration_tool.data_integration.data_set import DataSet
from calibration_tool.fileaccess.parameter import Parameters
from calibration_tool.logging_config import configure_logging
from calibration_tool.sumo.simulation_module import SumoInterface
from calibration_tool.sumo.sumo_project import SumoProject


configure_logging()
_LOGGER = logging.getLogger(__name__)
ITERATION = 1
START_TIME = None

class SimulationHandler:
    """class to handle the calibration process"""

    def __init__(self, directory: str, input_data: str,
                 param_keys: list = None,
                 num_workers: int = 100,
                 model: str = "eidm",
                 project_path: str = None):
        """
        :param directory:           output directory of calibration config and
                                    result data
        :param input_data:          directory to input data
        :param project_path:        path for the simulations
        :param model:               model to simulate
        :param param_keys:          list of paramters that are to be optimized,
                                    if left emtpy, then all except
                                    [`emergencyDecel`,`stepping`]
                                    will be tuned
        :param num_workers:         the initial population size, tuning
                                    paramter
        """
        self.directory = Path(directory)
        if not self.directory.exists():
            self.directory.mkdir(parents=True)
        self._input_data = Path(input_data)
        if not self._input_data.exists():
            raise FileNotFoundError("Input directory does not exist.")
        self._prepared = False
        self._model = model
        self._data_set = None
        self.selection_data = None
        self.num_workers =num_workers
        self.param_keys = param_keys or list(
            Parameters.get_defaults_dict().keys())
        self.meta_data = None
        if project_path:
            self.delete_path = False
            self.project_path = Path(project_path)
        else:
            self.delete_path = True
            self.project_path = self.directory / "sumo_project"
        self._lock = None
        self._identification = None
        self.x_0 = None
        self.pool = None
        self.manager = None
        self._check_options()

    def __del__(self):
        # if self.delete_path and self.project_path.exists():
        #     shutil.rmtree(self.project_path)
        if hasattr(self, "pool"):
            if self.pool:
                self._close_pool()
                del self.pool

    def _check_options(self):
        pass

    def _prepare_input_data(self):
        self._data_set = DataSet(self._input_data)
        self._data_set.prepare_reading()

    def prepare_selection(self, use_save_state=True):
        """
        Prepares the simulation
        """
        self._prepare_input_data()
        if use_save_state and (self._input_data / "selection.pickle").exists():
            _LOGGER.debug("Reading selection data from pickle file.")
            with open(self._input_data / "selection.pickle", "rb") as file:
                data_dump = pickle.load(file)
                self.selection_data = data_dump["selection"]
                self.meta_data = data_dump["meta_data"]
        else:
            _LOGGER.debug("Calculating selection data")
            self.selection_data, self.meta_data = self._get_selection()
            with open(self._input_data / "selection.pickle", "wb") as file:
                data_dump = {"selection": self.selection_data,
                             "meta_data": self.meta_data}
                pickle.dump(data_dump, file)
                self._create_selection_frames(self.selection_data)
            _LOGGER.debug("Wrote selection data to pickle file.")
        self._create_selection_frames(self.selection_data)
        self._prepared = True

    def _get_selection(self):
        _LOGGER.info("Starting to read the input data.")
        entry_data = []
        meta_datas = []
        for data_frame, meta_data, lane_data in \
            self._data_set.get_next_dataframes():
            entries = []
            pairs_count = 0
            rec = meta_data["recordingId"].values[0]
            _LOGGER.info("Read recording #%s.", rec)
            free_leaders = ""
            lanes = self._data_set.get_file_specific_options(rec).get("lanes")
            for leader in car_at_redlight(data_frame, lane_data, lanes):
                free_leaders += " " + str(leader)
                entries.append(("", leader))
            _LOGGER.info("Free leaders %s.", free_leaders)
            coord = self._data_set.get_file_specific_options(rec).get(
                "coordinate_system")
            traffic_light_time = self._data_set.get_file_specific_options(
                rec).get("traffic_light_time")
            if coord:
                use_xy = (coord == ["xCenter", "yCenter"])
            else:
                use_xy = False
            for leader, foll in following_cars(
                data_frame, lane_data, meta_data, use_xy, lanes=lanes,
                traffic_light_time=traffic_light_time):
                entries.append((leader, foll))
                pairs_count += 1
                lane = (
                    np.rint(np.mean(data_frame[data_frame["trackId"]==leader]
                                        ["lane"].values)).astype(int))
                meta_datas.append(self._get_meta_data(meta_data, entries, rec))
                _LOGGER.debug(
                    "Found leader=%d and follower=%d lane=%d.", leader, foll,
                    lane)
            _LOGGER.info("Found %d pairs in recording %d.", pairs_count, rec)
            entry_data.extend(self._get_entry_data(data_frame, entries, rec))
        dtypes = {"leader": str,
                  "follower": str,
                  "lane": int,
                  "leaderIntersectionCrossing":float,
                  "followerIntersectionCrossing": float,
                  "recordingId": int}
        selection_data = pd.DataFrame(entry_data).astype(dtype=dtypes)
        meta_data = pd.concat(meta_datas)
        order = ["recordingId", "leaderIntersectionCrossing", "lane"]
        selection_data.sort_values(by=order, inplace=True)
        self._create_selection_frames(selection_data)
        return selection_data, meta_data

    def _create_selection_frames(self, selection_data):
        for recording_id in selection_data["recordingId"].unique():
            chunk = selection_data[selection_data["recordingId"]==recording_id]
            data_file_name = Path(
                self._data_set.get_filename_by_id(recording_id))
            data_file_name = data_file_name.name
            name = self._input_data / f"{data_file_name[:-4]}_selection.csv"
            chunk.to_csv(name)

    def _get_meta_data(self, meta_data, entries, rec,
                       identification=None) -> pd.DataFrame:
        meta_data = self._data_set.get_meta_data_by_id(rec)
        entry_data = np.array(entries)
        leaders = list(entry_data[:, 0])
        followers = list(entry_data[:, 1])
        meta_data_entries = (
            meta_data[
            (meta_data["trackId"].isin(leaders + followers))
            & (meta_data["recordingId"] == rec)])
        if identification:
            meta_data_entries = meta_data_entries[
                (meta_data_entries["trackId"].isin(identification[:2]))
            ]
        return meta_data_entries

    def _get_entry_data(self, data_frame, entries, rec) -> dict:
        entry_data = []
        for leader, follower in entries:
            follower_frame = data_frame[data_frame["trackId"]==follower]
            lane = np.rint(np.mean(follower_frame["lane"].values))
            follower_crossing = (follower_frame["intersectionCrossing"]
                                   .values[0])
            if leader != "":
                leader_frame = data_frame[data_frame["trackId"]==leader]
                leader_crossing = (
                        leader_frame["intersectionCrossing"].values[0])
                leader_crossing = float(leader_crossing)
            else:
                leader_crossing = None
            entry_data.append({
                "leader": leader,
                "follower": follower,
                "lane": lane,
                "leaderIntersectionCrossing": leader_crossing,
                "followerIntersectionCrossing": follower_crossing,
                "recordingId": rec})
        return entry_data

    def _create_pool(self):
        if not self.manager:
            self.manager = Manager()
        # if (sys.platform == "linux"
        #         or self._optimization == "genetic_algorithm"):
        #     pool = manager.Pool(self.num_workers)
        # else:
        #     pool = ThreadPool(self.num_workers)
        pool = ThreadPool(self.num_workers)
        self.pool = pool

    def _close_pool(self):
        self.pool.close()
        self.pool.terminate()

def simulate_params(param_sets, data_path, identification, project_path):
    """simulates on param set and returns ground_truth and prediction"""
    project_path = Path(project_path)
    routes_path = Path(project_path) / "calibration_routes.rou.xml"
    SumoProject.create_sumo(project_path, len(param_sets))
    SumoProject.write_followers_leader(routes_path, param_sets)
    sumo = SumoInterface(project_path, data_path, gui=False)
    results = sumo.run_simulation(identification)[1]
    ground_truth , prediction = results["ground_truth"], results["prediction"]
    gt_chunks = []
    pred_chunks = []
    for idx in range(len(param_sets)):
        gt_chunk = ground_truth[ground_truth["counter"]==idx]
        gt_chunks.append(gt_chunk)
        pred_chunk = prediction[prediction["counter"]==idx]
        pred_chunks.append(pred_chunk)
    return gt_chunks, pred_chunks

def get_weighted_errors(param_sets, data_path, identification,
                        objective_function, project_path):
    """simulate one and return the weighted error"""
    gts, preds = simulate_params(
        param_sets, data_path, identification, project_path)
    weighted_errors = []
    for ground_truth, prediction in zip(gts, preds):
        weighted_errors.append(objective_function(ground_truth, prediction))
    return weighted_errors

# pylint: disable=E0401
"""module to handle sumo interfacing"""

import logging
import os
import re
from math import ceil
from pathlib import Path
import pickle
import datetime
import sys
import subprocess
from random import randint

import numpy as np
import pandas as pd

from carmodel_calibration.data_integration.data_set import DataSet
from carmodel_calibration.exceptions import (FolderNotFound, MissingConfiguration,
                                         MultipleConfigurations)
from carmodel_calibration.helpers import _get_starting_time, _get_vehicle_meta_data
from carmodel_calibration.sumo.sumo_project import SumoProject

_LOGGER = logging.getLogger(__name__)
# pylint: disable=C0103,W0603
traci = None

def _chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def _get_distance_df(leader_chunk: pd.DataFrame, follower_chunk: pd.DataFrame):
    pos_leader = leader_chunk[["xCenter", "yCenter"]].values
    pos_leader = np.atleast_2d(pos_leader)
    pos_follower = follower_chunk[["xCenter", "yCenter"]].values
    pos_follower = np.atleast_2d(pos_follower)
    return np.sqrt(np.sum((pos_leader-pos_follower)**2, axis=1).astype(np.float64))

def _create_leader_leader_chunk(track_chunk: pd.DataFrame):
    starting_time = _get_starting_time(track_chunk)
    leader_chunk = track_chunk[track_chunk["time"]>=starting_time].copy()
    # initial distance should be 2 meters
    len_vehicle = leader_chunk["length"].values[0]
    distance_per_axis = np.sqrt((2+len_vehicle)**2/2)
    distance = _get_distance_df(leader_chunk.iloc[1:],leader_chunk.iloc[:-1])
    distance_is = distance[0]
    difference = distance_per_axis - distance_is
    leader_chunk.loc[leader_chunk.index[0], "xCenter"] -= difference
    leader_chunk.loc[leader_chunk.index[0], "yCenter"] -= difference

    # after the initial distance, distance shuold be atleast 500m
    distance_per_axis = np.sqrt(500**2/2)
    difference = distance_per_axis - distance
    x_pos = leader_chunk.loc[leader_chunk.index[1:], "xCenter"] + difference
    y_pos = leader_chunk.loc[leader_chunk.index[1:], "yCenter"] + difference
    leader_chunk.loc[leader_chunk.index[1:], "xCenter"] = x_pos
    leader_chunk.loc[leader_chunk.index[1:], "yCenter"] = y_pos
    return leader_chunk

def _calculate_distance(x_series, y_series):
    pos = np.stack((x_series, y_series), axis=1)
    delta_pos = np.sqrt(np.sum((pos - np.roll(pos, -1, axis=0))**2, axis=1))
    delta_pos = np.roll(delta_pos, 1, axis=0)
    delta_pos[0] = 0
    covered_distance = np.cumsum(delta_pos)
    return covered_distance

def _interpolate_jumpy_start(leader_jumpy, follower_jumpy):
    leader_synced, follower_synced = leader_jumpy.copy(), follower_jumpy.copy()
    covered_distance_l = _get_distance_df(
                leader_synced.iloc[1:], leader_synced.iloc[:-1])
    covered_distance_f = _get_distance_df(
                follower_synced.iloc[1:], follower_synced.iloc[:-1])
    free_follower = (set(leader_synced["trackId"])
                     == set(follower_synced["trackId"]))
    columns = ["xCenter", "yCenter", "speed", "lon", "lat", "accel"]
    jumpy_points = []
    if np.any(np.abs(covered_distance_l) > 1):
        jumpy_points.extend(np.where(np.abs(covered_distance_l) > 1)[0])
    if np.any(np.abs(covered_distance_f) > 1):
        jumpy_points.extend(np.where(np.abs(covered_distance_f) > 1)[0])
    jumpy_points = np.unique(jumpy_points)
    if len(jumpy_points) == 0:
        return leader_jumpy, follower_jumpy
    start = max(1, jumpy_points[0]-8)
    stop = max(start+30, jumpy_points[-1]-8)
    x_averaged_f = np.convolve(
                follower_synced["xCenter"].values[start:stop],
                np.ones(16)/16,
                mode='valid')
    y_averaged_f = np.convolve(
                follower_synced["yCenter"].values[start:stop],
                np.ones(16)/16,
                mode='valid')
    indexes = follower_synced.index[start+7:stop-8]
    follower_synced.loc[indexes, "xCenter"] = x_averaged_f
    follower_synced.loc[indexes, "yCenter"] = y_averaged_f

    x_averaged_f = np.convolve(
                follower_synced["xCenter"].values[start:stop],
                np.ones(16)/16,
                mode='valid')
    y_averaged_f = np.convolve(
                follower_synced["yCenter"].values[start:stop],
                np.ones(16)/16,
                mode='valid')
    indexes = follower_synced.index[start+7:stop-8]
    follower_synced.loc[indexes, "xCenter"] = x_averaged_f
    follower_synced.loc[indexes, "yCenter"] = y_averaged_f
    
    x_averaged_l = np.convolve(
                leader_synced["xCenter"].values[start:stop],
                np.ones(16)/16,
                mode='valid')
    y_averaged_l = np.convolve(
                leader_synced["yCenter"].values[start:stop],
                np.ones(16)/16,
                mode='valid')
    indexes = leader_synced.index[start+7:stop-8]
    leader_synced.loc[indexes, "xCenter"] = x_averaged_l
    leader_synced.loc[indexes, "yCenter"] = y_averaged_l
    return leader_synced, follower_synced

class SumoInterface:
    """class that handles the sumo app"""

    def __init__(self, sumo_project_path: Path, leader_follower_path: Path, remote_port: int = randint(8000, 9000),
                 gui=False, file_buffer=None):
        self.args = (sumo_project_path, leader_follower_path)
        self.kwargs = {"gui": gui, "file_buffer": file_buffer}
        if not sumo_project_path.exists():
            _LOGGER.error("Could not find sumo path:\n%s.",
                          str(sumo_project_path))
            raise FileNotFoundError
        self.sumo_project_path = sumo_project_path
        if sumo_project_path.is_file():
            _LOGGER.error("Sumo_project_path must be directory .sumocfg etc.")
            raise FileExistsError
        self.network_file = sumo_project_path / "calibration_network.xml"
        if not self.network_file.exists():
            _LOGGER.error("Coud not find %s.",
                          str(self.network_file))
            raise FileNotFoundError
        self.routes_file = sumo_project_path / "calibration_routes.rou.xml"
        if not self.routes_file.exists():
            _LOGGER.error("Coud not find %s.",
                          str(self.network_file))
            raise FileNotFoundError
        _LOGGER.debug("Successfully entered the sumo project directory.")
        files = sumo_project_path.glob("*")
        configs = []
        for file in files:
            file = Path(file)
            if re.match(r"^.*(\.sumocfg|\.sumo\.cfg)$", file.name):
                configs.append(file)
        if len(configs) > 1:
            _LOGGER.error("Found multiple simulation configurations in %s.",
                        str(sumo_project_path))
            raise MultipleConfigurations
        if len(configs) == 0:
            _LOGGER.error("no config in %s",
                          str(sumo_project_path))
            raise MissingConfiguration
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            _LOGGER.error("Please declare environment variable 'SUMO_HOME'.")
            raise ModuleNotFoundError
        self.data_dir = Path(leader_follower_path)
        if not self.data_dir.exists():
            _LOGGER.error("Path with leader-follower data does not exist:\n%s",
                          str(self.data_dir))
            raise FolderNotFound
        self.traci_log = sumo_project_path / "traci.log"
        self.trajectories_file = sumo_project_path / "trajectories.xml"
        with open(self.traci_log, "w", encoding="utf-8") as file:
            now = str(datetime.datetime.now())
            file.write(f"beginning traci logging now:{now}\n")
        files = [file for file in self.data_dir.glob("*selection.csv")]
        if len(files) == 0:
            _LOGGER.error("Could not find any leader-follower data files, "
                          "looking for \'*_selection.csv\'.")
            raise FileNotFoundError("Leader-Follower data not found.")
        self.network_info = self._get_network_info()
        self._port = remote_port
        self.reload = [f"-c={str(configs[0])}"]
        initial_fcd_file = self.sumo_project_path / "01_trajectories.xml"
        self.cmd = [
            ("sumo-gui" if gui else "sumo"),
            self.reload[0],
            f"--remote-port={self._port}",
            f"--fcd-output={initial_fcd_file}",
            "--fcd-output.acceleration",
            "--fcd-output.max-leader-distance=1000",
            "--startup-wait-threshold=-1",
            "--quit-on-end",
            "--seed=2023"
        ]
        self.file_buffer = file_buffer
        self.proc = None
        self.selection_ids = None
        self.selection_data = None
        self.identification = None
        self.start_simulation_module()

    def release(self):
        """stops the simulation module and releases all files"""
        try:
            traci.close(False)
            self.proc.kill()
        # pylint: disable=W0702
        except:
            pass

    def start_simulation_module(self):
        """starts the simulation module"""
        self.proc = subprocess.Popen(
            self.cmd, stdout=self.file_buffer or subprocess.DEVNULL,
            stderr=self.file_buffer or subprocess.DEVNULL)
        self._init_traci()

    def _init_traci(self):
        global traci
        try:
            # pylint: disable=C0415
            import traci
        except Exception as exc:
            _LOGGER.error("Failed to import traci, fix SUMO_HOME:%s",
                        os.environ.get("SUMO_HOME"))
            raise exc
        try:
            label = str(self.sumo_project_path)
            traci.init(self._port, label=label)
        except exc:
            _LOGGER.debug("Traci init failed with message %s"
                          ", trying to continue...", str(exc))

    def serialized(self):
        """creates a serializable instance of itself"""
        return SumoInterfaceSerialized(*self.args, **self.kwargs)

    def _get_network_info(self):
        network_info = {}
        route_count = SumoProject.get_number_routes(self.routes_file)
        followers_count = SumoProject.get_number_followers(self.routes_file)
        network_info["count"] = min(route_count, followers_count)
        network_info.update(SumoProject.get_network_info(self.network_file))
        return network_info

    def run_simulation(self, identification=None) -> dict:
        """executes all steps and finishes simulation"""
        # simulation runs in steps within epochs
        # each epoch is chunk of cars, that fit into the sumo calibration
        # network.
        # each step moves the calibration leaders to their new distance from
        # their follower
        label = str(self.sumo_project_path)
        traci.switch(label)
        self.network_info = self._get_network_info()
        new_trajectories_file = (
            self.sumo_project_path / "01_trajectories.xml")
        traci.load(self.reload
                       + ["--fcd-output", str(new_trajectories_file),
                          "--fcd-output.acceleration",
                          "--startup-wait-threshold=-1",
                          "--fcd-output.max-leader-distance=1000"])
        return self._chunk(identification=identification)

    def _get_selection_data(self, identification = None):
        selection_file = self.data_dir / "selection.pickle"
        if not (selection_file).exists():
            raise FileNotFoundError(
                f"Selection file '{selection_file}' not found")
        selection_ids = []
        selection_data = []
        leader_follower = pd.DataFrame()
        with open(self.data_dir / "selection.pickle", "rb") as file:
            leader_follower = pickle.load(file)["selection"]
            recording_ids_set = set(leader_follower["recordingId"])
        if identification:
            recording_ids_set = [identification[2]]
        for recording_id in recording_ids_set:
            filename = DataSet(self.data_dir).get_filename_by_id(
                recording_id)
            data_chunk, meta_data, *_ = DataSet.read_file(
                Path(filename),
                skip_preprocessing=True)
            leader_follower_chunk = leader_follower[
                leader_follower["recordingId"]==recording_id].copy()
            leader = leader_follower_chunk["leader"].to_list()
            follower = leader_follower_chunk["follower"].dropna().to_list()
            track_ids = set(leader + follower)
            data_chunk = data_chunk[
                data_chunk["trackId"].isin(track_ids)].copy()
            for track_id, chunk in meta_data.groupby(by="trackId"):
                indexes = data_chunk[data_chunk["trackId"]==track_id].index
                data_chunk.loc[indexes, "recordingId"] = (
                    chunk["recordingId"].values[0])
                data_chunk.loc[indexes, "length"] = chunk["length"].values[0]
                data_chunk.loc[indexes, "width"] = chunk["width"].values[0]
            leader_follower_chunk.loc[:,"recordingId"] = recording_id
            selection_ids.append(leader_follower_chunk)
            selection_data.append(data_chunk)
        selection_ids_out = pd.concat(selection_ids).copy()
        selection_data = pd.concat(selection_data).dropna()
        if identification:
            selection_ids_out = selection_ids_out[
                (selection_ids_out["leader"]==identification[0])
                & (selection_ids_out["follower"]==identification[1])
                & (selection_ids_out["recordingId"]==identification[2])]
            follower = selection_ids_out["follower"].values[0]
            track_ids = identification[:2]
            selection_data = selection_data[
                (selection_data["trackId"].isin(track_ids))
                & (selection_data["recordingId"]==identification[2])]
        return selection_ids_out, selection_data

    def _chunk(self, identification=None):
        # epoch is chunk of cars that fit into network
        network_capacity = self.network_info["count"]
        if self.identification != identification:
            selection_ids, selection_data = self._get_selection_data(
                identification=identification)
            self.selection_ids = selection_ids
            self.selection_data = selection_data
            self.identification = identification
        else:
            selection_ids = self.selection_ids
            selection_data = self.selection_data
        if identification:
            selection_ids = (
                selection_ids.loc[
                    np.repeat(selection_ids.index, network_capacity)]
                    .reset_index(drop=True))
        num_chunks = ceil(len(selection_ids) / network_capacity)
        _LOGGER.debug("Running %d simulations, %d cars in total.",
                     num_chunks,
                     len(selection_ids))
        chunk_counter = 0
        chunk_inputs = []
        for chunk in _chunker(selection_ids, network_capacity):
            out  = self._process_chunk(selection_data, chunk)
            (max_time,
             max_leader_time,
             simulation_data,
             ground_truth) = out
            recording_ids = list(np.unique(ground_truth["recordingId"]))
            _LOGGER.debug(
                "[REC %s] Chunk %d with target simulation time %f seconds "
                "due to leader %s.",
                ",".join([str(item) for item in recording_ids]),
                chunk_counter + 1,
                max_time,
                max_leader_time)
            self._steps(simulation_data, max_time)
            # read simulation results
            chunk_inputs.append(ground_truth)
            chunk_counter += 1
            _LOGGER.debug("Finished %d/%d sumo chunks.",
                         chunk_counter,
                         num_chunks)
            new_trajectories_file = (
                self.sumo_project_path
                / f"{chunk_counter+1:02d}_trajectories.xml")
            traci.load(self.reload
                       + ["--fcd-output", str(new_trajectories_file),
                          "--fcd-output.acceleration",
                          "--startup-wait-threshold=-1",
                          "--fcd-output.max-leader-distance=1000"])
            self._read_simulation_prediction(chunk_counter)
        _LOGGER.debug("Finished simulations, gathering simulation results.")
        simulation_results = \
            self._create_simulation_dataframe(chunk_inputs)
        try:
            self._pickle_dump(simulation_results, "simulation_results")
        except Exception as exc:
            _LOGGER.error("Failed to dump pickle.")
            raise exc
        _LOGGER.debug("Dumped simulation results into pickle.")
        return simulation_results

    def _process_chunk(self, selection_data, chunk):
        track_ids = set(chunk["leader"].to_list()
                            + chunk["follower"].to_list())
        chunk_data = \
                selection_data[selection_data["trackId"].isin(track_ids)]
        max_time = 0
        max_leader_time = 0
        counter = 0
        simulation_data = {}
        ground_truths = []
        # calculate the length of steps at the beginning of the simulation to
        # build up reaction time. Cars stand still until t > 0.25s
        steps_reaction_time = np.rint(.3 // 0.04).astype(int) + 1
        for leader, follower, recording_id in zip(chunk["leader"].values,
                                        chunk["follower"].values,
                                        chunk["recordingId"].values):
            # 1: Check if leader exists
            # 2: Get time to simulate
            # 2: Get the starting time at which cars start moving
            # 3: Set leader and follower positions
            # 4: Determine initial distance
            # 5: Add 0.5 seconds standstill
            # 6: Set leader accumulated driven distance
            # 7: Set follower accumulated driven distance
            # 8: Set ground truth dataframe
            follower_chunk = chunk_data[
                        (chunk_data["trackId"] == follower)
                        & (chunk_data["recordingId"] == recording_id)
                        ].sort_values(by="time")
            if leader != "":
                # /W Leader
                condition = True
                leader_chunk = chunk_data[
                        (chunk_data["trackId"] == leader)
                        & (chunk_data["recordingId"] == recording_id)
                        ].sort_values(by="time")
            else:
                # W/O Leader
                condition = False
                leader_chunk = _create_leader_leader_chunk(
                    follower_chunk.copy())

            # len_f = follower_chunk["length"].values[0]
            len_l = leader_chunk["length"].values[0]
            starting_time_l = _get_starting_time(leader_chunk)
            starting_time_f = _get_starting_time(follower_chunk)
            starting_time = min(starting_time_l, starting_time_f)
            stopping_time = min(leader_chunk["time"].values[-1],
                                follower_chunk["time"].values[-1])
            # synced chunks have the same time values
            leader_synced = leader_chunk[
                leader_chunk["time"].between(
                    starting_time, stopping_time)]
            follower_synced = follower_chunk[
                follower_chunk["time"].between(
                    starting_time, stopping_time)]
            # initial distance at standstill
            initial_distance_ss = _get_distance_df(leader_synced,
                                                    follower_synced)[0]
            initial_distance_ss -= len_l
            # initial_distance_ss -= (len_f + len_l) / 2
            initial_distance_ss = max(0.5, initial_distance_ss)
            leader_synced, follower_synced = _interpolate_jumpy_start(
                leader_synced, follower_synced)

            # add 0.5 seconds of standstill to the synced chunks to build
            # up reaction time and adjust the time
            leader_ss = pd.concat(
                [leader_synced.iloc[0].to_frame().T] * steps_reaction_time)
            leader_ss["speed"] = 0
            leader_synced = pd.concat((leader_ss, leader_synced))
            new_time = np.arange(0, len(leader_synced), 1) * 0.04
            leader_synced.reset_index(drop=True, inplace=True)
            leader_synced["time"] = new_time
            follower_ss = pd.concat(
                [follower_synced.iloc[0].to_frame().T] * steps_reaction_time)
            follower_ss["speed"] = 0
            follower_synced = pd.concat((follower_ss, follower_synced))
            follower_synced.reset_index(drop=True, inplace=True)
            follower_synced["time"] = new_time

            #  covered distance per vehicle
            distance = (_get_distance_df(leader_synced, follower_synced)
                            - len_l)
            delta_distance = distance[1:] - distance[:-1]
            delta_distance = np.insert(
                delta_distance, 0, initial_distance_ss, axis=0)
            covered_distance_f = _get_distance_df(
                follower_synced.iloc[1:], follower_synced.iloc[:-1])
            accumulated_distance_f = np.cumsum(
                np.insert(covered_distance_f, 0, 0, axis=0))
            accumulated_distance_l = (
                np.cumsum(delta_distance)
                + accumulated_distance_f)
            new_speed = np.insert((accumulated_distance_l[1:]
                               - accumulated_distance_l[:-1]) / 0.04,
                               0, 0, axis=0)
            new_speed_padded = np.pad(new_speed, (5//2, 5-1-5//2), mode="edge")
            new_speed_smooth = np.convolve(new_speed_padded, np.ones((5,))/5,
                                           mode="valid")
            leader_synced["speed"] = np.clip(new_speed_smooth,0, np.inf)

            if leader != "":
                # distance = (_get_distance_df(leader_synced, follower_synced)
                #             - (len_l + len_f) / 2)
                distance = (_get_distance_df(leader_synced, follower_synced)
                            - len_l)
            else:
                distance = accumulated_distance_f
            # Update the simulation time that is required for all pairs to
            # finish
            if new_time[-1] > max_time:
                max_time = new_time[-1]
                max_leader_time = leader
            # Create the ground truth dataframe
            ground_truth = pd.DataFrame({
                "time": new_time-0.04,
                "xCenterLeader": leader_synced["xCenter"],
                "yCenterLeader": leader_synced["yCenter"],
                "xCenterFollower": follower_synced["xCenter"],
                "yCenterFollower": follower_synced["yCenter"],
                "coveredDistanceLeader": accumulated_distance_l,
                "coveredDistanceFollower": accumulated_distance_f,
                "distance": distance,
                "speedLeader": leader_synced["speed"],
                "speedFollower": follower_synced["speed"]})
            ground_truth["counter"] = counter
            ground_truth["leader"] = leader
            ground_truth["follower"] = follower
            ground_truth["recordingId"] = recording_id
            # cars are just inserted at timestep 0 thus exist from timestep + 1
            ground_truth = ground_truth[ground_truth["time"]!=-0.04]
            ground_truths.append(ground_truth)
            simulation_data[counter] = [leader,
                                        condition,
                                        recording_id,
                                        accumulated_distance_l+0.1,
                                        leader_synced["speed"].values]
            counter += 1
        ground_truth = pd.concat(ground_truths)
        out = (max_time,
               max_leader_time,
               simulation_data,
               ground_truth)
        return out

    @staticmethod
    def _get_xy_pos(pos_car, steps):
        x_car = np.hstack((np.tile(pos_car[0,0],
                                   steps),
                           pos_car[:, 0]))
        y_car = np.hstack((np.tile(pos_car[0,1],
                                   steps),
                           pos_car[:, 1]))
        return x_car, y_car

    def _read_simulation_prediction(self, chunk):
        results = None
        xml2csv = str(Path(os.environ["SUMO_HOME"]) / "tools" / "xml" /
                      "xml2csv.py")
        cmd = [sys.executable, xml2csv, "--separator=,",
               f"{self.sumo_project_path}{os.sep}{chunk:02d}_trajectories.xml"]
        try:
            _ = subprocess.check_output(cmd)
            _LOGGER.debug("Running %s",
                          " ".join(cmd))
            _LOGGER.debug("Converted %02d_trajectories.xml to csv",
                          chunk)
        except subprocess.CalledProcessError as exc:
            _LOGGER.error("Could not convert %02d_trajectories.xml to csv %s",
                                     chunk,
                                     exc)
        return results

    def _create_simulation_dataframe(self, ground_truth):
        names = ["time", "acc", "angle", "trackId", "counter", "distance",
                 "leaderTrackId", "speedLeader", "coveredDistanceFollower",
                 "slope", "speed", "type", "LocalXFollower", "LocalYFollower"]
        use_names = ["time", "acc", "trackId", "counter", "distance", "leaderTrackId",
                     "speedLeader", "coveredDistanceFollower", "speed",
                     "LocalXFollower", "LocalYFollower"]
        simulation_results = {}
        chunk_results = list(self.sumo_project_path.glob("*_trajectories.csv"))
        for chunk_ground_truth, chunk_result, idx in zip(ground_truth,
                                                    chunk_results,
                                                    range(len(chunk_results))):
            prediction = pd.read_csv(chunk_result, names=names, header=0,
                                     usecols=use_names)
            prediction["time"] -= 0.04
            prediction = prediction[~prediction["trackId"].isna()].copy()
            prediction["edgeName"] = prediction["counter"].str.replace(
                "_0", "")
            prediction["counter"] = (prediction["counter"]
                                     .str.extract(r"([0-9]+)")
                                     .astype(int))
            # initial_distances = TODO
            prediction = (prediction
                .groupby(["trackId"], group_keys=True)
                .apply(lambda gdf: gdf.assign(
                    coveredDistanceFollower=
                    lambda x: _calculate_distance(x["LocalXFollower"],
                                                x["LocalYFollower"])))
                .reset_index(drop=True)
            )
            condition = prediction["trackId"].str.contains("_leader")
            prediction_followers = (
                prediction[~condition].sort_values(["time", "counter"])
                .rename(columns={"acc": "accFollower"}))
            prediction_followers = prediction_followers.rename(
                columns={"speed": "speedFollower"})
            prediction_leaders = (
                prediction[condition]
                .drop(["leaderTrackId", "speedLeader"],
                      axis=1)
                .rename(columns=
                        {"coveredDistanceFollower": "coveredDistanceLeader",
                         "LocalXFollower": "LocalXLeader",
                         "LocalYFollower": "LocalYLeader",
                         "acc": "accLeader"})
                .sort_values(["time", "counter"])
                [["accLeader", "coveredDistanceLeader"]]
            )
            prediction_followers[["accLeader", "coveredDistanceLeader"]] =  (
                prediction_leaders.values)
            prediction_followers[["recordingId", "leader", "follower"]] = (
                chunk_ground_truth.sort_values(["time", "counter"])
                [["recordingId", "leader", "follower"]].values)
            prediction = prediction_followers
            nans_pred = prediction[prediction["leader"]==""]
            prediction.loc[nans_pred.index, "distance"] = (
                nans_pred["coveredDistanceFollower"])
            chunk_data = {
                "ground_truth": chunk_ground_truth,
                "prediction": prediction
            }
            simulation_results[idx + 1] = chunk_data
        return simulation_results

    def _pickle_dump(self, sim, file_name):
        with open(self.sumo_project_path / (file_name + ".pickle"),
                  "wb") as file:
            pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _steps(self, step_data, max_time):
        # move cars by id in a LOOP
        step = 0
        sim_steps = np.rint(max_time / 0.04) + 1
        while step <= sim_steps:
            try:
                for edge, (_, leader_exists, _, accumulated_distance,
                           leader_speed) in (
                    step_data.items()):
                    if step == 0:
                        dist = accumulated_distance[step]
                        if leader_exists:
                            self._init_leader(edge, dist, "blue")
                        else:
                            self._init_leader(edge, dist, "red")
                    elif step == accumulated_distance.shape[0]:
                        self._stop_cars(edge)
                    elif step >= accumulated_distance.shape[0]:
                        continue
                    else:
                        dist = accumulated_distance[step]
                        if dist > 1000:
                            # TODO: get max distance of route from network file
                            continue
                        speed = leader_speed[step]
                        self._move_car(edge, dist, speed)
            except Exception as exc:
                _LOGGER.error("Failed moving cars at simulation step %d.",
                              step)
                raise exc
            try:
                traci.simulationStep()
                # if np.isclose(traci.simulation.getTime(), 0.25, atol=0.5):
                #     print("debug")
            except Exception as exc:
                _LOGGER.error("Failed executing next simulation step.")
                raise exc
            step += 1
        while traci.simulation.getMinExpectedNumber() != 0:
            traci.simulationStep()

    def _init_leader(self, edge, dist, color):
        edge_name = self.network_info[edge]
        # depart_position = follower length + distance + leader length
        depart_position = 5.0 + dist + 5.0
        traci.vehicle.add(edge_name+"_leader",
                          f"route_{int(edge):d}",
                          "leader0",
                          depart=0.04,
                          departPos=depart_position)
        if color == "red":
            rgba = (255, 0, 0, 255)
        elif color == "blue":
            rgba = (0, 0, 255, 255)
        else:
            rgba = (255, 165, 0, 255)
        traci.vehicle.setColor(edge_name+"_leader", rgba)

    def _move_car(self, edge, dist, speed):
        # https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-moveTo
        current_position = 5.0 + dist + 5.0
        edge_name = self.network_info[edge]
        leader_name = edge_name + "_leader"
        traci.vehicle.moveTo(leader_name,
                             f"{edge_name}_0",
                             pos=current_position)
        traci.vehicle.setPreviousSpeed(leader_name, speed)

    def _stop_cars(self, edge):
        edge_name = self.network_info[edge]
        try:
            # stop the leader and the follower ✋✋✋
            traci.vehicle.remove(f"t_{edge}")
            traci.vehicle.remove(edge_name+"_leader")
        except Exception as exc:
            _LOGGER.warning(
                "Could not delete cars, the pair [ %s_leader, t_%s ]"
                " stopped prematurely: %s",
                edge_name,
                str(edge),
                str(exc))


class SumoInterfaceSerialized(SumoInterface):
    """serializable child of the SumoInterface"""

    def __init__(self, *args, **kwargs):
        kwargs["file_buffer"] = None
        super(SumoInterfaceSerialized, self).__init__(*args, **kwargs)
        self.proc = None

    def __del__(self):
        pass

    def start_simulation_module(self, *_):
        self.proc = None

    def _init_traci(self):
        pass

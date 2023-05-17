import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import scipy.optimize as op
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from calibration_tool.helpers import (_get_vehicle_meta_data, _acceleration_curve_factory)
# from calibration_tool.data_integration.case_identification import following_cars
from calibration_tool.data_integration.case_identification import *
from calibration_tool.data_integration.data_set import DataSet
from calibration_tool.sumo.simulation_module import _get_distance_df
rc_file = Path(__file__).parents[1] / "calibration_tool/data_config/matplotlib.rc"
plt.style.use(rc_file)
data_set = DataSet(Path(os.environ["DATA_DIR"]))


def _get_starting_time(pos_car):
    starting_indexes = np.argwhere(pos_car["speed"].values==0)
    if starting_indexes.shape[0] != 0:
        starting_index = starting_indexes[-1,0]
        # starting_index = np.clip(starting_indexes[-1,0] + 1, 0,
        #                          pos_car.values.shape[0]-1)
        return pos_car["time"].iloc[starting_index]
    else:
        condition = np.isclose(pos_car["speed"].values, 0, atol=0.1)
        if any(condition):
            starting_time = pos_car[condition].max()["time"]
            return starting_time
        else:
            min_speed = pos_car["speed"].min()
            return pos_car[pos_car["speed"]==min_speed].iloc[0]["time"]
        
        
def following_cars(data: pd.DataFrame, lane_data: pd.DataFrame,
                   meta_data: pd.DataFrame, use_xy=False, lanes: list = None,
                   classes : list = ["Car", "Van"]):
    """gets the following cars"""
    if not lanes:
        lanes = np.unique(lane_data["trackId"])
    times = np.sort(data["time"].unique())[:2]
    fps = np.rint(1 / (times[1] - times[0])).astype(int)
    track_ids = []
    trajectories = data.copy()
    for track_id, chunk in data.groupby(by=["trackId"]):
        lane_counter = Counter(chunk["lane"])
        dominant_lane = max(chunk["lane"], key=lane_counter.get)
        trajectories.loc[chunk.index, "dominantLane"] = dominant_lane
        if dominant_lane not in lanes:
            continue
        track_ids.append(track_id)
    trajectories = trajectories[trajectories["trackId"].isin(track_ids)]
    # get times as which most cars stop at the intersection
    redlight_peaks = intersection_peaks(trajectories, lane_data, lanes)
    condition = (
        (trajectories["time"].isin(redlight_peaks))
        & (trajectories["lane"].isin(lanes))
    )
    stop_frames = np.sort(trajectories[condition]["frame"].unique())
    frames_to_investigate = []
    ten_secs = 10 * fps
    for begin, stop in zip(stop_frames[:-1], stop_frames[1:]):
        # capture order every 10 seconds
        frames_to_investigate.extend(list(np.arange(begin, stop, ten_secs)))
    if len(frames_to_investigate) == 0:
        frames_to_investigate.append(0)
    condition = trajectories["frame"].isin(frames_to_investigate)
    redlight_situation = trajectories[condition]
    # identifiy pairs of leader follower at stopping positions
    pairs = pd.DataFrame()
    leaders = []
    for group_name, stopped_chunk in redlight_situation.groupby(by=["frame", "lane"]):
        sequence = stopped_chunk
        sequence["distanceIntersectionCrossing"] = np.abs(
            sequence["distanceIntersectionCrossing"])
        sequence = sequence.sort_values(by="distanceIntersectionCrossing",
                                        ascending=True)
        # if any(sequence["trackId"].isin(["109.0"])):
        #     print("debug")
        sequence = sequence[~sequence["trackId"].isin(leaders)]
        sequence = sequence[sequence["dominantLane"]==group_name[1]]
        sequence["follower"] = np.roll(sequence["trackId"].values, -1)
        sequence["classFollower"] = np.roll(sequence["class"].values, -1)
        sequence["order"] = np.arange((len(sequence)))
        sequence = sequence.iloc[:-1]
        leaders.extend(list(sequence["trackId"].values))
        pairs = pd.concat((pairs, sequence))
    situations = []
    for index, row in pairs.iterrows():
        situation = np.argmin(redlight_peaks - row["time"])
        situations.append(situation)
    pairs.reset_index(inplace=True)
    pairs["siutation"] = situations
    pairs = pairs.rename(columns={"trackId": "leader"})
    if len(pairs) == 0:
        return []
    pairs = pairs[pairs["class"].isin(classes)
                  & pairs["classFollower"].isin(classes)]
    drop_pairs = []
    speed_pairs = []
    l_n_straight = 0
    f_n_straight = 0
    i_crossing = 0
    between = 0
    not_stopping = 0
    dist_too_small = 0
    min_gap_small = 0
    for leader, chunk in pairs.groupby(by=["leader"]):
        chunk = chunk.iloc[0]
        follower = chunk["follower"]
        lane = chunk["dominantLane"]
        if chunk["intersectionCrossing"] == 0:
            drop_pairs.append(leader)
            i_crossing+=1
            continue
        recording_id = chunk["recordingId"]
        leader_chunk = data[data["trackId"]==leader]
        follower_chunk = data[data["trackId"]==follower]
        if not car_goes_straight(lane_data, follower_chunk):
            drop_pairs.append(leader)
            f_n_straight+=1
            continue
        if not car_goes_straight(lane_data, leader_chunk):
            l_n_straight+=1
            drop_pairs.append(leader)
            continue
        # accel_f = follower_chunk["acc"].values
        # accel_l = leader_chunk["acc"].values
        # if (np.any((accel_f[1:]-accel_f[:-1])>0.5)
        #     or np.any((accel_l[1:]-accel_l[:-1])>0.5)):
        #     drop_pairs.append(leader)
        #     accel_leaders.append(leader)
        #     continue
        intersection_crossing_f = follower_chunk.iloc[0]["intersectionCrossing"]
        intersection_crossing_l = chunk["intersectionCrossing"]
        # common_frames = np.intersect1d(leader_chunk["frame"],
        #                                 follower_chunk["frame"])
        condition = (
            (~data["trackId"].isin([leader, follower]))
            & (data["intersectionCrossing"].between(intersection_crossing_l,
                                                    intersection_crossing_f))
            & (data["lane"] == lane)
            & (data["time"] == intersection_crossing_l)
        )
        if np.any(condition):
            drop_pairs.append(leader)
            between += 1
            continue
        if not (any(np.isclose(leader_chunk["speed"], 0, atol=0.050))
                and any(np.isclose(follower_chunk["speed"], 0, atol=0.050))):
            drop_pairs.append(leader)
            speed_pairs.append(leader)
            not_stopping += 1
            continue
        leader_meta, follower_meta = _get_vehicle_meta_data(
            meta_data, (leader, follower, recording_id))
        pos_leader = trajectories[trajectories["trackId"]==leader]
        pos_follower = trajectories[trajectories["trackId"]==follower]
        starting_time = _get_starting_time(pos_leader)
        stopping_time = min(pos_leader["time"].values[-1],
                            pos_follower["time"].values[-1])
        leader_synced = pos_leader[
            pos_leader["time"].between(starting_time, stopping_time)]
        follower_synced = pos_follower[
            pos_follower["time"].between(starting_time, stopping_time)]
        distance = (_get_distance_df(leader_synced, follower_synced)
                - leader_meta["length"])
        if any(distance < 0.1):
            drop_pairs.append(leader)
            dist_too_small += 1
            continue
        pos_leader_ss = pos_leader[pos_leader["time"]==starting_time]
        pos_leader_ss = pos_leader_ss[["xCenter", "yCenter"]].values
        pos_follower_ss = follower_chunk[follower_chunk["time"]==starting_time]
        pos_follower_ss = pos_follower_ss[["xCenter", "yCenter"]].values
        min_gap = euclidian_distance(pos_leader_ss, pos_follower_ss)
        min_gap -= leader_meta["length"]
        # min_gap -= (leader_meta["length"] + follower_meta["length"]) / 2
        if not(leader_meta["class"] in classes
               and follower_meta["class"] in classes):
            drop_pairs.append(leader)
            continue
        if min_gap < 0.1:
            drop_pairs.append(leader)
            min_gap_small+=1
            continue
    # pairs = pairs[~pairs["leader"].isin(drop_pairs)]
    print("l_n_straight: ", l_n_straight)
    print("f_n_straight: ", f_n_straight)
    print("i_crossing: ", i_crossing)
    print("between: ", between)
    print("not_stopping: ", not_stopping)
    print("dist_too_small: ", dist_too_small)
    print("min_gap_small: ", min_gap_small)
    return pairs[["leader", "follower", "recordingId", "order"]]

for data, meta_data, lane_data in data_set.get_next_dataframes():
    pair_data = following_cars(data, lane_data, meta_data, False, [1,2])
    recording_id = pair_data.values[0,2]
    pair_data.to_csv(Path("/home/bookboi/Nextcloud/1_Docs/4_Master/20_Studienarbeit_paper/Kalibrierung_des_Extended_Intelligent_Driver_Model_mit_Drohnendaten/data") / f"{recording_id}_selection.csv", index=False)

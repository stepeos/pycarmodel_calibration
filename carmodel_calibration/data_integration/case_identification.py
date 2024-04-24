"""
module to identify car following case
-> a car turns into another road when initial_trajectory-final_trajectory > 30°
    -> discard such cars from data
-> cars with minimal distance from each other
-> drop cars, whichs' maximum speed<5km/h
"""
import numpy as np
import pandas as pd
from collections import Counter
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from carmodel_calibration.data_integration.helper import (
    get_lane, get_angle, get_difference)
from carmodel_calibration.helpers import _get_starting_time, _get_vehicle_meta_data


def _get_distance_df(leader_chunk: pd.DataFrame, follower_chunk: pd.DataFrame):
    pos_leader = leader_chunk[["xCenter", "yCenter"]].values
    pos_leader = np.atleast_2d(pos_leader)
    pos_follower = follower_chunk[["xCenter", "yCenter"]].values
    pos_follower = np.atleast_2d(pos_follower)
    return np.sqrt(np.sum((pos_leader-pos_follower)**2, axis=1).astype(np.float64))




def car_at_redlight(data: pd.DataFrame, lane_data: pd.DataFrame,
                    lanes = [1, 2]):
    """
    yields the ids of the cars, that wait at the red lights on lane 1 and 2
    """
    times = data["time"].unique()
    idxs = np.round(np.linspace(0,
                               len(times) - 1, 100)).astype(int)
    times = times[idxs]
    times = np.sort(times)
    cars_stopped = np.zeros_like(times)
    for idx, time in enumerate(times):
        conditions = (
            (data["time"]==time)
            & (data["speed"]==0)
            & (data["lane"].isin(lanes))
        )
        data_chunk = data[conditions]
        cars_stopped[idx] = data_chunk["trackId"].unique().shape[0]
    time_step = np.mean(times[1:]-times[:-1])
    time_dist = np.rint(30 / (time_step))
    idxs = find_peaks(cars_stopped, distance=time_dist)[0]
    if idxs.shape[0] == 0:
        # every ten seconds
        min_time = np.min(times)
        max_time = np.max(times)
        time_points = np.arange(min_time, max_time, 10)
        times = np.sort(data["time"].unique())
        idxs = np.array([np.argmin(np.abs(times - time_point))
                        for time_point in time_points])
        times = times[idxs]
    else:
        times = intersection_peaks(data, lane_data, lanes=None)
    scoped = data[
        (data["time"].isin(times))
        &(data["lane"]).isin(lanes)]
    free_leaders = []
    for time, time_chunk in scoped.groupby(by=["time"]):
        for _, lane_chunk in time_chunk.groupby(by=["lane"]):
            lane_chunk = lane_chunk[~(lane_chunk["class"].str.lower()=="two-wheeler")]
            distance_chunk = lane_chunk.copy()
            distance_chunk["distanceIntersectionCrossing"] = np.abs(
                distance_chunk["distanceIntersectionCrossing"])
            distance_chunk.sort_values(by="distanceIntersectionCrossing",
                                   inplace=True)
            # We collect the first car in the row standing at the the
            # intersection as free leader
            free_leader = distance_chunk.iloc[0]["trackId"]
            free_leaders.append(free_leader)
    free_leaders_data = data[data["trackId"].isin(free_leaders)]
    for track_id, track_chunk in free_leaders_data.groupby(by="trackId"):
        # Check if every leader fulfills the criteria like class, going staight
        if track_chunk["class"].iloc[0].lower() not in ["car", "van"]:
            free_leaders.remove(track_id)
            continue
        lane_vals = track_chunk["lane"].values
        if Counter(lane_vals).most_common()[0][0] not in lanes:
            free_leaders.remove(track_id)
            continue
        if not car_goes_straight(lane_data=lane_data, data_chunk=track_chunk):
            free_leaders.remove(track_id)
            continue
        distance_intersection = track_chunk[
                "distanceIntersectionCrossing"].values
        signchanges = ((np.roll(np.sign(distance_intersection), 1)
                        - np.sign(distance_intersection)) != 0).astype(int)
        signchanges[0] = 0
        if np.sum(signchanges) != 1:
            free_leaders.remove(track_id)
            continue
        starting_time = _get_starting_time(track_chunk)
        if track_chunk.iloc[-1]["time"] - starting_time < 5:
            free_leaders.remove(track_id)
            continue
    return free_leaders

def car_goes_straight(lane_data, data_chunk):
    """calculate distance (int) between cars from dataframe chunks"""
    if "lane" not in data_chunk.columns:
        data_chunk["lane"] = get_lane(data_chunk["lon"],
                                      data_chunk["lat"],
                                      lane_data)
    lane = data_chunk["lane"].values
    # TODO: instead of counter, integrate the lane over the traveled distance
    # for more accurate lane identification
    lane_counts = sorted(list(Counter(lane).values()), reverse=True)
    lane_mean = np.rint(np.mean(lane))
    lane_xy = (lane_data[lane_data["trackId"]==lane_mean]
               [["xCenter", "yCenter"]].values[::10])
    # xy_center = data_chunk[["xCenter", "yCenter"]].values[:-100:10]
    xy_center = data_chunk[["xCenter", "yCenter"]].values[::10]
    distance =  get_difference(xy_center[:,0], xy_center[:,1],
                              lane_xy[:,0], lane_xy[:,1])
    if any(distance > 15):
        return False
    if np.mean(distance) > 7:
        return False
    if len(lane_counts) < 2:
        return True
    # return True
    return np.sum(lane_counts[1:]) / lane_counts[0] < 0.1

def distance_between_cars(meta_data: pd.DataFrame,
                          car: int,
                          data_chunk: pd.DataFrame,
                          next_chunk: pd.DataFrame,
                          foll: int,
                          frame: int) -> int:
    """calculate distance between cars"""
    x_center1, y_center1 = data_chunk[["xCenter", "yCenter"]][
                    [data_chunk["frame"]==int(frame)]]
    x_center1, y_center1 = next_chunk[["xCenter", "yCenter"]][
                    [next_chunk["frame"]==int(frame)]]
    pos1, = np.array((x_center1, y_center1))
    pos2  = np.array((x_center1, y_center1))
    dist = euclidian_distance(pos1, pos2)
    len_foll = meta_data[["length"]][meta_data["trackId"]==foll]
    len_car = meta_data[["length"]][meta_data["trackId"]==car]
                # dist must consider car geometry
    dist = dist - sum(len_foll, len_car) / 2
    return dist

def angle_diff(traj_a, traj_b):
    """both trajectories in degree"""
    diff = traj_a - traj_b
    return (diff + 180) % 360 - 180

def euclidian_distance(pos1, pos2):
    """calculate the euclidian distance between 2points in n-dim space"""
    return np.sum(np.square(pos1-pos2), axis=1) ** 0.5


def following_cars(data: pd.DataFrame, lane_data: pd.DataFrame,
                   meta_data: pd.DataFrame, use_xy=False, lanes: list = None,
                   classes : list = ["car", "van"], traffic_light_time=60):
    """
    function that calculates the leader-follower pairs by their common features
    and the time that they cross the intersection
    # TODO: Improve this
    """
    classes = [c.lower() for c in classes]
    if not lanes:
        lanes = np.unique(lane_data["trackId"])
    # Filter all vehilces that are on the desired lanes
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
    
    
    regarded_times = None
    if traffic_light_time is None or traffic_light_time < 0:
        # TODO: Here we can implement an alternative way to calculate the pairs
        # which does not involve the traffic light time.
        # just sample the time points
        regarded_times = np.arange(data["time"].values.min(),
                                   data["time"].values.max(), 1)
    else:
        # get times as which most cars stop at the intersection
        redlight_peaks = intersection_peaks(
            trajectories, lane_data, lanes, traffic_light_time)
        regarded_times = redlight_peaks

    condition = (
        (trajectories["time"].isin(regarded_times))
        & (trajectories["lane"].isin(lanes))
    )
    stop_frames = np.sort(trajectories[condition]["frame"].unique())


    frames_to_investigate = []
    times = np.sort(data["time"].unique())[:2]
    fps = np.rint(1 / (times[1] - times[0])).astype(int)
    ten_secs = 5 * fps
    for begin, stop in zip(stop_frames[:-1], stop_frames[1:]):
        # capture order every 5 seconds
        frames_to_investigate.extend(list(np.arange(begin, stop, ten_secs)))
    if len(frames_to_investigate) == 0:
        frames_to_investigate.append(0)
    condition = trajectories["frame"].isin(frames_to_investigate)
    regarded_situation = trajectories[condition]
    # identifiy pairs of leader follower at stopping positions
    pairs = pd.DataFrame()
    leaders = []
    for group_name, stopped_chunk in regarded_situation.groupby(by=["frame", "lane"]):
        sequence = stopped_chunk
        sequence["distanceIntersectionCrossing"] = np.abs(
            sequence["distanceIntersectionCrossing"])
        sequence = sequence.sort_values(by="distanceIntersectionCrossing",
                                        ascending=True)
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
        situation = np.argmin(regarded_times - row["time"])
        situations.append(situation)
    pairs.reset_index(inplace=True)
    pairs["situation"] = situations
    pairs = pairs.rename(columns={"trackId": "leader"})
    if len(pairs) == 0:
        return []
    pairs = pairs[pairs["class"].str.lower().isin(classes)
                  & pairs["classFollower"].str.lower().isin(classes)]
    drop_reasons = {
        "intersection_crossing_zero": [],
        "not_going_straight_leader": [],
        "not_going_straight_follower": [],
        "signchanges_not_one": [],
    }
    drop_pairs = []
    for leader, chunk in pairs.groupby(by=["leader"]):
        chunk = chunk.iloc[0]
        follower = chunk["follower"]
        lane = chunk["dominantLane"]
        if chunk["intersectionCrossing"] == 0:
            drop_pairs.append(leader)
            drop_reasons["intersection_crossing_zero"].append(leader)
            continue
        recording_id = chunk["recordingId"]
        leader_chunk = data[data["trackId"]==leader]
        follower_chunk = data[data["trackId"]==follower]
        if not car_goes_straight(lane_data, follower_chunk):
            drop_pairs.append(leader)
            drop_reasons["not_going_straight_follower"].append(follower)
            continue
        if not car_goes_straight(lane_data, leader_chunk):
            drop_reasons["not_going_straight_leader"].append(leader)
            drop_pairs.append(leader)
            continue
        for chunk in [leader_chunk, follower_chunk]:
            distance_intersection = leader_chunk[
                "distanceIntersectionCrossing"].values
            signchanges = ((np.roll(np.sign(distance_intersection), 1)
                            - np.sign(distance_intersection)) != 0).astype(int)
            signchanges[0] = 0
            if np.sum(signchanges) != 1:
                drop_pairs.append(leader)
                drop_reasons["signchanges_not_one"].append(leader)
                continue
        # accel_f = follower_chunk["acc"].values
        # accel_l = leader_chunk["acc"].values
        # if (np.any((accel_f[1:]-accel_f[:-1])>0.5)
        #     or np.any((accel_l[1:]-accel_l[:-1])>0.5)):
        #     drop_pairs.append(leader)
        #     accel_leaders.append(leader)
        #     continue
        intersection_crossing_f = follower_chunk.iloc[0]["intersectionCrossing"]
        intersection_crossing_l = chunk.iloc[0]["intersectionCrossing"]
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
            continue
        if not (any(np.isclose(leader_chunk["speed"], 0, atol=0.050))
                and any(np.isclose(follower_chunk["speed"], 0, atol=0.050))):
            drop_pairs.append(leader)
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
        # at least 5 seconds after drive-off
        if stopping_time - starting_time < 5:
            drop_pairs.append(leader)
            continue
        distance = (_get_distance_df(leader_synced, follower_synced)
                    - leader_meta["length"])
        if any(distance < 0.1):
            drop_pairs.append(leader)
            continue
        pos_leader_ss = pos_leader[pos_leader["time"]==starting_time]
        pos_leader_ss = pos_leader_ss[["xCenter", "yCenter"]].values
        pos_follower_ss = follower_chunk[follower_chunk["time"]==starting_time]
        pos_follower_ss = pos_follower_ss[["xCenter", "yCenter"]].values
        min_gap = euclidian_distance(pos_leader_ss, pos_follower_ss)
        min_gap -= leader_meta["length"]
        # min_gap -= (leader_meta["length"] + follower_meta["length"]) / 2
        if not(leader_meta["class"].lower() in classes
               and follower_meta["class"].lower() in classes):
            drop_pairs.append(leader)
            continue
        if min_gap < 0.1:
            drop_pairs.append(leader)
            continue
    pairs = pairs[~pairs["leader"].isin(drop_pairs)]
    return pairs[["leader", "follower"]].values


def intersection_peaks(trajectories, lane_data, lanes=None,
                       traffic_light_time=60):
    """find peaks of when cars stop on lanes at the intersection"""
    if lanes:
        selected_lanes = lanes
    else:
        selected_lanes = np.unique(lane_data["trackId"])
    conditions = (
        (trajectories["lane"].isin(selected_lanes))
        & (trajectories["speed"]<=0.5)
    )
    sorted_trajectories = trajectories[conditions].sort_values(by="time")
    cars_stopped = []
    times = []
    for time, time_chunk in sorted_trajectories.groupby(by="time"):
        times.append(time)
        cars_stopped.append(len(time_chunk))
    if len(times) == 1:
        times.append(max(trajectories["time"]))
        cars_stopped.append(0)
    interpolator = interp1d(times, cars_stopped)
    new_num = (max(times) - min(times)) // 0.5
    new_times = np.linspace(min(times), max(times), num=int(new_num))
    new_cars_stopped = interpolator(new_times)
    time_step = np.mean(new_times[1:]-new_times[:-1])
    # we search for peaks every 60 seconds because of 60seconds traffic light
    # period
    time_dist = np.rint(traffic_light_time / (time_step))
    idxs = find_peaks(new_cars_stopped, distance=time_dist)[0]
    if idxs.shape[0] == 0:
        idxs = [0]
    time_points = new_times[idxs]
    # since time was scaled, we need to find the closest time points which
    # exist in the original data
    times = np.array(times)
    new_time_points = []
    for time_point in time_points:
        closest_idx = np.argmin(np.abs(times-time_point))
        new_time_points.append(times[closest_idx])
    return np.array(new_time_points)
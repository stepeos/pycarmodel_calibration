"""file that contains arbitrary helper functions used within the package"""

import logging

import numpy as np
import pandas as pd
import scipy.optimize as op
from scipy.signal import find_peaks
from carmodel_calibration.fileaccess.parameter import ModelEnum

_LOGGER = logging.getLogger(__name__)


def _get_starting_time(pos_car):
    starting_index = np.argwhere(pos_car["speed"].values==0)
    if starting_index.shape[0] != 0:
        starting_index = starting_index[-1,0] + 1
    else:
        return pos_car["time"].values[0]
    starting_time = pos_car["time"].values[starting_index]
    return starting_time

def _get_vehicle_meta_data(meta_data: pd.DataFrame,
                           identification: tuple) -> tuple:
    leader = identification[0]
    follower = identification[1]
    recording_id = identification[2]
    conditions = ((meta_data["trackId"]==leader)
                  & (meta_data["recordingId"]==recording_id))
    leader_data = meta_data[conditions].iloc[0].to_dict()
    conditions = ((meta_data["trackId"]==follower)
                  & (meta_data["recordingId"]==recording_id))
    if not any(conditions):
        return leader_data, None
    follower_data = meta_data[conditions].iloc[0].to_dict()
    return leader_data, follower_data

def _get_starting_time(pos_car_in):
    speed_threshold = 0.2
    pos_car = pos_car_in.copy()
    pos_car["speed"] = pos_car["speed"].rolling(window=5).mean()
    starting_indexes = np.argwhere(pos_car["speed"].values==0)
    if starting_indexes.shape[0] != 0:
        starting_index = starting_indexes[-1,0]
        if starting_index == pos_car["speed"].values.shape[0] - 1:
            return pos_car["time"].values[starting_index]
        starting_index = np.clip(starting_indexes[-1,0] + 1, 0,
                                 pos_car.values.shape[0]-1)
        driven = np.argwhere(
            pos_car["speed"].values[starting_index:] > speed_threshold)
        driveaway_index = max(0, driven[0,0]-1+starting_index)
        speed_chunk = pos_car["speed"].values[starting_index:driveaway_index]
        peaks = find_peaks(speed_chunk, 0.1)[0]
        if peaks.shape[0] > 0:
            peaks += starting_index
            driveaway_index = (np.argmin(
                pos_car["speed"].values[peaks[-1]+1:driveaway_index])
            + peaks[-1] + 1)
        else:
            time_frame = (pos_car["speed"].values[driveaway_index]
                          - pos_car["speed"].values[starting_index])
            condition = (
                pos_car["speed"].values[starting_index+1:driveaway_index]
                > pos_car["speed"].values[starting_index:driveaway_index-1])
            if np.all(condition) and time_frame < 0.5:
                driveaway_index = starting_index
        driveaway = pos_car["time"].iloc[driveaway_index]
        return driveaway
    else:
        condition = np.isclose(pos_car["speed"].values, 0, atol=0.05)
        if any(condition):
            starting_time = pos_car[condition].max()["time"]
            return starting_time
        else:
            min_speed = pos_car["speed"].min()
            return pos_car[pos_car["speed"]==min_speed].iloc[0]["time"]

def _estimate_parameters(identification: tuple,
                              data_chunk: pd.DataFrame,
                              meta_data: pd.DataFrame,
                              model):
    leader_chunk = data_chunk[data_chunk["trackId"]==identification[0]]
    follower_chunk = data_chunk[data_chunk["trackId"]==identification[1]]
    # estimate follower parameters
    starting_time_f = _get_starting_time(follower_chunk)
    follower_start = follower_chunk[follower_chunk["time"]>starting_time_f]
    follower_start.reset_index(drop=True, inplace=True)
    follower_time = follower_start["time"].values
    taccmax_f, m_beg, m_flat = _estimate_acceleration_curve(
        starting_time_f, follower_start, follower_time)
    taccmax_f += 0.8 # see EIDM paper by Salles D.
    # TODO: hardcoded speed limit 50km/h
    speed_factor_follower =  follower_chunk["speed"].max() / 13.8889
    if identification[0] == "":
        return None, taccmax_f, m_beg, m_flat, speed_factor_follower, None
    starting_time_l = _get_starting_time(leader_chunk)

    adjusted_t0 = np.clip(starting_time_l, follower_chunk["time"].min(),
                          follower_chunk["time"].max())
    pos_follower = follower_chunk[follower_chunk["time"]==adjusted_t0]
    pos_leader = leader_chunk[leader_chunk["time"]==adjusted_t0]
    leader_meta, follower_meta = _get_vehicle_meta_data(
        meta_data, identification)
    len_l, len_f = leader_meta["length"], follower_meta["length"]
    pos_follower_xy = pos_follower[["xCenter", "yCenter"]].values[0]
    pos_leader_xy = pos_leader[["xCenter", "yCenter"]].values[0]
    # need to subtract 0.2 when using the EIDM, because the EIDM-SUMO-Code uses an additional 0.25m as "safeguard" 
    if model == "eidm":
        min_gap_subtract = 0.2
    else:
        min_gap_subtract = 0
    min_gap = (np.sqrt(np.sum(np.square(pos_leader_xy-pos_follower_xy)))
               - len_l - min_gap_subtract)
    starting_time_f = _get_starting_time(follower_chunk)
    if model == "idm" or model == "krauss":
        startup_delay = 0.0001
    else:
        startup_delay = np.clip((starting_time_f - adjusted_t0 - 0.25),
                            0, 2)
    follower_start = follower_chunk[
        follower_chunk["time"]>=starting_time_f-0.8]
    speed_factor_follower =  follower_chunk["speed"].max() / 13.8889
    follower_start.reset_index(drop=True, inplace=True)
    follower_time = follower_start["time"].values
    taccmax_f, m_beg, m_flat = _estimate_acceleration_curve(
        starting_time_f, follower_start, follower_time)
    params = (min_gap, taccmax_f, m_beg, m_flat, speed_factor_follower,
            startup_delay)
    return params

def _estimate_acceleration_curve(starting_time_f, follower_start,
                                 follower_time):
    local_maxima_i = find_peaks(
        follower_start["acc"].values, threshold=0.25)[0]
    taccmax_idx_f = np.argmax(follower_start["acc"].values)
    if local_maxima_i.shape[0] == 0:
        taccmax_idx_f = np.argmax(follower_start["acc"].values)
    else:
        taccmax_idx_f = next(
            x for x in local_maxima_i if follower_start["acc"].values[x] > 1)
    taccmax_dur_f = follower_time[taccmax_idx_f] - follower_time[0]
    follower_t_acc = follower_start[["time", "acc"]].values[:taccmax_idx_f+1]
    curve = _acceleration_curve_factory(taccmax_dur_f, starting_time_f, False)
    solutions = ()
    try:
        solutions = op.curve_fit(
            curve,
            follower_t_acc[:,0],
            follower_t_acc[:,1] / follower_t_acc[-1,1])
    except RuntimeError:
        # could not estimate parameters
        pass
    m_beg, m_flat = None, None
    for solution in solutions:
        if np.all(solution != np.inf):
            m_beg, m_flat = solution
            # m_beg, m_flat = solution
            break
    return taccmax_dur_f, m_beg, m_flat


def _acceleration_curve_factory(taccmax_dur_f, t_off, y_off=False):
    def _acceleration_curve(time, m_beg, m_flat):
        # (tanh( ( (time - t_off) * 2 / taccmax_dur_f - m_beg) * m_flat) + 1) / 2
        result = (np.tanh( ( (time - t_off) * 2 / taccmax_dur_f - m_beg)
                          * m_flat) + 1) / 2
        # if t[0] == t_off then:
        # (tanh( ( time * 2 / taccmax_dur_f - m_beg) * m_flat) + 1) / 2
        # result = (np.tanh( ( time * 2 / taccmax_dur_f - m_beg) * m_flat) + 1) / 2
        return result
    def _acceleration_curve_yoff(time, m_beg, m_flat, y_off):
        # (tanh( ( (time - t_off) * 2 / taccmax_dur_f - m_beg) * m_flat) + 1) / 2
        result = (np.tanh( ( (time - t_off) * 2 / taccmax_dur_f - m_beg)
                          * m_flat) + 1) / 2 + y_off
        # if t[0] == t_off then:
        # (tanh( ( time * 2 / taccmax_dur_f - m_beg) * m_flat) + 1) / 2
        # result = (np.tanh( ( time * 2 / taccmax_dur_f - m_beg) * m_flat) + 1) / 2
        return result
    if y_off:
        return _acceleration_curve_yoff
    else:
        return _acceleration_curve

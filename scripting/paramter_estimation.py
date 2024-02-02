import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import scipy.optimize as op
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from carmodel_calibration.helpers import (_get_vehicle_meta_data, _acceleration_curve_factory,
                                      _get_starting_time)
from carmodel_calibration.data_integration.case_identification import following_cars
from carmodel_calibration.data_integration.data_set import DataSet
rc_file = Path(__file__).parents[1] / "carmodel_calibration/data_config/matplotlib.rc"
plt.style.use(rc_file)
data_set = DataSet(Path(os.environ["DATA_DIR"]))
data_set.interactive_plot(2)
# data, meta_data, lane_data = data_set.get_dataframes_by_id("05")
# to investigate:
# 1040.0 1047.0 5 Datenfehler: Fzg verschwindet und aus 1047 wird 1046.0
# 277.0 287.0.0 5 stimmt so
#  "" 480 5 bumpy start
# 654.0 637.0 2 -> stimmt so
# 941 947 3
# 1064 1067 3 distance springt
# 68.0 71.0 2 distane springt -> datenfehler

dtypes = {"leader": str, "follower": str}
# data.to_csv(".tmp/data.csv")
data = pd.read_csv(".tmp/data.csv", dtype=dtypes)
data["trackId"] = data["trackId"].astype(str)
# meta_data.to_csv(".tmp/meta_data.csv")
meta_data = pd.read_csv(".tmp/meta_data.csv", dtype=dtypes)
meta_data["trackId"] = meta_data["trackId"].astype(str)
# lane_data.to_csv(".tmp/lane_data.csv")
lane_data = pd.read_csv(".tmp/lane_data.csv")



# def _get_starting_time(pos_car):
#     starting_indexes = np.argwhere(pos_car["speed"].values==0)
#     if starting_indexes.shape[0] != 0:
#         starting_index = starting_indexes[-1,0]
#         # starting_index = np.clip(starting_indexes[-1,0] + 1, 0,
#         #                          pos_car.values.shape[0]-1)
#         return pos_car["time"].iloc[starting_index]
#     else:
#         condition = np.isclose(pos_car["speed"].values, 0, atol=0.1)
#         if any(condition):
#             starting_time = pos_car[condition].max()["time"]
#             return starting_time
#         else:
#             min_speed = pos_car["speed"].min()
#             return pos_car[pos_car["speed"]==min_speed].iloc[0]["time"]

def _estimate_parameters(identification: tuple,
                              data_chunk: pd.DataFrame,
                              meta_data: pd.DataFrame):
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
    min_gap = (np.sqrt(np.sum(np.square(pos_leader_xy-pos_follower_xy)))
               - (len_l + len_f) / 2) + 0.1
    starting_time_f = _get_starting_time(follower_chunk)
    startup_delay = np.clip((starting_time_f - adjusted_t0 - 0.25),
                            0, 2)
    follower_start = follower_chunk[
        follower_chunk["time"]>=starting_time_f-0.8]
    speed_factor_follower =  follower_chunk["speed"].max() / 13.8889
    follower_start.reset_index(drop=True, inplace=True)
    follower_time = follower_start["time"].values
    taccmax_f, m_beg, m_flat = _estimate_acceleration_curve(starting_time_f, follower_start, follower_time)
    params = (min_gap, taccmax_f, m_beg, m_flat, speed_factor_follower,
            startup_delay)
    # import matplotlib.pyplot as plt
    # ax = leader_chunk.plot(x="time", y="speed")
    # follower_chunk.plot(x="time", y="speed", ax=ax)
    # ax.scatter(starting_time_l, leader_chunk[leader_chunk["time"]==starting_time_l]["speed"].values[0])
    # ax.scatter(starting_time_f, follower_chunk[follower_chunk["time"]==starting_time_f]["speed"].values[0])
    # plt.show()
    return params

def _estimate_acceleration_curve(starting_time_f, follower_start, follower_time):
    local_maxima_i = find_peaks(
        follower_start["acc"].values)[0]
    taccmax_idx_f = np.argmax(follower_start["acc"].values)
    if local_maxima_i.shape[0] == 0:
        taccmax_idx_f = np.argmax(follower_start["acc"].values)
    else:
        taccmax_idx_f = next(
            x for x in local_maxima_i if follower_start["acc"].values[x] > 1)
    taccmax_f = follower_start["acc"].values[taccmax_idx_f]
    # taccmax_f += 0.8 # see EIDM paper by Salles D. # not needed with the new
    # files
    taccmax_dur_f = follower_time[taccmax_idx_f] - follower_time[0]
    follower_t_acc = follower_start[["time", "acc"]].values[:taccmax_idx_f+1]
    curve = _acceleration_curve_factory(taccmax_dur_f, starting_time_f, False)
    solutions = ()
    try:
        solutions = op.curve_fit(
            curve,
            follower_t_acc[:,0],
            follower_t_acc[:,1] / follower_t_acc[-1,1],
            # bounds=((0.7, 1.5), (1,5), (-100, 100))
            )
    except RuntimeError:
        # could not estimate parameters
        pass
    m_beg, m_flat = None, None
    for solution in solutions:
        if np.all(solution != np.inf):
            m_beg, m_flat = solution
            # m_beg, m_flat = solution
            break
    return taccmax_f, m_beg, m_flat

selection = following_cars(data, lane_data, meta_data, False, [1,2])
figures = []
for identification in selection:
    identification = np.insert(identification, identification.shape[0], meta_data["recordingId"].values[0])
    identification = ['887.0', '909.0', 2]
    _estimate_parameters(identification, data, meta_data)
    # figures.append(figure)
plt.close("all")
filepath = Path(__file__).parents[1]/ ".tmp/estimation.pdf"
pdf = matplotlib.backends.backend_pdf.PdfPages(
        filepath)
for figure in figures:
    pdf.savefig(figure)
pdf.close()

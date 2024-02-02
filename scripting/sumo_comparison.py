import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from carmodel_calibration.data_integration.data_set import DataSet
from carmodel_calibration.sumo.simulation_module import SumoInterface
from carmodel_calibration.sumo.sumo_project import SumoProject
from carmodel_calibration.fileaccess.parameter import EidmParameters
from carmodel_calibration.helpers import _get_starting_time
from carmodel_calibration.data_integration.case_identification import following_cars
def _get_distance_df(leader_chunk: pd.DataFrame, follower_chunk: pd.DataFrame):
    pos_leader = leader_chunk[["xCenter", "yCenter"]].values
    pos_leader = np.atleast_2d(pos_leader)
    pos_follower = follower_chunk[["xCenter", "yCenter"]].values
    pos_follower = np.atleast_2d(pos_follower)
    return np.sqrt(np.sum((pos_leader-pos_follower)**2, axis=1).astype(np.float64))

DATA_DIR = os.environ["DATA_DIR"]
data_path = Path(DATA_DIR)
identification = ['654.0', '637.0', 2]
identification = ['56.0', '60.0', 2]
identification = ['651.0', '653.0', 4]

# data_set = DataSet(data_path)
# data, meta_data, lane_data = data_set.get_dataframes_by_id(str(identification[2]))
# pairs = following_cars(data, lane_data, meta_data, lanes=[1,2])










project_dir = Path(".tmp/sumo_test")
params = {'speedFactor': 1.1822782725642353, 'minGap': 0.7795200105686237, 'accel': 3.4600572003620442, 'decel': 1.7495062578325644, 'emergencyDecel': 15, 'startupDelay': 0.1946886633177909, 'tau': 0.5707546714518943, 'delta': 2.8145495876579902, 'stepping': 0.25, 'tpreview': 4, 'tPersDrive': 3, 'tPersEstimate': 10, 'treaction': 0.5742938422375167, 'ccoolness': 0.99, 'sigmaleader': 0.0001, 'sigmagap': 0.0001, 'sigmaerror': 0.0001, 'jerkmax': 3, 'epsilonacc': 1, 'taccmax': 0.5479972668964113, 'Mflatness': 3.2858975316298378, 'Mbegin': 0.22611948527999615}
SumoProject.create_sumo(project_dir, 20)
params = {'speedFactor': 1.1822782725642353, 'minGap': 0.7795200105686237, 'accel': 3.4600572003620442, 'decel': 1.7495062578325644, 'emergencyDecel': 15, 'startupDelay': 0.1946886633177909, 'tau': 0.5707546714518943, 'delta': 2.8145495876579902, 'stepping': 0.25, 'tpreview': 4, 'tPersDrive': 3, 'tPersEstimate': 10, 'treaction': 0.5742938422375167, 'ccoolness': 0.99, 'sigmaleader': 0.0001, 'sigmagap': 0.0001, 'sigmaerror': 0.0001, 'jerkmax': 3, 'epsilonacc': 1, 'taccmax': 0.5479972668964113, 'Mflatness': 3.2858975316298378, 'Mbegin': 0.22611948527999615}
eidm = EidmParameters.create_eidm_parameter_set(".tmp/test.json", **params)
SumoProject.write_followers_leader(Path(".tmp/sumo_test/calibration_routes.rou.xml"), [eidm]*20)
sumo = SumoInterface(project_dir,
                        Path(os.environ["DATA_DIR"]), gui=False)
dtypes = {"ID": str}
data = pd.read_csv(data_path / f"{identification[2]:02d}_WorldPositions.csv",dtype=dtypes)
data["ID"] = data["ID"].astype(float).astype(str)
data = data.rename(columns={"Time": "time", "ID": "trackId", "LocalX": "xCenter", "LocalY": "yCenter", "Speed": "speed", "Length": "length"})
leader = data[data["trackId"]==identification[0]]
follower = data[data["trackId"]==identification[1]]
len_l = leader.iloc[0]["length"]
starting_time = _get_starting_time(leader)
starting_time = min(starting_time, _get_starting_time(follower))
leader = leader[leader["time"]>=starting_time]
follower = follower[follower["time"]>=starting_time]
max_time = min(leader["time"].iloc[-1], follower["time"].iloc[-1])
leader = leader[leader["time"]<=max_time]
follower = follower[follower["time"]<=max_time]

distance = _get_distance_df(leader, follower) - len_l
# covered_distance_l = _get_distance_df(
#                 leader.iloc[1:], leader.iloc[:-1])
# accumulated_distance_l = np.cumsum(
#     np.insert(covered_distance_l, 0, 0, axis=0))
# covered_distance_f = _get_distance_df(
#     follower.iloc[1:], follower.iloc[:-1])
# covered_distance_f = np.abs(follower["xCenter"].iloc[1:].values - follower["xCenter"].iloc[:-1].values)
# accumulated_distance_f = np.cumsum(
#                 np.insert(covered_distance_f, 0, 0, axis=0))
# distance2 = accumulated_distance_l - accumulated_distance_f + distance[0]
simulation_reults = sumo.run_simulation(identification)
pred = simulation_reults[1]["prediction"]
gt = simulation_reults[1]["ground_truth"]
gt_item = gt[gt["counter"]==0]
item = pred[pred["counter"]==0]
# plt.plot(item["time"].values, item["distance"].values)
plt.plot(leader["time"].values-leader["time"].values[0], distance)
plt.plot(gt_item["time"].values[7:] - gt_item["time"].values[7], gt_item["distance"].values[7:])
plt.show()

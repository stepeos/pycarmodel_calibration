"""
conversion helpers to manipulate dataframes
"""
import numpy as np
import pandas as pd

def moving_average(data: pd.DataFrame, n_average: int):
    """applies moving average filter to a dataframe or array"""
    if isinstance(data, pd.DataFrame):
        return data.apply(np.convolve, aixs = 0, raw=True,
            args=(np.ones(n_average)/n_average, ), mode='valid')
    elif isinstance(data, np.ndarray):
        return np.convolve(data, np.ones(n_average)/n_average, mode='valid')

def get_difference(x, y, lane_x, lane_y):
    """calculate distance between lane trajectory data and xy traj"""
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    # if min(x) < min(lane_x) or max(x) > max(lane_x) or lane_x.shape[0] < 100:
    #     # in case that extrapolation is required, or lane has low resolution
    #     f1 = ip.interp1d(lane_x.copy(), lane_y, fill_value="extrapolate")
    #     lane_x = x[:,0]
    #     lane_y = f1(x[:,0])
    lane_xy = np.hstack((lane_x[:, np.newaxis], lane_y[:, np.newaxis]))
    xy_stacked = np.repeat(
        np.hstack((x.reshape((1,1,-1)),y.reshape((1,1,-1)))),
        lane_xy.shape[0], axis=0)
    lane_xy_stacked = np.tile(lane_xy[:,:,np.newaxis], x.shape[0])
    vec = lane_xy_stacked - xy_stacked
    eucl_dist = np.sqrt(vec[:,0,:]**2 + vec[:,1,:]**2)
    dist = np.min(eucl_dist, axis=0)
    # FIXME: actually, dist must be
    # dist = np.abs(np.cross(pt2-pt1, pt1-coords)) / np.linalg.norm(pt2-pt1)
    # where pt1 and pt2 are the two closest points to the coords [x:, y:]
    return dist

def get_lane(lon: np.ndarray, lat:np.ndarray, lanes: pd.DataFrame,
             use_xy=False):
    """
    returns the closest lane for every x-y pair
    :param lon:         array with longitude values
    :param lat:         array with latitude values
    :param  lanes:      colums [ID, lat, lon]
    :param use_xy:      uses xy data instead
                        (lon has to be x and lat has to be y)
    :ret:               returns the closest lane ID for every lon-lat pair
    """
    abscissa = "xCenter" if use_xy else "lon"
    ordinate = "yCenter" if use_xy else "lat"
    lane_id = np.zeros((lon.shape[0],lanes["trackId"].unique().shape[0]))
    for idx, identifier in enumerate(np.sort(lanes["trackId"].unique())):
        trajectory_data = lanes[lanes["trackId"]==identifier]
        lane_lon = trajectory_data[abscissa].values
        lane_lat = trajectory_data[ordinate].values
        fit = get_difference(lon, lat, lane_lon, lane_lat)
        lane_id[:, idx] = fit
    replacements = {}
    for idx in range(lane_id.shape[1]):
        replacements[idx] = np.sort(lanes["trackId"].unique())[idx]
    lane_id = np.argmin(lane_id, axis=1)
    # remap from 0 ... len(lanes_id) to lane identifier
    # to enable 0, 2 , 5 as possible lanes
    for current, repl in replacements.items():
        lane_id[lane_id == current] = repl
    return lane_id

def get_heading(x_pos: np.ndarray, y_pos: np.ndarray) -> np.ndarray:
    """
    returns heading for each datapoint
    TODO:adjusted with moving average
    with north aligned to the x-axis
    :ret:           array with heading in degrees
    """
    if len(x_pos) < 2:
        return 0
    attack_vector_dx = x_pos[1:] - x_pos[:-1]
    attack_vector_dy = y_pos[1:] - y_pos[:-1]
    mask = (np.sqrt(np.square(attack_vector_dx) + np.square(attack_vector_dy))<0.05)
    for index in range(attack_vector_dx.shape[0], 0, -1):
        if mask[index-1]:
            new_index = np.clip(index, 0, attack_vector_dx.shape[0]-1, dtype=int)
            attack_vector_dx[index-1] = attack_vector_dx[new_index]
            attack_vector_dy[index-1] = attack_vector_dy[new_index]
    attack_vector = np.hstack((attack_vector_dx[:,np.newaxis],
                               attack_vector_dy[:,np.newaxis]))
    attack_vector_angle = get_angle(attack_vector)
    attack_vector_angle = np.hstack((attack_vector_angle[0],
                                     attack_vector_angle)).astype(np.float32)
    return attack_vector_angle

def get_angle(attack_vector: np.ndarray) -> np.ndarray:
    """
    :param attack_vector:       [dx, dy] of shape (N, 2)
    :ret:                       angle 0...360 degrees with shape (N,)
    """
    attack_vector_angle = np.degrees(np.arctan2(attack_vector[:,0],
                                                attack_vector[:,1]))
    attack_vector_angle -= 90
    attack_vector_angle = attack_vector_angle % 360
    return attack_vector_angle

def get_frame(time, fps):
    return np.rint(time * fps).astype(int)

def get_crossing_time(lon_lat: np.ndarray,
                      points: np.ndarray, time: np.ndarray):
    """returns time at which object crosses line"""
    pt1, pt2 = points[:2], points[2:]
    cross = np.cross(pt2-pt1, pt1-lon_lat)
    if np.unique(np.sign(cross)).shape[0] == 1:
        return 0
    dist = np.abs(cross) / np.linalg.norm(pt2-pt1)
    low, up = np.argsort(dist)[:2]
    return np.mean((time[low],time[up]))

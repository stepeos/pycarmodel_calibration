"""module to integrate data files"""

import glob
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from calibration_tool.data_integration.helper import (get_frame, get_heading,
                                                      get_lane, moving_average)
from calibration_tool.data_integration.integrator import Integrator
from calibration_tool.fileaccess.configs import JSON
from calibration_tool.helpers import _get_vehicle_meta_data

warnings.filterwarnings('ignore', message="^Columns.*")
_LOGGER = logging.getLogger(__name__)

class DataSet(Integrator):
    """class to handle data files in a directory to import data and integrate
    it into common database format"""

    def __init__(self, directory):
        super().__init__(directory)
        directory = Path(directory)
        self._config_file = \
            JSON(self._directory.absolute() / "data_config.json")
        self._config_file.load_values()
        self._read_dataset_config()

    def _get_metafile(self, data_file: Path):
        meta_file = Path(data_file).parent / data_file.name.replace(
            self._metafile_replace[0], self._metafile_replace[1])
        if not meta_file.exists():
            data = pd.read_csv(data_file, **self._df_kwargs)
            data = self._update_metadata(data)
            meta_items = []
            for track_id, chunk in data.groupby(by=["trackId"]):
                row = chunk.iloc[0]
                recording_id = int(float((meta_file.name).split("_")[0]))
                meta_items.append({
                    "trackId": track_id,
                    "length": row["length"],
                    "width": row["width"],
                    "class": row["class"],
                    "recordingId": recording_id})
            meta_data = pd.DataFrame(meta_items)
            return meta_data
        try:
            meta_data = pd.read_csv(meta_file)
            assert len(meta_data.index) > 0
        except AssertionError as ex:
            message = f"Metadatafile {meta_file} is of length 0"
            _LOGGER.error(message)
            raise AssertionError(message) from ex
        except Exception as ex:
            raise ex
        return meta_data

    def _essential_preprocessing(self, data, meta_data, options=None):
        # might not want ot drop all na rows, since they still contain
        # trajectory data
        data.dropna(inplace=True)
        options = options or {}
        recording_id = meta_data.iloc[0]["recordingId"]
        data.loc[:, "recordingId"] = recording_id
        drop_times = options.get("disregard_time") or []
        for drop_time in drop_times:
            drops = data[data["time"].between(float(drop_time[0]),
                                               float(drop_time[1]))]
            data.drop(drops.index, inplace=True)
        for _, value in self._df_config.items():
            column = value.get("new_name")
            ops = value.get("operations")
            if not isinstance(ops, list):
                continue
            if len(ops) == 0:
                continue
            for operation, args in zip(ops, value.get("operations_args")):
                if operation == "moving_average":
                    moving_average(data[[column]], args)
        if "heading" not in data.columns:
            data["heading"] = None
            data["heading"] = data["heading"].astype(float)
            for track_id in data["trackId"].unique():
                heading = get_heading(
                    data[data["trackId"]==track_id]["xCenter"].values,
                    data[data["trackId"]==track_id]["yCenter"].values)
                try:
                    data.loc[data["trackId"]==track_id, "heading"] = heading
                except ValueError as exc:
                    _LOGGER.error(exc)
                    raise exc
        if "time" not in data.columns and "frame" not in data.columns:
            raise ValueError("data does not contain time nor frame")
        if "time" not in data.columns:
            data["time"] = (data["frame"].values
                            / self._config_file.get_value("fps"))
        if "speed" not in data.columns:
            # TODO: implement speed calculation
            data["speed"] = None
            for track_id in data["trackId"].unique():
                data_chunk = data[data["trackId"]==track_id]
                dx = data_chunk["xCenter"].diff()
                dy = data_chunk["yCenter"].diff() 
                delta = np.sqrt(dx**2 + dy**2)
                time_diff = data_chunk["time"].diff()
                data_chunk.loc[:, "speed"] = delta / time_diff
                window_size = 3
                data_chunk.loc[:, "speed"] = data_chunk["speed"].rolling(
                    window=window_size, center=True).mean()
                data_chunk.loc[:, "speed"] = data_chunk["speed"].fillna(
                    method='bfill')
                data_chunk.loc[:, "speed"] = data_chunk["speed"].fillna(
                    method='ffill')
                data.loc[data["trackId"]==track_id, "speed"] = (
                    data_chunk["speed"].fillna(
                    method='ffill'))
            # change to numeric
            data["speed"] = pd.to_numeric(data["speed"], errors='coerce')
        if "class" not in data.columns:
            data["class"] = None
            for track_id in data["trackId"].unique():
                track_class = (
                    meta_data[meta_data["trackId"]==track_id]["class"].iloc[0])
                data.loc[data["trackId"]==track_id, "class"] = \
                    track_class
        if "frame" not in data.columns:
            data["frame"] = get_frame(
                    data["time"].values,
                    self._config_file.get_value("fps"))
        return data

    def _optional_preprocessing(self, data, meta_data, options=None):
        # might not want ot drop all na rows, since they still contain
        # trajectory data
        data.dropna(inplace=True)
        options = options or {}
        coordinate_system = options.get("coordinate_system") or ["lon", "lat"]
        abscissa = coordinate_system[0]
        ordinate = coordinate_system[1]
        lane_data = self.get_lane_df(None)
        recording_id = meta_data.iloc[0]["recordingId"]
        data.loc[:, "recordingId"] = recording_id
        if coordinate_system == ["xCenter", "yCenter"]:
            use_xy = True
        else:
            use_xy = False
        drop_times = options.get("disregard_time") or []
        for drop_time in drop_times:
            drops = data[data["time"].between(float(drop_time[0]),
                                               float(drop_time[1]))]
            data.drop(drops.index, inplace=True)
        if coordinate_system == ["lon", "lat"]:
            intersection_points = self._intersection_points_lonlat
        elif coordinate_system == ["xCenter", "yCenter"]:
            intersection_points = self._intersection_points_xy
        else:
            # placeholder
            intersection_points = self._intersection_points_lonlat
        for _, value in self._df_config.items():
            column = value.get("new_name")
            ops = value.get("operations")
            if not isinstance(ops, list):
                continue
            if len(ops) == 0:
                continue
            for operation, args in zip(ops, value.get("operations_args")):
                if operation == "moving_average":
                    moving_average(data[[column]], args)
        # drop tracks with speed average below 1 km/h
        valid_tracks = []
        for track_id, track_chunk in data.groupby(by=["trackId"]):
            if track_chunk["speed"].mean() > 1:
                valid_tracks.append(track_id)
        data = data[data["trackId"].isin(valid_tracks)]
        if "distanceIntersectionCrossing" not in data.columns:
            # calculate intersection crossing time
            pt1, pt2 = \
                intersection_points[:2], intersection_points[2:]
            for track_id in data["trackId"].unique():
                lon_lat = \
                    data[data["trackId"]==track_id][
                        [abscissa, ordinate]].values
                cross = np.cross(pt2-pt1, pt1-lon_lat)
                dist = cross / np.linalg.norm(pt2-pt1)
                # dist = np.abs(cross) / np.linalg.norm(pt2-pt1)
                data.loc[data["trackId"]==track_id,
                         "distanceIntersectionCrossing"] = dist
        if "intersectionCrossing" not in data.columns:
            data["intersectionCrossing"] = None
            for track_id in data["trackId"].unique():
                data_chunk = data[data["trackId"]==track_id]
                time = data_chunk["time"].values
                dist = data_chunk["distanceIntersectionCrossing"].values
                if np.unique(np.sign(dist)).shape[0] == 1:
                    data.loc[
                        data["trackId"]==track_id, "intersectionCrossing"] = 0
                    continue
                sign = np.sign(dist)
                sign_change = ((np.roll(sign, 1) - sign) != 0).astype(int)[1:]
                sign_change = np.where(sign_change == 1)
                if sign_change[0].shape[0] == 0:
                    crossing_time = 0
                else:
                    lower = sign_change[0][0]
                    upper = lower + 1
                    crossing_time = np.mean((time[lower],time[upper]),
                                            dtype=float)
                data.loc[data["trackId"]==track_id, "intersectionCrossing"] = \
                    crossing_time
        if "lane" in data.columns:
            if data.dtypes["lane"] != int:
                lanes = np.unique(data["lane"].values)
                new_lanes = np.arange(lanes.shape[0])
                for lane, new_lane in zip(lanes, new_lanes):
                    data.loc[:, "lane"] = data["lane"].replace(lane, new_lane)
        if "lane" not in data.columns:
            data["lane"] = None
            for track_id in data["trackId"].unique():
                data_chunk = data[data["trackId"]==track_id]
                data.loc[data["trackId"]==track_id, "lane"] = \
                    get_lane(data_chunk[abscissa].values,
                    data_chunk[ordinate].values,
                    lane_data,
                    use_xy=use_xy)
        
        return data

    def _update_metadata(self, data):
        rename = {}
        for key, value in self._df_config.items():
            rename.update({key: value.get("new_name") or key})
            if value.get("is_numeric"):
                if key in data.columns:
                    data[key] = pd.to_numeric(data[key], errors='coerce')
            if value.get("unit") == "kmh":
                if key in data.columns:
                    data[key] = data[key].div(3.6)
            # for key in list(data.keys()):
            # pylint: disable=C0201
            # if key not in list(rename.keys()):
            #     data.drop(key, inplace=True)
        data.rename(columns=rename, inplace=True)
        data.dropna(inplace=True)
        dtypes = {"trackId": str}
        data = data.astype(dtype=dtypes)
        return data

    def convert(self, identification=None, name=None,
                skip_preprocessing=False):
        for file in glob.glob(str(self._directory) + "/*"):
            file = Path(file)
            if not re.match(self._regex, file.name):
                continue
            if identification:
                if identification not in file.name:
                    continue
            if name is not None:
                if file.name != name:
                    continue
            data = pd.read_csv(file, **self._df_kwargs)
            data = self._update_metadata(data)
            meta_data = self._get_metafile(file)
            meta_data = self._update_metadata(meta_data)
            recording_id = meta_data["recordingId"].values[0]
            options = self.get_file_specific_options(recording_id)
            data = self._essential_preprocessing(data, meta_data, options)
            if not skip_preprocessing:
                data = self._optional_preprocessing(data, meta_data, options)
            lane_data = self.get_lane_df(data)
            yield data, meta_data, lane_data
            # https://www.ind-dataset.com/format

    def get_next_dataframes(self):
        self.prepare_reading()
        for conv in self.convert():
            yield conv

    def get_dataframes_by_id(self, identification: str):
        """gets the dataframes by id"""
        self.prepare_reading()
        return next(self.convert(identification=identification))

    def get_filename_by_id(self, identification: int):
        for file in glob.glob(str(self._directory) + "/*"):
            file = Path(file)
            if not re.match(self._regex, file.name):
                continue
            data = pd.read_csv(file, **self._df_kwargs)
            data = self._update_metadata(data)
            meta_data = self._get_metafile(file)
            meta_data = self._update_metadata(meta_data)
            recording_id = meta_data["recordingId"].values[0]
            if recording_id == identification:
                return str(file)

    def get_meta_data_by_id(self, identification: int):
        for file in glob.glob(str(self._directory) + "/*"):
            file = Path(file)
            if not re.match(self._regex, file.name):
                continue
            recording_id = (float(file.name.split("_")[0]))
            meta_data = self._get_metafile(file)
            meta_data = self._update_metadata(meta_data)
            recording_id = meta_data["recordingId"].values[0]
            if recording_id == identification:
                return meta_data

    def prepare_reading(self):
        for file in glob.glob(str(self._directory) + "/*"):
            file = Path(file)
            if not re.match(self._regex, file.name):
                continue
            try:
                self._prepare_reading(file)
            except Exception as ex:
                if self._skip_corrupt:
                    continue
                _LOGGER.error("could not read data file %s with skipping "
                    "disabled", str(file))
                raise ex

    def get_lane_df(self, _):
        for file in glob.glob(str(self._directory) + "/*"):
            file = Path(file)
            if not re.match(self._regex, file.name):
                continue
            filename = str(Path(file).parent / "01_lanes.csv")
            lane_data = pd.read_csv(filename, index_col=0)
            lane_data = self._update_metadata(lane_data)
            lane_data["trackId"] = lane_data["trackId"].astype(float)
            return lane_data

    @classmethod
    def read_file(cls, data_file: Path, skip_preprocessing=False) -> tuple:
        """
        reads single data file from path
        :ret:       returns (data_file, meta_data, lane_data)
        """
        data_set = DataSet(data_file.parent)
        if not data_file.exists():
            raise FileNotFoundError
        for data in data_set.convert(name=data_file.name,
                                     skip_preprocessing=skip_preprocessing):
            return data
        raise FileNotFoundError("could not find file, even though it exists")

    @staticmethod
    def create_data_config(target: Path, data_file_regex: str = None,
                           skip_corrupt: bool  = None,
                           intersection_points_xy: list = None,
                           intersection_points_lonlat: list = None,
                           file_specific_options: list = None,
                           input_format: dict = None,
                           output_format: dict = None,
                           df_config: dict = None,
                           df_kwargs: dict = None,
                           metafile_replace: list = None,
                           overwrite=False):
        """creates config file for a dataset"""
        if target.exists() and not overwrite:
            raise FileExistsError("Config file already exists, you must"
                                  " specify overwrite explicitly.")
        config = JSON(target)
        values = dict(
            data_file_regex=data_file_regex,
            skip_corrupt=skip_corrupt,
            intersection_points_xy=intersection_points_xy,
            intersection_points_lonlat=intersection_points_lonlat,
            file_specific_options=file_specific_options,
            input_format=input_format,
            output_format=output_format,
            df_config=df_config,
            df_kwargs=df_kwargs,
            metafile_replace=metafile_replace
        )
        config.set_values(values)
        # TODO: check if config is OK
        config.write_file()

    def get_data_chunk(self, identification, data_chunk=None):
        """gets chunk of leader follower pair"""
        # chunk = data_set.get_dataframes_by_id(f"{identification[2]:02d}")
        if data_chunk is None:
            file_path = Path(self.get_filename_by_id(identification[2]))
            data_chunk, _, _ = self.read_file(file_path,
                                    skip_preprocessing=True)
        conditions = data_chunk["trackId"].isin(identification[0:2])
        leader_follower_chunk = data_chunk[conditions].copy()
        leader_follower_chunk.loc[:, "recordingId"] = int(identification[2])
        return leader_follower_chunk

    def interactive_plot(self, recording_id, track_id=None):
        """
        interactive plot of a vehicle
        """
        rec = f"{int(recording_id):02d}"
        data_frame, meta_data, lane_data = self.get_dataframes_by_id(rec)
        identification = (track_id, None, recording_id)
        if track_id:
            time = data_frame[data_frame["trackId"]==track_id]["time"].values
        else:
            time = np.unique(data_frame["time"].values)
        max_f = data_frame["frame"].max()
        total_frame = np.arange(0, max_f+1)
        total_frame_track_ids = []
        for frame in total_frame:
            tracks = np.unique(
                data_frame[data_frame["frame"]==frame]["trackId"])
            total_frame_track_ids.append((tracks))
        initial_frames = {}
        for track, chunk in data_frame.groupby(by=["trackId"]):
            min_frame = min(chunk["frame"])
            initial_frames[track] = min_frame
        centroids = {}
        for track, chunk in data_frame.groupby(by=["trackId"]):
            centroids[track] = dict(zip(chunk["frame"].values,
                                        chunk[["xCenter", "yCenter"]].values))

        bboxes = {}
        for track_id in np.unique(data_frame["trackId"]):
            chunk = data_frame[data_frame["trackId"]==track_id]
            identification = (track_id, None, chunk["recordingId"].iloc[0])
            leader_meta, _ = _get_vehicle_meta_data(meta_data, identification)
            veh_l, veh_w = leader_meta["length"], leader_meta["width"]
            chunk_pos = chunk[["xCenter", "yCenter", "heading"]]
            bbox_vehicle = _get_bbox(chunk_pos[["xCenter", "yCenter"]].values,
                                     chunk_pos["heading"].values, veh_l, veh_w)
            frames_bbox = dict(zip(chunk["frame"].values, bbox_vehicle))
            bboxes[track_id] = frames_bbox
        drawn_items = {"bboxes": [], "annotations": [], "arrows": []}

        def draw_bboxes(frame):
            for key, items in drawn_items.items():
                for item in items:
                    item.remove()
                drawn_items[key] = []
            for track in total_frame_track_ids[frame]:
                bbox = bboxes[track][frame] # shape [4, 2]
                centroid = centroids[track][frame]
                max_frame = max(centroids[track].keys())
                max_next = np.clip(frame+25, 0, max_frame)
                dcentroid = centroids[track][max_next] - centroid
                plot_box = plt.Polygon(bbox, closed=True)
                anno = axis.annotate(track, [centroid[0], centroid[1]])
                drawn_items["annotations"].append(anno)
                axis.add_patch(plot_box)
                arrow = axis.arrow(*centroid, *dcentroid, color="r", zorder=100)
                drawn_items["arrows"].append(arrow)
                drawn_items["bboxes"].append(plot_box)

        def plot_lanes(lanes):
            lines = []
            for lane in np.unique(lanes["trackId"].values):
                chunk = lanes[lanes["trackId"]==lane]
                lines.append(chunk.plot(x="xCenter", y="yCenter", ax=axis))
            return lines
        # time_step = 1.0 / self._config_file.get_value("fps")



        fig = plt.figure(figsize=(8, 6),)
        axis = fig.add_subplot(111, aspect='equal')
        fig.subplots_adjust(bottom=0.25)
        lines = plot_lanes(lane_data)
        axis.set_xlabel('x [m]')
        axis.set_ylabel('y [m]')
        axis.set_aspect('equal')
        axis.set_xlim(min(data_frame["xCenter"].values)-10, max(data_frame["xCenter"].values)+10)
        axis.set_ylim(min(data_frame["yCenter"].values)-10, max(data_frame["yCenter"].values)+10)
        draw_bboxes(5)

        axes_time = fig.add_axes([0.2, 0.1, 0.65, 0.03])
        time_slider = Slider(
            ax=axes_time,
            label='Time [s]',
            valmin=time[1],
            valmax=time[-2],
            valinit=time[1],
        )

        def update(exact_time):
            frame = int(exact_time // (1 / self._config_file.get_value("fps")))
            draw_bboxes(frame)
        time_slider.on_changed(update)
        plt.show(block=True)

def _get_bbox(center, heading, veh_l, veh_w):
    heading_cos = np.cos(np.deg2rad(heading))
    heading_sin = np.sin(np.deg2rad(heading))
    _veh_l = veh_l / 2
    _veh_w = veh_w / 2
    _veh_l_sin = _veh_l * heading_sin
    _veh_l_cos = _veh_l * heading_cos
    _veh_w_sin = _veh_w * heading_sin
    _veh_w_cos = _veh_w * heading_cos
    bbox = np.zeros((center.shape[0], 4, 2))
    # bl
    bbox[:, 0, 0] = center[:,0] - _veh_l_cos + _veh_w_sin
    bbox[:, 0, 1] = center[:,1] - _veh_l_sin - _veh_w_cos

    # br
    bbox[:, 3, 0] = center[:,0] - _veh_l_cos - _veh_w_sin
    bbox[:, 3, 1] = center[:,1] - _veh_l_sin + _veh_w_cos

    # fl
    bbox[:, 1, 0] = center[:,0] + _veh_l_cos + _veh_w_sin
    bbox[:, 1, 1] = center[:,1] + _veh_l_sin - _veh_w_cos

    # fr
    bbox[:, 2, 0] = center[:,0] + _veh_l_cos - _veh_w_sin
    bbox[:, 2, 1] = center[:,1] + _veh_l_sin + _veh_w_cos
    return bbox

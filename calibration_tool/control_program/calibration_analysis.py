"""
Takes calibration results and renders plots with paramters for sensitivity
analysis into pdfs
"""
from pathlib import Path
import shutil

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibration_tool.fileaccess.parameter import EidmParameters
from calibration_tool.optimization import (_get_results, factory_wrapper)
from calibration_tool.sumo.simulation_module import SumoInterface
from calibration_tool.sumo.sumo_project import SumoProject


def _get_param_set(results_chunk: pd.DataFrame, iteration: int) -> dict:
    param_set = results_chunk[
        results_chunk["iteration"]==iteration].iloc[0].to_dict()
    keys = list(EidmParameters.get_defaults_dict().keys())
    return {key: param_set[key] for key in keys}

def _get_unique_calibrations(results: pd.DataFrame):
    results = results.copy()
    follower_dtype = results["follower"].dtype
    results["follower"] = results["follower"].astype(str)
    calibration = (
        results.groupby(["leader", "follower", "recordingId"], as_index=False))
    rows = (
        results[["leader", "follower", "recordingId"]]
        .drop_duplicates()
        .copy())
    for _, row in rows.iterrows():
        identification = (row["leader"],
                          row["follower"],
                          row["recordingId"])
        chunk = calibration.get_group(identification)
        if follower_dtype == np.float64:
            follower = float(row["follower"])
        else:
            follower = row["follower"]
        identification = (row["leader"],
                          follower,
                          row["recordingId"])
        chunk["follower"] = chunk["follower"].astype(follower_dtype)
        yield identification, chunk

def _simulate_single(identification, results_chunk, sumo_interface,
                    project_path):
    param_sets = {}
    for iteration in results_chunk["iteration"].unique():
        param_sets.update({iteration: _get_param_set(results_chunk, iteration)})
    simulation_results = _run_sumo(identification, sumo_interface, param_sets,
                                  project_path)
    return simulation_results

def _run_sumo(identification, sumo_interface, param_sets, project_path):
    # SumoProject.create_sumo(project_path, eidm0,  len(param_sets))
    eidms = []
    for idx in range(len(param_sets)):
        param_set = param_sets[idx+1]
        eidm = EidmParameters.create_eidm_parameter_set(f"set{idx}.json",
                                                        **param_set)
        eidms.append(eidm)
    SumoProject.write_followers_leader(
        project_path / "calibration_routes.rou.xml",
        eidms)
    # identification = (identification[0], float(identification[1]), identification[2])
    simulation_results = sumo_interface.run_simulation(
        identification=identification)
    return simulation_results[1]

def _get_sorted_params(param_sets, simulation_result, identification,
                      results_chunk, max_num, objective_function):
    errors = _get_weighted_errors(param_sets, simulation_result, identification,
                                 results_chunk, objective_function)
    weighted_errors = []
    for iteration, _ in param_sets.items():
        weighted_errors.append((iteration, errors[iteration-1]))
    weighted_errors = sorted(
        weighted_errors,
        key=lambda x: (x[1], len(errors) - x[0]))
    for iteration, _ in weighted_errors[:max_num]:
        yield iteration, param_sets[iteration]

def _get_weighted_errors(param_sets, sim_result, ident, results_chunk,
                        objective_function, force_simulated=False):
    weighted_errors = []
    for jdx, _ in param_sets.items():
        pred, gt = _get_results({1: sim_result}, ident, jdx-1)
        weighted_error = results_chunk["weightedError"].values[jdx-1]
        if weighted_error == 0 or force_simulated:
            weighted_error = objective_function(gt, pred)
        weighted_errors.append(weighted_error)
    return weighted_errors

def _drop_keys_from_param(param_sets, results_chunk):
    drop_keys = _key_drop_keys(results_chunk)
    for _, param_set in param_sets.items():
        for key in drop_keys:
            param_set.pop(key, None)
    return param_sets

def _key_drop_keys(results_chunk):
    data = results_chunk.values.T
    drop_keys = []
    if len(results_chunk) <= 1:
        drop_keys = []
        return drop_keys
    if "paramKeys" in results_chunk.columns:
        columns = list(results_chunk.columns)[1:]
        used_keys = results_chunk["paramKeys"].iloc[0].split(",")
        return list(set(columns) - set(used_keys))
    for column in range(1, data.shape[0]):
        if all(data[column] == data[column, 0]):
            drop_keys.append(list(results_chunk.columns)[column])
    return drop_keys

def _plot_single(identification, results_chunk, simulation_result,
                objective_function):
    param_sets = {}
    for iteration in results_chunk["iteration"].unique():
        param_sets.update({iteration: _get_param_set(results_chunk, iteration)})
    fig1, axes = plt.subplots(4, 1, figsize=(10, 10))
    axes[0].set_ylabel("coveredDistance")
    axes[1].set_ylabel("distance")
    axes[2].set_ylabel("speed")
    axes[2].set_xlabel("time")
    axes[3].set_xlabel("iteration")
    axes[3].set_ylabel("weighted error")
    title = "Leader={} Follower={} RecordingId={} best parameter set"
    axes[0].set_title(title.format(*identification))
    pred, ground_truth = _get_results(
        {1: simulation_result}, identification, len(param_sets)-1)
    time = ground_truth["time"]
    covered_distance = ground_truth["coveredDistanceFollower"]
    axes[0].plot(time[::5], covered_distance[::5],
                 label="ground_truth", marker="+")
    distance = ground_truth["coveredDistanceLeader"]
    if identification[0] != "":
        axes[0].plot(time[::5], distance[::5], label = "", linestyle="--", color="C7")
    distance = ground_truth["distance"]
    axes[1].plot(time[::5], distance[::5], label = "ground_truth", marker="+")
    speed = ground_truth["speedFollower"]
    axes[2].plot(time[::5], speed[::5], label = "ground_truth", marker="+")
    speed = ground_truth["speedLeader"]
    if identification[0] != "":
        axes[2].plot(time[::5], speed[::5], label = "", linestyle="--", color="C7")
    for idx, weighted_error in enumerate(
        _get_weighted_errors(param_sets, simulation_result, identification,
                            results_chunk, objective_function, True)):
        axes[3].scatter(idx, weighted_error, marker="v", color="b")
    for idx, weighted_error in enumerate(
        results_chunk["weightedError"].values):
        axes[3].scatter(idx, weighted_error, color="r")
    for jdx, param_set in _get_sorted_params(
        param_sets, simulation_result, identification, results_chunk, 1,
        objective_function):
        pred, gt = \
            _get_results(
                {1: simulation_result}, identification, jdx-1)
        try:
            weighted_error = objective_function(gt, pred)
        except FloatingPointError:
            pred, gt = _get_results(
                {1: simulation_result}, identification, jdx-1)
            weighted_error = results_chunk["weightedError"].values[jdx-1]
        time = pred["time"].values
        covered_distance = pred["coveredDistanceFollower"].values
        distance = pred["distance"].values
        speed = pred["speedFollower"].values
        axes[0].plot(time, covered_distance, label = f"param_set#{jdx}")
        axes[1].plot(time, distance,
                     label = f"param_set#{jdx} f(x)={weighted_error:.4f}")
        axes[2].plot(time, speed, label = f"param_set#{jdx}")
    axes[0].legend()
    axes[1].legend()
    plt.close()
    param_sets = _drop_keys_from_param(param_sets, results_chunk)
    n_rows = np.max((np.ceil(len(param_sets[1])/3), 1)).astype(int)
    fig2, axes = plt.subplots(n_rows,
                              3, figsize=(10, 10))
    for iteration, param_set in param_sets.items():
        for idx, (key, value) in enumerate(param_set.items()):
            axes[idx//3][idx%3].set_title(key)
            if idx//3 != axes.shape[0]-1:
                axes[idx//3][idx%3].xaxis.set_ticklabels([])
            axes[idx//3][idx%3].scatter(
                iteration, value, label=f"it={iteration}")
    for idx in range(len(param_sets[1]),
                     n_rows*3):
        if n_rows == 1:
            axis = axes[idx//3]
        else:
            axis = axes[idx//3][idx%3]
        axis.grid(False)
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
        axis.xaxis.set_ticklabels([])
        axis.yaxis.set_ticklabels([])
        for item in ["top", "right", "bottom", "left"]:
            axis.spines[item].set_visible(False)
    plt.tight_layout()
    plt.close()
    return fig1, fig2

def create_calibration_analysis(outputpath, data_directory):
    """creates calibration analysis pdfs from calibration results"""
    rc_file = Path(__file__).parents[1] / "data_config/matplotlib.rc"
    plt.style.use(rc_file)
    dtypes = {"leader": str, "follower": str}
    for calibration_result in outputpath.rglob("calibration_results_*.csv"):
        results_data = pd.read_csv(calibration_result, index_col=0,
                                   dtype=dtypes)
        results_data["leader"].fillna("", inplace=True)
        results_data = results_data.astype(
            dtype={"leader": str, "follower": str})
        objective_function = factory_wrapper(results_data)
        simulation_results = []
        for identification, results_chunk in _get_unique_calibrations(
            results_data):
            project_path = (outputpath
                            / ".tmp/sumo_project_analysis/")
            if not project_path.exists():
                project_path.mkdir(parents=True)
            num_param_sets = results_chunk.values.shape[0]
            SumoProject.create_sumo(project_path, num_param_sets)
            sumo = SumoInterface(project_path, data_directory, gui=False)
            simulation_result = _simulate_single(
                identification, results_chunk, sumo, project_path)
            simulation_results.append(simulation_result)
            sumo.release()
        figures = []
        for (identification, results_chunk), simulation_result in zip(
                _get_unique_calibrations(results_data), simulation_results):
            fig1, fig2 = _plot_single(identification, results_chunk,
                                     simulation_result, objective_function)
            figures.extend([fig1, fig2])
        filename = (outputpath
                    / f"{str(calibration_result.name)[:-4]}_plots.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
        for figure in figures:
            pdf.savefig(figure)
        pdf.close()
        shutil.rmtree(outputpath / ".tmp")

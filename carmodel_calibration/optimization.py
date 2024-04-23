"""module with optimizers"""

import logging
from multiprocessing import Pool
import os

import numpy as np
import pandas as pd

# from carmodel_calibration.exceptions import OptimizationFailed
from carmodel_calibration.fileaccess.parameter import ModelParameters
from carmodel_calibration.sumo.sumo_project import SumoProject

_LOGGER = logging.getLogger(__name__)

def measure_of_performance_factory(objectives=["distance"],
                                   weights=None, gof="rmse", **__):
    """
    this function will return the appropiate objective function handle
    depending on the desired MOP and the calibration object
    (leader-follower-pair, free leader)
    case1:  leader-follower-pair, sop: distance
    case2:  leader-follower-pair, mmop: distance and speed
    :param identification:  (leader identification, follower identification,
                            recordingId)
    :param objectives:      ['distance', 'speed', 'speedVariance',...] the
                            objectives to calibrate, weighted equally if not
                            specified differently by the `weights` parameter
    :param weights:         `tuple` of len(objectives), each item of type float
                            in range 0 < item < 1, must equal 1 in sum
    :param gof:             ['rmse', 'rmsep'] goodness-of-fit function `rmse`
                            per default
    """
    if not isinstance(gof, str):
        if gof is None or np.isnan(gof):
            gof = "rmse"
    try:
        assert len(objectives) > 0
        if weights:
            weights_values = np.array(weights)
            assert np.isclose(np.sum(weights_values), 1)
            assert weights_values.shape[0] == len(objectives)
        else:
            weights_values = np.ones(len(objectives)) / len(objectives)
        weights = {}
    except AssertionError as exc:
        _LOGGER.error("Failed creating objective function with given args.")
        raise exc
    for weights_value, objective in zip(weights_values, objectives):
        weights[objective] = weights_value
    def rmse(ground_truth, prediction):
        """RMSE(v)
        see 'About calibration of car-following dynamics of automated and
        human-driven vehicles: Methodology, guidelines and codes'
        """
        sums = np.zeros(len(objectives))
        for idx, sop in enumerate(objectives):
            sums[idx] = np.sum(np.square(
                prediction[sop].values
                - ground_truth[sop].values))
            sums[idx] = np.sqrt(sums[idx] / len(ground_truth)) * weights[sop]
        return np.sum(sums)
    def rmspe(ground_truth, prediction):
        """
        RMSPE(v)
        see 'About calibration of car-following dynamics of automated and
        human-driven vehicles: Methodology, guidelines and codes'
        """
        sums = np.zeros(len(objectives))
        for idx, sop in enumerate(objectives):
            sums[idx] = np.sum(np.square(
                (prediction[sop].values
                 - ground_truth[sop].values)
                / ground_truth[sop].values))
            sums[idx] = np.sqrt(sums[idx] / len(ground_truth)) * weights[sop]
        return np.sum(sums)
    def theils_u(ground_truth, prediction):
        """theils U(v)
        see 'About calibration of car-following dynamics of automated and
        human-driven vehicles: Methodology, guidelines and codes'
        """
        sums = np.zeros(len(objectives))
        for idx, sop in enumerate(objectives):
            rmse_error = np.sum(np.square(
                prediction[sop].values
                - ground_truth[sop].values))
            rmse_error = np.sqrt(rmse_error / len(ground_truth)) * weights[sop]
            sim_root = np.sqrt(np.sum(np.square(prediction[sop].values))
                               / len(ground_truth))
            gt_root = np.sqrt(np.sum(np.square(ground_truth[sop].values))
                              / len(ground_truth))
            sums[idx] = rmse_error / (sim_root + gt_root) * weights[sop]
        return np.sum(sums)

    def model_output(_, prediction):
        """only sum model outputs"""
        sums = np.zeros(len(objectives))
        for idx, sop in enumerate(objectives):
            sums[idx] = prediction[sop].values[-1] * weights[sop]
        return np.sum(sums)
    gof_handles = {"rmse": rmse, "rmsep": rmspe, "modelOutput": model_output,
                   "theils_u": theils_u}

    def get_weighted_error(ground_truth, prediction):
        """calculate weighted error on case1"""
        length = len(ground_truth)
        if (len(ground_truth) != len(prediction)) or length == 0:
            raise ValueError
        with np.errstate(divide='raise', invalid="raise"):
            weigthed_error = gof_handles[gof](ground_truth, prediction)
        return weigthed_error

    return get_weighted_error

def factory_wrapper(factory_kwargs):
    """invokes factory from results data"""
    if isinstance(factory_kwargs, pd.DataFrame):
        factory_kwargs = factory_kwargs.reset_index().copy()
        kwargs = factory_kwargs.iloc[0].to_dict()
        if not np.isnan(kwargs["weights"]):
            kwargs["weights"] = [
                float(item) for item in kwargs["weights"].split(",")]
        else:
            kwargs["weights"] = None
        kwargs["objectives"] = kwargs["objectives"].split(",")
        kwargs["recordingId"] = float(kwargs["recordingId"])
        return measure_of_performance_factory(**kwargs)
    elif isinstance(factory_kwargs, dict):
        return measure_of_performance_factory(**factory_kwargs)
    else:
        raise TypeError(
            "`factory_kwargs` must either be of type dict or pandas DattaFrame"
            )

def _get_results(simulation_results, identification, lane):
    """returns the simulation result for specific identification"""
    edge_name = f"B{lane}A{lane}"
    condition = (
        (simulation_results[1]["prediction"]["edgeName"]==edge_name)
    )
    prediction_result = simulation_results[1]["prediction"][condition]
    leader = identification[0]
    recording_id = identification[2]
    follower_condition = (
        simulation_results[1]["ground_truth"]["follower"]==identification[1])
    condition = (
        (simulation_results[1]["ground_truth"]["leader"]==leader)
        & (follower_condition)
        & (simulation_results[1]["ground_truth"]
            ["recordingId"]==recording_id)
        & (simulation_results[1]["ground_truth"]
            ["counter"]==lane)
    )
    ground_truth_result = (
        simulation_results[1]["ground_truth"][condition])
    return prediction_result, ground_truth_result

def _run_sumo(identification, sumo_interface, param_sets, project_path, model):
    cfmodels = []
    route_count = SumoProject.get_number_routes(
        project_path / "calibration_routes.rou.xml")
    if route_count != len(param_sets):
        sumo_interface.release()
        SumoProject.create_sumo(project_path, model, len(param_sets))
        sumo_interface.start_simulation_module()
    for idx, param_set in enumerate(param_sets):
        cfmodel = ModelParameters.create_parameter_set(f"set{idx}.json", model,
                                                        **param_set)
        cfmodels.append(cfmodel)
    SumoProject.write_followers_leader(
        project_path / "calibration_routes.rou.xml",
        cfmodels)
    simulation_results = sumo_interface.run_simulation(
        identification=identification)
    return simulation_results

def _calculate_performance(idx, simulation_results, identification,
                          objective_function):
    prediction, ground_truth = _get_results(
            simulation_results, identification, idx)
    objective_function = factory_wrapper(objective_function)
    return objective_function(ground_truth, prediction)

def _vectorized_target(params, *data):
    # params.shape = (Number params, number of solutions)
    params = np.atleast_2d(params).T
    data = data[0]
    identification = data["identification"]
    model = data["cfmodel"]
    sumo_interface = data["sumo_interface"]
    objective_function = data["objective_function"]
    project_path = data["project_path"]
    # keys of the paramters that are optimized
    param_names = data["param_names"]
    param_sets = []
    if params.shape[1]==1:
        params = params.T
    for solution in params:
        params_dict = data["default_parameters"].copy()
        params_dict.update(
            {key: value for key, value in zip(param_names, solution)})
        param_sets.append(params_dict)
    simulation_results = _run_sumo(
        identification, sumo_interface, param_sets, project_path, model)
    with Pool(os.cpu_count()//2) as pool:
        results = []
        for idx in range(len(params)):
            results.append((idx, simulation_results, identification,
                            objective_function))
        performance = list(pool.starmap(_calculate_performance, results))
    #performance = []
    #for idx in range(len(params)):
    #    performance.append(_calculate_performance(idx, simulation_results, identification,
    #                    objective_function))
    return performance


def target_factory(data, invert_error=False):
    """factory for creating a target callback"""
    def _vectorized_wrapper(params, solution_indexes):
        solutions = _vectorized_target(params.T, data)
        if solution_indexes is None:
            return None
        if invert_error:
            return [1 / solution for solution in solutions]
        else:
            return solutions
    return _vectorized_wrapper

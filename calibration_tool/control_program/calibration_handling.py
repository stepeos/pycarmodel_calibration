"""module to handle simulations"""
import io
import logging
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pygad
import scipy.optimize as op
from scipy.stats import truncnorm
from tqdm import tqdm

from calibration_tool.fileaccess.parameter import EidmParameters, Parameters
from calibration_tool.optimization import (target_factory,
                                           _vectorized_target)
from calibration_tool.sumo.simulation_module import SumoInterface
from calibration_tool.sumo.sumo_project import SumoProject
from calibration_tool.control_program.simulation_handler import SimulationHandler
from calibration_tool.control_program.calibration_analysis import (
    create_calibration_analysis)
from calibration_tool.helpers import (_estimate_parameters)
from calibration_tool.exceptions import OptimizationFailed

_LOGGER = logging.getLogger(__name__)
ITERATION = 0
START_TIME = None

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class CalibrationHandler(SimulationHandler):
    """class to handle the calibration process"""

    def __init__(self, directory: str, input_data: str, model: str = "eidm",
                 optimization: str = "differential_evolution",
                 max_iter:int = 100, param_keys: list = None,
                 population_size: int = 100,
                 project_path: str = None,
                 gof: str = "rmse",
                 mop: list = ["distance"],
                 seed: int = None,
                 force_recalculation: bool = False):
        """
        :param directory:           output directory of calibration config and
                                    result data
        :param input_data:          directory to input data
        :param model:               model to simulate
        :param optimization:        the type of optimization to be used, can be
                                    `differential_evolution` or `direct`
        :param param_keys:          list of paramters that are to be optimized,
                                    if left emtpy, then all except
                                    [`emergencyDecel`,`stepping`]
                                    will be tuned
        :param population_size:     size of the population, does not apply to
                                    `direct` optimization  >= 5
        :param gof:                 goodness-of-fit-function
        :param mop:                 list of measure of performances
        :param seed:                seed for random variables
        :param force_recalculation: forces to recalculate the selection
        """
        super().__init__(directory=directory, input_data=input_data,
                        param_keys=param_keys, model=model,
                        project_path=project_path)
        self.force_recalculation = force_recalculation
        self._optimization = optimization
        self._max_iter = max_iter
        self.parameters = []
        self.population_size = max(5, population_size)
        self.param_keys = param_keys or [
            "minGap", "accel", "decel", "tau", "delta", "tpreview",
            "tPersDrive", "tPersEstimate", "treaction", "ccoolness",
            "sigmaleader", "sigmagap", "sigmaerror", "jerkmax",
            "epsilonacc", "taccmax", "Mflatness", "Mbegin"]
        if project_path:
            self.delete_path = False
            self.project_path = Path(project_path)
        else:
            self.delete_path = True
            self.project_path = (self.directory
                                 / f"sumo_project_{self._optimization}")
        self.objectives = mop
        self.weights = None
        self.gof = gof
        self._lock = None
        self._identification = None
        self.iteration_results = None
        self.buf = None
        self.initial_population = None
        self.x_0 = None
        self.seed = seed
        self.data = None
        self.identification = None
        self.x_optimizing = None
        self.sumo_interface = None
        if self.seed:
            np.random.seed(self.seed)
        self._check_options()

    def run_calibration_cli(self, create_reports=False):
        """
        returns pretty string for cli usage
        :param create_reports:      if True, then it willl create pdfs after
                                    the cailbration procoess with plots of the
                                    iteration progress, parameters and the best
                                    param sets
        """
        result = self.run_calibration()
        if create_reports:
            create_calibration_analysis(self.directory, self._input_data)
        ids = result.groupby(by=["leader", "follower", "recordingId"])
        for ident, chunk in ids:
            last_iteration = np.max(chunk["iteration"].values)
            best = result[result["iteration"]==last_iteration]
            weighted_error = best["weightedError"].values[0]
            params = chunk.iloc[0].to_dict()
            # pylint: disable=C0201
            params = {key: params[key] for key in self.x_0.keys()}
            print(f"Results for leader = {ident[0]} "
                  f"follower = {ident[1]} recordingId = {ident[2]}:")
            print(f"weigthedError = {weighted_error}")
            for key, value in params.items():
                print(f"{key} = {value}")

    def run_calibration(self):
        """
        Main Control Program Routine
        1. Data processing: find selection of calibration pairs 👫
        2. Prepare Simulation Module
        3. Initialize Optimizer ⛏
        4. For each optimization iteration, do:
            - run sumo simulation with ground truth and get prediction of
                current parameter set 🖥
            - calculate objective function for weighted error
            - adjust parameters
            until stop-criterion 🛑
        """
        self.prepare_selection(not self.force_recalculation)
        if not self.is_calibration_prepared():
            raise RuntimeError("Calibration not prepared. Aborting process")
        # pylint: disable=W0603
        self._prepare_sumo_project()
        self.selection_data.to_csv(self.project_path / "selection.csv")
        bounds = Parameters.get_bounds_from_keys(self.param_keys, 0.04)
        # cons = EidmParameters.get_constraints()
        managed_results = []
        managed_results_out = None
        identifications = (
            self.selection_data[["leader", "follower", "recordingId"]])
        # identifications = identifications[identifications["leader"]==""]
        indexes = np.array(identifications.index)
        np.random.shuffle(indexes)
        identifications = identifications.loc[indexes]
        sumo_pipe = open(
            str(self.project_path / "traci.log"), "w",
            encoding="utf-8")
        sumo_interface = SumoInterface(self.project_path,
                                        self._input_data,
                                        file_buffer=sumo_pipe,
                                        gui=False)
        results_path = (self.directory
                        / f"calibration_results_{self._optimization}.csv")
        try:
            results_path.unlink()
        except FileNotFoundError:
            pass
        for idx, identification in enumerate(identifications.values):
            self.identification = identification
            _LOGGER.info("Starting optimization %d/%d",idx+1,
                         len(identifications))
            try:
                result = self._run_optimization(
                    bounds, sumo_interface, identification)
            except OptimizationFailed:
                _LOGGER.error("Optimization failed for %s", identification)
                continue
            except Exception as exc:
                _LOGGER.error("Optimization initialization failed for %s"
                              " with message %s",
                              identification, str(exc))
                continue
            iteration_results = self._tinker_results(
                identification, result)
            managed_results.append(iteration_results)
            _LOGGER.info("Finished %s optimization for %s", self._optimization,
                         identification)
            # pylint: disable=W0212
            if not isinstance(result, op._optimize.OptimizeResult):
                fun = 1 / result["fun"]
                best_solution = result["x"]
            else:
                fun = result.fun
                best_solution = result.x
            _LOGGER.info("result.fun= %s", fun)
            optimal_params = {}
            for idx, key in enumerate(self.param_keys):
                optimal_params[key] = best_solution[idx]
            _LOGGER.info("parameters= %s", optimal_params)
            managed_results_out = pd.concat(managed_results)
            if len(managed_results_out) > 0:
                managed_results_out.reset_index(drop=True, inplace=True)
                managed_results_out.to_csv(results_path)
        sumo_interface.release()
        del sumo_interface
        sumo_pipe.close()
        if managed_results_out is not None:
            return managed_results_out
        else:
            raise OptimizationFailed("No Optimization results.")

    def _tinker_results(self, identification, result):
        cols = (["iteration", "weightedError", "convergence"]
            + list(EidmParameters.get_defaults_dict().keys()))
        iteration_results = pd.DataFrame(self.iteration_results,
                                                columns=cols)
        if self._optimization == "direct":
            self.log_iteration(result.x, **{"weighted_error": result.fun})
            iteration_results = pd.DataFrame(self.iteration_results,
                                    columns=cols)
        iteration_results.loc[:,"leader"] = identification[0]
        iteration_results.loc[:,"follower"] = identification[1]
        iteration_results.loc[:,"recordingId"] = identification[2]
        iteration_results.loc[:,"algorithm"] = self._optimization
        iteration_results.loc[:,"pop-size"] = self.population_size
        iteration_results.loc[:,"objectives"] = ",".join(self.objectives)
        iteration_results.loc[:,"paramKeys"] = ",".join(self.param_keys)
        if self.weights:
            weights_str = [str(item) for item in self.weights]
        else:
            weights_str = ""
        iteration_results.loc[:,"weights"] = ",".join(weights_str)
        iteration_results.loc[:,"gof"] = self.gof
        return iteration_results

    def _run_optimization(self, bounds, sumo_interface, identification):
        global ITERATION
        global START_TIME
        identification = tuple(identification)
        _LOGGER.info("Starting %s for identification%s",
                     self._optimization, str(identification))
        ITERATION = 1
        self.iteration_results = []
        self._prepare_optimization(
            sumo_interface, identification)
        START_TIME = time.time()
        if self._optimization == "differential_evolution":
            self.initial_population = random_population_from_bounds(
                bounds, self.population_size)
            popsize = np.ceil(self.population_size
                              / len(self.x_optimizing)).astype(int)
            with io.StringIO() as self.buf, redirect_stdout(self.buf):
                result = op.differential_evolution(
                    _vectorized_target, bounds,
                    maxiter=self._max_iter,
                    args=(self.data,),
                    updating="deferred",
                    recombination=0.7,
                    mutation=(0.5, 1),
                    strategy="best1bin",
                    popsize=popsize,
                    disp=True,
                    seed=self.seed,
                    x0=self.x_optimizing,
                    polish=False,
                    callback=self.log_iteration,
                    vectorized=True)
                # TODO change to pymoode library
            self.buf = None
        elif self._optimization == "genetic_algorithm":
            self.initial_population = random_population_from_bounds(
                bounds, self.population_size)
            # Set the population genes with estimations
            for key in ["minGap", "taccmax", "Mbegin", "Mflatness",
                        "speed_factor", "startupDelay"]:
                if key in self.param_keys:
                    bnds = EidmParameters.get_bounds_from_keys([key], 0.04)[0]
                    std = bnds[1] - bnds[0] / 2
                    itera = get_truncated_normal(
                        self.x_0[key], std, bnds[0], bnds[1])
                    self.initial_population[:self.population_size//2,
                                            self.param_keys.index(key)] = (
                        itera.rvs(self.population_size//2))
            ga_instance = pygad.GA(
                num_generations=self._max_iter,
                num_parents_mating=self.initial_population.shape[0]//10,
                initial_population=self.initial_population,
                sol_per_pop=self.population_size,
                fitness_func=target_factory(self.data, True),
                suppress_warnings=True,
                mutation_by_replacement=True,
                # mutation_num_genes=np.clip(6, len(self.param_keys)),
                crossover_probability=0.4,
                keep_elitism=0,
                gene_space=np.array(bounds),
                keep_parents=self.initial_population.shape[1]//5,
                fitness_batch_size=self.population_size,
                random_seed=self.seed,
                mutation_type="random",
                # mutation_probability=[0.33, 0.15], # for adaptive
                mutation_probability=0.33,
                save_best_solutions=True,
                on_fitness=fitness_callback_factory(self))
            ga_instance.run()
            solution, solution_fitness, _ = (
                ga_instance.best_solution())
            result = {"fun": solution_fitness, "x": solution}
            del ga_instance
        elif self._optimization == "direct":
            result = op.direct(
                _vectorized_target,
                bounds,
                maxiter=self._max_iter,
                eps=0.000000001,
                args=(self.data, ),
                callback=self.log_iteration)
        return result

    def _prepare_optimization(self, sumo_interface, identification):
        data_chunk = self._data_set.get_data_chunk(identification)
        args = _estimate_parameters(identification, data_chunk,
                                    self.meta_data)
        x_0 = EidmParameters.get_defaults_dict(*args)
        self.x_0 = x_0
        x_optimizing = np.array([self.x_0[key] for key in self.param_keys])
        self.sumo_interface = sumo_interface
        proxy_objects = {
                "sumo_interface": sumo_interface,
                "project_path": self.project_path
        }
        objective_function = {
            "identification": identification,
            "objectives": self.objectives,
            "weights": self.weights,
            "gof": self.gof}
        data = {
                "identification": identification,
                "objective_function": objective_function,
                "default_parameters": self.x_0,
                "param_names": self.param_keys
        }
        data.update(proxy_objects)
        self.data = data
        self.x_optimizing = x_optimizing

    def is_calibration_prepared(self) -> bool:
        """returns true if calibration is ready to go"""
        return self._prepared

    def log_iteration(self, *args, **kwargs):
        """log each iteration to terminal"""
        # pylint: disable=W0603
        global ITERATION
        global START_TIME
        params = args[0]
        time_taken = time.time() - START_TIME
        params_dict = self.x_0
        params_dict.update({key: value for key, value in zip(self.param_keys,
                                                             params)})
        if self.buf:
            lines = self.buf.getvalue().split("\n")
            line = ""
            for line in lines:
                if f"differential_evolution step {ITERATION}: f(x)= " in line:
                    break
            weighted_error = float(line.split("f(x)= ")[-1])
        else:
            weighted_error = kwargs.get("weighted_error") or 0
        self.iteration_results.append(
            {"iteration": ITERATION, "weightedError": weighted_error,
             "convergence": kwargs.get("convergence") or -1, **params_dict})
        _LOGGER.debug("Current iteration %d with weighted_error=%f, "
                     " time taken:%f sec.\nConvergence:%f Parameters:%s",
                     ITERATION, weighted_error, time_taken,
                     kwargs.get("convergence") or -1, params_dict)
        _LOGGER.info(tqdm.format_meter(ITERATION, self._max_iter,
                          elapsed=time_taken,
                          prefix=str(self.identification))+
              f" f(x)={weighted_error:.3f}")
        ITERATION += 1
        # if we return True or when the convergence => 1 , then the polishing
        # step is initialized
        stop_criterion = False
        return stop_criterion

    def _prepare_sumo_project(self):
        project_path = self.project_path
        if not project_path.exists():
            project_path.mkdir()
        SumoProject.create_sumo(project_path, self.population_size)


def random_population_from_bounds(bounds: tuple, population_size: int,
                                  std_devs: tuple = ()):
    """
    creates a random population within boundaries
    :param bounds:              ((lb, up), (lb, up), (lb, ub),...)
    :param population_size:     specifies the number of unique populations
    :param std_devs:            (col1, col10,..) column ids to use std_dev
                                instead of uniform distribution
                                for each column
    :ret:                       output of shape [population_size, len(bounds)]
    """
    bounds_arr = np.array(bounds)
    population = np.zeros((population_size, len(bounds)))
    diff = bounds_arr[:, 1] - bounds_arr[:,0]
    for pop in range(population_size):
        # population[pop, i] = np.random.normal(bounds_arr[i, 0], std_devs[i], 1)
        if pop in std_devs:
            population[pop,:] = diff * np.random.normal(0.5,
                                                        std_devs[pop],
                                                        diff.shape[0])
            population[pop,:] = np.clip(population[pop,:], 0, 1)
        else:
            population[pop,:] = diff * np.random.uniform(0.0000000001,
                                                        0.9999999999,
                                                        diff.shape[0])
        population[pop,:] += bounds_arr[:,0]
    return population

def fitness_callback_factory(item):
    """a factory for the fitness callback function"""

    def on_fitness(ga_instance: pygad.GA, solutions):
        """provided callback for fitness"""
        best_solution = ga_instance.best_solutions[-1]
        best = np.max(ga_instance.last_generation_fitness)
        if best < np.max(solutions):
            best = np.max(solutions)
            best_solution = ga_instance.population[
                np.argmin(solutions).astype(int)]
        # TODO: log instead of 1 / 0
        kwargs = {"weighted_error": 1 / best}
        _ = item.log_iteration(best_solution, **kwargs)
    return on_fitness

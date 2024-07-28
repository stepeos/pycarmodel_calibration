"""module to handle simulations"""
import io
import logging
import time
from contextlib import redirect_stdout
from pathlib import Path
import traceback
from random import randint

import numpy as np
import pandas as pd
import pygad
import scipy.optimize as op
from scipy.stats import truncnorm
from tqdm import tqdm
from pymoo.optimize import minimize
from pymoode.algorithms import GDE3, NSDE, NSDER
from pymoode.survival import RankAndCrowding, ConstrRankAndCrowding
from pymoo.termination.default import DefaultSingleObjectiveTermination, DefaultMultiObjectiveTermination
from pymoo.core.callback import Callback

from carmodel_calibration.fileaccess.parameter import ModelParameters, Parameters
from carmodel_calibration.optimization import (target_factory,
                                               target_factory_nsga2,
                                               target_factory_mo_de,
                                               _vectorized_target)
from carmodel_calibration.sumo.simulation_module import SumoInterface
from carmodel_calibration.sumo.sumo_project import SumoProject
from carmodel_calibration.control_program.simulation_handler import SimulationHandler
from carmodel_calibration.control_program.calibration_analysis import (
    create_calibration_analysis)
from carmodel_calibration.helpers import (_estimate_parameters)
from carmodel_calibration.exceptions import OptimizationFailed

_LOGGER = logging.getLogger(__name__)
ITERATION = 0
START_TIME = None


def _get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class CalibrationHandler(SimulationHandler):
    """class to handle the calibration process"""

    def __init__(self, directory: str, input_data: str, model: str = "eidm",
                 remote_port: int = randint(8000, 9000),
                 timestep: float = 0.04,
                 optimization: str = "differential_evolution",
                 max_iter: int = 100, param_keys: list = None,
                 population_size: int = 100,
                 project_path: str = None,
                 gof: str = "rmse",
                 mop: list = ["distance"],
                 seed: int = None,
                 num_parents_mating: float = 2.,  # (2=50%) 4=25%, 4/3=75%
                 parent_selection_type: str = "sss",  # "sus", "rank"
                 crossover_type: str = "uniform",  # "single_point"
                 crossover_probability: float = 0.4,  # 0.2, 0.7
                 keep_elitism: float = 4.,  # (4=25%) 2=50%, 4/3=75%
                 mutation_type: str = "random",  # "scramble", "adaptive"
                 # 0.1, 0.6, 0.9, for adaptive [0.1, 0.4]
                 mutation_probability: float = 0.33,
                 strategy: str = "best1bin",  # best2bin, rand1bin
                 recombination: float = 0.7,  # 0.2, 0.5, 0.9
                 mutation: tuple = (0.5, 1.0),  # (0.1, 0.6), (1.1, 1.6)
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
                         param_keys=param_keys,
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
            "epsilonacc", "actionStepLength", "taccmax", "Mflatness", "Mbegin"]
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
        self.model = model
        self.timestep = timestep
        self._port = remote_port
        self._lock = None
        self._identification = None
        self.iteration_results = None
        self.all_iteration_results = None
        self.buf = None
        self.initial_population = None
        self.x_0 = None
        self.seed = seed
        self.num_parents_mating = num_parents_mating
        self.parent_selection_type = parent_selection_type
        self.crossover_type = crossover_type
        self.crossover_probability = crossover_probability
        self.keep_elitism = keep_elitism
        self.mutation_type = mutation_type
        self.mutation_probability = mutation_probability
        self.strategy = strategy
        self.recombination = recombination
        self.mutation = mutation
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
            create_calibration_analysis(
                self.directory, self._input_data, self.model, self._port, self.timestep)
        ids = result.groupby(by=["leader", "follower", "recordingId"])
        for ident, chunk in ids:
            last_iteration = np.max(chunk["iteration"].values)
            best = result[result["iteration"] == last_iteration]
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
        bounds = Parameters.get_bounds_from_keys(self.param_keys)
        # cons = ModelParameters.get_constraints()
        managed_results = []
        managed_results_out = None
        all_managed_results = []
        all_managed_results_out = None
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
                                       remote_port=self._port,
                                       gui=False,
                                       timestep=self.timestep)
        results_path = (self.directory
                        / f"calibration_results_{self._optimization}.csv")
        all_results_path = (self.directory
                            / f"calibration_all_results_{self._optimization}.csv")
        try:
            results_path.unlink()
            all_results_path.unlink()
        except FileNotFoundError:
            pass
        for idx, identification in enumerate(identifications.values):
            self.identification = identification
            _LOGGER.info("Starting optimization %d/%d", idx + 1,
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
                _LOGGER.error("Stack Trace: \n%s", traceback.format_exc())
                continue
            iteration_results = self._tinker_results(identification, result)
            managed_results.append(iteration_results)
            if self._optimization in ["nsde", "gde3", "nsga2"]:
                all_iteration_results = self._tinker_results(
                    identification, result, True)
                all_managed_results.append(all_iteration_results)
            _LOGGER.info("Finished %s optimization for %s", self._optimization,
                         identification)
            # pylint: disable=W0212
            if self._optimization == "differential_evolution":
                fun = result.fun
                best_solution = result.x
            elif self._optimization in ["genetic_algorithm", "direct"]:
                fun = 1 / result["fun"]
                best_solution = result["x"]
            else:
                fun = result["fun"]
                best_solution = result["x"]
            _LOGGER.info("result.fun= %s", fun)
            optimal_params = {}
            for idx, key in enumerate(self.param_keys):
                optimal_params[key] = best_solution[idx]
            _LOGGER.info("parameters= %s", optimal_params)
            managed_results_out = pd.concat(managed_results)
            if len(managed_results_out) > 0:
                managed_results_out.reset_index(drop=True, inplace=True)
                managed_results_out.to_csv(results_path)
            if self._optimization in ["nsde", "gde3", "nsga2"]:
                all_managed_results_out = pd.concat(all_managed_results)
                if len(managed_results_out) > 0:
                    all_managed_results_out.reset_index(drop=True, inplace=True)
                    all_managed_results_out.to_csv(all_results_path)
        sumo_interface.release()
        del sumo_interface
        sumo_pipe.close()
        if managed_results_out is not None:
            return managed_results_out
        else:
            raise OptimizationFailed("No Optimization results.")

    def _tinker_results(self, identification, result, all_results=False):
        cols = (["iteration", "weightedError", "convergence"]
                + list(ModelParameters.get_defaults_dict().keys()))
        res = self.iteration_results if not all_results else self.all_iteration_results
        if self._optimization == "direct":
            self.log_iteration(result.x, **{"weighted_error": result.fun})
            iteration_results = pd.DataFrame(self.iteration_results,
                                             columns=cols)
        else:
            iteration_results = pd.DataFrame(self.iteration_results,
                                             columns=cols)
        iteration_results.loc[:, "leader"] = identification[0]
        iteration_results.loc[:, "follower"] = identification[1]
        iteration_results.loc[:, "recordingId"] = identification[2]
        iteration_results.loc[:, "algorithm"] = self._optimization
        iteration_results.loc[:, "pop-size"] = self.population_size
        iteration_results.loc[:, "objectives"] = ",".join(self.objectives)
        iteration_results.loc[:, "paramKeys"] = ",".join(self.param_keys)
        if self.weights:
            weights_str = [str(item) for item in self.weights]
        else:
            weights_str = ""
        iteration_results.loc[:, "weights"] = ",".join(weights_str)
        iteration_results.loc[:, "gof"] = self.gof
        return iteration_results

    def _run_optimization(self, bounds, sumo_interface, identification):
        global ITERATION
        global START_TIME
        identification = tuple(identification)
        _LOGGER.info("Starting %s for identification%s",
                     self._optimization, str(identification))
        ITERATION = 1
        self.iteration_results = []
        self.all_iteration_results = []
        self._prepare_optimization(sumo_interface, identification)
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
                    recombination=self.recombination,
                    mutation=self.mutation,
                    strategy=self.strategy,
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
                    bnds = ModelParameters.get_bounds_from_keys([key])[0]
                    std = bnds[1] - bnds[0] / 2
                    itera = _get_truncated_normal(
                        self.x_0[key], std, bnds[0], bnds[1])
                    self.initial_population[:self.population_size // 2,
                                            self.param_keys.index(key)] = (
                        itera.rvs(self.population_size // 2))
            dict_bounds = []
            for bound_tuple in bounds:
                dict_bounds.append(
                    {'low': bound_tuple[0], 'high': bound_tuple[1]})
            ga_instance = pygad.GA(
                num_generations=self._max_iter,
                num_parents_mating=int(
                    self.initial_population.shape[0] // self.num_parents_mating),
                initial_population=self.initial_population,
                sol_per_pop=self.population_size,
                fitness_func=target_factory(self.data, True),
                parent_selection_type=self.parent_selection_type,
                suppress_warnings=True,
                mutation_by_replacement=False,
                # mutation_num_genes=np.clip(6, len(self.param_keys)),
                crossover_type=self.crossover_type,
                crossover_probability=self.crossover_probability,
                keep_elitism=int(
                    self.initial_population.shape[0] // self.keep_elitism),
                gene_space=dict_bounds,
                # keep_parents=self.initial_population.shape[0]//2-1,
                fitness_batch_size=self.population_size,
                random_seed=self.seed,
                random_mutation_min_val=-1.0,
                random_mutation_max_val=1.0,
                mutation_type=self.mutation_type,
                # mutation_probability=[0.33, 0.10], # for adaptive
                mutation_probability=self.mutation_probability,
                save_best_solutions=True,
                on_fitness=fitness_callback_factory(self, "pygad"))
            ga_instance.run()
            solution, solution_fitness, _ = (
                ga_instance.best_solution())
            result = {"fun": solution_fitness, "x": solution}
            del ga_instance
        elif self._optimization == "nsga2":
            self.initial_population = random_population_from_bounds(
                bounds, self.population_size)
            # Set the population genes with estimations
            for key in ["minGap", "taccmax", "Mbegin", "Mflatness",
                        "speed_factor", "startupDelay"]:
                if key in self.param_keys:
                    bnds = ModelParameters.get_bounds_from_keys([key])[0]
                    std = bnds[1] - bnds[0] / 2
                    itera = _get_truncated_normal(
                        self.x_0[key], std, bnds[0], bnds[1])
                    self.initial_population[:self.population_size // 2,
                                            self.param_keys.index(key)] = (
                        itera.rvs(self.population_size // 2))
            dict_bounds = []
            for bound_tuple in bounds:
                dict_bounds.append(
                    {'low': bound_tuple[0], 'high': bound_tuple[1]})
            nsga_instance = pygad.GA(
                num_generations=self._max_iter,
                num_parents_mating=int(
                    self.initial_population.shape[0] // self.num_parents_mating),
                initial_population=self.initial_population,
                sol_per_pop=self.population_size,
                fitness_func=target_factory_nsga2(self.data, True),
                parent_selection_type="nsga2",  # "nsga2" or "tournament_nsga2"
                suppress_warnings=True,
                mutation_by_replacement=False,
                # mutation_num_genes=np.clip(6, len(self.param_keys)),
                crossover_type=self.crossover_type,
                crossover_probability=self.crossover_probability,
                keep_elitism=int(
                    self.initial_population.shape[0] // self.keep_elitism),
                gene_space=dict_bounds,
                # keep_parents=self.initial_population.shape[0]//2-1,
                fitness_batch_size=self.population_size,
                random_seed=self.seed,
                random_mutation_min_val=-1.0,
                random_mutation_max_val=1.0,
                mutation_type=self.mutation_type,
                # mutation_probability=[0.33, 0.10], # for adaptive
                mutation_probability=self.mutation_probability,
                save_best_solutions=True,
                on_fitness=fitness_callback_factory(self, "pymoo"))
            nsga_instance.run()
            solution, solution_fitness, _ = (
                nsga_instance.best_solution())
            solution_fitness = np.sum(solution_fitness)
            result = {"fun": solution_fitness, "x": solution}
            del nsga_instance
        elif self._optimization == "nsde":
            # variant can be the same as in DE, but takes the form "DE/selection/n/crossover"
            # crossover is either 'bin' or 'exp'
            # Selection variants are: 'ranked', 'rand', 'best', 'current-to-best', 'current-to-best', 'current-to-rand', 'rand-to-best'
            nsde = NSDE(pop_size=self.population_size, variant="DE/rand/1/bin", F=self.mutation,
                        CR=self.recombination, survival=RankAndCrowding(crowding_func="cd"))
            termination_multi = DefaultMultiObjectiveTermination(
                xtol=1e-8,
                cvtol=1e-8,
                ftol=1e-8,
                period=20,
                n_max_gen=self._max_iter,
            )
            low_bounds = []
            high_bounds = []
            for bound_tuple in bounds:
                low_bounds.append(bound_tuple[0])
                high_bounds.append(bound_tuple[1])
            problem = target_factory_mo_de(self.data,
                                           n_var=len(self.param_keys),
                                           n_obj=len(
                                               self.data["objective_function"]["objectives"]),
                                           n_ieq_constr=0,
                                           n_eq_constr=0,
                                           xl=low_bounds,
                                           xu=high_bounds)
            nsde_instance = minimize(problem,
                                     nsde,
                                     termination_multi,
                                     seed=self.seed,
                                     # deepcopy in save_history results in "cannot pickle '_io.TextIOWrapper' object"
                                     save_history=False,
                                     verbose=False,
                                     callback=self.log_iteration)
            single_error = np.sum(nsde_instance.F, axis=1)
            solution = nsde_instance.X[np.argmin(single_error)]
            solution_fitness = min(single_error)
            result = {"fun": solution_fitness, "x": solution}
        elif self._optimization == "gde3":
            # variant can be the same as in DE, but takes the form "DE/selection/n/crossover"
            # crossover is either 'bin' or 'exp'
            # Selection variants are: 'ranked', 'rand', 'best', 'current-to-best', 'current-to-best', 'current-to-rand', 'rand-to-best'
            gde3 = GDE3(pop_size=self.population_size, variant="DE/rand/1/bin", F=self.mutation,
                        CR=self.recombination, survival=RankAndCrowding(crowding_func="pcd"))
            termination_multi = DefaultMultiObjectiveTermination(
                xtol=1e-8,
                cvtol=1e-8,
                ftol=1e-8,
                period=20,
                n_max_gen=self._max_iter,
            )
            low_bounds = []
            high_bounds = []
            for bound_tuple in bounds:
                low_bounds.append(bound_tuple[0])
                high_bounds.append(bound_tuple[1])
            problem = target_factory_mo_de(self.data,
                                           n_var=len(self.param_keys),
                                           n_obj=len(
                                               self.data["objective_function"]["objectives"]),
                                           n_ieq_constr=0,
                                           n_eq_constr=0,
                                           xl=low_bounds,
                                           xu=high_bounds)
            gde3_instance = minimize(problem,
                                     gde3,
                                     termination_multi,
                                     seed=self.seed,
                                     # deepcopy in save_history results in "cannot pickle '_io.TextIOWrapper' object"
                                     save_history=False,
                                     verbose=False,
                                     callback=self.log_iteration)
            single_error = np.sum(gde3_instance.F, axis=1)
            solution = gde3_instance.X[np.argmin(single_error)]
            solution_fitness = min(single_error)
            result = {"fun": solution_fitness, "x": solution}
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
                                    self.meta_data, self.model)
        x_0 = ModelParameters.get_defaults_dict(*args)
        self.x_0 = x_0
        for key in self.param_keys:
            if key not in x_0:
                raise ValueError(
                    f"Unknown key {key} give nas param-keys option")
        x_optimizing = np.array([self.x_0[key] for key in self.param_keys])
        self.sumo_interface = sumo_interface
        proxy_objects = {
            "sumo_interface": sumo_interface,
            "cfmodel": self.model,
            "timestep": self.timestep,
            "project_path": self.project_path
        }
        # by default, the mops's are reduced to a single value, except for the
        # optimization algorithms that explicitly handle mulitple measures of performance
        # by themselves
        reduce = True
        if self._optimization in ["nsga2", "gde3", "nsde"]:
            reduce = False
        objective_function = {
            "identification": identification,
            "objectives": self.objectives,
            "weights": self.weights,
            "reduce_mmop2smop": reduce,
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
        if self._optimization == "nsde":
            single_error = np.sum(args[0].opt.get("F"), axis=1)
            params = args[0].opt.get("X")[np.argmin(single_error)]
        elif self._optimization == "gde3":
            single_error = np.sum(args[0].opt.get("F"), axis=1)
            params = args[0].opt.get("X")[np.argmin(single_error)]
        else:
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
        elif self._optimization == "nsde":
            weighted_error = min(single_error)
        elif self._optimization == "gde3":
            weighted_error = min(single_error)
        else:
            weighted_error = kwargs.get("weighted_error") or 0
        _LOGGER.debug("Current iteration %d with weighted_error=%f, "
                      " time taken:%f sec.\nConvergence:%f Parameters:%s",
                      ITERATION, weighted_error, time_taken,
                      kwargs.get("convergence") or -1, params_dict)
        self.iteration_results.append(
            {"iteration": ITERATION, "weightedError": weighted_error,
             "convergence": kwargs.get("convergence") or -1, **params_dict})
        if self._optimization == "nsde":
            for idx, i in enumerate(args[0].opt.get("F")):
                nondom_error = i
                nondom_params = args[0].opt.get("X")[idx]
                nondom_params_dict = self.x_0
                nondom_params_dict.update({key: value for key, value
                                           in zip(self.param_keys, nondom_params)})
                self.all_iteration_results.append(
                    {"iteration": ITERATION, "weightedError": nondom_error,
                     "convergence": kwargs.get("convergence") or -1, **nondom_params_dict})
        elif self._optimization == "gde3":
            for idx, i in enumerate(args[0].opt.get("F")):
                nondom_error = i
                nondom_params = args[0].opt.get("X")[idx]
                nondom_params_dict = self.x_0
                nondom_params_dict.update({key: value for key, value
                                           in zip(self.param_keys, nondom_params)})
                self.all_iteration_results.append(
                    {"iteration": ITERATION, "weightedError": nondom_error,
                     "convergence": kwargs.get("convergence") or -1, **nondom_params_dict})
        elif self._optimization == "nsga2":
            for idx, i in enumerate(kwargs.get("nondom")):
                nondom_error = 1 / i[1]
                nondom_params = kwargs.get("pop")[i[0]]
                nondom_params_dict = self.x_0
                nondom_params_dict.update({key: value for key, value
                                           in zip(self.param_keys, nondom_params)})
                self.all_iteration_results.append(
                    {"iteration": ITERATION, "weightedError": nondom_error,
                     "convergence": kwargs.get("convergence") or -1, **nondom_params_dict})
        _LOGGER.info(tqdm.format_meter(ITERATION, self._max_iter,
                                       elapsed=time_taken,
                                       prefix=str(self.identification)) +
                     f" f(x)={weighted_error:.6f}")
        ITERATION += 1
        # if we return True or when the convergence => 1 , then the polishing
        # step is initialized
        stop_criterion = False
        return stop_criterion

    def _prepare_sumo_project(self):
        project_path = self.project_path
        if not project_path.exists():
            project_path.mkdir()
        SumoProject.create_sumo(project_path, self.model,
                                self.population_size, self.timestep)


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
    diff = bounds_arr[:, 1] - bounds_arr[:, 0]
    for pop in range(population_size):
        # population[pop, i] = np.random.normal(bounds_arr[i, 0], std_devs[i], 1)
        if pop in std_devs:
            population[pop, :] = diff * np.random.normal(0.5,
                                                         std_devs[pop],
                                                         diff.shape[0])
            population[pop, :] = np.clip(population[pop, :], 0, 1)
        else:
            population[pop, :] = diff * np.random.uniform(0.0000000001,
                                                          0.9999999999,
                                                          diff.shape[0])
        population[pop, :] += bounds_arr[:, 0]
    return population


def fitness_callback_factory(item, module_name):
    """a factory for the fitness callback function"""

    def on_fitness_pygad(ga_instance: pygad.GA, solutions):
        """provided callback for fitness on pygad iteration"""
        best_solution = ga_instance.best_solutions[-1]
        best = np.max(ga_instance.last_generation_fitness)
        if best < np.max(solutions):
            best = np.max(solutions)
            best_solution = ga_instance.population[
                np.argmin(solutions).astype(int)]
        # TODO: log instead of 1 / 0
        kwargs = {"weighted_error": 1 / best}
        _ = item.log_iteration(best_solution, **kwargs)

    def on_fitness_pymoo(nsga_instance, solutions_fitness):
        """provided callback for fitness for pymoo iteration"""
        _, solution_fitness, _ = (
            nsga_instance.best_solution(nsga_instance.last_generation_fitness))
        best_solution = nsga_instance.best_solutions[-1]
        best = np.sum([1 / sol for sol in solution_fitness])
        normed_solutions = np.sum(
            [1 / sol for sol in solutions_fitness], axis=1)
        if best > np.min(normed_solutions):
            best = np.min(normed_solutions)
            best_solution = nsga_instance.population[
                np.argmin(normed_solutions).astype(int)]
        # TODO: log instead of 1 / 0
        pareto, _ = nsga_instance.non_dominated_sorting(solutions_fitness)
        kwargs = {"weighted_error": best,
                  "nondom": pareto[0], "pop": nsga_instance.population}
        _ = item.log_iteration(best_solution, **kwargs)

    if module_name == "pygad":
        return on_fitness_pygad
    elif module_name == "pymoo":
        return on_fitness_pymoo
    else:
        return NotImplementedError("Module not implemented")


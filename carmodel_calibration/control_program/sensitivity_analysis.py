# pylint: disable=C0103
"""file with class for sensitivity analysis"""

from pathlib import Path
import logging
from random import randint

from tqdm import tqdm
import numpy as np
from SALib import ProblemSpec
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from carmodel_calibration.fileaccess.parameter import ModelParameters, Parameters
from carmodel_calibration.sumo.simulation_module import SumoInterface
from carmodel_calibration.sumo.sumo_project import SumoProject
from carmodel_calibration.control_program.simulation_handler import SimulationHandler
from carmodel_calibration.optimization import (_vectorized_target)

_LOGGER = logging.getLogger(__name__)


def _chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
class SensitivityAnalysisHandler(SimulationHandler):
    """class to handle Sensitivity Analysis"""

    def __init__(self, directory: str, input_data: str,
                 num_workers: int = 100,
                 project_path: str = None,
                 param_keys: list = None,
                 weights: list = None,
                 num_samples: int = 1000,
                 model: str = "eidm",
                 remote_port: int = randint(8000, 9000),
                 timestep: float = 0.04,
                 gof: str = "rmse",
                 mop: list = ["distance"],
                 method: str = "sobol",
                 seed: int = None,
                 force_recalculation: bool = False):
        """
        :param directory:           output directory of calibration config and
                                    result data
        :param input_data:          directory to input data
        :param project_path:        path for the simulations
        :param num_workers:         the count of parallel vehicles to be
                                    simulated at once
        :param param_keys:          list of paramters that are to be optimized,
                                    if left emtpy, then all except
                                    [`emergencyDecel`,`stepping`]
                                    will be tuned
        :param model:               `eidm` by default, currently the only
                                    option
        :param num_samples:         number of samples for the analysis for each
                                    parameter
        :param gof:                 goodness-of-fit-function
        :param mop:                 list of measure of performances
        :param weights:             weights for the mop, only supply if
                                    multiple mop, by default all weights are
                                    equal
        :param method:              method for sensitivity analysis, by default
                                    `sobol`, other options [`fast`]
        :param seed:                seeding random variables
        :param force_recalculation: forces to recalculate the selection
        """
        if not param_keys:
            param_keys = list(Parameters.get_defaults_dict().keys())
        keys = []
        for key in param_keys:
            if Parameters.get_bounds_from_keys([key]):
                keys.append(key)
        param_keys = keys
        super().__init__(directory=directory,
                         input_data=input_data,
                         param_keys=param_keys,
                         num_workers=num_workers,
                         project_path=project_path)
        if project_path:
            self.delete_path = False
            self.project_path = Path(project_path)
        else:
            self.delete_path = True
            self.project_path = (self.directory
                                 / "sumo_project_sensitivity_analysis")
        if not self.project_path.exists():
            self.project_path.mkdir(parents=True)
        self.objectives = mop
        self.gof = gof
        self.model = model
        self.timestep = timestep
        self._port = remote_port
        self.weights = weights
        self.method = method.lower()
        self.num_samples = num_samples
        self.problem = None
        self.sampler, self.analyzer = None, None
        self.target_function = None
        self.data = None
        self.sumo_pipe = None
        self.identifications = None
        self.sumo_interface = None
        self.sensitivity = None
        self.manager = None
        self.samples = None
        self.results = None
        self.default_params = None
        self.force_recalculation = force_recalculation
        self.seed = seed
        self.results_frames = []
        self._prepare_sa()

    def __del__(self):
        super().__del__()
        if self.sumo_pipe:
            try:
                self.sumo_pipe.close()
            except IOError as exc:
                _LOGGER.error("Failed closing sumo_pipe with message %s",
                              str(exc))

    def _prepare_sa(self):
        if self.seed:
            np.random.seed(self.seed)
        self.prepare_selection(not self.force_recalculation)
        if not self.identifications:
            identifications = (
                self.selection_data[~self.selection_data["follower"].isna()]
            [["leader", "follower", "recordingId"]].values)
            np.random.shuffle(identifications)
            self.identifications = identifications[:5]
        self._define_problem()
        self.sample, self.analyze = (
            _sensitivity_analysis_factory(self.method, self.num_samples,
                                          self.problem, False, self.seed))

    def _define_target_data(self, parallel_vehicles, identification):
        self.sumo_pipe = open(
            str(self.project_path / "traci.log"), "w",
            encoding="utf-8")
        cfmodel = ModelParameters.get_defaults_dict()
        self.default_params = cfmodel
        if self.sumo_interface is None:
            SumoProject.create_sumo(self.project_path, self.model, parallel_vehicles, self.timestep)
            sumo_interface = SumoInterface(self.project_path, self._input_data,
                                        remote_port=self._port, gui=False, file_buffer=self.sumo_pipe, timestep=self.timestep)
            self.sumo_interface = sumo_interface
        proxy_objects = {
                "sumo_interface": self.sumo_interface,
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
                "default_parameters": self.default_params,
                "param_names": self.param_keys
        }
        data.update(proxy_objects)
        self.data = data

    def _define_problem(self):
        self.problem = {
            "num_vars": len(self.param_keys),
            "names": self.param_keys,
            "groups": None,
            "bounds": Parameters.get_bounds_from_keys(self.param_keys),
            "outputs": ["weighted_error"]
        }

    def _sample_model(self):
        self.samples = self.sample()
        _LOGGER.info("Created sample data for method %s with %d samples "
                     "resulting in %d calculations in total, %d simulations in"
                     " total.",
                     self.method,
                     int(2 ** np.ceil(np.log2(self.num_samples))),
                     self.samples.samples.shape[0] * 5,
                     np.ceil(self.samples.samples.shape[0]/self.num_workers)*5)

    def _evaluate(self):
        _LOGGER.info("Starting evaluation for samples...")
        self.results = None
        results_all = []
        total = np.ceil(self.samples.samples.shape[0]/self.num_workers).astype(int)*5
        initial = 0
        for identification in self.identifications:
            self._define_target_data(self.num_workers, list(identification))
            results = []
            kwargs = {"initial": initial,
                      "total": total}
            for sample_chunk in tqdm(_chunker(self.samples.samples,
                                              self.num_workers),**kwargs):
                chunk_result = _vectorized_target(sample_chunk.T, self.data)
                # calculate weighted error for all samples on the specific
                # leader-follower pair
                results.extend(list(chunk_result))
            results_all.append(results)
            initial += np.ceil(self.samples.samples.shape[0]/self.num_workers)
        # average sample results for all leader-follower pairs
        results = np.array(results_all).T
        results = np.mean(results, axis=1)
        self.results = np.array(results)
        _LOGGER.info("Finished sensitivity analysis.")

    def _analyze(self):
        _LOGGER.info("Analyzing results from samples.")
        self.sensitivity = self.analyze(self.results)
        sensitivitiy_df = self.sensitivity.to_df()
        if self.method == "fast":
            sensitivitiy_df = [sensitivitiy_df]
        self.results_frames = []
        for df in sensitivitiy_df:
            df = df.sort_values(by=df.columns[0])
            name = df.columns[0]
            param_names = list(df.index)
            if isinstance(param_names[0], tuple):
                param_names = [",".join(item) for item in param_names]
            else:
                param_names = df.index
            df["parameterName"] = param_names
            df.to_csv(self.directory / f"sensitivity_results{name}.csv",
                            index=False)
            self.results_frames.append(df)

    def _show_plots(self):
        rc_file = Path(__file__).parents[1] / "data_config/matplotlib.rc"
        plt.style.use(rc_file)
        figures = []
        if self.method == "fast":
            axes = self.sensitivity.plot()
            plt.tight_layout()
            plt.close("all")
            figures.append(axes.figure)
        if self.method == "sobol":
            figs = [_plot_bar(item) for item in self.results_frames]
            plt.tight_layout()
            plt.close("all")
            figures += figs
        pdf = matplotlib.backends.backend_pdf.PdfPages(
            self.directory / f"sensitivity_analysis_{self.method}.pdf")
        for figure in figures:
            pdf.savefig(figure)
        pdf.close()

    def run_analysis(self):
        """run the sensitivity analysis routine"""
        self._sample_model()
        self._evaluate()
        self._analyze()
        self._show_plots()

def _sensitivity_analysis_factory(name, num_samples, problem,
                                  calc_second_order, seed=None):
    sp = ProblemSpec(**problem)
    if name.lower() == "fast" and calc_second_order:
        _LOGGER.warning(
            "Invalid option: %s with `calc_second_order` specified",
            name)
    # pylint: disable=E1101
    def sobol_sampler():
        n_calc=int(2 ** np.ceil(np.log2(num_samples)))
        return sp.sample_saltelli(
            n_calc, calc_second_order=calc_second_order)
    # pylint: disable=E1101
    def fast_sampler():
        n_calc = num_samples // len(problem["names"])
        interference_param = 4
        if n_calc <= 4 * interference_param ** 2:
            n_calc_min = 4 * interference_param ** 2
            num_samples_min = n_calc_min * len(problem["names"])
            raise ValueError(
                f"Number of samples must be atleast {num_samples_min}")
        return sp.sample_fast(n_calc, M=interference_param, seed=seed)

    # pylint: disable=E1101
    def sobol_analyzer(results: np.ndarray):
        sp.set_results(results)
        return sp.analyze_sobol(print_to_console=True,
                                calc_second_order=False)

    # pylint: disable=E1101
    def fast_analyzer(results: np.ndarray):
        sp.set_results(results)
        return sp.analyze_fast(print_to_console=True)

    samplers = {"sobol": sobol_sampler, "fast": fast_sampler}
    analyzers = {"sobol": sobol_analyzer, "fast": fast_analyzer}
    sample = samplers[name.lower()]
    analyze = analyzers[name.lower()]
    return sample, analyze

def _plot_bar(results):
    names = results["parameterName"].values
    sens_index = results.iloc[:, 0].values
    sens_index_error = results.iloc[:, 1].values
    fig, ax = plt.subplots(figsize=(12, 12))
    x_range = np.arange(len(results))
    ax.set_title(list(results.columns)[0])
    ax.bar(x_range, sens_index, yerr=sens_index_error, align="center")
    ax.set_xticks(x_range)
    ax.set_xticklabels(names, rotation=45, ha="right")
    fig.tight_layout()
    return fig

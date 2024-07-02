"""command line entry point for pycarmodel_calibration tool"""

import os
import sys
import argparse
from pathlib import Path
import logging
import json
from random import randint

sys.path.append(str(Path(__file__).parents[1]))

from carmodel_calibration.control_program.calibration_handling import (
    CalibrationHandler)
from carmodel_calibration.control_program.sensitivity_analysis import (
    SensitivityAnalysisHandler)
from carmodel_calibration.logging_config import configure_logging
from carmodel_calibration.exceptions import MissingRequirements
from carmodel_calibration.control_program.calibration_analysis import (
    create_calibration_analysis)
from carmodel_calibration.data_integration.data_set import DataSet

def _chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

configure_logging()
_LOGGER = logging.getLogger(__name__)

def main():
    """main entry point"""
    if os.environ.get("SUMO_HOME") is None:
        raise MissingRequirements(
            "Could not find SUMO_HOME environment Variable.")
    else:
        if not Path(os.environ.get("SUMO_HOME")).exists():
            raise MissingRequirements(
                "Sumo Environment Variable points to non-existing Path:"
                + os.environ.get("SUMO_HOME"))
    args_to_parse = sys.argv[1:]
    parser = _get_parser(args_to_parse)

    args = parser.parse_args(args_to_parse)
    if args.verbose:
        configure_logging("debug")
    if args.action == "calibrate":
        _calibrate(args)
    elif args.action == "create_reports":
        create_calibration_analysis(Path(args.output_dir), Path(args.data_dir), args.model.lower(), args.remote_port, args.timestep)
    elif args.action == "read_matrix":
        _LOGGER.error("Not implemented yet.")
        raise NotImplementedError
    elif args.action == "create_matrix":
        _LOGGER.error("Not implemented yet.")
        raise NotImplementedError
    elif args.action == "create_data_config":
        kwargs = _kwargs_prompt()
        if kwargs:
            DataSet.create_data_config(Path(args.file_path), overwrite=False,
                                       **kwargs)
    elif args.action == "sensitivity_analysis":
        _perform_sensitivity_analysis(args)

def _get_parser(args_to_parse):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    action_help = ("`create_reports` will create reports in an existing output"
        "-directory with existing calibration result data, "
        "`calibrate` will get the selection and calibrate it,"
        "`get_selection` will only output the selection to the output"
        " directory, `read_matrix` will read a json file with all the"
        " arguments for calibration runs, allows multiple calibrations"
        " at once. `sensitivity_analysis` can be performed as well.")
    action_choices = ["calibrate", "get_selection", "create_reports",
                      "read_matrix", "create_data_config",
                      "create_matrix", "sensitivity_analysis"]
    parser.add_argument("--action",
                        choices=action_choices,
                        help=action_help)
    args_to_check = ["--action=calibrate", "--action=get_selection",
                     "--action=create_reports",
                     "--action=sensitivity_analysis"]
    if _check_for_args(args_to_parse, args_to_check):
        # TODO create resume flag
        parser.add_argument("data_dir",
                            metavar="Data-Directory",
                            help="str, Path to where the measured data is.")
        parser.add_argument("output_dir",
                            metavar="Output-Directory",
                            help="str, Path to where results should be"
                            " stored.")
    args_to_check = ["--action=create_reports"]
    remote_port_help = "Remote port for Connection to SUMO, Default: randint(8000, 9000)"
    parser.add_argument("--remote_port",
                        type=int,
                        default=randint(8000, 9000),
                        help=remote_port_help)
    timestep_help = "Time step size of the input data and the SUMO simulation in seconds, Default: 0.04"
    parser.add_argument("--timestep",
                        type=float,
                        default=0.04,
                        help=timestep_help)
    num_parents_mating_help = "num_parents_mating"
    parser.add_argument("--num_parents_mating",
                        type=float,
                        default=2.,
                        help=num_parents_mating_help)
    parent_selection_type_help = "parent_selection_type"
    parser.add_argument("--parent_selection_type",
                        type=str,
                        default="sss",
                        help=parent_selection_type_help)
    crossover_type_help = "crossover_type"
    parser.add_argument("--crossover_type",
                        type=str,
                        default="uniform",
                        help=crossover_type_help)
    crossover_probability_help = "crossover_probability"
    parser.add_argument("--crossover_probability",
                        type=float,
                        default=0.4,
                        help=crossover_probability_help)
    keep_elitism_help = "keep_elitism"
    parser.add_argument("--keep_elitism",
                        type=float,
                        default=4.,
                        help=keep_elitism_help)
    mutation_type_help = "mutation_type"
    parser.add_argument("--mutation_type",
                        type=str,
                        default="random",
                        help=mutation_type_help)
    mutation_probability_help = "mutation_probability"
    parser.add_argument("--mutation_probability",
                        type=float,
                        default=0.33,
                        help=mutation_probability_help)
    strategy_help = "strategy"
    parser.add_argument("--strategy",
                        type=str,
                        default="best1bin",
                        help=strategy_help)
    recombination_help = "recombination"
    parser.add_argument("--recombination",
                        type=float,
                        default=0.7,
                        help=recombination_help)
    mutation_help = "mutation"
    parser.add_argument("--mutation",
                        type=_tuple_type,
                        default=(0.5,1.0),
                        help=mutation_help)
    if _check_for_args(args_to_parse, args_to_check):
        model_help = "Model under test, e.g. `eidm`."
        parser.add_argument("--model",
                            type=str,
                            default="eidm",
                            help=model_help)
    args_to_check = ["--action=calibrate"]
    if _check_for_args(args_to_parse, args_to_check):
        subparsers = parser.add_subparsers(
            title="Calibration Mode",
            dest="calibration_mode",
            help="Mode for optimization, "
            "differential_evolutions solves best")
        de_parser = subparsers.add_parser(
            "differential_evolution",
            help="Differential Evolution optimization")
        ga_parser = subparsers.add_parser(
            "genetic_algorithm", help="Genetic Algorithm optimization")
        nsga2_parser = subparsers.add_parser(
            "nsga2", help="Non-dominated Sorting Genetic Algorithm II optimization")
        nsde_parser = subparsers.add_parser(
            "nsde", help="Non-dominated Sorting Differential Evolution optimization")
        gde3_parser = subparsers.add_parser(
            "gde3", help="Generalized Differential Evolution 3 optimization")
        direct_parser = subparsers.add_parser(
            "direct", help="direct optimization")
        model_help = "Model under test, e.g. `eidm`."
        param_keys = ("speedFactor,minGap,accel,"
            "decel,startupDelay,tau,delta,"
            "treaction,taccmax,Mflatness,Mbegin")
        param_help = ("comma separated list of parameters e.g. "
        "`speedFactor,minGap,...`")
        force_selection_help = ("If the flag is passed, then the selection "
                                "is forced to be recalculateld.")
        for calibration_parser in [de_parser, ga_parser, nsga2_parser, nsde_parser, gde3_parser]:
            calibration_parser.add_argument("--population-size",
                                            type=int,
                                            default=5,
                                            help="Size of the population"
                                            "does not apply to direct.")
        for calibration_parser in [de_parser, ga_parser, nsga2_parser, nsde_parser, gde3_parser, direct_parser]:
            calibration_parser.add_argument("--seed",
                                            type=int,
                                            default=None,
                                            help="Seed for random variables.")
            calibration_parser.add_argument("--max-iter",
                                            type=int,
                                            default=1,
                                            help="Number of max iterations.")
            calibration_parser.add_argument("--model",
                                            type=str,
                                            default="eidm",
                                            help=model_help)
            calibration_parser.add_argument("--param-keys",
                                            type=str,
                                            default=param_keys,
                                            help=param_help)
            calibration_parser.add_argument("--force-selection",
                                            action="store_true",
                                            help=force_selection_help)
            calibration_parser.add_argument("--no-report",
                                            action="store_true",
                                            help="Will not create reports in"
                                            " the output directory.")
            calibration_parser.add_argument("--gof",
                                            default="rmse",
                                            type=str,
                                            help="goodness-of-fit-function,"
                                            "one of `rmse`, `nrmse`, `rmsep`, `theils_u`"
                                            )
            calibration_parser.add_argument("--mop",
                                            default="distance",
                                            type=str,
                                            help="measure of performance, "
                                            "comma separated list, default:"
                                            "`distance`, options are `distance`"
                                            "`speed`, `acceleration`")
    args_to_check = ["--action=read_matrix", "--action=create_matrix",
                     "--action=create_data_config"]
    if _check_for_args(args_to_parse, args_to_check):
        parser.add_argument("file_path",
                            metavar="path/to/file.json",
                            help="str, Path to where config is targeted.")
    args_to_check = ["--action=sensitivity_analysis"]
    if _check_for_args(args_to_parse, args_to_check):
        subparsers = parser.add_subparsers(
            title="Sensitivity Analysis",
            dest="sensitivity_method",
            help="Method for sensitivity analysis, "
            "default is `fast`")
        fast_parser = subparsers.add_parser(
            "fast",
            help="Fourier Amplitude Sampling Testing")
        sobol_parser = subparsers.add_parser(
            "sobol", help="variance based sensitivity analysis")
        model_help = "Model under test, e.g. `eidm`."
        param_help = ("comma separated list of parameters e.g. "
        "`speedFactor,minGap,...`")
        force_selection_help = ("If the flag is passed, then the selection "
                                "is forced to be recalculateld.")
        for sa_analsis_parser in [fast_parser, sobol_parser]:
            sa_analsis_parser.add_argument("--num-samples",
                                            type=int,
                                            default=80,
                                            help="Sample size per parameter")
            sa_analsis_parser.add_argument("--num-workers",
                                            type=int,
                                            default=500,
                                            help="Number of params per sim")
            sa_analsis_parser.add_argument("--seed",
                                            type=int,
                                            default=None,
                                            help="Seed for random variables.")
            sa_analsis_parser.add_argument("--model",
                                            type=str,
                                            default="eidm",
                                            help=model_help)
            sa_analsis_parser.add_argument("--force-selection",
                                action="store_true",
                                help=force_selection_help)
            sa_analsis_parser.add_argument("--param-keys",
                                            type=str,
                                            default=None,
                                            help=param_help)
            sa_analsis_parser.add_argument("--gof",
                                            default="rmse",
                                            type=str,
                                            help="goodness-of-fit-function")
            sa_analsis_parser.add_argument("--mop",
                                            default="distance",
                                            type=str,
                                            help="measure of performance, "
                                            "comma separated list, default:"
                                            "`distance`")
    return parser

def _calibrate(args):
    out_path = Path(args.output_dir)
    if not out_path.exists():
        out_path.mkdir(parents=True)
    handler = None
    if args.calibration_mode != "direct":
        population = args.population_size
    else:
        population = 1
    mop = args.mop.replace("\"", "").split(",")
    handler = CalibrationHandler(out_path, args.data_dir,
                            args.model.lower(),
                            args.remote_port,
                            args.timestep,
                            args.calibration_mode,
                            max_iter=args.max_iter,
                            param_keys=args.param_keys.split(","),
                            population_size=population,
                            force_recalculation=args.force_selection,
                            num_parents_mating=args.num_parents_mating,
                            parent_selection_type=args.parent_selection_type,
                            crossover_type=args.crossover_type,
                            crossover_probability=args.crossover_probability,
                            keep_elitism=args.keep_elitism,
                            mutation_type=args.mutation_type,
                            mutation_probability=args.mutation_probability,
                            strategy=args.strategy,
                            recombination=args.recombination,
                            mutation=args.mutation,
                            gof=args.gof,
                            mop=mop,
                            seed=args.seed)
    if handler:
        handler.run_calibration_cli(create_reports=not args.no_report)
        del handler

def _perform_sensitivity_analysis(args):
    out_path = Path(args.output_dir)
    if not out_path.exists():
        out_path.mkdir(parents=True)
    handler = None
    mop = args.mop.replace("\"", "").split(",")
    if args.param_keys:
        param_keys = args.param_keys.replace("\"", "")
        param_keys = param_keys.split(",")
    else:
        param_keys = None
    handler = SensitivityAnalysisHandler(
        directory=out_path,
        input_data=args.data_dir,
        num_workers=args.num_workers,
        project_path=None,
        param_keys=param_keys,
        weights=None,
        num_samples=args.num_samples,
        model=args.model,
        remote_port=args.remote_port,
        timestep=args.timestep,
        gof=args.gof,
        mop=mop,
        method=args.sensitivity_method,
        seed=args.seed,
        force_recalculation=args.force_selection
    )
    if handler:
        handler.run_analysis()

def _check_for_args(args_to_parse, args_to_check):
    for arg in args_to_check:
        # pylint: disable=W0640
        condition = (map(
            lambda x: arg in x, args_to_parse))
        if True in list(condition):
            return True
    return False

def _tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_float = map(float, strings.split(","))
    return tuple(mapped_float)

def _kwargs_prompt():
    keys = {
        "data_file_regex": ("Regex Pattern for the data files", ".*_WorldPositions.csv"),
        "skip_corrupt": ("If true then invalid data is just skipped", "y"),
        "intersection_points_xy": (
            "Intersection crossing points points in [x1, y1, x2, y2]",
            "-7.13,11,-7.13,-12"),
        "intersection_points_lonlat":
            ("Intersection crossing points points in [x1, y1, x2, y2]",
             "9.191343,48.788692,9.191638,48.788567"),
        "file_specific_options": ("File specific options[y/n]", "n"),
        "input_format": ("Filler", ""),
        "output_format": ("Filler", ""),
        "df_config": ("df config options, like changing column names[y/n]",
                      "n"),
        "df_kwargs": ("filler", ""),
        "metafile_replace": (".", "_meta.")
    }
    kwargs = {}
    for key, (prompt, default) in keys.items():
        new_prompt = key + "\n\t" + prompt + f" \n\tdefault is `{default}`:\n"
        resp = input(new_prompt).strip()
        if key == "file_specific_options":
            if resp.lower() == "y":
                file_specific_options = []
                while(input(
                    "Add another file specific option?y/n:").lower()=="y"):
                    option = {}
                    recording_id = int(float(input("recordingId (int):")))
                    if recording_id == "":
                        print("Aborted...")
                        continue
                    option["recordingId"] = recording_id
                    disregard_time = input(
                        "disregard_time comma separated in seconds:")
                    if disregard_time != "":
                        disregard_time = [float(item) for item in
                                        disregard_time.split(",")]
                        disregard_time = _chunker(disregard_time, 2)
                        disregard_time = [item for item in disregard_time]
                        option["disregard_time"] = disregard_time
                    coordinate_system = input(
                        "coordinate_system either `lon,lat` or `x,y`:")
                    if coordinate_system != "":
                        coordinate_system = [item for item in
                                        coordinate_system.split(",")]
                        option["coordinate_system"] = coordinate_system
                    file_specific_options.append(option)
                kwargs.update({key: file_specific_options})

        elif key == "df_config":
            if resp.lower() == "y":
                df_config = {}
                while(input(
                    "Add another df_config?y/n:").lower()=="y"):
                    original_name = input(
                        "Original name of the column to modify e.g."
                        "`ID`(str):").strip()
                    new_name = input(
                        "New name name of the column to modify e.g."
                        "`trackId`(str):").strip()
                    is_numeric = input(
                        "is the column numeric ?y/n:").strip()
                    if is_numeric != "":
                        is_numeric = is_numeric.lower() == "y"
                    else:
                        is_numeric = None

                    df_config.update({original_name:
                        {
                            "new_name": new_name,
                            "is_numeric": is_numeric
                            }
                        }
                    )
                kwargs.update({"df_config": df_config})
        elif (key == "intersection_points_xy" or
            key == "intersection_points_lonlat"):
            if resp.strip() == "":
                resp = default
            resp = [float(item) for item in resp.split(",")]
            kwargs.update({key: resp})
        else:
            if resp.strip().lower() == "y":
                resp = True
            elif resp.strip().lower() == "n":
                resp = False
            elif resp.strip() == "":
                resp = None
            kwargs.update({key: resp})
    drop_keys = []
    for key, value in kwargs.items():
        if value == "" or value is None:
            drop_keys.append(key)
    for key in drop_keys:
        kwargs.pop(key, None)
    print("#"* 50 + "\n")
    kwargs_string = json.dumps(kwargs, indent=4)
    print(f"The current configuration is\n{kwargs_string}\n")
    resp = input("Continue with current Configuration?y/n:")
    if resp.strip() != "y":
        return None
    return kwargs
if __name__ == "__main__":
    # __spec__ = None # May be uncommented for development
    main()

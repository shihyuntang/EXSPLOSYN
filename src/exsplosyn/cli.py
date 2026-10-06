"""Command-line entry point for EXSPLOSYN simulations."""

import argparse
from importlib.metadata import version
import tomllib
from pathlib import Path
from time import perf_counter

from .configuration import DEFAULT_CONFIG_FILE, PROJECT_ROOT
from .simulation_setup import init_setup_cfg
from .simulation import (
    main_model,
    prepare_simulation_spectra,
    setup_logging,
)


def _project_version():
    """Read checkout metadata or the installed distribution version."""
    if PROJECT_ROOT is not None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as source:
            return tomllib.load(source)["project"]["version"]
    return version("exsplosyn")


def main(argv=None):
    """Run EXSPLOSYN using command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="EXSPLOSYN",
        description="EXOplanetary Spotted Photosphere LOw-mass SYNthesizer.\n"
        "High-resolution spectral simulation for spotted low-mass stars.",
        epilog="Example (source checkout, default configuration):\n"
        "  python main.py -o my_run\n\n"
        "Contact: shihyun.tang@austin.utexas.edu, tang.shih.yun@gmail.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_project_version()}"
    )

    config_help = (
        "Configuration filename in the project's configs directory, or an explicit "
        "file path. If no extension is given, .cfg is added. "
        "Relative paths use the working directory. Default: %(default)s."
        if DEFAULT_CONFIG_FILE is not None
        else "Path to the configuration file (required). If no extension is given, .cfg is added. "
        "Relative paths use the working directory."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_FILE.name if DEFAULT_CONFIG_FILE is not None else None,
        required=DEFAULT_CONFIG_FILE is None,
        help=config_help,
    )

    parser.add_argument(
        "--paths",
        metavar="NAME",
        default=None,
        help="Use [paths:NAME] to override the base [paths] directories. "
        "If omitted, use [paths] only.",
    )

    parser.add_argument(
        "-o",
        "--output-suffix",
        "-out",
        help="Suffix appended to the output folder prefix, before the path profile.",
        default=None,
    )

    parser.add_argument(
        "--quiet",
        "-silence",
        action="store_true",
        help="Suppress progress bars and informational messages; keep warnings and errors.",
    )

    parser.add_argument(
        "--mode", choices=("simulation", "fitting", "debug"), default="simulation",
        help="Run a simulation, return fitting results in RAM, or export debug data.",
    )

    args = parser.parse_args(argv)
    run_start = perf_counter()
    mode = args.mode
    quiet = mode == "fitting" or args.quiet
    logger = setup_logging(debug=mode == "debug", quiet=quiet)

    logger.info("EXSPLOSYN v%s", _project_version())

    (
        star,
        bands,
        save_dir,
        nrho,
        grid_size,
        instrument_res,
        cpu_use,
        plot_flag,
        save_spectra,
        spectral_paths,
        phases,
        active_regions_number,
        active_region_box,
        filters,
        spectrum_vel_step,
    ) = init_setup_cfg(
        logger, args.config, output_suffix=args.output_suffix, path_profile=args.paths,
        mode=mode
    )

    logger.info("\nPreparing simulation spectra and RV templates...")
    band_spectra = prepare_simulation_spectra(
        bands, star, plot_flag, spectrum_vel_step, mode=mode
    )

    # Run the simulation for the configured bands and rotation phases.
    result = main_model(
        logger,
        bands,
        nrho,
        grid_size,
        star,
        phases,
        band_spectra,
        filters,
        active_regions_number,
        active_region_box,
        spectral_paths,
        instrument_res,
        cpu_use,
        save_dir,
        save_spectra,
        plot_flag,
        spectrum_vel_step,
        silence_pbar=quiet,
        mode=mode,
    )
    logger.info("\nSimulation complete: %.2f s", perf_counter() - run_start)
    if mode != "fitting":
        logger.info("Output directory: %s", Path("outputs") / save_dir)
    return result

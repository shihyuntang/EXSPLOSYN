"""Command-line entry point for EXSPLOSYN simulations."""

import argparse
from importlib.metadata import version
import tomllib
from pathlib import Path
from time import perf_counter

from .configuration import DEFAULT_CONFIG_FILE, PROJECT_ROOT


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
        "  python main.py --output-dir outputs/my_run\n\n"
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
        "--output-dir",
        metavar="PATH",
        help="Save directly into this folder instead of output_root/run_name. "
        "Relative paths start from your terminal folder; no names are added.",
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

    # Help/version exit during parsing, before loading scientific dependencies.
    from .simulation_setup import init_setup_cfg
    from .simulation import (
        main_model,
        prepare_simulation_spectra,
        setup_logging,
    )

    run_start = perf_counter()
    mode = args.mode
    quiet = mode == "fitting" or args.quiet
    logger = setup_logging(debug=mode == "debug", quiet=quiet)

    logger.info("EXSPLOSYN v%s", _project_version())

    (
        star,
        bands,
        output_dir,
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
        logger, args.config, output_dir=args.output_dir, path_profile=args.paths,
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
        output_dir,
        save_spectra,
        plot_flag,
        spectrum_vel_step,
        silence_pbar=quiet,
        mode=mode,
    )
    elapsed = perf_counter() - run_start
    if elapsed >= 120:
        # Round before splitting so seconds never display as 60.00.
        minutes, seconds = divmod(round(elapsed, 2), 60)
        duration = f"{int(minutes)} min {seconds:05.2f} s"
    else:
        duration = f"{elapsed:.2f} s"
    logger.info("\nSimulation complete: %s", duration)
    if mode != "fitting":
        logger.info("%19s: %s", "Model ID", star.model_hash[:10])
        logger.info("%19s: %s", "Output directory", Path(output_dir))
    return result

"""Run a performance benchmark from a source checkout.

Each case prepares its spectra once in a fresh Python process, then repeats the
same model in fitting mode. Downloads and preparation are excluded from the
steady-state timings. No results are uploaded automatically.
"""

import argparse
import configparser
from contextlib import contextmanager
from copy import copy
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from threading import Event, Thread
from time import perf_counter
import tomllib

ROOT = Path(__file__).resolve().parent
# Use the source checkout without requiring an editable/package installation.
sys.path.insert(0, str(ROOT / "src"))
from exsplosyn.leaderboard import load_catalog, show_rankings
CASES = ("photometry", "phx_rv", "synmast_feoh")
CASE_LABELS = {
    "photometry": "Photometry",
    "phx_rv": "PHX RV",
    "synmast_feoh": "SYNMAST FeOH",
}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
PACKAGES = (
    "numpy",
    "scipy",
    "numba",
    "llvmlite",
    "ldtk",
    "pyphot",
    "pandas",
    "astropy",
    "psutil",
    "threadpoolctl",
)


def command_output(command):
    """Read optional machine metadata without failing on unsupported platforms."""
    try:
        return subprocess.check_output(
            command, stderr=subprocess.DEVNULL, text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def system_info(psutil):
    """Collect hardware descriptions, excluding hostnames and serial numbers."""
    cpu = platform.processor() or None
    model = None
    translated = None
    if sys.platform == "darwin":
        cpu = command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
        model = command_output(["sysctl", "-n", "hw.model"])
        translated = command_output(["sysctl", "-n", "sysctl.proc_translated"]) == "1"
    elif sys.platform.startswith("linux"):
        try:
            entries = Path("/proc/cpuinfo").read_text().splitlines()
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in entries
                if line.startswith("model name")
            )
        except (OSError, StopIteration):
            pass
        try:
            model = Path("/sys/devices/virtual/dmi/id/product_name").read_text().strip()
        except OSError:
            pass
    logical = psutil.cpu_count(logical=True)
    try:
        available = len(psutil.Process().cpu_affinity())
    except (AttributeError, NotImplementedError, OSError, psutil.Error):
        available = logical
    return {
        "computer_model": model,
        "cpu_model": cpu,
        "os": "macOS" if sys.platform == "darwin" else platform.system(),
        "os_version": platform.mac_ver()[0] or platform.release(),
        "kernel_version": platform.release(),
        "architecture": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": logical,
        "available_logical_cpus": available,
        "ram_gib": psutil.virtual_memory().total / 1024**3,
        "rosetta_translated": translated,
        "python": platform.python_version(),
    }


def software_info():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project_version = tomllib.load(handle)["project"]["version"]
    status = command_output(["git", "-C", str(ROOT), "status", "--porcelain"])
    return {
        "exsplosyn": project_version,
        "packages": packages,
        "git_commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
    }


def read_config(path):
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    with Path(path).open(encoding="utf-8") as handle:
        cfg.read_file(handle)
    return cfg


def case_config(source, case, workers, *, paths_config=None):
    """Change only the selected observables and worker count for each case."""
    cfg = read_config(source)
    if paths_config is not None:
        # Reuse working demo paths without changing the fixed benchmark physics.
        path_settings = read_config(paths_config)
        for section in list(cfg.sections()):
            if section == "paths" or section.startswith("paths:"):
                cfg.remove_section(section)
        for section in path_settings.sections():
            if section == "paths" or section.startswith("paths:"):
                cfg.add_section(section)
                for key, value in path_settings.items(section, raw=True):
                    cfg.set(section, key, value)
    cfg.set("main", "cpu", str(workers))
    cfg.set(
        "spectroscopic",
        "enabled",
        {"photometry": "", "phx_rv": "phx_rv", "synmast_feoh": "FeOH"}[case],
    )
    if case != "photometry":
        cfg.set("photometry", "filters", "None")
    for key in ("make_plots", "make_gif_animation", "save_spectra"):
        cfg.set("output", key, "0")
    return cfg


def public_config(cfg):
    """Retain effective settings without including local spectral-library paths."""
    result = {
        section: dict(cfg.items(section, raw=True))
        for section in cfg.sections()
        if not section.startswith("paths")
    }
    for section, key in (("phase_sampling", "phase_file"), ("rv_ccf", "template_file")):
        if result.get(section, {}).get(key):
            result[section][key] = Path(result[section][key]).name
    return result


def feoh_available(cfg, profile):
    paths = dict(cfg.items("paths"))
    if profile:
        paths.update(dict(cfg.items(f"paths:{profile}")))
    key = cfg.get("band:FeOH", "library_path")
    path = Path(paths[key]).expanduser()
    return path.is_dir() and next(path.rglob("*.prf.gz"), None) is not None


@contextmanager
def sampled_memory(psutil):
    """Sample RSS summed across this case process and its live descendants.

    This is not unique physical RAM: shared pages can be counted more than once.
    Sampling at 0.1 s can miss brief peaks, so report it as a sampled RSS sum.
    """
    process = psutil.Process()
    stopped = Event()
    result = {"peak_rss_sum_mib": None, "children_accessible": True}

    def sample():
        try:
            processes = [process, *process.children(recursive=True)]
        except (psutil.Error, OSError):
            processes = [process]
            result["children_accessible"] = False
        total = 0
        sampled = False
        for item in processes:
            try:
                total += item.memory_info().rss
                sampled = True
            except (psutil.Error, OSError):
                result["children_accessible"] = False
        if sampled:
            result["peak_rss_sum_mib"] = max(
                result["peak_rss_sum_mib"] or 0.0, total / 1024**2
            )

    def monitor():
        while not stopped.wait(0.1):
            sample()

    sample()
    thread = Thread(target=monitor, daemon=True)
    thread.start()
    try:
        yield result
    finally:
        stopped.set()
        thread.join()
        sample()


def checked_observables(results, phases, case, np):
    """Check scientific outputs, allowing NaN only in unused diagnostic fields."""
    observables = {}
    for band, columns in results.items():
        if not np.array_equal(columns["phase"], phases):
            raise ValueError(f"{band}: returned phases differ from the input phases")
        for key in (
            "apparent_teff_K",
            "apparent_luminosity_erg_s",
            "active_filling_f",
            "cool_filling_f",
            "hot_filling_f",
        ):
            observables[f"{band}/{key}"] = np.asarray(columns[key])
        if case == "photometry":
            for name, values in columns["relative_flux"].items():
                observables[f"{band}/{name}_relative_flux"] = np.asarray(values)
        else:
            observables[f"{band}/rv_km_s"] = np.asarray(columns["rv_km_s"])
            if case == "synmast_feoh":
                observables[f"{band}/ewr_oh_fe"] = np.asarray(columns["ewr_oh_fe"])
    if not observables:
        raise ValueError("The model returned no observables")
    for name, values in observables.items():
        if values.shape != phases.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid model output: {name}")
    return observables


def run_case(args, psutil):
    # Preserve the caller's thread settings and numerical-library defaults.
    import numpy as np
    from numba import config as numba_config
    from threadpoolctl import threadpool_info

    sys.path.insert(0, str(ROOT / "src"))
    from exsplosyn.simulation import (
        main_model,
        prepare_simulation_spectra,
        setup_logging,
    )
    from exsplosyn.simulation_setup import init_setup_cfg

    paths_config = getattr(args, "paths_config", None)
    if paths_config is None:
        demo = "demo_lkca4.cfg" if args.worker_case == "synmast_feoh" else "demo.cfg"
        paths_config = ROOT / "configs" / demo
    cfg = case_config(
        args.config, args.worker_case, args.workers, paths_config=paths_config
    )
    record = {
        "case": args.worker_case,
        "status": "running",
        "workers": args.workers,
        "effective_configuration": public_config(cfg),
        "path_configuration": Path(paths_config).name,
    }
    if args.worker_case == "synmast_feoh" and not feoh_available(cfg, args.paths):
        record.update(
            status="skipped",
            reason="SYNMAST FeOH library not configured or unavailable",
        )
        print("  Skipped: SYNMAST FeOH library is unavailable.", flush=True)
        return record

    logger = setup_logging(quiet=True)
    # Keep warnings in both the terminal and the local report.
    warnings = []
    import logging

    class WarningRecorder(logging.Handler):
        def emit(self, item):
            text = (
                item.getMessage()
                .replace(str(ROOT), "<repository>")
                .replace(str(Path.home()), "~")
            )
            if text not in warnings:
                warnings.append(text)

    recorder = WarningRecorder(level=logging.WARNING)
    logger.addHandler(recorder)
    try:
        with tempfile.TemporaryDirectory(prefix="exsplosyn-benchmark-") as temporary:
            config_path = Path(temporary) / "benchmark.cfg"
            with config_path.open("w", encoding="utf-8") as handle:
                cfg.write(handle)
            print("  Preparing spectral inputs and templates...", flush=True)
            started = perf_counter()
            (
                star,
                bands,
                save_dir,
                nrho,
                grid,
                instrument_res,
                cpu,
                plot_flag,
                save_spectra,
                paths,
                phases,
                n_regions,
                region_box,
                filters,
                velocity_step,
            ) = init_setup_cfg(
                logger, config_path, path_profile=args.paths, mode="fitting"
            )
            prepared = prepare_simulation_spectra(
                bands, star, plot_flag, velocity_step, mode="fitting"
            )
            record["preparation_s"] = perf_counter() - started
            print(
                f"  Preparation complete: {record['preparation_s']:.2f} s",
                flush=True,
            )
            record["phases"] = np.asarray(phases).tolist()
            record["grid"] = grid
            record["spot_count"] = n_regions
            record["model_id"] = star.model_id
            record["bands"] = list(bands)
            record["filters"] = filters
            # Reading the configured limit avoids starting an otherwise unused
            # Numba parallel thread pool just to inspect its size.
            record["numba_threads_configured"] = int(numba_config.NUMBA_NUM_THREADS)
            record["thread_pools"] = [
                {key: value for key, value in info.items() if key != "filepath"}
                for info in threadpool_info()
            ]
            record["spectral_shapes"] = {
                band: {
                    str(temp): {
                        "shape": list(spec.spectrum.shape),
                        "padded_bins": spec.n_padded_bins,
                    }
                    for temp, spec in spectra.items()
                }
                for band, spectra in zip(bands, prepared)
                if spectra is not None
            }
            # The input phase values, not the local filename, define the workload.
            configuration_bytes = json.dumps(
                {
                    "settings": record["effective_configuration"],
                    "phases": record["phases"],
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            record["workload_sha256"] = hashlib.sha256(
                configuration_bytes
            ).hexdigest()

            def evaluate():
                # Reuse prepared spectra; isolate per-call Star state.
                evaluation_star = copy(star)
                evaluation_regions = tuple(
                    None if array is None else array.copy() for array in region_box
                )
                return main_model(
                    logger,
                    bands,
                    nrho,
                    grid,
                    evaluation_star,
                    phases,
                    prepared,
                    filters,
                    n_regions,
                    evaluation_regions,
                    paths,
                    instrument_res,
                    cpu,
                    save_dir,
                    save_spectra,
                    plot_flag,
                    velocity_step,
                    silence_pbar=True,
                    mode="fitting",
                )[1]

            record["warmup_times_s"] = []
            reference = None
            for index in range(args.warmup):
                started = perf_counter()
                result = evaluate()
                elapsed = perf_counter() - started
                reference = checked_observables(
                    result, phases, args.worker_case, np
                )
                del result
                record["warmup_times_s"].append(elapsed)
                print(
                    f"  Warm-up {index + 1}/{args.warmup}: {elapsed:.2f} s",
                    flush=True,
                )
            record["evaluations"] = []
            max_differences = {key: 0.0 for key in reference}
            for index in range(args.repeat):
                started = perf_counter()
                result = evaluate()
                elapsed = perf_counter() - started
                observed = checked_observables(result, phases, args.worker_case, np)
                for name, values in observed.items():
                    max_differences[name] = max(
                        max_differences[name],
                        float(np.max(np.abs(values - reference[name]))),
                    )
                del result, observed
                record["evaluations"].append(
                    {"evaluation": index + 1, "wall_time_s": elapsed}
                )
                print(
                    f"  Evaluation {index + 1}/{args.repeat}: {elapsed:.2f} s",
                    flush=True,
                )
            # Memory sampling is a separate identical evaluation, so its
            # monitoring overhead does not affect the timed evaluations.
            print("  Measuring memory in a separate evaluation...", flush=True)
            with sampled_memory(psutil) as memory:
                result = evaluate()
            checked_observables(result, phases, args.worker_case, np)
            del result
            record["memory_probe"] = memory
            times = [item["wall_time_s"] for item in record["evaluations"]]
            record.update(
                status="passed",
                average_s=statistics.mean(times),
                median_s=statistics.median(times),
                stdev_s=statistics.stdev(times) if len(times) > 1 else 0.0,
                min_s=min(times),
                max_s=max(times),
                sampled_peak_rss_sum_mib=memory["peak_rss_sum_mib"],
                repeat_max_abs_differences=max_differences,
                reference_observables={
                    key: values.tolist() for key, values in reference.items()
                },
            )
            print(
                f"  Average: {record['average_s']:.2f} s | Median: {record['median_s']:.2f} s",
                flush=True,
            )
    except Exception as error:
        import traceback

        traceback.print_exc()
        record.update(
            status="failed",
            reason=str(error)
            .replace(str(ROOT), "<repository>")
            .replace(str(Path.home()), "~"),
        )
    finally:
        logger.removeHandler(recorder)
        record["warnings"] = warnings
    return record


def write_reports(directory, report):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "benchmark_details.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    hardware = report["hardware"]
    fields = [
        "case",
        "status",
        "machine_label",
        "utc_started",
        "protocol",
        "cpu_model",
        "computer_model",
        "os",
        "os_version",
        "architecture",
        "physical_cores",
        "logical_cores",
        "ram_gib",
        "workers",
        "thread_policy",
        "grid",
        "spot_count",
        "phase_count",
        "repeat",
        "preparation_s",
        "average_s",
        "speedup_vs_1_process",
        "median_s",
        "stdev_s",
        "min_s",
        "max_s",
        "sampled_peak_rss_sum_mib",
        "rss_children_accessible",
        "exsplosyn",
        "git_commit",
        "git_dirty",
        "workload_sha256",
        "reason",
    ]
    with (directory / "benchmark_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in report["cases"]:
            row = {
                key: case.get(
                    key, hardware.get(key, report["software"].get(key, report.get(key)))
                )
                for key in fields
            }
            row["rss_children_accessible"] = case.get("memory_probe", {}).get(
                "children_accessible"
            )
            row["phase_count"] = len(case.get("phases", []))
            row["repeat"] = len(case.get("evaluations", []))
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.cfg")
    parser.add_argument("--paths", help="Named path profile in the path configuration.")
    parser.add_argument(
        "--paths-config", type=Path,
        help="Override demo path settings with [paths] from this cfg.",
    )
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Test only this process count; default: compare 1 and 4."
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Untimed warm-up evaluations; default: 2."
    )
    parser.add_argument(
        "--repeat", type=int, default=5, help="Timed evaluations; default: 5."
    )
    parser.add_argument(
        "--machine-label",
        default="",
        help="Optional descriptive computer name, not a hostname.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results")
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.workers or 1, args.warmup, args.repeat) < 1 or args.workers == 0:
        parser.error("workers, warmup, and repeat must be positive")
    args.config = args.config.resolve()
    if args.paths_config is not None:
        args.paths_config = args.paths_config.resolve()
    args.output_dir = args.output_dir.resolve()
    try:
        import psutil
        import threadpoolctl
    except ImportError:
        parser.error(
            "Install benchmark dependencies: conda install -c conda-forge psutil threadpoolctl"
        )
    os.chdir(ROOT)
    if args.worker_case:
        args.workers = args.workers or 1
        try:
            record = run_case(args, psutil)
        except Exception as error:
            import traceback

            traceback.print_exc()
            record = {
                "case": args.worker_case,
                "status": "failed",
                "workers": args.workers,
                "reason": str(error)
                .replace(str(ROOT), "<repository>")
                .replace(str(Path.home()), "~"),
            }
        args.worker_result.write_text(json.dumps(record, allow_nan=False))
        return int(record["status"] == "failed")

    hardware = system_info(psutil)
    if (
        hardware["available_logical_cpus"] is not None
        and args.workers is not None
        and args.workers > hardware["available_logical_cpus"]
    ):
        parser.error("workers exceeds the available logical CPU count")
    worker_counts = [args.workers] if args.workers is not None else [1, 4]
    # Fail on an invalid config before starting any benchmark subprocesses.
    read_config(args.config)
    report = {
        "protocol": "fixed_temperature_forward_model_v3",
        "worker_counts": worker_counts,
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "machine_label": args.machine_label,
        "hardware": hardware,
        "software": software_info(),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "configuration": args.config.name,
        "thread_policy": "inherit environment and library defaults",
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "timing_scope": "main_model in fitting mode; prepared spectra reused; no result exports; input state copied per call",
        "memory_metric": "separate evaluation: sampled sum of process-tree RSS every 0.1 s; includes shared pages more than once; may miss brief peaks",
        "cases": [],
    }
    print("EXSPLOSYN performance benchmark\n", flush=True)
    print(f"Computer:       {hardware['computer_model'] or 'unknown'}", flush=True)
    print(f"OS:             {hardware['os']} {hardware['os_version']}", flush=True)
    print(f"CPU:            {hardware['cpu_model'] or 'unknown'}", flush=True)
    print(
        f"CPU cores:      {hardware['physical_cores']} physical / {hardware['logical_cores']} logical",
        flush=True,
    )
    print(f"Process counts: {', '.join(map(str, worker_counts))}", flush=True)
    print("Threads:        existing environment / library defaults", flush=True)
    print("Preparation/downloads are excluded from the evaluation timings.", flush=True)
    # Fetch once, outside the timed evaluations; all process counts use the same snapshot.
    catalog, report["leaderboard_source"] = load_catalog()
    with tempfile.TemporaryDirectory(prefix="exsplosyn-benchmark-results-") as tmp:
        for workers in worker_counts:
            for case in dict.fromkeys(args.cases):
                available = hardware["available_logical_cpus"]
                if available is not None and workers > available:
                    print(f"Skipping {case} with {workers} processes: only {available} CPUs available.", flush=True)
                    report["cases"].append({
                        "case": case, "workers": workers, "status": "skipped",
                        "reason": "Requested process count exceeds available CPUs",
                    })
                    continue
                print(f"\n>>> {CASE_LABELS[case]}: {workers} process(es) <<<", flush=True)
                result_path = Path(tmp) / f"{case}_{workers}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(args.config),
                    "--workers",
                    str(workers),
                    "--warmup",
                    str(args.warmup),
                    "--repeat",
                    str(args.repeat),
                    "--worker-case",
                    case,
                    "--worker-result",
                    str(result_path),
                ]
                if args.paths:
                    command.extend(["--paths", args.paths])
                if args.paths_config is not None:
                    command.extend(["--paths-config", str(args.paths_config)])
                process = subprocess.run(command, check=False)
                if result_path.is_file():
                    case_result = json.loads(result_path.read_text())
                else:
                    case_result = {
                        "case": case,
                        "status": "failed",
                        "workers": workers,
                        "reason": f"Case process exited with code {process.returncode} without a report",
                    }
                report["cases"].append(case_result)
    # Compare matching cases only; a missing or failed run has no speedup.
    for case in dict.fromkeys(args.cases):
        passed = {
            row["workers"]: row for row in report["cases"]
            if row["case"] == case and row["status"] == "passed"
        }
        baseline = passed.get(1)
        if baseline is None:
            continue
        for workers, row in passed.items():
            if row["average_s"] > 0:
                row["speedup_vs_1_process"] = baseline["average_s"] / row["average_s"]
                if workers != 1:
                    print(
                        f"{CASE_LABELS[case]}: {workers} processes speedup = "
                        f"{row['speedup_vs_1_process']:.2f}x", flush=True,
                    )
    report["leaderboard"] = show_rankings(report, catalog_entries=catalog)
    report["utc_finished"] = datetime.now(timezone.utc).isoformat()
    directory = args.output_dir / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    write_reports(directory, report)
    print(f"\nResults saved to: {directory}", flush=True)
    return int(any(case["status"] == "failed" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())

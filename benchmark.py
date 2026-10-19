"""Time complete EXSPLOSYN CLI runs with one and four processes.

Use configs/performance.cfg: 200 phases, two warm-ups, five timed runs.
Each run reads the config and prepares all spectra through the normal CLI.
Times therefore include spectral/template preparation and surface integration,
as well as the enabled observables. Warm-up runs are excluded from statistics.
"""

import argparse
import configparser
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

# Allow execution from a source checkout without installing the package.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from exsplosyn.leaderboard import load_catalog, show_rankings

# Each process count gets its own warm-ups and repeated measurements.
# Change PROTOCOL if the timing method changes; config/phase changes are hashed.
WARMUP = 2
REPEAT = 5
PHASE_COUNT = 200
PROTOCOL = "complete_cli_simulation_v5"


def read_config(path):
    # Match the config syntax used by the simulation, including inline comments.
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    with path.open(encoding="utf-8") as handle:
        cfg.read_file(handle)
    return cfg


def hardware_info(psutil):
    """Record hardware without hostnames, serial numbers or local file paths."""
    cpu, model = platform.processor() or platform.machine(), None
    # platform.processor() can return only the architecture on macOS; sysctl
    # supplies the CPU name and a hardware model identifier (not a serial number).
    if sys.platform == "darwin":
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
    elif sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    # CPU affinity may restrict the cores available to this process on clusters.
    # macOS does not expose this through psutil, so fall back to the core count.
    try:
        available = len(psutil.Process().cpu_affinity())
    except (AttributeError, NotImplementedError, OSError, psutil.Error):
        available = psutil.cpu_count()
    # Keep the user-facing OS version separate from the kernel version.
    # Unknown macOS names fall back to the numeric version in the printed summary.
    system = platform.system()
    os_version = platform.mac_ver()[0] or platform.release()
    # Apple version names: https://support.apple.com/en-us/109033
    macos_names = {
        "11": "Big Sur",
        "12": "Monterey",
        "13": "Ventura",
        "14": "Sonoma",
        "15": "Sequoia",
        "26": "Tahoe",
        "27": "Golden Gate",
    }
    return dict(
        cpu_model=cpu,
        computer_model=model,
        os="macOS" if sys.platform == "darwin" else system,
        os_version=os_version,
        os_codename=(
            macos_names.get(os_version.split(".")[0])
            if sys.platform == "darwin"
            else None
        ),
        kernel_version=platform.release(),
        python_version=platform.python_version(),
        architecture=platform.machine(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(),
        available_logical_cpus=available,
        ram_gib=psutil.virtual_memory().total / 1024**3,
    )


def run_test(config_path, workers):
    """Use the unmodified CLI for every warm-up and timed run."""
    from exsplosyn.cli import main as run_simulation

    # Reuse the normal CLI entry point, including config loading and preparation.
    # Quiet suppresses routine output; fitting mode returns results in RAM.
    # Imports above occur before timing; lazy imports inside the CLI can occur
    # during warm-up. Input downloads, JIT compilation and caches may also warm up.
    times = []
    for index in range(WARMUP + REPEAT):
        # Monotonic wall time includes both computation and waiting for workers/I/O.
        started = perf_counter()
        run_simulation(["--config", str(config_path), "--quiet", "--mode", "fitting"])
        elapsed = perf_counter() - started
        # Show warm-up durations for feedback, but only retain the timed repeats.
        if index < WARMUP:
            print(f"  Warm-up {index + 1}/{WARMUP}: {elapsed:.2f} s", flush=True)
        else:
            times.append(elapsed)
            print(f"  Run {len(times)}/{REPEAT}: {elapsed:.2f} s", flush=True)
    return dict(
        workers=workers,
        status="passed",
        average_s=statistics.mean(times),
        median_s=statistics.median(times),
        stdev_s=statistics.stdev(times),
        evaluations=[{"wall_time_s": value} for value in times],
    )


def save_reports(directory, report):
    """Save full run details in JSON and one summary CSV row per process count."""
    directory.mkdir(parents=True)
    (directory / "benchmark_details.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    # Nested metadata and individual timings stay in JSON; CSV is for comparison.
    fields = (
        "case",
        "workers",
        "status",
        "average_s",
        "median_s",
        "stdev_s",
        "speedup_vs_1_process",
        "reason",
    )
    with (directory / "benchmark_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["cases"])


def main(argv=None):
    """Validate the workload, time both process counts, then save and compare results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/performance.cfg")
    args = parser.parse_args(argv)
    cfg = read_config(args.config.resolve())
    # --- Validate the shared workload ---
    # One fixed phase list makes the workload independent of regular phase sampling.
    phase_file = ROOT / "configs/phases/phases_benchmark.txt"
    phases = [float(value) for value in phase_file.read_text().splitlines()]
    if phases != [i / PHASE_COUNT for i in range(PHASE_COUNT)]:
        parser.error(
            "The benchmark phase list must contain 200 phases from 0.000 to 0.995 in steps of 0.005"
        )
    cfg.set("phase_sampling", "phase_file", str(phase_file))
    cfg.set("phase_sampling", "phase_step", "")
    # Keep plotting and spectral exports out of the workload, even for a custom cfg.
    if any(
        cfg.getboolean("output", key)
        for key in ("make_plots", "make_gif_animation", "save_spectra")
    ):
        parser.error("Disable plots, GIFs and spectrum exports in the performance cfg")

    # Delay optional benchmark imports until after argparse so --help stays light.
    import psutil
    import threadpoolctl
    from exsplosyn.cli import _project_version

    hardware = hardware_info(psutil)
    enabled = [
        name.strip()
        for name in cfg.get("spectroscopic", "enabled").split(",")
        if name.strip()
    ]
    # --- Check optional SYNMAST inputs ---
    # If the library is absent, still time the remaining observables. Such runs
    # have a different case/hash and do not compete with complete FeOH runs.
    # This checks for library files; the normal reader validates required corners.
    has_feoh = "FeOH" in enabled
    if has_feoh:
        library = Path(
            cfg.get("paths", cfg.get("band:FeOH", "library_path"))
        ).expanduser()
        if not library.is_absolute():
            library = ROOT / library
        if not any(library.glob("*.prf.gz")):
            enabled.remove("FeOH")
            cfg.set("spectroscopic", "enabled", ", ".join(enabled))
            has_feoh = False
            print(
                "FeOH library unavailable; running the remaining observables.",
                flush=True,
            )
    case_name = "synmast_feoh" if has_feoh else "simulation_without_feoh"
    # --- Collect metadata and print a compact summary ---
    # Record library versions for interpretation without changing their thread limits.
    report = dict(
        protocol=PROTOCOL,
        utc_started=datetime.now(timezone.utc).isoformat(),
        hardware=hardware,
        software={
            "exsplosyn": _project_version(),
            "packages": {
                name: version(name)
                for name in ("numpy", "scipy", "numba", "matplotlib", "ldtk")
            },
        },
        warmup=WARMUP,
        repeat=REPEAT,
        configuration=args.config.name,
        thread_policy="inherit environment and library defaults",
        timing_scope="complete quiet fitting-mode CLI call, including config, all spectral/template preparation and model; imports/downloads may occur during warm-up",
        cases=[],
    )
    os_label = " ".join(
        str(value)
        for value in (
            hardware["os"],
            hardware.get("os_codename"),
            hardware["os_version"],
        )
        if value
    )
    regions = len(cfg.get("active_regions", "sizes").split(","))
    grid = cfg.getint("main", "grid")
    summary = {
        "CPU": f"{hardware['cpu_model']} | {hardware['physical_cores']} physical / {hardware['logical_cores']} logical cores",
        "System": f"{os_label} ({hardware.get('architecture', platform.machine())})",
        "Memory": f"{hardware['ram_gib']:.0f} GiB | Python {hardware.get('python_version', platform.python_version())}",
        "Workload": f"{regions} regions | grid {grid} x {grid} | {len(phases)} phases",
        "Signals": " + ".join(enabled + [cfg.get("photometry", "filters")]),
        "Runs": f"CPU 1 vs 4 | {WARMUP} warm-ups + {REPEAT} timed runs each",
    }
    print(
        f"\nEXSPLOSYN {report['software']['exsplosyn']} | Performance benchmark",
        flush=True,
    )
    print("-" * 64, flush=True)
    for label, value in summary.items():
        print(f"{label:>8}: {value}", flush=True)
    print("-" * 64, flush=True)
    # Fetch reference scores before timing; the loader falls back to bundled scores.
    catalog, report["leaderboard_source"] = load_catalog()
    # --- Run the same model with one and four worker processes ---
    # Paths are resolved from the checkout, just as when running main.py there.
    previous_directory = Path.cwd()
    try:
        os.chdir(ROOT)
        # Write a temporary cfg so the CLI can load it normally. The supplied cfg
        # stays unchanged; only [main] cpu differs between the two timed workloads.
        with tempfile.TemporaryDirectory(prefix="exsplosyn-performance-") as temporary:
            config_path = Path(temporary) / "performance.cfg"
            for workers in (1, 4):
                print(f"\n>>> Complete simulation: cpu = {workers} <<<", flush=True)
                cfg.set("main", "cpu", str(workers))
                with config_path.open("w") as handle:
                    cfg.write(handle)
                # Hash parsed settings and actual phase values, excluding path profiles
                # and the absolute phase filename. Include cpu to distinguish the runs.
                # Sorted JSON keys make section/option ordering irrelevant to the hash.
                # The leaderboard also checks protocol/version before comparing scores.
                settings = {
                    name: dict(cfg[name])
                    for name in cfg
                    if name != "DEFAULT" and not name.startswith("paths")
                }
                settings["phase_sampling"]["phase_file"] = phase_file.name
                workload = hashlib.sha256(
                    json.dumps(
                        {"settings": settings, "phases": phases}, sort_keys=True
                    ).encode()
                ).hexdigest()
                record = dict(
                    case=case_name,
                    workers=workers,
                    workload_sha256=workload,
                    effective_configuration=settings,
                    phases=phases,
                )
                # Skip the four-process run if fewer than four CPUs are available.
                if (
                    hardware["available_logical_cpus"] is not None
                    and workers > hardware["available_logical_cpus"]
                ):
                    record.update(
                        status="skipped",
                        reason="Requested process count exceeds available CPUs",
                    )
                else:
                    try:
                        record.update(run_test(config_path, workers))
                        print(
                            f"  Average: {record['average_s']:.2f} s | Median: {record['median_s']:.2f} s",
                            flush=True,
                        )
                    except Exception as error:
                        # Preserve failures in the report and continue with the other
                        # process count. Remove common local path prefixes from errors.
                        record.update(
                            status="failed",
                            reason=str(error)
                            .replace(str(ROOT), "<repository>")
                            .replace(str(Path.home()), "~"),
                        )
                        print(f"  Failed: {record['reason']}", flush=True)
                report["cases"].append(record)
    finally:
        # Restore the caller's working directory even if a run or report setup fails.
        os.chdir(previous_directory)
    # --- Summarize successful runs and save the results ---
    # Speedup = mean time with 1 process / mean time with 4 processes.
    # Values below 1 mean the four-process workload took longer.
    passed = {
        row["workers"]: row for row in report["cases"] if row["status"] == "passed"
    }
    if 1 in passed and 4 in passed:
        speedup = passed[1]["average_s"] / passed[4]["average_s"]
        passed[4]["speedup_vs_1_process"] = speedup
        print(f"\n4-process speedup: {speedup:.2f}x", flush=True)
    # Retain an explicit skipped FeOH entry so the ranking helper can explain
    # how to enable the complete benchmark when the optional library is missing.
    if not has_feoh:
        report["cases"].append(
            dict(
                case="synmast_feoh", status="skipped", reason="FeOH library unavailable"
            )
        )
    # Inspect loaded numerical thread pools after the model has run. This records
    # their defaults without setting thread limits; omit machine-specific paths.
    report["thread_pools"] = [
        {key: value for key, value in item.items() if key != "filepath"}
        for item in threadpoolctl.threadpool_info()
    ]
    report["leaderboard"] = show_rankings(report, catalog_entries=catalog)
    # A UTC timestamp gives each invocation its own folder; sharing the whole
    # folder retains both the short CSV summary and the metadata needed to compare.
    directory = (
        ROOT
        / "benchmark_results"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    save_reports(directory, report)
    print(f"\nResults saved to: {directory}", flush=True)
    print(
        f'Please share the entire "{directory.name}" folder with the author.',
        flush=True,
    )
    # A failed run yields a nonzero exit status; skipped runs alone do not.
    return int(any(row["status"] == "failed" for row in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())

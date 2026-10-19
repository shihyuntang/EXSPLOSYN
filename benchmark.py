"""Time complete EXSPLOSYN CLI runs with one and four processes.

Use configs/performance.cfg: 100 phases, two warm-ups, five timed runs.
Each run reads the config and prepares all spectra through the normal CLI.
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from exsplosyn.leaderboard import load_catalog, show_rankings

WARMUP = 2
REPEAT = 5
PROTOCOL = "complete_cli_simulation_v5"


def read_config(path):
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    with path.open(encoding="utf-8") as handle:
        cfg.read_file(handle)
    return cfg


def hardware_info(psutil):
    """Record hardware without hostnames, serial numbers or local file paths."""
    cpu, model = platform.processor() or platform.machine(), None
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
    elif sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    try:
        available = len(psutil.Process().cpu_affinity())
    except (AttributeError, NotImplementedError, OSError, psutil.Error):
        available = psutil.cpu_count()
    return dict(
        cpu_model=cpu, computer_model=model, os=platform.system(),
        os_version=platform.mac_ver()[0] or platform.release(),
        architecture=platform.machine(), physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(), available_logical_cpus=available,
        ram_gib=psutil.virtual_memory().total / 1024**3,
    )


def run_test(config_path, workers):
    """Use the unmodified CLI for every warm-up and timed run."""
    from exsplosyn.cli import main as run_simulation

    times = []
    for index in range(WARMUP + REPEAT):
        started = perf_counter()
        run_simulation(["--config", str(config_path), "--quiet", "--mode", "fitting"])
        elapsed = perf_counter() - started
        if index < WARMUP:
            print(f"  Warm-up {index + 1}/{WARMUP}: {elapsed:.2f} s", flush=True)
        else:
            times.append(elapsed)
            print(f"  Run {len(times)}/{REPEAT}: {elapsed:.2f} s", flush=True)
    return dict(
        workers=workers, status="passed", average_s=statistics.mean(times),
        median_s=statistics.median(times), stdev_s=statistics.stdev(times),
        evaluations=[{"wall_time_s": value} for value in times],
    )


def save_reports(directory, report):
    directory.mkdir(parents=True)
    (directory / "benchmark_details.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fields = ("case", "workers", "status", "average_s", "median_s", "stdev_s", "speedup_vs_1_process", "reason")
    with (directory / "benchmark_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["cases"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/performance.cfg")
    args = parser.parse_args(argv)
    cfg = read_config(args.config.resolve())
    # One fixed phase list makes the workload independent of regular phase sampling.
    phase_file = ROOT / "configs/phases/phases_benchmark.txt"
    phases = [float(value) for value in phase_file.read_text().splitlines()]
    if phases != [i / 100 for i in range(100)]:
        parser.error("The benchmark phase list must contain 0.00 through 0.99 in steps of 0.01")
    cfg.set("phase_sampling", "phase_file", str(phase_file))
    cfg.set("phase_sampling", "phase_step", "")
    if any(cfg.getboolean("output", key) for key in ("make_plots", "make_gif_animation", "save_spectra")):
        parser.error("Disable plots, GIFs and spectrum exports in the performance cfg")

    import psutil
    import threadpoolctl
    from exsplosyn.cli import _project_version

    hardware = hardware_info(psutil)
    enabled = [name.strip() for name in cfg.get("spectroscopic", "enabled").split(",") if name.strip()]
    has_feoh = "FeOH" in enabled
    if has_feoh:
        library = Path(cfg.get("paths", cfg.get("band:FeOH", "library_path"))).expanduser()
        if not library.is_absolute():
            library = ROOT / library
        if not any(library.glob("*.prf.gz")):
            enabled.remove("FeOH")
            cfg.set("spectroscopic", "enabled", ", ".join(enabled))
            has_feoh = False
            print("FeOH library unavailable; running the remaining observables.", flush=True)
    case_name = "synmast_feoh" if has_feoh else "simulation_without_feoh"
    report = dict(
        protocol=PROTOCOL, utc_started=datetime.now(timezone.utc).isoformat(),
        hardware=hardware, software={"exsplosyn": _project_version(), "packages": {
            name: version(name) for name in ("numpy", "scipy", "numba", "matplotlib", "ldtk")
        }}, warmup=WARMUP, repeat=REPEAT, configuration=args.config.name,
        thread_policy="inherit environment and library defaults",
        timing_scope="complete quiet fitting-mode CLI call, including config, all spectral/template preparation and model; imports/downloads may occur during warm-up",
        cases=[],
    )
    print(f"CPU: {hardware['cpu_model']} | Cores: {hardware['physical_cores']} physical / {hardware['logical_cores']} logical", flush=True)
    print(f"OS: {hardware['os']} {hardware['os_version']} | Phases: 100", flush=True)
    catalog, report["leaderboard_source"] = load_catalog()
    # Paths are resolved from the checkout, just as when running main.py there.
    previous_directory = Path.cwd()
    try:
        os.chdir(ROOT)
        with tempfile.TemporaryDirectory(prefix="exsplosyn-performance-") as temporary:
            config_path = Path(temporary) / "performance.cfg"
            for workers in (1, 4):
                print(f"\n>>> Complete simulation: cpu = {workers} <<<", flush=True)
                cfg.set("main", "cpu", str(workers))
                with config_path.open("w") as handle:
                    cfg.write(handle)
                # Hash effective settings, excluding private paths and the phase filename.
                settings = {name: dict(cfg[name]) for name in cfg if name != "DEFAULT" and not name.startswith("paths")}
                settings["phase_sampling"]["phase_file"] = phase_file.name
                workload = hashlib.sha256(json.dumps({"settings": settings, "phases": phases}, sort_keys=True).encode()).hexdigest()
                record = dict(case=case_name, workers=workers, workload_sha256=workload,
                              effective_configuration=settings, phases=phases)
                if hardware["available_logical_cpus"] is not None and workers > hardware["available_logical_cpus"]:
                    record.update(status="skipped", reason="Requested process count exceeds available CPUs")
                else:
                    try:
                        record.update(run_test(config_path, workers))
                        print(f"  Average: {record['average_s']:.2f} s | Median: {record['median_s']:.2f} s", flush=True)
                    except Exception as error:
                        record.update(status="failed", reason=str(error).replace(str(ROOT), "<repository>").replace(str(Path.home()), "~"))
                        print(f"  Failed: {record['reason']}", flush=True)
                report["cases"].append(record)
    finally:
        os.chdir(previous_directory)
    passed = {row["workers"]: row for row in report["cases"] if row["status"] == "passed"}
    if 1 in passed and 4 in passed:
        speedup = passed[1]["average_s"] / passed[4]["average_s"]
        passed[4]["speedup_vs_1_process"] = speedup
        print(f"\n4-process speedup: {speedup:.2f}x", flush=True)
    if not has_feoh:
        report["cases"].append(dict(case="synmast_feoh", status="skipped", reason="FeOH library unavailable"))
    report["thread_pools"] = [{key: value for key, value in item.items() if key != "filepath"} for item in threadpoolctl.threadpool_info()]
    report["leaderboard"] = show_rankings(report, catalog_entries=catalog)
    directory = ROOT / "benchmark_results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    save_reports(directory, report)
    print(f"\nResults saved to: {directory}", flush=True)
    return int(any(row["status"] == "failed" for row in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare voluntarily shared FeOH timings without uploading machine data.

Only matching protocols, workloads, EXSPLOSYN versions, process counts and
sampling settings share a leaderboard. Reference snapshots contain only the
descriptive machine label, CPU/OS and timing fields needed for comparison.
"""

from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
import math
from pathlib import Path
from urllib.request import urlopen

CATALOG = Path(__file__).with_name("leaderboard.json")
# Stable public file link; upload new Box versions to preserve its token.
LEADERBOARD_URL = "https://utexas.box.com/shared/static/3hofr4h9v9ustavitee32bhnmuqdzzx5.json"
DOWNLOAD_TIMEOUT_S = 5
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
CASE = "synmast_feoh"
COMPATIBILITY = (
    "protocol", "exsplosyn", "workload_sha256", "workers", "warmup",
    "repeat", "thread_policy",
)
FEOH_URL = "https://utexas.box.com/s/34wcqyq0k0glfl4nc95kfidd1j5nr9kt"


def score_entries(report, label):
    """Select successful 1/4-process FeOH results and publish only needed fields."""
    hardware = report.get("hardware", {})
    entries = []
    for case in report.get("cases", []):
        if case.get("case") != CASE or case.get("status") != "passed":
            continue
        workers = case.get("workers")
        times = [row["wall_time_s"] for row in case.get("evaluations", [])]
        average = case.get("average_s")
        if (
            workers not in (1, 4) or not times
            or not isinstance(average, (int, float))
            or not math.isfinite(average) or average <= 0
            or any(not math.isfinite(value) or value <= 0 for value in times)
            or not math.isclose(average, sum(times) / len(times), rel_tol=1e-9)
        ):
            continue
        entry = {
            "protocol": report.get("protocol"),
            "exsplosyn": report.get("software", {}).get("exsplosyn"),
            "workload_sha256": case.get("workload_sha256"),
            "workers": workers,
            "warmup": report.get("warmup"),
            "repeat": len(times),
            "thread_policy": report.get("thread_policy"),
            "machine_label": label,
            "cpu_model": hardware.get("cpu_model") or "unknown CPU",
            "computer_model": hardware.get("computer_model"),
            "os": hardware.get("os"),
            "average_s": average,
        }
        if all(entry[key] is not None for key in COMPATIBILITY):
            entries.append(entry)
    return entries


def comparison_key(entry):
    return tuple(entry[key] for key in COMPATIBILITY)


def validate_catalog(document):
    """Check downloaded and bundled data before using it in ranks or terminal text."""
    if not isinstance(document, dict) or not isinstance(document.get("entries"), list):
        raise ValueError("Expected a leaderboard JSON object with an entries list")
    # Reject non-standard NaN/Infinity values anywhere in the snapshot.
    json.dumps(document, allow_nan=False)
    entries = document["entries"]
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid leaderboard entry")
        for key in ("protocol", "exsplosyn", "workload_sha256", "thread_policy", "machine_label", "cpu_model"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
                raise ValueError(f"Invalid leaderboard field: {key}")
        for key in ("workers", "warmup", "repeat"):
            if type(entry.get(key)) is not int or entry[key] < 1:
                raise ValueError(f"Invalid leaderboard field: {key}")
        if entry["workers"] not in (1, 4):
            raise ValueError("Leaderboard process count must be 1 or 4")
        average = entry.get("average_s")
        if type(average) not in (int, float) or not math.isfinite(average) or average <= 0:
            raise ValueError("Leaderboard contains an invalid timing")
        identity = (comparison_key(entry), entry["machine_label"])
        if identity in seen:
            raise ValueError("Duplicate machine score for the same benchmark settings")
        seen.add(identity)
    return entries


def read_catalog(path=CATALOG):
    return validate_catalog(json.loads(Path(path).read_text()))


def load_catalog(url=LEADERBOARD_URL, bundled_path=CATALOG):
    """Try the public snapshot once per run; preserve the repo's offline fallback."""
    metadata = {"source": "bundled", "online_status": "not_configured"}
    entries = None
    if url:
        print("Checking for the latest benchmark leaderboard...", flush=True)
        try:
            # Only JSON data is read. The remote file never replaces local source files.
            with urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
                body = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(body) > MAX_DOWNLOAD_BYTES:
                raise ValueError("Leaderboard download exceeds the size limit")
            entries = validate_catalog(json.loads(body))
            metadata.update(
                source="online", online_status="updated",
                retrieved_utc=datetime.now(timezone.utc).isoformat(),
            )
            print("Leaderboard: latest online scores loaded.", flush=True)
        except (OSError, HTTPException, ValueError, KeyError, TypeError) as error:
            metadata.update(online_status="unavailable", download_error=str(error))
            print(
                "Could not refresh the leaderboard; using the bundled offline scores.\n"
                "Check your internet connection and run again for the latest rankings.",
                flush=True,
            )
    else:
        print("Leaderboard: bundled scores (online updates are not configured yet).", flush=True)
    if entries is None:
        try:
            entries = read_catalog(bundled_path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            metadata.update(source="unavailable", bundled_error=str(error))
            entries = []
            print("Leaderboard unavailable; benchmark timings will still be saved.", flush=True)
    metadata["entry_count"] = len(entries)
    metadata["snapshot_sha256"] = hashlib.sha256(
        json.dumps({"entries": entries}, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return entries, metadata


def show_rankings(report, catalog_path=CATALOG, *, catalog_entries=None):
    """Print the two top-five tables and return the comparisons for the local JSON."""
    feoh = [case for case in report["cases"] if case["case"] == CASE]
    if not feoh:
        return []
    print("\n>>> FeOH performance leaderboard <<<", flush=True)
    if any(case.get("status") == "skipped" and "library" in case.get("reason", "").lower() for case in feoh):
        print(
            "Want to see where your computer ranks? Download the FeOH library,\n"
            "set synmast_feoh in configs/demo_lkca4.cfg, then run python benchmark.py again.\n"
            f"FeOH download: {FEOH_URL}", flush=True,
        )
    scores = score_entries(report, report.get("machine_label") or "Your computer")
    if not scores:
        return []
    try:
        catalog = read_catalog(catalog_path) if catalog_entries is None else catalog_entries
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Leaderboard unavailable: {error}. Benchmark timings are still saved.", flush=True)
        return []
    results = []
    for current in scores:
        candidates = [dict(entry) for entry in catalog if comparison_key(entry) == comparison_key(current)]
        # An explicitly supplied label replaces that machine's previous result.
        if report.get("machine_label"):
            candidates = [entry for entry in candidates if entry["machine_label"] != current["machine_label"]]
        print(f"\n{current['workers']} process(es) — mean evaluation time (lower is faster)", flush=True)
        if not candidates:
            print("No matching reference scores yet; your result is ready to share.", flush=True)
            results.append({"workers": current["workers"], "status": "no_reference_scores"})
            continue
        current = dict(current, is_current=True)
        candidates.append(current)
        candidates.sort(key=lambda entry: entry["average_s"])
        rank = 1 + sum(entry["average_s"] < current["average_s"] for entry in candidates)
        total = len(candidates)
        percentile = math.ceil(100 * rank / total)
        top_five = []
        print(" Rank | Mean time | Computer / CPU", flush=True)
        for entry in candidates[:5]:
            place = 1 + sum(row["average_s"] < entry["average_s"] for row in candidates)
            marker = " (you)" if entry.get("is_current") else ""
            print(
                f" {place:>4} | {entry['average_s']:>7.2f} s | "
                f"{entry['machine_label']} — {entry['cpu_model']}{marker}", flush=True,
            )
            top_five.append(dict(entry, rank=place))
        if rank <= 5:
            print(f"Your rank: #{rank} of {total} computers.", flush=True)
        else:
            print(f"Your result: top {percentile}% of {total} computers.", flush=True)
        print("Compared only with matching benchmark settings; this is a small participant sample.", flush=True)
        results.append({
            "workers": current["workers"], "status": "ranked", "rank": rank,
            "computer_count": total, "top_percent": percentile, "top_five": top_five,
        })
    return results

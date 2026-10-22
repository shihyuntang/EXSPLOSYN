"""Configuration resolution for simulation setup."""

import configparser
import os
from pathlib import Path

# Supported temperatures in Kelvin; cooler models are less reliable.
# The upper limit is relaxed to 10000 K; normalization above 7000 K needs checking.
MIN_TEMPERATURE_K = 2500
MAX_TEMPERATURE_K = 10000

# Only a source checkout has a project-level default configuration.
_source_directory = Path(__file__).resolve().parents[1]
PROJECT_ROOT = (
    _source_directory.parent
    if _source_directory.name == "src"
    and (_source_directory.parent / "pyproject.toml").is_file()
    else None
)
DEFAULT_CONFIG_FILE = (
    PROJECT_ROOT / "configs" / "demo.cfg" if PROJECT_ROOT is not None else None
)


def resolve_config_file(config_file) -> Path:
    """Resolve a config filename or explicit path, adding .cfg when no extension is given."""
    requested = Path(config_file).expanduser()
    # Bare filenames select checkout configs; explicit paths retain their meaning.
    if PROJECT_ROOT is not None and str(config_file) == requested.name:
        requested = PROJECT_ROOT / "configs" / requested
    if not requested.suffix:
        requested = requested.with_suffix(".cfg")
    if requested.is_file():
        return requested.resolve()
    raise FileNotFoundError(f"Configuration file not found: {requested}")


def resolve_temperature_table_file(table_file) -> Path:
    """Find a table in this code folder's configs/temperature_tables, or at a given path."""
    requested = Path(table_file).expanduser()
    # A filename alone selects a table from this code folder; paths select another location.
    if str(table_file) == requested.name:
        if PROJECT_ROOT is None:
            raise ValueError(
                "A filename alone works only when running EXSPLOSYN from a code folder. "
                "Give the table location, such as ./my_tables/custom.ini or /path/to/custom.ini."
            )
        requested = PROJECT_ROOT / "configs" / "temperature_tables" / requested
    if requested.is_file():
        return requested.resolve()
    raise FileNotFoundError(f"Temperature table not found: {requested}")


def resolve_output_directory(cfg, output_dir=None, *, path_profile="default") -> Path:
    """Choose output_root/run_name or the command-line destination.

    Relative paths start from the current folder; ~ and environment variables
    are supported. No folders are created here.
    """
    if output_dir is not None:
        destination = str(output_dir).strip()
        if not destination:
            raise ValueError("--output-dir needs a folder path, such as outputs/trial_01.")
        return Path(os.path.expandvars(destination)).expanduser().resolve()

    output_root = cfg.get("output", "output_root", fallback="outputs").strip()
    if not output_root:
        raise ValueError("[output] output_root needs a folder path, such as outputs.")
    run_name = cfg.get("output", "run_name", fallback="").strip()
    if not run_name or run_name in {".", ".."} or Path(run_name).name != run_name:
        raise ValueError("[output] run_name needs a folder name, such as trial_01, rather than a path.")
    if path_profile != "default":
        run_name = f"{run_name}_{path_profile}"
    return (Path(os.path.expandvars(output_root)).expanduser() / run_name).resolve()


def _collect_path_profiles(cfg: configparser.ConfigParser) -> dict[str, tuple[str, str]]:
    """Index configured profile names without embedding machine names in code."""
    profiles = {}
    for section in cfg.sections():
        if not section.startswith("paths:"):
            continue
        name = section.partition(":")[2].strip()
        key = name.casefold()
        if not name or key == "default" or key in profiles:
            raise ValueError(f"Invalid or duplicate path profile: {section!r}.")
        profiles[key] = (name, section)
    return profiles


def resolve_path_profile(
    cfg: configparser.ConfigParser, path_profile: str | None = None
) -> str:
    """Use base paths unless a named path profile is explicitly selected."""
    profiles = _collect_path_profiles(cfg)
    selected = "default" if path_profile is None else path_profile.strip()
    if selected.casefold() == "default":
        return "default"
    if selected.casefold() in profiles:
        return profiles[selected.casefold()][0]
    choices = ", ".join(["default", *(value[0] for value in profiles.values())])
    raise ValueError(f"Unknown path profile {selected!r}. Choose: {choices}.")


def resolve_spectral_paths(
    cfg: configparser.ConfigParser, path_profile: str
) -> dict[str, Path]:
    """Merge base/profile paths and expand user-home and environment variables.

    All nonempty entries are retained, including new library keys. Relative
    paths use the working directory. Readers check paths when they are needed;
    no directories are checked or created here.
    """
    selected = resolve_path_profile(cfg, path_profile)
    values = dict(cfg.items("paths")) if cfg.has_section("paths") else {}
    if selected != "default":
        section = _collect_path_profiles(cfg)[selected.casefold()][1]
        values.update(cfg.items(section))

    return {
        name: Path(os.path.expandvars(value.strip())).expanduser()
        for name, value in values.items()
        if value.strip()
    }

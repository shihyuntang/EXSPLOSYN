"""Build the EXSPLOSYN documentation with Markdown and Furo."""

from pathlib import Path
import tomllib


project = "EXSPLOSYN"
author = "Shih-Yun Tang"
copyright = "2026, EXSPLOSYN contributors"

# Share the package version without importing the simulation code.
with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as source:
    release = tomllib.load(source)["project"]["version"]
version = release

extensions = ["myst_parser"]
myst_enable_extensions = ["dollarmath"]
root_doc = "index"
language = "en"
exclude_patterns = ["_build", ".DS_Store"]

html_theme = "furo"
html_title = f"{project} {release}"

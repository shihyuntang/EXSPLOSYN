# Development installation

The package currently uses testing version 0.9.1. The first official public
release, v1.0.0, will include basic simulation functionality and documentation.
The validated local simulation environment uses Python 3.11. The package
metadata allows Python >=3.11,<3.14, matching the pinned Numba version's
compatibility range. Python 3.12 and 3.13 are not yet validated for EXSPLOSYN.

## Create an environment

Run from the EXSPLOSYN project root:

```sh
conda env create -f environment.yml
conda activate exsplosyn
```

The environment file contains the simulation dependencies. Conda resolves
platform-specific builds and compatible minor/patch releases from conda-forge;
nodefaults prevents mixing in the default Anaconda channels. Python stays on
3.11, and Numba stays at 0.62.1. Numba determines the compatible llvmlite version,
and NumPy is restricted to >=2,<2.4. Other dependencies use version ranges
rather than the exact versions of one workstation.

LDTk and pyphot are not currently available from conda-forge. The environment's
pip section installs them after Conda has installed their scientific dependencies.
Conda's PyTables package is named pytables; its Python import and PyPI package
are named tables.

This is a portable environment specification, not an exact dependency lock or
a guarantee of identical numerical results across dependency versions and
platforms. Editing these files does not modify an existing working environment.

## Bundled spectral utilities

EXSPLOSYN includes rds3v, rtint, rotational broadening, instrumental broadening,
and rebinning under exsplosyn.utilities. These modules are included in the wheel;
no syt_tools installation or personal_script checkout is required.

Numerical implementations were copied from the validated syt_tools 2026.9.28.1
snapshot, with original function bodies and source/citation comments retained.
Standalone demo blocks were omitted. Source information, citations and history
are recorded at the top of each utility module.
Updates to another installed syt_tools package do not affect these bundled tools.

## Install EXSPLOSYN

For development from this checkout:

```sh
python -m pip install -e . --no-deps --no-build-isolation
python -m exsplosyn --help
```

Editable installation lets package imports use the source files being edited.
To install a regular package instead, use
python -m pip install . --no-deps --no-build-isolation in the same prepared
environment. --no-deps preserves the environment's resolved dependencies;
--no-build-isolation uses its installed setuptools and wheel. These commands
assume the environment creation step has completed. The entry point is also
available as exsplosyn.

Check dependency consistency after installation:

```sh
python -m pip check
```

```sh
exsplosyn --config configs/my_simulation.cfg -out my_run
```

A regular installed package requires --config. Configurations, spectra, and
outputs are not included in the wheel. Keep the configuration and its ZDI table
together, and provide the library/tool paths required by the configuration.
Path selection is documented in [path_profiles.md](../dev/docs/path_profiles.md), including
--paths and machine overrides.
For the editable/source-checkout layout, omitting --config retains the existing
project default. The root python main.py launch command continues to work
without installing EXSPLOSYN itself, provided its dependencies are installed.
In this layout, --config my_simulation.cfg selects configs/my_simulation.cfg
under the checkout root. An explicit path is used directly; relative paths
are resolved from the working directory.

Use python main.py --version in a checkout, or exsplosyn --version after
installation. The checkout reads [project].version from pyproject.toml;
an installed package reports its distribution metadata version.

Installation does not change the working directory. Continue running from the
project root when using its existing relative input, cache, and output paths.
See [project_layout.md](../dev/docs/project_layout.md) for the path rules.

## External runtime requirements

- PHOENIX/SYNMAST libraries and the PHX intensity user cache must be prepared
  separately; installing the Python package does not supply spectra.
- PHX flux acquisition still uses curl when a needed file is missing.
- RV templates use a separate RASSINE checkout, launched with the same Python
  interpreter as EXSPLOSYN. The checkout must contain Rassine.py,
  Rassine_functions.py, and Rassine_config.py and accept input pickle, output
  directory, and checkout directory as positional arguments. An existing
  normalized HDF is reused without launching RASSINE.
- The local RASSINE checkout includes eight writable-array copies to support
  pandas 3 rolling results. A fresh 3800 K normalization passed with Python 3.11
  and pandas 3.0.2; an unpatched checkout may fail with read-only-array errors.
  Public installation still requires a documented, obtainable patched RASSINE
  revision. See [refactoring_notes.md](../dev/docs/refactoring_notes.md) for validation limits.
- GIF generation uses ImageMagick's convert command when available.

These execution paths have not been changed by the packaging step. Fitting
packages and scripts are outside the simulation dependency list for v1.0.

## Verification status

The installation specification now uses Conda-managed version ranges with a
fixed Numba version. A fresh environment must pass dependency checks and the
simulation CSV comparisons before being treated as a validated environment.
Conda's dry-run resolver succeeded on macOS arm64 for this specification; this
does not install or validate the pip-only packages or execute a simulation.

For reference, the previous validated Python 3.11 environment used the following
direct dependency versions. This record is not an installation requirements file:

| Package | Validated version | Package | Validated version |
| --- | --- | --- | --- |
| numpy | 2.3.5 | scipy | 1.15.2 |
| pandas | 3.0.2 | matplotlib | 3.10.1 |
| astropy | 7.0.1 | h5py | 3.14.0 |
| tables | 3.10.2 | numba | 0.62.1 |
| ldtk | 1.8.4 | pyphot | 2.1.1 |
| pqdm | 0.2.0 | specutils | 2.2.0 |
| tqdm | 4.67.1 | setuptools | 78.1.0 |
| wheel | 0.45.1 | | |

The wheel was built offline and installed into a temporary target directory,
without changing the existing scientific environment. Imports and console
entry-point help were exercised outside the source checkout. Missing --config
was rejected, and explicit config/output arguments reached simulation setup.
The bundled utilities retain the original numerical function definitions.
Real SYNMAST plain/gzip reads and standalone broadening/integration/rebinning
comparisons match the original implementation exactly. A fresh process blocked
all external utility imports and successfully imported simulation and CLI.
Source path regression tests and Python syntax checks passed.

This used existing installed dependencies. A new environment resolving and
installing all dependencies, a full simulation from that environment, and
external RASSINE validation on the fresh environment remain required before
v1.0 completion. One local fresh normalization has passed with the patched
checkout, but this does not validate a clean installation or exact RV equivalence.

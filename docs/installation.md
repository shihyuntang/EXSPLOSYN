# Installation

EXSPLOSYN was developed and tested with Python 3.11. Python 3.12 and 3.13 are
also supported by the package requirements, but have not yet been validated.
If you encounter issues with either version, please let us know.

We recommend using Conda to manage the environment, as shown below.

## 1. Create the environment

Download or clone this repository, open a terminal in its root directory, and run:

```sh
conda env create -f environment.yml
conda activate exsplosyn
```

The environment uses conda-forge packages and specifies Python 3.11 and
Numba 0.62.1 for compatibility. LDTk and pyphot are installed through the
`pip` section of `environment.yml`. Other dependencies have version ranges
rather than exact pins.

## 2. Install external tools and optional spectral data

### RASSINE for PHOENIX RV templates

EXSPLOSYN uses RASSINE to normalize PHOENIX (2013) spectra for RV templates.
Install the [EXSPLOSYN-compatible RASSINE fork](https://github.com/shihyuntang/Rassine_public/tree/exsplosyn-v0.9.1)
from the EXSPLOSYN repository root:

```sh
mkdir -p external
git clone --branch exsplosyn-v0.9.1 --single-branch \
  https://github.com/shihyuntang/Rassine_public.git external/Rassine_public
```

This tag (`exsplosyn-v0.9.1`) provides a fixed version with the normalization settings and
pandas 3 compatibility fixes used by EXSPLOSYN. For details, see the
[EXSPLOSYN usage notes](https://github.com/shihyuntang/Rassine_public/blob/exsplosyn-v0.9.1/EXSPLOSYN.md).
The fork is based on the [original RASSINE](https://github.com/MichaelCretignier/Rassine_public).

### Optional SYNMAST libraries

The following SYNMAST library is currently available:

- **FeOH:** approximately 1.5444–1.5679 $\mu$m.
  [Download from Box](https://utexas.box.com/s/34wcqyq0k0glfl4nc95kfidd1j5nr9kt).
<!-- - **Ti:** 2.2200–2.2322 $\mu$m. -->
<!-- - **CO:** 2.3052–2.3224 $\mu$m. -->

After downloading the library, set `synmast_feoh` in the `[paths]` section
of your configuration file to the directory containing the spectral files,
as described below.

## 3. Configure spectral data and external tools

Update the `[paths]` section of your configuration file so that EXSPLOSYN can
find RASSINE and the spectral libraries. Two demo configurations are provided
in `configs/`.

For example, `configs/demo.cfg` uses:

```ini
[paths]
phoenix_root           : ~/PHOENIX-ACES-AGSS-COND-2011
phoenix_intensity_root : ~/.ldtk/cache_visir-lowres
rassine_root           : ./external/Rassine_public
```

**These defaults can be kept if you follow the installation steps above and
use the default spectral-data locations.**

- `phoenix_root`: stores the downloaded PHOENIX (2013) high-resolution flux
  spectra. EXSPLOSYN automatically downloads missing spectra to this location.
  To store them elsewhere, such as on an external drive, change this path or
  create a symbolic link to the desired directory.
- `phoenix_intensity_root`: the default location where LDTk stores its
  low-resolution intensity spectra.
- `rassine_root`: the directory containing the EXSPLOSYN-compatible RASSINE fork.

For SYNMAST libraries, set the corresponding paths:

```ini
[paths]
synmast_feoh           : ./data/synmast/FeOH
```

<!-- synmast_co             : ./data/synmast/CO -->
<!-- synmast_ti             : ./data/synmast/Ti -->

SYNMAST bands are disabled by default in `demo.cfg`, so this example
does not require the SYNMAST library. The `demo_lkca4.cfg` example, however,
has FeOH enabled and thus requires the SYNMAST FeOH library.

For more options, see the [configuration manual](configuration.md).

## 4. Check the command-line entry point

We recommend running EXSPLOSYN directly from the source repository; no package
installation is required. With the `exsplosyn` Conda environment activated,
run from the repository root:

```sh
python main.py --help
```

This displays the available command-line options. To run your first simulation,
continue to [Running the demos](running_demos.md).

**The first simulation can take longer:** missing PHOENIX spectra are downloaded
and spectral inputs are prepared. Later runs can reuse these cached inputs.

## Optional: install EXSPLOSYN as a package

If you want to use the `exsplosyn` command or import EXSPLOSYN in your own
scripts from other directories, install the package from the repository root
with the `exsplosyn` environment activated:

```sh
python -m pip install . --no-deps
```

The `--no-deps` option keeps the dependencies installed in the previous step.
You can then use `exsplosyn` or `python -m exsplosyn` from any working directory
while this environment is active.

Configuration files and relative paths still need to be supplied as described in [Running the demos](running_demos.md).

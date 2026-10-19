# Installation

EXSPLOSYN was developed and tested with Python 3.11. Python 3.12 and 3.13 are
also supported by the package requirements, but have not yet been validated.
If you encounter issues with either version, please let us know.

Windows installation and simulations have not yet been validated for this release.
If you are a Windows user, and run into any issue, please let us know!

We recommend using Conda to manage the environment, as shown below.

## 1. Download EXSPLOSYN

Open a terminal in the directory where you want to keep EXSPLOSYN, then run:

```sh
git clone https://github.com/shihyuntang/EXSPLOSYN.git
cd EXSPLOSYN
```

Run the remaining commands from this repository directory unless stated otherwise.

## 2. Create the environment

Create and activate the Conda environment:

```sh
conda env create -f environment.yml
conda activate exsplosyn
```

The environment uses conda-forge packages and specifies Python 3.11,
Numba 0.62.1, and LDTk 1.8.4 to match the tested setup. LDTk and pyphot are
installed through the `pip` section of `environment.yml`. Other dependencies
have version ranges rather than exact pins.

## 3. Install external tools and optional spectral data

### RASSINE for PHOENIX RV templates

EXSPLOSYN uses RASSINE to normalize PHOENIX (2013) spectra for RV templates.
Install the [EXSPLOSYN-compatible RASSINE fork](https://github.com/shihyuntang/Rassine_public/tree/exsplosyn-v0.9.1)
from the EXSPLOSYN repository root:

```sh
mkdir -p external
git -c advice.detachedHead=false clone \
  --depth 1 --filter=blob:none --sparse --branch exsplosyn-v0.9.1 \
  https://github.com/shihyuntang/Rassine_public.git external/Rassine_public
```

This downloads the root-level files needed by EXSPLOSYN without the upstream
example spectra or full Git history. The `-c advice.detachedHead=false` option
suppresses Git's tag-checkout advice for this command only.

The tag (`exsplosyn-v0.9.1`) provides a fixed version with the normalization settings and
pandas 3 compatibility fixes used by EXSPLOSYN. For details, see the
[EXSPLOSYN usage notes](https://github.com/shihyuntang/Rassine_public/blob/exsplosyn-v0.9.1/EXSPLOSYN.md).
The fork is based on the [original RASSINE](https://github.com/MichaelCretignier/Rassine_public).

### Optional SYNMAST libraries

The SYNMAST FeOH library covers approximately 1.5444–1.5679 $\mu$m and is
available as a demo via [Box](https://utexas.box.com/s/34wcqyq0k0glfl4nc95kfidd1j5nr9kt),
with a full version coming soon:

- **Demo — `FeOH_synmast1.9_marcs_spirou_demo` (9.8 MB): recommended for
  running the supplied demos and performance benchmark.** Contains ten spectra
  spanning 3200–4100 K in 100 K steps, at $\log g = 4.0$, [Fe/H] = 0.0,
  and magnetic-field strength 0 G.
- **Full — `FeOH_synmast1.9_marcs_spirou_full` (approximately 10.3 GB): coming soon.**
  This version will provide the full parameter grid described in
  [Model applicability and spectral grids](model_limits.md). Please use the demo
  version until the full library is released.

Store the downloaded folder wherever you keep your synthetic spectral libraries.
Set `synmast_feoh` in `[paths]` to that folder; it does not need to be inside
the EXSPLOSYN repository. Choose the full library when exploring parameters
outside the demo subset.

## 4. Configure spectral data and external tools

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

For the recommended demo library, replace `/path/to` below with its parent
directory. If you downloaded the full library, use its `_full` folder instead:

```ini
[paths]
synmast_feoh           : /path/to/FeOH_synmast1.9_marcs_spirou_demo
```

SYNMAST bands are disabled by default in `demo.cfg`, so this example
does not require the SYNMAST library. The `demo_lkca4.cfg` example, however,
has FeOH enabled and thus requires the SYNMAST FeOH library.

For more options, see the [configuration manual](configuration.md).

## 5. Check the command-line entry point

We recommend running EXSPLOSYN directly from the source repository; no package
installation is required. With the `exsplosyn` Conda environment activated,
run from the repository root:

```sh
python main.py --help
```

This displays the available command-line options.

**Next step:** follow [Running the demos](running_demos.md) to run your first
simulation, including selecting a configuration file and setting relative paths.

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

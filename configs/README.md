# Simulation configurations

The active simulation configuration and its optional temperature lookup table
are stored here. Spectral locations use [paths] defaults and optional
[paths:NAME] overrides. See [path profiles](../dev/docs/path_profiles.md).

Optional phase-list inputs are stored in `phases/`. For example, set
`phase_file : configs/phases/phases_obs.txt` under [phase_sampling] when running from the
project root. `phase_file` is relative to the working directory. See [phase lists](phases/README.md).

Run the current default simulation from the project root:

```sh
python main.py
```

Select another configuration without editing the source code:

```sh
python main.py --config my_simulation.cfg
python main.py --config my_simulation.cfg -o my_run
```

Bare filenames are resolved in this directory. Explicit paths are also accepted;
if no extension is given, `.cfg` is added (for example, `--config dev`).
relative paths are resolved from the working directory. Run `python main.py --help`
to see the available options and default configuration filename.

SYNMAST bands accept arbitrary names and reference a path key:

```ini
[spectroscopic]
enabled : 123

[band:123]
source            : synmast
spectrum_mode     : intensity
library_path      : my_library
center            : 17500
window_half_width : 52

[paths]
my_library : ./data/synmast/my_library

[paths:Swing]
my_library : /data/local/my_library
```

`spectrum_mode` selects `intensity` (default) or `flux` for SYNMAST bands.
Intensity uses the native mu-dependent spectra. Flux uses `rds3v` disk
integration with zero rotation and the existing 2 km/s macroturbulence setting,
then applies quadratic LDTk limb darkening in EXSPLOSYN surface integration.
Both inputs remain unnormalized; continuum normalization is applied to the
simulated disk spectrum later. These are different angular treatments, so
switching modes is not expected to preserve identical RV/EWR results.

`library_path` is a key, not a directory. With `--paths Swing`, the selected
profile overrides `[paths]`; without it, the base value is used. The key need
not match the band name or source. Multiple bands may reference the same key.
Actual directory values expand `~` and environment variables; relative paths
use the working directory.

The current SYNMAST reader requires the existing `.prf.gz` layout; NPZ/CSV
readers are not yet implemented. Diagnostics default to `rv`; `ewr_FeOH`
remains restricted to FeOH. Band identifiers and output filenames now use the
section name directly (FeOH, CO, Ti, or a custom name). `compare_outputs.py`
can match old ewr_FeOH/spec_K_CO/spec_K_Ti benchmark filenames to the new names.
No separate `library` option is used.

The default `demo.cfg` enables `phx_rv`, uses the output prefix `demo`, and
contains only generic base paths. Custom SYNMAST bands remain as inactive
examples. `dev.cfg` preserves the development settings and local path profiles
for subsequent review and regression runs. The original research configuration
also remains available for benchmark runs.

Run the development configuration with the existing Swing paths:

```sh
python main.py --config dev.cfg --paths Swing
```

For `temperature_table`, a bare filename such as `temperature_table_lkca4_demo.ini`
selects `temperature_tables/` beside the selected configuration. Explicit relative
paths (including `./filename.ini`) use the working directory; absolute paths are
used directly. Keep the `.ini` extension.
The LkCa 4 example uses this shared table. Other relative data paths and
output/cache locations use the working directory.

The same filename/path rules apply when calling `init_setup_cfg()`, for example
`demo.cfg` or `configs/demo.cfg` from the project root.
Missing paths raise an error; former locations are not redirected.
Historical and fitting configurations formerly in the project root are stored
in `archive/`. Their contents and schemas have not been migrated to the current
simulation interface. Historical fitting scripts in `../scripts/archive/` still
reference their former locations and will be updated during the separately
scoped fitting upgrade.

The source package is in `src/exsplosyn/`. Moving modules does not change the
location of this configuration directory. The root `main.py` loads the source
package without requiring an installation step. Continue running simulations
from the project root to preserve the existing relative output and data paths.

Select a named machine profile explicitly when needed:

```sh
python main.py --config configs/my_simulation.cfg --paths Swing -o my_run
```

The development config preserves four optional path profiles at the end of the file.
The demo contains only `[paths]`. Without `--paths NAME`, only base directories are used. Its generic base paths must be edited before using a new machine.

## Advanced numerical settings

`[advanced] black_below_temperature` optionally approximates cooler regions
as non-emitting after rounding temperatures to integer Kelvin. Leave it empty
to disable; regions equal to the threshold remain emitting. Gaussian brightness
models ignore this temperature threshold.

The config ends with `[advanced] spec_oversample_factor : 14`. This controls
internal spectral velocity sampling relative to instrumental FWHM. Increase
it (for example to 28) when inspecting small RV signals, at increased runtime
and memory cost; it does not change resolving power or simulate CCD pixels.
If omitted, simulation uses 14. Positive integer values are accepted.
The previous `[spectroscopic]` option is no longer read; move custom values
to `[advanced]` when updating an existing configuration.

`[phase_sampling]` selects regular phase spacing (`phase_step`) or a phase file
(`phase_file`, which takes precedence). `[output]` controls `make_plots`,
`save_spectra`, and `save_folder_name`; the folder name also includes the
optional `--output-suffix` and selected machine profile. Plot and spectrum flags
use 0/1. Temperature-based simulations always calculate bolometric luminosity
and apparent effective temperature at the configured phases, shared across bands.
CSV outputs use `apparent_luminosity_erg_s` and `apparent_teff_K`; projected
visible-disk filling factors use `active_filling_f`, `cool_filling_f`, and
`hot_filling_f` (fractions from 0 to 1). Spectroscopic CSVs use `phase`,
`rv_km_s`, and `ewr_oh_fe`; Gaussian CSVs use `LSD_RV` for their first-moment
RV in km/s and omit the physical temperature/luminosity columns. Gaussian-only models use brightness instead of temperature.

Unused library/tool paths can be left empty. Enabled SYNMAST bands require
a `library_path` key referring to the merged `[paths]` settings; PHOENIX CCF templates require `rassine_root`. PHOENIX
intensity remains required for limb darkening, and the current non-Gaussian
setup requires `phoenix_root`. Missing paths are reported when they are used;
directory contents are checked by spectral readers. Path resolution accepts
new keys without code changes. New spectral formats require reader support; new SYNMAST windows can be added
as band sections without code changes.

## Region temperature inputs

`temperature_mode : absolute` reads `temperatures` in Kelvin.
`offset` and `relative` are aliases that read `temperature_offsets` in Kelvin:
`T_spot = T_phot + offset`, using the rounded photospheric temperature.
Negative offsets give cooler regions; positive offsets give hotter regions.
One temperature or offset can be shared by all regions; otherwise provide one per region.

`t_lookup_table` reads `temperature_levels` and `temperature_table` to look up
signed offsets for the rounded photospheric temperature. It replaces `zdi_level`.
All modes use the same final temperature rounding and non-emitting threshold.

All supplied temperatures are physical values. Non-emitting regions are selected
only by `[advanced] black_below_temperature`, after rounding; equality does not
select a region. An input of `9999.99` now rounds to a physical 10000 K region.

Temperature limits are defined once by `MIN_TEMPERATURE_K` and
`MAX_TEMPERATURE_K` in `src/exsplosyn/configuration.py`: currently 2500–10000 K
for the photosphere, emitting regions, and PHOENIX CCF templates. Region
temperatures must not exceed the upper limit before rounding; the lower limit
is checked after rounding and non-emitting selection. These checks also apply
to temperatures computed from offsets or lookup tables. Non-emitting regions
may be cooler than the lower limit because they require no spectrum.

Models below 2500 K are considered less reliable. The upper limit is currently
relaxed to 10000 K; continuum normalization above 7000 K needs verification.

Photometry CSVs store only `<filter>_relative_flux`, with a mean of 1 over the
supplied phase samples. No magnitude columns are written. In-memory photometry
results use `relative_flux`, keyed by the short filter name.

In-memory band results use the same physical field names and units as CSVs:
`phase`, `rv_km_s`, `ewr_oh_fe`, `apparent_teff_K`,
`apparent_luminosity_erg_s`, `active_filling_f`, `cool_filling_f`, and
`hot_filling_f`, as applicable to the band. Gaussian results use `LSD_RV`.
Photometry retains a nested `relative_flux` dictionary keyed by filter name.

Result CSV filenames end with `_<band>.csv` for spectroscopy, photometry,
and Gaussian bands. Parameter text is retained; diagnostic names are given by
the columns rather than `_RV_EWR` or `_RV` filename suffixes.
`compare_outputs.py` accepts the previous suffixes when comparing saved baselines.

Python callers use `star.photosphere_temperature` (Kelvin) and `star.bands`.
The unused `Star.psi` and `Star.rad_sun` attributes have been removed.
Rotation phases come from `[phase_sampling]`; `[star] psi` is no longer read.

### Prepared spectra

Generated spectral files are written below `prepared_spectra/`:

- `library_interpolated/`: library spectra interpolated to component temperatures.
- `rassine_continuum/`: PHOENIX flux and fitted continuum, plus temporary RASSINE files.
- `simulation_ready/`: unnormalized flux or intensity on the simulation velocity grid.
- `rv_templates/`: normalized PHOENIX spectra prepared for RV CCF measurements.

Filenames identify the source, band (when applicable), spectrum type, temperature,
log g, and metallicity. Prepared band spectra also include wavelength limits in
Angstroms and velocity spacing in km/s. SYNMAST flux filenames include `vrt2`,
the current macroturbulent velocity in km/s; intensity currently does not apply it.
PHOENIX library exports also identify alpha abundance. Existing continuum fits
under `syn_spec/RASSINE_PHX/` are copied into the new layout and reused when available;
old files are retained. Final exported NPZ templates remain in the run's
`outputs/<run>/ccf_template/` directory.

`[output] save_spectra` controls final spectra and CCF templates under `outputs/`.
The CLI `-DeBug` flag independently saves intermediate HDF spectra under
`prepared_spectra/`, including simulation inputs and pre-broadening RV templates.
PHOENIX HDF inputs required by RASSINE and its reusable continuum results are
stored regardless of these options. Turning either option off retains existing files.

### Spectral-grid interpolation

The current `phx` source uses PHOENIX 2013 (ACES). Library-specific Python
readers, exporters, filename helpers, preparation methods, and grid constants
use `phx2013` / `PHX2013` names to distinguish them from a future 2025 reader.
The official `PHOENIX-ACES-AGSS-COND-2011` filenames and existing cache names
remain unchanged. PHOENIX 2025 is not yet available in EXSPLOSYN.

PHOENIX readers interpolate temperature, log g, and metallicity. SYNMAST
readers also interpolate magnetic-field strength in Gauss. They use products
of linear axis weights, clamp requests to grid boundaries, and skip weights
below `1e-5`, following the model-spectrum reference implementation. Values
are accumulated in float64 without renormalizing the retained weights.
The PHOENIX 5000 K exclusion remains; alpha abundance is not a new interpolation
axis. All contributing spectra must share wavelength grids and array shapes.
SYNMAST also requires matching mu grids. PHOENIX mu values vary between
models; its intensity interpolation retains the existing column-position
approximation and the first corner's mu grid, without resampling in mu.
SYNMAST files are indexed by their actual paths; the trailing library name
may vary. Missing corners and duplicate parameter combinations raise errors.
Set `[advanced] photosphere_B` and `active_B` in Gauss (missing or blank values default to zero).
SYNMAST uses the first for the photosphere and the second for all emitting
active regions; PHOENIX ignores these settings. The temperature lookup is
unchanged. A SYNMAST region at the rounded photospheric temperature requires
identical B settings; otherwise configuration raises an error. Prepared
SYNMAST spectrum filenames include B to distinguish intermediate exports.

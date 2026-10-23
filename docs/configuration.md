# Configuration manual

Configuration files use named sections and `key : value` entries. The supplied
files are in `configs/`. For step-by-step examples, see
[Running the demos](running_demos.md).

Use `1` to enable and `0` to disable the output switches described below.

## Usual settings

### `[main]`

Set the spatial resolution, instrumental resolution, and parallel processing.

- `grid`: number of pixels along each side of the projected stellar disk grid,
  $n \times n$. A larger value gives finer spatial sampling. The default and
  recommended value is 200.
- `nrho`: number of points sampled along each circular region's boundary.
  The default and recommended value is 100.
- `instrument_reso`: instrumental resolving power,
  $R = \lambda/\Delta\lambda$, where $\Delta\lambda$ is the FWHM of the
  instrumental broadening profile. Set to `0` to disable instrumental broadening.
- `cpu`: number of subprocesses used for parallel surface integration.

### `[star]`

Define the quiet star and the stellar parameters used to select local spectra.

- `radius`: stellar radius in solar radii, $R_\ast$.
- `prot`: rotation period in days, $P_{\rm rot}$.
- `inclination_angle`: angle between the rotation axis and the line of sight,
  $i_\ast$, in degrees. `0` deg is pole-on; `90` deg is equator-on.
- `Tphot`: quiet photospheric temperature, $T_{\rm phot}$, in Kelvin;
  rounded to integer Kelvin.
- `logg`: base-10 logarithm of surface gravity in cgs.
- `metallicity`: metallicity, [Fe/H], in dex.


### `[active_regions]`

Define circular cool and hot regions on the stellar surface. **Each column
represents one region.** Keep all region lists in the same order, including
Gaussian `brightness` values when used.

- `longitudes`: region-center longitudes, $l$, in degrees.
- `latitudes`: region-center latitudes, $b$, in degrees.
- `sizes`: disk-center filling factors, $f_0$, in percent: the fraction of the
  projected stellar disk covered when the region center is at disk center.
  Individual sizes above 80% are unsupported.
- `temperature_mode`: how region temperatures are supplied:
  `absolute`, `offset` (or `relative`), or `t_lookup_table`.
  - `temperatures`: temperatures in Kelvin; used with `absolute`.
  - `temperature_offsets`: signed offsets in Kelvin from `Tphot`; used with
    `offset` or `relative`. Negative values are cooler, positive values hotter
    compared to the photosphere.
  - `temperature_levels`: case-sensitive table labels, one per region; used
    with `t_lookup_table`.
    - `temperature_table`: table of temperature offsets; used with
      `t_lookup_table`. Each section names a photospheric temperature in Kelvin.
      The section matching `Tphot` must exist; table sections are not interpolated.

For `temperatures` and `temperature_offsets`, one value may be shared by all
regions; otherwise supply one value per region. Final temperatures are rounded
to integer Kelvin. Cooler non-emitting regions can be selected with
`black_below_temperature`.

A filename alone uses `configs/temperature_tables/` inside the code folder you
are running, regardless of the configuration file's location. To use a table
elsewhere, give `./my_tables/custom.ini` from your terminal's current folder,
or a full path such as `/path/to/custom.ini`. An installed package requires
one of these paths. Include `.ini` in the filename.

### `[spectroscopic]`

Select the spectroscopic configurations to run.

- `enabled`: comma-separated names. `phx_rv` uses `[phx_rv]`; other names
  select `[band:NAME]` sections, such as `gaussian` or `FeOH`. Unlisted
  spectroscopic sections are inactive.

Photometry is controlled separately through `[photometry] filters`; **do not**
add `photometry` to `enabled`.

### `[phx_rv]`

Define a regularly spaced set of PHOENIX (2013) RV bands. All wavelength
settings below are in **Angstroms**.

- `wavelength_start`: first band center.
- `wavelength_end`: upper limit for the band-center grid.
- `wavelength_step`: positive spacing between band centers.
- `window_half_width`: positive wavelength extent on either side of each
  center. The total band width is twice this value.
  The default and recommended value is ~100 Angstroms.

Half-widths of 50 Å or less trigger a warning because narrow windows can make
continuum fitting unreliable, particularly for cool-star spectra.

### `[band:gaussian]`

Define an analytic local absorption profile, enabled by listing `gaussian`
in `[spectroscopic] enabled`. RV is measured by the profile's **first moment**,
without CCF.

- `source`: use `gaussian_profile`.
- `center`: line-center wavelength in Angstroms.
- `velocity_half_range`: positive extent on either side of zero velocity,
  in km/s.
- `velocity_step`: positive velocity-bin spacing in km/s.
- `doppler_width`: positive Gaussian width parameter in km/s; not the FWHM.
  The local profile is $1-d\exp[-(v/w)^2]$, where $d$ is `line_depth` and
  $w$ is `doppler_width`.
- `line_depth`: absorption depth relative to the local continuum; between
  `0` and `1`, excluding both endpoints.
- `brightness`: one nonnegative value per active region, in configuration
  order. The quiet surface has brightness `1`; `0` is dark and values above
  `1` are brighter than quiet.

Gaussian regions use `brightness` instead of temperature and ignore
`black_below_temperature`. `Tphot` still sets their limb-darkening coefficients.
**Do NOT** add a `diagnostics` option to a Gaussian band.

### `[band:FeOH]` and custom `[band:NAME]` sections

Define individual bands from a SYNMAST library. Add the section's name to
`[spectroscopic] enabled` to run it. Custom bands may use other names.

- `source`: use `synmast`.
- `spectrum_mode`: `intensity` (default) or `flux`. Intensity uses the library's
  $\mu$-dependent spectra; flux uses disk-integrated local spectra and applies
  quadratic limb darkening during surface integration.
- `library_path`: the **name of a key in `[paths]`**, such as `synmast_feoh`,
  not the directory itself. Multiple bands may share a key.
- `center`: band-center wavelength in Angstroms.
- `window_half_width`: positive wavelength extent on either side of the
  center, in Angstroms. Half-widths of 50 Å or less trigger a warning.
- `diagnostics`: comma-separated diagnostics; defaults to `rv`.
  `rv` measures CCF radial velocity. `ewr_FeOH` measures the OH/Fe
  equivalent-width ratio and is supported only in `[band:FeOH]`.

Libraries must use the supported SYNMAST `.prf.gz` format. The flux and
intensity modes use different angular treatments and need not give identical
results.

<!-- Flux currently includes 2 km/s macroturbulent broadening; intensity -->
<!-- has no corresponding broadening. -->

### `[photometry]`

Select the filters used to calculate relative light curves.

- `filters`: comma-separated identifiers from the
  [SVO Filter Profile Service](https://svo2.cab.inta-csic.es/theory/fps/).
  Leave empty or use `None` to disable photometry. Filter coverage must remain
  within 500–50,000 Å (0.05–5.0 μm).

Relative flux is normalized to a mean of `1` over the supplied phase samples.

### `[rv_ccf]`

Set the CCF RV search and reference spectrum. Gaussian first-moment RV does
not use these settings.

- `rv_half_range`: initial CCF search extent on either side of zero RV, in km/s;
  must exceed `1`. An initial search with an edge or missing peak triggers one
  retry at twice this range.
- `template_source`: `phx` (default) or `file`.
- `template_use_photosphere`: for `phx`, `1` uses `Tphot`; `0` uses
  `template_temperature`.
  - `template_temperature`: PHOENIX template temperature in Kelvin, within
    2500–10000 K; used only when `template_use_photosphere : 0`.
- `template_file`: required for `file`. An NPZ containing matching 1D arrays
  `wavelength_angstrom` (vacuum Angstroms) and `flux` (continuum-normalized).
  Relative paths use the configuration file's directory; absolute paths are
  accepted. Include sufficient wavelength margins for RV shifts.
- `template_rotation_vel`: additional rotational broadening for file
  templates, in km/s ($v\sin i$). `0` leaves the supplied rotation unchanged.
- `template_resolution`: additional instrumental broadening for file
  templates, expressed as resolving power. `0` leaves the supplied profile
  unchanged.

### `[phase_sampling]`

Choose the rotation phases to simulate, in cycles.

- `phase_step`: positive spacing for regular sampling, spanning approximately
  −0.75 to +0.75 cycles. Used when `phase_file` is empty or `None`.
- `phase_file`: text file whose first column contains the phases. Overrides
  `phase_step`. Relative paths use the working directory; absolute paths are
  accepted.

### `[output]`

Control final outputs in **simulation mode**.

- `make_plots`: `1` saves summary PDFs and continuum-fitting plots; `0` skips
  them. Continuum-fitting plots are saved in a `continuum_fitting/` subfolder
  inside the chosen results folder.
- `make_gif_animation`: `1` saves phase animations; `0` skips them. Independent
  of `make_plots`.
- `save_spectra`: `1` saves final NPZ spectra and RV CCF templates under
  `simulated_spotted_spectra/` and `rv_ccf_templates/` in the run folder;
  `0` skips these exports.
- `output_root`: folder that holds your runs; defaults to `outputs`.
  Relative paths start from your terminal's current folder, not the configuration
  file's folder. Full paths, `~`, and environment variables such as `$HOME` work too.
- `run_name`: folder name for one run inside `output_root`, such as `LkCa4_trial_01`.
  Required unless you use `--output-dir`. Using `--paths NAME` adds `_NAME`
  to the folder name.

For example:

```ini
[output]
output_root : /path/to/EXSPLOSYN/results
run_name    : LkCa4_trial_01
```

This saves results in `/path/to/EXSPLOSYN/results/LkCa4_trial_01/`.
`--output-dir PATH` replaces both settings and saves directly into `PATH`,
without adding any names. Missing folders are created when saving results.
Rename the old `save_folder_name` setting to `run_name`.

CSV results are saved in simulation mode regardless of these plot/spectrum
switches. For fitting and debug output behavior, see
[command-line options](command_line.md).

### `[paths]`

Set the directories containing spectral libraries and external tools.

- `phoenix_root`: PHOENIX (2013) high-resolution flux library and download
  directory.
- `phoenix_intensity_root`: PHOENIX intensity library used by LDTk and the
  simulation intensity reader.
- `rassine_root`: directory containing the EXSPLOSYN-compatible RASSINE fork.
- `synmast_feoh`: SYNMAST FeOH library directory.
- Custom keys: add a directory entry and reference its key through a band's
  `library_path`.

Directory values support `~` and environment variables. Relative paths use
the working directory; absolute paths are accepted. Unused library/tool paths
may be left empty.

## Advanced settings

### `[advanced]`

Keep the values in `demo.cfg` for typical use. Adjust magnetic fields or the
non-emitting approximation when needed. Refine numerical sampling when checking
whether the CCF RV results have converged.

**Magnetic fields (SYNMAST only)**

- `photosphere_B`: photospheric SYNMAST magnetic-field strength in Gauss.
  Missing or empty values use `0`; PHOENIX ignores this setting.
- `active_B`: SYNMAST magnetic-field strength in Gauss, shared by all emitting
  active regions. Missing or empty values use `0`; PHOENIX ignores this setting.
  If a region has the same rounded temperature as the photosphere, its magnetic
  field must also match: `active_B` must equal `photosphere_B`.

**Non-emitting regions**

- `black_below_temperature`: approximate regions below this temperature in
  Kelvin as non-emitting, after rounding temperatures to integer Kelvin.
  Leave empty to disable. Regions equal to the threshold remain emitting.
  Gaussian brightness models ignore this setting.

**Spectral sampling and CCF RV-search steps**

- `spec_oversample_factor`: positive integer controlling internal spectral
  samples per instrumental FWHM; default `14`. This changes sampling, not
  resolving power. Actual velocity spacing is limited to 0.1–1.0 km/s;
  disabling instrumental broadening uses 1.0 km/s.
- `ccf_template_oversample_factor`: minimum template sampling density relative
  to the simulated spectrum; default `5`. Denser input templates are retained.
- `ccf_rv_oversample_factor`: controls the spacing between trial velocities
  in the fine CCF RV search; default `50`. The search step equals the simulated
  spectrum's velocity-bin spacing divided by this factor. For example,
  0.306 km/s divided by `50` gives a search step of 0.00612 km/s (6.12 m/s).
  This refines the RV-search grid without changing the spectral sampling or
  resolving power.

Finer sampling increases computation time and may increase memory use.

(named-path-profiles)=
### `[paths:NAME]`

Optionally override `[paths]` for a named machine or data location.

- Entries use the same directory keys as `[paths]`. Only listed entries are
  overridden; all others are inherited.
- Select a profile explicitly with `--paths NAME`. Without this option,
  EXSPLOSYN uses only `[paths]`; it never selects a profile automatically.

The end of `demo.cfg` contains a `[paths:demo]` example. Replace its example
paths with your own before selecting it with `--paths demo`.

### `[references]`

Optional space for citation comments associated with a configuration.
It contains no simulation controls.


### Shared CCF implementation

RV correlation, common-interval selection, coarse/fine peak search and retries
are supplied by the shared [`rv-ccf` package](https://github.com/shihyuntang/rv-ccf). EXSPLOSYN passes its known
`spectrum_vel_step`, so the fine step remains
`spectrum_vel_step / ccf_rv_oversample_factor`; it does not use the package's
wavelength-based sampling estimate. The 1 km/s coarse step, one retry at twice
the range, debug profiles, and finite-phase mean subtraction are unchanged.
Template preparation and spectrum normalization remain in EXSPLOSYN.

EXSPLOSYN pins the tested `rv-ccf` release in `environment.yml` and
`pyproject.toml`. Use the same release for observation processing; update the
pin in both files only after validating a new release.

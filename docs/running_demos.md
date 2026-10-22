# Running the demos

Complete the [installation steps](installation.md) before running these examples.
Activate the `exsplosyn` Conda environment and run all commands below from the
EXSPLOSYN repository root.

Two configurations are supplied in `configs/`:

- **Basic demo — `demo.cfg`:** simulate a star with four active regions using
  PHOENIX spectra and several photometric filters.
- **LkCa 4 demo — `demo_lkca4.cfg`:** explore a 25-region surface model inspired
  by a Zeeman–Doppler imaging (ZDI) reconstruction, using a temperature lookup
  table and the SYNMAST FeOH library.

These examples introduce selected features. For the full set of configuration
options, see the [configuration manual](configuration.md).

## 1. Basic demo: four active regions

Open `configs/demo.cfg`. This example uses PHOENIX spectra, which EXSPLOSYN can
download automatically; no SYNMAST library is required. Check the `[paths]`
settings as described in the installation guide.

### Define the star and active regions

The `[star]` section defines the stellar radius, rotation period, inclination
angle, quiet photospheric temperature (`Tphot`), surface gravity (`logg`), and
metallicity. Together, these determine the rotating stellar geometry and the
local spectra used in the simulation. This demo uses `Tphot : 3670`,
`logg : 4.0`, and `metallicity : 0.0`.

The `[active_regions]` section defines each circular region's longitude,
latitude, size, and temperature. **Each column represents one region:** keep
corresponding entries in the same order across all lists.

A region's `size` is its **disk-center filling factor**, $f_0$: the percentage
of the projected stellar disk covered by that region when its center is at
disk center. It is NOT a percentage of the entire stellar surface. Individual
sizes above 80% are currently unsupported.

The basic demo contains **three cool regions and one hot region**, with
temperatures specified directly:

```ini
temperature_mode : absolute
temperatures     : 3278, 3377, 3484, 3722
```

Alternatively, use signed offsets from `Tphot`:

```ini
temperature_mode    : offset
temperature_offsets : -392, -293, -186, 52
```

With `Tphot : 3670`, these two examples produce the same region temperatures.
`relative` is an alias for `offset`. The LkCa 4 demo below introduces a third
option: a temperature lookup table.

### Choose the spectral bands

After defining the star and active regions, select the spectroscopic
configurations to run in `[spectroscopic]`:

```ini
[spectroscopic]
enabled : phx_rv
```

`phx_rv` uses the PHOENIX (2013) library and the wavelength grid defined in
`[phx_rv]`:

```ini
[phx_rv]
wavelength_start  : 15400
wavelength_end    : 15801
wavelength_step   : 200
window_half_width : 100
```

All four values are in **Angstroms**. `wavelength_start`, `wavelength_end`, and
`wavelength_step` define the band centers; `window_half_width` sets the coverage
on either side of each center. This configuration produces three bands:

| Band | Center (Å) | Wavelength coverage (μm) |
| --- | ---: | ---: |
| `w154` | 15,400 | 1.53–1.55 |
| `w156` | 15,600 | 1.55–1.57 |
| `w158` | 15,800 | 1.57–1.59 |

The default half-width of 100 Å is a starting point that balances wavelength
coverage and computation time. Narrow windows can make continuum fitting
unreliable, particularly for cool stars with strong molecular absorption.
When changing the wavelength coverage, inspect the continuum-fitting plots.
These are saved during a normal simulation when `[output]` includes:

```ini
[output]
make_plots : 1
```

### Choose the photometric filters

The `[photometry]` section selects filters using identifiers from the
[SVO Filter Profile Service](https://svo2.cab.inta-csic.es/theory/fps/):

```ini
[photometry]
filters : TESS/TESS.Red, SLOAN/SDSS.z, 2MASS/2MASS.Ks
```

Separate multiple filters with commas. **Photometry runs when filters are
specified**; it is controlled separately from `[spectroscopic] enabled`. Leave
`filters` empty or set it to `None` to disable photometry.

The selected filters must have wavelength coverage within
**500 Å–5.0 μm**.

### Run the basic demo

```sh
python main.py --config demo
```

A bare configuration name is resolved inside `configs/`, and `.cfg` is added
when omitted. Explicit relative paths and absolute paths are also accepted.

This demo uses `output_root : outputs` and `run_name : demo` under `[output]`,
saving to `outputs/demo/` in your terminal's current folder. To save elsewhere:

```bash
python main.py --config demo --output-dir /path/to/results/demo_trial_02
```

The demo enables summary PDF plots, GIF animations, and saved
final spectra through its `[output]` settings. Reusable spectral data are
stored separately in `spectral_cache/`.

## 2. LkCa 4 demo: 25 active regions

Open `configs/demo_lkca4.cfg`. This example contains **15 cool and 10 hot
regions** and extends the basic demo with temperature labels, SYNMAST FeOH
diagnostics, and a supplied list of rotation phases.

**Before running:** download the SYNMAST FeOH library using the link in the
[installation guide](installation.md), and set `synmast_feoh` under `[paths]`
to the directory containing its spectral files.

### Assign temperatures with a lookup table

In `[active_regions]`, the temperature settings are:

```ini
temperature_mode  : t_lookup_table
temperature_table : temperature_table_lkca4_demo.ini
```

This filename uses `configs/temperature_tables/temperature_table_lkca4_demo.ini`
inside the code folder you are running, regardless of the configuration file's
location. You can also give a relative or full path; see
[temperature-table paths](configuration.md#active-regions).

The `temperature_levels` list assigns one table label to each region. The
first three entries in this example are:

```ini
# First three entries only; retain all 25 entries in the configuration.
temperature_levels : b, c, b
```

The table contains a section for each supported photospheric temperature.
For example:

```ini
[3600]
a : -322
b : -223
c : -116
d :  122
e :  259
f :  426

[3700]
a : -341
b : -235
c : -121
d :  133
e :  293
f :  512
```

**Table values are temperature offsets in Kelvin**, added to `Tphot`.
For `Tphot : 3600`, EXSPLOSYN uses `[3600]`, so the first three regions have:

| Region | Label | Offset (K) | Region temperature (K) |
| --- | --- | ---: | ---: |
| 1 | `b` | −223 | 3377 |
| 2 | `c` | −116 | 3484 |
| 3 | `b` | −223 | 3377 |

Labels are case-sensitive. The table must contain a section matching `Tphot`;
EXSPLOSYN does not interpolate between table sections.

### Add FeOH RV and equivalent-width diagnostics

This demo enables both the PHOENIX RV bands and the SYNMAST FeOH band:

```ini
[spectroscopic]
enabled : phx_rv, FeOH
```

The `[phx_rv]` settings use `wavelength_start : 10000`,
`wavelength_end : 10501`, and `wavelength_step : 200`, producing three bands:
**w100**, **w102**, and **w104**, centered at 10000, 10200, and 10400 Å.
Each has `window_half_width : 100` Å. Together with FeOH, this demo runs
four spectroscopic bands. The FeOH band uses the following configuration:

```ini
[band:FeOH]
source            : synmast
spectrum_mode     : intensity
library_path      : synmast_feoh
center            : 15557.0
window_half_width : 94.0
diagnostics       : ewr_FeOH, rv
```

`library_path : synmast_feoh` refers to the **key in `[paths]`**, where the
library directory is specified. `center` and `window_half_width` are in
Angstroms. `diagnostics : ewr_FeOH, rv` requests the **OH/Fe equivalent-width
ratio** and the **CCF radial velocity**. 

**The `ewr_FeOH` diagnostic is supported only for `[band:FeOH]`.**

### Use a supplied phase list

The `[phase_sampling]` section selects specific rotation phases:

```ini
[phase_sampling]
phase_step :
phase_file : configs/phases/phases_obs.txt
```

Providing `phase_file` overrides regular sampling through `phase_step`.
The first column of the text file must contain rotation phases in cycles.
The file path can be relative to the working directory or absolute.
Replace this file with your own phase list to sample the model at your
observation phases.

### Run the LkCa 4 demo

```sh
python main.py --config demo_lkca4
```

With `output_root : outputs` and `run_name : demo_lkca4` under `[output]`,
results are written to `outputs/demo_lkca4/`.
This demo saves summary PDFs and CSV results. GIF animations are disabled
(`make_gif_animation : 0`) to keep the run shorter; set this option to `1`
if you would like an animation. Final spectral export is disabled
(`save_spectra : 0`).

The default `cpu : 2` uses two worker processes. Use `cpu : 1` to reduce
memory use, or increase it if your computer has sufficient CPU and memory resources.

See [Understanding the outputs](outputs.md) for the saved files and column definitions.

# Understanding the outputs

Results go to `output_root/run_name`, set under `[output]`, or directly into
`--output-dir PATH` when supplied. `output_root` defaults to `outputs` in your
terminal's current folder. See [output settings](configuration.md#output)
for path rules and the effect of `--paths`.

Examples below use `outputs/demo/`. Replace it with your results folder.

## Files in the run folder

- **CSV files:** one per band, including the photometry band (`photo`). These
  contain the phase-dependent measurements.
- **PDF plots:** phase summaries and light curves, enabled by `make_plots`.
- **GIF animations:** enabled independently by `make_gif_animation`.
- `continuum_fitting/`: normalization checks for the simulated spectra and
  applicable RV templates, enabled by `make_plots`.
- `simulated_spotted_spectra/`: final NPZ spectral archives, enabled by
  `save_spectra`.
- `rv_ccf_templates/`: NPZ reference spectra used for CCF RV measurements,
  also enabled by `save_spectra`.
- `model_ids.json`: records the model IDs and their full hashes.

Debug mode adds intermediate diagnostics and a configuration snapshot named
`configuration_id-<model_id>.cfg`. See the [command-line options](command_line.md)
for the differences between simulation, fitting, and debug modes.

## Output filenames

Final outputs use the order **spectral source, band, model ID**. CSV filenames
also retain the model parameters between the band and ID, without equals signs.
For example:

```text
synmast_FeOH_P3.4_i70.0_Tp3600_logg4.0_SpotNum25_Ta3278-3377-3484-3722-3859-4026_id-T3600i70_a262a0bef0.csv
synmast_FeOH_id-T3600i70_a262a0bef0.pdf
synmast_FeOH_id-T3600i70_a262a0bef0.gif
synmast_FeOH_id-T3600i70_a262a0bef0.npz
```

Photometry-only plots include `lightcurves` before the ID. RV-template archives
include their purpose and template temperature, for example:

```text
phx_photo_lightcurves_id-T3600i70_a262a0bef0.pdf
phx_FeOH_rv_ccf_template_T3800K_id-T3600i70_a262a0bef0.npz
```

The source prefix of an RV-template archive describes the **template**, which
may differ from the source of the simulated spectra. User-supplied templates
use `file` as the prefix; an unknown template temperature is labeled `Tunknown`.

## CSV columns

Each row corresponds to one simulated phase. Columns depend on the band:

| Column | Meaning | Units |
| --- | --- | --- |
| `phase` | Rotation phase | Cycles |
| `rv_km_s` | CCF radial velocity relative to the selected template | km/s |
| `ewr_oh_fe` | OH/Fe equivalent-width ratio | Dimensionless |
| `<filter>_relative_flux` | Photometric flux divided by its mean over the supplied phases | Dimensionless |
| `apparent_teff_K` | Apparent effective temperature from the bolometric calculation | K |
| `apparent_luminosity_erg_s` | Apparent bolometric luminosity | erg/s |
| `active_filling_f` | Total projected visible-disk active-region filling factor | Fraction |
| `cool_filling_f` | Projected visible-disk cool-region filling factor | Fraction |
| `hot_filling_f` | Projected visible-disk hot-region filling factor | Fraction |
| `first_moment_rv_km_s` | Gaussian-profile first-moment radial velocity | km/s |

**Filling factors are fractions, not percentages:** `0.25` means 25% of the
projected visible disk. These vary with phase and differ from the input
`sizes`, which specify disk-center filling factors in percent.

A diagnostic that was not requested can appear as `NaN`; for example,
`ewr_oh_fe` is not calculated for a PHOENIX RV band. Photometry CSVs 
contain one relative-flux column per filter, together with apparent 
effective temperature, apparent bolometric luminosity, and active-region 
filling factors. Gaussian CSVs contain only `phase`, `first_moment_rv_km_s`, 
and `active_filling_f`.

## Saved spectra

With `save_spectra : 1`, each physical band has one NPZ archive containing all
its phases. Spectroscopic archives include:

| Array | Contents |
| --- | --- |
| `phase` | Rotation phases |
| `wavelength_angstrom` | Shared wavelength grid in Angstroms |
| `velocity_kms` | Shared velocity grid in km/s |
| `quiet_photosphere_flux` | Quiet-star spectrum before phase-dependent replacements |
| `active_replacement_quiet_flux` | Quiet emission removed at active-region pixels, at each phase |
| `active_region_flux` | Active-region emission added at each phase |
| `spotted_flux` | Combined spectrum before scaling and instrumental broadening |
| `processed_spotted_flux` | Spectrum after phase-by-phase scaling and instrumental broadening, when enabled |
| `normalized_processed_spotted_flux` | Processed spectrum after continuum normalization |
| `model_id`, `model_hash` | Physical-model identifiers |

Phase-dependent spectra have shape `(number of phases, number of wavelength
samples)`. The shared grids and quiet-star spectrum are one-dimensional.
The raw flux arrays combine directly:

```python
spotted_flux = (
    quiet_photosphere_flux
    - active_replacement_quiet_flux
    + active_region_flux
)
```

Start with the quiet-star spectrum, subtract the quiet emission at pixels
occupied by active regions, then add the emission from those regions.
This gives the spotted spectrum at each phase, before phase-by-phase scaling,
instrumental broadening, or continuum normalization. The processed and
normalized arrays include these additional steps and should not be substituted
into this equation.

Photometry archives store the unnormalized spectra and their quiet/active
contributions used to calculate flux through the selected filters.
Gaussian archives use separate arrays, including `raw_phase_spectra` and
`normalized_phase_spectra`.

To load the first saved `w154` spectrum from the basic demo:

```python
from pathlib import Path
import numpy as np

path = next(Path("outputs/demo/simulated_spotted_spectra").glob("phx_w154_id-*.npz"))
with np.load(path, allow_pickle=False) as data:
    phase = data["phase"].copy()
    wavelength = data["wavelength_angstrom"].copy()
    flux = data["normalized_processed_spotted_flux"].copy()

first_phase_spectrum = flux[0]
```

RV-template archives contain `wavelength_angstrom`, `velocity_kms`, and
continuum-normalized `flux`, together with the model identifiers. Debug mode
also includes `metadata_json` with additional preparation information.

## Model IDs and rerunning a model

Output filenames include an ID such as `id-T3670i70_0123456789` (illustrative).
The readable prefix gives the photospheric temperature and inclination; the
hash identifies effective stellar and active-region parameters, including
the disk grid size. Use this ID to match CSVs, spectra, and plots from the
same physical model.

**A model ID is not a unique run ID.** It does not include every calculation
setting, such as the phase list, spectral bands, instrumental resolution, or
CCF sampling. Rerunning a model can overwrite files with matching names.
Choose a different `run_name` or `--output-dir` folder to keep runs separate.

## Spectral cache

`spectral_cache/` holds spectra and prepared data for reuse between runs,
including additional spectra saved in debug mode. It stays in your terminal's
current folder when you change the results location.

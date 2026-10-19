# Model applicability and spectral grids

EXSPLOSYN can simulate a wide range of stellar parameters when the required
spectra are available. However, not all of these parameters fall within the
scientifically tested range of the model.

## Scientific scope

The current version of EXSPLOSYN targets **M dwarfs and T Tauri stars (TTS)**.
Testing so far has focused on stars with photospheric temperatures
**below about 4500 K**. This is the tested range, not a hard upper temperature
limit.

A hotter model, such as a **6000 K photosphere using PHOENIX**, can run when
the required spectra are available and the input parameters are accepted.
However, a successful run does not mean that all relevant physics is included.
Keep the following assumptions and limitations in mind:

- The star rotates as a rigid body; differential rotation is not included.
- Convective blueshift is not simulated.
- Hot regions use the same type of one-dimensional LTE photospheric spectra
  as cool regions, with different temperatures.

For a more detailed discussion, see **Tang, Johns-Krull, and Prato 2026 (in prep.)**.
Continuum normalization above 7000 K also requires careful inspection.

## Spectral-library coverage

Coverage differs between flux and intensity libraries. A library's overall
range does not guarantee that every combination of temperature, gravity,
metallicity, and magnetic field is present.

### PHOENIX (2013)

The published PHOENIX library spans 2300–12000 K, $\log g$ 0–6, and [Fe/H] −4 to
+1; see [Husser et al. (2013)](https://doi.org/10.1051/0004-6361/201219058).
The current EXSPLOSYN reader uses the following grid axes:

| Parameter | High-resolution flux | Intensity, shared with LDTk |
| --- | --- | --- |
| Temperature | 2300–11600 K | 2300–12000 K |
| Temperature spacing | 100 K below 7000 K; 200 K from 7000 K | 100 K through 7000 K; 200 K above 7000 K |
| $\log g$ | 0–6, in steps of 0.5 | 0–6, in steps of 0.5 |
| [Fe/H] | −4, −3, −2, −1.5, −1, −0.5, 0, +0.5, +1 | −4, −3, −2, −1.5, −1, 0, +0.5, +1 |
| Magnetic field | Not a model axis | Not a model axis |

Some intensity-grid combinations lack files. For a missing model, the reader
preserves $\log g$ and metallicity and searches for the nearest available
temperature. This substitution does not expand the scientifically tested
regime. Inspect debug output when checking the spectra used for interpolation.

Photometry currently requires all selected filter wavelengths to lie within
**500–50,000 Å (0.05–5.0 μm)**.

### SYNMAST FeOH

The currently available SYNMAST FeOH library covers:

| Parameter | Grid coverage | Spacing |
| --- | --- | --- |
| Temperature | 2500–5600 K | 100 K |
| $\log g$ | 3–5 | 0.5 |
| [Fe/H] | −1, −0.5, 0, +0.5, +1 | 0.5 dex |
| Magnetic-field strength | 0, 2000, 4000, 6000 G | 2000 G |
| Wavelength | Approximately 1.5444–1.5679 μm | Native library sampling |

## Choosing the spectral inputs

EXSPLOSYN uses different spectral inputs for different calculations, balancing
wavelength coverage, spectral detail, and computation time. The following
choices are used in the supplied configurations:

1. **Broad-band photometry and apparent bolometric luminosity:** use
   low-resolution PHOENIX-ACES **specific intensity spectra**, which describe
   how the emergent spectrum varies from disk center to limb. Their broad,
   continuous wavelength coverage makes them suitable for integration over
   photometric filters and for estimating bolometric luminosity.

2. **RV studies across optical and near-infrared wavelengths:** use
   high-resolution PHOENIX-ACES **flux spectra**, selected through `phx_rv`.
   Their broad wavelength coverage makes them well suited to studying
   spot-induced RV signals across multiple wavelength regions.

3. **Individual spectral features and RV studies in narrow wavelength
   regions:** use high-resolution SYNMAST **specific intensity spectra**,
   selected with `source : synmast` and `spectrum_mode : intensity` in a
   `[band:NAME]` section. These custom libraries are tuned to better reproduce
   observed spectral features while retaining their viewing-angle dependence.

Specific intensity spectra are often described as **$\mu$-dependent**, where
$\mu$ is the cosine of the angle between the local surface normal and the
line of sight. Flux spectra have already been integrated over the outgoing
angles, so they require a separate limb-darkening treatment in the simulation.

For diagnostics based on individual line depths and shapes, the quality of
the adopted line list is particularly important. The SYNMAST library uses
fine-tuned line lists to better reproduce these features, making it useful
for temperature-sensitive diagnostics such as the FeOH ratio.

The **RV CCF template is a separate choice**, set in `[rv_ccf]`. Its default is
a normalized PHOENIX flux spectrum, even when the simulated band uses SYNMAST.
See the [configuration manual](configuration.md) for the available settings.

## Program limits and interpolation behavior

The current version accepts temperatures between **2500 and 10000 K** for
the photosphere, emitting active regions, and PHOENIX RV templates. The spectra
needed for your model must also be available in the selected library. These
input limits do not change the scientific scope described above.

For cooler regions, `black_below_temperature` can be used to approximate them
as completely dark. These regions do not require a local spectrum.

When the requested parameters fall between grid points, EXSPLOSYN interpolates
in **temperature, surface gravity, and metallicity**. For SYNMAST spectra,
it also interpolates in **magnetic-field strength**. The surrounding grid
files must be available; missing SYNMAST grid points raise an error.

Parameters outside a library's grid use the **nearest boundary value**;
EXSPLOSYN does not extrapolate. For example, requesting 6000 K from the FeOH
library above uses the 5600 K boundary spectrum. This does not produce a
6000 K FeOH spectrum.

Check the library coverage for both the photosphere and active regions before
running your model. To see which spectra and grid points are used, run with
`--mode debug`; see the [command-line options](command_line.md).

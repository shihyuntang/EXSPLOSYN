# EXSPLOSYN

**EXOplanetary Spotted Photosphere LOw-mass SYNthesizer**

EXSPLOSYN simulates high-resolution optical and infrared spectra of spotted
low-mass stars, including T Tauri stars and M dwarfs. It integrates local
synthetic spectra across a rotating stellar disk with cool and hot active
regions, allowing you to explore how surface inhomogeneities affect spectral
lines, radial velocities, and photometric variability.

## Features

- **High-resolution spectral modeling.** Generate phase-resolved spectra using
  temperature-dependent local spectra, stellar rotation, and limb darkening.
  Multiple circular active regions have individually specified positions,
  sizes, and temperatures.
- **Spectroscopic and photometric predictions.** Predict line-profile
  variability, spot-induced radial velocities, the FeOH equivalent-width ratio,
  and relative light curves from the same stellar surface configuration.
- **Configurable RV reference spectra.** Measure radial velocities through
  cross-correlation with a PHOENIX template or a user-supplied spectrum, allowing
  you to minimize differences between observations and simulations.
- **Flexible spectral grids.** Use PHOENIX (2013) or custom SYNMAST libraries,
  with interpolation in temperature, surface gravity, and metallicity.
  SYNMAST grids also support interpolation in magnetic-field strength.
- **Multi-band photometry.** Integrate the simulated spectra over filter
  transmission curves from the SVO Filter Profile Service to compare activity
  signals across photometric bands.
- **Phase-dependent apparent stellar properties.** Calculate apparent effective
  temperature and apparent bolometric luminosity, together with the visible-disk
  filling factors of cool and hot regions.

Support for PHOENIX 2025 is planned; see the [roadmap](version_roadmap.md).

## Getting started

Start with the [installation instructions](installation.md), then follow
[Running the demos](running_demos.md) to run your first simulations and learn
the key settings. [Understanding the outputs](outputs.md) explains the saved
results. Use the [configuration manual](configuration.md) and
[command-line options](command_line.md) as references when adjusting your model.

```{toctree}
:maxdepth: 1
:caption: Documentation

installation
running_demos
outputs
configuration
command_line
model_limits
benchmarking
version_roadmap
```

## Reference

If you use EXSPLOSYN in your research, please cite:

**Tang, Johns-Krull, and Prato 2026 (in prep.).**

## Contact

If you have questions or would like to report a bug, please
[open a GitHub issue](https://github.com/shihyuntang/EXSPLOSYN/issues) or email
[shihyun.tang@austin.utexas.edu](mailto:shihyun.tang@austin.utexas.edu).

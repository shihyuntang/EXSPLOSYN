"""Prepare spectral inputs, simulation spectra, and RV templates."""

import logging
import os
from pathlib import Path

import numpy as np

from . import spectral_io
from . import diagnostics
from . import spectral_preprocessing
from . import rassine_adapter
from .configuration import MIN_TEMPERATURE_K, MAX_TEMPERATURE_K

logger = logging.getLogger(__name__)


def _component_magnetic_field(star, temperature):
    """Select the SYNMAST field strength in Gauss for this temperature."""
    if temperature == star.photosphere_temperature:
        return getattr(star, "photosphere_B", 0.0)
    return getattr(star, "active_B", 0.0)


def prepare_flux_spectra(star, spectrum_vel_step, plot, savehdf=None):
    """Prepare simulation flux spectra for all component temperatures.

    Args:
        spectrum_vel_step (float): Velocity sampling interval in km/s.
        plot (bool): If True, plot intermediate results.
        savehdf (bool, optional): Whether to save prepared spectra.
    """
    # Include the photosphere and all cool and hot regions.
    component_temperatures = np.unique(
        np.concatenate([[star.photosphere_temperature], star.T_cool, star.T_hot])
    )
    star.flux_spec = {}

    for band, (
        band_center,
        window_half_width,
        _,
        spectrum_mode,
        spectrum_source,
    ) in star.bands.items():
        # Only process bands that require flux spectra
        if spectrum_mode != "flux":
            continue
        if spectrum_source.lower() == "gaussian_profile":
            continue

        logger.debug("Simulation spectrum: band=%s, source=%s, mode=%s",
                     band, spectrum_source, spectrum_mode)
        # Define wavelength window for this band
        wavelength_min = int(band_center - window_half_width)
        wavelength_max = int(band_center + window_half_width)

        for component_temperature in component_temperatures:
            if spectrum_source.lower() == "phx":
                prepared_path = spectral_io.prepared_spectrum_path(
                    star,
                    "simulation_ready",
                    spectrum_source,
                    spectrum_mode,
                    component_temperature,
                    band,
                    wavelength_min,
                    wavelength_max,
                    spectrum_vel_step,
                )
                native_spectrum = star.phx_flux_spectra[component_temperature]
                prepared_spectrum = spectral_preprocessing.prepare_flux_spectrum(
                    spectrum_source,
                    native_spectrum,
                    prepared_path,
                    wavelength_min,
                    wavelength_max,
                    spectrum_vel_step,
                    plot,
                    flatten=False,
                    savehdf=savehdf,
                )

            elif spectrum_source.lower() == "synmast":
                magnetic_field = _component_magnetic_field(star, component_temperature)
                native_spectrum = spectral_io.load_synmast_flux_spectrum(
                    component_temperature,
                    star.logg,
                    star.metallicity,
                    star.band_library_paths[band],
                    magnetic_field=magnetic_field,
                )
                prepared_path = spectral_io.prepared_spectrum_path(
                    star,
                    "simulation_ready",
                    spectrum_source,
                    spectrum_mode,
                    component_temperature,
                    band,
                    wavelength_min,
                    wavelength_max,
                    spectrum_vel_step,
                    magnetic_field=magnetic_field,
                )
                prepared_spectrum = spectral_preprocessing.prepare_flux_spectrum(
                    spectrum_source,
                    native_spectrum,
                    prepared_path,
                    wavelength_min,
                    wavelength_max,
                    spectrum_vel_step,
                    plot,
                    flatten=False,
                    savehdf=savehdf,
                )

            else:
                logger.error("Unsupported spectrum source: %s", spectrum_source)
                continue

            # Keep the prepared arrays for this band and temperature.
            velocity = prepared_spectrum["v"].values.astype(np.float32, copy=False)
            values = prepared_spectrum["f"].values.astype(np.float64, copy=False)
            prefix = (
                f"{band}_{spectrum_source}_{spectrum_mode}_{component_temperature:1.4f}"
            )
            spec = {
                f"{prefix}_v": velocity,
                f"{prefix}_w": prepared_spectrum["wavelength_angstrom"].to_numpy(
                    dtype=np.float64, copy=False
                ),
                f"{prefix}_f": values,
            }

            star.flux_spec.update(spec)


def prepare_spectral_inputs(star, rv_template_temperature=None):
    """Initialize spectral storage and prepare required PHOENIX flux files.

    PHOENIX flux simulations require spectra at all component
    temperatures. PHOENIX-based RV templates require only the
    selected template temperature.

    Args:
        rv_template_temperature (float, optional): RV CCF template
            temperature in Kelvin. None selects the photospheric temperature.
    """
    star.out_path = os.path.join(os.getcwd(), "spectral_cache")
    star.phx_flux_file = {}
    star.phx_flux_spectra = {}
    star.spec_flatten = {}

    diagnostics_by_band = getattr(star, "band_diagnostics", {})
    temperatures = []

    for band, params in star.bands.items():
        _, _, diagnostic_mode, spectrum_mode, spectrum_source = params

        if spectrum_source.lower() == "phx" and spectrum_mode == "flux":
            temperatures.extend(
                np.concatenate(
                    [
                        [star.photosphere_temperature],
                        star.T_cool,
                        star.T_hot,
                    ]
                )
            )

        selected_diagnostics = diagnostics_by_band.get(band, ("rv",))
        needs_template = (
            diagnostic_mode == "rv"
            and "rv" in selected_diagnostics
            and spectrum_source.lower() != "gaussian_profile"
            and getattr(star, "ccf_template_source", "phx") == "phx"
        )
        if needs_template:
            temperatures.append(
                star.photosphere_temperature
                if rv_template_temperature is None
                else rv_template_temperature
            )

    if temperatures:
        temperatures = np.unique(np.asarray(temperatures, dtype=np.float64))
        logger.debug(
            "Preparing PHOENIX flux inputs at temperatures %s K",
            temperatures,
        )
        star.prepare_phx2013_flux_inputs(temperatures, alpha=0)


def prepare_rv_templates(
    star, spectrum_vel_step, plot, rv_template_temperature=None, savehdf=None
):
    """Prepare PHOENIX-based templates for bands requesting RV.

    Call prepare_spectral_inputs() before this method.

    Args:
        spectrum_vel_step (float): Velocity sampling interval in km/s.
        plot (bool): Whether to generate preprocessing plots.
        rv_template_temperature (float, optional): Template temperature in Kelvin.
            None selects the photospheric temperature.
        savehdf (bool, optional): Whether to export intermediate template spectra.
        star: Initialized object with library/tool paths in star.spectral_paths.
    """
    star.spec_flatten = {}
    if getattr(star, "ccf_template_source", "phx") == "file":
        return

    paths = star.spectral_paths
    diagnostics_by_band = getattr(star, "band_diagnostics", {})
    template_temperature = np.float64(
        star.photosphere_temperature
        if rv_template_temperature is None
        else rv_template_temperature
    )

    for band, (
        band_center,
        window_half_width,
        diagnostic_mode,
        spectrum_mode,
        spectrum_source,
    ) in star.bands.items():
        selected_diagnostics = diagnostics_by_band.get(band, ("rv",))
        if diagnostic_mode != "rv" or "rv" not in selected_diagnostics:
            continue
        if spectrum_source.lower() == "gaussian_profile":
            continue
        if spectrum_source.lower() not in {"phx", "synmast"}:
            logger.error("Unsupported spectrum source: %s", spectrum_source)
            continue

        wavelength_min = int(band_center - window_half_width)
        wavelength_max = int(band_center + window_half_width)

        # The default RV template uses PHOENIX even for SYNMAST simulations.
        continuum_path = rassine_adapter.flatten_phx_with_rassine(
            spectral_io.require_spectral_path(paths, "rassine_root"),
            star.out_path,
            star.phx_flux_file[template_temperature],
            keep_intermediate=bool(savehdf),
        )
        logger.debug("RV CCF template for %s: PHOENIX T=%g K; continuum=%s",
                     band, template_temperature, Path(continuum_path).absolute())
        prepared_path = spectral_io.prepared_spectrum_path(
            star,
            "rv_ccf_templates",
            "phx",
            "normalized_flux",
            template_temperature,
            band,
            wavelength_min,
            wavelength_max,
            spectrum_vel_step,
        )
        # Normalize by the continuum, cut the band, and resample in velocity.
        prepared_template = spectral_preprocessing.prepare_flux_spectrum(
            "phx",
            continuum_path,
            prepared_path,
            wavelength_min,
            wavelength_max,
            spectrum_vel_step,
            plot,
            flatten=True,
            savehdf=savehdf,
        )
        velocity = prepared_template["v"].values.astype(np.float32, copy=False)
        flux = prepared_template["f"].values.astype(np.float64, copy=False)
        # Lookup keys follow the simulation band; the template source is PHOENIX.
        prefix = f"{band}_{spectrum_source}_{spectrum_mode}_{template_temperature:1.4f}"
        star.spec_flatten.update({f"{prefix}_v": velocity, f"{prefix}_f": flux})


def prepare_intensity_spectra(star, spectrum_vel_step, plot, savehdf=None):
    """Prepare simulation intensity spectra for all component temperatures.

    Args:
        spectrum_vel_step (float): Velocity sampling interval in km/s.
        plot (bool): If True, plot intermediate results.
        savehdf (bool, optional): Whether to save prepared spectra.
        star: Initialized object with library/tool paths in star.spectral_paths.
    """
    # Include the photosphere and all cool and hot regions.
    component_temperatures = np.unique(
        np.concatenate([[star.photosphere_temperature], star.T_cool, star.T_hot])
    )
    star.int_spec = {}
    # Keep only compact bolometric integrals from this preparation.
    star.bolometric_intensities = {}
    paths = star.spectral_paths

    for band, (
        band_center,
        window_half_width,
        _,
        spectrum_mode,
        spectrum_source,
    ) in star.bands.items():
        # Only process bands that require intensity spectra
        if spectrum_mode != "intensity":
            continue

        logger.debug("Simulation spectrum: band=%s, source=%s, mode=%s",
                     band, spectrum_source, spectrum_mode)
        # Band wavelength limits are in Angstroms.
        wavelength_min = int(band_center - window_half_width)
        wavelength_max = int(band_center + window_half_width)

        for component_temperature in component_temperatures:
            if spectrum_source.lower() == "phx":
                native_spectrum = spectral_io.load_phx2013_intensity_spectrum(
                    component_temperature,
                    star.logg,
                    star.metallicity,
                    0,
                    library_path=spectral_io.require_spectral_path(
                        paths, "phoenix_intensity_root"
                    ),
                )
                if component_temperature not in star.bolometric_intensities:
                    star.bolometric_intensities[component_temperature] = (
                        diagnostics.calculate_bolometric_intensities(
                            native_spectrum, component_temperature
                        )
                    )
                if savehdf:
                    spectral_io.save_phx2013_intensity_spectrum(
                        native_spectrum, star.out_path
                    )
                prepared_path = spectral_io.prepared_spectrum_path(
                    star,
                    "simulation_ready",
                    spectrum_source,
                    spectrum_mode,
                    component_temperature,
                    band,
                    wavelength_min,
                    wavelength_max,
                    spectrum_vel_step,
                )
                prepared_spectrum = spectral_preprocessing.prepare_intensity_spectrum(
                    spectrum_source,
                    native_spectrum,
                    prepared_path,
                    wavelength_min,
                    wavelength_max,
                    spectrum_vel_step,
                    plot,
                    savehdf=savehdf,
                    trim_edges=False,
                )

            elif spectrum_source.lower() == "synmast":
                magnetic_field = _component_magnetic_field(star, component_temperature)
                library_path = star.band_library_paths[band]
                native_spectrum = spectral_io.load_synmast_intensity_spectrum(
                    component_temperature, star.logg, star.metallicity, library_path,
                    magnetic_field=magnetic_field,
                )
                if savehdf:
                    spectral_io.save_synmast_intensity_spectrum(
                        native_spectrum, star.out_path, band_name=band
                    )
                prepared_path = spectral_io.prepared_spectrum_path(
                    star,
                    "simulation_ready",
                    spectrum_source,
                    spectrum_mode,
                    component_temperature,
                    band,
                    wavelength_min,
                    wavelength_max,
                    spectrum_vel_step,
                    magnetic_field=magnetic_field,
                )
                prepared_spectrum = spectral_preprocessing.prepare_intensity_spectrum(
                    spectrum_source,
                    native_spectrum,
                    prepared_path,
                    wavelength_min,
                    wavelength_max,
                    spectrum_vel_step,
                    plot,
                    savehdf=savehdf,
                )

            else:
                logger.error("Unsupported spectrum source: %s", spectrum_source)
                continue

            # Extract mu grid and intensity columns
            mu_columns = [
                col for col in prepared_spectrum.columns if col.startswith("mu_")
            ]
            intensity_columns = prepared_spectrum[mu_columns]

            velocity = prepared_spectrum["v"].to_numpy(
                dtype=np.float32, copy=False
            )  # (n_v,)
            values = intensity_columns.to_numpy(
                dtype=np.float64, copy=False
            )  # (n_v, n_mu)

            prefix = (
                f"{band}_{spectrum_source}_{spectrum_mode}_{component_temperature:1.4f}"
            )
            spec = {
                f"{prefix}_v": velocity,
                f"{prefix}_w": prepared_spectrum["wavelength_angstrom"].to_numpy(
                    dtype=np.float64, copy=False
                ),
                f"{prefix}_f": values,
            }
            star.int_spec.update(spec)

        # Store the viewing-angle cosines alongside the intensity spectra.
        mu_grid = [
            float(col.split("_")[1])
            for col in prepared_spectrum.columns
            if col.startswith("mu_")
        ]
        star.int_spec[f"mu_{spectrum_source}"] = np.array(mu_grid, dtype=np.float32)


def prepare_phx2013_flux_inputs(star, temperatures, alpha=0):
    """Load PHOENIX 2013 flux spectra and export HDF files for RASSINE.

    Args:
        temperatures (array): Component and template temperatures in Kelvin.
        alpha (float): Alpha-element abundance enhancement [alpha/Fe].
    """
    star.phx_flux_file = {}
    star.phx_flux_spectra = {}
    paths = star.spectral_paths
    for component_temperature in temperatures:
        if (
            component_temperature is None
            or not np.isfinite(component_temperature)
            or not MIN_TEMPERATURE_K <= component_temperature <= MAX_TEMPERATURE_K
        ):
            raise ValueError(
                f"PHOENIX flux temperature must be between {MIN_TEMPERATURE_K} and {MAX_TEMPERATURE_K} K; "
                f"got {component_temperature!r}."
            )
        spectrum = spectral_io.load_phx2013_flux_spectrum(
            component_temperature,
            star.logg,
            star.metallicity,
            alpha,
            spectral_io.require_spectral_path(paths, "phoenix_root"),
        )
        star.phx_flux_spectra[component_temperature] = spectrum
        # Retain the HDF export required by the RASSINE template path.
        phx_flux_path = spectral_io.save_phx2013_flux_spectrum(spectrum, star.out_path)
        star.phx_flux_file[component_temperature] = Path(phx_flux_path).name

"""Read simulation settings and run each spectral band."""

import logging
import json
from time import time

import numpy as np
from astropy.table import Table

from . import diagnostics, spectral_processing, ccf_search
from . import surface_integration as stsp
from . import visualization as vis
from .spectral_processing import apply_instrumental_broadening
from .gaussian_profiles import integrate_gaussian_profiles
from .spectral_data import PreparedSpectrum
from .simulation_setup import SOLAR_RADIUS
from .output_options import resolve_output_options
from .model_identity import assign_model_identity, register_model_identity, identity_arrays, output_stem
from pathlib import Path


class _UniqueGridWarning(logging.Filter):
    """Report each missing PHOENIX corner once per logging setup/run."""

    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record):
        message = record.getMessage()
        if message.removeprefix("Warning: ").startswith("Missing official PHOENIX intensity corner:"):
            if message in self.seen:
                return False
            self.seen.add(message)
        return True


def setup_logging(debug: bool = False, *, quiet: bool = False) -> logging.Logger:
    """Set the EXSPLOSYN log level and return its terminal logger."""
    level = logging.WARNING if quiet else (logging.DEBUG if debug else logging.INFO)
    logger = logging.getLogger("exsplosyn")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    formatter = logging.Formatter(
        "[%(levelname)s] %(message)s" if debug else "%(message)s"
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)
        for log_filter in list(handler.filters):
            if isinstance(log_filter, _UniqueGridWarning):
                handler.removeFilter(log_filter)
        handler.addFilter(_UniqueGridWarning())
    return logger


def prepare_simulation_spectra(
        bands, star, plot_flag, spectrum_vel_step, savehdf=None, *, mode=None
        ):
    """Prepare local component spectra for surface integration.

    Args:
        bands (dict): Configured bands, in simulation order.
        star (Star): Stellar settings and prepared spectral inputs.
        plot_flag (bool): Whether to plot spectral preprocessing results.
        spectrum_vel_step (float): Velocity sampling interval in km/s.
        savehdf (bool or None): Whether to save prepared spectra.

    Returns:
        list: One temperature-to-PreparedSpectrum dictionary per band. Gaussian entries
        are None; their profiles are stored in star.gaussian_profiles.
    """
    options = getattr(star, "output_options", None)
    if mode is not None:
        options = resolve_output_options(mode)
    if options is not None:
        savehdf = options.save_intermediate
        plot_flag = int(options.save_intermediate)

    # Include the photosphere and all cool and hot component temperatures.
    # Reset even when this run does not prepare PHX intensity spectra.
    star.bolometric_intensities = {}
    component_temperatures = np.unique(
        np.concatenate(([star.photosphere_temperature], star.T_cool, star.T_hot))
    )

    # Gaussian limb darkening uses the photosphere; region brightness is set separately.
    physical_bands = {
        name: p for name, p in bands.items() if p[4] != "gaussian_profile"
    }
    gaussian_bands = {
        name: p for name, p in bands.items() if p[4] == "gaussian_profile"
    }
    # Gaussian profiles do not require library flux or intensity spectra.
    spectrum_modes = {params[3] for params in physical_bands.values()}

    ld_coefficients = {}
    # LDTk defines a shared limb from the mean profile of the selected bands.
    # Removing intensity/photo bands changes flux-band LD coefficients and RV.
    for selected, temperatures in (
        (physical_bands, component_temperatures),
        (gaussian_bands, [star.photosphere_temperature]),
    ):
        if selected:
            star.prepare_quadratic_limb_darkening(
                temperatures,
                list(selected),
                np.array([p[0] for p in selected.values()], dtype=np.float32),
                np.array([p[1] for p in selected.values()], dtype=np.float32),
            )
            ld_coefficients.update(star.LD_coeff)
    star.LD_coeff = ld_coefficients

    # Prepare local flux spectra when required.
    if "flux" in spectrum_modes:
        star.prepare_flux_spectra(spectrum_vel_step, plot_flag, savehdf=savehdf)

    if gaussian_bands:
        star.prepare_gaussian_profiles()

    if "intensity" in spectrum_modes or "photo" in spectrum_modes:
        star.prepare_intensity_spectra(spectrum_vel_step, plot_flag, savehdf=savehdf)

    star.prepare_black_region_spectra()

    spectrum_temperatures = component_temperatures

    if getattr(star, "has_black_spots", False):
        spectrum_temperatures = np.unique(
            np.concatenate(
                [
                    component_temperatures,
                    [star.black_spot_sentinel],
                ]
            )
        )
    # Keep per-band spectra aligned with the band order used by main_model.
    band_spectra = []
    for band_name, params in bands.items():
        _, _, _, spectrum_mode, spectrum_source = params
        if spectrum_source == "gaussian_profile":
            band_spectra.append(None)
            continue
        spec_dict = {
            np.round(temperature, 4): PreparedSpectrum(
                star=star,
                band=band_name,
                temperature=temperature,
                spectrum_mode=spectrum_mode,
                spectrum_source=spectrum_source,
            )
            for temperature in spectrum_temperatures
        }
        band_spectra.append(spec_dict)

    return band_spectra


def simulation_plot(
    grid_size,
    nrho,
    instrument_res,
    star,
    spectrum_source,
    save_dir,
    active_regions_number,
    active_region_box,
    phases,
    integrated_stellar_spectrum,
    ccf_rv_shift,
    final_2d_surface_flux_sum,
    outdir,
    cpu_use,
    ewr_oh_fe,
    band,
    spectrum_mode,
    filters,
    relative_flux_curves,
    *,
    region_cube=None,
    region_brightness=None,
    region_fspot=None,
    region_order=None,
    photometry_for_plot=None,
):
    """Plot spectral diagnostics or relative photometric flux.

    Fixed model information summarizes all configured regions, including
    zero-emission regions. Physical regions use temperature offsets from the
    photosphere; Gaussian regions use relative brightness.
    """
    if band == "photo":
        vis.save_lightcurve_plot(
            phases, filters, relative_flux_curves,
            Path("outputs") / save_dir
            / f"{output_stem(star, spectrum_source, band, details='lightcurves')}.pdf",
        )
        return

    gaussian = spectrum_source == "gaussian_profile"
    if gaussian:
        # Gaussian regions describe brightness relative to the quiet photosphere.
        region_values = np.asarray(region_brightness)[1:]
        region_counts = (
            f"{np.count_nonzero(region_values < 1)} dim, "
            f"{np.count_nonzero(region_values > 1)} bright, "
            f"{np.count_nonzero(region_values == 1)} neutral"
        )
        zero_emission_count = np.count_nonzero(region_values == 0)
        value_label = "Brightness"
        value_unit = ""
    else:
        # Physical temperatures include black regions without the internal marker.
        physical_temperatures = getattr(star, "region_temperatures", None)
        if physical_temperatures is None:
            physical_temperatures = active_region_box[3]
        region_values = np.asarray(physical_temperatures) - star.photosphere_temperature
        region_counts = (
            f"{np.count_nonzero(region_values < 0)} cool, "
            f"{np.count_nonzero(region_values > 0)} hot, "
            f"{np.count_nonzero(region_values == 0)} neutral"
        )
        zero_emission_count = np.count_nonzero(getattr(star, "region_black_mask", []))
        value_label = r"$\Delta T$"
        value_unit = " K"

    # --- Plot the simulated results ---
    source_label = {"phx": "PHX2013", "synmast": "SYNMAST",
                    "gaussian_profile": "Gaussian"}.get(spectrum_source, spectrum_source)
    # Three compact lines retain stellar parameters, library source and grid settings.
    stellar_rows = (
        (
            rf"$T_{{\rm phot}}$ = {star.photosphere_temperature:.0f} K",
            rf"$R_\ast$ = {star.rad / SOLAR_RADIUS:.2f} $R_{{\odot}}$",
            rf"$P_{{\rm rot}}$ = {star.prot:.2f} d",
            rf"$i_\ast$ = {star.inclination_angle:.1f} deg",
        ),
        (
            rf"$v\sin i$ = {star.vsini:.1f} km/s",
            f"log $g$ = {star.logg:.2f}",
            f"[Fe/H] = {star.metallicity:+.2f}",
            f"R = {instrument_res:d}",
        ),
        (
            rf"Grid = ${grid_size}\times {grid_size}$",
            f"n_rho = {nrho}",
            f"Band = {band}",
            f"{source_label}/{spectrum_mode}",
        ),
    )
    # Separate cells with |; visualization measures their rendered widths, including symbols.
    column_widths = [max(len(value) for value in column) for column in zip(*stellar_rows)]
    stellar_info_text = "\n".join(
        " | ".join(f"{value:<{width}}" for value, width in zip(row, column_widths)).rstrip()
        for row in stellar_rows
    )
    # Count all configured regions, independent of phase visibility or sorting.
    region_info_text = f"Active regions: {active_regions_number} ({region_counts})\n"
    if region_values.size:
        minimum, maximum = np.min(region_values), np.max(region_values)
        number_format = ".3f" if gaussian else "+.0f"
        region_info_text += (
            fr"min({value_label}) = ${minimum:{number_format}}${value_unit} | "
            fr"max({value_label}) = ${maximum:{number_format}}${value_unit}" + "\n"
        )
    # These regions remain in the geometry but contribute no emitted flux.
    region_info_text += f"Zero-emission regions: {zero_emission_count}"
    options = getattr(star, "output_options", None)
    vis.save_simulation_plots(
        star,
        grid_size,
        phases,
        integrated_stellar_spectrum,
        ccf_rv_shift,
        final_2d_surface_flux_sum,
        outdir,
        cpu_use,
        ewr_oh_fe,
        stellar_info_text,
        region_info_text,
        band,
        spectrum_mode,
        spectrum_source,
        region_cube=region_cube,
        region_brightness=region_brightness,
        region_fspot=region_fspot,
        photometry_for_plot=photometry_for_plot,
        make_plots=getattr(options, "make_plots", True),
        make_gif_animation=getattr(options, "make_gif_animation", False),
    )


def _run_gaussian_band(
    star,
    band,
    phases,
    region_cube,
    brightness,
    instrument_res,
    save_dir,
    save_spectra,
    plot_flag,
    in_memory,
    write_csv,
    *,
    region_geometry=None,
    region_order=None,
    nrho=100,
    cpu_use=1,
    photometry_for_plot=None,
):
    """Integrate brightness, normalize outer wings and measure first-moment RV.

    Gaussian tests have no region-temperature diagnostics or CCF template.
    Saved RV is in km/s; fspot is the projected area fraction assigned to regions,
    including regions with unit brightness. Surface maps retain spectral-bin sums.
    """
    if len(phases) == 0:
        raise ValueError("Gaussian simulation requires at least one phase.")
    parameters = star.gaussian_parameters[band]
    limb_coeffs = star.LD_coeff[f"{band}_{np.round(star.photosphere_temperature, 4)}"]
    raw_spectra, flux_maps, velocity, wavelength = integrate_gaussian_profiles(
        parameters,
        star.rotation_vel,
        star.inclination_angle,
        limb_coeffs,
        region_cube,
        brightness,
    )
    # Match spectral mode's scaling and instrumental convolution.
    spectra = np.asarray([flux / np.percentile(flux, 99) for flux in raw_spectra])
    if instrument_res > 0:
        spectra = apply_instrumental_broadening(instrument_res, wavelength, spectra)
    normalized = spectral_processing.normalize_gaussian_profiles(
        star, spectra, phases, band, "flux", "gaussian_profile", save_dir,
        getattr(getattr(star, "output_options", None), "make_plots", False)
    )
    rv = diagnostics.calculate_first_moment_rv(
        star,
        normalized,
        band,
        "flux",
        "gaussian_profile",
        velocity_limit=60.0,
        subtract_mean=False,
    )
    disk_pixels = np.count_nonzero(region_cube[0] >= 0)
    fspot = np.count_nonzero(region_cube > 0, axis=(1, 2)) / disk_pixels
    result = {"phase": np.asarray(phases), "first_moment_rv_km_s": rv, "active_filling_f": fspot}
    csv_parameters = (
        f"P{star.prot:.1f}_i{star.inclination_angle:.1f}"
        f"_Tp{star.photosphere_temperature:.0f}_logg{star.logg:.1f}"
    )
    outdir = str(
        Path("outputs") / save_dir
        / output_stem(star, "gaussian_profile", band, details=csv_parameters)
    )
    if not in_memory:
        if save_spectra:
            geometry_metadata = {}
            if region_geometry is not None:
                geometry_metadata = dict(
                    zip(
                        (
                            "region_longitude_deg",
                            "region_latitude_deg",
                            "region_radius_fraction",
                        ),
                        region_geometry,
                    )
                )
                geometry_metadata["region_config_index"] = (
                    np.arange(len(region_geometry[0]))
                    if region_order is None
                    else np.asarray(region_order)
                )
            # Keep the local profile, physical settings and processed spectra together.
            np.savez(
                Path("outputs") / save_dir / "simulated_spotted_spectra"
                / f"{output_stem(star, 'gaussian_profile', band)}.npz",
                **identity_arrays(star),
                phase=np.asarray(phases),
                velocity_kms=velocity,
                wavelength_angstrom=wavelength,
                local_flux=star.gaussian_profiles[band][2],
                raw_phase_spectra=raw_spectra,
                normalized_phase_spectra=normalized,
                surface_flux_bin_sum=flux_maps,
                region_brightness=brightness,
                photosphere_t=star.photosphere_temperature,
                logg=star.logg,
                metallicity=star.metallicity,
                rotation_vel_kms=star.rotation_vel,
                inclination_angle=star.inclination_angle,
                limb_coeffs=np.asarray(limb_coeffs),
                instrument_resolution=instrument_res,
                center_angstrom=parameters.center,
                velocity_half_range_kms=parameters.velocity_half_range,
                velocity_step_kms=parameters.velocity_step,
                doppler_width_kms=parameters.doppler_width,
                line_depth=parameters.line_depth,
                configuration_file=getattr(star, "simulation_config_file", ""),
                **geometry_metadata,
            )
        if write_csv:
            table = Table([phases, rv, fspot], names=("phase", "first_moment_rv_km_s", "active_filling_f"))
            table["phase"].format = "%.6f"
            table["first_moment_rv_km_s"].format = "%.6f"
            table["active_filling_f"].format = "%.6f"
            table.write(f"{outdir}.csv", format="csv", overwrite=True)
        if plot_flag:
            if region_geometry is None:
                raise ValueError("Gaussian visualization requires region geometry.")
            simulation_plot(
                region_cube.shape[1],
                nrho,
                instrument_res,
                star,
                "gaussian_profile",
                save_dir,
                len(brightness) - 1,
                (*region_geometry, None),
                phases,
                normalized,
                rv,
                flux_maps,
                outdir,
                cpu_use,
                np.full(len(phases), np.nan),
                band,
                "flux",
                [],
                None,
                region_cube=region_cube,
                region_brightness=brightness,
                region_fspot=fspot,
                region_order=region_order,
                photometry_for_plot=photometry_for_plot,
            )
    return outdir, result


def _sort_active_regions(star, active_region_box):
    """Order black, cool, hot and neutral regions, keeping their properties paired."""
    # Gaussian-only uses config order; mixed models retain temperature-based order.
    (
        active_region_longitudes,
        active_region_latitudes,
        active_region_radii,
        region_temperatures,
    ) = active_region_box

    n_regions = len(active_region_longitudes)
    if region_temperatures is None:
        # Gaussian-only regions use brightness, so retain their config order.
        region_order = np.arange(n_regions)
    else:
        black_temp = getattr(
            star,
            "black_spot_sentinel",
            None,
        )

        if black_temp is None:
            black_mask = np.zeros(
                region_temperatures.shape,
                dtype=bool,
            )
        else:
            black_mask = np.isclose(
                region_temperatures,
                float(black_temp),
                # Distinguish the float32 marker from physical integer temperatures.
                atol=0.001,
                rtol=0.0,
            )

        # A black sentinel is logically a cool region even though its
        # numerical marker is greater than Tphot.
        # Compare each region with the photosphere; include black regions as cool.
        cool_mask = (region_temperatures < star.photosphere_temperature) | black_mask

        hot_mask = (region_temperatures > star.photosphere_temperature) & (~black_mask)

        # Convert the masks to indices into the original region arrays.
        cool_candidates = np.flatnonzero(cool_mask)
        hot_candidates = np.flatnonzero(hot_mask)

        # Treat black spots as cooler than every physical atmosphere.
        cool_sort_temperature = np.where(
            black_mask[cool_candidates],
            -np.inf,
            region_temperatures[cool_candidates],
        )

        # Black regions first, then physical cool regions from coldest to warmest.
        cool_idx = cool_candidates[np.argsort(cool_sort_temperature)]

        # Hot regions run from hottest to coolest.
        hot_idx = hot_candidates[np.argsort(region_temperatures[hot_candidates])[::-1]]

        # Neutral regions have no temperature contrast and are processed last.
        neutral_idx = np.flatnonzero((region_temperatures == star.photosphere_temperature) & (~black_mask))

        # Store original indices in the final black/cool -> hot -> neutral order.
        region_order = np.concatenate(
            [
                cool_idx,
                hot_idx,
                neutral_idx,
            ]
        )

        if region_order.size != n_regions:
            raise ValueError(
                "Active-region sorting lost one or more regions: "
                f"input={n_regions}, sorted={region_order.size}."
            )

    # Earlier regions take priority where their projected areas overlap.
    # Apply the same order to coordinates, radii and temperatures to keep them paired.
    active_region_longitudes = np.asarray(
        active_region_longitudes, dtype=np.float32
        )[region_order]
    active_region_latitudes = np.asarray(
        active_region_latitudes, dtype=np.float32
        )[region_order]
    active_region_radii = np.asarray(
        active_region_radii, dtype=np.float32
        )[region_order]

    if region_temperatures is not None:
        region_temperatures = np.asarray(
            region_temperatures, dtype=np.float64
            )[region_order]

    active_region_box = (
        active_region_longitudes,
        active_region_latitudes,
        active_region_radii,
        region_temperatures,
    )

    return active_region_box, region_order


def _build_output_parameters(star, active_regions_number, active_region_box):
    """Retain the detailed model parameters in CSV filenames, without equals signs."""
    longitudes, latitudes, radii, temperatures = active_region_box

    def format_array(values):
        if values is None:
            return "none"
        return (
            np.array2string(
                values, separator="-", formatter={"float_kind": lambda x: f"{int(x):d}"}
            )
            .strip("[]")
            .replace(" ", "")
        )

    parameters = (
        f"P{star.prot:.1f}_i{star.inclination_angle:.1f}"
        f"_Tp{star.photosphere_temperature:.0f}_logg{star.logg:.1f}"
    )
    if active_regions_number < 4:
        return (
            f"{parameters}_Ta{format_array(temperatures)}"
            f"_lon{format_array(longitudes)}_lat{format_array(latitudes)}"
            f"_size{format_array(radii**2 * 100.0)}"
        )
    unique_temperatures = None if temperatures is None else np.unique(temperatures)
    return (
        f"{parameters}_SpotNum{active_regions_number:.0f}"
        f"_Ta{format_array(unique_temperatures)}"
    )


def _build_band_result(
    star, band_phases, bolometric_teff_curve, bolometric_luminosity_curve,
    *, ccf_rv_shift=None, ewr_oh_fe=None, filters=(), relative_flux_curves=None,
):
    """Package results with shared physical columns and band-specific diagnostics."""
    result = {"phase": band_phases}
    if relative_flux_curves is not None:
        result["relative_flux"] = {
            filter_name.split("/")[1]: np.asarray(relative_flux_curves[index], dtype=np.float64)
            for index, filter_name in enumerate(filters)
        }
        filling_factors = (star._phase_fspot, star._phase_fcool, star._phase_fhot)
    else:
        result.update(rv_km_s=ccf_rv_shift, ewr_oh_fe=ewr_oh_fe)
        filling_factors = tuple(np.asarray(values) for values in (
            star._phase_fspot, star._phase_fcool, star._phase_fhot,
        ))
    # Copy shared curves so each band's returned values remain independent.
    result.update(
        apparent_teff_K=bolometric_teff_curve.copy(),
        apparent_luminosity_erg_s=bolometric_luminosity_curve.copy(),
        active_filling_f=filling_factors[0].copy(),
        cool_filling_f=filling_factors[1].copy(),
        hot_filling_f=filling_factors[2].copy(),
    )
    return result


def _write_band_csv(
    star, band_phases, bolometric_teff_curve, bolometric_luminosity_curve,
    save_dir, outdir, band, write_csv,
    *, ccf_rv_shift=None, ewr_oh_fe=None, filters=(), relative_flux_curves=None,
):
    """Write physical-band CSV columns at their existing stored precision."""
    tab = Table([band_phases], names=["phase"])
    if relative_flux_curves is not None:
        for index, filter_name in enumerate(filters):
            column = f"{filter_name.split('/')[1]}_relative_flux"
            tab[column] = np.asarray(relative_flux_curves[index], dtype=np.float64)
            tab[column].format = "%.6f"
    else:
        tab["rv_km_s"] = ccf_rv_shift
        tab["ewr_oh_fe"] = ewr_oh_fe
        tab["rv_km_s"].format = "%.4f"
        tab["ewr_oh_fe"].format = "%.5f"

    tab["apparent_teff_K"] = bolometric_teff_curve
    tab["apparent_luminosity_erg_s"] = bolometric_luminosity_curve
    tab["active_filling_f"] = star._phase_fspot
    tab["cool_filling_f"] = star._phase_fcool
    tab["hot_filling_f"] = star._phase_fhot
    tab["phase"].format = "%.3f"
    tab["apparent_teff_K"].format = "%d"
    tab["apparent_luminosity_erg_s"].format = "%.3e"
    for column in ("active_filling_f", "cool_filling_f", "hot_filling_f"):
        tab[column].format = "%.4f"
    if write_csv:
        tab.write(
            f"{outdir}.csv",
            format="csv", overwrite=True,
        )


def _save_spotted_spectra(star, save_dir, band, arrays):
    """Save one archive per band, retaining each processing stage's flux scale."""
    output_arrays = dict(arrays, **identity_arrays(star))
    if star.save_ccf_template_metadata:
        output_arrays["metadata_json"] = np.asarray(json.dumps({
            "band": band,
            "configuration": getattr(star, "simulation_config", None),
            "raw_flux": "disk-integrated quiet and active contributions before scaling and instrumental broadening",
            "processed_flux": "phase percentile-99 scaling followed by instrumental broadening; spectroscopic bands only",
            "normalized_flux": "continuum normalization of processed flux; spectroscopic bands only",
        }, allow_nan=False))
    directory = Path("outputs") / save_dir / "simulated_spotted_spectra"
    directory.mkdir(parents=True, exist_ok=True)
    spectrum_source = star.bands[band][4]
    np.savez(directory / f"{output_stem(star, spectrum_source, band)}.npz", **output_arrays)


def main_model(
    logger,
    bands,
    nrho,
    grid_size,
    star,
    phases,
    band_spectra,
    filters,
    active_regions_number,
    active_region_box,
    spectral_paths,
    instrument_res,
    cpu_use,
    save_dir,
    save_spectra,
    plot_flag,
    spectrum_vel_step,
    silence_pbar,
    in_memory: bool = False,
    write_csv: bool = True,
    *,
    mode: str | None = None,
):
    """
    Run the per-band spotted-star simulations and write outputs.

    Build the surface temperature cube once per call after ordering the active
    regions, and share it across bands. Spectra and surface flux maps remain
    band-specific.

    Args:
        logger (logging.Logger): Logger for status and timing.
        bands (dict): Mapping band_name -> [center, width, mode, spectrum_mode, spectrum_source].
        nrho (int): Circumference resolution.
        grid_size (int): Surface grid resolution (N x N).
        star (Star): Initialized star object with spectra prepared.
        phases (array[float32]): Input rotation phases.
        band_spectra (list[dict[float, PreparedSpectrum] | None]): Per-band temperature lookup;
            Gaussian entries are None.
        filters (list[str]): Photometry filters from SVO (if 'photo' band used).
        active_regions_number (int): Number of active regions.
        active_region_box (tuple): Longitudes, latitudes, radius fractions and
            temperatures; temperatures is None for Gaussian-only models.
        spectral_paths (dict): Library/tool paths keyed by config option name.
        instrument_res (int): Instrumental resolution.
        cpu_use (int): # of logical CPUs to use downstream.
        save_dir (str): Output directory name under ./outputs/.
        save_spectra (int): Save final spectra and CCF templates in simulation mode.
        plot_flag (int): Produce final model plots in simulation mode.
        mode (str or None): simulation, fitting or debug. Fitting returns results
            in RAM without exports; debug enables all existing exports.
            None uses the setup mode, or the legacy in_memory choice.
        silence_pbar (bool): Silence progress bars in inner routines.

    Returns:
        str: Output path when in_memory is False.
        tuple: (output_path, results_by_band) when in_memory is True.
            Result keys match CSV column names. Physical bands include phase,
            apparent_teff_K, apparent_luminosity_erg_s, active_filling_f,
            cool_filling_f and hot_filling_f.
            Spectroscopic bands also include rv_km_s and ewr_oh_fe; disabled
            diagnostics contain NaN. Photometry includes a relative_flux dictionary
            keyed by short filter name. Gaussian bands return phase, first_moment_rv_km_s and
            active_filling_f, without physical temperature or luminosity.
    """

    # Resolve output choices once before calling any band-specific routines.
    configured_options = getattr(star, "output_options", None)
    selected_mode = mode or ("fitting" if in_memory else getattr(configured_options, "mode", "simulation"))
    options = resolve_output_options(
        selected_mode, save_spectra,
        getattr(configured_options, "make_plots", plot_flag),
        getattr(configured_options, "make_gif_animation", False),
    )
    in_memory = options.in_memory
    silence_pbar = silence_pbar or in_memory
    write_csv = options.write_csv if mode is not None or selected_mode != "simulation" else write_csv
    save_spectra = int(options.save_spectra)
    plot_flag = int(options.make_plots or options.make_gif_animation)
    # Continuum checks accompany summary plots; CCF debug plots remain debug-only.
    continuum_plot_flag = int(options.make_plots)
    star.output_options = options
    star.save_ccf_template_metadata = options.save_intermediate

    assign_model_identity(star, active_region_box, grid_size)
    if not in_memory and (write_csv or save_spectra or plot_flag):
        register_model_identity(Path("outputs") / save_dir, star)

    active_region_box, region_order = _sort_active_regions(star, active_region_box)
    original_region_indices = region_order
    (
        active_region_longitudes,
        active_region_latitudes,
        active_region_radii,
        region_temperatures,
    ) = active_region_box

    output_parameters = _build_output_parameters(star, active_regions_number, active_region_box)

    results_by_band = {}  # only filled if in_memory=True
    photometry_for_plot = None

    # --- Surface maps shared across bands ---
    # Build each required map for all phases once per simulation.
    band_names = list(bands.keys())  # freeze order to align with band_spectra
    if any(bands[name][4] != "gaussian_profile" for name in band_names):
        # Physical spectra use pixel temperatures, including the black-region marker.
        surface_t_cube, surface_y, surface_z = stsp.build_surface_temperature_cube(
            star,
            phases,
            grid_size,
            nrho,
            active_regions_number,
            active_region_box,
        )
        if options.save_intermediate:
            # Temperature maps are shared across bands; save them once in debug mode.
            vis.save_surface_temperature_frames(
                surface_t_cube, surface_y, surface_z,
                outdir=Path("outputs") / save_dir / "debug" / "surface_temperature"
                / f"id-{star.model_id}",
                phase_values=phases,
                black_spot_sentinel=getattr(star, "black_spot_sentinel", None),
            )

    if any(bands[name][4] == "gaussian_profile" for name in band_names):
        # Gaussian profiles use region IDs to look up each region's brightness.
        surface_region_cube, _, _ = stsp.build_surface_region_cube(
            star, phases, grid_size, nrho, active_region_box[:3]
        )

    bolometric_results = None
    if any(params[4] != "gaussian_profile" for params in bands.values()):
        # Both diagnostics use the shared maps and configured phases, once per run.
        star._phase_fspot, star._phase_fcool, star._phase_fhot = (
            diagnostics.calculate_surface_filling_factors(
                surface_t_cube, surface_y, surface_z, star.photosphere_temperature,
                black_spot_sentinel=getattr(star, "black_spot_sentinel", None),
            )
        )
        bolometric_results = diagnostics.calculate_bolometric_luminosity_curve(
            star, grid_size, surface_t_cube, phases
        )
        (
            bolometric_luminosity_curve, bolometric_teff_curve,
            mean_apparent_luminosity, mean_apparent_teff,
        ) = bolometric_results
        star._phase_teff = bolometric_teff_curve

    # --- Main loop over bands ---
    for band_index, band in enumerate(band_names):
        start_time = time()

        _, _, _, spectrum_mode, spectrum_source = bands[band]

        if band == "photo":
            logger.info("\n>>> Calculating photometry (photo) <<<")
        else:
            logger.info("\n>>> Calculating band: %s <<<", band)

        if spectrum_source == "gaussian_profile":
            # Index 0 is the photosphere; reorder region brightness to match the map IDs.
            brightness = np.concatenate(
                ([1.0], star.gaussian_brightness[band][1:][region_order])
            )
            outdir, gaussian_result = _run_gaussian_band(
                star,
                band,
                phases,
                surface_region_cube,
                brightness,
                instrument_res,
                save_dir,
                save_spectra,
                plot_flag,
                in_memory,
                write_csv,
                region_geometry=active_region_box[:3],
                region_order=region_order,
                nrho=nrho,
                cpu_use=cpu_use,
                photometry_for_plot=photometry_for_plot,
            )
            if in_memory:
                results_by_band[band] = gaussian_result
            logger.info(
                "Band %s complete: %.2f s",
                band,
                time() - start_time,
            )
            continue

        spectra_by_temperature = band_spectra[band_index]

        outdir = str(
            Path("outputs") / save_dir
            / output_stem(star, spectrum_source, band, details=output_parameters)
        )

        spectral_output = {} if save_spectra else None

        # Integrate local spectra over the visible disk at each rotation phase.
        (
            band_phases,
            integrated_stellar_spectrum,
            quiet_2d_surface_flux_sum,
            final_2d_surface_flux_sum,
        ) = stsp.calculate_activity_signal_spec(
            grid_size,
            star,
            phases,
            spectra_by_temperature,
            instrument_res,
            cpu_use,
            save_dir,
            band,
            save_spectra,
            spectrum_mode.lower(),
            silence_pbar,
            spectral_output=spectral_output,
            surface_t_cube=surface_t_cube,
            y=surface_y,
            z=surface_z,
        )
        # The quiet map is unused; the final flux map is needed only for plots.
        del quiet_2d_surface_flux_sum
        if plot_flag != 1:
            del final_2d_surface_flux_sum
        band_phases = np.asarray(band_phases, dtype=np.float32)

        if band == "photo":
            if save_spectra:
                _save_spotted_spectra(star, save_dir, band, spectral_output)
            # Photometry only
            relative_flux_curves = diagnostics.calculate_relative_photometry(
                integrated_stellar_spectrum,
                star,
                band,
                spectrum_mode,
                spectrum_source,
            )
            if plot_flag == 1:
                # Show the first two filters in combined plots; standalone plots keep all.
                photometry_for_plot = {
                    "phases": band_phases.copy(),
                    "curves": [
                        (name, np.asarray(curve).copy())
                        for name, curve in zip(filters[:2], relative_flux_curves[:2])
                    ],
                }
            ccf_rv_shift = np.full_like(band_phases, np.nan, dtype=np.float32)
            ewr_oh_fe = np.full_like(band_phases, np.nan, dtype=np.float32)

            if in_memory:
                results_by_band[band] = _build_band_result(
                    star, band_phases, bolometric_teff_curve, bolometric_luminosity_curve,
                    filters=filters, relative_flux_curves=relative_flux_curves,
                )
            else:
                _write_band_csv(
                    star, band_phases, bolometric_teff_curve, bolometric_luminosity_curve,
                    save_dir, outdir, band, write_csv,
                    filters=filters, relative_flux_curves=relative_flux_curves,
                )

            if options.make_plots:
                simulation_plot(
                    grid_size,
                    nrho,
                    instrument_res,
                    star,
                    spectrum_source,
                    save_dir,
                    active_regions_number,
                    active_region_box,
                    band_phases,
                    integrated_stellar_spectrum,
                    ccf_rv_shift,
                    final_2d_surface_flux_sum,
                    outdir,
                    cpu_use,
                    ewr_oh_fe,
                    band,
                    spectrum_mode,
                    filters,
                    relative_flux_curves,
                    region_order=region_order,
                )

            # Free big arrays
            del (
                integrated_stellar_spectrum,
                relative_flux_curves,
                ccf_rv_shift,
                ewr_oh_fe,
            )
            if plot_flag == 1:
                del final_2d_surface_flux_sum
            spectral_output = None
            logger.info("Photometry complete: %.2f s", time() - start_time)
            continue

        # Normalize the pseudo-continuum before measuring EWR and CCF RV.
        integrated_stellar_spectrum = spectral_processing.normalize_simulated_spectra(
            star,
            integrated_stellar_spectrum,
            band_phases,
            band,
            spectrum_mode,
            spectrum_source,
            save_dir,
            continuum_plot_flag,
        )
        if save_spectra:
            spectral_output["normalized_processed_spotted_flux"] = integrated_stellar_spectrum
            _save_spotted_spectra(star, save_dir, band, spectral_output)

        # Run enabled diagnostics; keep NaN columns for those not requested.
        selected_diagnostics = getattr(star, "band_diagnostics", {}).get(band, ("rv",))

        if "ewr_FeOH" in selected_diagnostics:
            ewr_oh_fe = diagnostics.calculate_oh_fe_ew_ratio(
                star,
                integrated_stellar_spectrum,
                band_phases,
                band,
                spectrum_mode,
                spectrum_source,
            ).astype(np.float32, copy=False)
        else:
            ewr_oh_fe = np.full_like(band_phases, np.nan, dtype=np.float32)

        if "rv" in selected_diagnostics:
            logger.info("Calculating RV with CCF...")
            ccf_rv_shift = ccf_search.calculate_ccf_rv(
                integrated_stellar_spectrum,
                star,
                band_phases,
                band,
                spectrum_mode,
                spectrum_source,
                save_dir,
                spectrum_vel_step,
                continuum_plot_flag,
                save_spectra,
            )
            ccf_rv_shift = np.asarray(ccf_rv_shift, dtype=np.float32)
        else:
            ccf_rv_shift = np.full_like(band_phases, np.nan, dtype=np.float32)

        if in_memory:
            results_by_band[band] = _build_band_result(
                star, band_phases, bolometric_teff_curve, bolometric_luminosity_curve,
                ccf_rv_shift=ccf_rv_shift, ewr_oh_fe=ewr_oh_fe,
            )
        else:
            _write_band_csv(
                star, band_phases, bolometric_teff_curve, bolometric_luminosity_curve,
                save_dir, outdir, band, write_csv,
                ccf_rv_shift=ccf_rv_shift, ewr_oh_fe=ewr_oh_fe,
            )

            if plot_flag == 1:
                simulation_plot(
                    grid_size,
                    nrho,
                    instrument_res,
                    star,
                    spectrum_source,
                    save_dir,
                    active_regions_number,
                    active_region_box,
                    band_phases,
                    integrated_stellar_spectrum,
                    ccf_rv_shift,
                    final_2d_surface_flux_sum,
                    outdir,
                    cpu_use,
                    ewr_oh_fe,
                    band,
                    spectrum_mode,
                    filters,
                    relative_flux_curves=None,
                    region_order=region_order,
                    photometry_for_plot=photometry_for_plot,
                )

        spectral_output = None
        # Free big arrays
        del (
            integrated_stellar_spectrum,
            ccf_rv_shift,
            ewr_oh_fe,
        )
        if plot_flag == 1:
            del final_2d_surface_flux_sum

        logger.info("Band %s complete: %.2f s", band, time() - start_time)

    if bolometric_results is not None:
        logger.info("\n%s", f"{'Phase Average Values':^33}")
        logger.info("%33s: %.2f %%", "Active-region filling factor", np.mean(star._phase_fspot) * 100)
        logger.info("%33s: %.2f %%", "Cool-region filling factor", np.mean(star._phase_fcool) * 100)
        logger.info("%33s: %.2f %%", "Hot-region filling factor", np.mean(star._phase_fhot) * 100)
        logger.info("%33s: %.3e erg/s", "Apparent luminosity", mean_apparent_luminosity)
        logger.info("%33s: %.1f K", "Apparent effective temperature", mean_apparent_teff)

    else:
        # Gaussian-only coverage includes all region IDs, independent of brightness.
        disk_pixels = np.count_nonzero(surface_region_cube[0] >= 0)
        gaussian_coverage = np.count_nonzero(
            surface_region_cube > 0, axis=(1, 2)
            ) / disk_pixels
        logger.info("Mean active-region filling factor: %.2f %%", np.mean(gaussian_coverage) * 100)

    # Restore original order for downstream'
    restore = np.argsort(original_region_indices)
    active_region_longitudes = active_region_longitudes[restore]
    active_region_latitudes = active_region_latitudes[restore]
    active_region_radii = active_region_radii[restore]
    if region_temperatures is not None:
        region_temperatures = region_temperatures[restore]
    active_region_box = (
        active_region_longitudes,
        active_region_latitudes,
        active_region_radii,
        region_temperatures,
    )

    if in_memory:
        return outdir, results_by_band
    return outdir

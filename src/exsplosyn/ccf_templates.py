"""Read, prepare and export spectral templates for Doppler RV measurement.

configure_ccf_template reads settings and optional file input once during setup.
prepare_ccf_template prepares the reference spectrum for each RV band;
save_ccf_template exports its arrays, physical parameters and model association.
CCF evaluation and RV search settings are handled separately in ccf_search.
"""

from .model_identity import identity_arrays, identity_suffix, output_stem

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from .visualization import PLOT_STYLE
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.interpolate import CubicSpline

from .spectral_processing import fit_template_continuum, apply_instrumental_broadening
from .utilities.rotint import rotint

logger = logging.getLogger(__name__)


def validate_ccf_template(wavelength, flux):
    """Require finite, matching 1D arrays on a positive, increasing wavelength grid."""
    if wavelength.ndim != 1 or flux.ndim != 1 or wavelength.shape != flux.shape:
        raise ValueError(
            "CCF template wavelength_angstrom and flux must be matching 1D arrays."
        )
    if wavelength.size < 4:
        raise ValueError("CCF template must contain at least four wavelength samples.")
    # Non-finite samples would propagate through interpolation and the CCF.
    if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(flux)):
        raise ValueError(
            "CCF template wavelength and flux must contain only finite values."
        )
    # Interpolation requires increasing wavelengths with no duplicate samples.
    if np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0):
        raise ValueError(
            "CCF template wavelengths must be positive and strictly increasing."
        )
    # Mean subtraction removes a constant spectrum, leaving no measurable RV.
    if np.ptp(flux) == 0:
        raise ValueError(
            "CCF template must contain spectral features; constant flux has no RV peak."
        )


def configure_ccf_template(star, cfg, config_file):
    """Read template settings; resolve file paths relative to the configuration file.

    File templates contain wavelength_angstrom (vacuum Angstroms) and normalized
    flux. They are used as supplied, with optional additional broadening. PHOENIX
    templates retain the simulation's rotation and instrumental broadening.
    """
    source = cfg.get("rv_ccf", "template_source", fallback="phx").strip().lower()

    if source not in ("phx", "file"):
        raise ValueError("[rv_ccf] template_source must be 'phx' or 'file'.")

    # Keep the input settings with exported templates so their model can be traced.
    # raw=True preserves configuration strings without interpolating placeholders.
    star.simulation_config = {
        section: dict(cfg.items(section, raw=True)) for section in cfg.sections()
    }
    star.ccf_template_source = source
    if source == "phx":
        # spectral_setup prepares PHOENIX inputs later, separately for each band.
        return

    filename = cfg.get("rv_ccf", "template_file", fallback="").strip()
    if not filename:
        raise ValueError("[rv_ccf] template_file is required for template_source=file.")

    # Resolve relative paths from the config, independent of the working directory.
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = Path(config_file).resolve().parent / path
    if path.suffix.lower() != ".npz":
        raise ValueError("[rv_ccf] template_file must be an NPZ file.")

    # Copy arrays before closing the archive; retain the input for reuse across bands.
    # Metadata is a JSON string, so loading requires no pickled Python objects.
    logger.debug("Reading file RV CCF template: %s", path.resolve())
    with np.load(path, allow_pickle=False) as data:
        if not {"wavelength_angstrom", "flux"}.issubset(data.files):
            raise ValueError(
                "CCF template NPZ must contain wavelength_angstrom and flux."
            )
        wave = np.array(data["wavelength_angstrom"], dtype=np.float64, copy=True)
        flux = np.array(data["flux"], dtype=np.float64, copy=True)
        # Physical metadata is optional; missing values remain unknown on export.
        metadata = {}
        if "metadata_json" in data.files:
            try:
                metadata = json.loads(str(data["metadata_json"].item()))
            except (ValueError, TypeError) as error:
                raise ValueError(
                    "CCF template metadata_json must be a scalar JSON object."
                ) from error
            if not isinstance(metadata, dict):
                raise ValueError(
                    "CCF template metadata_json must contain a JSON object."
                )
            # Accept the exported schema or a simple dictionary of physical parameters.
            physical = metadata.get("template", metadata)
            if not isinstance(physical, dict):
                raise ValueError(
                    "CCF template metadata 'template' must be a JSON object."
                )
            for key in ("temperature_k", "logg", "metallicity_feh"):
                value = physical.get(key)
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                ):
                    raise ValueError(
                        f"CCF template metadata {key} must be a finite number or null."
                    )
    # Keep the original metadata distinct from the current simulation parameters.
    star.ccf_template_input_file = str(path.resolve())
    star.ccf_template_input_metadata = metadata
    validate_ccf_template(wave, flux)
    star.ccf_file_template = (wave, flux)
    # Zero adds no broadening: a supplied disk spectrum may already be broadened.
    for key in ("template_rotation_vel", "template_resolution"):
        value = cfg.getfloat("rv_ccf", key, fallback=0.0)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"[rv_ccf] {key} must be finite and non-negative.")
        setattr(star, f"ccf_{key}", value)


def normalize_ccf_template(wavelength, flux, band, save_dir, plot_template):
    """Normalize a CCF template using a continuum fitted to spectral peaks.

    Divide by the peak-fitted continuum, then rescale to the 99th percentile.
    This operates on the template spectrum, not on the resulting CCF profile.

    Parameters
    ----------
    wavelength : array_like
        Template wavelengths in Angstroms.
    flux : array_like
        Template spectral values, matching wavelength.
    band : str
        Band name used in plot filenames.
    save_dir : str
        Output directory name under outputs.
    plot_template : bool
        Whether to save the normalization plot.

    Returns
    -------
    ndarray
        Continuum-normalized template spectrum.
    """
    # Separate candidate continuum peaks by 4% of the spectrum length.
    # Fit their envelope to remove the continuum trend left after broadening.
    _mpd = int(len(wavelength) * 0.04)
    cont, peaks = fit_template_continuum(
        flux, plot=False, poly_order=2, spline_k=2, mpd=_mpd
    )
    fluxp_flat = flux / cont
    # Set the upper flux level near unity without using a single maximum sample.
    fluxp_flat = fluxp_flat / np.nanpercentile(fluxp_flat, 99)

    if plot_template == 1:
        with plt.rc_context(PLOT_STYLE):
            fig, ax = plt.subplots(1, 1, figsize=(7, 3), facecolor="white", dpi=250)
            ax.plot(wavelength, flux, "-k", lw=1, label="Org. model spec", alpha=0.6)
            ax.plot(
                wavelength, fluxp_flat, "-", c="tab:red", lw=0.8, label="Flatten model spec"
            )
            ax.plot(
                wavelength[peaks],
                flux[peaks],
                "o",
                c="tab:blue",
                ms=4,
                zorder=5,
                label="peaks",
            )
            ax.plot(wavelength, cont, "-", c="tab:orange", lw=1.0)
            ax.axhline(1, ls="-", lw=0.5, color="k", alpha=0.3)
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.legend(loc="lower right", fontsize="x-small", ncol=4)
            ax.set_xlabel("Wavelength [Angstrom]")
            ax.set_ylabel("Flux")
            ax.set_ylim(0.4, 1.05)
            fig.savefig(
                f"./outputs/{save_dir}/continuum_fitting/flatten_{band}_CCF_template.png",
                bbox_inches="tight",
            )
            plt.close(fig)
    return np.array(fluxp_flat)


def prepare_ccf_template(
    star,
    band,
    spectrum_mode,
    spectrum_source,
    wave_obs,
    w_center,
    save_dir,
    plot_template,
    save_template,
):
    """Prepare a disk-integrated spectral template for Doppler RV measurement.

    The template provides the reference line shapes compared with the simulated
    stellar spectrum. PHOENIX templates use flattened local flux spectra, apply
    stellar rotation and instrumental broadening, then normalize the continuum.
    File templates are used as supplied, with only explicitly requested extra
    broadening; no continuum fit is applied to them.

    When needed, resample the template with a cubic spline to the requested
    sampling density (default: five times the simulated spectrum). Keep the
    full template interval so its wavelength margins remain available for Doppler
    shifts. Finer sampling does not increase the physical spectral resolution.

    Parameters
    ----------
    star : Star
        Stellar parameters, prepared spectra and CCF template settings.
    band : str
        Band identifier used for spectral lookup and output filenames.
    spectrum_mode : str
        Simulation spectrum representation: 'flux' or 'intensity'. PHOENIX
        CCF templates use flux spectra for either representation.
    spectrum_source : str
        Simulation spectral source; a file template overrides the default source.
    wave_obs : ndarray
        Increasing wavelength grid of the simulated spectra, in Angstroms.
    w_center : float
        Reference wavelength in Angstroms for converting velocity to wavelength.
    save_dir : str
        Run directory name under outputs.
    plot_template : bool
        Whether to save the PHOENIX continuum-normalization plot.
    save_template : int
        If 1, save the final template arrays in an NPZ file after resampling.
        Include metadata only when star.save_ccf_template_metadata is True.
        Applies to every template source.

    Returns
    -------
    wavelength, flux, velocity : ndarray
        Prepared template arrays. Wavelength is in Angstroms; velocity is in
        km/s relative to w_center. Flux retains the file's supplied scale or
        the built-in template's continuum normalization.
    """
    c_kms = 299792.458

    # Select PHOENIX or an explicitly supplied spectral template.
    if getattr(star, "ccf_template_source", "phx") == "file":
        template_wavelength, template_flux = star.ccf_file_template

        # Work on local copies so preparing one band does not alter the input.
        template_wavelength = template_wavelength.copy()
        template_flux = template_flux.copy()

        # The supplied disk spectrum may already include both broadening effects.
        # Apply additional broadening only when explicitly requested.
        if star.ccf_template_rotation_vel > 0:
            template_wavelength, template_flux = rotint(
                template_wavelength, template_flux, star.ccf_template_rotation_vel
            )
        if star.ccf_template_resolution > 0:
            template_flux = apply_instrumental_broadening(
                star.ccf_template_resolution, template_wavelength, template_flux
            )
    else:
        # PHOENIX supplies the default template even for SYNMAST simulations.
        # The storage key identifies the simulation band; the input is PHOENIX flux.
        template_temperature = star.ccf_temp
        template_velocity = star.spec_flatten[
            f"{band}_{spectrum_source}_{spectrum_mode}_{template_temperature:1.4f}_v"
        ]
        template_flux = star.spec_flatten[
            f"{band}_{spectrum_source}_{spectrum_mode}_{template_temperature:1.4f}_f"
        ]
        template_velocity = np.asarray(template_velocity, dtype=np.float64)
        template_flux = np.asarray(template_flux, dtype=np.float64)

        template_wavelength = template_velocity * w_center / c_kms + w_center

        # Rotational broadening
        template_wavelength, template_flux = rotint(
            template_wavelength, template_flux, star.vsini
        )

        # Apply the instrument profile when a resolving power is specified.
        if star.inst_res > 0:
            template_flux = apply_instrumental_broadening(
                star.inst_res, template_wavelength, template_flux
            )

        # Refit the continuum after broadening and scale its 99th percentile.
        template_flux = normalize_ccf_template(
            template_wavelength,
            template_flux,
            band + identity_suffix(star),
            save_dir,
            plot_template,
        )

    validate_ccf_template(template_wavelength, template_flux)

    # Compare sampling densities only over the simulated wavelength interval.
    valid_temp_mask = (template_wavelength >= wave_obs[0]) & (
        template_wavelength <= wave_obs[-1]
    )
    n_obs = len(wave_obs)
    n_temp_in_range = np.sum(valid_temp_mask)

    if n_temp_in_range == 0:
        raise ValueError(f"CCF template does not overlap band {band!r}.")

    # Denser template sampling limits interpolation error during Doppler shifts.
    # It adds sampling points, not new spectral information or resolving power.
    oversample_factor = getattr(star, "ccf_template_oversample_factor", 5.0)

    # Count template wavelength points within the simulated wavelength interval.
    # For example, if the simulated spectrum has 1,000 points, the target is
    # about 5,000 template points at the default oversample_factor = 5.
    # Interpolate onto a finer grid only when the template has too few points.
    if n_temp_in_range < oversample_factor * n_obs:
        # Increase sampling across the full template, including the shift margins.
        current_temp_size = len(template_wavelength)
        # Scale the total sample count by the density ratio required within the band.
        target_temp_size = int(
            current_temp_size * (oversample_factor * n_obs / max(1, n_temp_in_range))
        )

        # Use a uniform wavelength grid spanning the original template endpoints.
        fine_template_wavelength = np.linspace(
            template_wavelength[0], template_wavelength[-1], target_temp_size
        )

        # Interpolate spectral values onto the finer wavelength grid.
        cs = CubicSpline(template_wavelength, template_flux)
        fine_template_flux = cs(fine_template_wavelength)

        # Use the finer wavelength and flux arrays.
        template_wavelength = fine_template_wavelength
        template_flux = fine_template_flux

    template_velocity = c_kms * (template_wavelength - w_center) / w_center
    logger.debug("RV CCF template for %s: %.6f–%.6f Angstroms (%d samples)",
                 band, template_wavelength[0], template_wavelength[-1], template_wavelength.size)

    # Export after resampling so the saved spectrum is exactly the CCF input.
    # All sources use this same archive format and metadata layout.
    if save_template == 1:
        save_ccf_template(
            star,
            band,
            spectrum_mode,
            spectrum_source,
            save_dir,
            w_center,
            wave_obs,
            template_wavelength,
            template_flux,
            template_velocity,
        )
    return template_wavelength, template_flux, template_velocity


def _finite_parameter(obj, name):
    """Represent an unavailable physical parameter as JSON null."""
    # None becomes JSON null; do not invent a numerical value for unknown physics.
    value = getattr(obj, name, None)
    return float(value) if value is not None and np.isfinite(value) else None


def save_ccf_template(
    star,
    band,
    spectrum_mode,
    spectrum_source,
    save_dir,
    wavelength_center,
    wave_obs,
    wavelength,
    flux,
    velocity,
):
    """Save CCF arrays, adding template and simulation provenance in debug mode.

    File-template physical parameters come from their input metadata, never from
    the simulated star. JSON metadata keeps archives readable with allow_pickle=False.
    Unknown input-file parameters are null.
    """
    is_file = getattr(star, "ccf_template_source", "phx") == "file"
    if is_file:
        input_metadata = star.ccf_template_input_metadata
        physical = input_metadata.get("template", input_metadata)
        temperature = physical.get("temperature_k")
    else:
        temperature = float(star.ccf_temp)

    save_metadata = getattr(star, "save_ccf_template_metadata", False)
    if save_metadata:
        # Identify the template source independently of the simulated spectral library.
        if is_file:
            # Read atmospheric parameters from the supplied template, not the target star.
            # Record extra broadening separately from any broadening already in the input.
            input_metadata = star.ccf_template_input_metadata
            physical = input_metadata.get("template", input_metadata)
            template = {
                "source": "file",
                "temperature_k": physical.get("temperature_k"),
                "logg": physical.get("logg"),
                "metallicity_feh": physical.get("metallicity_feh"),
                "input_file": star.ccf_template_input_file,
                "additional_rotation_vel_kms": star.ccf_template_rotation_vel,
                "additional_instrument_resolution": star.ccf_template_resolution,
                "continuum_normalization": "as_supplied",
            }
        else:
            # Built-in PHOENIX templates use the selected Teff and stellar logg/[Fe/H].
            temperature = float(star.ccf_temp)
            template = {
                "source": "phx",
                "temperature_k": temperature,
                "logg": _finite_parameter(star, "logg"),
                "metallicity_feh": _finite_parameter(star, "metallicity"),
                "applied_rotation_vel_kms": _finite_parameter(star, "vsini"),
                "applied_instrument_resolution": _finite_parameter(star, "inst_res"),
                "continuum_normalization": "peak_fit_percentile_99",
            }
            template["input_file"] = (
                str(getattr(star, "phx_flux_file", {}).get(temperature, "")) or None
            )

        # Separate template physics from the model that used it. The configuration
        # snapshot preserves region settings and other inputs alongside the run label.
        metadata = {
            "schema_version": 1,
            "template": template,
            "simulation": {
                "run_directory": str(save_dir),
                "config_file": getattr(star, "simulation_config_file", None),
                "configuration": getattr(star, "simulation_config", None),
                "band": band,
                "spectrum_source": spectrum_source,
                "spectrum_representation": spectrum_mode,
                "photosphere_temperature_k": _finite_parameter(
                    star, "photosphere_temperature"
                ),
                "logg": _finite_parameter(star, "logg"),
                "metallicity_feh": _finite_parameter(star, "metallicity"),
                "rotation_vel_kms": _finite_parameter(star, "vsini"),
                "instrument_resolution": _finite_parameter(star, "inst_res"),
                "wavelength_center_angstrom": float(wavelength_center),
                "wavelength_min_angstrom": float(wave_obs[0]),
                "wavelength_max_angstrom": float(wave_obs[-1]),
            },
            "wavelength_convention": "vacuum_angstrom",
            "velocity_convention": "km/s relative to simulation wavelength_center_angstrom",
            "sampling": {
                "target_density_ratio": float(
                    getattr(star, "ccf_template_oversample_factor", 5.0)
                ),
                "interpolation": "cubic_spline_if_needed",
            },
        }
        if is_file:
            # Preserve the original source physics and previous model association.
            metadata["input_metadata"] = input_metadata
    # Use the template temperature in the filename; full parameters live in metadata.
    temperature_label = f"T{temperature:g}K" if temperature is not None else "Tunknown"
    directory = Path("outputs") / save_dir / "rv_ccf_templates"
    directory.mkdir(parents=True, exist_ok=True)
    template_source = "file" if is_file else "phx"
    stem = output_stem(
        star, template_source, band, details=f"rv_ccf_template_{temperature_label}"
    )
    path = directory / f"{stem}.npz"
    arrays = dict(
        wavelength_angstrom=wavelength,
        flux=flux,
        velocity_kms=velocity,
        **identity_arrays(star),
    )
    if save_metadata:
        # A scalar JSON string is readable without pickle.
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
    np.savez(path, **arrays)
    return path

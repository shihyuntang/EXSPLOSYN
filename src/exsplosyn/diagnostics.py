"""Measure photometry, line diagnostics and surface properties from simulations."""

import logging
import os

import astropy.constants as const
import numpy as np
from numba import njit
from pyphot import Filter, config

from . import spectral_io

logger = logging.getLogger(__name__)
unit = config.units


def calculate_bolometric_luminosity_curve(star, grid_size, surface_t_cube, phases):
    """Estimate phase-dependent isotropic-equivalent bolometric luminosity and Teff.

    Integrate native PHOENIX intensity spectra over wavelength for each component
    temperature and mu, including the long-wavelength Planck tail added by
    integrate_bolometric_intensity. Select the nearest mu at each surface
    pixel and sum I_bol(mu) times its projected area. The spectra supply the
    angular dependence; no analytic limb-darkening scaling is applied.

    Reuse native bolometric integrals from spectral preparation when available;
    otherwise load and integrate the PHOENIX intensity spectra here.

    Convert the projected intensity sum to isotropic-equivalent luminosity using
    L_iso = 4*pi*R_star**2 * sum(I_bol * projected_pixel_area). Derive Teff from
    L_iso = 4*pi*R_star**2 * sigma_SB * Teff**4. For an inhomogeneous star, these
    are viewing-dependent quantities, not its total emission into all directions.

    Parameters
    ----------
    star : Star
        Component temperatures in Kelvin, radius in km, atmospheric parameters
        and PHOENIX intensity library location.
    grid_size : int
        Number of pixels along each projected surface axis.
    surface_t_cube : ndarray, shape (n_phases, grid_size, grid_size)
        Precomputed temperature maps indexed as [phase, iy, iz].
    phases : array_like, shape (n_phases,)
        Rotation phases; their count sets the number of maps processed.

    Returns
    -------
    apparent_luminosity_curve : ndarray, shape (n_phases,)
        Isotropic-equivalent luminosity in erg/s for each phase.
    apparent_teff_curve : ndarray, shape (n_phases,)
        Apparent effective temperature in Kelvin, derived from luminosity.
    mean_apparent_luminosity : float
        Arithmetic mean of the sampled luminosities, without phase weighting.
    mean_apparent_teff : float
        Arithmetic mean of the sampled temperatures, without phase weighting.

    Notes
    -----
    Pixel coordinates use -1 + index*2/grid_size, matching the supplied
    temperature cube. The disk area is approximated by equal-area pixels.
    """
    logger.info(
        "Calculating apparent luminosity and effective temperature..."
    )

    n_phases = len(phases)
    component_temperatures = np.unique(
        np.concatenate(
            [
                [star.photosphere_temperature],
                star.T_cool,
                star.T_hot,
            ]
        )
    )

    # Integrate each temperature/mu spectrum once, then reuse it across phases.
    bolometric_intensity_lookup = {}
    mu_grid = None
    prepared_intensities = getattr(star, "bolometric_intensities", {})

    for component_temperature in component_temperatures:
        if component_temperature in prepared_intensities:
            component_mu, bolometric_intensity_mu = prepared_intensities[
                component_temperature
            ]
        else:
            # Flux-only and SYNMAST runs still need native PHX bolometric inputs.
            spectrum = spectral_io.load_phx2013_intensity_spectrum(
                component_temperature,
                star.logg,
                star.metallicity,
                0,
                library_path=spectral_io.require_spectral_path(
                    star.spectral_paths, "phoenix_intensity_root"
                ),
            )
            component_mu, bolometric_intensity_mu = calculate_bolometric_intensities(
                spectrum, component_temperature
            )

        if mu_grid is None:
            # Retain the first component's mu grid for the shared lookup.
            # PHX mu varies with temperature; this keeps the existing approximation.
            mu_grid = component_mu

        bolometric_intensity_lookup[np.round(component_temperature, 4)] = (
            bolometric_intensity_mu
        )

    # Projected pixel area is dimensionless here; stellar radius is applied below.
    grid_step = 2.0 / grid_size
    pixel_area = grid_step**2
    # Match the temperature cube and integration kernels' pixel coordinates.
    y = -1.0 + np.arange(grid_size) * grid_step
    z = -1.0 + np.arange(grid_size) * grid_step
    yy, zz = np.meshgrid(y, z, indexing="ij")

    projected_radius_squared = yy**2 + zz**2
    disk_mask = projected_radius_squared <= 1.0
    surface_mu = np.sqrt(np.clip(1.0 - projected_radius_squared, 0.0, 1.0))

    # Each projected pixel has a fixed viewing angle; reuse its nearest mu index.
    nearest_mu_indices = np.argmin(
        np.abs(surface_mu[..., None] - mu_grid[None, None, :]),
        axis=-1,
    )

    apparent_luminosity_curve = np.zeros(n_phases, dtype=np.float64)
    stellar_radius_cm = star.rad * 1e5
    stellar_surface_area = 4.0 * np.pi * (stellar_radius_cm**2)

    # Sum observer-facing intensity over projected pixels for each phase.
    for phase_index in range(n_phases):
        surface_temperatures = np.round(surface_t_cube[phase_index], 4)
        projected_intensity = np.zeros((grid_size, grid_size), dtype=np.float64)
        for component_temperature in component_temperatures:
            temperature_key = np.round(component_temperature, 4)
            mask = (
                np.isclose(surface_temperatures, temperature_key, atol=1e-2) & disk_mask
            )

            if np.any(mask):
                mu_indices = nearest_mu_indices[mask]

                # Retrieve the corresponding integrated specific intensity
                pixel_intensity = bolometric_intensity_lookup[temperature_key][
                    mu_indices
                ]

                # Flux contribution to the observer = I(mu) * projected_pixel_area
                projected_intensity[mask] = pixel_intensity * pixel_area

        # Isotropic equivalent luminosity = 4 * pi * R^2 * Integral(Flux_obs)
        apparent_luminosity_curve[phase_index] = stellar_surface_area * np.sum(
            projected_intensity
        )

    # Infer effective temperature from the isotropic-equivalent luminosity.
    sigma_sb = const.sigma_sb.cgs.value
    apparent_teff_curve = (
        apparent_luminosity_curve / (stellar_surface_area * sigma_sb)
    ) ** 0.25

    # Return arithmetic means over the supplied phase samples.
    if n_phases > 1:
        mean_apparent_luminosity = np.mean(apparent_luminosity_curve)
        mean_apparent_teff = np.mean(apparent_teff_curve)
    else:
        mean_apparent_luminosity = apparent_luminosity_curve[0]
        mean_apparent_teff = apparent_teff_curve[0]

    return (
        apparent_luminosity_curve,
        apparent_teff_curve,
        mean_apparent_luminosity,
        mean_apparent_teff,
    )


def calculate_relative_photometry(
    integrated_stellar_spectrum: np.ndarray,
    star,
    band: str,
    spectrum_mode: str,
    spectrum_source: str,
):
    """Compute relative photometric light curves from disk-integrated spectra.

    Integrate each phase spectrum through the requested filter profiles using
    pyphot and each filter's photon/energy detector convention. Divide band
    fluxes by their mean over the supplied phases.

    Parameters
    ----------
    integrated_stellar_spectrum : ndarray, shape (n_phases, n_spectrum_bins)
        Unnormalized disk-integrated spectra from the intensity simulation.
    star : Star
        Prepared intensity storage, band definitions and ordered filter names.
    band : str
        Photometric band label used to locate its velocity grid.
    spectrum_mode : str
        Storage label, normally 'intensity' for photometry.
    spectrum_source : str
        Library label used in spectral storage keys.

    Returns
    -------
    relative_flux_curves : ndarray, shape (n_filters, n_phases)
        Relative fluxes with sample mean one for each filter.

    Notes
    -----
    Reuse Filter objects prepared during initialization. If absent, read profiles
    from bandpass_cache and retain them on star for repeated calls.
    Missing local profiles raise FileNotFoundError.
    Means weight the supplied phase samples equally, not a full rotation.
    """

    # Reconstruct wavelengths using the preprocessing window midpoint.
    velocity = star.int_spec[
        f"{band}_{spectrum_source}_{spectrum_mode}_{star.photosphere_temperature:1.4f}_v"
    ]  # (n_v,)
    band_center_wavelength, window_half_width = star.bands[band][0], star.bands[band][1]
    wavelength_min = int(band_center_wavelength - window_half_width)
    wavelength_max = int(band_center_wavelength + window_half_width)

    c_kms = 299_792.458
    band_center = (wavelength_min + wavelength_max) * 0.5  # Å (float64)
    wavelength = (velocity * band_center / c_kms + band_center) * unit.U(
        "AA"
    )  # Quantity (n_v,)

    # Parse locally cached filters once per star; this function does not download them.
    if getattr(star, "_pyphot_filters", None) is None:
        star._pyphot_filters = {}
        cache_dir = "./bandpass_cache"

        for filter_name in star.filters:
            cache_filename = filter_name.replace("/", "_") + ".csv"
            cache_path = os.path.join(cache_dir, cache_filename)

            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    f"Cached filter profile {cache_path} not found. "
                    "Ensure the initialization block downloaded it successfully."
                )

            with open(cache_path, "r") as cache_file:
                first_line = cache_file.readline()
                # The header records the detector convention, e.g. "# dtype:photon".
                filter_dtype = first_line.strip().split(":")[1]

            # Read wavelength and transmission columns; loadtxt skips the header.
            data = np.loadtxt(cache_path, delimiter=",")

            filter_wavelength = data[:, 0] * unit.U("AA")
            filter_transmission = data[:, 1]

            star._pyphot_filters[filter_name] = Filter(
                filter_wavelength,
                filter_transmission,
                name=filter_name,
                dtype=filter_dtype,
            )

    photometric_filters = [
        star._pyphot_filters[filter_name] for filter_name in star.filters
    ]

    # Preserve filter order on axis 0 and phase order on axis 1.
    n_phases = integrated_stellar_spectrum.shape[0]
    n_filters = len(photometric_filters)
    relative_flux_curves = np.empty((n_filters, n_phases), dtype=np.float64)

    # Convert and attach units once, shared by all filters.
    phase_spectra = np.asarray(
        integrated_stellar_spectrum, dtype=np.float64, order="C"
    ) * unit.U("erg/s/cm**2/AA")

    for filter_index, (filter_name, photometric_filter) in enumerate(
        zip(star.filters, photometric_filters)
    ):
        logger.debug("Calculating photometric light curve for %s...", filter_name)

        # All phases share one wavelength grid. Integrate along spectral bins.
        band_flux = np.asarray(
            photometric_filter.get_flux(wavelength, phase_spectra, axis=-1).value,
            dtype=np.float64,
        ).reshape(n_phases)

        # Normalize each filter by its mean over the supplied phase samples.
        relative_flux_curves[filter_index] = band_flux / np.nanmean(band_flux)

    return relative_flux_curves


def calculate_first_moment_rv(
    star,
    phase_spectra,
    band,
    spectrum_mode,
    spectrum_source,
    velocity_limit=60.0,
    subtract_mean=False,
):
    """
    Calculate the absorption-profile first-moment RV.

    RV = integral[v * (1 - I(v)) dv]
         --------------------------------
         integral[(1 - I(v)) dv]

    Parameters
    ----------
    star : Star
        EXSPLOSYN star object.
    phase_spectra : array-like, shape (n_phase, n_velocity)
        Continuum-normalized disk-integrated line profiles.
    band, spectrum_mode, spectrum_source : str
        Spectrum identifiers.
    velocity_limit : float or None
        Integrate over +/- velocity_limit km/s. If None, use the
        complete velocity grid.
    subtract_mean : bool
        If True, subtract the mean RV over all phases.

    Returns
    -------
    rv_curve : ndarray, shape (n_phase,)
        First-moment RV in km/s.
    """
    if spectrum_source == "gaussian_profile":
        velocity = star.gaussian_profiles[band][0]
    else:
        storage = star.flux_spec if spectrum_mode == "flux" else star.int_spec
        velocity_key = f"{band}_{spectrum_source}_{spectrum_mode}_{star.photosphere_temperature:1.4f}_v"
        velocity = storage[velocity_key]

    velocity = np.asarray(velocity, dtype=np.float64)
    profiles = np.atleast_2d(np.asarray(phase_spectra, dtype=np.float64))

    if profiles.shape[1] != velocity.size:
        raise ValueError(
            "Profile and velocity-grid sizes do not match: "
            f"{profiles.shape[1]} versus {velocity.size}."
        )

    if velocity_limit is None:
        velocity_mask = np.ones(velocity.size, dtype=bool)
    else:
        velocity_mask = np.abs(velocity) <= float(velocity_limit)

    selected_velocity = velocity[velocity_mask]
    rv_curve = np.full(profiles.shape[0], np.nan, dtype=np.float64)

    for phase_index, profile in enumerate(profiles):
        normalized_flux = profile[velocity_mask]

        # Absorption profile relative to continuum.
        absorption = 1.0 - normalized_flux

        # Tiny negative values may occur from interpolation/convolution.
        # They do not represent physical emission in this model.
        absorption = np.clip(absorption, 0.0, None)

        denominator = np.trapezoid(absorption, selected_velocity)

        if not np.isfinite(denominator) or denominator <= 0.0:
            continue

        numerator = np.trapezoid(selected_velocity * absorption, selected_velocity)
        rv_curve[phase_index] = numerator / denominator

    if subtract_mean:
        rv_curve = rv_curve - np.nanmean(rv_curve)

    return rv_curve


def calculate_oh_fe_ew_ratio(
    star, phase_spectra, phases, band, spectrum_mode, spectrum_source
):
    """Calculate EW(OH) divided by the mean EW of the two Fe lines per phase.

    Equivalent width is computed as:
        EW = ∫ (1 - F_norm) dλ
    with F_norm = F / continuum, where the continuum level is the 95th percentile
    within 1.561–1.565 µm for each phase.

    Return EW(OH) / (0.5 * (EW(Fe1) + EW(Fe2))). Integrate wavelengths in
    microns; wavelength units cancel in the ratio.

    Returns
    -------
    np.ndarray, shape (n_phase,)
        OH/Fe equivalent-width ratio per phase.
    """
    # --- Resolve velocity grid for this band at the photosphere temp ---
    key_v = f"{band}_{spectrum_source}_{spectrum_mode}_{star.photosphere_temperature:1.4f}_v"
    if spectrum_mode == "flux":
        velocity = star.flux_spec[key_v]
    else:
        velocity = star.int_spec[key_v]
    velocity = np.asarray(velocity, dtype=np.float64, order="C")

    # Convert the velocity grid to wavelength around the configured band center.
    band_center = float(star.bands[band][0])  # in Angstrom
    c_kms = 299_792.458
    wavelength_angstrom = band_center + velocity * (band_center / c_kms)  # Å
    wavelength_micron = wavelength_angstrom / 1e4  # µm
    wavelength_micron = np.asarray(wavelength_micron, dtype=np.float64, order="C")

    # --- Masks: normalization region and EW sub-ranges (in µm) ---
    continuum_mask = (wavelength_micron > 1.5610) & (wavelength_micron < 1.5650)

    # Fe, OH, Fe windows (µm)
    ew_windows = [
        (1.56230, 1.56290),  # Fe 1
        (1.56290, 1.56335),  # OH
        (1.56335, 1.56395),  # Fe 2
    ]
    line_masks = [
        (wavelength_micron >= a) & (wavelength_micron <= b) for (a, b) in ew_windows
    ]

    # Reuse the line wavelength arrays across phases.
    fe1_wavelength = wavelength_micron[line_masks[0]]
    oh_wavelength = wavelength_micron[line_masks[1]]
    fe2_wavelength = wavelength_micron[line_masks[2]]

    # --- Loop phases (need per-phase percentile) ---
    n_phases = len(phases)
    ew_fe1 = np.empty(n_phases, dtype=np.float64)
    ew_fe2 = np.empty(n_phases, dtype=np.float64)
    ew_oh = np.empty(n_phases, dtype=np.float64)

    for phase_index in range(n_phases):
        # ensure contiguous float64
        flux = np.asarray(phase_spectra[phase_index], dtype=np.float64, order="C")

        # Continuum estimate (95th percentile) in the normalization region
        continuum_flux = flux[continuum_mask]
        continuum_level = np.nanpercentile(continuum_flux, 95.0)

        if not np.isfinite(continuum_level) or continuum_level <= 0:
            # An invalid continuum leaves all three equivalent widths undefined.
            ew_fe1[phase_index] = ew_fe2[phase_index] = ew_oh[phase_index] = np.nan
            continue

        # Normalize only the three line windows by this phase's continuum level.
        fe1_absorption = 1.0 - flux[line_masks[0]] / continuum_level
        oh_absorption = 1.0 - flux[line_masks[1]] / continuum_level
        fe2_absorption = 1.0 - flux[line_masks[2]] / continuum_level

        # Integrate with trapezoid in µm (units cancel in the ratio)
        # Empty line windows produce NaN equivalent widths.
        ew_fe1[phase_index] = (
            np.trapz(fe1_absorption, fe1_wavelength) if fe1_absorption.size else np.nan
        )
        ew_oh[phase_index] = (
            np.trapz(oh_absorption, oh_wavelength) if oh_absorption.size else np.nan
        )
        ew_fe2[phase_index] = (
            np.trapz(fe2_absorption, fe2_wavelength) if fe2_absorption.size else np.nan
        )

    # Average the two Fe EWs, then form the OH/Fe ratio.
    fe_mean = 0.5 * (ew_fe1 + ew_fe2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ew_ratio = ew_oh / fe_mean
    # Avoid infinities from ~zero Fe mean
    ew_ratio[~np.isfinite(ew_ratio)] = np.nan
    return ew_ratio


@njit(cache=True)
def _isclose_absolute(value, reference, tolerance):
    """Match np.isclose with zero relative tolerance, including nonfinite values."""
    return value == reference or (
        np.isfinite(value)
        and np.isfinite(reference)
        and abs(value - reference) <= tolerance
    )


@njit(cache=True, error_model="numpy")
def _count_surface_pixels(
    surface_t_cube,
    disk,
    photosphere_temperature,
    sentinel_offdisk,
    scale,
    black_spot_sentinel,
):
    """Count valid disk, active, cool and hot pixels without whole-cube masks."""
    counts = np.zeros((4, surface_t_cube.shape[0]), dtype=np.int64)
    for phase_index in range(surface_t_cube.shape[0]):
        for iy in range(surface_t_cube.shape[1]):
            for iz in range(surface_t_cube.shape[2]):
                # Match the original float64 conversion before applying scale.
                temperature = np.float64(surface_t_cube[phase_index, iy, iz]) / scale
                if not disk[iy, iz] or temperature == sentinel_offdisk:
                    continue
                counts[0, phase_index] += 1

                is_black = False
                if black_spot_sentinel is not None:
                    is_black = _isclose_absolute(
                        temperature, black_spot_sentinel, 0.001
                    )
                is_photosphere = (
                    _isclose_absolute(temperature, photosphere_temperature, 0.1)
                    and not is_black
                )
                if not is_photosphere:
                    counts[1, phase_index] += 1
                    # Non-emitting black regions contribute to the cool filling factor.
                    if temperature < photosphere_temperature or is_black:
                        counts[2, phase_index] += 1
                    if temperature > photosphere_temperature and not is_black:
                        counts[3, phase_index] += 1
    return counts


def calculate_surface_filling_factors(
    surface_t_cube,
    y,
    z,
    photosphere_temperature,
    sentinel_offdisk=0,
    scale=1.0,
    black_spot_sentinel=None,
):
    """Return active, cool and hot projected-disk filling factors per phase.

    Count equal-area visible pixels without limb-darkening weights. Black regions
    count as cool; photospheric-temperature pixels are excluded from active regions.
    Return fractions between zero and one, in active/cool/hot order.
    """
    _, grid_size, other_axis = surface_t_cube.shape
    assert grid_size == other_axis, "grid must be square"

    yy, zz = np.meshgrid(y, z, indexing="ij")
    disk = yy * yy + zz * zz <= 1.0
    counts = _count_surface_pixels(
        surface_t_cube,
        disk,
        float(photosphere_temperature),
        sentinel_offdisk,
        float(scale),
        None if black_spot_sentinel is None else float(black_spot_sentinel),
    )
    disk_pixels = counts[0]
    if np.any(disk_pixels == 0):
        bad = np.flatnonzero(disk_pixels == 0)
        raise ValueError(f"No valid on-disk pixels for phases: {bad}")

    return tuple(counts[index] / disk_pixels for index in (1, 2, 3))


def calculate_bolometric_intensities(spectrum, temperature):
    """Return native mu and bolometric intensity at each angle, before band cuts."""
    tail_integral = integrate_planck_tail(spectrum.wavelength, temperature)
    intensities = np.zeros(len(spectrum.mu), dtype=np.float64)
    for mu_index in range(len(spectrum.mu)):
        intensities[mu_index] = integrate_bolometric_intensity(
            spectrum.wavelength,
            spectrum.values[:, mu_index],
            temperature,
            tail_integral=tail_integral,
        )
    # Keep native PHX order; prepared band spectra reverse mu and columns together.
    return spectrum.mu.copy(), intensities


def integrate_bolometric_intensity(
    wavelength_angstrom,
    native_intensity,
    temperature,
    *,
    tail_integral=None,
):
    """Integrate one component's native intensity spectrum at one mu.

    Integrates the synthetic intensity spectrum via the trapezoidal rule
    and appends a numerical Planck function tail for wavelengths beyond
    the template limits (e.g., lambda > 5.5 um).

    Parameters
    ----------
    wavelength_angstrom : numpy.ndarray
        Wavelength array in vacuum Angstroms.
    native_intensity : numpy.ndarray
        Native PHOENIX intensity at one mu, in erg s^-1 cm^-2 sr^-1 cm^-1.
        Convert from per-cm to per-Angstrom intensity before integration.
    temperature : float
        Effective temperature of the specific model region [K].
    tail_integral : float, optional
        Precomputed Planck tail for this temperature and wavelength range.
        If omitted, calculate the tail here.

    Returns
    -------
    I_bol : float
        Total integrated specific intensity in erg s^-1 cm^-2 sr^-1.
    """
    # -------------------------------------------------------------------------
    # PHOENIX native units are erg / s / cm^2 / cm / sr.
    # We must multiply by 1e-8 to convert to erg / s / cm^2 / Angstrom / sr
    # -------------------------------------------------------------------------
    intensity_per_angstrom = native_intensity * 1e-8

    # 1. Integrate the available empirical template range
    spectrum_integral = np.trapezoid(intensity_per_angstrom, wavelength_angstrom)

    if tail_integral is None:
        tail_integral = integrate_planck_tail(wavelength_angstrom, temperature)

    return spectrum_integral + tail_integral


def integrate_planck_tail(wavelength_angstrom, temperature):
    """Integrate B_lambda from the spectrum's longest wavelength to 1 cm.

    Use 2000 logarithmically spaced wavelengths and trapezoidal integration.
    Return specific intensity in erg s^-1 cm^-2 sr^-1, shared by all mu angles.
    """
    max_wavelength_m = np.max(wavelength_angstrom) * 1e-10
    tail_wavelength_m = np.logspace(np.log10(max_wavelength_m), np.log10(1e-2), 2000)

    h = const.h.cgs.value
    c = const.c.cgs.value
    k = const.k_B.cgs.value

    tail_wavelength_cm = tail_wavelength_m * 100.0

    # Planck function B_lambda gives Specific Intensity (erg s^-1 cm^-2 cm^-1 sr^-1)
    planck_intensity_per_cm = (2.0 * h * c**2 / (tail_wavelength_cm**5)) / (
        np.exp((h * c) / (tail_wavelength_cm * k * temperature)) - 1.0
    )

    # Convert to Angstrom grid for consistent integration
    tail_wavelength_angstrom = tail_wavelength_cm * 1e8
    planck_intensity_per_angstrom = (
        planck_intensity_per_cm * 1e-8
    )  # erg s^-1 cm^-2 sr^-1 A^-1

    # Integrate the specific intensity tail
    tail_integral = np.trapezoid(
        planck_intensity_per_angstrom, tail_wavelength_angstrom
    )

    return tail_integral

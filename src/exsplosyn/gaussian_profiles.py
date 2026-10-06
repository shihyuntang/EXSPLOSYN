"""Inputs and local absorption profiles for brightness-based Gaussian tests."""

from dataclasses import dataclass

import numpy as np
from numba import njit

from . import surface_integration

C_KMS = 299792.458


@dataclass(frozen=True)
class GaussianProfileParameters:
    """Line center in Angstroms; velocity quantities in km/s.

    The absorption profile is exp[-(v / doppler_width)**2].
    doppler_width is its 1/e half-width, not its FWHM.
    line_depth is relative to a unit local continuum.
    """

    center: float
    velocity_half_range: float
    velocity_step: float
    doppler_width: float
    line_depth: float

    def __post_init__(self):
        for name in ("center", "velocity_half_range", "velocity_step", "doppler_width"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Gaussian {name} must be finite and positive.")
        if self.velocity_half_range >= C_KMS:
            raise ValueError(
                "Gaussian velocity_half_range must be below the speed of light."
            )
        if not np.isfinite(self.line_depth) or not 0 < self.line_depth < 1:
            raise ValueError("Gaussian line_depth must lie strictly between 0 and 1.")

    @property
    def wavelength_half_width(self):
        """Requested velocity window expressed as a wavelength half-width."""
        return self.center * self.velocity_half_range / C_KMS


def read_gaussian_parameters(cfg, section):
    """Read one [band:name] Gaussian definition without spectral-library paths."""
    return GaussianProfileParameters(
        **{
            name: cfg.getfloat(section, name)
            for name in GaussianProfileParameters.__dataclass_fields__
        }
    )


def read_gaussian_brightness(cfg, section, n_regions):
    """Return brightness by region ID: quiet ID 0 is one, followed by config order.

    Read brightness from the Gaussian band section. Each active region requires
    one finite, nonnegative value. Zero is dark;
    values above one are brighter than the quiet surface. Temperatures do not
    determine brightness. Off-disk ID -1 must be masked before lookup.
    """
    raw = cfg.get(section, "brightness")
    values = np.asarray(raw.replace(",", " ").split(), dtype=np.float64)
    if values.size != n_regions:
        raise ValueError(f"[{section}] brightness requires one value per region.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"[{section}] brightness must be finite and nonnegative.")
    return np.concatenate(([1.0], values))


def build_gaussian_profile(parameters):
    """Return velocity, wavelength and unit-continuum local flux arrays.

    Round the requested velocity extent up to a whole bin to keep a symmetric
    grid containing zero. All regions share this line shape; surface integration
    applies their brightness separately, without temperature-keyed copies.
    """
    n_side = int(np.ceil(parameters.velocity_half_range / parameters.velocity_step))
    velocity = (
        np.arange(-n_side, n_side + 1, dtype=np.float64) * parameters.velocity_step
    )
    wavelength = parameters.center * (1.0 + velocity / C_KMS)
    flux = 1.0 - parameters.line_depth * np.exp(
        -((velocity / parameters.doppler_width) ** 2)
    )
    return velocity, wavelength, flux


# This kernel reuses the same fractional shift and endpoint padding as flux mode.
@njit(cache=True)
def _integrate_brightness_columns(region_cube, brightness, column_spectra, limb_map):
    """Sum brightness-weighted limb factors before accumulating each column."""
    n_phases, grid_size, _ = region_cube.shape
    spectra = np.zeros((n_phases, column_spectra.shape[1]), np.float64)
    flux_maps = np.zeros((n_phases, grid_size, grid_size), np.float64)

    # Column spectra do not change with phase; sum each one only once.
    column_spectral_bin_sums = np.empty(grid_size, np.float64)
    for y_index in range(grid_size):
        column_spectral_bin_sums[y_index] = np.sum(column_spectra[y_index])

    for phase_index in range(n_phases):

        # Pixels at fixed y share the same rotational Doppler shift.
        for y_index in range(grid_size):
            column_total_weight = 0.0
            column_spectral_bin_sum = column_spectral_bin_sums[y_index]

            for z_index in range(grid_size):
                region_id = region_cube[phase_index, y_index, z_index]
                if region_id >= 0:
                    weight = brightness[region_id] * limb_map[y_index, z_index]
                    column_total_weight += weight

                    # Region IDs use [y, z]; displayed maps use [z, y].
                    flux_maps[phase_index, z_index, y_index] = column_spectral_bin_sum * weight
            # Add this column's weighted spectrum to the current phase.
            for velocity_index in range(column_spectra.shape[1]):
                spectra[phase_index, velocity_index] += (
                    column_spectra[y_index, velocity_index] * column_total_weight
                )
    return spectra, flux_maps


def integrate_gaussian_profiles(
    parameters, rotation_vel, inclination_angle, limb_coeffs, region_cube, brightness
):
    """Integrate one shared local Gaussian using region brightness and quiet LDC.

    At fixed y, all pixels share both the local line shape and rotation velocity.
    Shift once per column, then sum brightness times quadratic limb darkening
    over its visible pixels for each phase. Region IDs determine overlap ownership;
    temperatures are not used. Return cropped raw spectra, padded surface bin sums,
    velocity and wavelength grids. Surface maps sum the padded spectral bins
    for brightness display, without multiplying by wavelength or velocity spacing.
    Instrumental broadening is applied downstream.
    """
    velocity, wavelength, local_flux = build_gaussian_profile(parameters)

    if region_cube.ndim != 3 or region_cube.shape[1] != region_cube.shape[2]:
        raise ValueError(
            "Region cube must have shape (n_phases, grid_size, grid_size)."
        )

    grid_size = region_cube.shape[1]
    if np.any(region_cube < -1) or np.any(region_cube >= len(brightness)):
        raise ValueError("Region IDs do not match the brightness lookup.")

    step = parameters.velocity_step
    # Whole-bin padding preserves zero and makes room for all rotational shifts.
    padding_bins = int(np.ceil(max(1.1 * abs(rotation_vel), 20.0) / step))
    n_bins = velocity.size
    n_padded = n_bins + 2 * padding_bins
    column_spectra = np.empty((grid_size, n_padded), np.float64)
    limb_map = np.zeros((grid_size, grid_size), np.float64)
    u1, u2 = limb_coeffs
    sin_inclination_angle = np.sin(np.deg2rad(inclination_angle))

    # Coordinates are in stellar-radius units. Rotation velocity is constant
    # along each column at fixed y, so shift its local spectrum only once.
    for y_index in range(grid_size):
        y = -1.0 + y_index * (2.0 / grid_size)
        local_vel = y * rotation_vel * sin_inclination_angle
        # Split the Doppler shift into whole bins and a remaining fraction.
        # Interpolate the fraction, then place the result at the whole-bin offset.
        integer_shift = int(surface_integration.rndup(local_vel / step, 0))
        shifted = surface_integration.shifting_spectrum(
            velocity, local_flux, local_vel - integer_shift * step, n_bins
        )
        # Extend both ends with their edge flux to avoid gaps after shifting.
        start = padding_bins + integer_shift
        column_spectra[y_index, :start] = shifted[0]
        column_spectra[y_index, start : start + n_bins] = shifted
        column_spectra[y_index, start + n_bins :] = shifted[-1]
        # mu is the cosine of the angle between the surface normal and observer.
        # Apply quadratic limb darkening inside the disk; outside stays zero.
        for z_index in range(grid_size):
            z = -1.0 + z_index * (2.0 / grid_size)
            radius_squared = y * y + z * z
            if radius_squared <= 1.0:
                mu = np.sqrt(1.0 - radius_squared)
                limb_map[y_index, z_index] = (
                    1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
                )
    # Combine each shifted column with the phase-dependent region brightness
    # and the fixed limb-darkening map.
    padded_spectra, flux_maps = _integrate_brightness_columns(
        region_cube, brightness, column_spectra, limb_map
    )
    return (
        padded_spectra[:, padding_bins : padding_bins + n_bins],
        flux_maps,
        velocity,
        wavelength,
    )

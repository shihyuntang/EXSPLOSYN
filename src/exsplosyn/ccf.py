"""Batch Doppler cross-correlation and three-point RV peak refinement."""

import numpy as np
from numba import njit


@njit(cache=True)
def _calculate_doppler_ccfs(
    rv_grid,
    wavelength,
    phase_spectra,
    template_wavelength,
    template_flux,
    c_kms,
    weights,
):
    """Correlate Doppler-shifted template features with each phase spectrum.

    Subtract each spectrum's mean and sum its product with the mean-subtracted
    shifted template. This measures spectral alignment at each trial RV.
    Interpolate the template once per RV and reuse it across phases, keeping
    only the current shifted template in memory. Return (n_phases, n_rvs) CCFs.
    """
    n_phases = phase_spectra.shape[0]
    centered_spectra = np.empty_like(phase_spectra)
    for phase_index in range(n_phases):
        centered_spectra[phase_index] = phase_spectra[phase_index] - np.mean(
            phase_spectra[phase_index]
        )

    ccf_curves = np.zeros((n_phases, rv_grid.size), dtype=np.float64)
    for rv_index in range(rv_grid.size):
        # Stream one RV at a time; do not retain a full shifted-template bank.
        # lambda_shifted = lambda_template * (1 + RV/c); positive RV is a redshift.
        shifted_wavelength = template_wavelength * (1.0 + rv_grid[rv_index] / c_kms)
        shifted_template = np.interp(wavelength, shifted_wavelength, template_flux)
        shifted_template *= weights
        shifted_template -= np.mean(shifted_template)

        for phase_index in range(n_phases):
            # Sum the spectral products in wavelength order for each phase.
            correlation = 0.0
            for wavelength_index in range(wavelength.size):
                correlation += (
                    centered_spectra[phase_index, wavelength_index]
                    * shifted_template[wavelength_index]
                )
            ccf_curves[phase_index, rv_index] = correlation
    return ccf_curves


def calculate_ccf_batch(
    wavelength,
    phase_spectra,
    template_wavelength,
    template_flux,
    rv_min,
    rv_max,
    rv_step,
    edge_pixels=0,
):
    """Compute Doppler CCFs for phases sharing one wavelength grid.

    Shift the template at each trial RV and compare its spectral features with
    all phases using mean-subtracted dot products. Template interpolation is
    shared across phases; calculation runs in a single process with unit weights.

    Args:
        wavelength: Common phase-spectrum wavelength grid in Angstroms.
        phase_spectra: Flux array shaped (n_phases, n_wavelengths).
        template_wavelength, template_flux: Template wavelength and flux arrays.
        rv_min, rv_max, rv_step: Trial RV limits and spacing in km/s;
            np.arange includes rv_min and excludes rv_max.
        edge_pixels: Number of samples removed from each spectrum edge (default 0).

    Returns:
        Float64 CCF array shaped (n_phases, n_rvs), or None if the shifted
        template cannot cover the measurement interval at every trial RV.

    Raises:
        ValueError: Input array sizes or dimensions are invalid, or edge trimming
            removes every spectrum sample.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    phase_spectra = np.asarray(phase_spectra, dtype=np.float64)
    template_wavelength = np.asarray(template_wavelength, dtype=np.float64)
    template_flux = np.asarray(template_flux, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size == 0:
        raise ValueError(
            "Spectrum wavelength must be a nonempty one-dimensional array."
        )
    if (
        template_wavelength.ndim != 1
        or template_flux.ndim != 1
        or template_wavelength.size == 0
        or template_wavelength.size != template_flux.size
    ):
        raise ValueError(
            "Template wavelength and flux must be nonempty one-dimensional arrays "
            "with matching sizes."
        )
    if phase_spectra.ndim != 2 or phase_spectra.shape[1] != wavelength.size:
        raise ValueError("Phase spectra must have shape (n_phases, n_wavelengths).")

    if edge_pixels > 0:
        wavelength = wavelength[edge_pixels:-edge_pixels]
        phase_spectra = phase_spectra[:, edge_pixels:-edge_pixels]
    if wavelength.size == 0:
        raise ValueError("Edge trimming leaves no usable spectrum wavelength samples.")
    # Keep each phase's wavelength samples contiguous after edge slicing.
    phase_spectra = np.ascontiguousarray(phase_spectra)
    weights = np.ones_like(wavelength, dtype=np.float64)
    c_kms = 299792.458
    # The template must cover the measurement interval at both RV limits.
    covered_max = template_wavelength[-1] * (1.0 + rv_min / c_kms)
    covered_min = template_wavelength[0] * (1.0 + rv_max / c_kms)
    if covered_min > wavelength[0] or covered_max < wavelength[-1]:
        return None
    rv_grid = np.arange(rv_min, rv_max, rv_step, dtype=np.float64)
    return _calculate_doppler_ccfs(
        rv_grid,
        wavelength,
        phase_spectra,
        template_wavelength,
        template_flux,
        c_kms,
        weights,
    )


def _refine_ccf_peak(rv_grid, ccf, peak_index):
    """Estimate the CCF peak RV between samples using a three-point parabola.

    Fit the selected local maximum and its two neighbours on an equally spaced
    RV grid. Return the parabola's vertex in the same units as rv_grid. This
    approximates the peak position.
    Return the selected sample's RV if a neighbour is missing or a is zero.
    """
    if peak_index <= 0 or peak_index >= ccf.size - 1:
        return rv_grid[peak_index]
    x0, x1, x2 = rv_grid[peak_index - 1], rv_grid[peak_index], rv_grid[peak_index + 1]
    y0, y1, y2 = ccf[peak_index - 1], ccf[peak_index], ccf[peak_index + 1]
    # Approximate the three CCF samples by y = a*u**2 + b*u + c,
    # where u = -1, 0, +1 at the left, central and right samples.
    a = 0.5 * (y0 - 2 * y1 + y2)
    b = 0.5 * (y2 - y0)
    if a == 0.0:
        return x1
    # The vertex is at u = -b/(2*a), relative to the central sample.
    peak_offset_bins = -b / (2 * a)
    # Keep the estimate within the two neighbouring samples.
    if peak_offset_bins < -1.0:
        peak_offset_bins = -1.0
    if peak_offset_bins > 1.0:
        peak_offset_bins = 1.0
    # Convert the offset to RV using the spacing of the supplied grid.
    rv_step = (x2 - x0) * 0.5
    return x1 + peak_offset_bins * rv_step

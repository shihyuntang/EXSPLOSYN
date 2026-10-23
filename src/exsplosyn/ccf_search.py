"""Adapt EXSPLOSYN spectra, templates and debug output to the rv-ccf package."""

import logging
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from . import ccf_templates
from rv_ccf import (
    measure_rv,
    coarse_to_fine_ccf_rv_search,
    find_common_wavelength_slice,
)

logger = logging.getLogger(__name__)


def _save_ccf_debug_profile(directory, phases, band, *, phase_index, **profile):
    """Save profiles immediately, without retaining fine CCFs across groups."""
    from .visualization import plot_ccf_profile

    plot_ccf_profile(
        directory / f"phase_{phase_index:05d}.png",
        phase=float(phases[phase_index]),
        band=band,
        **profile,
    )


def calculate_ccf_rv(
    phase_spectra,
    star,
    phases,
    band,
    spectrum_mode,
    spectrum_source,
    output_dir,
    spectrum_vel_step,
    plot_template,
    save_template,
):
    """Measure relative RVs from Doppler CCF peaks for one simulation band.

    Shift a reference template across a trial velocity grid and compare its line
    shapes with every phase spectrum. All phases share the wavelength grid, so
    each shifted template is reused across phases in a single-process batch.
    Template input, broadening and normalization are handled by ccf_templates.

    Search every phase on a wide coarse grid, then refine near its peak using
    a shared fine grid for phases with the same coarse peak. The measurement
    wavelength interval is covered by the shifted template throughout the fixed
    search range. Every phase uses the same interval, independent of its RV.
    Retry once at twice the range for an edge or missing coarse peak. After a
    retry, any remaining phase failure raises an error. Without a retry, failed
    phases remain NaN.

    Parameters
    ----------
    phase_spectra : ndarray
        Continuum-normalized disk-integrated spectra with shape
        (n_phases, n_wavelengths).
    star : Star
        Stellar parameters, spectral grids and CCF template settings.
    phases : ndarray
        Rotation phases corresponding to the spectrum rows.
    band : str
        Simulation band identifier.
    spectrum_mode : str
        Local spectrum representation: 'flux' or 'intensity'.
    spectrum_source : str
        Simulation spectral source, used to locate its wavelength grid.
    output_dir : str or Path
        Complete run directory path for template exports and normalization plots.
    spectrum_vel_step : float
        Simulation velocity sampling in km/s; sets the base CCF RV step.
    plot_template : bool
        Whether to save PHOENIX template-normalization plots.
    save_template : int
        If 1, export the prepared CCF template; debug mode also includes metadata.

    Returns
    -------
    ndarray
        Relative RVs in km/s, in phase order. Without a retry, unmeasurable
        peaks are NaN; remaining failures after a retry raise ValueError.

    Notes
    -----
    Subtract the mean of the finite phase RVs to set the relative RV zero point.
    This is an arithmetic phase-sample mean.
    """
    phases = np.asarray(phases, dtype=np.float64)
    phase_spectra = np.asarray(phase_spectra, dtype=np.float64)
    n_phases = phases.size
    if (
        phases.ndim != 1
        or phase_spectra.ndim != 2
        or phase_spectra.shape[0] != n_phases
    ):
        raise ValueError("Spectra rows must match the one-dimensional phase array.")

    storage = star.flux_spec if spectrum_mode == "flux" else star.int_spec
    velocity_key = f"{band}_{spectrum_source}_{spectrum_mode}_{star.photosphere_temperature:1.4f}_v"
    velocity = storage[velocity_key]
    velocity = np.asarray(velocity, dtype=np.float64)

    # Reconstruct wavelengths from the stored velocity bins and band limits.
    band_center, window_half_width = star.bands[band][0], star.bands[band][1]
    wavelength_min = int(band_center - window_half_width)
    wavelength_max = int(band_center + window_half_width)
    c_kms = 299792.458
    wavelength_center = 0.5 * (wavelength_min + wavelength_max)
    wavelength = velocity * wavelength_center / c_kms + wavelength_center  # Å

    template_wavelength, template_flux, _ = ccf_templates.prepare_ccf_template(
        star,
        band,
        spectrum_mode,
        spectrum_source,
        wavelength,
        wavelength_center,
        output_dir,
        plot_template,
        save_template,
    )

    # Preserve the exact simulation sampling; other callers may infer it.
    debug_callback = None
    if getattr(getattr(star, "output_options", None), "save_intermediate", False):
        directory = (
            Path(output_dir)
            / "debug"
            / "ccf_profiles"
            / f"id-{getattr(star, 'model_id', 'model')}"
            / band
        )

        def debug_callback(*, attempt_index, **profile):
            _save_ccf_debug_profile(
                directory / f"attempt_{attempt_index + 1:02d}",
                phases,
                band,
                **profile,
            )

    result = measure_rv(
        wavelength,
        phase_spectra,
        template_wavelength,
        template_flux,
        spectrum_vel_step=spectrum_vel_step,
        ccf_rv_oversample_factor=getattr(star, "ccf_rv_oversample_factor", 50.0),
        rv_half_range=getattr(star, "ccf_rv_half_range", 40.0),
        n_retries=1,
        debug_callback=debug_callback,
    )
    rv_curve, report = result.rv, result.report

    start, stop = report["wavelength_start"], report["wavelength_stop"]
    logger.debug(
        "CCF measurement interval for %s: %.6f–%.6f Angstroms (%d samples)",
        band,
        wavelength[start],
        wavelength[stop - 1],
        stop - start,
    )

    for phase_index, reason in enumerate(report["failures"]):
        if reason:
            logger.warning(
                "CCF peak unavailable for %s at phase %.3f: %s",
                band,
                phases[phase_index],
                reason,
            )

    finite_phases = np.isfinite(rv_curve)
    if np.any(finite_phases):
        rv_curve = rv_curve - np.mean(rv_curve[finite_phases])
    return rv_curve.astype(np.float64, copy=False)

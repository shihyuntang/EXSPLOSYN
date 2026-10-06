"""Search CCF peaks on coarse and fine RV grids; report failures per phase."""

import logging
from functools import partial
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from . import ccf_templates
from .ccf import calculate_ccf_batch, _refine_ccf_peak

logger = logging.getLogger(__name__)


def find_common_wavelength_slice(wavelength, template_wavelength, rv_min, rv_max):
    """Select mock-spectrum wavelengths covered by the template at every trial RV.

    Doppler shifting changes the template's wavelength range. Trim the mock
    spectrum where needed so every trial from rv_min to rv_max can be evaluated
    without extrapolating the template flux. Use this same interval for all
    phases when measuring their relative RVs.

    Inputs are increasing wavelength arrays in Angstroms and RV limits in km/s.

    Returns:
        (start, stop): Indices selecting wavelength[start:stop]. The start sample is
        included and the stop sample is excluded. Apply the same slice to each
        phase spectrum.
    """
    c_kms = 299792.458
    # At the largest redshift the template's blue edge moves furthest right;
    # at the largest blueshift its red edge moves furthest left.
    covered_min = template_wavelength[0] * (1 + rv_max / c_kms)
    covered_max = template_wavelength[-1] * (1 + rv_min / c_kms)
    start = int(np.searchsorted(wavelength, covered_min, side="left"))
    stop = int(np.searchsorted(wavelength, covered_max, side="right"))
    if stop - start < 4:
        raise ValueError(
            "Template coverage leaves fewer than four usable wavelength points."
        )
    return start, stop


def coarse_to_fine_ccf_rv_search(
    wavelength,
    phase_spectra,
    template_wavelength,
    template_flux,
    *,
    half_range=40.0,
    coarse_step=1.0,
    fine_step=0.005,
    max_half_range=None,
    debug_callback=None,
):
    """Measure template-relative RVs using wide coarse and local fine searches.

    Doppler-shift the template and correlate it with every phase spectrum.
    First locate the most prominent peak on a coarse grid. Then search near
    that peak on a fine grid, select the local peak closest to the coarse peak,
    and refine its maximum with a three-point parabola.
    Phases sharing a coarse peak are batched so each shifted template is reused.

    Args:
        wavelength: Common spectrum wavelength grid in Angstroms.
        phase_spectra: Normalized flux array, shape (n_phases, n_wavelengths).
        template_wavelength, template_flux: Template wavelength and flux arrays.
        half_range: Symmetric RV search limit in km/s.
        coarse_step, fine_step: Trial RV spacing in km/s.
        max_half_range: Optional expansion limit for experiments. By default,
            the search range stays fixed; edge peaks are reported as failures.
        debug_callback: Optional consumer of the already calculated coarse/fine
            profiles, called once per phase after the selected search attempt.

    Returns:
        Raw RVs in km/s, in input phase order, and a diagnostic dictionary with
        wavelength slice bounds, search attempts and per-phase failures.
        Zero RV is valid; failed phases remain NaN. No phase mean is subtracted.
    """
    if max_half_range is None:
        max_half_range = half_range

    if not (0 < fine_step < coarse_step < half_range <= max_half_range):
        raise ValueError(
            "Require 0 < fine_step < coarse_step < half_range <= max_half_range."
        )
    wavelength = np.asarray(wavelength, dtype=np.float64)
    phase_spectra = np.asarray(phase_spectra, dtype=np.float64)
    template_wavelength = np.asarray(template_wavelength, dtype=np.float64)
    template_flux = np.asarray(template_flux, dtype=np.float64)

    if phase_spectra.ndim != 2 or phase_spectra.shape[1] != wavelength.size:
        raise ValueError("Spectra must have shape (n_phases, n_wavelengths).")
    if (
        not np.all(np.isfinite(phase_spectra))
        or not np.all(np.isfinite(template_flux))
        or not np.all(np.diff(wavelength) > 0)
        or not np.all(np.diff(template_wavelength) > 0)
    ):
        raise ValueError(
            "Search requires finite spectra and increasing wavelength grids."
        )
    n_phases = len(phase_spectra)
    if n_phases == 0:
        return np.empty(0), {"attempts": [], "failures": []}

    report = {"attempts": []}
    while True:
        start, stop = find_common_wavelength_slice(
            wavelength, template_wavelength, -half_range, half_range
        )
        selected_wavelength = wavelength[start:stop]
        selected_spectra = phase_spectra[:, start:stop]

        coarse_grid = np.arange(-half_range, half_range, coarse_step, dtype=np.float64)
        coarse_ccfs = calculate_ccf_batch(
            selected_wavelength,
            selected_spectra,
            template_wavelength,
            template_flux,
            -half_range,
            half_range,
            coarse_step,
        )
        if coarse_ccfs is None:
            raise ValueError(
                "Template coverage check failed after common-interval selection."
            )
        # Initialize every phase's peak index to -1 (no valid peak found yet).
        # Replace it with the coarse-grid index when a valid peak is found.
        coarse_peak_indices = np.full(n_phases, -1, dtype=int)
        # Start with no failure reason for each phase. Failed searches record
        # a reason below; successful searches leave the string empty.
        failure_reasons = [""] * n_phases

        for phase_index, curve in enumerate(coarse_ccfs):
            span = np.ptp(curve)
            if not np.isfinite(span) or span == 0:
                failure_reasons[phase_index] = "flat_ccf"
                continue
            # An edge maximum cannot be accepted as a measured RV.
            if np.argmax(curve) in (0, len(curve) - 1):
                failure_reasons[phase_index] = "coarse_edge"
                continue
            # Normalize only for prominence selection; refinement uses raw CCFs.
            candidates, properties = find_peaks(
                (curve - curve.min()) / span,
                prominence=0.1,
            )
            if len(candidates):
                coarse_peak_indices[phase_index] = candidates[
                    np.argmax(properties["prominences"])
                ]
            else:
                failure_reasons[phase_index] = "no_prominent_peak"
        report["attempts"].append(
            {
                "half_range_kms": half_range,
                "coarse_rv_points": len(coarse_grid),
                "failed_phases": int(np.sum(coarse_peak_indices < 0)),
            }
        )
        # A wider search may reveal an edge or missing peak, but cannot fix a
        # flat CCF. Retry all phases on the newly covered wavelength interval.
        if (
            any(
                reason in ("coarse_edge", "no_prominent_peak")
                for reason in failure_reasons
            )
            and half_range < max_half_range
        ):
            half_range = min(half_range * 2, max_half_range)
            continue
        break

    rv_curve = np.full(n_phases, np.nan)

    def export_profile(phase_index, fine_grid=None, fine_ccf=None):
        # Consume the current profiles without saving additional CCF arrays.
        coarse_index = coarse_peak_indices[phase_index]
        debug_callback(
            phase_index=int(phase_index),
            coarse_grid=coarse_grid,
            coarse_ccf=coarse_ccfs[phase_index],
            coarse_peak_rv=coarse_grid[coarse_index] if coarse_index >= 0 else None,
            fine_grid=fine_grid,
            fine_ccf=fine_ccf,
            raw_rv=rv_curve[phase_index],
            failure_reason=failure_reasons[phase_index],
        )

    if debug_callback is not None:
        for phase_index in np.flatnonzero(coarse_peak_indices < 0):
            export_profile(phase_index)
    fine_groups = []
    # Group by coarse peak location, not phase order. This is serial batching,
    # not process parallelism; each group shares the same shifted templates
    for coarse_index in np.unique(coarse_peak_indices[coarse_peak_indices >= 0]):
        phase_indices = np.flatnonzero(coarse_peak_indices == coarse_index)
        # Search between the neighbouring coarse samples. np.arange excludes
        # its stop, so add one fine step to include the right neighbour when
        # possible. Only interior local peaks can be refined below.
        rv_min = max(-half_range, coarse_grid[coarse_index] - coarse_step)
        rv_max = min(
            half_range,
            coarse_grid[coarse_index] + coarse_step + fine_step,
        )
        fine_grid = np.arange(rv_min, rv_max, fine_step, dtype=np.float64)
        fine_ccfs = calculate_ccf_batch(
            selected_wavelength,
            selected_spectra[phase_indices],
            template_wavelength,
            template_flux,
            rv_min,
            rv_max,
            fine_step,
        )
        if fine_ccfs is None:
            raise ValueError("Fine search exceeds the selected template coverage.")

        # Follow the peak selected by coarse prominence: choose the local peak
        # closest to its coarse RV, rather than a higher neighbouring peak.
        # Refine that peak using its two adjacent samples.
        for phase_index, curve in zip(phase_indices, fine_ccfs):
            candidates, _ = find_peaks(curve)
            if not len(candidates):
                failure_reasons[phase_index] = (
                    "fine_edge"
                    if np.argmax(curve) in (0, len(curve) - 1)
                    else "no_fine_peak"
                )
                if debug_callback is not None:
                    export_profile(phase_index, fine_grid, curve)
                continue
            # Measure each local peak's RV distance from the selected coarse peak.
            distances = np.abs(fine_grid[candidates] - coarse_grid[coarse_index])
            # Follow the nearest peak, even if another peak is higher.
            peak_index = candidates[np.argmin(distances)]
            # Fit this peak and its two neighbours to refine the RV between samples.
            rv_curve[phase_index] = _refine_ccf_peak(fine_grid, curve, peak_index)
            if debug_callback is not None:
                export_profile(phase_index, fine_grid, curve)
        fine_groups.append(
            {
                "center_kms": float(coarse_grid[coarse_index]),
                "phases": len(phase_indices),
                "fine_rv_points": len(fine_grid),
            }
        )
    report.update(
        wavelength_start=start,
        wavelength_stop=stop,
        usable_wavelength_points=stop - start,
        fine_groups=fine_groups,
        failures=failure_reasons,
        half_range_kms=half_range,
    )
    return rv_curve, report


def _save_ccf_debug_profile(directory, phases, band, *, phase_index, **profile):
    """Save profiles immediately, without retaining fine CCFs across groups."""
    from .visualization import plot_ccf_profile

    plot_ccf_profile(
        directory / f"phase_{phase_index:05d}.png",
        phase=float(phases[phase_index]), band=band, **profile,
    )


def calculate_ccf_rv(
    phase_spectra,
    star,
    phases,
    band,
    spectrum_mode,
    spectrum_source,
    save_dir,
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
    save_dir : str
        Run directory under outputs for template exports and normalization plots.
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
        save_dir,
        plot_template,
        save_template,
    )

    # Fine CCF trial spacing = simulation velocity-bin width / this factor.
    # For example, a 0.306 km/s spectrum bin and factor 50 give trials spaced
    # by 0.00612 km/s (6.12 m/s). A larger factor gives a finer RV search grid.
    rv_oversample_factor = getattr(star, "ccf_rv_oversample_factor", 50.0)
    fine_step = spectrum_vel_step / rv_oversample_factor

    half_range = getattr(star, "ccf_rv_half_range", 40.0)

    # Trim the mock spectra at both ends according to the RV search range,
    # so the shifted template covers the measurement interval at every trial RV.
    # All phases and both coarse and fine searches use this same interval.
    # If the range is doubled for a retry, recalculate the interval for all phases.
    for attempt in range(2):
        logger.debug("CCF search for %s, attempt %d: +/-%.3f km/s; "
                     "coarse step=1 km/s, fine step=%.6f km/s",
                     band, attempt + 1, half_range, fine_step)
        debug_options = {}
        if getattr(getattr(star, "output_options", None), "save_intermediate", False):
            directory = (
                Path("outputs") / save_dir / "debug" / "ccf_profiles"
                / f"id-{getattr(star, 'model_id', 'model')}" / band
                / f"attempt_{attempt + 1:02d}"
            )
            debug_options["debug_callback"] = partial(
                _save_ccf_debug_profile, directory, phases, band,
            )
        rv_curve, report = coarse_to_fine_ccf_rv_search(
            wavelength,
            phase_spectra,
            template_wavelength,
            template_flux,
            half_range=half_range,
            coarse_step=1.0,
            fine_step=fine_step,
            max_half_range=half_range,
            **debug_options,
        )
        failures = report["failures"]
        if attempt == 1:
            if any(failures):
                raise ValueError(
                    f"CCF peak search failed for {band!r} after retrying at +/-{half_range:g} km/s. "
                    "Check rv_half_range, the spectra and the template. "
                    f"Phase failure reasons: {failures}"
                )
            break
        if "coarse_edge" not in failures and "no_prominent_peak" not in failures:
            break

        # If a peak is at the search boundary or no prominent peak is found,
        # double the RV range and try again once. The wider range requires more
        # trimming at the spectrum edges, so rerun every phase to measure all RVs
        # over the same wavelength interval.
        logger.warning(
            "CCF edge or missing peak for %s within +/-%.1f km/s; retrying all phases at +/-%.1f km/s.",
            band,
            half_range,
            half_range * 2,
        )
        half_range *= 2

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

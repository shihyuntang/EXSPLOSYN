"""Instrumental broadening, continuum fitting and spectrum normalization."""

from .model_identity import identity_suffix

import matplotlib.pyplot as plt

from .visualization import PLOT_STYLE
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy import interpolate
from scipy.special import expit

from .utilities.macbro_dynamic import macbro_dyn


def apply_instrumental_broadening(instrument_res, wavelength, spectra):
    """Apply a Gaussian instrument profile to one spectrum or all phases.

    Resolving power sets the profile FWHM to wavelength / R. Each spectrum
    uses macbro_dyn's wavelength-dependent convolution, with endpoint padding
    and input sample-sum preservation. Line depths decrease as lines broaden;
    no original-depth rescaling is applied.

    Args:
        instrument_res: Positive resolving power, R = wavelength / FWHM.
        wavelength: Shared wavelength grid in Angstroms.
        spectra: One spectrum of shape (n_samples,), or phase spectra of shape
            (n_phases, n_samples).

    Returns:
        Broadened spectra with the same shape. The input is not modified.
    """
    spectra = np.asarray(spectra)
    if spectra.ndim == 1:
        return macbro_dyn(wavelength, spectra, R=instrument_res)
    if spectra.ndim != 2:
        raise ValueError("Instrumental broadening requires 1D or 2D spectra.")

    broadened_spectra = np.empty_like(spectra)
    for phase_index in range(len(spectra)):
        broadened_spectra[phase_index] = macbro_dyn(
            wavelength, spectra[phase_index], R=instrument_res
        )
    return broadened_spectra


def detect_peaks(
    x,
    mph=None,
    mpd=1,
    threshold=0,
    edge="rising",
    kpsh=False,
    valley=False,
    show=False,
    ax=None,
):
    """
    Detect peaks in a 1D array.

    Args:
        x (array): 1D data array.
        mph (float, optional): Minimum peak height.
        mpd (int, optional): Minimum peak distance.
        threshold (float, optional): Minimum threshold between peak and neighbors.
        edge (str, optional): 'rising', 'falling', 'both', or None.
        kpsh (bool, optional): Keep peaks with same height even if closer than mpd.
        valley (bool, optional): If True, detect valleys (local minima) instead of peaks.
        show (bool, optional): If True, plot the result.
        ax (matplotlib axis, optional): Axis to plot on.

    Returns:
        ind (np.ndarray): Indices of detected peaks.
    """
    y = np.asarray(x, dtype=np.float64)
    n = y.size
    if n < 3:
        return np.empty(0, dtype=int)

    if valley:
        y = -y

    # Finite mask & convenient view that ignores NaNs
    finite = np.isfinite(y)
    if not finite.all():
        # Replace NaNs with -inf so they cannot be peaks,
        # and we can compute differences without special-casing
        y = y.copy()
        y[~finite] = -np.inf

    # First/last cannot be peaks
    dy = y[1:] - y[:-1]

    # Candidate edges
    # Rising edge: dy switches + or 0 right before, and - after
    # Falling edge: dy switches - before, and + or 0 after
    # (We emulate your <= / >= conventions)
    left = np.hstack((dy, 0.0))
    right = np.hstack((0.0, dy))

    cand = np.empty(0, dtype=int)
    if edge is None:
        # strictly interior local maxima
        cand = np.flatnonzero((left < 0.0) & (right > 0.0))
    else:
        e = edge.lower()
        idx = []
        if e in ("rising", "both"):
            idx.append(np.flatnonzero((left <= 0.0) & (right > 0.0)))
        if e in ("falling", "both"):
            idx.append(np.flatnonzero((left < 0.0) & (right >= 0.0)))
        if idx:
            cand = np.unique(np.concatenate(idx))  # sorted union

    if cand.size == 0:
        return cand

    # Remove boundary points (already implied by construction, but keep explicit)
    cand = cand[(cand > 0) & (cand < n - 1)]

    # Drop anything touching NaN neighbors
    if not finite.all():
        bad = ~(finite[cand] & finite[cand - 1] & finite[cand + 1])
        if bad.any():
            cand = cand[~bad]
            if cand.size == 0:
                return cand

    # Minimum peak height
    if mph is not None:
        cand = cand[y[cand] >= mph]
        if cand.size == 0:
            return cand

    # Threshold vs neighbors
    if threshold > 0.0:
        left_diff = y[cand] - y[cand - 1]
        right_diff = y[cand] - y[cand + 1]
        mindiff = np.minimum(left_diff, right_diff)
        cand = cand[mindiff >= threshold]
        if cand.size == 0:
            return cand

    # Minimum peak distance (greedy keep largest first)
    if mpd > 1:
        order = np.argsort(y[cand])[::-1]  # tallest → shortest
        cand_sorted = cand[order]
        keep = np.ones(cand_sorted.size, dtype=bool)

        # O(k^2) worst-case, but k << n typically; simple & robust
        for i in range(cand_sorted.size):
            if not keep[i]:
                continue
            lo = cand_sorted[i] - mpd
            hi = cand_sorted[i] + mpd
            # If kpsh=False, remove all within window.
            # If kpsh=True, keep equal-height ties.
            too_close = (cand_sorted >= lo) & (cand_sorted <= hi)
            if kpsh:
                # do not delete those with the exact same height
                same_height = y[cand_sorted] == y[cand_sorted[i]]
                too_close &= ~same_height
            keep |= False  # no-op to keep type checker calm
            keep[too_close] = False
            keep[i] = True  # keep the current best

        cand = np.sort(cand_sorted[keep])

    # Optional plotting hook preserved (no-op here)
    if show and ax is not None:
        pass
    return cand


def fit_spectral_continuum(
    spectrum, *, poly_order=2, spline_k=2, mpd=100, transition_mad=0.25
):
    """Fit a pseudo-continuum without hard residual-based anchor rejection.

    Detect candidate peaks and estimate their polynomial/MAD residuals.
    Replace the -2/+6 MAD cutoffs by continuous influence weights. The
    transition width is expressed in MAD units, not wavelength or flux units.
    Coefficients are estimated independently for each spectrum: phases are
    neither averaged nor smoothed together.

    Returns continuum, candidate peak indices and influence weights. SciPy's
    spline squares its supplied weights, so pass sqrt(influence) to give the
    intended weighted least-squares contribution. This is a
    pseudo-continuum fit; it does not establish a physical continuum or RV accuracy.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if (
        spectrum.ndim != 1
        or spectrum.size <= max(poly_order, spline_k)
        or not np.all(np.isfinite(spectrum))
    ):
        raise ValueError(
            "Expected a finite one-dimensional spectrum with sufficient samples."
        )

    if not np.isfinite(transition_mad) or transition_mad <= 0:
        raise ValueError("transition_mad must be finite and positive.")

    x = np.arange(spectrum.size)
    minimum = max(poly_order + 1, spline_k + 1, 3)

    # Local maxima trace the upper envelope, which may be a pseudo-continuum
    # rather than the physical continuum in spectra with dense absorption lines.
    peaks = detect_peaks(spectrum, mpd=mpd)
    if peaks.size < minimum:
        # Relax peak spacing when too few anchors remain for either fit.
        peaks = detect_peaks(spectrum, mpd=max(1, mpd // 2))
    if peaks.size < minimum:
        # Featureless spectra still need enough samples to define a spline.
        peaks = np.linspace(0, spectrum.size - 1, minimum).astype(int)

    # An initial unweighted trend measures how far each anchor lies above or
    # below the envelope. MAD sets the residual scale without assuming noise errors.
    polynomial = np.poly1d(np.polyfit(x[peaks], spectrum[peaks], poly_order))

    residual = spectrum[peaks] - polynomial(x[peaks])
    mad = np.median(np.abs(residual - np.median(residual)))

    # Spectra are scaled near unity; smaller values are treated as numerical zero.
    zero_tolerance = 1e-14
    if mad <= zero_tolerance:
        # Avoid dividing by a vanishing residual scale, e.g. for a flat spectrum.
        influence = np.ones(peaks.size)
    else:
        score = residual / mad
        # Replace hard rejection below -2 or above +6 MAD with sigmoid weights.
        # At either threshold the influence is approximately 0.5; transition_mad
        # controls how gradually it changes. No neighbouring phases are used.
        # expit(x) = 1 / (1 + exp(-x)): a smooth transition from 0 to 1.
        lower_weight = expit((score + 2) / transition_mad)  # Downweight below -2 MAD.
        upper_weight = expit((6 - score) / transition_mad)  # Downweight above +6 MAD.
        # Anchors must satisfy both bounds to receive nearly full influence.
        influence = lower_weight * upper_weight

    median = np.median(spectrum)
    if median > zero_tolerance:
        # Preserve the original 0.1-of-median low-flux criterion as a smooth
        # transition too; this has negligible effect on near-continuum peaks.
        influence *= expit((spectrum[peaks] / median - 0.1) / 0.01)

    # Keep spline weights positive even for anchors with negligible influence.
    influence = np.maximum(influence, 1e-8)
    smoothing, hits = 1e6, 0
    for _ in range(10):
        # SciPy minimizes sum((w * residual)**2): sqrt(influence) gives each
        # anchor its intended contribution, rather than squaring it again.
        continuum = interpolate.UnivariateSpline(
            x[peaks], spectrum[peaks], w=np.sqrt(influence), k=spline_k, s=smoothing
        )(x)
        # Increase the smoothing budget until the envelope has at most one
        # local maximum on two consecutive fits, instead of tracking line peaks.
        if detect_peaks(continuum).size <= 1:
            hits += 1
            if hits >= 2:
                break
        else:
            hits = 0
        smoothing *= 10
    if not np.all(np.isfinite(continuum)) or np.any(continuum <= 0):
        raise ValueError("Weighted fit produced a nonpositive or nonfinite continuum.")
    return continuum, peaks, influence


def fit_template_continuum(spectrum, plot=False, poly_order=1, spline_k=1, mpd=100):
    """Fit a pseudo-continuum to spectral peaks for RV templates.

    Select local maxima and reject anchors below 0.1 times the median flux
    or outside the initial polynomial trend minus 2 MAD / plus 6 MAD.
    MAD = median(abs(residual - median(residual))). If too few anchors remain,
    retry peak detection with closer spacing, then use uniformly spaced samples.
    Fit a smooth spline through the selected anchors.

    This retains hard anchor rejection; simulated spectra use the continuously
    weighted fit_spectral_continuum instead.

    Args:
        spectrum (ndarray): One-dimensional template flux.
        plot (bool): Unused; normalization plots are handled by the caller.
        poly_order (int): Polynomial order for initial fit.
        spline_k (int): Spline degree.
        mpd (int): Minimum peak distance for peak detection.

    Returns:
        continuum (np.ndarray): Fitted pseudo-continuum on the input samples.
        peaks (np.ndarray): Indices of the final spline anchors.
    """
    x = np.arange(spectrum.size)
    # initial peak pick with min peak distance
    peaks = detect_peaks(spectrum, mpd=mpd)

    # robust prune peaks against poly fit
    if peaks.size >= (poly_order + 1):
        f = np.polyfit(x[peaks], spectrum[peaks], poly_order)
        q = np.poly1d(f)
        resid = spectrum[peaks] - q(x[peaks])
        MAD = np.median(np.abs(resid - np.median(resid)))

        mask = np.ones(peaks.size, dtype=bool)
        # low S/N or deep lines
        mask &= spectrum[peaks] / np.nanmedian(spectrum) >= 0.1
        # reject strong devs from the poly trend
        mask &= spectrum[peaks] >= q(x[peaks]) - 2 * MAD
        mask &= spectrum[peaks] <= q(x[peaks]) + 6 * MAD
        peaks = peaks[mask]

    # Fallback: if we lost too many peaks, repopulate with coarse Mpd
    if peaks.size < max(3, spline_k + 1):
        peaks = detect_peaks(spectrum, mpd=max(1, mpd // 2))
        if peaks.size < max(3, spline_k + 1):
            # last resort: take a coarse uniform subset to avoid spline crash
            idx = np.linspace(0, spectrum.size - 1, num=max(3, spline_k + 1)).astype(
                int
            )
            peaks = idx

    # Grow smoothing geometrically; stop once the spline envelope yields ~one broad bump twice
    s = 1e6
    hits = 0
    continuum = None
    for _ in range(10):  # at most 10 spline fits per tier
        f = interpolate.UnivariateSpline(x[peaks], spectrum[peaks], k=spline_k, s=s)
        continuum = f(x)
        # count envelopes' peaks
        pk2 = detect_peaks(continuum)
        if pk2.size <= 1:
            hits += 1
            if hits >= 2:
                break
        else:
            hits = 0
        s *= 10.0  # geometric growth

    # If somehow didn't converge, ensure continuum is at least smooth-ish
    if continuum is None:
        f = interpolate.UnivariateSpline(x[peaks], spectrum[peaks], k=spline_k, s=1e8)
        continuum = f(x)

    return continuum, peaks


def _get_spectrum_velocity_grid(star, band, spectrum_mode, spectrum_source):
    """Read one band's prepared velocity grid from its spectral storage."""
    storage = star.flux_spec if spectrum_mode == "flux" else star.int_spec
    key = f"{band}_{spectrum_source}_{spectrum_mode}_{star.photosphere_temperature:1.4f}_v"
    return storage[key]


def normalize_gaussian_profiles(
    star,
    phase_spectra,
    phases,
    band,
    spectrum_mode,
    spectrum_source,
    save_dir,
    plot,
    continuum_velocity_min=65.0,
):
    """Continuum-normalize the simulated disk-integrated Gaussian profiles.

    Estimate one continuum level per phase from the median of both outer wings,
    where abs(velocity) >= continuum_velocity_min. Divide the entire profile
    by that level to remove its continuum brightness while retaining the line
    shape. Unlike peak-based normalization, this method fits no continuum curve.

    If fewer than ten bins meet the cutoff, use abs(velocity) >= 0.8*max_abs_vel.
    Wing samples are assumed to represent the continuum outside the line.

    Parameters
    ----------
    star : Star
        Stellar parameters and prepared spectral storage.
    phase_spectra : array_like, shape (n_phase, n_velocity)
        Disk-integrated profiles. A single 1D profile is treated as one phase.
    phases : array_like, shape (n_phase,)
        Rotation phases in cycles, used for plot filenames.
    band : str
        Configured band name used to find the velocity grid.
    spectrum_mode : str
        Spectral storage mode: 'flux' or 'intensity'.
    spectrum_source : str
        Profile label used in storage keys, normally 'gaussian_profile'.
    save_dir : str
        Output directory name under outputs.
    plot : bool
        Whether to save a normalization plot for each phase.
    continuum_velocity_min : float
        Minimum absolute velocity in km/s for continuum samples.

    Returns
    -------
    normalized_profiles : ndarray, shape (n_phase, n_velocity)
        Continuum-normalized_profiles profiles as a new float64 array.

    Raises
    ------
    ValueError
        If any phase has a nonfinite or nonpositive continuum estimate.
    """
    do_plot = (plot == 1) or (plot is True)
    phases = np.asarray(phases)

    if spectrum_source == "gaussian_profile":
        velocity = star.gaussian_profiles[band][0]
    else:
        velocity = _get_spectrum_velocity_grid(
            star, band, spectrum_mode, spectrum_source
        )

    velocity = np.asarray(velocity, dtype=np.float64)
    profiles = np.atleast_2d(np.asarray(phase_spectra, dtype=np.float64))

    continuum_mask = np.abs(velocity) >= float(continuum_velocity_min)

    # Too few wing samples: use the outer 20% of the absolute velocity range.
    if np.count_nonzero(continuum_mask) < 10:
        max_abs_velocity = np.nanmax(np.abs(velocity))
        continuum_mask = np.abs(velocity) >= 0.8 * max_abs_velocity

    normalized_profiles = np.empty_like(profiles)

    for phase_index, profile in enumerate(profiles):
        # Use the combined wings to estimate one continuum level for this phase.
        continuum = np.nanmedian(profile[continuum_mask])

        if not np.isfinite(continuum) or continuum <= 0.0:
            raise ValueError(
                f"Invalid Gaussian continuum at phase index {phase_index}: {continuum}"
            )

        normalized_profiles[phase_index] = profile / continuum

        # Save the original and normalized_profiles profiles for comparison.
        if do_plot:
            with plt.rc_context(PLOT_STYLE):
                fig, ax = plt.subplots(1, 1, figsize=(7, 3), facecolor="white", dpi=250)
                ax.plot(velocity, profile, "-k", lw=1, label="Original profile", alpha=0.6)
                ax.plot(
                    velocity,
                    normalized_profiles[phase_index],
                    "-",
                    lw=0.8,
                    label="Normalized profile",
                )

                ax.axhline(1, ls="-", lw=0.5, color="k", alpha=0.3)
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.legend(loc="lower right", fontsize="x-small", ncol=4)
                ax.set_xlabel("Velocity [km/s]")
                ax.set_ylabel("Flux")
                ax.set_ylim(0.9, 1.02)
                fig.savefig(
                    f"./outputs/{save_dir}/debug/continuum_fitting/flatten_{band}{identity_suffix(star)}_{phases[phase_index]:1.2f}.png",
                    bbox_inches="tight",
                )
                plt.close(fig)

    return normalized_profiles


def normalize_simulated_spectra(
    star,
    phase_spectra: np.ndarray,
    phases: np.ndarray,
    band: str,
    spectrum_mode: str,
    spectrum_source: str,
    save_dir: str,
    plot: bool,
) -> np.ndarray:
    """Continuum-normalize the simulated stellar spectra for all rotation phases.

    Estimate a smooth pseudo-continuum from continuously weighted spectral
    peaks separately for each phase,
    divide the disk-integrated spectrum by that continuum, then rescale its
    99th percentile to one. This removes the continuum shape and overall scale
    before spectral diagnostics. RV templates are normalized separately by
    normalize_ccf_template; Gaussian profiles use a wing-based continuum.

    Parameters
    ----------
    star : Star
        Stellar parameters and prepared spectral storage.
    phase_spectra : ndarray, shape (n_phases, n_velocity)
        Disk-integrated simulated spectra on a common velocity grid.
    phases : array_like, shape (n_phases,)
        Rotation phases in cycles, used for plot filenames.
    band : str
        Configured band name used to find the velocity grid.
    spectrum_mode : str
        Spectral storage mode: 'flux' or 'intensity'.
    spectrum_source : str
        Library label used in spectral storage keys.
    save_dir : str
        Output directory name under outputs.
    plot : bool
        Whether to save a normalization plot for each phase.

    Returns
    -------
    normalized_spectra : ndarray, shape (n_phases, n_velocity)
        Normalized spectra as a new float64 array.

    Notes
    -----
    Nonpositive continuum values are replaced by NaN. If the normalized
    spectrum's 99th percentile is nonfinite or nonpositive, the additional
    percentile scaling is skipped.
    """
    # All phases use the same prepared velocity grid for this band.
    velocity = _get_spectrum_velocity_grid(star, band, spectrum_mode, spectrum_source)

    phases = np.asarray(phases)
    n_phases = phase_spectra.shape[0]
    n_velocity_bins = velocity.shape[0]

    # Use a common minimum peak separation: 4% of the spectral bin count.
    minimum_peak_distance = int(n_velocity_bins * 0.04) if n_velocity_bins > 0 else 1

    normalized_spectra = np.empty((n_phases, n_velocity_bins), dtype=np.float64)

    do_plot = (plot == 1) or (plot is True)

    for phase_index in range(n_phases):
        flux = np.asarray(phase_spectra[phase_index], dtype=np.float64, order="C")
        # Weight candidate peaks continuously to avoid abrupt threshold rejection.
        # Each phase is fitted independently; no phase averaging is applied.
        continuum, peaks, influence = fit_spectral_continuum(
            flux,
            poly_order=2,
            spline_k=2,
            mpd=minimum_peak_distance,
            transition_mad=0.25,
        )

        # Mark nonphysical continuum values instead of dividing by zero or negatives.
        continuum = np.where(continuum <= 0.0, np.nan, continuum)

        normalized_flux = flux / continuum
        # Set the near-continuum level using the 99th percentile; no clipping is applied.
        percentile_scale = np.nanpercentile(normalized_flux, 99.0)
        if not np.isfinite(percentile_scale) or percentile_scale <= 0.0:
            percentile_scale = 1.0
        normalized_flux = normalized_flux / percentile_scale

        normalized_spectra[phase_index, :] = normalized_flux

        # Save the input spectrum, fitted continuum and normalized result.
        if do_plot:
            with plt.rc_context(PLOT_STYLE):
                fig, ax = plt.subplots(1, 1, figsize=(7, 3), facecolor="white", dpi=250)
                ax.plot(velocity, flux, "-k", lw=1, label="Original spectrum", alpha=0.6)
                ax.plot(velocity, normalized_flux, "-", lw=0.8, label="Normalized spectrum")
                if peaks.size:
                    ax.scatter(
                        velocity[peaks],
                        flux[peaks],
                        s=24 * influence,
                        zorder=5,
                        label="anchors (size reflects weight)",
                    )
                ax.plot(velocity, continuum, "-", lw=1.0, label="Pseudo-continuum")
                ax.axhline(1, ls="-", lw=0.5, color="k", alpha=0.3)
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.legend(loc="lower right", fontsize="x-small", ncol=4)
                ax.set_xlabel("Velocity [km/s]")
                ax.set_ylabel("Flux")
                ax.set_ylim(0.4, 1.05)
                fig.savefig(
                    f"./outputs/{save_dir}/debug/continuum_fitting/flatten_{band}{identity_suffix(star)}_{phases[phase_index]:1.2f}.png",
                    bbox_inches="tight",
                )
                plt.close(fig)

    return normalized_spectra

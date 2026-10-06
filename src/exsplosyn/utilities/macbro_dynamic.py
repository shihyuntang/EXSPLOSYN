"""Apply wavelength-dependent Gaussian instrumental broadening.

Python version bundled from syt_tools 2026.9.28.1.
"""

import numpy as np
from scipy.interpolate import make_interp_spline
from astropy.convolution import convolve


def macbro_dyn(w, s, hwhmlist=None, R=None):
    """
    Apply wavelength-dependent Gaussian instrumental broadening to a spectrum.

    Parameters
    ----------
    w : ndarray of shape (N,)
        Wavelength grid of the input spectrum. Any consistent wavelength
        unit may be used.
    s : ndarray of shape (N,)
        Input spectrum sampled on `w`.
    hwhmlist : ndarray of shape (N,), optional
        Half-width at half maximum (HWHM) of the Gaussian instrumental
        profile at each wavelength. Must use the same wavelength unit as `w`.
        Exactly one of `hwhmlist` or `R` must be provided.
    R : float, optional
        Spectral resolving power, defined as ``R = lambda / FWHM``.
        If provided, the wavelength-dependent Gaussian HWHM is calculated as

            HWHM(lambda) = lambda / (2 R).

        Exactly one of `hwhmlist` or `R` must be provided.

    Returns
    -------
    sout0 : ndarray of shape (N,)
        Instrumentally broadened spectrum on the original wavelength grid.

    Notes
    -----
    The instrumental Gaussian width may vary with wavelength. To perform the
    broadening with a single Gaussian kernel, the wavelength coordinate is
    first transformed so that the local instrumental HWHM becomes the constant
    reference width `hwhm_ref`.

    Each wavelength interval is rescaled according to

        dw_stretched = dw * hwhm_ref / hwhm_mid,

    where `hwhm_mid` is the average instrumental HWHM across that interval.
    The spectrum is then interpolated onto a uniformly sampled grid in the
    stretched coordinate.

    A Gaussian kernel with HWHM = `hwhm_ref` is constructed and truncated at
    approximately +/- 4 sigma. The stretched spectrum is convolved with this
    kernel and then interpolated back to the stretched coordinates
    corresponding to the original wavelength samples.

    Finally, the output spectrum is globally renormalized to preserve the
    total summed flux.
    """
    w = np.asarray(w)
    s = np.asarray(s)

    if len(w) != len(s):
        raise ValueError("`w` and `s` must have the same length.")
    
    # Require exactly one instrumental-width specification.
    if (hwhmlist is None) == (R is None):
        raise ValueError("Provide exactly one of `hwhmlist` or `R`.")

    # Convert a constant resolving power into wavelength-dependent HWHM.
    if R is not None:
        if R <= 0:
            raise ValueError("`R` must be positive.")

        hwhmlist = w / (2.0 * R)

    else:
        hwhmlist = np.asarray(hwhmlist)

        if len(hwhmlist) != len(w):
            raise ValueError(
                "`hwhmlist` must have the same length as `w`."
            )

    # Reference instrumental width used in the stretched coordinate.
    hwhm_ref = np.min(hwhmlist)

    # Average HWHM within each wavelength interval.
    hwhm_mid = 0.5 * (hwhmlist[:-1] + hwhmlist[1:])

    # Original wavelength intervals.
    dw = w[1:] - w[:-1]

    # Rescale each interval so that its local instrumental HWHM becomes
    # hwhm_ref in the stretched wavelength coordinate.
    dw_stretched = dw * hwhm_ref / hwhm_mid

    # Construct the stretched wavelength coordinate.
    w_stretched = np.concatenate(
        (
            np.array([w[0]]),
            w[0] + np.cumsum(dw_stretched),
        )
    )

    # Construct a uniformly sampled grid in the stretched coordinate.
    dw_fine = np.min(dw)
    w_fine = np.arange(
        np.min(w_stretched),
        np.max(w_stretched) + dw_fine,
        dw_fine,
    )

    # Interpolate the spectrum onto the uniform stretched grid.
    spline = make_interp_spline(w_stretched, s, k=3)
    s_fine = spline(w_fine)

    # --- Construct the Gaussian instrumental kernel ---
    # G(x) = exp[-ln(2) * (x / HWHM)^2]

    nw = len(w_fine)

    # Truncate the Gaussian at approximately +/- 4 sigma.
    # HWHM = sqrt(2 ln(2)) * sigma.
    nhalf = int(
        4.0 / np.sqrt(2.0 * np.log(2.0))
        * hwhm_ref / dw_fine
    )

    # Sample the kernel symmetrically on the dw_fine grid.
    ng = 2 * nhalf + 1
    wg = dw_fine * (np.arange(ng, dtype=float) - nhalf)

    # Gaussian profile expressed in terms of HWHM.
    gpro = np.exp(-np.log(2.0) * (wg / hwhm_ref)**2)

    # Normalize the discrete kernel to unit sum.
    gpro /= np.sum(gpro)

    # Pad with endpoint fluxes to reduce convolution edge effects.
    npad = nhalf + 2
    spad = np.concatenate(
        (
            np.full(npad, s_fine[0]),
            s_fine,
            np.full(npad, s_fine[-1]),
        )
    )

    # Convolve on the uniformly sampled stretched grid.
    # sout = np.convolve(spad, gpro, mode="same")
    sout = convolve(spad, gpro, boundary="fill", normalize_kernel=False)

    # Remove the padding.
    sout = sout[npad:npad + nw]

    # Evaluate the broadened spectrum at the stretched coordinates
    # corresponding to the original wavelength samples.
    spline = make_interp_spline(w_fine, sout, k=3)
    sout0 = spline(w_stretched)

    # Preserve the total summed flux.
    sout0 *= np.sum(s) / np.sum(sout0)

    return sout0

"""Rotational broadening adapted from CMJ's IDL ROTINT routine.

The original documentation cites Smith (1994, A&A, "in press") for the
differential-rotation law; that historical wording is retained below.


Edit History
------------
11-May-1994 CMJ Written.
30-Sep-2026 SYT Updated the Python implementation documentation and
                clarified the disk-sampling, limb-darkening, and weighting
                scheme. Renamed intermediate variables for readability
                without changing the disk-integration algorithm.
                Changed the default disk sampling from nr=10, ntheta=20
                to nr=20, ntheta=40 based on convergence testing against
                PyAstronomy rotBroad.

Python version bundled from syt_tools 2026.9.28.1.
"""

import numpy as np
from numba import njit

@njit
def vsini_broaden_cmj(w, s, vsini, eps=0.4, nr=20, ntheta=40, dif=0):
    """
    Modified from CMJ's IDL routine ROTINT.
    Apply rotational broadening to a spectrum by direct stellar-disk integration.

    The stellar disk is normalized to unit radius and divided into radial
    annuli and azimuthal sectors. Each surface element contributes a
    Doppler-shifted copy of the input spectrum, weighted by its projected
    area and a linear limb-darkening law.

    Parameters
    ----------
    w : ndarray of shape (N,)
        Wavelength grid of the input spectrum. Any consistent wavelength
        unit may be used.
    s : ndarray of shape (N,)
        Input spectrum sampled on `w`.
    vsini : float
        Projected stellar rotation velocity, in km/s.
    eps : float, optional
        Linear limb-darkening coefficient. The limb-darkening law is

            I(mu) / I(1) = 1 - eps * (1 - mu),

        where ``mu`` is the cosine of the viewing angle. Default is 0.4.
    nr : int, optional
        Number of radial annuli used to sample the projected stellar disk.
        Default is 20.
    ntheta : int, optional
        Controls the azimuthal sampling of the disk. The number of azimuthal
        sectors in an annulus at normalized radius ``r`` is

            nphi = int(ntheta * r).

        Default is 40.
    dif : float, optional
        Differential-rotation parameter. If nonzero, the local projected
        velocity is scaled according to

            Omega(theta) / Omega(eq)
                = 1 - dif/2 - (dif/2) * cos(2 * theta),

        where ``theta`` in this expression is the stellar co-latitude.
        Default is 0, corresponding to rigid rotation.

    Returns
    -------
    w : ndarray of shape (N,)
        The original wavelength grid.
    ns : ndarray of shape (N,)
        Rotationally broadened spectrum on the same wavelength grid as `w`.

    Notes
    -----
    The projected stellar radius is normalized to 1. The disk is sampled
    using concentric annuli with radial spacing ``dr = 1 / nr``. Each annulus
    is further divided into azimuthal sectors, with the number of sectors
    increasing approximately linearly with radius so that surface elements
    remain roughly similar in size across the projected disk.

    For each surface element, the local line-of-sight velocity is calculated
    from `vsini`. The input spectrum is shifted by this velocity and added to
    the disk-integrated spectrum with a weight equal to the projected surface
    area multiplied by the linear limb-darkening weight.

    The current implementation uses ``numpy.interp`` for the Doppler-shifted
    spectra. This performs linear interpolation and uses the endpoint values
    outside the wavelength range rather than linear extrapolation.

    Default disk sampling
    ---------------------
    The default disk sampling (nr=20, ntheta=40) was selected from a convergence
    test against PyAstronomy rotBroad. Relative to the original 10/20 sampling,
    the RMS flux difference decreased from 5.58e-4 to 1.43e-4, while runtime
    increased from ~0.004 s to ~0.015 s. Increasing the sampling further provided
    only modest accuracy improvements at substantially higher computational cost.

    Original documentation
    ----------------------
    This routine reads in a spectrum, s, on a wavelength scale, w, and a vsini
    with which to rotationally broaden the spectrum. The rotationally broadened
    spectrum is returned in ns. Parameters that can be set are the coefficient
    of the limb darkening law, eps (0.6 default), the number of radial steps on
    the disk, nr (default = 10), and the maximum number of steps in angle around
    the disk, ntheta (default = 100). Final optional parameter dif allows for
    differential rotation according to the law

        Omeg(th) / Omeg(eq) = 1 - dif/2 - (dif/2) cos(2 th).

    Dif = .675 nicely reproduces the law proposed by Smith, 1994, A&A, in press,
    to unify WTTS and CTTS. Dif = .23 is similar to observed solar differential
    rotation. Note: the th in the above expression is the stellar co-latitude,
    not the same as the integration variable used below. This is a disk
    integration routine.

    Edit History
    ------------
    11-May-1994 CMJ Written.
    30-Sep-2026 SYT Updated the Python implementation documentation and
                    clarified the disk-sampling, limb-darkening, and weighting
                    scheme. Renamed intermediate variables for readability
                    without changing the disk-integration algorithm.
                    Changed the default disk sampling from nr=10, ntheta=20
                    to nr=20, ntheta=40 based on convergence testing against
                    PyAstronomy rotBroad.
    """

    c = 2.99792458e5
    tarea = 0.0

    dr = 1. / nr  # Stellar radius is normalized to 1.
    ns = np.zeros(len(s), dtype=np.float64)

    for j in range(nr):
        # Sample the center of each radial annulus.
        r = dr/2. + j*dr

        # Viewing-angle cosine and linear limb-darkening weight
        # for this annulus.
        mu = np.sqrt(1. - r**2)
        ld_weight = 1. - eps * (1. - mu)

        # Projected area of the annulus, apart from a common factor of pi.
        # annulus_area = ((r + dr/2.)**2 - (r - dr/2.)**2)
        annulus_area = 2. * r * dr

        # Number of azimuthal sectors ("pizza slices").
        # Scale with r so that sectors have approximately similar sizes
        # across the stellar disk.
        nphi = int(ntheta * r)

        # Projected area weight of one sector.
        pizza_area = annulus_area / nphi

        # Apply limb darkening to the sector weight.
        pizza_weight = pizza_area * ld_weight
        # Total weight of this annulus.
        tarea += nphi * pizza_weight 

        dtheta = 2. * np.pi / nphi

        for k in range(nphi):
            # Sample the center of each azimuthal sector.
            # theta = np.pi/nphi + k*dtheta
            theta = (k + 0.5) * dtheta

            if dif != 0:
                vl = vsini * r*np.sin(theta) * (
                    1. - dif/2. - dif/2. * np.cos(
                        2.*np.arccos(r*np.cos(theta))
                    )
                )
            else:
                vl = vsini * r*np.sin(theta)

            ns += pizza_weight * np.interp(w, w + w*vl/c, s)
            # tarea += pizza_weight

    ns /= tarea

    return w, ns

# Name used by simulation rotational broadening calls.
rotint = vsini_broaden_cmj

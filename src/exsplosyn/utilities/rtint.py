"""Disk integration translated from the IDL RTINT routine.

Published reference used by EXSPLOSYN:
    Valenti, J. A. & Piskunov, N. (1996), A&AS, 118, 595-603.
    "Spectroscopy made easy: A new tool for fitting observations with
    synthetic spectra."
    https://doi.org/10.1051/aas:1996222
    https://ui.adsabs.harvard.edu/abs/1996A%26AS..118..595V/abstract

IDL source distributed by Oleg Kochukhov with RDS3V:
    https://www.astro.uu.se/~oleg/synth3.html
    https://www.astro.uu.se/~oleg/files/synth3/rds3v.pro

Original author history and citation request (original wording):
ORIGINAL SOURCE COMMENTS
------------------------

AUTHOR'S REQUEST:
    If you use this algorithm in work that you publish, please cite
      Valenti & Anderson 1996, PASP, currently in preparation.

MODIFICATION HISTORY:
       Feb-88  GM   Created ANA version.
    13-Oct-92 JAV   Adapted from G. Marcy's ANA routine of the same name.
    03-Nov-93 JAV   Switched to annular convolution technique.
    12-Nov-93 JAV   Fixed bug. Intensity components not added when vsini=0.
    14-Jun-94 JAV   Reformatted for "public" release. Heavily commented.
                    Pass deltav instead of 2.998d5/deltav. Added osamp
                      keyword. Added rebinning logic at end of routine.
                    Changed default osamp from 3 to 1.
    20-Feb-95 JAV   Added mu as an argument to handle arbitrary mu sampling
                      and remove ambiguity in intensity profile ordering.
                    Interpret VTURB as sqrt(2)*sigma instead of just sigma.
                    Replaced call_external with call to spl_{init|interp}.
    03-Apr-95 JAV   Multiply flux by !pi to give observed flux.
    24-Oct-95 JAV   Force "nmk" padding to be at least 3 pixels.
    18-Dec-95 JAV   Renamed from dskint() to rtint(). No longer make local
                      copy of intensities. Use radial-tangential instead
                      of isotropic Gaussian macroturbulence.
    26-Jan-99 JAV   For NMU=1 and VSINI=0, assume resolved solar surface#
                      apply R-T macro, but supress vsini broadening.
    01-Apr-99 GMH   Use annuli weights, rather than assuming equal area.
    01-Oct-26 SYT   Replaced the custom convolution routine with Astropy convolution.
                    Astropy was also ~2-5x faster in RTINT benchmarks.

Python version bundled from syt_tools 2026.9.28.1.
"""

import numpy as np
from scipy.interpolate import make_interp_spline
from astropy.convolution import convolve



def rtint(mu, inten, deltav, vsini_in, vrt_in, osamp=None):
    """
    Integrate mu-dependent intensity spectra over the visible stellar disk.

    The stellar disk is divided into annuli corresponding to the supplied
    limb-angle coordinates `mu`. For each annulus, the local intensity
    spectrum is broadened by the line-of-sight velocity distributions
    produced by stellar rotation and radial-tangential macroturbulence.
    The broadened annular spectra are then weighted by projected area and
    summed to produce the disk-integrated flux spectrum.

    Parameters
    ----------
    mu : ndarray of shape (n_mu,)
        Cosine of the angle between the local surface normal and the
        observer's line of sight, ``mu = cos(theta)``.
    inten : ndarray of shape (n_points, n_mu)
        Specific-intensity spectra corresponding to each value of `mu`.
    deltav : float
        Velocity spacing between adjacent spectral pixels. Must have the
        same velocity unit as `vsini_in` and `vrt_in`.
    vsini_in : float
        Projected equatorial rotational velocity, typically in km/s.
    vrt_in : float
        Radial-tangential macroturbulence parameter in the same velocity
        units as `deltav`. Following the original RTINT convention,
        ``vrt = sqrt(2) * sigma`` for the underlying Gaussian turbulent
        velocity distribution.
    osamp : int, optional
        Internal oversampling factor used during convolution. If greater
        than 1, each intensity spectrum is cubic-spline interpolated onto
        a finer velocity grid before broadening. Default is 1.

    Returns
    -------
    flux : ndarray of shape (n_points,)
        Disk-integrated flux spectrum.

    Notes
    -----
    Each supplied `mu` intensity is associated with a projected stellar
    annulus. Annulus boundaries are chosen at equal-area midpoints between
    adjacent `mu` samples, and each annulus is weighted by its projected
    area.

    Rotational broadening is represented by a normalized line-of-sight
    velocity probability distribution (LOS-velocity PDF) derived
    analytically from the geometry of each annulus.

    Radial-tangential macroturbulence is represented by a 50/50 mixture of
    radial and tangential Gaussian LOS-velocity PDFs. Their projected
    widths depend on `mu`.

    The final area-weighted mean intensity is multiplied by pi to produce
    the disk-integrated flux.

    This implementation is translated from the IDL RTINT routine described
    in the original source comments. EXSPLOSYN cites Valenti & Piskunov
    (1996), A&AS, 118, 595-603, https://doi.org/10.1051/aas:1996222.
    The historical citation request below is preserved verbatim.
    
    ORIGINAL SOURCE COMMENTS
    ------------------------
    
    AUTHOR'S REQUEST:
    	If you use this algorithm in work that you publish, please cite
    	  Valenti & Anderson 1996, PASP, currently in preparation.
    
    MODIFICATION HISTORY:
    	   Feb-88  GM	Created ANA version.
    	13-Oct-92 JAV	Adapted from G. Marcy's ANA routine of the same name.
    	03-Nov-93 JAV	Switched to annular convolution technique.
    	12-Nov-93 JAV	Fixed bug. Intensity components not added when vsini=0.
    	14-Jun-94 JAV	Reformatted for "public" release. Heavily commented.
    			Pass deltav instead of 2.998d5/deltav. Added osamp
    			  keyword. Added rebinning logic at end of routine.
    			Changed default osamp from 3 to 1.
    	20-Feb-95 JAV	Added mu as an argument to handle arbitrary mu sampling
    			  and remove ambiguity in intensity profile ordering.
    			Interpret VTURB as sqrt(2)*sigma instead of just sigma.
    			Replaced call_external with call to spl_{init|interp}.
    	03-Apr-95 JAV	Multiply flux by !pi to give observed flux.
    	24-Oct-95 JAV	Force "nmk" padding to be at least 3 pixels.
    	18-Dec-95 JAV	Renamed from dskint() to rtint(). No longer make local
    			  copy of intensities. Use radial-tangential instead
    			  of isotropic Gaussian macroturbulence.
    	26-Jan-99 JAV	For NMU=1 and VSINI=0, assume resolved solar surface#
    			  apply R-T macro, but supress vsini broadening.
    	01-Apr-99 GMH	Use annuli weights, rather than assuming equal area.
        01-Oct-26 SYT   Replaced the custom convolution routine with Astropy convolution.
                        Astropy was also ~2-5x faster in RTINT benchmarks.
    """

    # Make local copies because vsini may be modified below.
    vsini = float(vsini_in)
    vrt = float(vrt_in)

    # Internal oversampling factor.
    if osamp is None:
        os = 1
    else:
        if int(osamp) != osamp or osamp < 1:
            raise ValueError("`osamp` must be a positive integer.")
        os = int(osamp)

    # --- Define projected stellar annuli ---

    # Convert mu = cos(theta) to projected stellar radius
    # r = sin(theta) = sqrt(1 - mu^2).
    rmu = np.sqrt(1.0 - mu**2)

    # Sort annuli from disk center to limb.
    isort = np.argsort(rmu)
    rmu = rmu[isort]

    nmu = len(mu)

    # With only one mu angle, treat the spectrum as a spatially resolved
    # surface intensity and suppress rotational broadening.
    if nmu == 1:
        vsini = 0.0


    if nmu > 1 or vsini != 0:

        # Boundaries between neighboring mu samples are placed at the
        # midpoint in projected area rather than midpoint in radius.
        #
        # Since enclosed projected area is proportional to r^2,
        #
        #     r_boundary^2 = (r_i^2 + r_{i+1}^2) / 2.
        #
        r = np.sqrt(
            0.5 * (
                rmu[:-1]**2 +
                rmu[1:]**2
            )
        )

        # Add the disk center and stellar limb as the innermost and
        # outermost boundaries.
        r = np.concatenate(
            (
                np.array([0.0]),
                r,
                np.array([1.0]),
            )
        )

        # Projected-area fraction represented by each annulus.
        # The common factor pi cancels because the weights are normalized
        # to the area of a unit stellar disk.
        wt = r[1:]**2 - r[:-1]**2

    else:
        wt = np.array([1.0])
    
    # --- Construct the internal velocity-sampling grid ---

    npts = inten.shape[0]
    xpix = np.arange(npts)

    nfine = os * npts

    # Fine-grid coordinates are centered so that exactly `os` fine
    # samples correspond to each original spectral pixel.
    xfine = (
        0.5 / os
        * (2 * np.arange(nfine) - os + 1)
    )

    flux = np.zeros(nfine)

    
    for imu in range(nmu):

        # Specific-intensity spectrum represented by this annulus.
        ypix = inten[:, isort[imu]]

        if os == 1:
            yfine = ypix
        else:
            # Cubic-spline oversampling for convolution on a finer
            # velocity grid.
            spline = make_interp_spline(xpix, ypix, k=3)
            yfine = spline(xfine)

        # --- Rotational broadening ---

        # For rigid rotation, each annulus has a characteristic distribution
        # of projected line-of-sight velocities. `rkern` is the normalized
        # LOS-velocity PDF for the current annulus.
        if vsini > 0:

            r1 = r[imu]       # inner annulus radius
            r2 = r[imu + 1]   # outer annulus radius

            dv = deltav / os
            maxv = vsini * r2

            nrk = 2 * int(maxv / dv) + 3

            v = dv * (
                np.arange(nrk) -
                (nrk - 1) / 2
            )

            rkern = np.zeros(nrk)

            # Regime 1:
            # |v| < vsini*r1
            #
            # The constant-velocity chord crosses both the outer and inner
            # boundaries of the annulus. The available projected length is
            # therefore proportional to:
            #
            #     outer chord - inner chord.
            #
            j1 = np.where(np.abs(v) < vsini * r1)[0]

            if len(j1) > 0:
                rkern[j1] = (
                    np.sqrt((vsini * r2)**2 - v[j1]**2)
                    -
                    np.sqrt((vsini * r1)**2 - v[j1]**2)
)

            # Regime 2:
            # vsini*r1 < |v| < vsini*r2
            #
            # The constant-velocity chord lies outside the inner boundary,
            # so only the outer chord contributes.
            j2 = np.where(
                (np.abs(v) > vsini * r1)
                & (np.abs(v) < vsini * r2)
            )[0]

            if len(j2) > 0:
                rkern[j2] = np.sqrt(
                    (vsini * r2)**2 - v[j2]**2
                )

            # Normalize the annulus LOS-velocity PDF.
            rkern /= np.sum(rkern)


            # Convolve the local intensity spectrum with the rotational
            # LOS-velocity PDF. Extend the endpoint intensities beyond the
            # spectral boundaries to avoid artificial zero-padding.
            if nrk > 3:
                yfine = convolve( yfine, rkern, boundary="extend",
                normalize_kernel=False)

        # --- Radial-tangential macroturbulence ---

        muval = mu[isort[imu]]

        # The RT parameter is defined as sqrt(2)*sigma_v.
        # Convert the Gaussian sigma from velocity units to fine-grid pixels.
        sigma = (os * vrt / np.sqrt(2.0) / deltav)

        # LOS projections of the radial and tangential turbulent velocities.
        sigr = sigma * muval
        sigt = sigma * np.sqrt(1.0 - muval**2)

        # Use a sufficiently wide kernel while keeping it smaller than
        # approximately half the spectrum.
        nmk = min(int(10 * sigma), int((nfine - 3) / 2))
        nmk = max(nmk, 3)

        # Radial turbulent LOS-velocity PDF.
        if sigr > 0:
            xarg = (np.arange(2 * nmk + 1) - nmk) / sigr

            power = -0.5 * xarg**2
            power[power < -20] = -20

            mrkern = np.exp(power)
            mrkern /= np.sum(mrkern)

        else:
            mrkern = np.zeros(2 * nmk + 1)
            mrkern[nmk] = 1.0

        # Tangential turbulent LOS-velocity PDF.
        if sigt > 0:
            xarg = (np.arange(2 * nmk + 1) - nmk) / sigt

            power = -0.5 * xarg**2
            power[power < -20] = -20

            mtkern = np.exp(power)
            mtkern /= np.sum(mtkern)

        else:
            mtkern = np.zeros(2 * nmk + 1)
            mtkern[nmk] = 1.0

        # Radial-tangential macroturbulence assumes equal projected
        # surface fractions for radial and tangential components.
        mkern = 0.5 * mrkern + 0.5 * mtkern

        # Pad with endpoint intensities before convolution to reduce
        # edge artifacts.
        yfine_expand = np.concatenate(
            (np.ones(nmk) * yfine[0], yfine, np.ones(nmk) * yfine[-1],)
        )

        # Convolve with the radial-tangential LOS-velocity PDF.
        # yfine = convol(yfine_expand, mkern, False)
        yfine = convolve(yfine_expand, mkern, boundary="fill", 
                         fill_value=0.0,
                         normalize_kernel=False)
        
        yfine = yfine[nmk : nmk + nfine]

        # Add this annulus to the disk-integrated spectrum using its
        # projected-area fraction.
        flux += wt[imu] * yfine

    # Rebin the oversampled spectrum to the original wavelength sampling.
    flux = np.reshape(flux, (os, npts),)

    flux = np.sum(flux, axis=0) / os

    # The annulus weights sum to unit projected area. Multiply by the
    # projected area pi of a unit stellar disk to convert the weighted
    # mean intensity into disk-integrated flux.
    return np.pi * flux

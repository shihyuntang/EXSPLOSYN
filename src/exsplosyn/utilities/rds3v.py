"""Read SYNMAST spectral profiles using the translated IDL RDS3V routine.

IDL source supplied by Oleg Kochukhov for SYNMAST/Synth3 work.
Public source pages:
    https://www.astro.uu.se/~oleg/soft.html
    https://www.astro.uu.se/~oleg/synth3.html
Public IDL source:
    https://www.astro.uu.se/~oleg/files/synth3/rds3v.pro

Original IDL history (rds3v.pro):
    23-Feb-07 Kochukhov Adapted from rdmag.pro

The original IDL routine calls RTINT; see rtint.py for its citation request.
Python version bundled from syt_tools 2026.9.28.1.
"""

import numpy as np
import gzip
from scipy.interpolate import interp1d
from .rtint import rtint
from .macbro_dynamic import macbro_dyn

def rds3v(
    infile,
    vsini=0.0,
    vrt=0.0,
    resol=1e6,
    instrum=None,
    stokes=False,
    get_intensity=False,
    verbose=False,
    ):
    """
    Read a Synmast spectral synthesis file (.prf file) and construct a 1D spectrum.

    The mu-dependent specific intensities from Synmast are interpolated onto
    a logarithmic wavelength grid with constant velocity spacing. By default,
    the intensities are then integrated over the stellar disk using `rtint`,
    including optional rotational and radial-tangential macroturbulent
    broadening. Instrumental broadening may also be applied at a specified
    resolving power.

    Parameters
    ----------
    infile : str
        Path to the Synmast ``.prf`` input file. Gzip-compressed files
        (``.gz``) are also supported.
    vsini : float, optional
        Projected rotational velocity in km/s. Default is 0.
    vrt : float, optional
        Radial-tangential macroturbulent velocity in km/s. Default is 0.
    resol : float, optional
        Internal resolving power used to construct the logarithmic wavelength
        grid before disk integration. This determines the approximately
        constant velocity spacing of the working spectrum. Default is 1e6.
    instrum : float, optional
        Instrumental resolving power, ``R = lambda / FWHM``. If provided,
        Gaussian instrumental broadening is applied after disk integration.
        Default is None.
    stokes : bool, optional
        If True, read and process Stokes Q, U, and V in addition to Stokes I.
        This option must match the polarization content of the input file.
        Default is False.
    get_intensity : bool, optional
        If True, return the mu-dependent specific-intensity spectra on the
        constant-velocity wavelength grid without performing disk integration.
        Default is False.
    verbose : bool, optional
        If True, print the velocity spacing of the internal wavelength grid.
        Default is False.

    Returns
    -------
    x : ndarray of shape (N,)
        Logarithmically sampled wavelength grid with constant velocity spacing.
        Wavelength units are inherited from the input Synmast file.
    spectrum : ndarray
        Stokes-I spectrum in units of erg s^-1 cm^-2 cm^-1.
        For the default disk-integrated output, shape is ``(N,)``.
        If ``get_intensity=True``, returns the mu-dependent specific
        intensities with shape ``(N, n_mu)``.
    continuum : ndarray
        Continuum spectrum in units of
        erg s^-1 cm^-2 cm^-1, with the same dimensionality as `spectrum`.

    Other Returns
    -------------
    mu : ndarray of shape (n_mu,)
        ``mu = cos(theta)``. Returned only when
        ``get_intensity=True``.
    q, u, v : ndarray
        Stokes Q, U, and V spectra in units of
        erg s^-1 cm^-2 cm^-1. Returned only when ``stokes=True``.
        Their shapes match that of `spectrum`.
    
    When ``stokes=True``, the disk-integrated return order is
        ``x, I, continuum, Q, U, V``.
    When ``get_intensity=True``, the return order is
        ``x, mu, I, continuum, (Q, U, V)``.

    Notes
    -----
    Synmast intensities are multiplied by ``1e13`` before being returned so
    that the output flux scale is consistent with PHOENIX spectra in units of
    erg s^-1 cm^-2 cm^-1.

    The returned spectrum is not continuum normalized. A normalized Stokes-I
    spectrum can be constructed as ``spectrum / continuum``. Stokes Q, U,
    and V can be normalized in the same way.

    The `resol` and `instrum` parameters serve different purposes. `resol`
    controls the internal wavelength sampling used for disk integration,
    whereas `instrum` specifies the physical instrumental resolving power
    used for Gaussian broadening.
    """

    # Read full data from input file
    open_func = gzip.open if infile.endswith('.gz') else open
    try:
        with open_func(infile, 'rt') as f:
            data = f.readlines()
    except Exception as e:
        print(f"Error reading input file '{infile}': {e}!")
        raise

    # --- Read mu-dependent specific-intensity spectra ---
    # Number of spectral lines listed in the prf header
    nlines = int(data[0].split()[0])

    # Get the number of mu angles, where mu = cos(theta).
    nmu = int(data[nlines + 1])
    mu = np.array(list(map(float, data[nlines + 2].split())))
    
    # Read continuum intensities
    nwc = int(data[nlines + 3])
    continuum_data = np.array(
        [list(map(float, line.split())) 
         for line in data[nlines + 4: nlines + 4 + nwc]]
        ).T
    wlc = continuum_data[0]
    spc = continuum_data[1:]
    
    # Read line intensities
    line_start = nlines + 4 + nwc
    nwl = int(data[line_start])

    line_data = np.array(
        [list(map(float, line.split())) 
         for line in data[line_start + 1: line_start + 1 + nwl]]
         ).T
    wll = line_data[0]
    spl = line_data[1:]
    
    # Read polarization spectra if stokes parameter is enabled
    next_row = line_start + 1 + nwl
    if stokes:
        q_data = np.array(
            [list(map(float, line.split()))
                for line in data[next_row : next_row + nwl]
            ]
        ).T
        spq = q_data[1:]
        next_row += nwl

        u_data = np.array(
            [list(map(float, line.split()))
                for line in data[next_row : next_row + nwl]
            ]
        ).T
        spu = u_data[1:]
        next_row += nwl

        v_data = np.array(
            [list(map(float, line.split()))
                for line in data[next_row : next_row + nwl]
            ]
        ).T
        spv = v_data[1:]
        next_row += nwl

    # --- Construct a constant-velocity wavelength grid ---
    
    wfirst, wlast = wlc[0], wlc[-1]

    # Estimate the number of wavelength pixels requested by the input 
    # internal resolving power (resol). This sets approximately constant
    # fractional wavelength spacing, dλ/λ ~ 1/resol. We can get
    # λ_nx = λ_0 (1 + 1/resol)^(nx-1), where nx is the number of wavelength points.
    #    --> nx = log(λ_nx / λ_0) / log(1 + 1/resol) + 1
    
    nx = int(np.floor(
        np.log10(wlast / wfirst) / np.log10(1.0 + 1.0 / resol)
        ) + 1)

    # Adjust the effective resolving power so that the logarithmic grid
    # spans exactly from wfirst to wlast with nx points.
    rstep = 1.0 / (
        (wlast / wfirst) ** (1.0 / (nx - 1)) - 1.0
        )
    # Velocity spacing per wavelength pixel [km/s].
    deltav = 2.99792458e5 / rstep

    if verbose:
        print(f"Velocity spacing per pixel: {deltav:.3f} km/s")
    
    # Logarithmically spaced wavelength grid with constant velocity spacing.
    x = wfirst * (1.0 + 1.0 / rstep) ** np.arange(nx)
    
    sp1l = np.zeros((nx, nmu))
    sp1c = np.zeros((nx, nmu))

    if stokes:
        sp1q = np.zeros_like(sp1l)
        sp1u = np.zeros_like(sp1l)
        sp1v = np.zeros_like(sp1l)
    
    # --- Interpolate intensity values onto the new wavelength grid ---
    for imu in range(nmu):
        sp1l[:, imu] = interp1d(wll, spl[imu], fill_value="extrapolate")(x)
    
        if stokes:
            sp1q[:, imu] = interp1d(wll, spq[imu], fill_value="extrapolate")(x)
            sp1u[:, imu] = interp1d(wll, spu[imu], fill_value="extrapolate")(x)
            sp1v[:, imu] = interp1d(wll, spv[imu], fill_value="extrapolate")(x)

        sp1c[:, imu] = interp1d(wlc, spc[imu], fill_value="extrapolate")(x)

    # Return the mu-dependent specific-intensity spectra before disk integration.
    if get_intensity:
        if stokes:
            return x, mu, sp1l*1e13, sp1c*1e13, sp1q*1e13, sp1u*1e13, sp1v*1e13
        else:
            return x, mu, sp1l*1e13, sp1c*1e13

    # ----------------------------------------------------------------
    # --- Disk integration and broadening ---
    # Integrate the mu-dependent intensities over the stellar disk.
    # rtint also applies rotational broadening (vsini) and
    # radial-tangential macroturbulence (vrt).

    spl = rtint(mu, sp1l, deltav, vsini, vrt)
    spc = rtint(mu, sp1c, deltav, vsini, vrt)
    
    if stokes:
        spq = rtint(mu, sp1q, deltav, vsini, vrt)
        spu = rtint(mu, sp1u, deltav, vsini, vrt)
        spv = rtint(mu, sp1v, deltav, vsini, vrt)
    
    # Apply instrumental broadening if specified
    if instrum is not None:
        spl = macbro_dyn(x, spl, R=instrum)

        if stokes:
            spq = macbro_dyn(x, spq, R=instrum)
            spu = macbro_dyn(x, spu, R=instrum)
            spv = macbro_dyn(x, spv, R=instrum)

    if stokes:
        return x, spl*1e13, spc*1e13, spq*1e13, spu*1e13, spv*1e13
    else:
        return x, spl*1e13, spc*1e13

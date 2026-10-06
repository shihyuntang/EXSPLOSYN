"""Spectral rebinning translated through IDL/ANA and Python versions.


Edit History:
 10-Oct-90 JAV Create.
 22-Sep-91 JAV Translated from IDL to ANA.
 27-Aug-93 JAV Fixed bug in endpoint check: the "or" was essentially an "and".
 26-Aug-94 JAV Made endpoint check less restrictive so that identical old and
           new endpoints are now allowed. Switched to new Solaris library in call_external.
 29-Sep-26 SYT Modernized the spline interface using
           scipy.interpolate.make_interp_spline and updated the
           documentation. The interpolation branch was validated against
           the previous scipy.interpolate.interp1d implementation using
           the test spectrum in test_data.

Python version bundled from syt_tools 2026.9.28.1.
"""

import numpy as np
from scipy.interpolate import make_interp_spline

def rebin_jv(Wold, Sold, Wnew, verbose=False, logger=None):
    """
    Resample a spectrum onto a new wavelength grid using cubic splines.

    Parameters
    ----------
    Wold : ndarray of shape (Nold,)
        Original wavelength coordinates in a consistent unit, such as
        angstroms. Supply finite, strictly increasing wavelengths.
    Sold : ndarray of shape (Nold,)
        Spectrum values corresponding to `Wold`.
    Wnew : ndarray of shape (Nnew,)
        Finite, strictly increasing target wavelengths in the same units as
        `Wold`.
    verbose : bool, optional
        Enable the wavelength-coverage diagnostic when `logger` is provided.
        Default is False.
    logger : logging.Logger or compatible object, optional
        Object with a ``debug(message)`` method. The default, None, disables
        the coverage diagnostic.

    Returns
    -------
    Snew : ndarray
        Resampled spectrum with shape `(Nnew,)`, in the same units as `Sold`.

    Notes
    -----
    If the target grid has an equal or smaller pixel scale
    (`PSnew <= PSold`), the spectrum is evaluated directly on `Wnew` using
    a cubic spline fitted to the original spectrum.

    If the target grid is coarser (`PSnew > PSold`), each target pixel is
    divided into `XFac` subpixels. These subpixel positions form a finer
    sampling grid, `W`. The cubic spline fitted to the original spectrum is
    then evaluated at `W`, and the resulting subpixel values are averaged to
    estimate the flux of each target pixel.

    The coarser-grid branch therefore performs a numerical subpixel average,
    rather than an analytic spline integration.

    Local target-pixel widths are estimated from neighboring values in
    `Wnew`, allowing the target grid to have locally varying spacing. However,
    the choice between interpolation and subpixel averaging is based only on
    the average pixel scales of the two grids.

    Original documentation
    ----------------------
    Interpolates OR integrates a spectrum onto a new wavelength scale, depending
    on whether number of pixels per angstrom increases or decreases. Integration
    is effectively done analytically under a cubic spline fit to old spectrum.

    Parameters
    ----------
    Wold : old wavelngth scale.
    Sold : old spectrum to be binned.
    Wnew : new wavelength spectrum.

    Returns
    -------
    Snew : newly binned spectrum.

    Edit History:
     10-Oct-90 JAV Create.
     22-Sep-91 JAV Translated from IDL to ANA.
     27-Aug-93 JAV Fixed bug in endpoint check: the "or" was essentially an "and".
     26-Aug-94 JAV Made endpoint check less restrictive so that identical old and
               new endpoints are now allowed. Switched to new Solaris library in call_external.
     29-Sep-26 SYT Modernized the spline interface using
               scipy.interpolate.make_interp_spline and updated the
               documentation. The interpolation branch was validated against
               the previous scipy.interpolate.interp1d implementation using
               the test spectrum in test_data.
    """
    Nold = len(Wold)
    Nnew = len(Wnew)
    PSold = (Wold[-1] - Wold[0]) / (Nold-1) # avg. old pixel scale
    PSnew = (Wnew[-1] - Wnew[0]) / (Nnew-1) # avg. new pixel scale

    # Log endpoint coverage only; this does not prevent extrapolation.
    if verbose and (logger is not None):
        if (Wnew[0] < Wold[0]) or (Wnew[-1] > Wold[-1]):
            logger.debug('New wavelength scale not subset of old.')

    # Select integration or interpolation depending on change in dispersion (pixel scale, PS).
    spline = make_interp_spline(Wold, Sold, k=3)

    if PSnew <= PSold:
        # Pixel scale decreased --> Interpolation by cubic spline.
        Snew = spline(Wnew)
    else:
        # Pixel scale increased --> Integration under cubic spline.
        # The new wavelength grid is coarser than the original grid.

        # Number of subpixels used inside each new wavelength pixel.
        XFac = int(PSnew / PSold + 0.5)
        
        # Estimate the local width of each pixel in Wnew.
        # This allows the new wavelength grid to have locally varying spacing.
        dW = 0.5 * (Wnew[2:Nnew] - Wnew[0:Nnew-2])
        # Extrapolate the last pixel width from the two preceding estimates.
        dW = np.concatenate((dW, [2*dW[Nnew-3] - dW[Nnew-4]]))
        # Extrapolate the first pixel width from the first two estimates.
        dW = np.concatenate(([2*dW[0] - dW[1]], dW))

        # Construct XFac evenly spaced subpixel centers inside each Wnew pixel.
        # W initially has shape (XFac, Nnew):
        #   each row = one subpixel position across all new pixels.
        W = np.empty((XFac, Nnew))

        for i in range(XFac):
            # Fractional position of this subpixel relative to the pixel center.
            offset = float(2*i + 1) / (2.0*XFac) - 0.5
            W[i, :] = Wnew + dW * offset

        # Rearrange so that all subpixels belonging to the same output pixel
        # are adjacent, then flatten to a 1-D wavelength array for spline evaluation.
        W = W.T
        W = W.ravel()

        # Evaluate the original-spectrum spline at every subpixel center.
        S = spline(W)

        # Convert the subpixel values into an average contribution.
        S /= XFac

        # Group every XFac subpixels back into one output pixel and sum them.
        # Because S was divided by XFac above, this produces the mean sampled
        # flux within each new wavelength pixel.
        Snew = np.sum(S.reshape(Nnew, XFac), axis=1)

    return Snew

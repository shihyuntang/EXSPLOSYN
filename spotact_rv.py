import sys, os, logging
logger = logging.getLogger()

import numpy as np
from numba import njit
from scipy.signal import find_peaks

import visualization as vis

# @njit
# def _ccf(drvs, mode, w, f, tw, tf, c, weights, meanWl):
#     """
#     Fast cross-correlation calculation using Numba.

#     Args:
#         drvs (np.ndarray): Array of RV shifts to compute [km/s].
#         mode (str): 'lin' or 'doppler' for shift method.
#         w (np.ndarray): Wavelength grid of observation.
#         f (np.ndarray): Flux of observation.
#         tw (np.ndarray): Wavelength grid of template.
#         tf (np.ndarray): Flux of template.
#         c (float): Speed of light [km/s].
#         weights (np.ndarray): Weights for each point.
#         meanWl (float): Mean wavelength (for 'lin' mode).

#     Returns:
#         np.ndarray: Cross-correlation function for each RV shift.
#     """
#     cc = np.zeros(len(drvs), dtype=float)
#     df = f - np.nansum(f)/len(f)

#     i = 0
#     for rv in drvs:
#         if mode == "lin":
#             # Linear shift
#             f_shifted = np.interp(w, tw + meanWl * (rv / c), tf) * weights
#             dfs = f_shifted - np.nansum(f_shifted)/len(f_shifted)
#         elif mode == "doppler":
#             # Doppler shift
#             f_shifted = np.interp(w, tw * (1.0 + rv / c), tf) * weights
#             dfs = f_shifted - np.nansum(f_shifted)/len(f_shifted)
#         i += 1
#         cc[i] = np.nansum(df * dfs * weights)
#     return cc

@njit(cache=True)
def _ccf_doppler(drvs, w, f, tw, tf, c, weights):
    """
    Doppler-shift the template for each rv in `drvs`, linearly interpolate
    onto `w`, zero-mean both vectors, and compute weighted dot product.
    Returns CCF at each rv (same length as drvs).
    """
    nrv = drvs.size
    cc  = np.zeros(nrv, dtype=np.float64)

    # observation centered once
    df  = f - np.mean(f)

    for i in range(nrv):
        rv = drvs[i]
        # Doppler shift: lambda' = tw * (1 + rv/c)
        tw_shift = tw * (1.0 + rv / c)

        # Linear interp template → obs grid
        # np.interp expects ascending x; tw is ascending from the spectra pipeline.
        t_shift = np.interp(w, tw_shift, tf)

        # apply weights once (not twice)
        t_shift *= weights

        # center template *after* interpolation (keeps dot-product unbiased)
        t_shift -= np.mean(t_shift)

        # weighted cross-correlation (df already centered; multiply by weights once)
        # df*weights isn’t needed—weights already in t_shift
        s = 0.0
        for k in range(df.size):
            s += df[k] * t_shift[k]
        cc[i] = s

    return cc


@njit(cache=True)
def _ccf_lin(drvs, w, f, tw, tf, c, weights, meanWl):
    """
    Linear shift kernel (λ' = tw + meanWl * rv/c)
    """
    nrv = drvs.size
    cc  = np.zeros(nrv, dtype=np.float64)
    df  = f - np.mean(f)

    for i in range(nrv):
        rv = drvs[i]
        tw_shift = tw + meanWl * (rv / c)
        t_shift  = np.interp(w, tw_shift, tf)
        t_shift *= weights
        t_shift -= np.mean(t_shift)
        
        s = 0.0
        for k in range(df.size):
            s += df[k] * t_shift[k]
        cc[i] = s
    return cc


# def ccf_rv(
#     w,
#     f,
#     tw,
#     tf,
#     rvmin,
#     rvmax,
#     drv,
#     mode="doppler",
#     skipedge=0,
#     edgeTapering=None,
#     weights=None,
#     meanwvl=None,
#     ):
#     """
#     Cross-correlate a spectrum with a template.
#     modified from 
#     https://github.com/sczesla/PyAstronomy/blob/master/src/pyasl/asl/crosscorr.py#L138

#     The algorithm implemented here works as follows: For
#     each RV shift to be considered, the wavelength axis
#     of the template is shifted, either linearly or using
#     a proper Doppler shift depending on the `mode`. The
#     shifted template is then linearly interpolated at
#     the wavelength points of the observation
#     (spectrum) to calculate the cross-correlation function.

#     (see, e.g., `Lookwood et al. 2014, ApJ 783 <https://ui.adsabs.harvard.edu/abs/2014ApJ...783L..29L/abstract>`_).
    
#     Parameters
#     ----------
#     w : array
#         The wavelength axis of the observation.
#     f : array
#         The flux axis of the observation.
#     tw : array
#         The wavelength axis of the template.
#     tf : array
#         The flux axis of the template.
#     rvmin : float
#         Minimum radial velocity for which to calculate
#         the cross-correlation function [km/s].
#     rvmax : float
#         Maximum radial velocity for which to calculate
#         the cross-correlation function [km/s].
#     drv : float
#         The width of the radial-velocity steps to be applied
#         in the calculation of the cross-correlation
#         function [km/s].
#     mode : string, {lin, doppler}, optional
#         The mode determines how the wavelength axis will be
#         modified to represent a RV shift. If "lin" is specified,
#         a mean wavelength shift will be calculated based on the
#         mean wavelength of the observation. The wavelength axis
#         will then be shifted by that amount. If "doppler" is
#         specified (the default), the wavelength axis will
#         properly be Doppler-shifted.
#     skipedge : int, optional
#         If larger zero, the specified number of bins will be
#         skipped from the begin and end of the observation. This
#         may be useful if the template does not provide sufficient
#         coverage of the observation.
#     edgeTapering : float or tuple of two floats
#         If not None, the method will "taper off" the edges of the
#         observed spectrum by multiplying with a sine function. If a float number
#         is specified, this will define the width (in wavelength units)
#         to be used for tapering on both sides. If different tapering
#         widths shall be used, a tuple with two (positive) numbers
#         must be given, specifying the width to be used on the low- and
#         high wavelength end. If a nonzero 'skipedge' is given, it
#         will be applied first. Edge tapering can help to avoid
#         edge effects (see, e.g., Gullberg and Lindegren 2002, A&A 390).
#     weights : array of floats, optional
#         If given, it specifies weights for individual data points (given by f).
#         A possible choice for weights is 1/sigma_i**2 where sigma_i is the
#         uncertainty of the i-th data point.
#     meanwvl : float, optional
#         If given, this value will be adopted for the mean wavelength
#         :math:`\\bar{w}` to calculate shifts in 'lin' mode instead of
#         the mean of the input array w.

#     Returns
#     -------
#     dRV : array
#         The RV axis of the cross-correlation function. The radial
#         velocity refer to a shift of the template, i.e., positive
#         values indicate that the template has been red-shifted and
#         negative numbers indicate a blue-shift of the template.
#         The numbers are given in km/s.
#     CC : array
#         The cross-correlation function.
#     """

    

#     if weights is None:
#         weights = np.ones_like(w, dtype=float)
#     else:
#         weights = weights.copy()

#     # Copy and cut wavelength and flux arrays
#     w, f = w.copy(), f.copy()
#     if skipedge > 0:
#         w, f = w[skipedge:-skipedge], f[skipedge:-skipedge]
#         weights = weights[skipedge:-skipedge]

#     if edgeTapering is not None:
#         # Smooth the edges using a sine
#         if isinstance(edgeTapering, (float, int)):
#             edgeTapering = [float(edgeTapering), float(edgeTapering)]

#         # Carry out edge tapering (left edge)
#         indi = np.where(w < w[0] + edgeTapering[0])[0]
#         f[indi] *= np.sin((w[indi] - w[0]) / edgeTapering[0] * np.pi / 2.0)
#         # Carry out edge tapering (right edge)
#         indi = np.where(w > (w[-1] - edgeTapering[1]))[0]
#         f[indi] *= np.sin(
#             (w[indi] - w[indi[0]]) / edgeTapering[1] * np.pi / 2.0 + np.pi / 2.0
#         )

#     # Speed of light in km/s
#     c = 299792.458

#     # Check whether template is large enough
#     if mode == "lin":
#         if meanwvl is None:
#             meanWl = np.mean(w)
#         else:
#             meanWl = meanwvl
#         dwlmax = meanWl * (rvmax / c)
#         dwlmin = meanWl * (rvmin / c)
#         if (tw[0] + dwlmax) > w[0]:
#             print("The minimum wavelength is not covered by the template for all indicated RV shifts.")
#             return None
               
#         if (tw[-1] + dwlmin) < w[-1]:
#             print("The maximum wavelength is not covered by the template for all indicated RV shifts.")
#             return None
#     elif mode == "doppler":
#         # Ensure that the template covers the entire observation range for all Doppler shifts
#         maxwl = tw[-1] * (1.0 + rvmin / c)
#         minwl = tw[0] * (1.0 + rvmax / c)

#         if minwl > w[0]:
#             print("The minimum wavelength is not covered by the template for all indicated RV shifts.")
#             print(f"  Template too short after applying RV: min shifted λ = {minwl:1.4f}, but observation starts at {w[0]:1.4f}")
#             return None

#         if maxwl < w[-1]:
#             print("The maximum wavelength is not covered by the template for all indicated RV shifts.")
#             print(f"  Template not long enough after applying RV: max shifted λ = {maxwl:1.4f}, but observation ends at {w[-1]:1.4f}")
#             return None
#     else:
#         print('Error')
#         return None
    
#     # Calculate the cross correlation
#     drvs = np.arange(rvmin, rvmax, drv, dtype=float)

#     # --- using numba ---
#     # if mode == "lin":
#     #     cc = _ccf(drvs, mode, w, f, tw, tf, c, weights, meanWl)
#     # elif mode == "doppler":
#     #     meanWl = 0
#     #     cc = _ccf(drvs, mode, w, f, tw, tf, c, weights, meanWl)
#     if mode == "lin":
#         return _ccf_lin(drvs, w, f, tw, tf, c, weights, meanWl)
#     else:
#         return _ccf_doppler(drvs, w, f, tw, tf, c, weights)

    # ------------------
        
    # return cc

def ccf_rv(
    w, f, tw, tf, rvmin, rvmax, drv, mode="doppler",
    skipedge=0, edgeTapering=None, weights=None, meanwvl=None
):
    """
    Cross-correlate spectrum (w,f) with template (tw,tf) over rv grid [rvmin,rvmax) with step drv.
    Returns the CCF array (same length as drvs).  (Calling code already builds drvs.)
    """
    # weights
    if weights is None:
        weights = np.ones_like(w, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).copy()

    # copy & trim
    w   = np.asarray(w,  dtype=np.float64)
    f   = np.asarray(f,  dtype=np.float64).copy()
    tw  = np.asarray(tw, dtype=np.float64)
    tf  = np.asarray(tf, dtype=np.float64)

    if skipedge > 0:
        w, f = w[skipedge:-skipedge], f[skipedge:-skipedge]
        weights = weights[skipedge:-skipedge]

    # optional sine taper at edges (helpful vs. ringing)
    if edgeTapering is not None:
        if isinstance(edgeTapering, (float, int)):
            edgeTapering = (float(edgeTapering), float(edgeTapering))
        wL, wR = edgeTapering
        if wL > 0:
            idx = np.where(w < w[0] + wL)[0]
            if idx.size:
                f[idx] *= np.sin((w[idx] - w[0]) / wL * (np.pi * 0.5))
        if wR > 0:
            idx = np.where(w > w[-1] - wR)[0]
            if idx.size:
                f[idx] *= np.sin((w[idx] - (w[-1] - wR)) / wR * (np.pi * 0.5) + np.pi * 0.5)

    # speed of light
    c = 299792.458

    # coverage checks (same logic as your version, but silent returns None on failure)
    if mode == "lin":
        meanWl = float(np.mean(w) if (meanwvl is None) else meanwvl)
        dwlmax = meanWl * (rvmax / c)
        dwlmin = meanWl * (rvmin / c)
        if (tw[0] + dwlmax) > w[0]:
            return None
        if (tw[-1] + dwlmin) < w[-1]:
            return None
    elif mode == "doppler":
        maxwl = tw[-1] * (1.0 + rvmin / c)
        minwl = tw[0]  * (1.0 + rvmax / c)
        if minwl > w[0] or maxwl < w[-1]:
            return None
    else:
        return None

    # RV grid
    drvs = np.arange(rvmin, rvmax, drv, dtype=np.float64)

    # Call specialized kernel (no string branching inside Numba)
    if mode == "lin":
        return _ccf_lin(drvs, w, f, tw, tf, c, weights, meanWl)
    else:
        return _ccf_doppler(drvs, w, f, tw, tf, c, weights)
    

def mc_ccf(
        wave_obs, flux_obs_nor_box, wave_tem, flux_tem_nor, 
        ccf_lw, ccf_up, step_size, skipedge, visible_phi, i):
    """
    Compute the CCF for a single phase (for parallelization).

    Args:
        wave_obs (np.ndarray): Wavelength grid of observation.
        flux_obs_nor_box (np.ndarray): 2D array of normalized observed flux (N_phases x N_wave).
        wave_tem (np.ndarray): Wavelength grid of template.
        flux_tem_nor (np.ndarray): Normalized template flux.
        ccf_lw (float): Lower RV bound [km/s].
        ccf_up (float): Upper RV bound [km/s].
        step_size (float): RV step size [km/s].
        skipedge (int): Number of edge bins to skip.
        visible_phi (np.ndarray): Array of phases.
        i (int): Index of the phase.

    Returns:
        np.ndarray: CCF for this phase.
    """
    flux_obs_nor = flux_obs_nor_box[i]
    cc = ccf_rv(
        wave_obs, flux_obs_nor, wave_tem, flux_tem_nor, 
        ccf_lw, ccf_up, step_size, mode="doppler", 
        skipedge=skipedge)
    return cc


def _quad_refine(x, y, k):
    """
    Parabolic refine around index k using points (k-1,k,k+1).
    Returns refined x_peak (float). Falls back to x[k] at edges / degenerate cases.
    """
    n = y.size
    if k <= 0 or k >= n - 1:
        return x[k]
    x0, x1, x2 = x[k-1], x[k], x[k+1]
    y0, y1, y2 = y[k-1], y[k], y[k+1]
    # fit parabola vertex analytically
    denom = (y0 - 2*y1 + y2)
    if denom == 0.0:
        return x1
    delta = 0.5 * (y0 - y2) / denom  # offset from center in grid units
    # clamp to [-1,1] to be safe
    if delta < -1.0: delta = -1.0
    if delta >  1.0: delta =  1.0
    # x is uniform grid here; but we pass the real x for rigor:
    dx = (x2 - x0) * 0.5
    return x1 + delta * dx


def get_ccf_peak(
        ccf_box, drvs, visible_phi, band, outdir, refine=True, prominence=0.1):
    """
    Locate the RV of the strongest (most prominent) CCF peak per phase.
    If `refine` is True, apply a tiny quadratic refinement for sub-grid precision.
    """
    rvs = np.full(len(ccf_box), np.nan, dtype=np.float64)

    for i in range(len(ccf_box)):
        ccf = ccf_box[i]
        if ccf is None or len(ccf) == 0:
            continue
        
        # --- Normalize the CCF to [0, 1] ---
        ccf_min = np.nanmin(ccf)
        ccf_max = np.nanmax(ccf)
        if ccf_max > ccf_min:
            ccf_norm = (ccf - ccf_min) / (ccf_max - ccf_min)
        else:
            ccf_norm = ccf

        peaks, props = find_peaks(ccf_norm, prominence=prominence)
        
        # Optionally:
        # vis.plot_cc(drvs, ccf, rvs[i], visible_phi[i], band, props)

        if peaks.size == 0:
            print(f' >> CCF cannot find peaks at {band} phase {visible_phi[i]:.3f}')
            vis.plot_cc(drvs, ccf, None, visible_phi[i], band, props, outdir)
            continue

        # most prominent peak
        k = peaks[np.argmax(props["prominences"])]
        rv_peak = drvs[k]

        # edge guard
        if k == 0 or k == len(ccf) - 1:
            print(f' >> CCF peak at edge at {band} phase {visible_phi[i]:.3f}')
            rvs[i] = np.nan
            continue

        # optional sub-grid refine
        rvs[i] = _quad_refine(drvs, ccf, k) if refine else rv_peak

        # Optionally:
        # vis.plot_cc(drvs, ccf, rvs[i], visible_phi[i], band, props, outdir)
    return rvs





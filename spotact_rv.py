import sys, os, logging
logger = logging.getLogger()

import numpy as np
from numba import njit
from scipy.signal import find_peaks

@njit
def _ccf(drvs, mode, w, f, tw, tf, c, weights, meanWl):
    """
    Fast cross-correlation calculation using Numba.

    Args:
        drvs (np.ndarray): Array of RV shifts to compute [km/s].
        mode (str): 'lin' or 'doppler' for shift method.
        w (np.ndarray): Wavelength grid of observation.
        f (np.ndarray): Flux of observation.
        tw (np.ndarray): Wavelength grid of template.
        tf (np.ndarray): Flux of template.
        c (float): Speed of light [km/s].
        weights (np.ndarray): Weights for each point.
        meanWl (float): Mean wavelength (for 'lin' mode).

    Returns:
        np.ndarray: Cross-correlation function for each RV shift.
    """
    cc = np.zeros(len(drvs), dtype=float)
    df = f - np.nansum(f)/len(f)

    i = 0
    for rv in drvs:
        if mode == "lin":
            # Linear shift
            f_shifted = np.interp(w, tw + meanWl * (rv / c), tf) * weights
            dfs = f_shifted - np.nansum(f_shifted)/len(f_shifted)
        elif mode == "doppler":
            # Doppler shift
            f_shifted = np.interp(w, tw * (1.0 + rv / c), tf) * weights
            dfs = f_shifted - np.nansum(f_shifted)/len(f_shifted)
        i += 1
        cc[i] = np.nansum(df * dfs * weights)
    return cc

def ccf_rv(
    w,
    f,
    tw,
    tf,
    rvmin,
    rvmax,
    drv,
    mode="doppler",
    skipedge=0,
    edgeTapering=None,
    weights=None,
    meanwvl=None,
    ):
    """
    Cross-correlate a spectrum with a template.
    modified from 
    https://github.com/sczesla/PyAstronomy/blob/master/src/pyasl/asl/crosscorr.py#L138

    The algorithm implemented here works as follows: For
    each RV shift to be considered, the wavelength axis
    of the template is shifted, either linearly or using
    a proper Doppler shift depending on the `mode`. The
    shifted template is then linearly interpolated at
    the wavelength points of the observation
    (spectrum) to calculate the cross-correlation function.

    (see, e.g., `Lookwood et al. 2014, ApJ 783 <https://ui.adsabs.harvard.edu/abs/2014ApJ...783L..29L/abstract>`_).
    
    Parameters
    ----------
    w : array
        The wavelength axis of the observation.
    f : array
        The flux axis of the observation.
    tw : array
        The wavelength axis of the template.
    tf : array
        The flux axis of the template.
    rvmin : float
        Minimum radial velocity for which to calculate
        the cross-correlation function [km/s].
    rvmax : float
        Maximum radial velocity for which to calculate
        the cross-correlation function [km/s].
    drv : float
        The width of the radial-velocity steps to be applied
        in the calculation of the cross-correlation
        function [km/s].
    mode : string, {lin, doppler}, optional
        The mode determines how the wavelength axis will be
        modified to represent a RV shift. If "lin" is specified,
        a mean wavelength shift will be calculated based on the
        mean wavelength of the observation. The wavelength axis
        will then be shifted by that amount. If "doppler" is
        specified (the default), the wavelength axis will
        properly be Doppler-shifted.
    skipedge : int, optional
        If larger zero, the specified number of bins will be
        skipped from the begin and end of the observation. This
        may be useful if the template does not provide sufficient
        coverage of the observation.
    edgeTapering : float or tuple of two floats
        If not None, the method will "taper off" the edges of the
        observed spectrum by multiplying with a sine function. If a float number
        is specified, this will define the width (in wavelength units)
        to be used for tapering on both sides. If different tapering
        widths shall be used, a tuple with two (positive) numbers
        must be given, specifying the width to be used on the low- and
        high wavelength end. If a nonzero 'skipedge' is given, it
        will be applied first. Edge tapering can help to avoid
        edge effects (see, e.g., Gullberg and Lindegren 2002, A&A 390).
    weights : array of floats, optional
        If given, it specifies weights for individual data points (given by f).
        A possible choice for weights is 1/sigma_i**2 where sigma_i is the
        uncertainty of the i-th data point.
    meanwvl : float, optional
        If given, this value will be adopted for the mean wavelength
        :math:`\\bar{w}` to calculate shifts in 'lin' mode instead of
        the mean of the input array w.

    Returns
    -------
    dRV : array
        The RV axis of the cross-correlation function. The radial
        velocity refer to a shift of the template, i.e., positive
        values indicate that the template has been red-shifted and
        negative numbers indicate a blue-shift of the template.
        The numbers are given in km/s.
    CC : array
        The cross-correlation function.
    """

    

    if weights is None:
        weights = np.ones_like(w, dtype=float)
    else:
        weights = weights.copy()

    # Copy and cut wavelength and flux arrays
    w, f = w.copy(), f.copy()
    if skipedge > 0:
        w, f = w[skipedge:-skipedge], f[skipedge:-skipedge]
        weights = weights[skipedge:-skipedge]

    if edgeTapering is not None:
        # Smooth the edges using a sine
        if isinstance(edgeTapering, (float, int)):
            edgeTapering = [float(edgeTapering), float(edgeTapering)]

        # Carry out edge tapering (left edge)
        indi = np.where(w < w[0] + edgeTapering[0])[0]
        f[indi] *= np.sin((w[indi] - w[0]) / edgeTapering[0] * np.pi / 2.0)
        # Carry out edge tapering (right edge)
        indi = np.where(w > (w[-1] - edgeTapering[1]))[0]
        f[indi] *= np.sin(
            (w[indi] - w[indi[0]]) / edgeTapering[1] * np.pi / 2.0 + np.pi / 2.0
        )

    # Speed of light in km/s
    c = 299792.458

    # Check whether template is large enough
    if mode == "lin":
        if meanwvl is None:
            meanWl = np.mean(w)
        else:
            meanWl = meanwvl
        dwlmax = meanWl * (rvmax / c)
        dwlmin = meanWl * (rvmin / c)
        if (tw[0] + dwlmax) > w[0]:
            print("The minimum wavelength is not covered by the template for all indicated RV shifts.")
            return None
               
        if (tw[-1] + dwlmin) < w[-1]:
            print("The maximum wavelength is not covered by the template for all indicated RV shifts.")
            return None
    elif mode == "doppler":
        # Ensure that the template covers the entire observation range for all Doppler shifts
        maxwl = tw[-1] * (1.0 + rvmin / c)
        minwl = tw[0] * (1.0 + rvmax / c)

        if minwl > w[0]:
            print("The minimum wavelength is not covered by the template for all indicated RV shifts.")
            print(f"  Template too short after applying RV: min shifted λ = {minwl:1.4f}, but observation starts at {w[0]:1.4f}")
            return None

        if maxwl < w[-1]:
            print("The maximum wavelength is not covered by the template for all indicated RV shifts.")
            print(f"  Template not long enough after applying RV: max shifted λ = {maxwl:1.4f}, but observation ends at {w[-1]:1.4f}")
            return None
    else:
        print('Error')
        return None
    
    # Calculate the cross correlation
    drvs = np.arange(rvmin, rvmax, drv, dtype=float)

    # --- using numba ---
    if mode == "lin":
        cc = _ccf(drvs, mode, w, f, tw, tf, c, weights, meanWl)
    elif mode == "doppler":
        meanWl = 0
        cc = _ccf(drvs, mode, w, f, tw, tf, c, weights, meanWl)
    # ------------------
        
    return cc

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

def get_ccf_peak(ccf_box, drvs, visible_phi, band):
    """
    Find the RV shift for each phase by locating the most prominent peak in the CCF.

    Args:
        ccf_box (np.ndarray): 2D array of CCFs (N_phases x N_drvs).
        drvs (np.ndarray): Array of RV shifts corresponding to CCF axis.
        visible_phi (np.ndarray): Array of phases.
        band (str): Band name (for logging).

    Returns:
        np.ndarray: Array of RV shifts (one per phase).
    """
    new_rv_box = []

    for i in range(len(ccf_box)):
        ccf = ccf_box[i]
        peaks, properties = find_peaks(ccf, prominence=0.1) 

        if len(peaks) != 0:
            prominences = properties["prominences"]
            _loc = np.argmax(prominences)
            new_rv = drvs[peaks][_loc]

            # Check if the peak is at the edge of the RV grid
            if -np.abs(new_rv) == drvs[0]:
                new_rv = np.nan
                print(f'\n     >>>> CCF hit edge {new_rv}km/s at {band} phase {visible_phi[i]:1.3f}!')
        else:
            new_rv = np.nan
            print(f'\n     >>>> CCF cannot find peaks at {band} phase {visible_phi[i]:1.3f}!')
        
        # Optionally: vis.plot_cc(drvs, ccf, new_rv, visible_phi[i], band, properties)
        new_rv_box.append(new_rv)

    return np.array(new_rv_box)


import sys, os, logging
logger = logging.getLogger()
from functools import partial

import numpy as np
from numpy import cos, sin
from numba import njit
from scipy import signal,interpolate
from scipy.signal import find_peaks
from astropy.table import Table
import astropy.units as u
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from specutils import Spectrum1D
from specutils.analysis import equivalent_width

import starspot as stsp
import visualization as vis
import synspecprep as ssp
from basic_spec import rotint
from macbro_dynamic import macbro_dyn
from pqdm.processes import pqdm

from ldtk import LDPSetCreator, BoxcarFilter, TabulatedFilter
from pyphot import astropy as pyphot
from pyphot import svo, unit


from rebin_jv import rebin_jv


class Star:
    def __init__(
        self, prot, vrot, incl, psi, rad_sun, rad, 
        Temp, Tspot, Tfaculae, logg, FeH, bands2use
    ):
        """
        Initialize a Star object for SOAP-TTS simulation.

        Args:
            prot (float): Rotational period [days]
            vrot (float): Rotational velocity [km/s]
            incl (float): Inclination [deg]
            psi (float): Phase
            rad_sun (float): Sun radius [km]
            rad (float): Radius [Rsol]
            Temp (float): Effective temperature of the star [K]
            Tspot (float or array): Spot temperature(s) [K]
            Tfaculae (float or array): Faculae temperature(s) [K]
            logg (float): log(g)
            FeH (float): [Fe/H]
            bands2use (dict): Band definitions
        """
        self.prot = prot
        self.vrot = vrot
        self.incl = incl
        self.psi = psi
        self.rad_sun = rad_sun
        self.rad = rad
        self.Temp = Temp
        self.Tspot = Tspot
        self.Tfaculae = Tfaculae
        self.logg = logg
        self.FeH = FeH
        self.bands2use = bands2use

    def get_quadratic_limbdarken(self, temperature, line_centers, l_widths):
        """
        Calculate quadratic limb-darkening coefficients for each temperature and band.

        Args:
            temperature (array): List of temperatures to compute LD for.
            line_centers (array): Central wavelengths (AA).
            l_widths (array): Band widths (AA).
        """
        LD_teff_wv_coeff = {}
        for tt in temperature:
            # Ensure arrays for line centers and widths
            if len(line_centers) == 1:
                line_centers = np.array([line_centers])
                l_widths = np.array([l_widths])
            elif len(line_centers) > 1:
                line_centers = np.array(line_centers)
                l_widths = np.array(l_widths)

            # Convert AA to nm for ldtk
            lc_s = line_centers / 10
            _width = l_widths / 10

            # Create boxcar filters for each band
            filters = [
                BoxcarFilter(
                    f'lc{lc_s[i]//1000}', lc_s[i]-_width[i], lc_s[i]+_width[i]
                ) for i in range(len(lc_s))
            ]
            # ldtk needs a small std for Bayesian sampling
            sc = LDPSetCreator(
                teff=(tt, 1), logg=(self.logg, 0.01), z=(self.FeH, 0.01),
                filters=filters, dataset='visir-lowres'
            )
            ps = sc.create_profiles(nsamples=10)
            qc, _ = ps.coeffs_qd()
            LD_wv_coeff = {
                f"w{int(line_centers[i] // 100)}_{int(tt)}": qc[i]
                for i in range(len(line_centers))
            }
            # Special handling for FeOH line center
            if 15631.35 in line_centers:
                _loc = np.where(line_centers == 15631.35)[0][0]
                LD_wv_coeff[f'ewr_FeOH_{int(tt)}'] = qc[_loc]
            LD_teff_wv_coeff.update(LD_wv_coeff)
        self.LD_coeff = LD_teff_wv_coeff

    def find_fluxspec(self, Vel_bin_size, plot, syn_model_path=None):
        """
        Find and store flux spectra for the star, spot, and plage at different temperatures.

        Args:
            Vel_bin_size (float): Velocity bin size for spectra.
            plot (bool): If True, plot intermediate results.
            syn_model_path (dict): Paths to synthetic models.
        """
        # Build temperature list for all components (star, spot, plage)
        temp_list = np.concatenate([[self.Temp], self.Tspot, self.Tfaculae])
        temp_list = np.unique(temp_list)
        self.flux_spec = {}

        wv_regions = list(self.bands2use.keys())
        for band in wv_regions:
            line_cen = self.bands2use[band][0]
            delta_wv = self.bands2use[band][1]
            intORflux = self.bands2use[band][3]
            model_type = self.bands2use[band][4]

            # Only process bands that require flux spectra
            if intORflux != 'flux':
                continue

            # Define wavelength window for this band
            wvmin, wvmax = int(line_cen - delta_wv), int(line_cen + delta_wv)

            for tt in temp_list:
                # --- Read PHX SYN Spectra ---
                if model_type.lower() == 'phx':
                    sim_use_spec = f'SimUse_PHX_ACES_{tt}_{self.logg}_{self.FeH}_{band}.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_use_spec}'
                    unflattened_spec_dir = f'{self.out_path}/RASSINE_PHX/{self.phx_flux_file[tt]}'
                    ssp.final_flux_spec(
                        model_type, unflattened_spec_dir, sim_spec_path, wvmin, wvmax, 
                        Vel_bin_size, plot, flatten=False)
                    final_spec = pd.read_hdf(sim_spec_path)

                # --- Read SYNmast Spectra ---
                elif model_type.lower() == 'synmast':
                    # prf_dir = syn_model_path[model_type]
                    unflattened_spec_dir = ssp.prep_synmast_flux_spec(
                        tt, self.logg, self.FeH, self.out_path)
                    sim_spec_name = f'SimUse_Synmast_{tt}_{self.logg}_{self.FeH}_{band}_flux.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_spec_name}'
                    ssp.final_flux_spec(
                        model_type, unflattened_spec_dir, sim_spec_path, wvmin, wvmax, 
                        Vel_bin_size, plot, flatten=False)
                    final_spec = pd.read_hdf(sim_spec_path)

                else:
                    logger.error('\n[ERROR] Can only handle PHX and Synmast spec!\n')
                    continue

                # Store velocity and flux arrays for this band/temp/model
                spec = {
                    f'{band}_{model_type}_{intORflux}_{tt:1.0f}_v': final_spec['v'].to_numpy(),
                    f'{band}_{model_type}_{intORflux}_{tt:1.0f}_f': final_spec['f'].to_numpy()
                }
                self.flux_spec.update(spec)

    def find_fluxspec_flatten(self, Vel_bin_size, plot, syn_model_path=None, fit_temp=None):
        """
        Prepare and flatten synthetic spectra for all bands and temperatures.

        Args:
            Vel_bin_size (float): Velocity bin size for spectra.
            plot (bool): If True, plot intermediate results.
            syn_model_path (dict): Paths to synthetic models.
            fit_temp_range (tuple, optional): Range of temperatures for fitting.
        """
        self.out_path = f'{os.getcwd()}/syn_spec'

        # Build temperature list for all components
        if fit_temp is None:
            temp_list_org = np.concatenate([[self.Temp], self.Tspot, self.Tfaculae])
        else:
            # fitting_photo_T = np.arange(fit_temp_range[0], fit_temp_range[1]+1, 100)
            temp_list_org = np.concatenate([[self.Temp], self.Tspot, self.Tfaculae, [fit_temp]])
        temp_list_org = np.unique(temp_list_org)

        logger.info(f'\nChecking model spectra for temperatures {temp_list_org} K')
        self.dw_phx_flux(temp_list_org, alpha=0)
        self.spec_flatten = {}

        wv_regions = list(self.bands2use.keys())
        for band in wv_regions:
            line_cen = self.bands2use[band][0]
            delta_wv = self.bands2use[band][1]
            rvORphoto = self.bands2use[band][2]
            intORflux = self.bands2use[band][3]
            model_type = self.bands2use[band][4]

            # Build temperature list for this band
            if fit_temp is None:
                if intORflux == 'intensity':
                    temp_list = np.array([self.Temp])
                elif intORflux == 'flux':
                    temp_list = temp_list_org.copy()
            else:
                if intORflux == 'intensity':
                    temp_list = np.concatenate([[self.Temp], [fit_temp]])
                elif intORflux == 'flux':
                    temp_list = np.concatenate([temp_list_org.copy(), [fit_temp]])

            # Only process RV bands
            if rvORphoto != 'rv':
                continue

            wvmin, wvmax = int(line_cen - delta_wv), int(line_cen + delta_wv)

            for tt in temp_list:
                # --- Read PHX SYN Spectra ---
                if model_type.lower() == 'phx':
                    # 1. Flatten PHX spectra at a large wavelength range (3000 - 30000 AA)
                    flat_spec_dir = ssp.flatten_spec(
                        syn_model_path[model_type], self.out_path, self.phx_flux_file[tt])
                    sim_use_spec = f'SimUse_PHX_ACES_{tt}_{self.logg}_{self.FeH}_{band}_flatten.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_use_spec}'
                    # 2. Cut and convert to velocity space
                    ssp.final_flux_spec(
                        model_type, flat_spec_dir, sim_spec_path, wvmin, wvmax,
                        Vel_bin_size, plot, flatten=True)
                    flatten_spec = pd.read_hdf(sim_spec_path)
                # --- Read Synmast Spectra (flattened using PHX as template) ---
                elif model_type.lower() == 'synmast':
                    flat_spec_dir = ssp.flatten_spec(
                        syn_model_path['phx'], self.out_path, self.phx_flux_file[tt])
                    sim_use_spec = f'SimUse_PHX_ACES_{tt}_{self.logg}_{self.FeH}_{band}_flatten.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_use_spec}'
                    ssp.final_flux_spec(
                        'phx', flat_spec_dir, sim_spec_path, wvmin, wvmax,
                        Vel_bin_size, plot, flatten=True)
                    flatten_spec = pd.read_hdf(sim_spec_path)
                else:
                    logger.error('\n[ERROR] Can only handle PHX and Synmast spec!\n')
                    continue

                # Store velocity and flux arrays for this band/temp/model
                spec = {
                    f'{band}_{model_type}_{intORflux}_{tt:1.0f}_v': flatten_spec['v'].to_numpy(),
                    f'{band}_{model_type}_{intORflux}_{tt:1.0f}_f': flatten_spec['f'].to_numpy()
                }
                self.spec_flatten.update(spec)

    def find_intensityspec(self, Vel_bin_size, plot, syn_model_path=None):
        """
        Find and store intensity spectra for the star, spot, and plage at different temperatures.

        Args:
            Vel_bin_size (float): Velocity bin size for spectra.
            plot (bool): If True, plot intermediate results.
            syn_model_path (dict): Paths to synthetic models.
        """
        # Build temperature list for all components (star, spot, plage)
        temp_list = np.concatenate([[self.Temp], self.Tspot, self.Tfaculae])
        temp_list = np.unique(temp_list)
        self.int_spec = {}

        wv_regions = list(self.bands2use.keys())
        for band in wv_regions:
            line_cen = self.bands2use[band][0]
            delta_wv = self.bands2use[band][1]
            intORflux = self.bands2use[band][3]
            model_type = self.bands2use[band][4]

            # Only process bands that require intensity spectra
            if intORflux != 'intensity':
                continue

            wvmin, wvmax = int(line_cen - delta_wv), int(line_cen + delta_wv)

            for tt in temp_list:
                # --- Read PHX SYN Intensity Spectra ---
                if model_type.lower() == 'phx':
                    unflattened_spec_dir = ssp.prep_phx_intensity_spec(
                        tt, self.logg, self.FeH, 0, self.out_path)
                    sim_spec_name = f'PhotoUse_phx_{tt}_{self.logg}_{self.FeH}_{band}.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_spec_name}'
                    ssp.final_intensity_spec(
                        model_type, unflattened_spec_dir, sim_spec_path, wvmin, wvmax,
                        Vel_bin_size, plot, flatten=False)
                    final_spec = pd.read_hdf(sim_spec_path)

                # --- Read SYNmast Intensity Spectra ---
                elif model_type.lower() == 'synmast':
                    prf_dir = syn_model_path[model_type]
                    unflattened_spec_dir = ssp.prep_synmast_intensity_spec(
                        tt, self.logg, self.FeH, self.out_path, prf_dir)
                    sim_spec_name = f'SimUse_Synmast_{tt}_{self.logg}_{self.FeH}_{band}.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_spec_name}'
                    ssp.final_intensity_spec(
                        model_type, unflattened_spec_dir, sim_spec_path, wvmin, wvmax,
                        Vel_bin_size, plot, flatten=False)
                    final_spec = pd.read_hdf(sim_spec_path)

                else:
                    print('\n[ERROR] Can only handle PHX and Synmast spec!\n')
                    continue

                # Extract mu grid and intensity columns
                mu_s = [float(col.split('_')[1]) for col in final_spec.columns if col.startswith("mu_")]
                mu_cols = [col for col in final_spec.columns if col.startswith("mu_")]
                df_mu = final_spec[mu_cols]
                spec = {
                    f'{band}_{model_type}_{intORflux}_{tt:1.0f}_v': final_spec['v'].to_numpy(),
                    f'{band}_{model_type}_{intORflux}_{tt:1.0f}_f': df_mu.to_numpy(),
                }
                self.int_spec.update(spec)

            # Store mu grid for this model type
            self.int_spec[f'mu_{model_type}'] = np.array(mu_s, dtype=float)

    def dw_phx_flux(self, temp_list, alpha=0):
        """
        Download and prepare PHOENIX flux spectra for a range of temperatures.

        Args:
            temp_list (array): Array of temperatures.
            alpha (float): Microturbulence parameter.
        """
        self.phx_flux_file = {}
        for tt in temp_list:
            if (tt > 7000) | (tt < 2300) | (tt is None):
        
                raise Exception(
                    f'   [ERROR] Rassine normalization cfg not tuned for {tt}K\n'
                    '   Please check the temperature range in the config file!'
                )
            else:
                phx_flux_file = ssp.get_phx_flux_spec(
                    tt, self.logg, self.FeH, alpha, self.phx_path, self.out_path)
                self.phx_flux_file[tt] = phx_flux_file
                

class Active_region:
    def __init__(self, long, lat, size, active_region_type, check=False):
        """
        Initialize an active region (spot or plage).

        Args:
            long (float): Longitude [deg]
            lat (float): Latitude [deg]
            size (float): Spot radius [Rstar]
            active_region_type (int): 0 if spot, 1 if plage
            check (bool, optional): 1 if the active region is considered, 
                0 if not. Defaults to False.

        Notes:
            For size: Active region's size [Rstar]. 
            If you want a spot coverage S1 (%) in area of the visible hemisphere, 
            you have to put here sqrt(2*S1), because:
                S1 = Area_spot / Area_visible_hemisphere 
                   = pi*(size1*Rstar)**2 / (2*pi*Rstar**2) 
                   = size1**2 / 2 
                -> size1 = sqrt(2*S1).
            For 0.1% of the visible hemisphere, put 0.045
        """
        self.long = long    
        self.lat  = lat     
        self.s    = size    
        self.active_region_type = active_region_type 
        if self.active_region_type == 0:
            self.active_region_type_str = 'spot'  # active region type as string
        else:
            self.active_region_type_str = 'plage' # active region type as string
        self.check = check                           

    def calc_maps(self, star, grid, nrho):
        """
        Calculate the visibility of the active region.

        Args:
            star (Star): Star object with basic info.
            grid (int): Grid resolution NxN.
            nrho (int): Resolution for the spot's circumference.
        """
        # Calculates the initial coordinates of the active region circumference
        xyz = stsp.spot_location_init(self.s, self.long, self.lat, star.incl, nrho) 
        self.xyz = xyz
        
        # Calculates the coordinates of the active region circumference at phase psi
        xyz2 = stsp.get_spot_phase_coord(xyz, star.incl, nrho, star.psi) 
        self.xyz2 = xyz2

        # Check if the active region is visible at phase psi
        # and return the min and max y and z values for the circumference
        (self.visible, self.iminy, self.iminz, self.imaxy, self.imaxz) =\
            stsp.spot_area(xyz2, nrho, grid) 


class Spec:
    def __init__(self, star, band, temperature, intORflux, model_type):
        """
        Spectrum object for a given band and temperature.

        Args:
            star (Star): Star object containing spectral data.
            band (str): Band name.
            temperature (float): Temperature [K].
            intORflux (str): 'flux' or 'intensity'.
            model_type (str): Model type ('phx' or 'synmast').

        Attributes:
            rv (np.ndarray): Radial velocity grid.
            intensity (np.ndarray): Intensity or flux array.
            mu (np.ndarray): Mu grid (for intensity spectra only).
            width (float): Maximum velocity.
            step (float): Velocity step size.
            n (int): Number of velocity points.
            n_v (int): Number of points for extrapolated CCF.
            v_interval (float): Half-width of the extrapolated CCF in velocity.
        """
        # Select the correct spectrum arrays based on type
        if intORflux == 'flux':
            self.rv = star.flux_spec[f'{band}_{model_type}_{intORflux}_{temperature:1.0f}_v']
            self.intensity = star.flux_spec[f'{band}_{model_type}_{intORflux}_{temperature:1.0f}_f']
        elif intORflux == 'intensity':
            self.rv = star.int_spec[f'{band}_{model_type}_{intORflux}_{temperature:1.0f}_v']
            self.intensity = star.int_spec[f'{band}_{model_type}_{intORflux}_{temperature:1.0f}_f']
            self.mu = star.int_spec[f'mu_{model_type}']

        self.width = np.max(self.rv)
        self.step = self.rv[1] - self.rv[0]
        self.n = len(self.rv)  # Number of points in the CCF

        # Calculate n_v: number of points needed to cover rotation broadening
        # Ensure n_v is odd for symmetry
        n_v_est = self.n + round(((1.1 * star.vrot) / self.step) * 2)
        if np.round(((1.1 * star.vrot) / self.step) * 2) % 2 == 0:
            self.n_v = int(n_v_est)
        else:
            self.n_v = int(n_v_est + 1)

        # v_interval: half the velocity range of the extrapolated CCF
        # (n_v-1) intervals, each of size self.step
        self.v_interval = self.step * (self.n_v - 1) / 2.

def detect_peaks(
        x, mph=None, mpd=1, threshold=0, edge='rising',
        kpsh=False, valley=False, show=False, ax=None):
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
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # Find indices of all peaks
    dx = x[1:] - x[:-1]
    # Handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # Remove peaks near NaNs
    if ind.size and indnan.size:
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # First and last values cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # Remove peaks below minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # Remove peaks with neighbors too close in height
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # Remove small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # Keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # Remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        # _plot(x, mph, mpd, threshold, edge, valley, ax, ind)
    return ind

def a0cont(a0vcut, plot=False, poly_order=1, spline_k=1, mpd=100):
    """
    Performs first-pass fit to blaze shape of telluric standard spectrum.

    Args:
        a0vcut (array): Flux of telluric standard spectrum.
        plot (bool): If True, plot the result.
        poly_order (int): Polynomial order for initial fit.
        spline_k (int): Spline degree.
        mpd (int): Minimum peak distance for peak detection.

    Returns:
        continuum (np.ndarray): Best fit blaze model.
        peaks (np.ndarray): Indices of detected peaks.
    """
    x = np.arange(len(a0vcut))

    # Detect peaks that are at least separated by minimum peak distance
    peaks = detect_peaks(a0vcut, mpd=mpd)
    xtimes = 1

    for _ in range(xtimes):
        mask = np.ones(len(peaks), dtype=bool)

        # Fit a polynomial to the detected peaks
        f = np.polyfit(x[peaks], a0vcut[peaks], poly_order)
        q = np.poly1d(f)
        residual = a0vcut[peaks] - q(x[peaks])
        MAD = np.median(np.abs(residual - np.median(residual)))

        # Mask out outliers and low points
        mask[(a0vcut[peaks]/np.nanmedian(a0vcut) < .1)] = False
        mask[(a0vcut[peaks] < q(x[peaks])-2*MAD)] = False
        mask[(a0vcut[peaks] > q(x[peaks])+6*MAD)] = False
        peaks = peaks[mask]

    c = 0
    for smoothing in np.arange(1e6, 1e8, 1e6):
        # Fit a spline to the filtered peaks
        f = interpolate.UnivariateSpline(
            x[peaks], a0vcut[peaks], k=spline_k, s=smoothing)
        continuum = f(x)
        peaks2 = detect_peaks(continuum)
        if len(peaks2) == 1:
            c += 1
        if c == 2:
            break

    # If not converged, try larger smoothing
    if smoothing == 99000000.0:
        for smoothing in np.arange(1e8, 1e10, 1e8):
            f = interpolate.UnivariateSpline(
                x[peaks], a0vcut[peaks], k=spline_k, s=smoothing)
            continuum = f(x)
            peaks2 = detect_peaks(continuum)
            if len(peaks2) == 1:
                c += 1
            if c == 2:
                break

    return continuum, peaks

def flatten_ccf_model_spec(vel, flux, band, RVandEWR_dir, plot):
    """
    Flatten a model spectrum by dividing by its continuum, estimated via spline fit to peaks.

    Args:
        vel (array): Velocity grid.
        flux (array): Flux array.
        band (str): Band name.
        RVandEWR_dir (str): Output directory for plots.
        plot (bool): If True, save a plot of the flattening process.

    Returns:
        np.ndarray: Flattened flux array.
    """
    _mpd = int(len(vel) * 0.04)
    cont, peaks = a0cont(flux, plot=False, poly_order=2, spline_k=2, mpd=_mpd)
    fluxp_flat = flux / cont
    fluxp_flat = fluxp_flat / np.nanpercentile(fluxp_flat, 99)

    if plot == 1:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3), facecolor='white', dpi=250)
        ax.plot(vel, flux, '-k', lw=1, label='Org. model spec', alpha=0.6)
        ax.plot(vel, fluxp_flat, '-', c='tab:red', lw=0.8, label='Flatten model spec')
        ax.plot(vel[peaks], flux[peaks], 'o', c='tab:blue', ms=4, zorder=5, label='peaks')
        ax.plot(vel, cont, '-', c='tab:orange', lw=1.)
        ax.axhline(1, ls='-', lw=0.5, color='k', alpha=0.3)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.legend(loc='lower right', fontsize='x-small', ncol=4)
        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Flux')
        ax.set_ylim(0.4, 1.05)
        fig.savefig(
            f'./outputs/{RVandEWR_dir}/normal_spec/flatten_{band}_CCF_template.png', bbox_inches='tight'
        )
        plt.close(fig)
    return np.array(fluxp_flat)

def flatten_model_spec(
        Star, flux_phi, visible_phi, band, intORflux, model_type, 
        RVandEWR_dir, plot):
    """
    Flatten a set of model spectra for all phases.

    Args:
        Star (Star): Star object.
        flux_phi (array): 2D array of flux for each phase.
        visible_phi (array): Array of phases.
        band (str): Band name.
        intORflux (str): 'flux' or 'intensity'.
        model_type (str): Model type.
        RVandEWR_dir (str): Output directory for plots.
        plot (bool): If True, save plots for each phase.

    Returns:
        np.ndarray: Array of flattened spectra for all phases.
    """
    if intORflux == 'flux':
        vel = Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']
    elif intORflux == 'intensity':
        vel = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']

    nomral_spec_box = []
    for i in range(len(visible_phi)):
        flux = flux_phi[i]
        _mpd = int(len(vel) * 0.04)
        cont, peaks = a0cont(flux, plot=False, poly_order=2, spline_k=2, mpd=_mpd)
        fluxp_flat = flux / cont
        fluxp_flat = fluxp_flat / np.nanpercentile(fluxp_flat, 99)

        if plot == 1:
            fig, ax = plt.subplots(1, 1, figsize=(7, 3), facecolor='white', dpi=250)
            ax.plot(vel, flux, '-k', lw=1, label='Org. model spec', alpha=0.6)
            ax.plot(vel, fluxp_flat, '-', c='tab:red', lw=0.8, label='Flatten model spec')
            ax.plot(vel[peaks], flux[peaks], 'o', c='tab:blue', ms=4, zorder=5, label='peaks')
            ax.plot(vel, cont, '-', c='tab:orange', lw=1.)
            ax.axhline(1, ls='-', lw=0.5, color='k', alpha=0.3)
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.legend(loc='lower right', fontsize='x-small', ncol=4)
            ax.set_xlabel('Velocity [km/s]')
            ax.set_ylabel('Flux')
            ax.set_ylim(0.4, 1.05)
            fig.savefig(
                f'./outputs/{RVandEWR_dir}/normal_spec/flatten_{band}_{visible_phi[i]:1.2f}.png', bbox_inches='tight'
            )
            plt.close(fig)
        nomral_spec_box.append(fluxp_flat)
    return np.array(nomral_spec_box)

def get_equivalent_width(
        Star, flux_phi, visible_phi, band, intORflux, model_type):
    """
    Calculate the equivalent width ratio (OH/Fe) for a given band and all phases.

    Args:
        Star (Star): Star object.
        flux_phi (array): 2D array of flux for each phase.
        visible_phi (array): Array of phases.
        band (str): Band name.
        intORflux (str): 'flux' or 'intensity'.
        model_type (str): Model type.

    Returns:
        np.ndarray: Array of OH/Fe equivalent width ratios for all phases.
    """
    ew_fe1 = []
    ew_fe2 = []
    ew_oh0 = []

    ew_cal_regions = [
        [1.56230, 1.56290], 
        [1.56290, 1.56335], 
        [1.56335, 1.56395]
    ]

    band_center_wv = Star.bands2use[band][0]

    if intORflux == 'flux':
        vel = Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']
    elif intORflux == 'intensity':
        vel = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']

    c_kms = 299792.458
    del_wv = vel * band_center_wv / c_kms
    wave = (band_center_wv + del_wv) / 1e4  # Convert to microns
    nor_region = np.where((wave > 1.5610) & (wave < 1.5650))[0]
    
    for i in range(len(visible_phi)):
        flux = flux_phi[i]
        nor_level = np.percentile(flux[nor_region], 95)
        flux = flux / nor_level
        # --- EW ---
        for r_udx, rr in enumerate(ew_cal_regions):
            use_r = (wave >= rr[0]) & (wave <= rr[1])
            spec = Spectrum1D(
                spectral_axis=wave[use_r] * u.um, 
                flux=flux[use_r] * u.Unit('erg cm-2 s-1 AA-1')
            )
            ew = equivalent_width(spec)
            if r_udx == 0:
                ew_fe1.append(ew.value)
            elif r_udx == 1:
                ew_oh0.append(ew.value)
            elif r_udx == 2:
                ew_fe2.append(ew.value)

    oh_line = np.array(ew_oh0)
    fe_line = (np.array(ew_fe1) + np.array(ew_fe2)) / 2
    line_ratio = oh_line / fe_line

    return line_ratio

def get_line_depth(
        Star, CCFstar_tot, visible_phi, band, intORflux, model_type):
    """
    Measure the depth and velocity location of Fe and OH lines in the CCF for all phases.

    Args:
        Star (Star): Star object.
        CCFstar_tot (array): 2D array of CCFs for each phase.
        visible_phi (array): Array of phases.
        band (str): Band name.
        intORflux (str): 'flux' or 'intensity'.
        model_type (str): Model type.

    Returns:
        tuple: (depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, depth_l_fe2, depth_l_oh0)
            Each is a list of depths or velocities for Fe1, Fe2, and OH0 lines.
    """
    depth_d_fe1 = []
    depth_d_fe2 = []
    depth_d_oh0 = []
    depth_l_fe1 = []
    depth_l_fe2 = []
    depth_l_oh0 = []

    if intORflux == 'flux':
        vel = Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']
    elif intORflux == 'intensity':
        vel = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']

    for i in range(len(visible_phi)):
        CCFstar_tot_flip = 1 + -1 * CCFstar_tot[i]

        fe_1 = (vel > -150) & (vel < -50)
        oh_0 = (vel > -50) & (vel < 50)
        fe_2 = (vel > 50) & (vel < 250)

        fe_1_loc = np.argmax(CCFstar_tot_flip[fe_1])
        oh_0_loc = np.argmax(CCFstar_tot_flip[oh_0])
        fe_2_loc = np.argmax(CCFstar_tot_flip[fe_2])

        depth_d_fe1.append(CCFstar_tot_flip[fe_1][fe_1_loc])
        depth_d_fe2.append(CCFstar_tot_flip[fe_2][fe_2_loc])
        depth_d_oh0.append(CCFstar_tot_flip[oh_0][oh_0_loc])
        
        depth_l_fe1.append(vel[fe_1][fe_1_loc])
        depth_l_fe2.append(vel[fe_2][fe_2_loc])
        depth_l_oh0.append(vel[oh_0][oh_0_loc])

    return depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, depth_l_fe2, depth_l_oh0


def calculate_activity_signal_spec(
    nrho, grid_size, star, psi, 
    Spec,
    active_regions_number, active_region_box,
    instrumental_resolution, cpu_use, save_dir, band, 
    save_spec_flag, fluxORint, silence_pbar
    ):
    """
    Calculate the integrated spectrum and surface flux for a star with active regions
    (spots/plages) over a grid of phases.

    Args:
        nrho (int): Resolution for the spot's circumference.
        grid_size (int): Model grid resolution NxN.
        star (Star): Star object with parameters and spectra.
        psi (array): Array of rotational phases to simulate.
        Spec (dict): Dictionary of Spec objects for each temperature.
        active_regions_number (int): Number of active regions.
        active_region_box (list): List of arrays: [longitudes, latitudes, sizes, temperatures].
        instrumental_resolution (float): Instrumental resolution (R).
        cpu_use (int): Number of CPUs to use for multiprocessing.
        save_dir (str): Output directory for saving intermediate results.
        band (str): Band name.
        save_spec_flag (int): If True, save intermediate spectra.
        fluxORint (str): 'flux' or 'intensity'.
        silence_pbar (bool): If True, silence progress bars.

    Returns:
        tuple:
            visible_phi (np.array): Array of visible phases.
            integrated_stellar_spectrum (np.array): 2D array of integrated spectra for each phase.
            quite_2d_surface_flux_sum (np.array): 2D array of quiet surface flux.
            final_2d_surface_flux_sum (np.array): 2D array of final surface flux (with active regions).
    """

    # --- Step 1: Compute the quiet photosphere spectrum and surface flux ---
    Spec_photosphere = Spec[int(star.Temp)]
    if fluxORint == 'flux':
        # Get limb-darkening coefficients for the quiet photosphere
        limba1_q = star.LD_coeff[f"{band}_{int(star.Temp)}"][0]
        limba2_q = star.LD_coeff[f"{band}_{int(star.Temp)}"][1]
        # Integrate the quiet photosphere spectrum over the visible disk
        integrated_stellar_spectrum, quite_2d_surface_flux_sum = stsp.itot_flux(
            star.vrot, star.incl, limba1_q, limba2_q, grid_size, 
            Spec_photosphere.rv, Spec_photosphere.intensity, Spec_photosphere.v_interval, 
            Spec_photosphere.n_v, Spec_photosphere.n
        )
    elif fluxORint == 'intensity':    
        # For intensity, limb darkening is handled via the intensity grid (mu)
        integrated_stellar_spectrum, quite_2d_surface_flux_sum = stsp.itot_int(
            star.vrot, star.incl, None, None, grid_size, 
            Spec_photosphere.rv, Spec_photosphere.intensity, Spec_photosphere.v_interval, 
            Spec_photosphere.n_v, Spec_photosphere.n, Spec_photosphere.mu
        )

    # Copy the quiet surface flux for later modification
    final_2d_surface_flux_sum = quite_2d_surface_flux_sum.copy()

    # Unpack active region properties
    (active_region_longs, active_region_lats, 
        active_region_sizes, active_region_Temp) = active_region_box

    # --- Step 2: Loop over all active regions and update the spectrum/flux ---
    logger.info('\nCalculate Activate Regions...')

    # This flag array tracks which grid cells are covered by active regions at each phase
    active_2d_surface_flag_psi = np.zeros(
        (len(psi), grid_size, grid_size), dtype=int)   
            
    for rr in range(active_regions_number):
        logger.info(f'\nCalculate Active Region {rr+1}')

        # Calculate the position of the spot/plage initialized at the disc center
        xyz = stsp.spot_location_init(
            active_region_sizes[rr], active_region_longs[rr], active_region_lats[rr], 
            star.incl, nrho
        )
        
        active_temp = int(active_region_Temp[rr])
        active_intensity = Spec[active_temp].intensity

        # For each phase, scan the grid and compute the "non-contribution" of the active region
        # (i.e., the flux/intensity that is blocked or replaced by the spot/plage)
        # This is subtracted from the quiet spectrum and replaced with the active region's spectrum.
        if fluxORint == 'flux':
            mu_s = None
            limba1 = np.array([
                limba1_q, star.LD_coeff[f"{band}_{active_temp}"][0]])
            limba2 = np.array([
                limba2_q, star.LD_coeff[f"{band}_{active_temp}"][1]])
        elif fluxORint == 'intensity':
            mu_s = Spec[active_temp].mu
            limba1 = None
            limba2 = None

        (intergrated_total_stellar_spectrum, intergrated_total_quiet_spectrum, 
         visible_phi, active_2d_surface_flux_sum, quiet_2d_surface_flux_sum,
         active_2d_surface_flag_psi) =\
                stsp.spot_scan_npsi(
                    xyz, nrho, psi, len(psi), star.vrot, star.incl, 
                    limba1, limba2, grid_size, 
                    Spec_photosphere.rv, Spec_photosphere.intensity, active_intensity, 
                    Spec_photosphere.v_interval, Spec_photosphere.n_v, Spec_photosphere.n, mu_s,
                    active_region_sizes[rr], active_region_longs[rr], 
                    active_region_lats[rr], cpu_use, fluxORint, active_2d_surface_flag_psi,
                    silence_pbar
                )

        # --- Step 2a: Optionally save the intermediate spectra for this region ---
        if save_spec_flag:
            istart = (Spec_photosphere.n_v-len(Spec_photosphere.rv))//2         
            iend   = istart+len(Spec_photosphere.rv)
            spec_qs = integrated_stellar_spectrum[istart:iend]

            spec_qs_duplicated = np.array(intergrated_total_quiet_spectrum)
            spec_qs_duplicated = spec_qs_duplicated[:,istart:iend]

            spec_ss = np.array(intergrated_total_stellar_spectrum)
            spec_ss = spec_ss[:,istart:iend]

            # Save to .npy files for later analysis
            np.save(f'./outputs/{save_dir}/spec/{band}_qs_{rr+1}.npy', spec_qs)
            np.save(f'./outputs/{save_dir}/spec/{band}_ss_{rr+1}.npy', spec_ss)
            np.save(f'./outputs/{save_dir}/spec/{band}_qsdup_{rr+1}.npy', spec_qs_duplicated)
            vel = np.array(Spec_photosphere.rv)
            np.save(f'./outputs/{save_dir}/spec/{band}_vel_{rr+1}.npy', vel)
        
        # --- Step 2b: Update the integrated spectrum and surface flux ---
        # Subtract the quiet region's contribution and add the active region's contribution
        integrated_stellar_spectrum  = integrated_stellar_spectrum \
            - np.array(intergrated_total_quiet_spectrum) \
            + np.array(intergrated_total_stellar_spectrum)  
        final_2d_surface_flux_sum = final_2d_surface_flux_sum \
            - np.array(quiet_2d_surface_flux_sum) \
            + np.array(active_2d_surface_flux_sum)

    # --- Step 3: Truncate the extrapolated spectrum to match the original velocity grid ---
    # The CCF is extrapolated to cover rotation broadening; here we cut it back to the original range.
    istart = (Spec_photosphere.n_v-len(Spec_photosphere.rv))//2
    iend   = istart+len(Spec_photosphere.rv)
    integrated_stellar_spectrum = integrated_stellar_spectrum[:, istart:iend]

    # --- Step 4: Normalize the spectra for each phase (except for photometry bands) ---
    if band != 'photo': 
        integrated_stellar_spectrum = np.array(
            [f/np.percentile(f,99) for f in integrated_stellar_spectrum], 
            dtype=np.float64
        )
    
    # --- Step 5: Convolve with the instrumental profile if needed ---
    # This simulates the effect of finite spectrograph resolution.
    if (instrumental_resolution != 0) and (band != 'photo'):
        integrated_stellar_spectrum = _apply_instrumental_broadening(
            instrumental_resolution, Spec_photosphere.rv, integrated_stellar_spectrum)

    return (visible_phi, integrated_stellar_spectrum, quite_2d_surface_flux_sum, 
            final_2d_surface_flux_sum)


def _apply_instrumental_broadening(inst_reso, vel_grid, spec_flux):
    """
    Apply instrumental broadening to a spectrum by convolving it with a Gaussian kernel.

    Args:
        inst_reso (float): Instrumental resolution (R = lambda / delta_lambda).
        vel_grid (np.ndarray): Radial velocity grid in km/s.
        spec_flux (np.ndarray): 2D array of spectra (N_phases x N_vel).

    Returns:
        np.ndarray: Broadened spectra (same shape as input).
    """
    c_kms = 299792.458  # speed of light in km/s

    fwhm = c_kms / inst_reso
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Gaussian kernel on the RV grid
    kernel = np.exp(-vel_grid**2 / (2 * sigma**2))

    broadened_spec = np.empty_like(spec_flux)
    for i in range(len(spec_flux)):
        # Convolve (1 - flux) to preserve line depth, then invert back
        convolved = signal.convolve(1 - spec_flux[i], kernel, mode='same')
        min_val = np.min(spec_flux[i])
        max_val = np.max(convolved)
        broadened_spec[i] = 1 - convolved * (1 - min_val) / max_val

    return broadened_spec


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

def cal_ccf_rv(
        star_tot, Star, visible_phi, band, intORflux, model_type, rv_ccf_pars, prf_dir, 
        cpu_use, RVandEWR_dir, Vel_bin_size, plot, silence_pbar, save_spec_flag):
    """
    Calculate the RV shift for each phase by cross-correlating with a template.

    Args:
        star_tot (np.ndarray): 2D array of spectra for all phases.
        Star (Star): Star object.
        visible_phi (np.ndarray): Array of phases.
        band (str): Band name.
        intORflux (str): 'flux' or 'intensity'.
        model_type (str): Model type.
        rv_ccf_pars (tuple): Parameters for CCF template (Teff, logg, FeH, vsini, resolution).
        prf_dir (str): Directory for template profiles.
        cpu_use (int): Number of CPUs for parallelization.
        RVandEWR_dir (str): Output directory for CCF plots/templates.
        Vel_bin_size (float): Velocity bin size for CCF.
        plot (bool): If True, plot CCFs and templates.
        silence_pbar (bool): If True, silence progress bars.
        save_spec_flag (int): If True, save CCF templates.

    Returns:
        np.ndarray: Array of RV shifts (one per phase).
    """
    visible_phi = np.array(visible_phi)
    if Star.ccf_template_photosphereORnot == 1:
        ccf_tempalate_temp = int(np.round(Star.Temp, -2))
    else:
        ccf_tempalate_temp = Star.ccf_temp

    # Get velocity grid for the spectra
    if intORflux == 'flux':
        vel = Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']
    elif intORflux == 'intensity':
        vel = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']
    vel = np.array([stsp.rndup(i, 4) for i in vel])

    # --- Get CCF template ---
    line_cen = Star.bands2use[band][0]
    delta_wv = Star.bands2use[band][1]
    wvmin, wvmax = (line_cen-delta_wv, line_cen+delta_wv)
    wvmin = int(wvmin); wvmax = int(wvmax)

    # Convert velocity grid to wavelength grid
    c_kms = 299792.458
    w_center = (wvmin + wvmax) / 2
    wave = vel * w_center / c_kms + w_center

    ccf_tt, ccf_logg, ccf_FeH, ccf_vsini, ccf_resolution = rv_ccf_pars

    # Use user-given parameters to generate CCF template if provided
    if ccf_tt != 99999:
        spec_base_name = f'mc_{ccf_tt:1.0f}{ccf_logg*100:1.0f}+{ccf_FeH*100:03.0f}_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf'
        spec_dir = f'{prf_dir}/{spec_base_name}'
        sim_spec_name = f'RVCCFUse_Synmast_spec_{ccf_tt:1.0f}_{ccf_logg:1.2f}_{ccf_FeH:1.2f}_vsini{ccf_vsini:1.02f}_res{ccf_resolution:1.0f}_wv{wvmin}_{wvmax}_flatten.p'
        sim_spec_path = f'{Star.out_path}{sim_spec_name}'

        ssp.final_flux_spec(
            'ccf_temp', spec_dir, sim_spec_path, wvmin, wvmax, 
            Vel_bin_size, flatten=True, pars=rv_ccf_pars)
        _temp = pd.read_pickle(sim_spec_path)
        spec_temp = _temp['f'].to_numpy()
    else:
        # Use spot coverage to estimate Teff for CCF template
        vel_temp = Star.spec_flatten[f'{band}_{model_type}_{intORflux}_{ccf_tempalate_temp:1.0f}_v']
        spec_temp = Star.spec_flatten[f'{band}_{model_type}_{intORflux}_{ccf_tempalate_temp:1.0f}_f']

        wave_temp = vel_temp * w_center / c_kms + w_center
        # Apply vsini broadening
        wave_temp, spec_temp = rotint(wave_temp, spec_temp, Star.vsini)
        # Apply instrumental broadening
        hwhm_list = wave_temp / Star.inst_res / 2
        spec_temp = macbro_dyn(wave_temp, spec_temp, hwhm_list)

        vel_temp = c_kms * (wave_temp - w_center) / w_center
        spec_temp = flatten_ccf_model_spec(
            wave_temp, spec_temp, band, RVandEWR_dir, plot)

        # Save CCF template if requested
        if save_spec_flag == 1:
            np.save(f'./outputs/{RVandEWR_dir}/ccf_template/{band}_ccf_template.npy', spec_temp)
            np.save(f'./outputs/{RVandEWR_dir}/ccf_template/{band}_ccf_template_wave.npy', wave_temp)

    # --- Determine CCF RV range and step ---
    ccf_step = Vel_bin_size/100 * 2  # x2 for faster forward fitting

    if len(visible_phi) <= 10:
        # For small number of phases, use a fixed RV window
        _edge = 40 
        clip_obs_edges = _edge / np.diff(vel_temp)[0]
        clip_obs_edges = int(clip_obs_edges*5)
        
        func = partial(
            mc_ccf, wave, star_tot, wave_temp, spec_temp, 
            -_edge, _edge, ccf_step, clip_obs_edges, visible_phi)
        outs = pqdm(np.arange(len(star_tot)), func, n_jobs=cpu_use)

        drvs = np.arange(-_edge, _edge, ccf_step)
        ccf_rv_shift = get_ccf_peak(np.array(outs), drvs, visible_phi, band)
        ccf_rv_shift = ccf_rv_shift - np.nanmin(ccf_rv_shift)
    else:
        # For many phases, do trial CCFs to estimate RV window
        _loc0 = np.argmin(np.abs(visible_phi-0.0))
        _loc1 = np.argmin(np.abs(visible_phi-0.15))
        _loc2 = np.argmin(np.abs(visible_phi-0.2))
        _loc3 = np.argmin(np.abs(visible_phi-0.25))
        _loc4 = np.argmin(np.abs(visible_phi-0.3))
        _loc5 = np.argmin(np.abs(visible_phi-0.5))
        trial_locs = np.array([_loc0, _loc1, _loc2, _loc3, _loc4, _loc5], dtype=int)

        _edge = 10
        clip_obs_edges = _edge / np.diff(vel_temp)[0]
        clip_obs_edges = int(clip_obs_edges*5)
        
        func = partial(
            mc_ccf, wave, star_tot[trial_locs], wave_temp, spec_temp, 
            -_edge, _edge, ccf_step, clip_obs_edges, visible_phi)
        outs = pqdm(
            np.arange(len(trial_locs)), func, n_jobs=6, disable=silence_pbar)

        drvs = np.arange(-_edge, _edge, ccf_step)
        _rv_shift = get_ccf_peak(
            np.array(outs), drvs, visible_phi[trial_locs], band) 
        logger.debug('_rv_shift', _rv_shift)

        ccf_bounds = np.round(_rv_shift, decimals=3)
        ccf_bound = np.nanmax(np.abs(ccf_bounds))
        logger.debug(f'org ccf_bound: {ccf_bound}')
        
        if ccf_bound <= 0.02:
            ccf_rv_shift = np.zeros(len(star_tot), dtype=np.float64)
            ccf_rv_shift[:] = np.nan
            return ccf_rv_shift
        
        elif (ccf_bound > 0.02) and (ccf_bound<=0.5):
            ccf_bound = 2
            ccf_step = ccf_step        
        elif (ccf_bound > 0.5) and (ccf_bound<=5):
            ccf_bound = 10
            ccf_step = ccf_step*10
        else:
            ccf_bound = ccf_bound * 4
            ccf_step = ccf_step*100

        logger.debug(f'ccf_bound: {ccf_bound}, ccf_step: {ccf_step}')
        clip_obs_edges = (ccf_bound) / np.diff(vel_temp)[0]
        clip_obs_edges = int(clip_obs_edges*5)

        func = partial(
            mc_ccf, wave, star_tot, wave_temp, spec_temp, 
            -ccf_bound, ccf_bound, ccf_step, clip_obs_edges, visible_phi)
        ccf_rv_shift_box = pqdm(
            np.arange(len(star_tot)), func, n_jobs=cpu_use, disable=silence_pbar)

        drvs_full = np.arange(-ccf_bound, ccf_bound, ccf_step)
        ccf_rv_shift = get_ccf_peak(np.array(ccf_rv_shift_box), drvs_full, visible_phi, band)
        ccf_rv_shift = ccf_rv_shift - np.nanmean(ccf_rv_shift)

    return ccf_rv_shift


def get_photometric(
        integrated_stellar_spectrum, Star, visible_phi, band, intORflux, model_type):
    """
    Calculate photometric light curves (magnitude and normalized flux) for each filter.

    Args:
        integrated_stellar_spectrum (np.ndarray): 2D array of integrated spectra for each phase.
        Star (Star): Star object containing filter and band info.
        visible_phi (np.ndarray): Array of phases.
        band (str): Band name.
        intORflux (str): 'intensity' (required for photometry).
        model_type (str): Model type.

    Returns:
        tuple: (lightcurve_mag_box, lightcurve_flux_box)
            lightcurve_mag_box (np.ndarray): Relative magnitude light curves for each filter.
            lightcurve_flux_box (np.ndarray): Normalized flux light curves for each filter.
    """
    vel = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']

    # Convert velocity grid back to wavelength grid (in Angstroms)
    line_center = Star.bands2use[band][0]
    delta_wave = Star.bands2use[band][1]
    wvmin, wvmax = (line_center - delta_wave, line_center + delta_wave)
    wvmin = int(wvmin)
    wvmax = int(wvmax)

    c_kms = 299792.458
    w_center = (wvmin + wvmax) / 2
    wave = vel * w_center / c_kms + w_center
    wave = wave * unit['AA']

    lightcurve_mag_box = []
    lightcurve_flux_box = []
    for filt in Star.filters:
        logger.info(f'Calculating photometric light curve for {filt} filter...')
        mag_box = []
        flux_box = []
        for flux in integrated_stellar_spectrum:
            flux = flux * unit['erg/s/cm**2/AA']
            f = svo.get_pyphot_filter(filt)
            band_flux = f.get_flux(wave, flux).value
            mag = -2.5 * np.log10(band_flux)  # Vega zero point not applied

            flux_box.append(band_flux)
            mag_box.append(mag)

        # Compute relative magnitude and normalized flux for this filter
        relative_mag = np.array(mag_box) - np.nanmean(mag_box)
        normalized_flux = np.array(flux_box) / np.nanmean(flux_box)
        lightcurve_flux_box.append(normalized_flux)
        lightcurve_mag_box.append(relative_mag)

    return np.array(lightcurve_mag_box), np.array(lightcurve_flux_box)

def grid_resolution_check(grid):
    """
    Check the accuracy of the grid area approximation for the stellar disk.

    The final spectra are approximated by summing all flux in each grid cell,
    so a finer grid better conserves the flux scale. This function checks how
    close the summed area of the grid cells inside the disk is to pi.

    Args:
        grid (int): Grid resolution (number of cells per axis).

    Returns:
        None. Prints a warning if the area is less than 99.9% of pi.
    """
    grid_side_length = 2. / grid  # Grid goes from -1 to 1, so total length is 2

    # Scan each cell on the grid (y-z plane)
    _area = 0.0
    for iy in range(grid):
        y = -1 + iy * grid_side_length  # y from -1 to 1
        for iz in range(grid):
            z = -1. + iz * grid_side_length  # z from -1 to 1
            if (y * y + z * z) <= 1:
                # If the cell is inside the disk, add its area
                _area += grid_side_length ** 2
    accuracy = _area / np.pi
    if accuracy < 0.999:
        logger.warning(
            f'Grid resolution check WARNING.\n'
            f'   Grid {grid}x{grid} has an area/pi={accuracy:1.2%} < 99.9%.\n'
            '   Consider increasing the number of grid to improve accuracy.'
        )


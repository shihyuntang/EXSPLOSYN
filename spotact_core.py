import sys, os, logging
logger = logging.getLogger()
from functools import partial

import numpy as np
# from numpy import cos, sin
# from scipy import signal,interpolate
# from scipy.signal import find_peaks
# from astropy.table import Table
# import astropy.units as u
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pqdm.processes import pqdm


from ldtk import LDPSetCreator, BoxcarFilter
from pyphot import svo, unit

from basic_spec import rotint
from macbro_dynamic import macbro_dyn
import spotact_spot as stsp
# import visualization as vis
import synspecprep as ssp
# from rebin_jv import rebin_jv
# from spotact_utils import _apply_instrumental_broadening, a0cont, save_phase_pngs, coverage_and_teff
import spotact_utils as utils
from spotact_rv import *


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
    cont, peaks = utils.a0cont(flux, plot=False, poly_order=2, spline_k=2, mpd=_mpd)
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
        cont, peaks = utils.a0cont(flux, plot=False, poly_order=2, spline_k=2, mpd=_mpd)
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

    # -------------------------------------------------------------------------
    # --- Step 2-1: Loop over all active regions and phases to make spot map ---
    logger.info('\nCalculate Activate Regions...')

    # Make a cube to store each grid's temperautre at different phases
    # For example, 200x200x45 for 200x200 grid and 45 phases
    active_2d_surface_temperature_psi = np.zeros(
        (len(psi), grid_size, grid_size), dtype=np.int16)   
    
    # assign the temperature of the quiet photosphere to all phases
    # check grid within radius of 1
    y = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    Y, Z = np.meshgrid(y, z, indexing="ij")   # Y.shape = Z.shape = (grid_size, grid_size)

    # 2) Inside-disk mask (unit circle): y^2 + z^2 <= 1
    disk_mask = (Y * Y + Z * Z) <= 1.0        # shape (grid_size, grid_size), bool

    # 3) Prepare a single-layer template with photosphere temp on-disk, sentinel off-disk
    #    Cast to int16; clip just in case
    single_layer = np.where(
        disk_mask, int(star.Temp), 0).astype(np.int16)  # (grid_size, grid_size)

    # 4) Broadcast to all phase-layers
    active_2d_surface_temperature_psi[...] = single_layer[None, ...]
    
    # >>>> Could use numba <<<<<
    for rr in range(active_regions_number):
        # logger.info(f'\nCalculate Active Region {rr+1}')

        # Calculate the circumference position of the spot/plage 
        xyz = stsp.spot_location_init(
            active_region_sizes[rr], active_region_longs[rr], active_region_lats[rr], 
            star.incl, nrho
        )
        
        # spot changes location at difference phases, loop over all phases
        for ipsi in np.arange(len(psi)):
            xyz_psi = stsp.get_spot_phase_coord(xyz, star.incl, nrho, psi[ipsi])
            
            # narrow down the area of the spot/plage to speed up calculations
            _, idx_miny, idx_minz, idx_maxy, idx_maxz = stsp.spot_area(
                xyz_psi, nrho, grid_size)
            
            active_2d_surface_temperature = stsp.spot_scan(
                star.incl, grid_size, active_region_sizes[rr], active_region_longs[rr], 
                active_region_lats[rr], psi[ipsi],
                idx_miny, idx_minz, idx_maxy, idx_maxz, int(active_region_Temp[rr]),
                active_2d_surface_temperature_psi[ipsi], int(star.Temp))
            
            active_2d_surface_temperature_psi[ipsi] = active_2d_surface_temperature
            

    # --- plot for checking the active region temperature map ---
    utils.save_phase_pngs(
        active_2d_surface_temperature_psi,
        y, z,
        outdir="phase_maps",
        sentinel_offdisk=0,
        cmap="inferno",
        add_colorbar=True,
        phase_values=psi,  # optional; remove if you don’t want it in titles
        )

    Teff = utils.teff_obs_rotavg(active_2d_surface_temperature_psi, y, z, sentinel_offdisk=0, scale=1.0)
    logger.info(f'\nAverage Teff of the active regions: {Teff:.1f} K')

    fspot, fspot_phi = utils.fspot_global_from_cube(
        active_2d_surface_temperature_psi, y, z, star.Temp, sentinel_offdisk=0, scale=1.0, mu_min=0.05)
    logger.info(f'\nTotal spot coverage: {fspot:.3%} of the visible hemisphere')
     
    # -------------------------------------------------------------------------
    # --- Step 2-2: run the simulation based on the temperature map ---
    
    if fluxORint == 'flux':
        star_LD_coeff = utils.LD_dict2array(star.LD_coeff, band) # returns temps_tab, coeffs_tab
        Spec_dict = utils.Spec_intensity_dict2array(Spec, fluxORint='flux') # returns temps_tab, spectrum_tab, mu_s_tab
    elif fluxORint == 'intensity':   
        star_LD_coeff = (None, None)
        Spec_dict = utils.Spec_intensity_dict2array(Spec, fluxORint='intensity')

    # For each phase, scan the grid and compute the "non-contribution" of the active region
    # (i.e., the flux/intensity that is blocked or replaced by the spot/plage)
    # This is subtracted from the quiet spectrum and replaced with the active region's spectrum.

    (intergrated_total_stellar_spectrum, intergrated_total_quiet_spectrum, 
        visible_phi, active_2d_surface_flux_sum, quiet_2d_surface_flux_sum) =\
            stsp.spot_scan_npsi(
                xyz, nrho, psi, len(psi), star.vrot, star.incl, star.Temp,
                grid_size, Spec_dict, star_LD_coeff,
                Spec_photosphere.rv, Spec_photosphere.intensity, 
                Spec_photosphere.v_interval, Spec_photosphere.n_v, Spec_photosphere.n,
                active_region_sizes[rr], active_region_longs[rr], 
                active_region_lats[rr], cpu_use, fluxORint, active_2d_surface_temperature_psi,
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
        integrated_stellar_spectrum = utils._apply_instrumental_broadening(
            instrumental_resolution, Spec_photosphere.rv, integrated_stellar_spectrum)

    return (visible_phi, integrated_stellar_spectrum, quite_2d_surface_flux_sum, 
            final_2d_surface_flux_sum)


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
import sys, os, logging, time
logger = logging.getLogger()
from functools import partial
from datetime import timedelta

import numpy as np
from scipy.interpolate import CubicSpline
# from numpy import cos, sin
# from scipy import signal,interpolate
# from scipy.signal import find_peaks
# from astropy.table import Table
# import astropy.units as u
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pqdm.processes import pqdm
# from pqdm.threads import pqdm

from ldtk import LDPSetCreator, BoxcarFilter
from ldtk.ldtk import LDPSet
from pyphot import svo, Filter
from pyphot import config
unit = config.units


from basic_spec import rotint
from macbro_dynamic import macbro_dyn
import spotact_spot as stsp
# import visualization as vis
import syn_specprep as ssp
from rebin_jv import rebin_jv
# from spotact_utils import _apply_instrumental_broadening, a0cont, save_phase_pngs, coverage_and_teff
import spotact_utils as utils
from spotact_rv import *

# ---- For ldtk, avoid mu=0 in resampling ----
def _resample_linear_mu_safe(self, nmu=50, eps=1e-6):
    self.resample(mu=np.linspace(eps, 1.0, nmu))
LDPSet.resample_linear_mu = _resample_linear_mu_safe

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
        self.Temp = np.round(Temp, 4)
        self.Tspot = np.round(Tspot, 4)
        self.Tfaculae = np.round(Tfaculae, 4)
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
        # Normalize inputs once
        line_centers = np.atleast_1d(np.array(line_centers, dtype=float))
        l_widths     = np.atleast_1d(np.array(l_widths, dtype=float))

        # Convert Å -> nm for ldtk
        lc_nm   = line_centers / 10.0
        wid_nm  = l_widths / 10.0

        # Cache filters (BoxcarFilter expects nm)
        filters = [
            BoxcarFilter(f"lc{int(lc_nm[i]//1000)}", lc_nm[i]-wid_nm[i], lc_nm[i]+wid_nm[i])
            for i in range(len(lc_nm))
        ]

        LD_teff_wv_coeff = {}
        # small sampling is fine if you just want a single estimate, not a posterior
        nsamples = 12

        for tt in np.atleast_1d(temperature).astype(float):
            # tiny stds are still needed for LDPSetCreator's sampling
            sc = LDPSetCreator(
                teff=(tt, 1.0),
                logg=(self.logg, 0.01),
                z=(self.FeH, 0.01),
                filters=filters,
                dataset='visir-lowres'
            )

            # Create profiles; our monkeypatch prevents μ=0 crash
            ps = sc.create_profiles(nsamples=nsamples)

            # Get quadratic coefficients: returns (means, covariance)
            qc, _ = ps.coeffs_qd()  # shape: (n_filters, 2) for (u1, u2)

            # Package results
            for i in range(len(line_centers)):
                key = f"w{int(line_centers[i] // 100)}_{np.round(tt, 4)}"
                LD_teff_wv_coeff[key] = qc[i]

            # Robust FeOH aliasing (use tolerance instead of exact float compare)
            _wave_cen = 15557.0 #15631.35  # Å
            locs = np.where(np.isclose(line_centers, _wave_cen, atol=1e-3))[0]
            if locs.size: LD_teff_wv_coeff[f"ewr_FeOH_{np.round(tt, 4)}"] = qc[locs[0]]

            _wave_cen = 15562.5
            locs = np.where(np.isclose(line_centers, _wave_cen, atol=1e-3))[0]
            if locs.size: LD_teff_wv_coeff[f"spec_FeOH_{np.round(tt, 4)}"] = qc[locs[0]]

            _wave_cen = 23138.0
            locs = np.where(np.isclose(line_centers, _wave_cen, atol=1e-3))[0]
            if locs.size: LD_teff_wv_coeff[f"spec_K_CO_{np.round(tt, 4)}"] = qc[locs[0]]

            _wave_cen = 22263.0
            locs = np.where(np.isclose(line_centers, _wave_cen, atol=1e-3))[0]
            if locs.size: LD_teff_wv_coeff[f"spec_K_Ti_{np.round(tt, 4)}"] = qc[locs[0]]

        self.LD_coeff = LD_teff_wv_coeff

    def find_fluxspec(
            self, Vel_bin_size, plot, savehdf=None):
        """
        Find and store flux spectra for the star, spot, and plage at different temperatures.

        Args:
            Vel_bin_size (float): Velocity bin size for spectra.
            plot (bool): If True, plot intermediate results.
            syn_model_path (dict): Paths to synthetic models.
        """
        # Build temperature list for all components (star, spot, plage)
        temp_list = np.unique(
            np.concatenate([[self.Temp], self.Tspot, self.Tfaculae]))
        self.flux_spec = {}

        for band, (line_cen, delta_wv, rvORphoto, intORflux, model_type) in self.bands2use.items():
            # Only process bands that require flux spectra
            if intORflux != 'flux':
                continue

            # Define wavelength window for this band
            wvmin = int(line_cen - delta_wv)
            wvmax = int(line_cen + delta_wv)

            for tt in temp_list:
                # --- Read PHX SYN Spectra ---
                if model_type.lower() == 'phx':
                    sim_use_spec = f'SimUse_PHX_ACES_{tt}_{self.logg}_{self.FeH}_{band}.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_use_spec}'
                    unflattened_spec = f'{self.out_path}/RASSINE_PHX/{self.phx_flux_file[tt]}'
                    final_spec = ssp.final_flux_spec(
                        model_type, unflattened_spec, sim_spec_path, wvmin, wvmax, 
                        Vel_bin_size, plot, flatten=False, savehdf=savehdf)
                    # final_spec = pd.read_hdf(sim_spec_path)

                # --- Read SYNmast Spectra ---
                elif model_type.lower() == 'synmast':
                    # prf_dir = syn_model_path[model_type]
                    unflattened_spec = ssp.prep_synmast_flux_spec(
                        tt, self.logg, self.FeH, self.out_path, savehdf=savehdf)
                    sim_spec_name = f'SimUse_Synmast_{tt}_{self.logg}_{self.FeH}_{band}_flux.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_spec_name}'
                    final_spec = ssp.final_flux_spec(
                        model_type, unflattened_spec, sim_spec_path, wvmin, wvmax, 
                        Vel_bin_size, plot, flatten=False, savehdf=savehdf)
                    # final_spec = pd.read_hdf(sim_spec_path)

                else:
                    logger.error('\n[ERROR] Can only handle PHX and Synmast spec!\n')
                    continue

                # Store velocity and flux arrays for this band/temp/model
                v_arr = final_spec['v'].values.astype(np.float32, copy=False)
                f_arr = final_spec['f'].values.astype(np.float64, copy=False)
                spec = {
                    f'{band}_{model_type}_{intORflux}_{tt:1.4f}_v': v_arr,
                    f'{band}_{model_type}_{intORflux}_{tt:1.4f}_f': f_arr
                    }
                
                self.flux_spec.update(spec)

    def find_fluxspec_flatten(
            self, Vel_bin_size, plot, syn_model_path=None, fit_temp=None, savehdf=None):
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
            temp_list_all = np.unique(
                np.concatenate([[self.Temp], self.Tspot, self.Tfaculae]))
        else:
            temp_list_all = np.unique(
                np.concatenate([[self.Temp], self.Tspot, self.Tfaculae, [fit_temp]]))

        logger.info(f'\nChecking model spectra for temperatures {temp_list_all} K')
        self.dw_phx_flux(temp_list_all, alpha=0)

        self.spec_flatten = {}

        for band, (line_cen, delta_wv, rvORphoto, intORflux, model_type) in self.bands2use.items():
            # Only process RV bands            
            if rvORphoto != 'rv':
                continue

            # Build temperature list for this band
            wvmin = int(line_cen - delta_wv)
            wvmax = int(line_cen + delta_wv)

            # band-specific temperature set
            if fit_temp is None:
                # temp_list = np.array([self.Temp], dtype=np.int32) if intORflux == 'intensity' else temp_list_all
                temp_list = np.array([self.Temp], dtype=np.float64)
            else:
                temp_list = np.array([fit_temp], dtype=np.float64)
                # if intORflux == 'intensity':
                #     temp_list = np.unique(np.array([self.Temp, fit_temp], dtype=np.int32))
                # else:
                #     temp_list = np.unique(np.concatenate([temp_list_all, [fit_temp]]).astype(np.int32))

            for tt in temp_list:
                # --- Read PHX model Spectra ---
                if model_type.lower() == 'phx':
                    # 1. Flatten PHX spectra at a large wavelength range (3000 - 30000 AA)
                    flat_spec_dir = ssp.flatten_phx_with_rassine(
                        syn_model_path[model_type], self.out_path, self.phx_flux_file[tt])
                    
                    sim_use_spec = f'SimUse_PHX_ACES_{tt}_{self.logg}_{self.FeH}_{band}_flatten.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_use_spec}'
                    # 2. Cut and convert to velocity space
                    flatten_spec = ssp.final_flux_spec(
                        model_type, flat_spec_dir, sim_spec_path, wvmin, wvmax,
                        Vel_bin_size, plot, flatten=True, savehdf=savehdf)
                    # flatten_spec = pd.read_hdf(sim_spec_path)

                # --- Read Synmast Spectra >>(All RV ccf use PHX as template, thus here for synmast we also have PHX flattened) ---
                elif model_type.lower() == 'synmast':
                    flat_spec_dir = ssp.flatten_phx_with_rassine(
                        syn_model_path['phx'], self.out_path, self.phx_flux_file[tt])
                    
                    sim_use_spec = f'SimUse_PHX_ACES_{tt}_{self.logg}_{self.FeH}_{band}_flatten.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_use_spec}'
                    flatten_spec = ssp.final_flux_spec(
                        'phx', flat_spec_dir, sim_spec_path, wvmin, wvmax,
                        Vel_bin_size, plot, flatten=True, savehdf=savehdf)
                    # flatten_spec = pd.read_hdf(sim_spec_path)
                else:
                    logger.error('\n[ERROR] Can only handle PHX and Synmast spec!\n')
                    continue

                # Store velocity and flux arrays for this band/temp/model
                v_arr = flatten_spec['v'].values.astype(np.float32, copy=False)
                f_arr = flatten_spec['f'].values.astype(np.float64, copy=False)
                spec = {
                    f'{band}_{model_type}_{intORflux}_{tt:1.4f}_v': v_arr,
                    f'{band}_{model_type}_{intORflux}_{tt:1.4f}_f': f_arr
                    }
                self.spec_flatten.update(spec)

    def find_intensityspec(
            self, Vel_bin_size, plot, syn_model_path=None, savehdf=None):
        """
        Find and store intensity spectra for the star, spot, and plage at different temperatures.

        Args:
            Vel_bin_size (float): Velocity bin size for spectra.
            plot (bool): If True, plot intermediate results.
            syn_model_path (dict): Paths to synthetic models.
        """
        # Build temperature list for all components (star, spot, plage)
        temp_list = np.unique(
                np.concatenate([[self.Temp], self.Tspot, self.Tfaculae]))
        self.int_spec = {}

        for band, (line_cen, delta_wv, rvORphoto, intORflux, model_type) in self.bands2use.items():
            # Only process bands that require intensity spectra
            if intORflux != 'intensity':
                continue

            # Build temperature list for this band
            wvmin = int(line_cen - delta_wv)
            wvmax = int(line_cen + delta_wv)

            for tt in temp_list:
                # --- Read PHX SYN Intensity Spectra ---
                if model_type.lower() == 'phx':
                    unflattened_spec = ssp.prep_phx_intensity_spec(
                        tt, self.logg, self.FeH, 0, self.out_path, savehdf=savehdf)
                    sim_spec_name = f'PhotoUse_phx_{tt}_{self.logg}_{self.FeH}_{band}.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_spec_name}'
                    final_spec = ssp.final_intensity_spec(
                        model_type, unflattened_spec, sim_spec_path, wvmin, wvmax,
                        Vel_bin_size, plot, flatten=False, savehdf=savehdf)
                    # final_spec = pd.read_hdf(sim_spec_path)

                # --- Read SYNmast Intensity Spectra ---
                elif model_type.lower() == 'synmast':
                    prf_dir = syn_model_path[model_type][band.split("_")[-1]]
                    unflattened_spec = ssp.prep_synmast_intensity_spec(
                        tt, self.logg, self.FeH, self.out_path, prf_dir, savehdf=savehdf)
                    sim_spec_name = f'SimUse_Synmast_{tt}_{self.logg}_{self.FeH}_{band}.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_spec_name}'
                    final_spec = ssp.final_intensity_spec(
                        model_type, unflattened_spec, sim_spec_path, wvmin, wvmax,
                        Vel_bin_size, plot, flatten=False, savehdf=savehdf)
                    # final_spec = pd.read_hdf(sim_spec_path)

                else:
                    print('\n[ERROR] Can only handle PHX and Synmast spec!\n')
                    continue

                # Extract mu grid and intensity columns
                mu_cols = [col for col in final_spec.columns if col.startswith("mu_")]
                df_mu = final_spec[mu_cols]
                
                v_arr = final_spec['v'].to_numpy(dtype=np.float32, copy=False)    # (n_v,)
                f_arr = df_mu.to_numpy(dtype=np.float64, copy=False)  # (n_v, n_mu)

                spec = {
                    f'{band}_{model_type}_{intORflux}_{tt:1.4f}_v': v_arr,
                    f'{band}_{model_type}_{intORflux}_{tt:1.4f}_f': f_arr,
                }
                self.int_spec.update(spec)
            
            # Store mu grid for this model type
            mu_s = [float(col.split('_')[1]) for col in final_spec.columns if col.startswith("mu_")]
            self.int_spec[f'mu_{model_type}'] = np.array(mu_s, dtype=np.float32)

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
            stsp.spot_area(xyz2, grid) 


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
            self.rv = star.flux_spec[f'{band}_{model_type}_{intORflux}_{temperature:1.4f}_v']
            self.intensity = star.flux_spec[f'{band}_{model_type}_{intORflux}_{temperature:1.4f}_f']
        elif intORflux == 'intensity':
            self.rv = star.int_spec[f'{band}_{model_type}_{intORflux}_{temperature:1.4f}_v']
            self.intensity = star.int_spec[f'{band}_{model_type}_{intORflux}_{temperature:1.4f}_f']
            self.mu = star.int_spec[f'mu_{model_type}']

        self.width = np.max(self.rv)
        self.step = self.rv[1] - self.rv[0]
        self.n = len(self.rv)  # Number of points in the CCF

        # # Calculate n_v: number of points needed to cover rotation broadening
        # # Ensure n_v is odd for symmetry
        # n_v_est = self.n + round(((1.1 * star.vrot) / self.step) * 2)
        # if np.round(((1.1 * star.vrot) / self.step) * 2) % 2 == 0:
        #     self.n_v = int(n_v_est)
        # else:
        #     self.n_v = int(n_v_est + 1)
        
        # Calculate n_v: number of points needed to cover rotation broadening + CCF shifts
        # Enforce a minimum padding of 20 km/s to prevent CCF 'NaN' failures at large bin sizes
        pad_vel = max(1.1 * star.vrot, 20.0) 
        n_v_est = self.n + round((pad_vel / self.step) * 2)
        
        if np.round((pad_vel / self.step) * 2) % 2 == 0:
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
    Star,
    flux_phi: np.ndarray,
    visible_phi: np.ndarray,
    band: str,
    intORflux: str,
    model_type: str,
    RVandEWR_dir: str,
    plot: bool
) -> np.ndarray:
    """
    Flatten a set of model spectra for all phases by dividing by a spline-
    estimated continuum (found via peak picking), then rescale to the 99th
    percentile. Plots are optional.

    Parameters
    ----------
    Star : Star
    flux_phi : (n_phase, n_vel) array
    visible_phi : (n_phase,) array
    band, intORflux, model_type : str
    RVandEWR_dir : str
    plot : bool (1/0 or True/False)

    Returns
    -------
    flat : (n_phase, n_vel) float64 array
    """
    # --- velocity grid for this band/type ---
    if intORflux == 'flux':
        vel = Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v']
    else:  # 'intensity'
        vel = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v']

    visible_phi = np.asarray(visible_phi)
    n_phase = flux_phi.shape[0]
    n_vel   = vel.shape[0]

    # Precompute mpd once per band (same grid length each phase)
    mpd = int(n_vel * 0.04) if n_vel > 0 else 1

    # Preallocate output
    flat = np.empty((n_phase, n_vel), dtype=np.float64)

    # Plotting helper (so we don’t recreate styles endlessly)
    do_plot = (plot == 1) or (plot is True)

    # Main loop over phases
    for i in range(n_phase):
        flux = np.asarray(flux_phi[i], dtype=np.float64, order='C')
        # continuum via your utils method (peak → spline)
        cont, peaks = utils.a0cont(flux, plot=False, poly_order=2, spline_k=2, mpd=mpd)

        # guard against zeros in continuum
        # (rare, but better than infs if spline dips locally)
        cont = np.where(cont <= 0.0, np.nan, cont)

        spec_flat = flux / cont
        # robust scale to ~continuum=1; clip tiny negatives from numerical noise
        p99 = np.nanpercentile(spec_flat, 99.0)
        if not np.isfinite(p99) or p99 <= 0.0:
            p99 = 1.0
        spec_flat = spec_flat / p99

        # Write into preallocated array
        flat[i, :] = spec_flat

        # optional plot
        if do_plot:
            fig, ax = plt.subplots(1, 1, figsize=(7, 3), facecolor='white', dpi=250)
            ax.plot(vel, flux, '-k', lw=1, label='Org. model spec', alpha=0.6)
            ax.plot(vel, spec_flat, '-', lw=0.8, label='Flatten model spec')
            if peaks.size:
                ax.plot(vel[peaks], flux[peaks], 'o', ms=4, zorder=5, label='peaks')
            ax.plot(vel, cont, '-', lw=1., label='continuum')
            ax.axhline(1, ls='-', lw=0.5, color='k', alpha=0.3)
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.legend(loc='lower right', fontsize='x-small', ncol=4)
            ax.set_xlabel('Velocity [km/s]')
            ax.set_ylabel('Flux')
            ax.set_ylim(0.4, 1.05)
            fig.savefig(
                f'./outputs/{RVandEWR_dir}/normal_spec/flatten_{band}_{visible_phi[i]:1.2f}.png',
                bbox_inches='tight'
            )
            plt.close(fig)

    return flat


def calculate_activity_signal_spec(
    nrho, grid_size, star, psi, 
    Spec,
    active_regions_number, active_region_box,
    instrumental_resolution, cpu_use, save_dir, band, 
    save_spec_flag, fluxORint, silence_pbar, plot_flag,
    *, map_debug_plots: bool = False
):
    """
    Calculate the integrated spectrum and surface flux for a star with active regions
    (spots/plages) over a grid of phases.

    Returns:
        tuple:
            visible_phi (np.ndarray): Array of visible phases.
            integrated_stellar_spectrum (np.ndarray): 2D array of integrated spectra for each phase.
            quite_2d_surface_flux_sum (np.ndarray): 2D array of quiet surface flux.
            final_2d_surface_flux_sum (np.ndarray): 2D array of final surface flux (with active regions).
    """
    # --- Step 1: Quiet photosphere baseline (spectrum + surface flux map) ---
    Spec_photosphere = Spec[np.round(star.Temp, 4)]
    if fluxORint == 'flux':
        u1, u2 = star.LD_coeff[f"{band}_{np.round(star.Temp, 4)}"]
        integrated_stellar_spectrum, quite_2d_surface_flux_sum = stsp.itot_flux(
            star.vrot, star.incl, u1, u2, grid_size, 
            Spec_photosphere.rv, Spec_photosphere.intensity, Spec_photosphere.v_interval, 
            Spec_photosphere.n_v, Spec_photosphere.n
        )
    else:  # 'intensity'
        integrated_stellar_spectrum, quite_2d_surface_flux_sum = stsp.itot_int(
            star.vrot, star.incl, None, None, grid_size, 
            Spec_photosphere.rv, Spec_photosphere.intensity, Spec_photosphere.v_interval, 
            Spec_photosphere.n_v, Spec_photosphere.n, Spec_photosphere.mu
        )

    final_2d_surface_flux_sum = quite_2d_surface_flux_sum.copy()

    # Unpack active region properties
    (active_region_longs, active_region_lats, 
     active_region_sizes, active_region_Temp) = active_region_box

    # --- Step 2-1: Build temperature cube over phases (float32; sentinel=0 off-disk) ---
    nphi = len(psi)
    temp_quiet = star.Temp

    y = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    disk_mask = (Y * Y + Z * Z) <= 1.0

    single_layer = np.where(disk_mask, temp_quiet, 0).astype(np.float32)

    # set active_2d_surface_temperature to float32 to save memory;
    active_2d_surface_temperature_psi = np.empty(
        (nphi, grid_size, grid_size), dtype=np.float32
        )
    active_2d_surface_temperature_psi[...] = single_layer[None, ...]

    logger.info('\nCalculate Active Regions...')
    for rr in range(active_regions_number):
        xyz = stsp.spot_location_init(
            active_region_sizes[rr], active_region_longs[rr], active_region_lats[rr], 
            star.incl, nrho
        )
        for ipsi in range(nphi):
            xyz_psi = stsp.get_spot_phase_coord(xyz, star.incl, nrho, psi[ipsi])
            _, idx_miny, idx_minz, idx_maxy, idx_maxz = stsp.spot_area(xyz_psi, grid_size)
            active_2d_surface_temperature_psi[ipsi] = stsp.spot_scan(
                star.incl, grid_size, active_region_sizes[rr], active_region_longs[rr], 
                active_region_lats[rr], psi[ipsi],
                idx_miny, idx_minz, idx_maxy, idx_maxz, np.round(active_region_Temp[rr], 4),
                active_2d_surface_temperature_psi[ipsi], temp_quiet
            )

    # Optional diagnostics (plots/coverage) — off by default
    if map_debug_plots:
        utils.save_phase_pngs(
            active_2d_surface_temperature_psi,
            y, z,
            outdir="phase_maps",
            sentinel_offdisk=0,
            cmap="inferno",
            add_colorbar=True,
            phase_values=psi,
        )
    
    # --- Step 2-2: Integrate spectra over phases using the temperature cube ---
    if fluxORint == 'flux':
        star_LD_coeff = utils.LD_dict2array(star.LD_coeff, band)        # (temps_tab, coeffs_tab)
        Spec_dict     = utils.Spec_intensity_dict2array(Spec, 'flux')   # (temps_tab, spectrum_tab, _)
    else:
        star_LD_coeff = (None, None)
        Spec_dict     = utils.Spec_intensity_dict2array(Spec, 'intensity')

    (intergrated_total_stellar_spectrum, intergrated_total_quiet_spectrum,
        visible_phi, active_2d_surface_flux_sum, quiet_2d_surface_flux_sum
        ) = stsp.spot_scan_npsi(
            psi, nphi, star.vrot, star.incl, star.Temp,
            grid_size, Spec_dict, star_LD_coeff,
            Spec_photosphere.rv, Spec_photosphere.intensity, 
            Spec_photosphere.v_interval, Spec_photosphere.n_v, Spec_photosphere.n,
            cpu_use, fluxORint, active_2d_surface_temperature_psi,
            silence_pbar
            )

    # --- Step 2a: Optional save of intermediate spectra (quiet vs active) ---
    if save_spec_flag:
        istart = (Spec_photosphere.n_v - len(Spec_photosphere.rv)) // 2
        iend   = istart + len(Spec_photosphere.rv)

        spec_qs = integrated_stellar_spectrum[istart:iend]
        spec_qs_duplicated = np.asarray(intergrated_total_quiet_spectrum)[:, istart:iend]
        spec_ss = np.asarray(intergrated_total_stellar_spectrum)[:, istart:iend]
        vel = np.asarray(Spec_photosphere.rv)

        np.save(f'./outputs/{save_dir}/spec/{band}_qs_1.npy',     spec_qs)
        np.save(f'./outputs/{save_dir}/spec/{band}_ss_1.npy',     spec_ss)
        np.save(f'./outputs/{save_dir}/spec/{band}_qsdup_1.npy',  spec_qs_duplicated)
        np.save(f'./outputs/{save_dir}/spec/{band}_vel_1.npy',    vel)
    
    if (band != 'photo') & (silence_pbar != True):
        # Calculate average Teff and spot coverage fraction per phase
        f_spot, f_cool, f_hot, T4_mean_phase, Teff = utils.coverage_and_teff(
            active_2d_surface_temperature_psi, y, z, star.Temp, star.LD_coeff, band,
            sentinel_offdisk=0, scale=1.0)
        
        logger.info(f'\n  >> Average Teff over phases: {Teff:1.0f} K')
        logger.info(f'  >> Spot coverage fraction over phases: {np.mean(f_spot)*100:1.2f} %')

        star._phase_teff = T4_mean_phase**0.25
        star._phase_fspot = f_spot
        star._phase_fcool = f_cool
        star._phase_fhot  = f_hot
    
    # --- Step 2b: Combine quiet baseline with active-region contributions ---
    integrated_stellar_spectrum = (
        integrated_stellar_spectrum
        - np.asarray(intergrated_total_quiet_spectrum)
        + np.asarray(intergrated_total_stellar_spectrum)
    )
    final_2d_surface_flux_sum = (
        final_2d_surface_flux_sum
        - np.asarray(quiet_2d_surface_flux_sum)
        + np.asarray(active_2d_surface_flux_sum)
    )

    # --- Step 3: Truncate extrapolated spectrum back to original grid ---
    istart = (Spec_photosphere.n_v - len(Spec_photosphere.rv)) // 2
    iend   = istart + len(Spec_photosphere.rv)
    integrated_stellar_spectrum = integrated_stellar_spectrum[:, istart:iend]

    # --- Step 4: Normalize per-phase (non-photo bands) ---
    if band != 'photo': 
        integrated_stellar_spectrum = np.array(
            [f / np.percentile(f, 99) for f in integrated_stellar_spectrum],
            dtype=np.float64
        )

    # --- Step 5: Instrumental broadening ---
    if (instrumental_resolution != 0) and (band != 'photo'):
        integrated_stellar_spectrum = utils._apply_instrumental_broadening(
            instrumental_resolution, Spec_photosphere.rv, integrated_stellar_spectrum
        )

    return (
        visible_phi,
        integrated_stellar_spectrum,
        quite_2d_surface_flux_sum,
        final_2d_surface_flux_sum
    )


def get_photometric(
    integrated_stellar_spectrum: np.ndarray,
    Star,
    visible_phi: np.ndarray,
    band: str,
    intORflux: str,
    model_type: str,
):
    """
    Compute relative magnitudes and normalized fluxes per filter.

    Notes
    -----
    - Assumes `integrated_stellar_spectrum` are INTENSITY spectra integrated
      over the disc for each phase on a UNIFORM velocity grid corresponding
      to `band` (your pipeline ensures that).
    - Vega zeropoint is NOT applied (relative mags only, as before).
    """

    # ---- grid: velocity -> wavelength (Å) ----
    vel = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v']  # (n_v,)
    line_center, delta_wave = Star.bands2use[band][0], Star.bands2use[band][1]
    wvmin = int(line_center - delta_wave)
    wvmax = int(line_center + delta_wave)

    c_kms = 299_792.458
    w_center = (wvmin + wvmax) * 0.5  # Å (float64)
    wave = (vel * w_center / c_kms + w_center) * unit.U('AA')  # Quantity (n_v,)

    # # ---- preload filters once ----
    # # Optional tiny cache on the Star object so repeated calls are cheap
    # if getattr(Star, "_pyphot_filters", None) is None:
    #     Star._pyphot_filters = {f: svo.get_pyphot_filter(f) for f in Star.filters}
    # filters_resolved = [Star._pyphot_filters[f] for f in Star.filters]
    
    # ---- preload filters purely from cache ----
    if getattr(Star, "_pyphot_filters", None) is None:
        Star._pyphot_filters = {}
        cache_dir = './bandpass_cash'

        for f in Star.filters:
            safe_fname = f.replace('/', '_') + '.csv'
            cache_path = os.path.join(cache_dir, safe_fname)

            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    f"Cached filter profile {cache_path} not found. "
                    f"Ensure the initialization block downloaded it successfully."
                )

            # Load strictly from local cache
            # logger.info(f"Loading filter {f} from local cache: {cache_path}") # Optional: comment out to reduce log spam
            
            with open(cache_path, 'r') as ff:
                first_line = ff.readline()
                # Extract dtype from something like "# dtype:photon"
                date_type = first_line.strip().split(':')[1]
                
            # 2. Read the numeric array (np.loadtxt automatically skips lines starting with '#')
            data = np.loadtxt(cache_path, delimiter=',')
                    
            cached_wave = data[:, 0] * unit.U('AA')
            cached_transmit = data[:, 1]

            Star._pyphot_filters[f] = Filter(
                cached_wave, cached_transmit, name=f, dtype=date_type
            )

    filters_resolved = [Star._pyphot_filters[f] for f in Star.filters]

    # ---- pre-allocate outputs ----
    n_phases = integrated_stellar_spectrum.shape[0]
    n_filts  = len(filters_resolved)
    lightcurve_mag_box  = np.empty((n_filts, n_phases), dtype=np.float64)
    lightcurve_flux_box = np.empty((n_filts, n_phases), dtype=np.float64)

    # ---- constant for units ----
    erg_s_cm2_AA = unit.U('erg/s/cm**2/AA')

    # ---- per filter, loop phases ----
    for i, (filt_name, filt_obj) in enumerate(zip(Star.filters, filters_resolved)):
        logger.info(f'Calculating photometric light curve for {filt_name}...')

        # temporary buffers per filter
        mag_buf  = np.empty(n_phases, dtype=np.float64)
        flux_buf = np.empty(n_phases, dtype=np.float64)

        for p in range(n_phases):
            # attach units only once per phase spectrum
            flux_q = integrated_stellar_spectrum[p].astype(np.float64, copy=False) * erg_s_cm2_AA
            band_flux = filt_obj.get_flux(wave, flux_q).value  # scalar float64

            # safe log10 just in case (avoid -inf)
            band_flux = float(band_flux)
            flux_buf[p] = band_flux
            mag_buf[p]  = -2.5 * np.log10(np.clip(band_flux, 1e-300, None))

        # relative mag and normalized flux
        mag_mean  = np.nanmean(mag_buf)
        flux_mean = np.nanmean(flux_buf)

        lightcurve_mag_box[i, :]  = mag_buf  - mag_mean
        lightcurve_flux_box[i, :] = flux_buf / flux_mean

    return lightcurve_mag_box, lightcurve_flux_box


def _pick_trials(visible_phi, n_keep=6):
    """Pick trial indices for the 1st-pass CCF.
    1) Prefer canonical phases; 2) fall back to evenly spaced PHASE targets.
    """
    phi = np.asarray(visible_phi, dtype=np.float64)
    n = phi.size
    n_keep = min(n_keep, n)

    # 1) preferred anchors (clip to actual phi span in case it’s not [0,1] exactly)
    anchors = np.array([0.00, 0.15, 0.20, 0.25, 0.30, 0.50], dtype=np.float64)
    anchors = anchors[(anchors >= phi.min()) & (anchors <= phi.max())]
    idx = [int(np.argmin(np.abs(phi - t))) for t in anchors]
    idx = np.unique(np.asarray(idx, dtype=int))

    # 2) fallback: evenly spaced *phase* targets, then map to nearest indices
    if idx.size < n_keep:
        targets = np.linspace(phi.min(), phi.max(), n_keep)
        extra = [int(np.argmin(np.abs(phi - t))) for t in targets]
        idx = np.unique(np.concatenate([idx, extra]))

    # Keep at most n_keep, sorted
    return np.sort(idx[:n_keep])


def cal_ccf_rv(
        star_tot, Star, visible_phi, band, intORflux, model_type, rv_ccf_pars, prf_dir, 
        cpu_use, RVandEWR_dir, Vel_bin_size, plot, silence_pbar, save_spec_flag, outdir):
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
    visible_phi = np.asarray(visible_phi, dtype=np.float64)
    nphi = visible_phi.size

    # --- Obs velocity grid on which star_tot lives ---
    if intORflux == 'flux':
        vel = Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v']
    else:
        vel = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v']
    vel = np.asarray(vel, dtype=np.float64)
    dv_obs = float(vel[1] - vel[0])

    # --- Build wavelength grid for obs spectra (same as your code) ---
    line_cen, delta_wv = Star.bands2use[band][0], Star.bands2use[band][1]
    wvmin, wvmax = int(line_cen - delta_wv), int(line_cen + delta_wv)
    c_kms = 299792.458
    w_center = 0.5 * (wvmin + wvmax)
    wave_obs = vel * w_center / c_kms + w_center  # Å

    # --- Build/choose template exactly like before ---
    ccf_tt, ccf_logg, ccf_FeH, ccf_vsini, ccf_resolution = rv_ccf_pars
    if Star.ccf_template_photosphereORnot == 1:
        ccf_template_temp = int(np.round(Star.Temp, -2))
    else:
        ccf_template_temp = Star.ccf_temp

    if ccf_tt != 99999:
        # Use user-provided CCF template parameters
        spec_base_name = (
            f'mc_{ccf_tt:1.4f}{ccf_logg*100:1.0f}+{ccf_FeH*100:03.0f}_'
            'b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf'
        )
        spec_dir = f'{prf_dir}/{spec_base_name}'
        sim_spec_name = (
            f'RVCCFUse_Synmast_spec_{ccf_tt:1.4f}_{ccf_logg:1.2f}_{ccf_FeH:1.2f}_'
            f'vsini{ccf_vsini:1.02f}_res{ccf_resolution:1.0f}_wv{wvmin}_{wvmax}_flatten.p'
        )
        sim_spec_path = f'{Star.out_path}{sim_spec_name}'
        ssp.final_flux_spec(
            'ccf_temp', spec_dir, sim_spec_path, wvmin, wvmax, Vel_bin_size,
            flatten=True, pars=rv_ccf_pars
        )
        _temp = pd.read_pickle(sim_spec_path)
        spec_temp = _temp['f'].to_numpy(dtype=float)
        # Template wavelength grid equals wave_obs after final_flux_spec → we rederive vel_temp below:
        vel_temp = (wave_obs - w_center) * c_kms / w_center
    else:
        # Build PHX-based template at ccf_template_temp with broadening + flattening
        vel_temp = Star.spec_flatten[f'{band}_{model_type}_{intORflux}_{ccf_template_temp:1.4f}_v']
        spec_temp = Star.spec_flatten[f'{band}_{model_type}_{intORflux}_{ccf_template_temp:1.4f}_f']
        vel_temp = np.asarray(vel_temp, dtype=np.float64)
        spec_temp = np.asarray(spec_temp, dtype=np.float64)

        wave_temp = vel_temp * w_center / c_kms + w_center
        # vsini broadening
        wave_temp, spec_temp = rotint(wave_temp, spec_temp, Star.vsini)
        # instrumental broadening
        hwhm_list = wave_temp / Star.inst_res / 2.0
        spec_temp = macbro_dyn(wave_temp, spec_temp, hwhm_list)
        # flatten
        spec_temp = flatten_ccf_model_spec(wave_temp, spec_temp, band, RVandEWR_dir, plot)
        vel_temp = c_kms * (wave_temp - w_center) / w_center

        if save_spec_flag == 1:
            np.save(f'./outputs/{RVandEWR_dir}/ccf_template/{band}_ccf_template_Tp{Star.ccf_temp:1.4f}.npy', spec_temp)
            np.save(f'./outputs/{RVandEWR_dir}/ccf_template/{band}_ccf_template_wave_Tp{Star.ccf_temp:1.4f}.npy', wave_temp)

    # --- Determine CCF RV range and step ---
    base_step = (Vel_bin_size / 100.0) * 2.0

    # --- make sure the template has a higher resolution of the "obs" ---
    valid_temp_mask = (wave_temp >= wave_obs[0]) & (wave_temp <= wave_obs[-1])
    n_obs = len(wave_obs)
    n_temp_in_range = np.sum(valid_temp_mask)

    oversample_factor = 5

    if n_temp_in_range < oversample_factor * n_obs:
        # print(f"Upsampling template: current ratio is {n_temp_in_range/n_obs:.2f}x. Target: {oversample_factor}x")
        
        # 2. Calculate the new size needed for the ENTIRE template array 
        # to maintain the oversample_factor in the overlapping region.
        # We upscale the whole array to ensure we don't lose the edges needed for RV shifting.
        current_temp_size = len(wave_temp)
        target_temp_size = int(current_temp_size * (oversample_factor * n_obs / max(1, n_temp_in_range)))
        
        # Create the new fine wavelength grid
        wave_temp_fine = np.linspace(wave_temp[0], wave_temp[-1], target_temp_size)
        
        # 3. Apply Cubic Spline interpolation
        cs = CubicSpline(wave_temp, spec_temp)
        spec_temp_fine = cs(wave_temp_fine)
        
        # Overwrite the original arrays with the up-sampled versions
        wave_temp = wave_temp_fine
        spec_temp = spec_temp_fine

    # Helper to run mc_ccf across an index list safely
    def _run_batch(idx_array, flux_box, vmin, vmax, step, clip_px, _cpu_use, _silence_pbar):
        # partial with closed-over arrays that are not pickled each call

        func = partial(
            mc_ccf, wave_obs, flux_box, wave_temp, spec_temp, vmin, vmax, step, 
            clip_px, visible_phi)
        # pqdm feeds us the integer index; call mc_ccf with star_tot[i] as 2nd arg
        try:
            out = pqdm(idx_array, func, n_jobs=_cpu_use, disable=_silence_pbar)
        except KeyboardInterrupt:
            print("\n[Ctrl+C detected] Terminating worker processes and freeing memory...")
            sys.exit(1)
        return out

    # Edge clipping in pixels 
    # Use template dv for conversion 
    dv_tpl = float(vel_temp[1] - vel_temp[0]) if vel_temp.size > 1 else dv_obs
    # guard: dvt can be tiny; avoid divide-by-zero
    clip_obs_edges_default = int((10.0 / dv_tpl) * 5) if dv_tpl != 0.0 else 0

    # --- PASS 1: probe a handful of phases to estimate required bounds/step ---
    if nphi <= 10:
        # just do one pass with a wide window
        edge = 40.0
        trial_idx = np.arange(nphi, dtype=int)
        outs = _run_batch(
            trial_idx, star_tot, -edge, edge, base_step, clip_obs_edges_default, min(cpu_use, nphi), silence_pbar)
        
        drvs = np.arange(-edge, edge, base_step, dtype=np.float64)
        rv_coarse = get_ccf_peak(
            np.asarray(outs, dtype=object), drvs, visible_phi[trial_idx], band, outdir)
        rv_final = rv_coarse - np.nanmin(rv_coarse)
        return rv_final.astype(np.float64, copy=False)

    # --- PASS 1: trials to get rough RV and width ---
    trial_idx  = _pick_trials(visible_phi, n_keep=6)
    trial_edge = 10.0
    drvs_trial = np.arange(-trial_edge, trial_edge, base_step, dtype=np.float64)
    outs_trial = _run_batch(
        trial_idx, star_tot, -trial_edge, trial_edge, base_step, clip_obs_edges_default, 
        min(cpu_use, len(trial_idx)), silence_pbar)
    rv_trial   = get_ccf_peak(
        np.asarray(outs_trial, dtype=object), drvs_trial, visible_phi[trial_idx], band, outdir)

    bound = float(np.nanmax(np.abs(np.round(rv_trial, 3)))) if rv_trial.size else np.nan
    if not np.isfinite(bound) or bound <= 0.02:
        return np.full(nphi, np.nan, dtype=np.float64)

    # ladder for window/step
    if bound <= 0.5:
        ccf_bound = 2.0
        ccf_step  = base_step
        # print(f'\n   Bound = {bound:1.3f} <= 0.5 km/s → using tight window ±{ccf_bound} km/s and fine step {ccf_step:1.5f} km/s')
    elif bound <= 5.0:
        ccf_bound = 10.0
        # ccf_bound = 5
        ccf_step  = base_step * 10.0
        # print(f'\n   Bound = {bound:1.3f} <= 5.0 km/s → using medium window ±{ccf_bound} km/s and medium step {ccf_step:1.5f} m/s')
    else:
        ccf_bound = bound * 4.0
        ccf_step  = base_step * 100.0
        # print(f'\n   Bound = {bound:1.3f} > 5.0 km/s → using wide window ±{ccf_bound} km/s and coarse step {ccf_step:1.3f} km/s')

    clip_obs_edges = int((ccf_bound / dv_tpl) * 5) if dv_tpl != 0.0 else 0

    # --- PASS 2: full run with chosen window/step (FINAL) ---
    drvs_full = np.arange(-ccf_bound, ccf_bound, ccf_step, dtype=np.float64)
    outs_full = _run_batch(
        np.arange(nphi, dtype=int), star_tot, -ccf_bound, ccf_bound, ccf_step, 
        clip_obs_edges, min(cpu_use, nphi), silence_pbar)
    rv_full   = get_ccf_peak(
        np.asarray(outs_full, dtype=object), drvs_full, visible_phi, band, outdir
    )
    rv_full   = rv_full - np.nanmean(rv_full)

    return rv_full.astype(np.float64, copy=False)

# === Standard library ===
import argparse
import configparser
import logging
import os, sys
import socket
import re
import warnings
from pathlib import Path
from time import time

# === Suppress warnings ===
warnings.filterwarnings("ignore")

# === Third-party libraries ===
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from astropy.table import Table

# === Custom modules ===
import soapcore
import starspot as stsp
import visualization as vis
from pyphot import astropy as pyphot
from pyphot import svo
from soapcore import flatten_model_spec

# CONFIG_FILE = 'config_LkCa4base_SizeDelT_study.cfg'
# CONFIG_FILE = 'config_LkCa4_2SPOTS_study.cfg'
# CONFIG_FILE = 'config_LkCa4_multi_SPOTS_study.cfg'
CONFIG_FILE = 'config_LkCa4_multi_SPOTS_fitting_9spots.cfg'
# CONFIG_FILE = 'config_teff_study.cfg'
# SAVE_DIR = f'RVandEWR_SizeDelT_width100'

# SimSpecWinSize = 100 # in AA
SIMSPEC_VEL_BIN_SIZE = 100/1000 # in km/s
SOLAR_RADIUS = 695700 # km
# ------------------------------------------------
hostname = socket.gethostname() 
hostname = hostname.split('.')[0].split('-')[0]

def setup_logging(debug: bool = False) -> logging.Logger:
    """
    Set up and return a root logger with stream handler.

    Args:
        debug (bool): If True, set logging level to DEBUG, else INFO.

    Returns:
        logging.Logger: Configured root logger.
    """
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        # Uncomment to customize log format:
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def init_setup_cfg(
    logger: logging.Logger,
    config_file: str,
    post_fix: str = None,
    fit_temp_range=None
):
    """
    Initialize and parse configuration, set up directories, and prepare simulation parameters.

    Args:
        logger (logging.Logger): Logger for status and warnings.
        config_file (str): Path to the configuration file.
        post_fix (str, optional): Suffix for output directory naming.
        fit_temp_range (optional): Temperature range for fitting.

    Returns:
        tuple: (STAR, bands, SAVE_DIR, nrho, grid_size, instrument_res, cpu_use, plot_flag,
                save_spec_flag, sim_spec_window_size, syn_model_path, rv_ccf_pars, PSI,
                active_regions_number, active_region_box, sim_spec_line_centers, filters)
    """
    # --- Read config ---
    config = configparser.ConfigParser(inline_comment_prefixes=';')
    config.read(config_file)
    save_folder_name = config.get('output', 'save_folder_name', fallback='').strip()
    hostname = socket.gethostname().split('.')[0].split('-')[0]
    SAVE_DIR = f"{save_folder_name}_{post_fix}_{hostname}" if post_fix else f"{save_folder_name}_{hostname}"

    # --- Spectroscopic Parameters ---
    wv_start = int(config.get('spectroscopic', 'wavelength_start'))
    wv_end = int(config.get('spectroscopic', 'wavelength_end'))
    wv_step = int(config.get('spectroscopic', 'wavelength_step'))
    sim_spec_window_size = float(config.get('spectroscopic', 'simulate_spectra_window'))

    # --- Photometric Parameters ---
    filters = config.get('photometry', 'filters').split(',')

    # --- Bands setup ---
    sim_spec_line_centers = np.arange(wv_start, wv_end + 1, wv_step)
    bands = {
        f"w{wave_center // 100}": [
            float(wave_center),
            sim_spec_window_size,
            'rv',
            'flux',
            'phx'
        ]
        for wave_center in sim_spec_line_centers
    }
    
    bands = {}
    # Add Equivalent Width Ratio (EWR) band
    bands['ewr_FeOH'] = [15557, 90, 'rv', 'intensity', 'synmast']

    # --- Photometry band setup ---
    if (len(filters) == 1 and 'None' in filters[0]) or (len(filters) == 1 and filters[0] == ''):
        logger.info('No filters specified. Will not generate photometry.')
    else:
        logger.info(f'Detect filters. Will generate photometry for {filters}.')
        filter_profiles = [svo.get_pyphot_filter(filt) for filt in filters]
        filter_wavelengths = np.concatenate([f.wavelength for f in filter_profiles]) * 10  # nm to AA
        filter_wv_min, filter_wv_max = np.min(filter_wavelengths), np.max(filter_wavelengths)
        if filter_wv_min < 500 or filter_wv_max > 50000:
            logger.warning(
                f'Filter wavelength range {filter_wv_min:.1f} - {filter_wv_max:.1f} AA is not in the range of 500 - 50000 AA.'
            )
        photo_center = int((filter_wv_max + filter_wv_min) / 2)
        photo_width = int((filter_wv_max - filter_wv_min) / 2)
        bands['photo'] = [photo_center, photo_width, 'photo', 'intensity', 'phx']

    # --- Path setup ---
    path_sub = 1 if hostname == 'Swing' else 2
    RASSINE_PATH = config.get('path', f'rassine_path{path_sub}')
    SYNmast_PATH = config.get('path', f'synmast_prf{path_sub}')
    syn_model_path = {'phx': RASSINE_PATH, 'synmast': SYNmast_PATH}
    phx_path = config.get('path', 'PHOENIX_path')

    # --- Output and model parameters ---
    plot_flag = int(config.get('output', 'plot'))
    save_spec_flag = int(config.get('output', 'save_spec'))
    grid_size = int(config.get('main', 'grid'))
    nrho = int(config.get('main', 'nrho'))
    instrument_res = int(config.get('main', 'instrument_reso'))
    cpu_use = 6 if hostname == 'Swing' else 30

    # --- Stellar parameters ---
    stellar_radius = float(config.get('star', 'radius')) * SOLAR_RADIUS
    rotation_period = float(config.get('star', 'prot'))
    incl = float(config.get('star', 'I'))
    psi = float(config.get('star', 'psi'))
    Temp = int(config.get('star', 'Tstar'))
    logg = float(config.get('star', 'logg'))
    FeH = float(config.get('star', 'FeH'))

    # --- Active region parameters ---
    cool_active_regions_number = int(config.get('cool_active_regions', 'spot_number'))
    hot_active_regions_number = int(config.get('hot_active_regions', 'spot_number'))
    active_regions_number = cool_active_regions_number + hot_active_regions_number

    logger.info(
        f"Found {cool_active_regions_number} cool region(s), "
        f"and {hot_active_regions_number} hot region(s)."
    )

    def parse_region(key, region_type):
        return np.array([
            float(val) for val in re.split(r'[,\s]+', config.get(region_type, key).strip()) if val
        ])

    cool_active_region_longs = parse_region('longitudes', 'cool_active_regions')
    cool_active_region_lats = parse_region('latitudes', 'cool_active_regions')
    cool_active_region_sizes = parse_region('sizes', 'cool_active_regions')
    cool_active_region_dTemp = np.array([
        int(val) for val in re.split(r'[,\s]+', config.get('cool_active_regions', 'Tcool_spots').strip()) if val
    ])

    # Sort cool spots by dTemp descending
    sort_idx = np.argsort(cool_active_region_dTemp)[::-1]
    cool_active_region_longs = cool_active_region_longs[sort_idx]
    cool_active_region_lats = cool_active_region_lats[sort_idx]
    cool_active_region_sizes = cool_active_region_sizes[sort_idx]
    cool_active_region_dTemp = cool_active_region_dTemp[sort_idx]
    cool_active_region_Temp = Temp - cool_active_region_dTemp

    hot_active_region_longs = parse_region('longitudes', 'hot_active_regions')
    hot_active_region_lats = parse_region('latitudes', 'hot_active_regions')
    hot_active_region_sizes = parse_region('sizes', 'hot_active_regions')
    hot_active_region_dTemp = np.array([
        int(val) for val in re.split(r'[,\s]+', config.get('hot_active_regions', 'Thot_faculae').strip()) if val
    ])
    sort_idx = np.argsort(hot_active_region_dTemp)[::-1]
    hot_active_region_longs = hot_active_region_longs[sort_idx]
    hot_active_region_lats = hot_active_region_lats[sort_idx]
    hot_active_region_sizes = hot_active_region_sizes[sort_idx]
    hot_active_region_dTemp = hot_active_region_dTemp[sort_idx]
    hot_active_region_Temp = Temp + hot_active_region_dTemp

    # Convert area % to spot radius
    cool_active_region_sizes_percent = cool_active_region_sizes.copy()
    cool_active_region_sizes = np.sqrt(cool_active_region_sizes / 100)
    hot_active_region_sizes_percent = hot_active_region_sizes.copy()
    hot_active_region_sizes = np.sqrt(hot_active_region_sizes / 100)

    # Concatenate all active regions
    active_region_longs = np.concatenate([cool_active_region_longs, hot_active_region_longs])
    active_region_lats = np.concatenate([cool_active_region_lats, hot_active_region_lats])
    active_region_Temp = np.concatenate([cool_active_region_Temp, hot_active_region_Temp])
    active_region_sizes = np.concatenate([cool_active_region_sizes, hot_active_region_sizes])
    active_region_box = [active_region_longs, active_region_lats, active_region_sizes, active_region_Temp]

    # --- Star object ---
    STAR = soapcore.Star(
        prot=rotation_period,
        vrot=(2 * np.pi * stellar_radius) / (rotation_period * 86400),
        incl=incl,
        psi=psi,
        rad_sun=SOLAR_RADIUS,
        rad=stellar_radius,
        Temp=Temp,
        Tspot=cool_active_region_Temp,
        Tfaculae=hot_active_region_Temp,
        logg=logg,
        FeH=FeH,
        bands2use=bands,
    )
    STAR.vsini = STAR.vrot * np.sin(np.deg2rad(STAR.incl))
    if 'photo' in bands.keys():
        STAR.filters = filters
    STAR.inst_res = instrument_res

    # --- Output directories ---
    dirs_to_create = [
        "syn_spec", "syn_spec/RASSINE_PHX", "syn_spec/SimUse", "syn_spec/Photometry_use_Spec",
        f"outputs/{SAVE_DIR}", f"outputs/{SAVE_DIR}/normal_spec", f"outputs/{SAVE_DIR}/spec",
        f"outputs/{SAVE_DIR}/ccf_template", "outputs/CCF",
    ]
    for path in dirs_to_create:
        os.makedirs(path, exist_ok=True)

    # --- RV CCF template parameters ---
    rv_ccf_pars = np.zeros(5) + 99999

    # --- Phase setup ---
    phase_step = float(config.get('output', 'ph_step'))
    phase_in = config.get('output', 'ph_in', fallback='').strip()
    if phase_in and phase_in.lower() != 'none':
        with open(phase_in) as f:
            PSI = np.array([float(l.split()[0]) for l in f.readlines()])
    elif phase_step > 0:
        PSI = np.arange(-0.75, 0.751, phase_step)
    else:
        logger.warning("Put a positive phase step ('ph_step') or provide a phase file ('ph_in').")

    # --- PHOENIX mode path ---
    STAR.phx_path = Path(f'{phx_path}/PHOENIX-ACES-AGSS-COND-2011').expanduser()
    os.makedirs(STAR.phx_path, exist_ok=True)
    os.makedirs(STAR.phx_path / 'SpecInt', exist_ok=True)
    os.makedirs(STAR.phx_path / 'SpecFlux', exist_ok=True)

    # --- Grid check and flatten synthetic spectra ---
    soapcore.grid_resolution_check(grid_size)
    # STAR.find_fluxspec_flatten(SIMSPEC_VEL_BIN_SIZE, plot_flag, syn_model_path, fit_temp_range)

    STAR.ccf_template_photosphereORnot = float(config.get('ccf', 'Use_photosphere_teff'))
    if STAR.ccf_template_photosphereORnot == 1:
        STAR.ccf_temp = STAR.Temp
        fit_temp = None
    else:
        STAR.ccf_temp = float(config.get('ccf', 'Teff'))
        fit_temp = STAR.ccf_temp

    STAR.find_fluxspec_flatten(SIMSPEC_VEL_BIN_SIZE, plot_flag, syn_model_path, fit_temp)

    return (
        STAR, bands, SAVE_DIR, nrho, grid_size, instrument_res, cpu_use, plot_flag,
        save_spec_flag, sim_spec_window_size, syn_model_path, rv_ccf_pars, PSI,
        active_regions_number, active_region_box, sim_spec_line_centers, filters
    )

def setup_activate_spectra(bands, STAR, plot_flag, syn_model_path):
    """
    Prepare synthetic spectra for all bands and temperatures for the spotted star simulation.

    Args:
        bands (dict): Band definitions with wavelength, width, mode, etc.
        STAR (Star): The star object with physical and spectral properties.
        plot_flag (int): If True, enable plotting during spectrum generation.
        syn_model_path (dict): Paths to synthetic model spectra.

    Returns:
        list: List of dictionaries mapping temperature to Spec objects for each band.
    """
    # Determine which intensity modes are needed
    intensity_modes = np.unique(np.vstack(list(bands.values()))[:, 3])
    temp_list = np.concatenate([[STAR.Temp], STAR.Tspot, STAR.Tfaculae])

    # Generate limb darkening and spectra as needed
    line_centers = [band[0] for band in bands.values()]
    line_widths = [band[1] for band in bands.values()]

    if 'flux' in intensity_modes:
        STAR.get_quadratic_limbdarken(temp_list, line_centers, line_widths)
        STAR.find_fluxspec(SIMSPEC_VEL_BIN_SIZE, plot_flag, syn_model_path)

    if 'intensity' in intensity_modes:
        STAR.get_quadratic_limbdarken(temp_list, line_centers, line_widths)
        STAR.find_intensityspec(SIMSPEC_VEL_BIN_SIZE, plot_flag, syn_model_path)

    # Build Spec objects for each band and temperature
    SPEC_box = []
    for band_name, band_params in bands.items():
        int_or_flux = band_params[3]
        model_type = band_params[4]
        spec_dict = {
            int(temp): soapcore.Spec(
                star=STAR,
                band=band_name,
                temperature=temp,
                intORflux=int_or_flux,
                model_type=model_type
            )
            for temp in temp_list
        }
        SPEC_box.append(spec_dict)

    return SPEC_box


def simulation_plot(grid_size, nrho, instrument_res, STAR,
                    model_type, SAVE_DIR,
                    active_regions_number, active_region_box,
                    visible_phi, integrated_stellar_spectrum,
                    ccf_rv_shift, final_2d_surface_flux_sum,
                    outdir, cpu_use, ohfe_ratio, band, intORflux, filters,
                    lightcurves_mag, lightcurves_flux):
    """
    Plot the simulated results for the spotted star simulation.

    Args:
        grid_size (int): Grid resolution.
        nrho (int): Circumference resolution.
        instrument_res (int): Instrumental resolution.
        STAR (Star): Star object.
        model_type (str): Model type.
        SAVE_DIR (str): Output directory.
        active_regions_number (int): Number of active regions.
        active_region_box (list): Active region properties.
        visible_phi (array): Visible phases.
        integrated_stellar_spectrum (array): Integrated spectrum.
        ccf_rv_shift (array): CCF RV shifts.
        final_2d_surface_flux_sum (array): 2D surface flux.
        outdir (str): Output directory for plots.
        cpu_use (int): Number of CPUs used.
        ohfe_ratio (array): OH/Fe ratio.
        band (str): Band name.
        intORflux (str): 'intensity' or 'flux'.
        filters (list): List of filter names.
        lightcurves_mag (array): Photometric lightcurves (magnitude).
        lightcurves_flux (array): Photometric lightcurves (flux).
    """
    (active_region_longs, active_region_lats, active_region_sizes, 
        active_region_Temp) = active_region_box
    active_region_sizes_percent = np.array(active_region_sizes)**2 * 100
    active_region_dTemp = np.array(active_region_Temp) - STAR.Temp

    # --- Plot the simulated results ---
    text1 = '{0:18} = {1:<6d}, {2:18} = {3:<6d}\n'.format(    'Grid res.',  grid_size,      'Circumference res.', nrho) + \
            '{0:18} = {1:<6d}, {2:18} = {3:<6.2f}\n'.format(  'Inst. res.', instrument_res, 'St Prot [d]', STAR.prot) + \
            '{0:18} = {1:<6.2f}, {2:18} = {3:<6.1f}\n'.format('St radius [Rsun]', STAR.rad/SOLAR_RADIUS,     'St incl', STAR.incl) + \
            '{0:18} = {1:<6.2f}, {2:18} = {3:<6.0f}\n'.format('St vsini [km/s]', STAR.vsini,         'St T_photo [K]', STAR.Temp) + \
            '{0:18} = {1:<6}, {2:5} = {3:<4.2f}, {4:5} = {5:<4.2f}'.format('Syn template', model_type, 'logg', STAR.logg, '[Fe/H]', 0.0)

    text2 = '{} active regions:\n'.format(active_regions_number) + \
            '  lon [deg]= {}\n'.format(np.array2string(
                active_region_longs, separator=',', formatter={'float_kind': lambda x: f'{int(x):4d}'})) + \
            '  lat [deg]= {}\n'.format(np.array2string(
                active_region_lats, separator=',', formatter={'float_kind': lambda x: f'{int(x):4d}'})) + \
            '   size [%]= {}\n'.format(np.array2string(
                active_region_sizes_percent, separator=',', formatter={'float_kind': lambda x: f'{int(x):4d}'})) + \
            '   Temp [K]= {}\n'.format(np.array2string(
                active_region_dTemp, separator=',', formatter={'float_kind': lambda x: fr'${int(x):+5d}$'}))
    
    if band != 'photo':
        vis.spot_view(
            STAR, grid_size, visible_phi, integrated_stellar_spectrum, ccf_rv_shift,
            final_2d_surface_flux_sum, outdir, cpu_use, ohfe_ratio, text1, text2,
            band, intORflux, model_type)
    else:
        visible_phi = np.array(visible_phi)
        fig, ax = plt.subplots(2, 1, figsize=(4, 3), dpi=300)    
        for i, filt in enumerate(filters):
            sc = ax[0].plot(
                visible_phi[1:-1]-1, lightcurves_mag[i][1:-1], 
                label=filt.split('/')[1], marker='o', ms=2, alpha=0.7)
            ax[0].plot(
                visible_phi[1:-1], lightcurves_mag[i][1:-1], 
                label='__', marker='o', ms=2, alpha=0.7, color=sc[0].get_color())
            ax[0].plot(
                visible_phi[1:-1]+1, lightcurves_mag[i][1:-1], 
                label='__', marker='o', ms=2, alpha=0.7, color=sc[0].get_color())
            
            ax[1].plot(
                visible_phi[1:-1]-1, lightcurves_flux[i][1:-1], 
                label='__', marker='o', ms=2, alpha=0.7, color=sc[0].get_color())
            ax[1].plot(
                visible_phi[1:-1], lightcurves_flux[i][1:-1], 
                label='__', marker='o', ms=2, alpha=0.7, color=sc[0].get_color())
            ax[1].plot(
                visible_phi[1:-1]+1, lightcurves_flux[i][1:-1], 
                label='__', marker='o', ms=2, alpha=0.7, color=sc[0].get_color())
        
        for ii in range(2):
            ax[ii].xaxis.set_minor_locator(AutoMinorLocator(4))
            ax[ii].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[1].set_xlabel('Phase', size='small')
        ax[0].set_ylabel('Relative magnitude', size='small')
        ax[1].set_ylabel('Normalized flux', size='small')
        ax[0].legend(fontsize='xx-small', edgecolor='white')
        fig.savefig(
            f'./outputs/{SAVE_DIR}/{outdir.split("/")[-1]}_{band}_lightcurves.png', 
            bbox_inches='tight')

def main_model(
    logger, bands, nrho, grid_size, STAR, PSI, SPEC_box, filters,
    active_regions_number, active_region_box, rv_ccf_pars, syn_model_path,
    instrument_res, cpu_use, SAVE_DIR, save_spec_flag, plot_flag, silence_pbar
):
    """
    Run the main simulation for all bands and output results.

    Args:
        logger (logging.Logger): Logger for status and warnings.
        bands (dict): Band definitions.
        nrho (int): Circumference resolution.
        grid_size (int): Grid resolution.
        STAR (Star): Star object.
        PSI (array): Phases.
        SPEC_box (list): List of Spec dictionaries for each band.
        filters (list): List of filter names.
        active_regions_number (int): Number of active regions.
        active_region_box (list): Active region properties.
        rv_ccf_pars (array): RV CCF template parameters.
        syn_model_path (dict): Paths to synthetic model spectra.
        instrument_res (int): Instrumental resolution.
        cpu_use (int): Number of CPUs used.
        SAVE_DIR (str): Output directory.
        save_spec_flag (int): If True, save spectra.
        plot_flag (int): If True, plot results.
        silence_pbar (bool): If True, silence progress bars.

    Returns:
        str: Output directory used for saving results.
    """
    # sort active regions by temperature so do coolest first for cool spots, and 
    # hottest first for hot spots
    (active_region_longs, active_region_lats, active_region_sizes, 
        active_region_Temp) = active_region_box
    active_region_org_idx = np.arange(len(active_region_longs))

    temp_cool_idx = active_region_Temp - STAR.Temp < 0
    temp_hot_idx = active_region_Temp - STAR.Temp > 0

    sort_idx = np.argsort(STAR.Tspot)
    # Apply the same order to all arrays
    cool_active_region_longs = active_region_longs[temp_cool_idx][sort_idx]
    cool_active_region_lats = active_region_lats[temp_cool_idx][sort_idx]
    cool_active_region_sizes = active_region_sizes[temp_cool_idx][sort_idx]
    cool_active_region_Temp = active_region_Temp[temp_cool_idx][sort_idx]
    cool_active_region_org_idx = active_region_org_idx[temp_cool_idx][sort_idx]

    sort_idx = np.argsort(STAR.Tfaculae)[::-1]  # ascending; use [::-1] for descending
    # Apply the same order to all arrays
    hot_active_region_longs = active_region_longs[temp_hot_idx][sort_idx]
    hot_active_region_lats = active_region_lats[temp_hot_idx][sort_idx]
    hot_active_region_sizes = active_region_sizes[temp_hot_idx][sort_idx]
    hot_active_region_Temp = active_region_Temp[temp_hot_idx][sort_idx]
    hot_active_region_org_idx = active_region_org_idx[temp_hot_idx][sort_idx]

    active_region_longs = np.concatenate(
        [cool_active_region_longs, hot_active_region_longs])
    active_region_lats = np.concatenate(
        [cool_active_region_lats, hot_active_region_lats])
    active_region_sizes = np.concatenate(
        [cool_active_region_sizes, hot_active_region_sizes])
    active_region_Temp = np.concatenate(
        [cool_active_region_Temp, hot_active_region_Temp])
    active_region_org_idx = np.concatenate(
        [cool_active_region_org_idx, hot_active_region_org_idx])
    # -------------

    active_region_box = (
        active_region_longs, active_region_lats, active_region_sizes, 
        active_region_Temp)

    for i, band in enumerate(bands.keys()):
        start_time = time()
        intORflux = bands[band][3]
        model_type = bands[band][4]
        logger.info(f'\n\nNow calculating the activity signal for the band: {band}')
        SPEC = SPEC_box[i]

        # --- Main simulation ---
        (visible_phi, integrated_stellar_spectrum, _, final_2d_surface_flux_sum) = soapcore.calculate_activity_signal_spec(
            nrho, grid_size, STAR, PSI, SPEC, active_regions_number, active_region_box,
            instrument_res, cpu_use, SAVE_DIR, band, save_spec_flag, intORflux.lower(), silence_pbar
        )

        # --- Flatten spectra ---
        if band != 'photo':
            integrated_stellar_spectrum = flatten_model_spec(
                STAR, integrated_stellar_spectrum, visible_phi, band,
                intORflux, model_type, SAVE_DIR, plot_flag
            )
            if save_spec_flag == 1:
                np.save(
                    f'./outputs/{SAVE_DIR}/spec/{band}_flat_final_spec.npy',
                    integrated_stellar_spectrum
                )

        # --- Equivalent Width Ratio (EWR) or placeholder ---
        if band == 'ewr_FeOH':
            ohfe_ratio = soapcore.get_equivalent_width(
                STAR, integrated_stellar_spectrum, visible_phi, band, intORflux, model_type
            )
        else:
            ohfe_ratio = np.zeros_like(visible_phi) + np.nan

        # --- Calculate CCF RV ---
        if band != 'photo':
            logger.info('\nCalculating the RV with CCF...\n')
            ccf_rv_shift = soapcore.cal_ccf_rv(
                integrated_stellar_spectrum, STAR, visible_phi, band, intORflux, model_type,
                rv_ccf_pars, syn_model_path['synmast'], cpu_use, SAVE_DIR, SIMSPEC_VEL_BIN_SIZE,
                plot_flag, silence_pbar, save_spec_flag
            )
        else:
            ccf_rv_shift = np.zeros_like(visible_phi) + np.nan

        # --- Calculate relative photometric variation ---
        if band == 'photo':
            lightcurves_mag, lightcurves_flux = soapcore.get_photometric(
                integrated_stellar_spectrum, STAR, visible_phi, band, intORflux, model_type
            )
        else:
            lightcurves_mag, lightcurves_flux = [None, None]

        # --- Prepare output folder names ---
        (active_region_longs, active_region_lats, active_region_sizes, active_region_Temp) = active_region_box
        active_region_sizes_percent = np.array(active_region_sizes) ** 2 * 100

        temp_str = np.array2string(active_region_Temp, separator='-', formatter={'float_kind': lambda x: f'{int(x):d}'})
        temp_str_uniq = np.array2string(np.unique(active_region_Temp), separator='-', formatter={'float_kind': lambda x: f'{int(x):d}'})
        long_str = np.array2string(active_region_longs, separator='-', formatter={'float_kind': lambda x: f'{int(x):d}'})
        lat_str = np.array2string(active_region_lats, separator='-', formatter={'float_kind': lambda x: f'{int(x):d}'})
        size_str = np.array2string(active_region_sizes_percent, separator='-', formatter={'float_kind': lambda x: f'{int(x):d}'})

        # Remove square brackets and spaces
        temp_str = temp_str.strip('[]').replace(' ', '')
        temp_str_uniq = temp_str_uniq.strip('[]').replace(' ', '')
        long_str = long_str.strip('[]').replace(' ', '')
        lat_str = lat_str.strip('[]').replace(' ', '')
        size_str = size_str.strip('[]').replace(' ', '')

        if active_regions_number < 4:
            outdir = "./outputs/{}/{}_P={:.1f}_i={:.1f}_Tp={:1.0f}_logg={:1.1f}_Ta={}_lon={}_lat={}_size={}".format(
                SAVE_DIR, model_type, STAR.prot, STAR.incl, STAR.Temp, STAR.logg,
                temp_str, long_str, lat_str, size_str)
        else:
            outdir = "./outputs/{}/{}_P={:.1f}_i={:.1f}_Tp={:1.0f}_logg={:1.1f}_SpotNum={:1.0f}_Ta={}".format(
                SAVE_DIR, model_type, STAR.prot, STAR.incl, STAR.Temp, STAR.logg,
                active_regions_number, temp_str_uniq)

        # --- Save output table ---
        if band != 'photo':
            tab_save = Table(
                data=(visible_phi, ccf_rv_shift, ohfe_ratio),
                names=('Phase', 'RV', 'EWR')
            )
            tab_save['Phase'].format = '%.3f'
            tab_save['RV'].format = '%.4f'
            tab_save['EWR'].format = '%.5f'
        else:
            tab_save = Table([visible_phi], names=['Phase'])
            for i, filt in enumerate(filters):
                filter_name = filt.split('/')[1]
                tab_save[f'{filter_name}_mag'] = lightcurves_mag[i]
                tab_save[f'{filter_name}_mag'].format = '%.6f'
                tab_save[f'{filter_name}_flux'] = lightcurves_flux[i]
                tab_save[f'{filter_name}_flux'].format = '%.6f'
            tab_save['Phase'].format = '%.3f'

        tab_save.write(
            f'./outputs/{SAVE_DIR}/{outdir.split("/")[-1]}_{band}_RV_EWR.csv',
            format='csv', overwrite=True
        )

        # --- Plotting ---
        if plot_flag == 1:
            simulation_plot(
                grid_size, nrho, instrument_res, STAR, model_type, SAVE_DIR,
                active_regions_number, active_region_box, visible_phi, integrated_stellar_spectrum,
                ccf_rv_shift, final_2d_surface_flux_sum, outdir, cpu_use, ohfe_ratio,
                band, intORflux, filters, lightcurves_mag, lightcurves_flux
            )

        end_time = time()
        logger.warning(f'\n--> {band} run time: {(end_time-start_time)/60:1.2f} min')

    # order active region back to original order
    org_idx = np.argsort(active_region_org_idx)
    active_region_longs = active_region_longs[org_idx]
    active_region_lats = active_region_lats[org_idx]
    active_region_sizes = active_region_sizes[org_idx]
    active_region_Temp = active_region_Temp[org_idx]
    active_region_box = (active_region_longs, active_region_lats, 
                         active_region_sizes, active_region_Temp)
    return outdir
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog        = 'SPOTACT-TTS simulation',
        description = '''
        NaN
        ''', epilog = 'sytang@rice.edu'
        )

    # parser.add_argument(
    #     '-cfg', dest="cfg", action="store", help="Enter the .cfg file to run.",
    #     type=str, default=None)
    # parser.add_argument(
    #     '-st', dest="st", action="store", help="The start dates id to run.",
    #     type=int, default=None)
    # parser.add_argument(
    #     '-ed', dest="ed", action="store", help="The end dates id to run.",
    #     type=int, default=None)
        
    # ---- Not normal setting ----
    # parser.add_argument(
    #     '-delT', dest="delT", action="store", help="Delta T for spot.",
    #     type=float, default=None)
    # parser.add_argument(
    #     '-Scover', dest="Scover", action="store", help="Spot coverage in % for visible DISK.",
    #     type=float, default=None)
    # parser.add_argument(
    #     '-Lat', dest="Lat", action="store", help="Manually set the spot latitude.",
    #     type=float, default=None)
    # parser.add_argument(
    #     '-Incl', dest="Incl", action="store", help="Manually set the Stellar incl. angle",
    #     type=float, default=None)
    parser.add_argument(
        '-out', dest="out", action="store", help="Out dur name.",
        type=str, default=None)
    # parser.add_argument(
    #     '-plot', dest="plot", action="store", help="If make gif.",
    #     type=str, default=None)
    parser.add_argument(
        '-DeBug', dest="debug", action="store_true",
        help="If set, DeBug logging will be output, as well as (lots of) \
                extra plots."
        )
    
    args = parser.parse_args()
    logger = setup_logging(args.debug)

    ################################################################################################
    
    # read in cfg, and setup classes
    (STAR, bands, SAVE_DIR, nrho, grid_size, instrument_res, cpu_use,
        plot_flag, save_spec_flag, sim_spec_window_size, syn_model_path,
        rv_ccf_pars, PSI, active_regions_number, active_region_box,
        sim_spec_line_centers, filters) = init_setup_cfg(
            logger, CONFIG_FILE, post_fix=args.out)

    (active_region_longs, active_region_lats, active_region_sizes, 
        active_region_Temp) = active_region_box
    
    # --- Setup the synthetic spectra for Spotted spectra simulation ---
    SPEC_box = setup_activate_spectra(bands, STAR, plot_flag, syn_model_path)

    ################################################################################################
    # run the simulation
    main_model(
        logger, bands, nrho, grid_size, STAR, PSI, SPEC_box, filters,
        active_regions_number, active_region_box, 
        rv_ccf_pars, syn_model_path, instrument_res, 
        cpu_use, SAVE_DIR, save_spec_flag, plot_flag, silence_pbar=False)
    
    
    # ######################################################################
    # # --- Exrtra Plotting for Diagnostic plot for RV and EWR
    # if len(sim_spec_line_centers) > 1:
    #     center_wvs = np.array([bands[band][0]/1e4 for band in bands.keys()])

    #     norm = mpl.colors.Normalize(vmin=center_wvs.min(), vmax=center_wvs.max())
    #     s_m = mpl.cm.ScalarMappable(cmap=mpl.cm.coolwarm, norm=norm)
    #     s_m.set_array([])

    #     fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.5), dpi=300)    
        
    #     amp_box = []
        
    #     for i, ss in zip(
    #         range(len(SPEC_photosphere_box)), itertools.cycle('ox>^+d*s')):
    #         band = list(bands.keys())[i]
    #         cen_wv = center_wvs[i]

    #         tab = Table.read(
    #             f'./outputs/{SAVE_DIR}/{outdir.split("/")[-1]}_{band}_RV_EWR.csv', 
    #             format='csv'
    #             )
            
    #         ax.plot(
    #             tab['Phase'], tab['RV'], '.-', 
    #             label=fr'$\sim${cen_wv:>4.2f}um {band}', 
    #             c=s_m.to_rgba(cen_wv), marker=ss, ms=2,
    #             alpha=0.7)
            
    #         amp_box.append(np.nanmax(tab['RV']) - np.nanmin(tab['RV']))
        
    #     ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    #     ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    #     ax.set_xlabel('Phase')
    #     ax.set_ylabel('RV [km/s]')
    #     ax.legend(
    #         fontsize=4, ncol=2, edgecolor='white', bbox_to_anchor=(1.05, 1.))
    #     # ax.set_ylim(-20, 20)
    #     fig.savefig(
    #         f'./outputs/{SAVE_DIR}/{outdir.split("/")[-1]}_RV_{sim_spec_line_centers[0]}_{sim_spec_line_centers[-1]}.png', 
    #         bbox_inches='tight')

    #     ## ---

    #     fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=300)    
    #     ax.plot(center_wvs, amp_box, 'o', ms=3, c='k')

    #     ax.set_xlabel('Wavelength [um]')
    #     ax.set_ylabel('RV amplitude [km/s]')
    #     ax.set_xlim(0, 3.0)
    #     # ax.set_ylim(0, np.round(np.max(amp_box))+1)

    #     fig.savefig(
    #         f'./outputs/{SAVE_DIR}/{outdir.split("/")[-1]}_RV_amp_{sim_spec_line_centers[0]}_{sim_spec_line_centers[-1]}.png', 
    #         bbox_inches='tight')


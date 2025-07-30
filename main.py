# === Standard library ===
import argparse
import configparser
import itertools
import logging
import os
import shutil
import socket
import sys
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
CONFIG_FILE = 'config_LkCa4_multi_SPOTS_study.cfg'
# CONFIG_FILE = 'config_teff_study.cfg'
# SAVE_DIR = f'RVandEWR_SizeDelT_width100'

# SimSpecWinSize = 100 # in AA
SIMSPEC_VEL_BIN_SIZE = 100/1000 # in km/s
SOLAR_RADIUS = 695700 # km
# ------------------------------------------------
hostname = socket.gethostname() 
hostname = hostname.split('.')[0].split('-')[0]

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler()
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # stream_handler.setFormatter(formatter)
        # formatter = logging.Formatter('%(message)s')
        # stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


# def main_model():

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog        = 'SOAP-TTS simulation',
        description = '''
        NaN
        ''', epilog = 'sytang@rice.edu'
        )

    parser.add_argument(
        '-cfg', dest="cfg", action="store", help="Enter the .cfg file to run.",
        type=str, default=None)
    # parser.add_argument(
    #     '-st', dest="st", action="store", help="The start dates id to run.",
    #     type=int, default=None)
    # parser.add_argument(
    #     '-ed', dest="ed", action="store", help="The end dates id to run.",
    #     type=int, default=None)
        
    # ---- Not normal setting ----
    parser.add_argument(
        '-delT', dest="delT", action="store", help="Delta T for spot.",
        type=float, default=None)
    parser.add_argument(
        '-Scover', dest="Scover", action="store", help="Spot coverage in % for visible DISK.",
        type=float, default=None)
    parser.add_argument(
        '-Lat', dest="Lat", action="store", help="Manually set the spot latitude.",
        type=float, default=None)
    parser.add_argument(
        '-Incl', dest="Incl", action="store", help="Manually set the Stellar incl. angle",
        type=float, default=None)
    parser.add_argument(
        '-out', dest="out", action="store", help="Out dur name.",
        type=str, default=None)
    parser.add_argument(
        '-plot', dest="plot", action="store", help="If make gif.",
        type=str, default=None)
    parser.add_argument(
        '-DeBug', dest="debug", action="store_true",
        help="If set, DeBug logging will be output, as well as (lots of) \
                extra plots."
        )
    
    args = parser.parse_args()
    logger = setup_logging(args.debug)

    # ------------------------------------------------------------
    if args.cfg is not None:
        if not os.path.exists(args.cfg):
            raise FileNotFoundError(f"\nConfig file {args.cfg} does not exist.")
        else:
            CONFIG_FILE = args.cfg
            logger.info(f'\nUsing config file: {CONFIG_FILE}')
    else:
        logger.info(f'\nUsing default config file: {CONFIG_FILE}')

################################################################################################
    # --------------------- INITIALIZATION -----------------------
    config = configparser.ConfigParser(inline_comment_prefixes=';')   
    config.read(CONFIG_FILE)
    save_folder_name = config.get(
        'output', 'save_folder_name', fallback='').strip()

    if save_folder_name and save_folder_name.lower() != 'none':
        if args.out is not None:
            SAVE_DIR = f'{save_folder_name}_{args.out}_{hostname}'
        else:
            SAVE_DIR = f'{save_folder_name}_{hostname}'
    else:
        SAVE_DIR = f'{args.out}_{hostname}'
        
    # ------------------------------------------------------------
    # --- Spectroscopic Parameters ---
    wv_start = config.get('spectroscopic', 'wavelength_start')
    wv_end   = config.get('spectroscopic', 'wavelength_end')
    wv_step  = config.get('spectroscopic', 'wavelength_step')
    sim_spec_window_size = config.get('spectroscopic', 'simulate_spectra_window')

    # --- Photometric Parameters ---
    filters = config.get('photometry', 'filters').split(',')
    
    # ------------------------------------------------------------
    # >>>>>>>>> temporary setup <<<<<<<<<
    path_sub = 1 if hostname == 'Swing' else 2
    RASSINE_PATH = config.get('path',f'rassine_path{path_sub}')
    SYNmast_PATH = config.get('path',f'synmast_prf{path_sub}')

    # ---- plotting parameters ----
    phx_path = config.get('path','PHOENIX_path')
    plot_flag = int(config.get('output','plot' ))
        
    save_spec_flag = int(config.get('output','save_spec' ))

    # ---- model basic setup ----
    grid_size = int(config.get('main','grid' ))
    nrho      = int(config.get('main','nrho' )) # Resolution for the spot's circumference
    instrument_res = int(config.get('main','instrument_reso' ))
    cpu_use = int(config.get('main','cpu' ))
    cpu_use = 6 if hostname == 'Swing' else 31
    
    stellar_radius = float(config.get('star','radius' )) * SOLAR_RADIUS
    rotation_period = float(config.get('star','prot' ))
    incl = float(config.get('star','I'))
    psi  = float(config.get('star','psi'))
        
    Temp = int(config.get('star','Tstar'))
    # Tdiff_spot = int(config.get('star','Tdiff_spot'))
    # Tdiff_faculae = int(config.get('star','Tdiff_faculae'))
    logg = float(config.get('star', 'logg' ))
    FeH  = float(config.get('star', 'FeH' ))

    if args.Incl is not None: incl = float(args.Incl)
        
    # --- Overwrite pars if been args updated ---
    # for model grid generation
    # if args.delT is not None: Tdiff_spot = int(args.delT)

    ######################################################################
    # --- Count and Load Active Regions from config ---
    cool_active_regions_number = int(
        config.get('cool_active_regions', 'spot_number' ))
    hot_active_regions_number = int(
        config.get('hot_active_regions', 'spot_number' ))
    active_regions_number = cool_active_regions_number + hot_active_regions_number

    logger.info(
        f"\nFound {cool_active_regions_number} cool region(s), "
        f"and {hot_active_regions_number} hot region(s).")

    # -----
    # Read properties of each active region
    cool_active_region_longs = np.array([float(val) for val in re.split(
        r'[,\s]+', config.get('cool_active_regions', 'longitudes').strip()
        ) if val])    
    cool_active_region_lats = np.array([float(val) for val in re.split(
        r'[,\s]+', config.get('cool_active_regions', 'latitudes').strip()
        ) if val])
    cool_active_region_sizes = np.array([float(val) for val in re.split(
        r'[,\s]+', config.get('cool_active_regions', 'sizes').strip()
        ) if val])
    
    _pars = len(cool_active_region_longs) + len(cool_active_region_lats) + len(cool_active_region_sizes)
    if _pars != 3 * cool_active_regions_number:
        raise ValueError(
            f"Number of pars in longitudes or latitudes or sizes for cool active regions ({_pars}) "
            f"does not match the number of active regions ({cool_active_regions_number})."
            )
    # Tphoto = 4100K
    # spot_temp_dict = {
    #     'a': 445,  
    #     'b': 323,  
    #     'c': 179, 
    #     'd': 290, 
    #     'e': 704, 
    #     'f': 800, 
    # }
    
    # Tphoto = 3800K
    # spot_temp_dict = {
    #     'a': 343,  
    #     'b': 238,  
    #     'c': 124, 
    #     'd': 150, 
    #     'e': 343, 
    #     'f': 659, 
    # }
    
    # Tphoto = 3700K
    spot_temp_dict = {
        'a': 321,  
        'b': 222,  
        'c': 115, 
        'd': 128, 
        'e': 283, 
        'f': 491, 
    }

    # spot_temp_dict = {
    #     'a': 1000,  
    #     'b': 500,  
    #     'c': 200, 
    #     'd': 200, 
    #     'e': 500, 
    #     'f': 1000, 
    # }
    cool_active_region_dTemp = np.array([int(spot_temp_dict[val]) for val in re.split(
        r'[,\s]+', config.get('cool_active_regions', 'Tcool_spots').strip()
        ) if val])

    # Get the sort order (indices) based on dTemp
    # for cool spot, the lowest temperature will be processed first,
    #   thus, the descending order
    sort_idx = np.argsort(cool_active_region_dTemp)[::-1]  # ascending; use [::-1] for descending

    # Apply the same order to all arrays
    cool_active_region_dTemp = cool_active_region_dTemp[sort_idx]
    cool_active_region_longs = cool_active_region_longs[sort_idx]
    cool_active_region_lats = cool_active_region_lats[sort_idx]
    cool_active_region_sizes = cool_active_region_sizes[sort_idx]
    cool_active_region_Temp = Temp - cool_active_region_dTemp

    # -----
    hot_active_region_longs = np.array([float(val) for val in re.split(
        r'[,\s]+', config.get('hot_active_regions', 'longitudes').strip()
        ) if val])
    hot_active_region_lats = np.array([float(val) for val in re.split(
        r'[,\s]+', config.get('hot_active_regions', 'latitudes').strip()
        ) if val])
    hot_active_region_sizes = np.array([float(val) for val in re.split(
        r'[,\s]+', config.get('hot_active_regions', 'sizes').strip()
        ) if val])
    _pars = len(hot_active_region_longs) + len(hot_active_region_lats) + len(hot_active_region_sizes)
    if _pars != 3 * hot_active_regions_number:
        raise ValueError(
            f"Number of pars in longitudes or latitudes or sizes for hot active regions ({_pars}) "
            f"does not match the number of active regions ({hot_active_regions_number})."
            )
    hot_active_region_dTemp = np.array([int(spot_temp_dict[val]) for val in re.split(
        r'[,\s]+', config.get('hot_active_regions', 'Thot_faculae').strip()
        ) if val])
    
    # Get the sort order (indices) based on dTemp
    # for hot spot (plages), the hottest spot will be process first,
    # thus, the descending order
    sort_idx = np.argsort(hot_active_region_dTemp)[::-1]  # ascending; use [::-1] for descending

    # Apply the same order to all arrays
    hot_active_region_dTemp = hot_active_region_dTemp[sort_idx]
    hot_active_region_longs = hot_active_region_longs[sort_idx]
    hot_active_region_lats = hot_active_region_lats[sort_idx]
    hot_active_region_sizes = hot_active_region_sizes[sort_idx]
    hot_active_region_Temp = Temp + hot_active_region_dTemp

    # The "size" of the active region used in SOAP is the radius of the spot [Rstar],
    #   which is not easy to interpret. Therefore, we let the user to input 
    #   the % of the spot area coverage to the visible disk as the "size".
    #   The input area coverage % is then transfer back to spot radius via:
    #     >>>  sqrt(area% / 100).
    # With size1 as the size of the spot and S1 as the percentage of the spot coverage:
    #   >>> S1 = Area_spot / Area_visible_disk 
    #          = pi*(size1*Rstar)**2 / pi*(Rstar**2) 
    #          = size1**2
    #  --> size1 = sqrt(S1)
    # btw., visible hemisphere is twice the size of a visible disk......
    cool_active_region_sizes_percent = cool_active_region_sizes.copy()
    cool_active_region_sizes = np.sqrt(np.array(cool_active_region_sizes)/100)

    hot_active_region_sizes_percent = hot_active_region_sizes.copy()
    hot_active_region_sizes = np.sqrt(np.array(hot_active_region_sizes)/100)

    # --- Overwrite pars if been args updated ---
    # if args.Scover is not None:
    #     _spot_cover_size = np.sqrt(np.array(args.Scover)/100)
    #     active_region_sizes = [_spot_cover_size]
    
    # if args.Lat is not None:
    #     active_region_lats = [float(args.Lat)]

    active_region_longs = np.concatenate([cool_active_region_longs, hot_active_region_longs])
    active_region_lats = np.concatenate([cool_active_region_lats, hot_active_region_lats])
    active_region_sizes = np.concatenate([cool_active_region_sizes, hot_active_region_sizes])
    active_region_sizes_percent = np.concatenate([cool_active_region_sizes_percent, hot_active_region_sizes_percent])
    active_region_Temp = np.concatenate([cool_active_region_Temp, hot_active_region_Temp])
    active_region_dTemp = np.concatenate([-cool_active_region_dTemp, +hot_active_region_dTemp])

    active_region_box = [
        active_region_longs, active_region_lats, 
        active_region_sizes, active_region_Temp]

    # cool_active_region_box = [
    #     cool_active_region_longs, cool_active_region_lats, 
    #     cool_active_region_sizes, cool_active_region_dTemp]

    # hot_active_region_box = [
    #     hot_active_region_longs, hot_active_region_lats, 
    #     hot_active_region_sizes, hot_active_region_dTemp]

    ######################################################################
    
    dirs_to_create = [
        "syn_spec",
        "syn_spec/RASSINE_PHX",
        "syn_spec/SimUse",
        "syn_spec/Photometry_use_Spec",
        f"outputs/{SAVE_DIR}",
        f"outputs/{SAVE_DIR}/normal_spec",
        f"outputs/{SAVE_DIR}/spec",
        f"outputs/{SAVE_DIR}/ccf_template",
        "outputs/CCF",
    ]

    for path in dirs_to_create:
        os.makedirs(path, exist_ok=True)
        
    ######################################################################
    # --- RV CCF template parameters ---
    # use_photosphere_teff = int(config.get('ccf', 'Use_photosphere_teff'))

    # teff = float(config.get('ccf', 'Teff'))
    # logg = float(config.get('ccf', 'logg'))
    # feh = float(config.get('ccf', 'FeH'))
    # vsini = float(config.get('ccf', 'vsini'))
    # vsini = STAR.vsini if vsini == -1 else vsini  # override if -1

    # rv_ccf_pars = np.array([teff, logg, feh, vsini, instrument_res])
    # if use_photosphere_teff == 1: rv_ccf_pars[:] = 99999
    rv_ccf_pars = np.zeros(5) + 99999

        # -- Estimated the Teff and use it for the LD Teff --
        # spot_portion = 0 # visible "DISK" portion
        # for ss in active_region_sizes:
        #     spot_portion = spot_portion + ss**2

        # est_teff = self.Temp**4*(1-spot_portion) \
        #     + self.Tspot**4*spot_portion
        # est_teff = int(est_teff**0.25)
        # print(f'\nEstimated Teff for calculating Limb darkening: {est_teff:1.0f} K') 
    
    # -- Phase at which the signal will be generated --
    phase_step  = float(config.get('output','ph_step'))
    phase_in    = config.get('output','ph_in', fallback='').strip()
################################################################################################
    # ------------- Setup wavelength to run for spec/RV/photometry --------------
    # bands hold wavelength info for simulation
    sim_spec_line_centers = np.arange(
        int(wv_start), int(wv_end)+1, int(wv_step)
        ) # in Angstroms

    # Create the dictionary
    bands = {
        f"w{wave_center // 100}": [
            float(wave_center), 
            float(sim_spec_window_size), 
            'rv', 
            'flux', 
            'phx'
        ]
        for wave_center in sim_spec_line_centers
    }
    # bands = {}
    # --- Add Equivalent Width Ratio (EWR) band ---
    # bands['ewr_FeOH'] = [15631.35, 30, 'rv', 'flux', 'synmast']
    # bands['ewr_FeOH'] = [15631.35, 100, 'rv', 'intensity', 'synmast']
    bands['ewr_FeOH'] = [15557, 90, 'rv', 'intensity', 'synmast']

    # --- band pass for photometry ---
    if (len(filters) == 1 and 'None' in filters[0]) or (len(filters) == 1 and filters[0]==''):
        logger.info('\nNo filters specified. Will not generate photometry.')    
    else:
        logger.info(f'\nDetect filers. Will generate photometry for {filters}.')
    
        # Load filter profiles from SVO
        filter_profiles = [svo.get_pyphot_filter(filt) for filt in filters]
        
        # Extract and flatten all wavelengths from filters
        filter_wavelengths = [f.wavelength for f in filter_profiles]
        filter_wavelengths = np.concatenate(filter_wavelengths)
        filter_wavelengths = filter_wavelengths * 10 # nm to AA
        
        # Calculate filter range
        filter_wv_min = np.min(filter_wavelengths)
        filter_wv_max = np.max(filter_wavelengths)
        
        if filter_wv_min < 500 or filter_wv_max > 50000:
            logger.warning(
                f'Filter wavelength range {filter_wv_min:.1f} - {filter_wv_max:.1f} AA is not in the range of 500 - 50000 AA. '
                'Please check the filter wavelengths.'
                )

        photo_center = int((filter_wv_max + filter_wv_min) / 2)
        photo_width = int((filter_wv_max - filter_wv_min) / 2)

        # bands  = {}
        bands['photo'] = [photo_center, photo_width, 'photo', 'intensity', 'phx']

    # --- synthetic spectra paths ---
    syn_model_path = {
        'phx': RASSINE_PATH, 
        'synmast': SYNmast_PATH
        }
    
    if args.plot is not None:
        plot_flag = 1 if 't' in args.plot.lower() else 0

    # --- Stellar condition setup ---
    STAR = soapcore.Star(
        prot = rotation_period,
        vrot = (2 * np.pi*stellar_radius) / (rotation_period * 86400),
        incl = incl,
        psi  = psi,
        rad_sun = SOLAR_RADIUS,
        rad  = stellar_radius,
        Temp = Temp,
        Tdiff_spot = cool_active_region_dTemp,
        Tdiff_faculae = hot_active_region_dTemp,
        logg = logg,
        FeH  = FeH,
        bands2use = bands,
        )

    if 'photo' in bands.keys(): STAR.filters = filters

    STAR.vsini = STAR.vrot * np.sin(np.deg2rad(STAR.incl))
    STAR.inst_res = instrument_res

    # --- PHOENIX mode path ---
    # Currently, we almost always use the PHOENIX spectra for RV CCF
    STAR.phx_path = Path(f'{phx_path}/PHOENIX-ACES-AGSS-COND-2011').expanduser()
    os.makedirs(STAR.phx_path, exist_ok=True)
    os.makedirs(STAR.phx_path / 'SpecInt', exist_ok=True)
    os.makedirs(STAR.phx_path / 'SpecFlux', exist_ok=True)

    # --- Check how fine the grid is ---
    soapcore.grid_resolution_check(grid_size)

    # --- Get flatten synthetic spectra for RV CCF ---
    #     >>>> spec_type == phx for now, can update in the future <<<<
    #     >>>> as using PHX for CCF templates, syn_spec_path == RASSINE_PATH <<<<
    STAR.find_fluxspec_flatten(
        SIMSPEC_VEL_BIN_SIZE, plot_flag, syn_model_path)
    
    # --- Setup the synthetic spectra for Spotted spectra simulation ---
    # check model spectra types, e.g., ['photo' 'flux' 'intensity']
    intensity_mode_flags = np.unique(np.vstack(list(bands.values()))[:, 3])

    temp_list = np.concatenate(
        [[STAR.Temp], STAR.Tspot, STAR.Tfaculae]
        )
        
    if 'flux' in intensity_mode_flags:
        # Flux mode: disk integration with limb darkening
        line_centers = [band[0] for band in bands.values()]
        line_widths = [band[1] for band in bands.values()]

        STAR.get_quadratic_limbdarken(
            temp_list, line_centers, line_widths)
        # LD_coeff_spot = STAR.get_quadratic_limbdarken(
        #     STAR.Tspot, line_centers, line_widths)
        # LD_coeff_faculae = STAR.get_quadratic_limbdarken(
        #     STAR.Tfaculae, line_centers, line_widths)
  
        # -- Setup the synthetic spectra --
        STAR.find_fluxspec(SIMSPEC_VEL_BIN_SIZE, plot_flag, syn_model_path)

    if 'intensity' in intensity_mode_flags:
        STAR.find_intensityspec(SIMSPEC_VEL_BIN_SIZE, plot_flag, syn_model_path)

    # SPEC_photosphere_box = []
    # SPEC_spot_box = []
    # SPEC_plage_box = []
    SPEC_box = []

    for band in bands.keys():
        spec_box_tt = {
            int(tt): soapcore.Spec(
                star = STAR, band = band, temperature=tt,
                intORflux=bands[band][3], model_type=bands[band][4]
                ) for tt in temp_list
            }
        SPEC_box.append(spec_box_tt)
        # SPEC_photosphere_box.append(
        #     soapcore.Spec(
        #         type = 'photosphere', star = STAR, band = band, 
        #         intORflux=bands[band][3], model_type=bands[band][4]
        #         ))
        # SPEC_spot_box.append(
        #     soapcore.Spec(
        #         type = 'spot', star = STAR, band = band, 
        #         intORflux=bands[band][3], model_type=bands[band][4]
        #         ))
        # SPEC_plage_box.append(
        #     soapcore.Spec(
        #         type = 'plage', star = STAR, band = band, 
        #         intORflux=bands[band][3], model_type=bands[band][4]
        #         ))
          
    ######################################################################

    # Read the phase in the file given in the "phase_in" argument in "config.cfg"
    if phase_in and phase_in.lower() != 'none':
        with open(phase_in) as f:
            PSI = np.array([float(l.split()[0]) for l in f.readlines()])
    # Otherwise create a evenly sampled phase between 0 and 1 with the step given
    # by the "ph_step" argument in "config.cfg"
    elif phase_step > 0:
        PSI = np.arange(-0.5, 0.51, phase_step)
        # PSI = np.arange(-0.1, 0.5, phase_step)
    else:
        logger.warning("WARNING: put a positive phase step ('ph_step' argument in 'config.cfg') \
or give a file name that contain the phases at which you want to compute the model \
('phase_in' argument in 'config.cfg')")

    # --- Run model per active region ---
    for i in range(len(bands.keys())):
        start_time = time()
        band = list(bands.keys())[i]
        intORflux = bands[band][3]
        model_type = bands[band][4]

        # if intORflux.lower() == 'flux':

        #     STAR.limba1 = [
        #         LD_coeff_photo[band][0],
        #         LD_coeff_spot[band][0],
        #         LD_coeff_faculae[band][0]
        #     ]
        #     STAR.limba2 = [
        #         LD_coeff_photo[band][1],
        #         LD_coeff_spot[band][1],
        #         LD_coeff_faculae[band][1]
        #     ]
        # elif intORflux.lower() == 'intensity':
        #     STAR.limba1 = [0.0, 0.0, 0.0]
        #     STAR.limba2 = [0.0, 0.0, 0.0]

        logger.info(f'\n\nNow calculating the activity signal for the band: {band}')

        # SPEC_photosphere = SPEC_photosphere_box[i]
        # SPEC_spot = SPEC_spot_box[i]
        # SPEC_plage = SPEC_plage_box[i]
        SPEC = SPEC_box[i]

        (visible_phi, integrated_stellar_spectrum, quite_2d_surface_flux_sum, 
         final_2d_surface_flux_sum) = soapcore.calculate_activity_signal_spec(
            nrho, grid_size, STAR, PSI, 
            # SPEC_photosphere, SPEC_spot, SPEC_plage,
            SPEC,
            active_regions_number, active_region_box, 
            instrument_res, cpu_use, SAVE_DIR, band, save_spec_flag, 
            intORflux.lower()
            )

        # --- flatten spectra ---
        if band != 'photo':
            integrated_stellar_spectrum = flatten_model_spec(
                STAR, integrated_stellar_spectrum, visible_phi, band, 
                intORflux, model_type, SAVE_DIR, plot_flag)
        
            if save_spec_flag == 1:
                np.save(
                    f'./outputs/{SAVE_DIR}/spec/{band}_flat_final_spec.npy', 
                    integrated_stellar_spectrum)

        if band == 'ewr_FeOH':        
            # depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, depth_l_fe2, depth_l_oh0 = soap2.get_line_depth(
            #     STAR, SPECstar_tot, visible_phi, band
            #     )
            ohfe_ratio = soapcore.get_equivalent_width(
                STAR, integrated_stellar_spectrum, visible_phi, band, intORflux, model_type)
            # print(f'--> {band} EWR amp. {np.max(ohfe_ratio) - np.min(ohfe_ratio):1.4f}')
        else:
            depth_d_fe1, depth_d_fe2, depth_d_oh0 = [None, None, None]
            ohfe_ratio = np.zeros_like(visible_phi) + np.nan

        # -- Calculate CCF RV --
        if band != 'photo':
            logger.info('\nCalculating the RV with CCF...\n')
            ccf_rv_shift = soapcore.cal_ccf_rv(
                integrated_stellar_spectrum, STAR, visible_phi, band, intORflux, model_type, 
                rv_ccf_pars, SYNmast_PATH, cpu_use, SAVE_DIR, SIMSPEC_VEL_BIN_SIZE, 
                plot_flag)
        else:
            ccf_rv_shift = np.zeros_like(visible_phi) + np.nan
        # print(f'--> {band} RV amp. {np.nanmax(ccf_rv_shift) - np.nanmin(ccf_rv_shift):1.6f} km/s')

        # --- Calculate relative photometric variation ---
        if band == 'photo':
            lightcurves_mag, lightcurves_flux = soapcore.get_photometric(
                integrated_stellar_spectrum, STAR, visible_phi, band, intORflux, model_type
            )
        
        
        ###########################################################################
        # --- Plot the simulated results ---
        # Clean string formatting for output folder names
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
        
        text1 = '{0:18} = {1:<6d}, {2:18} = {3:<6d}\n'.format(    'Grid res.',  grid_size,      'Circumference res.', nrho) + \
                '{0:18} = {1:<6d}, {2:18} = {3:<6.2f}\n'.format(  'Inst. res.', instrument_res, 'St Prot [d]', STAR.prot) + \
                '{0:18} = {1:<6.2f}, {2:18} = {3:<6.1f}\n'.format('St radius [Rsun]', STAR.rad/SOLAR_RADIUS,     'St incl', STAR.incl) + \
                '{0:18} = {1:<6.2f}, {2:18} = {3:<6.0f}\n'.format('St vsini [km/s]', STAR.vsini,         'St T_photo [K]', STAR.Temp) + \
                '{0:18} = {1:<6}, {2:5} = {3:<4.2f}, {4:5} = {5:<4.2f}'.format('Syn template', model_type, 'logg', STAR.logg, '[Fe/H]', 0.0)
        # '{:12} = {}\n'.format('LimDark lin', [float(np.round(ll, 3)) for ll in STAR.limba1]) + \
        # '{:12} = {}\n'.format('LimDark quad', [float(np.round(ll, 3)) for ll in STAR.limba2]) + \

        # if active_regions_number == 1:
        #     text2 = 'ActR {0}: lon={1:<5.1f}, lat={2:<4.1f}, size = {3:<5.3f} [Rsun] (={4:.2f}%)\n'.format(
        #         '1',active_region_longs[0],active_region_lats[0],active_region_sizes[0],active_region_sizes_percent[0])
        # elif active_regions_number == 2:
        #     text2 = 'ActR {0}: lon={1:<5.1f}, lat={2:<4.1f}, size = {3:<5.3f} [Rsun] (={4:.2f}%)\n'.format(
        #         '1',active_region_longs[0],active_region_lats[0],active_region_sizes[0],active_region_sizes_percent[0]) +\
        #             'ActR {0}: lon={1:<5.1f}, lat={2:<4.1f}, size = {3:<5.3f} [Rsun] (={4:.2f}%)\n'.format(
        #         '2',active_region_longs[1],active_region_lats[1],active_region_sizes[1],active_region_sizes_percent[1])
        # elif active_regions_number == 3:
        #     text2 = 'ActR {0}: lon={1:<5.1f}, lat={2:<4.1f}, size = {3:<5.3f} [Rsun] (={4:.2f}%)\n'.format(
        #         '1',active_region_longs[0],active_region_lats[0],active_region_sizes[0],active_region_sizes_percent[0]) +\
        #             'ActR {0}: lon={1:<5.1f}, lat={2:<4.1f}, size = {3:<5.3f} [Rsun] (={4:.2f}%)\n'.format(
        #         '2',active_region_longs[1],active_region_lats[1],active_region_sizes[1],active_region_sizes_percent[1]) +\
        #             'ActR {0}: lon={1:<5.1f}, lat={2:<4.1f}, size = {3:<5.3f} [Rsun] (={4:.2f}%)\n'.format(
        #         '3',active_region_longs[2],active_region_lats[2],active_region_sizes[2],active_region_sizes_percent[2])
        # else:
        #     text2 = 'Number of ActR > 3'
        # text2 = '{} active rigons:\n'.format(active_regions_number) + \
        #         '   lon [deg]= {}\n'.format(active_region_longs) + \
        #         '   lat [deg]= {}\n'.format(active_region_lats) + \
        #         '   size [%]= {}\n'.format(active_region_sizes_percent) + \
        #         '   Temp [K]= {}\n'.format(active_region_dTemp)
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
            tab_save = Table(
                data=(visible_phi, ccf_rv_shift, ohfe_ratio), 
                names=('Phase', 'RV', 'EWR'))
            tab_save['Phase'].format = '%.3f'
            tab_save['RV'].format = '%.4f'
            tab_save['EWR'].format = '%.5f'
        
        else:
            tab_save = Table([visible_phi], names=['Phase'])
            for i in range(len(filters)):
                filter_name = filters[i].split('/')[1]
                tab_save[f'{filter_name}_mag'] = lightcurves_mag[i]
                tab_save[f'{filter_name}_mag'].format = '%.6f'

                tab_save[f'{filter_name}_flux'] = lightcurves_flux[i]
                tab_save[f'{filter_name}_flux'].format = '%.6f'

            tab_save['Phase'].format = '%.3f'
            
        tab_save.write(
            f'./outputs/{SAVE_DIR}/{outdir.split("/")[-1]}_{band}_RV_EWR.csv', 
            format='csv', overwrite=True)

        if plot_flag == 1:
            if band != 'photo':
                vis.spot_view(
                    STAR, grid_size, visible_phi, integrated_stellar_spectrum, ccf_rv_shift,
                    final_2d_surface_flux_sum, outdir, cpu_use, ohfe_ratio, text1, text2,
                    band, intORflux, model_type)
            else:
                visible_phi = np.array(visible_phi)
                fig, ax = plt.subplots(2, 1, figsize=(4, 3), dpi=300, )    
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
        
        end_time = time()
        # print run time in min
        print(f'\n--> {band} run time: {(end_time-start_time)/60:1.2f} min')
    
    ######################################################################
    ## --- Diagnostic plot for RV and EWR
    if len(sim_spec_line_centers) > 1:
        center_wvs = np.array([bands[band][0]/1e4 for band in bands.keys()])

        norm = mpl.colors.Normalize(vmin=center_wvs.min(), vmax=center_wvs.max())
        s_m = mpl.cm.ScalarMappable(cmap=mpl.cm.coolwarm, norm=norm)
        s_m.set_array([])

        fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.5), dpi=300)    
        
        amp_box = []
        
        for i, ss in zip(
            range(len(SPEC_photosphere_box)), itertools.cycle('ox>^+d*s')):
            band = list(bands.keys())[i]
            cen_wv = center_wvs[i]

            tab = Table.read(
                f'./outputs/{SAVE_DIR}/{outdir.split("/")[-1]}_{band}_RV_EWR.csv', 
                format='csv'
                )
            
            ax.plot(
                tab['Phase'], tab['RV'], '.-', 
                label=fr'$\sim${cen_wv:>4.2f}um {band}', 
                c=s_m.to_rgba(cen_wv), marker=ss, ms=2,
                alpha=0.7)
            
            amp_box.append(np.nanmax(tab['RV']) - np.nanmin(tab['RV']))
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_xlabel('Phase')
        ax.set_ylabel('RV [km/s]')
        ax.legend(
            fontsize=4, ncol=2, edgecolor='white', bbox_to_anchor=(1.05, 1.))
        # ax.set_ylim(-20, 20)
        fig.savefig(
            f'./outputs/{SAVE_DIR}/{outdir.split("/")[-1]}_RV_{sim_spec_line_centers[0]}_{sim_spec_line_centers[-1]}.png', 
            bbox_inches='tight')

        ## ---

        fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=300)    
        ax.plot(center_wvs, amp_box, 'o', ms=3, c='k')

        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel('RV amplitude [km/s]')
        ax.set_xlim(0, 3.0)
        # ax.set_ylim(0, np.round(np.max(amp_box))+1)

        fig.savefig(
            f'./outputs/{SAVE_DIR}/{outdir.split("/")[-1]}_RV_amp_{sim_spec_line_centers[0]}_{sim_spec_line_centers[-1]}.png', 
            bbox_inches='tight')


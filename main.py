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
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from astropy.table import Table

# === Custom modules ===
import spotact_core, spotact_utils
import spotact_spot as stsp
import visualization as vis


# CONFIG_FILE = 'config_LkCa4base_SizeDelT_study.cfg'
# CONFIG_FILE = 'config_LkCa4_2SPOTS_study.cfg'
# CONFIG_FILE = 'config_LkCa4_multi_SPOTS_study.cfg'
# CONFIG_FILE = 'config_LkCa4_multi_SPOTS_fitting_8spots-1.cfg'
# CONFIG_FILE = 'config_LkCa4_multi_SPOTS_donati_oeg_21spots.cfg'
# CONFIG_FILE = 'config_LkCa4_one_SPOT_demo.cfg'
# CONFIG_FILE = 'config_2TemplateFit_test_1.cfg'
# CONFIG_FILE = 'config_2TemplateFit_test_exsplosyn-1.cfg'
# CONFIG_FILE = 'config_LimbDarkeningMu_test.cfg'
# CONFIG_FILE = 'config_Chenming_HD113816.cfg'
# SAVE_DIR = f'RVandEWR_SizeDelT_width100'
# CONFIG_FILE = 'config_LkCa4_multi_SPOTS_mimicking_ZDI.cfg'
# CONFIG_FILE = 'config_LkCa4_multi_SPOTS_chi2_fitting_ZDI_Tinit_SBPLX.cfg'
# CONFIG_FILE = 'config_LkCa4_multi_SPOTS_chi2_fitting_ZDI_Tinit_BOBYQA.cfg'
# CONFIG_FILE = 'config_max_size.cfg'
CONFIG_FILE = 'config_LkCa4_multi_SPOTS_mcmc_fitting_ZDI_Tinit-5spots.cfg'

# SimSpecWinSize = 100 # in AA
SOLAR_RADIUS = 695700 # km
SYNMAST_FeOH = 'LisaGrid_synmast_v1.9_spirou_marcs'
SYNMAST_Ti = 'ChrisGrid_Ti_synmast_v1.9_spirou_marcs'
SYNMAST_CO = 'ChrisGrid_CO_synmast_v1.9_spirou_marcs'
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
    *,
    host_override: str | None = None,  # optional convenience for testing
):
    """
    Initialize and parse configuration, set up directories, and prepare simulation parameters.

    Returns:
        (STAR, bands, SAVE_DIR, nrho, grid_size, instrument_res, cpu_use, plot_flag,
         save_spec_flag, sim_spec_window_size, syn_model_path, rv_ccf_pars, PSI,
         active_regions_number, active_region_box, sim_spec_line_centers, filters)
    """ 
    # --- Read config ---
    cfg = configparser.ConfigParser(inline_comment_prefixes=';')
    read_ok = cfg.read(config_file)
    if not read_ok:
        raise FileNotFoundError(f"Config file not found or unreadable: {config_file}")

    # --- Host & save dir ---
    hostname = (host_override or socket.gethostname()).split('.')[0].split('-')[0]
    save_folder_name = cfg.get('output', 'save_folder_name', fallback='').strip()
    SAVE_DIR = f"{save_folder_name}_{post_fix}_{hostname}" if post_fix else f"{save_folder_name}_{hostname}"

    # ==========================================================
    # HPC RESTART FIX: Swap dynamic SLURM node name for "nots"
    # Example: "MyDir_15spots_bb3u16c1" -> "MyDir_15spots_nots"
    # ==========================================================
    # if hostname is not "Swing" it "taurus", replace the last part of SAVE_DIR with "nots"
    if hostname not in ["Swing", "taurus"]:
        SAVE_DIR = SAVE_DIR.rsplit('_', 1)[0] + "_nots"

    # --- Spectroscopic parameters ---
    wv_start = cfg.getint('spectroscopic', 'wavelength_start')
    wv_end   = cfg.getint('spectroscopic', 'wavelength_end')
    wv_step  = cfg.getint('spectroscopic', 'wavelength_step')
    sim_spec_window_size = cfg.getfloat('spectroscopic', 'simulate_spectra_window')

    sim_spec_line_centers = np.arange(wv_start, wv_end + 1, wv_step, dtype=np.float32)

    # --- Photometry parameters ---
    filters_raw = cfg.get('photometry', 'filters', fallback='').strip()
    filters = [f.strip() for f in filters_raw.split(',')] if filters_raw else []
    has_filters = len(filters) > 0 and filters[0].lower() not in ('', 'none')

    # --- Bands setup ---
    bands = {
        f"w{int(w)//100}": [float(w), sim_spec_window_size, 'rv', 'flux', 'phx']
        for w in sim_spec_line_centers
    }
    bands = {}
    bands['ewr_FeOH'] = [15557.0, 94.0, 'rv', 'intensity', 'synmast']
    # bands['ewr_FeOH'] = [15557.0, 90.0, 'rv', 'flux', 'synmast']
    # bands['spec_FeOH'] = [15562.5, 108, 'rv', 'intensity', 'synmast']
    # bands['spec_K_CO'] = [23138.0, 81.0, 'rv', 'intensity', 'synmast']
    # bands['spec_K_Ti'] = [22263.0, 52.0, 'rv', 'intensity', 'synmast']

    # if has_filters:
    #     logger.info(f'\nDetect filters. Will generate photometry for {filters}.')
    #     try:
    #         from pyphot import svo  # import here so unit tests without pyphot still pass init for spec-only runs
    #         profiles = [svo.get_pyphot_filter(filt) for filt in filters]
    #         # nm -> Å
    #         fwv = np.concatenate([p.wavelength for p in profiles]).astype(np.float32) * 10.0
    #         wmin, wmax = float(np.min(fwv)), float(np.max(fwv))
    #         if wmin < 500 or wmax > 50000:
    #             logger.warning(f'Filter wavelength range {wmin:.1f}–{wmax:.1f} Å is outside 500–50000 Å.')
    #         photo_center = int((wmax + wmin) * 0.5)
    #         photo_width  = int((wmax - wmin) * 0.5)
    #         bands['photo'] = [float(photo_center), float(photo_width), 'photo', 'intensity', 'phx']
    #     except Exception as e:
    #         logger.warning(f'Photometry filters could not be loaded ({e}). Photometry will be skipped.')
    #         has_filters = False
    # else:
    #     logger.info('No filters specified. Will not generate photometry.')
    if has_filters:
        logger.info(f'\nDetect filters. Will generate photometry for {filters}.')
        try:
            from pyphot import svo, Filter
            from pyphot import config
            unit = config.units

            cache_dir = './bandpass_cash'
            os.makedirs(cache_dir, exist_ok=True)
            
            profiles_fwv = []
            
            for filt in filters:
                safe_fname = filt.replace('/', '_') + '.csv'
                cache_path = os.path.join(cache_dir, safe_fname)
                
                if os.path.exists(cache_path):
                    # 1. Read the top line manually to extract the dtype
                    with open(cache_path, 'r') as f:
                        first_line = f.readline()
                        # Extract dtype from something like "# dtype:photon"
                        date_type = first_line.strip().split(':')[1]
                        
                    # 2. Read the numeric array (np.loadtxt automatically skips lines starting with '#')
                    data = np.loadtxt(cache_path, delimiter=',')
                    fwv_angstrom = data[:, 0]
                    # transmit = data[:, 1]  # if you need the transmission array here as well
                    
                else:
                    # Download, save to cache, and extract wavelength
                    logger.info(f"Downloading filter {filt} from SVO to cache...")
                    filt_obj = svo.get_pyphot_filter(filt)
                    
                    # Force conversion to Angstroms immediately
                    fwv_angstrom = filt_obj.wavelength.to('AA').value
                    date_type = filt_obj.dtype  # 'energy' or 'photon'
                    
                    save_data = np.column_stack((
                        fwv_angstrom, 
                        filt_obj.transmit
                    ))
                    
                    # Embed the dtype into a multi-line header
                    header_text = f"dtype:{date_type}\nwavelength_AA,transmission"
                    
                    np.savetxt(
                        cache_path, 
                        save_data, 
                        delimiter=',', 
                        header=header_text
                        # The default is comments='# ', so the file header will look like:
                        # # dtype:photon
                        # # wavelength_AA,transmission
                    )
                
                profiles_fwv.append(fwv_angstrom)

            # Concatenate all wavelengths. No * 10.0 needed since they are already in Angstroms.
            fwv = np.concatenate(profiles_fwv).astype(np.float32)
            wmin, wmax = float(np.min(fwv)), float(np.max(fwv))
            
            if wmin < 500 or wmax > 50000:
                logger.warning(f'Filter wavelength range {wmin:.1f}–{wmax:.1f} Å is outside 500–50000 Å.')
            
            photo_center = int((wmax + wmin) * 0.5)
            photo_width  = int((wmax - wmin) * 0.5)
            bands['photo'] = [float(photo_center), float(photo_width), 'photo', 'intensity', 'phx']
            
        except Exception as e:
            logger.warning(f'Photometry filters could not be loaded ({e}). Photometry will be skipped.')
            has_filters = False
    else:
        logger.info('No filters specified. Will not generate photometry.')


    # --- Paths ---
    if hostname == 'Swing':
        path_sub = 1
    elif hostname == 'taurus':
        path_sub = 2
    else: 
        path_sub = 3
    print('hostname: ', hostname)
    RASSINE_PATH = cfg.get('path', f'rassine_path{path_sub}')
    SYNmast_PATH = cfg.get('path', f'synmast_prf{path_sub}')
    phx_path     = cfg.get('path', 'PHOENIX_path')
    syn_model_path = {
        'phx': RASSINE_PATH, 
        'synmast': {
            'FeOH': os.path.join(SYNmast_PATH, SYNMAST_FeOH),
            'Ti': os.path.join(SYNmast_PATH, SYNMAST_Ti),
            'CO': os.path.join(SYNmast_PATH, SYNMAST_CO)
        }
    }

    # --- Output & model params ---
    plot_flag      = cfg.getint('output', 'plot')
    save_spec_flag = cfg.getint('output', 'save_spec')
    grid_size      = cfg.getint('main', 'grid')
    nrho           = cfg.getint('main', 'nrho')
    instrument_res = cfg.getint('main', 'instrument_reso')

    # cpu_use = 6 if hostname == 'Swing' else (int(os.cpu_count()*0.9) or 8)
    # cpu_use = 6 if hostname == 'Swing' else 30
    cpu_use = int(cfg.get('main', 'cpu'))

    spec_oversample_factor = cfg.getint('spectroscopic', 'spec_oversample_factor')
    # ====================================================================
    # DYNAMIC VELOCITY BIN CALCULATION
    # ====================================================================
    c_kms = 299792.458
    if instrument_res > 0:
        inst_fwhm_kms = c_kms / instrument_res
        sim_spec_vel_bin_size = inst_fwhm_kms / spec_oversample_factor

        # Hard cap: Maximum 1.0 km/s to prevent CCF degradation
        if sim_spec_vel_bin_size > 1.0:
            sim_spec_vel_bin_size = 1.0
            
        # Hard cap: Minimum 0.1 km/s to prevent infinite RAM usage
        elif sim_spec_vel_bin_size < 0.1:
            nyquist_limit = inst_fwhm_kms / 3.0
            
            if 0.1 > nyquist_limit:
                logger.warning(
                    f"\n [WARNING] Instrumental resolution (R={instrument_res}) is exceptionally high! "
                    f"Capping vel bin at 0.1 km/s violates the Nyquist limit (<3 points per FWHM). "
                    f"Your spectral lines will be severely undersampled."
                )
            else:
                logger.info(
                    f"\n [NOTE] Capping velocity bin at minimum 0.1 km/s. "
                    f"Will not achieve 15x oversampling for R={instrument_res}, but Nyquist is preserved."
                )
            sim_spec_vel_bin_size = 0.1
    else:
        # Fallback for photometry-only runs where R=0
        sim_spec_vel_bin_size = 1.0 
        
    logger.warning(f"  >> Dynamic Velocity Bin Size set to: {sim_spec_vel_bin_size:.3f} km/s")
    # ====================================================================

    # --- Stellar params ---
    stellar_radius  = cfg.getfloat('star', 'radius') * SOLAR_RADIUS
    rotation_period = cfg.getfloat('star', 'prot')
    incl            = cfg.getfloat('star', 'I')
    psi             = cfg.getfloat('star', 'psi')
    Temp            = cfg.getfloat('star', 'Tstar')
    logg            = cfg.getfloat('star', 'logg')
    FeH             = cfg.getfloat('star', 'FeH')

    # --- Fast parsers for lists ---
    def parse_region(section: str, key: str, dtype):
        raw = cfg.get(section, key).strip()
        cleaned = re.sub(r'[,\s]+', ' ', raw).strip()
        if not cleaned:
            return np.array([], dtype=dtype)
        
        # Use split() inside np.array instead of np.fromstring. 
        # This natively supports both numbers (np.float32) and strings (str).
        return np.array(cleaned.split(), dtype=dtype)

    # Temperature dict for LkCa 4 ZDI image
    cool_hot_dict = {
        3600: {'a': 322.0, 'b': 223.0, 'c': 116.0, 'd': 122.0, 'e': 259.0, 'f': 426.0},
        3700: {'a': 341.0, 'b': 235.0, 'c': 121.0, 'd': 133.0, 'e': 293.0, 'f': 512.0},
        3800: {'a': 360.0, 'b': 248.0, 'c': 128.0, 'd': 155.0, 'e': 354.0, 'f': 678.0},
        3900: {'a': 386.0, 'b': 269.0, 'c': 144.0, 'd': 178.0, 'e': 454.0, 'f': 855.0},
        4000: {'a': 417.0, 'b': 295.0, 'c': 161.0, 'd': 224.0, 'e': 584.0, 'f': 1022.0},
        4100: {'a': 456.0, 'b': 330.0, 'c': 183.0, 'd': 292.0, 'e': 701.0, 'f': 1143.0},
        # 4000: {'a': 433, 'b': 298, 'c': 156, 'd': 173, 'e': 370, 'f': 586},
        # 3900: {'a': 416, 'b': 285, 'c': 149, 'd': 159, 'e': 341, 'f': 545},
        # 3800: {'a': 396, 'b': 272, 'c': 139, 'd': 153, 'e': 318, 'f': 509},
        # 3700: {'a': 377, 'b': 263, 'c': 136, 'd': 142, 'e': 297, 'f': 469},
    }
    # Cool regions
    cool_longs = parse_region('cool_active_regions', 'longitudes', np.float32)
    cool_lats  = parse_region('cool_active_regions', 'latitudes',  np.float32)
    cool_sizes = parse_region('cool_active_regions', 'sizes',      np.float32)
    cool_dT    = parse_region('cool_active_regions', 'Tcool_spots', np.float64)

    if (cool_dT.size == 1) and (cool_sizes.size > 1):
        cool_dT = np.full_like(cool_sizes, cool_dT[0], dtype=np.float64)
    
    # cool_dT    = parse_region('cool_active_regions', 'Tcool_spots', str)
    # cool_dT    = np.array([cool_hot_dict.get(int(Temp), {}).get(t, 0) for t in cool_dT], dtype=np.float64)

    if cool_dT.size:
        idx = np.argsort(cool_dT)[::-1]  # largest |ΔT| first
        cool_longs, cool_lats, cool_sizes, cool_dT = cool_longs[idx], cool_lats[idx], cool_sizes[idx], cool_dT[idx]
    cool_T = Temp - cool_dT

    # Hot regions
    hot_longs = parse_region('hot_active_regions', 'longitudes', np.float32)
    hot_lats  = parse_region('hot_active_regions', 'latitudes',  np.float32)
    hot_sizes = parse_region('hot_active_regions', 'sizes',      np.float32)
    hot_dT    = parse_region('hot_active_regions', 'Thot_faculae', np.float64)
    
    if (hot_dT.size == 1) and (hot_sizes.size > 1):
        hot_dT = np.full_like(hot_sizes, hot_dT[0], dtype=np.float64)

    # hot_dT    = parse_region('hot_active_regions', 'Thot_faculae', str)
    # hot_dT    = np.array([cool_hot_dict.get(int(Temp), {}).get(t, 0) for t in hot_dT], dtype=np.float64)

    if hot_dT.size:
        idx = np.argsort(hot_dT)[::-1]
        hot_longs, hot_lats, hot_sizes, hot_dT = hot_longs[idx], hot_lats[idx], hot_sizes[idx], hot_dT[idx]
    hot_T = Temp + hot_dT

    # --- Active region counts ---
    # cool_n = cfg.getint('cool_active_regions', 'spot_number')
    # hot_n  = cfg.getint('hot_active_regions',  'spot_number')
    cool_n = cool_longs.size
    hot_n  = hot_longs.size
    active_regions_number = cool_n + hot_n
    logger.info(f"Found {cool_n} cool region(s), and {hot_n} hot region(s).")
    # -----------------------------

    # Convert % area -> radius fraction (sqrt of fractional area)
    cool_sizes = np.sqrt(cool_sizes / 100.0, dtype=np.float32)
    hot_sizes  = np.sqrt(hot_sizes  / 100.0, dtype=np.float32)

    # Concatenate all regions
    active_region_longs = np.concatenate([cool_longs, hot_longs]).astype(np.float32, copy=False)
    active_region_lats  = np.concatenate([cool_lats,  hot_lats ]).astype(np.float32, copy=False)
    active_region_sizes = np.concatenate([cool_sizes, hot_sizes]).astype(np.float64, copy=False)
    active_region_Temp  = np.concatenate([cool_T,     hot_T    ]).astype(np.float32, copy=False)
    active_region_box   = [active_region_longs, active_region_lats, active_region_sizes, active_region_Temp]

    # --- Star object ---
    STAR = spotact_core.Star(
        prot=rotation_period,
        vrot=(2.0 * np.pi * stellar_radius) / (rotation_period * 86400.0),
        incl=incl,
        psi=psi,
        rad_sun=SOLAR_RADIUS,
        rad=stellar_radius,
        Temp=Temp,
        Tspot=cool_T,
        Tfaculae=hot_T,
        logg=logg,
        FeH=FeH,
        bands2use=bands,
    )
    STAR.vsini = float(STAR.vrot * np.sin(np.deg2rad(STAR.incl)))
    if 'photo' in bands:
        STAR.filters = filters
    STAR.inst_res = instrument_res

    # --- Output directories ---
    out_base = Path("outputs") / SAVE_DIR
    for p in [
        Path("syn_spec"),
        Path("syn_spec/RASSINE_PHX"),
        Path("syn_spec/SimUse"),
        Path("syn_spec/Photometry_use_Spec"),
        out_base,
        out_base / "normal_spec",
        out_base / "spec",
        out_base / "ccf_template",
        Path("outputs/CCF"),
    ]:
        p.mkdir(parents=True, exist_ok=True)

    # --- RV CCF template parameters ---
    rv_ccf_pars = np.zeros(5, dtype=np.float32) + 99999.0

    # --- Phase setup ---
    phase_step = cfg.getfloat('output', 'ph_step')
    phase_in   = cfg.get('output', 'ph_in', fallback='').strip()

    if phase_in and phase_in.lower() != 'none':
        with open(phase_in, 'r') as f:
            PSI = np.fromiter((float(line.split()[0]) for line in f), dtype=np.float32)
    elif phase_step > 0:
        # PSI = np.arange(-0.75, 0.751, phase_step, dtype=np.float32)
        PSI = np.arange(-0.5, 0.5, phase_step, dtype=np.float32)
    else:
        logger.warning("Put a positive phase step ('ph_step') or provide a phase file ('ph_in').")
        PSI = np.array([], dtype=np.float32)

    # --- PHOENIX paths ---
    phx_root = Path(phx_path).expanduser() / 'PHOENIX-ACES-AGSS-COND-2011'
    STAR.phx_path = phx_root
    (phx_root / 'SpecInt').mkdir(parents=True, exist_ok=True)
    (phx_root / 'SpecFlux').mkdir(parents=True, exist_ok=True)

    # --- Grid check and flatten synthetic spectra ---
    spotact_utils.grid_resolution_check(grid_size)

    STAR.ccf_template_photosphereORnot = float(cfg.get('ccf', 'Use_photosphere_teff'))
    if STAR.ccf_template_photosphereORnot == 1.0:
        STAR.ccf_temp = STAR.Temp
        fit_temp = None
    else:
        STAR.ccf_temp = float(cfg.get('ccf', 'Teff'))
        fit_temp = STAR.ccf_temp

    STAR.find_fluxspec_flatten(sim_spec_vel_bin_size, plot_flag, syn_model_path, fit_temp)

    return (
        STAR, bands, SAVE_DIR, nrho, grid_size, instrument_res, cpu_use, plot_flag,
        save_spec_flag, sim_spec_window_size, syn_model_path, rv_ccf_pars, PSI,
        active_regions_number, active_region_box, sim_spec_line_centers, filters,
        sim_spec_vel_bin_size
        )

def setup_activate_spectra(
        bands, STAR, plot_flag, syn_model_path, sim_spec_vel_bin_size, savehdf=None):
    """
    Prepare synthetic spectra for all bands and temperatures for the spotted star simulation.

    Args:
        bands (dict): {band_name: [center, width, mode, 'intensity'|'flux'|'photo', 'phx'|'synmast']}
        STAR (Star): Star object with physical/spectral props already set up.
        plot_flag (int): 0/1 toggles plotting inside STAR methods.
        syn_model_path (dict): paths for synthetic model spectra.

    Returns:
        list[dict[int, spotact_core.Spec]]: same ordering as bands.keys()
    """

    # Which intensity modes do we need overall?
    # (Set avoids duplicates; also robust if someone adds more bands later.)
    intensity_modes = {params[3] for params in bands.values()}

    # Keep photosphere + all spot/facula temps
    temp_list = np.unique(
        np.concatenate(([STAR.Temp], STAR.Tspot, STAR.Tfaculae))
    )

    # Vectorize band centers/widths once, cast to float32 to save RAM
    line_centers = np.array([params[0] for params in bands.values()], dtype=np.float32)
    line_widths  = np.array([params[1] for params in bands.values()], dtype=np.float32)

    # Limb darkening only needs to be computed once for the temp×band grid
    # (Both flux and intensity paths rely on LD.)

    if len(line_centers) != 0: 
        STAR.get_quadratic_limbdarken(temp_list, line_centers, line_widths)

    # Build synthetic spectra “banks” only for the modes we actually need
    if 'flux' in intensity_modes:
        STAR.find_fluxspec(
            sim_spec_vel_bin_size, plot_flag, savehdf=savehdf)
    if 'intensity' in intensity_modes or 'photo' in intensity_modes:
        STAR.find_intensityspec(
            sim_spec_vel_bin_size, plot_flag, syn_model_path, savehdf=savehdf)

    # Build per-band Spec dictionaries (temp -> Spec)
    # Keep the same order as bands.keys(), so SPEC_box[i] matches enumerate(bands.keys()) in main_model
    SPEC_box = []
    for band_name in bands.keys():
        center, width, mode, int_or_flux, model_type = bands[band_name]
        spec_dict = {
            np.round(t, 4): spotact_core.Spec(
                star=STAR,
                band=band_name,
                temperature=t,
                intORflux=int_or_flux,
                model_type=model_type,
            )
            for t in temp_list
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
            '{0:18} = {1:<6}, {2:5} = {3:<4.2f}, {4:5} = {5:<4.2f}'.format('Syn template', model_type, 'logg', STAR.logg, '[Fe/H]', 0.0)\

    text2 = '{} active regions:\n'.format(active_regions_number) + \
            '  lon [deg]= {}\n'.format(np.array2string(
                active_region_longs, separator=',', formatter={'float_kind': lambda x: f'{int(x):4d}'})) + \
            '  lat [deg]= {}\n'.format(np.array2string(
                active_region_lats, separator=',', formatter={'float_kind': lambda x: f'{int(x):4d}'})) + \
            '   size [%]= {}\n'.format(np.array2string(
                active_region_sizes_percent, separator=',', formatter={'float_kind': lambda x: f'{float(x):4.1f}'})) + \
            '   Temp [K]= {}\n'.format(np.array2string(
                active_region_dTemp, separator=',', formatter={'float_kind': lambda x: fr'${int(x):+5.1f}$'}))
    
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
    instrument_res, cpu_use, SAVE_DIR, save_spec_flag, plot_flag, sim_spec_vel_bin_size,
    silence_pbar, in_memory: bool = False, write_csv: bool = True,
):
    """
    Run the per-band spotted-star simulations and write outputs.

    Args:
        logger (logging.Logger): Logger for status and timing.
        bands (dict): Mapping band_name -> [center, width, mode, intORflux, model_type].
        nrho (int): Circumference resolution.
        grid_size (int): Surface grid resolution (N x N).
        STAR (spotact_core.Star): Initialized star object with spectra prepared.
        PSI (array[float32]): Phase array.
        SPEC_box (list[dict[int, spotact_core.Spec]]): Per-band dict: temperature -> Spec object.
        filters (list[str]): Photometry filters from SVO (if 'photo' band used).
        active_regions_number (int): Number of active regions.
        active_region_box (tuple[np.ndarray]): (longitudes, latitudes, sizes_rfrac, temperatures).
        rv_ccf_pars (np.ndarray): RV CCF template parameters.
        syn_model_path (dict): Model paths {'phx': ..., 'synmast': ...}.
        instrument_res (int): Instrumental resolution.
        cpu_use (int): # of logical CPUs to use downstream.
        SAVE_DIR (str): Output directory name under ./outputs/.
        save_spec_flag (int): If 1, save flattened spectra arrays.
        plot_flag (int): If 1, produce plots via visualization helpers.
        silence_pbar (bool): Silence progress bars in inner routines.

    Returns:
        if in_memory == False: str (final output folder path, as before)
        if in_memory == True : (str outdir, dict results_by_band)
            where results_by_band[band] is a dict with keys:
              - 'phase' : (N,) float32
              - 'rv'    : (N,) float32 or NaN (for photo-only band)
              - 'ewr'   : (N,) float32 or NaN (for non-EWR bands)
              - 'photo': dict only for 'photo' band with:
                    'mag': {<filter short name>: (N,) float32}
                    'flux': {<filter short name>: (N,) float32}
              - 'flattened_spec': (N, M) float32 if you want to keep them (optional; here we don’t store to save RAM)
    """

    # ---- Normalize active-region order ONCE (coolest cools first, hottest hots first) ----
    (active_region_longs, active_region_lats,
     active_region_sizes, active_region_Temp) = active_region_box

    n_regions = len(active_region_longs)
    active_region_org_idx = np.arange(n_regions)

    cool_mask = (active_region_Temp - STAR.Temp) < 0
    hot_mask  = (active_region_Temp - STAR.Temp) > 0

    cool_sort = np.argsort(STAR.Tspot)            # cool temps low->high, 2300, 2400, 2500, ...
    hot_sort  = np.argsort(STAR.Tfaculae)[::-1]   # hot temps high->low, 4100, 4000, 3900, ...

    cool_idx = np.flatnonzero(cool_mask)[cool_sort]
    hot_idx  = np.flatnonzero(hot_mask)[hot_sort]
    seq_idx  = np.concatenate([cool_idx, hot_idx]) if (cool_idx.size + hot_idx.size) else np.arange(n_regions)
    # e.g., 2300, 2400, 2500, ..., 4100, 4000, 3900, ... for a star with both cool and hot regions.

    # Reorder + compact dtypes
    active_region_longs = np.asarray(active_region_longs, dtype=np.float32)[seq_idx]
    active_region_lats  = np.asarray(active_region_lats,  dtype=np.float32)[seq_idx]
    active_region_sizes = np.asarray(active_region_sizes, dtype=np.float32)[seq_idx]
    active_region_Temp  = np.asarray(active_region_Temp,  dtype=np.float64)[seq_idx]
    active_region_org_idx = active_region_org_idx[seq_idx]

    active_region_box = (
        active_region_longs, active_region_lats, active_region_sizes, active_region_Temp
    )

    # ---- Build outdir base (independent of band) ----
    sizes_percent = (active_region_sizes ** 2) * 100.0
    def _fmt_arr(a):
        return np.array2string(a, separator='-', formatter={'float_kind': lambda x: f'{int(x):d}'})\
                 .strip('[]').replace(' ', '')

    temp_str      = _fmt_arr(active_region_Temp)
    temp_str_uniq = _fmt_arr(np.unique(active_region_Temp))
    long_str      = _fmt_arr(active_region_longs)
    lat_str       = _fmt_arr(active_region_lats)
    size_str      = _fmt_arr(sizes_percent)

    if active_regions_number < 4:
        outdir_base = "./outputs/{}/{}_P={:.1f}_i={:.1f}_Tp={:1.0f}_logg={:1.1f}_Ta={}_lon={}_lat={}_size={}".format(
            SAVE_DIR, 'MODEL', STAR.prot, STAR.incl, STAR.Temp, STAR.logg,
            temp_str, long_str, lat_str, size_str)
    else:
        outdir_base = "./outputs/{}/{}_P={:.1f}_i={:.1f}_Tp={:1.0f}_logg={:1.1f}_SpotNum={:1.0f}_Ta={}".format(
            SAVE_DIR, 'MODEL', STAR.prot, STAR.incl, STAR.Temp, STAR.logg,
            active_regions_number, temp_str_uniq)

    # ---- Main loop over bands ----
    results_by_band = {}  # only filled if in_memory=True

    band_names = list(bands.keys())  # freeze order to align with SPEC_box
    for i, band in enumerate(band_names):
        start_time = time()

        center, width, mode, intORflux, model_type = bands[band]
        logger.info(f'\n\nNow calculating the activity signal for the band: {band}')
        SPEC = SPEC_box[i]

        outdir = outdir_base.replace('/MODEL_', f'/{model_type}_')

        # Core simulation
        visible_phi, integrated_stellar_spectrum, _, final_2d_surface_flux_sum = spotact_core.calculate_activity_signal_spec(
            nrho, grid_size, STAR, PSI, SPEC, active_regions_number, active_region_box,
            instrument_res, cpu_use, SAVE_DIR, band, save_spec_flag, intORflux.lower(), silence_pbar, plot_flag
        )
        visible_phi = np.asarray(visible_phi, dtype=np.float32)

        if band == 'photo':
            # Photometry only
            lightcurves_mag, lightcurves_flux = spotact_core.get_photometric(
                integrated_stellar_spectrum, STAR, visible_phi, band, intORflux, model_type, 
            )
            ccf_rv_shift = np.full_like(visible_phi, np.nan, dtype=np.float32)
            ohfe_ratio   = np.full_like(visible_phi, np.nan, dtype=np.float32)

            if in_memory:
                # pack photometry into dicts keyed by short filter name
                mag_dict  = {}
                flux_dict = {}
                for j, filt in enumerate(filters):
                    fname = filt.split('/')[1]
                    mag_dict[fname]  = np.asarray(lightcurves_mag[j],  dtype=np.float32)
                    flux_dict[fname] = np.asarray(lightcurves_flux[j], dtype=np.float64)
                results_by_band[band] = {
                    'phase': visible_phi,
                    # 'rv':    ccf_rv_shift,
                    # 'ewr':   ohfe_ratio,
                    # 'mag': mag_dict, 
                    'flux': flux_dict
                }
            else:
                # write CSV (as before)
                from astropy.table import Table
                tab = Table([visible_phi], names=['Phase'])
                for j, filt in enumerate(filters):
                    fname = filt.split('/')[1]
                    tab[f'{fname}_mag']  = np.asarray(lightcurves_mag[j],  dtype=np.float32)
                    tab[f'{fname}_flux'] = np.asarray(lightcurves_flux[j], dtype=np.float64)
                    tab[f'{fname}_mag'].format  = '%.6f'
                    tab[f'{fname}_flux'].format = '%.6f'
                tab['Phase'].format = '%.3f'
                if write_csv:
                    tab.write(f'./outputs/{SAVE_DIR}/{outdir.split("/")[-1]}_{band}_RV_EWR.csv',
                              format='csv', overwrite=True)

            if plot_flag == 1:
                simulation_plot(
                    grid_size, nrho, instrument_res, STAR, model_type, SAVE_DIR,
                    active_regions_number, active_region_box, visible_phi, integrated_stellar_spectrum,
                    ccf_rv_shift, final_2d_surface_flux_sum, outdir, cpu_use, ohfe_ratio,
                    band, intORflux, filters, lightcurves_mag, lightcurves_flux
                )

            # Free big arrays
            del integrated_stellar_spectrum, final_2d_surface_flux_sum, lightcurves_mag, lightcurves_flux
            logger.info(f'\n--> {band} run time: {(time()-start_time)/60:1.2f} min')
            continue

        # Spectroscopy branch
        integrated_stellar_spectrum = spotact_core.flatten_model_spec(
            STAR, integrated_stellar_spectrum, visible_phi, band,
            intORflux, model_type, SAVE_DIR, plot_flag
        )
        if save_spec_flag == 1 and not in_memory and write_csv:
            np.save(f'./outputs/{SAVE_DIR}/spec/{band}_flat_final_spec.npy',
                    integrated_stellar_spectrum)

        if band == 'ewr_FeOH':
            ohfe_ratio = spotact_utils.get_equivalent_width(
                STAR, integrated_stellar_spectrum, visible_phi, band, intORflux, model_type
            ).astype(np.float32, copy=False)
        else:
            ohfe_ratio = np.full_like(visible_phi, np.nan, dtype=np.float32)

        logger.info('\nCalculating the RV with CCF...\n')
        ccf_rv_shift = spotact_core.cal_ccf_rv(
            integrated_stellar_spectrum, STAR, visible_phi, band, intORflux, model_type,
            rv_ccf_pars, syn_model_path['synmast'], cpu_use, SAVE_DIR, sim_spec_vel_bin_size,
            plot_flag, silence_pbar, save_spec_flag, outdir
        )
        ccf_rv_shift = np.asarray(ccf_rv_shift, dtype=np.float32)

        if in_memory:
            results_by_band[band] = {
                'phase': visible_phi,
                'rv':    ccf_rv_shift,
                'ewr':   ohfe_ratio,
            }
        else:
            # write CSV (as before)
            from astropy.table import Table
            tab = Table(data=(visible_phi, ccf_rv_shift, ohfe_ratio, STAR._phase_teff, STAR._phase_fspot),
                        names=('Phase', 'RV', 'EWR', 'model_Teff', 'fspot'))
            tab['Phase'].format = '%.3f'
            tab['RV'].format    = '%.4f'
            tab['EWR'].format   = '%.5f'
            tab['model_Teff'].format = '%d'
            tab['fspot'].format      = '%.4f'

            if write_csv:
                tab.write(f'./outputs/{SAVE_DIR}/{outdir.split("/")[-1]}_{band}_RV_EWR.csv',
                          format='csv', overwrite=True)

            if plot_flag == 1:
                simulation_plot(
                    grid_size, nrho, instrument_res, STAR, model_type, SAVE_DIR,
                    active_regions_number, active_region_box, visible_phi, integrated_stellar_spectrum,
                    ccf_rv_shift, final_2d_surface_flux_sum, outdir, cpu_use, ohfe_ratio,
                    band, intORflux, filters, lightcurves_mag=[None, None], lightcurves_flux=[None, None]
                )

        # Free big arrays
        del integrated_stellar_spectrum, final_2d_surface_flux_sum, ccf_rv_shift, ohfe_ratio
        logger.info(f'\n--> {band} run time: {(time()-start_time)/60:1.2f} min')

    # Restore original order for downstream
    restore = np.argsort(active_region_org_idx)
    active_region_longs = active_region_longs[restore]
    active_region_lats  = active_region_lats[restore]
    active_region_sizes = active_region_sizes[restore]
    active_region_Temp  = active_region_Temp[restore]
    active_region_box = (active_region_longs, active_region_lats, active_region_sizes, active_region_Temp)

    if in_memory:
        return outdir, results_by_band
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
    
    # silence progress bars setting, default False
    parser.add_argument(
        '-silence', action="store", type=bool, default=False)
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
    logger.info("\nStarting SPOTACT-TTS simulation...")
    logger.info("Initital setup...")
    # read in cfg, and setup classes
    (STAR, bands, SAVE_DIR, nrho, grid_size, instrument_res, cpu_use,
        plot_flag, save_spec_flag, sim_spec_window_size, syn_model_path,
        rv_ccf_pars, PSI, active_regions_number, active_region_box,
        sim_spec_line_centers, filters, sim_spec_vel_bin_size) = init_setup_cfg(
            logger, CONFIG_FILE, post_fix=args.out)

    (active_region_longs, active_region_lats, active_region_sizes, 
        active_region_Temp) = active_region_box
    
    # --- Setup the synthetic spectra for Spotted spectra simulation ---
    SPEC_box = setup_activate_spectra(
        bands, STAR, plot_flag, syn_model_path, sim_spec_vel_bin_size, 
        savehdf = save_spec_flag)

    ################################################################################################
    
    # run the simulation
    main_model(
        logger, bands, nrho, grid_size, STAR, PSI, SPEC_box, filters,
        active_regions_number, active_region_box, 
        rv_ccf_pars, syn_model_path, instrument_res, 
        cpu_use, SAVE_DIR, save_spec_flag, plot_flag, sim_spec_vel_bin_size, 
        silence_pbar=args.silence)
    
    
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


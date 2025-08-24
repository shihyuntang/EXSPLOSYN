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
    bands['ewr_FeOH'] = [15557.0, 90.0, 'rv', 'intensity', 'synmast']

    if has_filters:
        logger.info(f'\nDetect filters. Will generate photometry for {filters}.')
        try:
            from pyphot import svo  # import here so unit tests without pyphot still pass init for spec-only runs
            profiles = [svo.get_pyphot_filter(filt) for filt in filters]
            # nm -> Å
            fwv = np.concatenate([p.wavelength for p in profiles]).astype(np.float32) * 10.0
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
    path_sub = 1 if hostname == 'Swing' else 2
    RASSINE_PATH = cfg.get('path', f'rassine_path{path_sub}')
    SYNmast_PATH = cfg.get('path', f'synmast_prf{path_sub}')
    phx_path     = cfg.get('path', 'PHOENIX_path')
    syn_model_path = {'phx': RASSINE_PATH, 'synmast': SYNmast_PATH}

    # --- Output & model params ---
    plot_flag      = cfg.getint('output', 'plot')
    save_spec_flag = cfg.getint('output', 'save_spec')
    grid_size      = cfg.getint('main', 'grid')
    nrho           = cfg.getint('main', 'nrho')
    instrument_res = cfg.getint('main', 'instrument_reso')

    # Keep it simple: Swing=6, else use system cpu_count (reasonable cap)
    default_cpus = 6 if hostname == 'Swing' else (os.cpu_count() or 8)
    cpu_use = min(max(default_cpus, 1), 64)

    # --- Stellar params ---
    stellar_radius  = cfg.getfloat('star', 'radius') * SOLAR_RADIUS
    rotation_period = cfg.getfloat('star', 'prot')
    incl            = cfg.getfloat('star', 'I')
    psi             = cfg.getfloat('star', 'psi')
    Temp            = cfg.getint('star', 'Tstar')
    logg            = cfg.getfloat('star', 'logg')
    FeH             = cfg.getfloat('star', 'FeH')

    # --- Active region counts ---
    cool_n = cfg.getint('cool_active_regions', 'spot_number')
    hot_n  = cfg.getint('hot_active_regions',  'spot_number')
    active_regions_number = cool_n + hot_n
    logger.info(f"Found {cool_n} cool region(s), and {hot_n} hot region(s).")

    # --- Fast parsers for lists ---
    def parse_region(section: str, key: str, dtype):
        raw = cfg.get(section, key).strip()
        cleaned = re.sub(r'[,\s]+', ' ', raw).strip()
        if not cleaned:
            return np.array([], dtype=dtype)
        return np.fromstring(cleaned, sep=' ', dtype=dtype)

    # Cool regions
    cool_longs = parse_region('cool_active_regions', 'longitudes', np.float32)
    cool_lats  = parse_region('cool_active_regions', 'latitudes',  np.float32)
    cool_sizes = parse_region('cool_active_regions', 'sizes',      np.float32)
    cool_dT    = parse_region('cool_active_regions', 'Tcool_spots', np.int16)

    if cool_dT.size:
        idx = np.argsort(cool_dT)[::-1]  # largest |ΔT| first
        cool_longs, cool_lats, cool_sizes, cool_dT = cool_longs[idx], cool_lats[idx], cool_sizes[idx], cool_dT[idx]
    cool_T = np.int32(Temp) - cool_dT.astype(np.int32)

    # Hot regions
    hot_longs = parse_region('hot_active_regions', 'longitudes', np.float32)
    hot_lats  = parse_region('hot_active_regions', 'latitudes',  np.float32)
    hot_sizes = parse_region('hot_active_regions', 'sizes',      np.float32)
    hot_dT    = parse_region('hot_active_regions', 'Thot_faculae', np.int16)

    if hot_dT.size:
        idx = np.argsort(hot_dT)[::-1]
        hot_longs, hot_lats, hot_sizes, hot_dT = hot_longs[idx], hot_lats[idx], hot_sizes[idx], hot_dT[idx]
    hot_T = np.int32(Temp) + hot_dT.astype(np.int32)

    # Convert % area -> radius fraction (sqrt of fractional area)
    cool_sizes = np.sqrt(cool_sizes / 100.0, dtype=np.float32)
    hot_sizes  = np.sqrt(hot_sizes  / 100.0, dtype=np.float32)

    # Concatenate all regions
    active_region_longs = np.concatenate([cool_longs, hot_longs]).astype(np.float32, copy=False)
    active_region_lats  = np.concatenate([cool_lats,  hot_lats ]).astype(np.float32, copy=False)
    active_region_sizes = np.concatenate([cool_sizes, hot_sizes]).astype(np.float32, copy=False)
    active_region_Temp  = np.concatenate([cool_T,     hot_T   ]).astype(np.int32,   copy=False)
    active_region_box   = [active_region_longs, active_region_lats, active_region_sizes, active_region_Temp]

    # --- Star object ---
    STAR = spotact_core.Star(
        prot=rotation_period,
        vrot=(2.0 * np.pi * stellar_radius) / (rotation_period * 86400.0),
        incl=incl,
        psi=psi,
        rad_sun=SOLAR_RADIUS,
        rad=stellar_radius,
        Temp=int(Temp),
        Tspot=cool_T,        # already int32
        Tfaculae=hot_T,      # already int32
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
        PSI = np.arange(-0.75, 0.751, phase_step, dtype=np.float32)
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

    # Unique temperature list (int32) to avoid duplicate Spec objects
    # Keep photosphere + all spot/facula temps
    temp_list = np.unique(
        np.concatenate(([np.int32(STAR.Temp)], STAR.Tspot.astype(np.int32), STAR.Tfaculae.astype(np.int32)))
    )

    # Vectorize band centers/widths once, cast to float32 to save RAM
    line_centers = np.array([params[0] for params in bands.values()], dtype=np.float32)
    line_widths  = np.array([params[1] for params in bands.values()], dtype=np.float32)

    # Limb darkening only needs to be computed once for the temp×band grid
    # (Both flux and intensity paths rely on LD.)
    STAR.get_quadratic_limbdarken(temp_list, line_centers, line_widths)

    # Build synthetic spectra “banks” only for the modes we actually need
    if 'flux' in intensity_modes:
        STAR.find_fluxspec(SIMSPEC_VEL_BIN_SIZE, plot_flag, syn_model_path)
    if 'intensity' in intensity_modes or 'photo' in intensity_modes:
        STAR.find_intensityspec(SIMSPEC_VEL_BIN_SIZE, plot_flag, syn_model_path)

    # Build per-band Spec dictionaries (temp -> Spec)
    # Keep the same order as bands.keys(), so SPEC_box[i] matches enumerate(bands.keys()) in main_model
    SPEC_box = []
    for band_name in bands.keys():
        center, width, mode, int_or_flux, model_type = bands[band_name]
        spec_dict = {
            int(t): spotact_core.Spec(
                star=STAR,
                band=band_name,
                temperature=int(t),
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
        str: The final output folder path string (same base for each band; band suffix varies).
    """

    # ---- Normalize active-region order ONCE (coolest cools first, hottest hots first) ----
    (active_region_longs, active_region_lats,
     active_region_sizes, active_region_Temp) = active_region_box

    n_regions = len(active_region_longs)
    active_region_org_idx = np.arange(n_regions)

    cool_mask = (active_region_Temp - STAR.Temp) < 0
    hot_mask  = (active_region_Temp - STAR.Temp) > 0

    cool_sort = np.argsort(STAR.Tspot)            # cool temps low->high
    hot_sort  = np.argsort(STAR.Tfaculae)[::-1]   # hot temps high->low

    cool_idx = np.flatnonzero(cool_mask)[cool_sort]
    hot_idx  = np.flatnonzero(hot_mask)[hot_sort]
    seq_idx  = np.concatenate([cool_idx, hot_idx]) if (cool_idx.size + hot_idx.size) else np.arange(n_regions)

    # Reorder + compact dtypes
    active_region_longs = np.asarray(active_region_longs, dtype=np.float32)[seq_idx]
    active_region_lats  = np.asarray(active_region_lats,  dtype=np.float32)[seq_idx]
    active_region_sizes = np.asarray(active_region_sizes, dtype=np.float32)[seq_idx]
    active_region_Temp  = np.asarray(active_region_Temp,  dtype=np.int32)[seq_idx]
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
            instrument_res, cpu_use, SAVE_DIR, band, save_spec_flag, intORflux.lower(), silence_pbar
        )
        visible_phi = np.asarray(visible_phi, dtype=np.float32)

        if band == 'photo':
            # Photometry only
            lightcurves_mag, lightcurves_flux = spotact_core.get_photometric(
                integrated_stellar_spectrum, STAR, visible_phi, band, intORflux, model_type
            )
            ccf_rv_shift = np.full_like(visible_phi, np.nan, dtype=np.float32)
            ohfe_ratio   = np.full_like(visible_phi, np.nan, dtype=np.float32)

            # Save (Astropy Table)
            tab = Table([visible_phi], names=['Phase'])
            for j, filt in enumerate(filters):
                fname = filt.split('/')[1]
                tab[f'{fname}_mag']  = np.asarray(lightcurves_mag[j],  dtype=np.float32)
                tab[f'{fname}_flux'] = np.asarray(lightcurves_flux[j], dtype=np.float32)
                tab[f'{fname}_mag'].format  = '%.6f'
                tab[f'{fname}_flux'].format = '%.6f'
            tab['Phase'].format = '%.3f'
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
            logger.warning(f'\n--> {band} run time: {(time()-start_time)/60:1.2f} min')
            continue

        # Spectroscopy branch
        integrated_stellar_spectrum = spotact_core.flatten_model_spec(
            STAR, integrated_stellar_spectrum, visible_phi, band,
            intORflux, model_type, SAVE_DIR, plot_flag
        )
        if save_spec_flag == 1:
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
            rv_ccf_pars, syn_model_path['synmast'], cpu_use, SAVE_DIR, SIMSPEC_VEL_BIN_SIZE,
            plot_flag, silence_pbar, save_spec_flag
        )
        ccf_rv_shift = np.asarray(ccf_rv_shift, dtype=np.float32)

        # Save (Astropy Table)
        tab = Table(data=(visible_phi, ccf_rv_shift, ohfe_ratio),
                    names=('Phase', 'RV', 'EWR'))
        tab['Phase'].format = '%.3f'
        tab['RV'].format    = '%.4f'
        tab['EWR'].format   = '%.5f'
        tab.write(f'./outputs/{SAVE_DIR}/{outdir.split("/")[-1]}_{band}_RV_EWR.csv',
                  format='csv', overwrite=True)

        # Plot
        if plot_flag == 1:
            simulation_plot(
                grid_size, nrho, instrument_res, STAR, model_type, SAVE_DIR,
                active_regions_number, active_region_box, visible_phi, integrated_stellar_spectrum,
                ccf_rv_shift, final_2d_surface_flux_sum, outdir, cpu_use, ohfe_ratio,
                band, intORflux, filters, lightcurves_mag=[None, None], lightcurves_flux=[None, None]
            )

        # Free big arrays
        del integrated_stellar_spectrum, final_2d_surface_flux_sum, ccf_rv_shift, ohfe_ratio
        logger.warning(f'\n--> {band} run time: {(time()-start_time)/60:1.2f} min')

    # Restore original order for downstream
    restore = np.argsort(active_region_org_idx)
    active_region_longs = active_region_longs[restore]
    active_region_lats  = active_region_lats[restore]
    active_region_sizes = active_region_sizes[restore]
    active_region_Temp  = active_region_Temp[restore]
    active_region_box = (active_region_longs, active_region_lats, active_region_sizes, active_region_Temp)

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


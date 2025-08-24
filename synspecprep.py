"""
File modified from https://github.com/YinanZhao21/SOAP_GPU/blob/main/SOAP_code/phoenix_extraction.py
Download PHX synthetic spectra and flatten them using Rassine

Shih-Yun Tang 20241017
"""
import warnings, logging, subprocess, re
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import configparser
from astropy.io import fits
import astropy.units as u
from scipy import interpolate
# from specutils import Spectrum1D
import h5py

from rds3v import rds3v

config = configparser.ConfigParser()

def vac2air(wavelength):
    '''
    # Transform wavelength in vacuum to air
    # See VALD website here
    # http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    # The formula comes from Birch and Downs (1994, Metrologia, 31, 315)
    '''
    s = 1.e4 / wavelength # s = 10^4 / lamda_vac, lamda_vac is in Angstrom
    n_air = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    return wavelength / n_air


def air2vac(wavelength):
    '''
    # Transform wavelength in air to vacuum
    # See VALD website here
    # http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    # The formula comes from N. Piskunov
    '''
    s = 1.e4 / wavelength # s = 10^4 / lamda_vac, lamda_vac is in Angstrom
    n_air = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 \
        - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    return wavelength * n_air

def PHOENIX_name(temp, logg_val, FeH_val, alpha_val, fluxORint='flux'):
    if FeH_val <= 0.0:
        prefix_subdir = 'Z-{:.1f}'.format(np.abs(FeH_val))
        if alpha_val == 0.0:
            name = 'lte0{}-{:.2f}-{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(
                int(temp), logg_val, np.abs(FeH_val))
        elif alpha_val > 0.0:
            name = 'lte0{}-{:.2f}-{:.1f}.Alpha=+{:.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(
                int(temp), logg_val, np.abs(FeH_val), alpha_val)
        else:
            name = 'lte0{}-{:.2f}-{:.1f}.Alpha={:.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(
                int(temp), logg_val, np.abs(FeH_val), alpha_val)
    else:
        prefix_subdir = 'Z+{:.1f}'.format(np.abs(FeH_val))
        name = 'lte0{}-{:.2f}+{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(
            int(temp), logg_val, np.abs(FeH_val))
        
    if fluxORint == 'int':
        name = name.replace('HiRes', 'SpecInt')
    return prefix_subdir, name


LOGG_ARRAY = np.linspace(0.0,6.0,13)
TEMP_ARRAY = np.zeros(73)
FeH_ARRAY = np.concatenate( (np.linspace(-4.0, -3.0, 2), np.linspace(-2.0, 1.0, 7)) )
FeH_ARRAY = FeH_ARRAY.round(decimals=2)
ALPHA_ARRAY = -0.2+np.arange(8)*0.2
ALPHA_ARRAY = ALPHA_ARRAY.round(decimals=2)

for j in np.arange(73):
    if j < 48:
        TEMP_ARRAY[j] = 2300.0 + 100.0*(j)
    else:
        TEMP_ARRAY[j] = 7000.0 + 200.0*(j-49)

# remove 5000 from TEMP_ARRAY
idx_remove = np.where(TEMP_ARRAY == 5000.0)[0]
if idx_remove.size > 0:
    TEMP_ARRAY = np.delete(TEMP_ARRAY, idx_remove[0])

# --- Flux spectra (all mu integrated) ---
PHX_FTP_prefix = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'
PHX_FTP_wave_prefix = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/'
WAVE_FILE = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

# --- Intensity spectra (per mu angle) ---
PHX_FTP_prefix_Int = 'ftp://phoenix.astro.physik.uni-goettingen.de/v3.0/SpecIntFITS/'

def get_phx_flux_spec(temp, logg, FeH, alpha, phx_path, out_path):

    wvmin, wvmax = (3000, 30000)

    index_logg = np.argmin(np.abs(LOGG_ARRAY - logg))
    index_FeH = np.argmin(np.abs(FeH_ARRAY - FeH))
    index_alpha = np.argmin(np.abs(ALPHA_ARRAY - alpha))

    logg_val = LOGG_ARRAY[index_logg]
    FeH_val = FeH_ARRAY[index_FeH]

    if FeH_val > 0.0:
        alpha_val = 0.0
    else:
        alpha_val = ALPHA_ARRAY[index_alpha]

    index_temp = np.argmin(np.abs(TEMP_ARRAY - temp))
    if temp > TEMP_ARRAY[index_temp]:
        temp_low = TEMP_ARRAY[index_temp]
        temp_high = TEMP_ARRAY[index_temp+1]
    else:
        temp_low = TEMP_ARRAY[index_temp-1]
        temp_high = TEMP_ARRAY[index_temp]

    sub_dir_low, name_low = PHOENIX_name(
        temp_low, logg_val, FeH_val, alpha_val)
    sub_dir_high, name_high = PHOENIX_name(
        temp_high, logg_val, FeH_val, alpha_val)
    
    _, name_use = PHOENIX_name(temp, logg, FeH, alpha)
    name_use = name_use.replace('.fits', '').replace('/', '')
    
    
    wave_file = Path(f'{phx_path}/SpecFlux/{WAVE_FILE}')
    name_low_file = Path(f'{phx_path}/SpecFlux/{sub_dir_high}/{name_low}')
    name_high_file = Path(f'{phx_path}/SpecFlux/{sub_dir_high}/{name_high}')
    
    if not wave_file.is_file():
        logger.info('Download wavelength data...')
        os.system(f'curl {PHX_FTP_wave_prefix}/{WAVE_FILE} -o {phx_path}/SpecFlux/{WAVE_FILE}')
    else:
        logger.debug('Wavelength data exists...')
    
    os.makedirs(f'{phx_path}/SpecFlux/{sub_dir_low}', exist_ok=True)
    if not name_low_file.is_file():
        logger.info('Download spectral data...')
        os.system(f'curl {PHX_FTP_prefix}/{sub_dir_low}/{name_low} -o {phx_path}/SpecFlux/{sub_dir_low}/{name_low}')
    else:
        logger.debug(f'{name_low} exists...')
    if not name_high_file.is_file():
        logger.info('Download spectral data...')
        os.system(f'curl {PHX_FTP_prefix}/{sub_dir_high}/{name_high} -o {phx_path}/SpecFlux/{sub_dir_high}/{name_high}')
    else:
        logger.debug(f'{name_high} exists...')

    wave_data_vac = fits.getdata(f'{phx_path}/SpecFlux/{WAVE_FILE}')
    waveIndex = (wave_data_vac >= (wvmin-500)) & (wave_data_vac <= (wvmax+500))
    
    wavelength_vac = wave_data_vac[waveIndex]
    spec_low_vac = fits.getdata(f'{phx_path}/SpecFlux/{sub_dir_high}/{name_low}')[waveIndex]
    spec_high_vac = fits.getdata(f'{phx_path}/SpecFlux/{sub_dir_high}/{name_high}')[waveIndex]

    # wavelength_air = vac2air(wavelength_vac)
    # waveIndex_new = (wavelength_air >= wvmin) & (wavelength_air <= wvmax)
    # wavelength_air_new = wavelength_air[waveIndex_new]
    # spec_low_air = spec_low[waveIndex_new]
    # spec_high_air = spec_high[waveIndex_new]
    
    # flux_temp = np.zeros(wavelength_vac.size)
    # for i in np.arange(wavelength_vac.size):
    #     flux_temp[i] = spec_low_vac[i] + ( (temp - temp_low)/(temp_high - temp_low) )*(spec_high_vac[i] - spec_low_vac[i])
    
    flux_temp = spec_low_vac + ( (temp - temp_low)/(temp_high - temp_low) )\
                    *(spec_high_vac - spec_low_vac)
    
    spec_df = pd.DataFrame({'wave':wavelength_vac,'flux':flux_temp})
    
    data_name = f'wv{wvmin}_{wvmax}_{name_use}.h5'
    spec_df.to_hdf(
        f'{out_path}/RASSINE_PHX/{data_name}', key='spec', complevel=5, index=False)

    return data_name


def prep_phx_intensity_spec(temp, logg, FeH, alpha, out_path):
    wvmin, wvmax = (500, 50000)
    # Currently, phx int spectra is only use for photometry.
    # The intensity spectra from PHX has too low resolution in wavelength.
    # Use the low-res (50AA res) int spectra that already dw by ldtk pkg
    phx_low_res = Path(f'~/.ldtk/cache_visir-lowres').expanduser()

    index_logg = np.argmin(np.abs(LOGG_ARRAY - logg))
    index_FeH = np.argmin(np.abs(FeH_ARRAY - FeH))
    index_alpha = np.argmin(np.abs(ALPHA_ARRAY - alpha))
    logg_val = LOGG_ARRAY[index_logg]
    FeH_val = FeH_ARRAY[index_FeH]

    if FeH_val > 0.0:
        alpha_val = 0.0
    else:
        alpha_val = ALPHA_ARRAY[index_alpha]

    index_temp = np.argmin(np.abs(TEMP_ARRAY - temp))
    if temp > TEMP_ARRAY[index_temp]:
        temp_low = TEMP_ARRAY[index_temp]
        temp_high = TEMP_ARRAY[index_temp+1]
    else:
        temp_low = TEMP_ARRAY[index_temp-1]
        temp_high = TEMP_ARRAY[index_temp]

    sub_dir_low, name_low = PHOENIX_name(
        temp_low, logg_val, FeH_val, alpha_val, fluxORint='int')
    sub_dir_high, name_high = PHOENIX_name(
        temp_high, logg_val, FeH_val, alpha_val, fluxORint='int')
    
    _, name_use = PHOENIX_name(temp, logg, FeH, alpha, fluxORint='int')
    
    dir_low_file = phx_low_res / sub_dir_low / name_low
    dir_high_file = phx_low_res / sub_dir_high / name_high

    with fits.open(dir_low_file) as hdul:
        wl0 = hdul[0].header['crval1'] # Wavelength at d[:,0] [AA]
        dwl = hdul[0].header['cdelt1'] # Delta wavelength     [AA]
        nwl = hdul[0].header['naxis1'] # Number of wl samples
        wave_data_vac = wl0 + np.arange(nwl) * dwl
        mu = hdul[1].data
        spec_low_vac = hdul[0].data

    with fits.open(dir_high_file) as hdul:
        # wl0 = hdul[0].header['crval1'] # Wavelength at d[:,0] [AA]
        # dwl = hdul[0].header['cdelt1'] # Delta wavelength     [AA]
        # nwl = hdul[0].header['naxis1'] # Number of wl samples
        # wave_data_vac = wl0 + np.arange(nwl) * dwl
        # mu = hdul[1].data
        spec_high_vac = hdul[0].data
    
    # there are 78 mu angles from PHX.
    # We only use [0.067653, 0.13025, 0.2545, 0.37875, 0.503, 0.62725, 0.7515, 0.87575, 1]
    mu = mu[-17::2]
    spec_low_vac = spec_low_vac[-17::2, :]
    spec_high_vac = spec_high_vac[-17::2, :]

    waveIndex = (wave_data_vac >= (wvmin-500)) & (wave_data_vac <= (wvmax+500))
    
    wavelength_vac = wave_data_vac[waveIndex]
    spec_low_vac = spec_low_vac[:, waveIndex]
    spec_high_vac = spec_high_vac[:, waveIndex]
    flux_temp = spec_low_vac + ( (temp - temp_low)/(temp_high - temp_low) )*(spec_high_vac - spec_low_vac)

    # spec_df = pd.DataFrame({'wave':wavelength_vac, 'flux':flux_temp, 'mu':mu})    
    # spec_df.to_hdf(out_name, key='spec', complevel=5, index=False)

    out_name = f"{out_path}/Photometry_use_Spec/{name_use.replace('.fits', '.h5')}"
    with h5py.File(out_name, 'w') as f:
        f.create_dataset('wave', data=wavelength_vac, compression='gzip', compression_opts=5)
        f.create_dataset('mu', data=mu, compression='gzip', compression_opts=5)
        f.create_dataset('flux', data=flux_temp, compression='gzip', compression_opts=5)

    return out_name

def parse_synmast_filename(name: str):
    """
    Parse (Teff, logg, FeH) from a SYNMAST .prf filename by fixed slices.

    Expected prefix and layout like:
      mc_5400400+000_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf
         ^^^^ ^^^ ^^^^
          T   g*100  [Fe/H]*100 (signed, 3 digits)

    Returns
    -------
    (Teff:int, logg:float, FeH:float)
    """
    base = Path(name).name
    if not base.startswith("mc_") or not base.endswith(".prf"):
        raise ValueError(f"Not a SYNMAST .prf file: {name}")
    core = base[3:]              # drop "mc_"
    T_str   = core[0:4]          # 4 digits
    g_str   = core[4:7]          # 3 digits (×100)
    fe_str  = core[7:11]         # signed 3 digits (×100), e.g. +000, -050
    try:
        T   = int(T_str)
        logg = int(g_str) / 100.0
        FeH  = int(fe_str) / 100.0
    except Exception as e:
        raise ValueError(f"Unexpected filename layout for {name}") from e
    return T, logg, FeH

def build_synmast_filename(T: int, logg: float, FeH: float) -> str:
    """
    Assemble a SYNMAST filename stem using the same fixed layout.

    Returns e.g.:
      mc_5400400+000_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf
    """
    # 4-digit Teff
    T_str = f"{int(round(T)):04d}"
    # logg ×100 as 3 digits
    g_str = f"{int(round(logg*100)):03d}"
    # signed FeH ×100 as ±ddd
    val = abs(int(round(FeH*100)))
    fe_str = f"{'+' if FeH >= 0 else '-'}{val:03d}"
    return (
        f"mc_{T_str}{g_str}{fe_str}"
        "_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf"
    )


def prep_synmast_intensity_spec(temp, logg, FeH, out_path, syn_marcs):
    """
    Build a SYNMAST *intensity* spectrum I(λ, μ) at requested (temp, logg, FeH)
    by interpolating in Teff between the two nearest grid files. Keeps full μ grid.

    Saves HDF5:
      /wave : float32 (vac Å), shape (n_wave,)
      /mu   : float32,           shape (n_mu,)
      /flux : float64,           shape (n_wave, n_mu)

    Returns
    -------
    out_name (str): path to written HDF5
    """
    syn_marcs = Path(syn_marcs)
    out_path = Path(out_path)

    # Collect grid points from filenames
    Ts, Gs, Zs = [], [], []
    for p in syn_marcs.glob("*.prf"):
        try:
            T_i, g_i, z_i = parse_synmast_filename(p.name)
        except ValueError:
            continue
        Ts.append(T_i); Gs.append(g_i); Zs.append(z_i)

    if not Ts:
        raise FileNotFoundError(f"No SYNMAST .prf files found in {syn_marcs}")

    Ts = np.array(sorted(set(Ts)), dtype=np.int32)
    Gs = np.array(sorted(set(Gs)), dtype=np.float32)
    Zs = np.array(sorted(set(Zs)), dtype=np.float32)

    # Snap logg/[FeH] to nearest available values
    logg_val = float(Gs[np.argmin(np.abs(Gs - logg))])
    FeH_val  = float(Zs[np.argmin(np.abs(Zs - FeH))])

    # Find bracketing Teff (or exact match)
    iT = int(np.argmin(np.abs(Ts - temp)))
    if Ts[iT] == int(temp):
        T_low = T_high = int(Ts[iT]); alpha = 0.0
    else:
        if temp > Ts[iT]:
            i_low  = iT
            i_high = min(iT + 1, Ts.size - 1)
        else:
            i_low  = max(iT - 1, 0)
            i_high = iT
        T_low, T_high = int(Ts[i_low]), int(Ts[i_high])
        denom = float(T_high - T_low) if T_high != T_low else 1.0
        alpha = float((temp - T_low) / denom)

    # Filenames
    name_low  = build_synmast_filename(T_low,  logg_val, FeH_val)
    name_high = build_synmast_filename(T_high, logg_val, FeH_val)
    name_use  = build_synmast_filename(int(round(temp)), logg, FeH)  # label only

    path_low  = syn_marcs / name_low
    path_high = syn_marcs / name_high
    if not path_low.exists():
        raise FileNotFoundError(f"Missing grid file: {path_low}")
    if not path_high.exists():
        raise FileNotFoundError(f"Missing grid file: {path_high}")

    # Read endpoints as intensity (get_intensity=True); convert to vacuum
    wave_air, mu, I_low = rds3v(str(path_low),  vsini=0, vrt=2.0, if_flat=False, get_intensity=True)
    _,       _,  I_high = rds3v(str(path_high), vsini=0, vrt=2.0, if_flat=False, get_intensity=True)
    wave_vac = air2vac(wave_air)

    # Ensure shapes: I: (n_wave, n_mu), μ: (n_mu,)
    I_low  = I_low[:, None]  if I_low.ndim  == 1 else I_low
    I_high = I_high[:, None] if I_high.ndim == 1 else I_high
    if I_low.shape != I_high.shape:
        raise ValueError(f"Low/high intensity shapes differ: {I_low.shape} vs {I_high.shape}")

    # Linear interpolation in Teff (float64 math)
    if T_low == T_high:
        I_interp = I_low.astype(np.float64, copy=False)
    else:
        I_interp = (I_low.astype(np.float64) * (1.0 - alpha) +
                    I_high.astype(np.float64) * alpha)

    # Save (wave→f32, mu→f32, flux→f64)
    out_name = out_path / "SimUse" / name_use.replace(".prf", "_intensity.h5")
    with h5py.File(out_name, "w") as h:
        h.create_dataset("wave", data=wave_vac.astype(np.float32, copy=False),
                         compression="gzip", compression_opts=5, shuffle=True)
        h.create_dataset("mu",   data=np.asarray(mu, dtype=np.float32),
                         compression="gzip", compression_opts=5, shuffle=True)
        h.create_dataset("flux", data=I_interp,  # float64
                         compression="gzip", compression_opts=5, shuffle=True)
        # provenance
        h.attrs["grid_T_low"]  = T_low
        h.attrs["grid_T_high"] = T_high
        h.attrs["alpha"]       = alpha
        h.attrs["grid_logg"]   = logg_val
        h.attrs["grid_FeH"]    = FeH_val
        h.attrs["target_Teff"] = float(temp)
        h.attrs["target_logg"] = float(logg)
        h.attrs["target_FeH"]  = float(FeH)

    return str(out_name)



def prep_synmast_flux_spec(temp, logg, FeH, out_path):
    """
    Build a SYNMAST *flux* spectrum at the requested (temp, logg, FeH) by
    linear interpolation in Teff between the two nearest grid files, with
    logg and FeH snapped to the closest available values.

    Saves HDF5:
      /wave : float32 (vacuum Å), shape (n_wave,)
      /flux : float64, shape (n_wave, n_mu)  (or (n_wave, 1) if only one μ)

    Returns
    -------
    out_name (str): path to written HDF5
    """
    # location of the SYNMAST MARCS grid
    syn_marcs = Path(
        "./data/synmast/LisaGrid_synmast_v1.9_spirou_marcs"
    )
    out_path = Path(out_path)

    # collect grid points (from filenames)
    Ts, Gs, Zs = [], [], []
    prf_files = []
    for p in syn_marcs.glob("*.prf"):
        try:
            T_i, g_i, z_i = parse_synmast_filename(p.name)
        except ValueError:
            continue
        Ts.append(T_i); Gs.append(g_i); Zs.append(z_i)
        prf_files.append(p)

    if not prf_files:
        raise FileNotFoundError(f"No SYNMAST .prf files found in {syn_marcs}")

    Ts = np.array(sorted(set(Ts)), dtype=np.int32)
    Gs = np.array(sorted(set(Gs)), dtype=np.float32)
    Zs = np.array(sorted(set(Zs)), dtype=np.float32)

    # snap logg, FeH to nearest grid values
    logg_val = float(Gs[np.argmin(np.abs(Gs - logg))])
    FeH_val  = float(Zs[np.argmin(np.abs(Zs - FeH))])

    # bracketing Teff (or exact match)
    iT = int(np.argmin(np.abs(Ts - temp)))
    if Ts[iT] == int(temp):
        T_low = T_high = int(Ts[iT]); alpha = 0.0
    else:
        if temp > Ts[iT]:
            i_low  = iT
            i_high = min(iT + 1, Ts.size - 1)
        else:
            i_low  = max(iT - 1, 0)
            i_high = iT
        T_low, T_high = int(Ts[i_low]), int(Ts[i_high])
        denom = float(T_high - T_low) if T_high != T_low else 1.0
        alpha = float((temp - T_low) / denom)

    # filenames for endpoints and label for output
    name_low  = build_synmast_filename(T_low,  logg_val, FeH_val)
    name_high = build_synmast_filename(T_high, logg_val, FeH_val)
    name_use  = build_synmast_filename(int(round(temp)), logg, FeH)  # label only

    path_low  = syn_marcs / name_low
    path_high = syn_marcs / name_high
    if not path_low.exists():
        raise FileNotFoundError(f"Missing grid file: {path_low}")
    if not path_high.exists():
        raise FileNotFoundError(f"Missing grid file: {path_high}")

    # read endpoints (no flatten), convert to vacuum
    wave_air, spec_low = rds3v(str(path_low),  vsini=0, vrt=2.0, if_flat=False, get_intensity=False)
    _,        spec_high = rds3v(str(path_high), vsini=0, vrt=2.0, if_flat=False, get_intensity=False)
    wave_vac = air2vac(wave_air)

    # ensure 2-D flux shape (n_wave, n_mu) for consistency
    spec_low  = spec_low[:, None]  if spec_low.ndim  == 1 else spec_low
    spec_high = spec_high[:, None] if spec_high.ndim == 1 else spec_high
    if spec_low.shape != spec_high.shape:
        raise ValueError(f"Low/high spectra shapes differ: {spec_low.shape} vs {spec_high.shape}")

    # linear interpolation in Teff (float64 math)
    if T_low == T_high:
        flux_interp = spec_low.astype(np.float64, copy=False)
    else:
        flux_interp = (spec_low.astype(np.float64) * (1.0 - alpha) +
                       spec_high.astype(np.float64) * alpha)

    # save: wave→float32, flux→float64
    out_name = out_path / "SimUse" / name_use.replace(".prf", "_flux.h5")
    with h5py.File(out_name, "w") as h:
        h.create_dataset("wave", data=wave_vac.astype(np.float32, copy=False),
                         compression="gzip", compression_opts=5, shuffle=True)
        h.create_dataset("flux", data=flux_interp,  # float64
                         compression="gzip", compression_opts=5, shuffle=True)
        # provenance for reproducibility
        h.attrs["grid_T_low"]  = T_low
        h.attrs["grid_T_high"] = T_high
        h.attrs["alpha"]       = alpha
        h.attrs["grid_logg"]   = logg_val
        h.attrs["grid_FeH"]    = FeH_val
        h.attrs["target_Teff"] = float(temp)
        h.attrs["target_logg"] = float(logg)
        h.attrs["target_FeH"]  = float(FeH)

    return str(out_name)



def flatten_phx_with_rassine(rassine_path, out_path, phx_flux_file):
    """
    Run RASSINE to continuum-flatten a PHX spectrum HDF and save a compact HDF
    with only wave/flux/continuum_cubic.

    Inputs
    ------
    out_path      : base output directory (string or Path)
    phx_flux_file : filename of the PHX spectrum HDF under RASSINE_PHX/ (with or without .h5)
    rassine_path  : path to the RASSINE repo (directory containing Rassine.py)

    Returns
    -------
    outname : Path to the flattened HDF at RASSINE_PHX/RASSINE_<phx_flux_file>.h5
    """
    out_path = Path(out_path)
    rassine_dir = out_path / "RASSINE_PHX"
    # rassine_dir.mkdir(parents=True, exist_ok=True)

    # canonicalize input/output files
    base = phx_flux_file.replace(".h5", "")
    in_hdf  = rassine_dir / f"{base}.h5"                 # input PHX HDF
    in_pkl  = rassine_dir / f"{base}.p"                  # temp pickle for RASSINE
    out_hdf = rassine_dir / f"RASSINE_{base}.h5"         # final compact HDF
    out_pkl = rassine_dir / f"RASSINE_{base}.p"          # RASSINE output pickle

    # short-circuit if already done
    if out_hdf.exists():
        logger.debug(f"RASSINE flattened spectrum already exists: {out_hdf}")
        return str(out_hdf)

    logger.info(f"Flattening PHX spectrum with RASSINE → {out_hdf}")

    # 1) RASSINE takes a pickle; convert HDF → pickle (no unnecessary copies)
    #    Expect columns: 'wave', 'flux', maybe others. Keep full DF; RASSINE needs metadata.
    try:
        df_in = pd.read_hdf(in_hdf)  # key='spec' by default; adjust if you used a different key
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Input PHX HDF not found: {in_hdf}") from e

    df_in.to_pickle(in_pkl)  # protocol chosen by pandas; fine for RASSINE

    # 2) Run RASSINE
    rassine_py = Path(rassine_path) / "Rassine.py"
    if not rassine_py.exists():
        raise FileNotFoundError(f"RASSINE script not found: {rassine_py}")

    # RASSINE CLI: Rassine.py <input_pickle> <output_dir> <rassine_root_dir>
    # ensure trailing slash for dirs Rassine concatenates with filenames
    out_dir = str(rassine_dir).rstrip("/") + "/"
    root_dir = str(rassine_path).rstrip("/") + "/"

    cmd = [
        "python3",
        str(rassine_py),
        str(in_pkl),
        out_dir,   # <output_dir> with trailing '/'
        root_dir,  # <rassine_root_dir> with trailing '/'
    ]
    logger.info("\nUse RASSINE to flatten the spectrum ...")
    logger.debug("Running > %s", " ".join(cmd))

    # run and capture output; raise on non-zero
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if res.stdout:
            logger.debug("RASSINE stdout:\n%s", res.stdout.strip())
        if res.stderr:
            # RASSINE is chatty on stderr sometimes; keep it at debug unless non-empty and concerning
            logger.debug("RASSINE stderr:\n%s", res.stderr.strip())
    except subprocess.CalledProcessError as e:
        # keep the temp files for debugging
        logger.error("RASSINE failed (exit %s). STDERR:\n%s", e.returncode, e.stderr)
        raise

    # 3) Read RASSINE output pickle and save a compact HDF with only needed columns
    try:
        rass = pd.read_pickle(out_pkl)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Expected RASSINE output pickle not found: {out_pkl}") from e
    
    # Expected structure: wave / flux / output.continuum_cubic
    try:
        wave = rass["wave"].astype(np.float32, copy=False)
        flux = rass["flux"].astype(np.float64, copy=False)
        cont = rass["output"]["continuum_cubic"].astype(np.float64, copy=False)
    except Exception as e:
        raise RuntimeError("RASSINE output format unexpected; missing keys for wave/flux/continuum_cubic") from e

    df_out = pd.DataFrame(
        {"wave": wave, "flux": flux, "continuum_cubic": cont}
    )
    df_out.to_hdf(out_hdf, key="spec", complevel=5, index=False)

    # 4) Clean up only the temporary pickles we made/used
    for p in (in_pkl, out_pkl):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            # non-fatal; leave crumbs if FS is weird
            pass

    return str(out_hdf)


def final_flux_spec(
    spec_type, spec_dir, sim_spec_path, wvmin, wvmax, Vel_bin_size, plot,
    flatten=True, pars=None
    ):
    """
    Load a synthetic spectrum, optionally continuum-flatten it, cut to [wvmin,wvmax],
    convert to velocity space around the band center, resample to a uniform velocity
    grid, and save as an HDF table with columns 'v' and 'f'.

    Args
    ----
    spec_type      : {'phx','synmast','ccf_temp'}
    spec_dir       : input file path (PHX HDF, Synmast HDF5, or s3v path)
    sim_spec_path  : output HDF path for the (v,f) spectrum
    wvmin, wvmax   : wavelength cut [Å]
    Vel_bin_size   : velocity step (km/s) for resampling
    plot           : truthy → also write a quick PNG preview next to the HDF
    flatten        : if True, divide by continuum (when supported)
    pars           : extra parameters, used for 'ccf_temp' (expects
                     (_, _, _, ccf_vsini, ccf_resolution))

    Notes
    -----
    * PHX: expects columns 'wave', 'flux', 'continuum_cubic' in the input HDF.
    * Synmast: expects HDF5 datasets '/wave' and '/flux'.
    * Output 'v' is centered at (wvmin+wvmax)/2 and sampled uniformly with step Vel_bin_size.
    * We trim 200 points on each edge after spline resampling to reduce edge artifacts.
    """

    # ---------- Load ----------
    spec = spec_type.lower()
    if spec == 'phx':
        # Pandas is convenient; use .values to avoid extra copies.
        df = pd.read_hdf(spec_dir)
        wave = df['wave'].values.astype(np.float32, copy=False)
        if flatten:
            flux = (df['flux'].values / df['continuum_cubic'].values).astype(np.float32, copy=False)
        else:
            flux = df['flux'].values.astype(np.float64, copy=False)

    elif spec == 'synmast':
        # Read raw HDF5 datasets; avoid pandas overhead.
        with h5py.File(spec_dir, 'r') as f:
            wave = f['wave'][:].astype(np.float32, copy=False)
            flux = f['flux'][:].astype(np.float64, copy=False)   # shape (n_wv, [n_mu] or 1)

    elif spec == 'ccf_temp':
        # Example: read with your s3v reader, then convert to vacuum wavelengths.
        # pars expected: (_, _, _, ccf_vsini, ccf_resolution)
        _, _, _, ccf_vsini, ccf_resolution = pars
        wave, flux = rds3v(spec_dir, vsini=ccf_vsini, vrt=2.0, instrum=ccf_resolution, if_flat=flatten)
        wave = air2vac(wave).astype(np.float32, copy=False)
        flux = np.asarray(flux, dtype=np.float64)

    else:
        raise ValueError(f"Unknown spec_type: {spec_type}")


    # ---------- Band cut ----------
    # Keep only wavelengths within [wvmin, wvmax]
    sel = (wave >= wvmin) & (wave <= wvmax)
    w = wave[sel]
    f = flux[sel]

    # ---------- Wavelength → velocity ----------
    # Center velocity at the middle of the wavelength interval.
    c_kms = 299_792.458
    w_center = 0.5 * (wvmin + wvmax)
    v = c_kms * (w - w_center) / w_center  # km/s

    # ---------- Resample to uniform velocity grid ----------
    # Fit a cubic spline to (v, f) and evaluate on a uniform grid with step Vel_bin_size.
    # We build a symmetric grid around zero using the min extent.
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    v_span = min(-v_min, v_max)  # symmetric half-span about 0
    # Build symmetric grid [-v_span, +v_span] with step Vel_bin_size (inclusive on left).
    n_steps = int(np.floor((2.0 * v_span) / Vel_bin_size)) + 1
    v_new = (-v_span) + Vel_bin_size * np.arange(n_steps, dtype=np.float64)

    # Spline & evaluate
    tck = interpolate.splrep(v, f, s=0)
    f_new = interpolate.splev(v_new, tck, der=0)

    # Trim edges to mitigate spline ringing (keep center region)
    trim = 200
    if v_new.size > 2 * trim:
        v_new = v_new[trim:-trim]
        f_new = f_new[trim:-trim]

    # ---------- Save ----------
    simulation_spec = pd.DataFrame({
        'v': np.asarray(v_new, dtype=np.float32),
        'f': np.asarray(f_new, dtype=np.float64)
        })
    simulation_spec.to_hdf(sim_spec_path, key='spec', complevel=5, index=False)
    
    ########################################################
    if plot == 1:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(6, 4), facecolor='white', dpi=200)
        
        ax1.plot(w, f, '-', lw=1)
        ax2.plot(v_new, f_new, '-', lw=1)
        
        ax1.set_xlabel('Wavelength (AA)', fontsize='small')
        ax2.set_xlabel('Velocity (km/s)', fontsize='small')
        ax1.set_ylabel('Flux', fontsize='small')
        ax2.set_ylabel('Flux', fontsize='small')

        if flatten:
            ax1.set_ylim(0, 1.1)
            ax2.set_ylim(0, 1.1)
        fig.savefig(sim_spec_path.replace('.h5', '.png'))    



def final_intensity_spec(
    model_type, spec_dir, sim_spec_path, wvmin, wvmax, Vel_bin_size, plot,
    flatten=True, pars=None
    ):
    """
    Load an *intensity* spectrum I(λ, μ) (PHX / Synmast / template),
    cut to [wvmin, wvmax], convert to velocity space about the band center,
    resample each μ column to a uniform v-grid, and save:

        HDF columns: 'v', 'mu_<value0>', 'mu_<value1>', ...

    Notes
    -----
    * PHX: expects HDF5 datasets: /wave, /mu, /flux. For PHX we reverse μ to descending
      and align columns accordingly. Flattening is *not* supported for PHX intensity.
    * Synmast: expects /wave, /mu, /flux; we keep the 2-D shape (n_wave, n_mu).
    * We trim 200 points at the ends after spline resampling to reduce edge artifacts.
    """

    spec = model_type.lower()
    # ---------- Read wave, mu, fluxes ----------
    if spec == 'phx':
        with h5py.File(spec_dir, 'r') as f:
            wave   = f['wave'][:].astype(np.float32, copy=False)          # (n_wave,)
            mu     = f['mu'][:].astype(np.float32, copy=False)[::-1]      # descending μ
            fluxes = f['flux'][:].astype(np.float64, copy=False).T[:, ::-1]  # (n_wave, n_mu), align with μ
        if flatten:
            raise ValueError('Flattening PHX intensity spectra is not supported.')

    elif spec == 'synmast':
        with h5py.File(spec_dir, 'r') as f:
            wave   = f['wave'][:].astype(np.float32, copy=False)
            mu     = f['mu'][:].astype(np.float32, copy=False)
            fluxes = f['flux'][:].astype(np.float64, copy=False)           # (n_wave, n_mu) or (n_wave,)
        if fluxes.ndim == 1:
            fluxes = fluxes[:, None]  # promote to (n_wave, 1)

    elif spec == 'ccf_temp':
        # Example placeholder; adapt if you truly need intensity from this path.
        _, _, _, ccf_vsini, ccf_resolution = pars
        wave, flux1d = rds3v(spec_dir, vsini=ccf_vsini, vrt=2.0,
                             instrum=ccf_resolution, if_flat=flatten)
        wave = air2vac(wave).astype(np.float32, copy=False)
        fluxes = np.asarray(flux1d, dtype=np.float64)
        if fluxes.ndim == 1:
            fluxes = fluxes[:, None]
        # If there is no μ grid from this source, synthesize a trivial one:
        mu = np.array([1.0], dtype=np.float32)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ---------- Band cut ----------
    sel = (wave >= wvmin) & (wave <= wvmax)
    w = wave[sel]           # (n_cut,)
    F = fluxes[sel, :]      # (n_cut, n_mu)
    if w.size < 10:
        raise ValueError(f"Too few points in [{wvmin},{wvmax}] to resample (got {w.size}).")

    # ---------- Wavelength → velocity ----------
    # Center v at the midpoint of the wavelength window.
    c_kms = 299_792.458
    w_center = 0.5 * (wvmin + wvmax)
    v = c_kms * (w - w_center) / w_center    # (n_cut,)

    # Choose velocity step:
    # - PHX: if you want to preserve the native sampling, use native Δv
    # - else: use the requested Vel_bin_size
    if spec == 'phx':
        dv_native = float(np.max(np.diff(v))) if v.size > 1 else Vel_bin_size
        dv = dv_native
    else:
        dv = float(Vel_bin_size)

    # Build a *symmetric* uniform v-grid about 0 with step dv
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    v_span = min(-v_min, v_max)                      # symmetric half-span we can trust on both sides
    n_steps = int(np.floor((2.0 * v_span) / dv)) + 1
    v_new = (-v_span) + dv * np.arange(n_steps, dtype=np.float64)   # (n_v,)

    # ---------- Spline-resample each μ independently ----------
    n_mu = F.shape[1]
    f_new = np.empty((v_new.size, n_mu), dtype=np.float64)
    for i in range(n_mu):
        tck = interpolate.splrep(v, F[:, i], s=0)
        f_new[:, i] = interpolate.splev(v_new, tck, der=0)

    # Trim edges to mitigate spline ringing (keep center region only)
    trim = 200
    if 'Photo' not in sim_spec_path:
        v_new = v_new[trim:-trim]
        f_new = f_new[trim:-trim, :]

    # ---------- Save (HDF with one column per μ) ----------
    # Cast to float32 for storage; math above used float64.
    v_out = v_new.astype(np.float32, copy=False)
    f_out = f_new.astype(np.float64, copy=False)

    # Build a tidy DataFrame: 'v', 'mu_<val0>', 'mu_<val1>', ...
    # Ensure μ labels are consistent with column order
    df_dict = {'v': v_out}
    for i in range(n_mu):
        df_dict[f'mu_{mu[i]}'] = f_out[:, i]

    simulation_spec = pd.DataFrame(df_dict)
    simulation_spec.to_hdf(sim_spec_path, key='spec', complevel=5, index=False)
    
    ########################################################
    if plot == 1:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(6, 4), facecolor='white', dpi=200)
        
        for i in range(F.shape[1]):
            ax1.plot(w, F[:, i], '-', lw=1, label=f'mu={mu[i]}')
            ax2.plot(v_new, f_new[:, i], '-', lw=1, label=f'mu={mu[i]}')
        # ax1.plot(w, f, '-', lw=1)
        # ax2.plot(v_new, f_new, '-', lw=1)
        ax1.legend(fontsize='xx-small', loc='upper right')
        ax2.legend(fontsize='xx-small', loc='upper right')
        ax1.set_xlabel('Wavelength (AA)', fontsize='small')
        ax2.set_xlabel('Velocity (km/s)', fontsize='small')
        ax1.set_ylabel('Flux', fontsize='small')
        ax2.set_ylabel('Flux', fontsize='small')

        if flatten:
            ax1.set_ylim(0, 1.1)
            ax2.set_ylim(0, 1.1)

        fig.savefig(sim_spec_path.replace('.h5', '.png'))    



if __name__ == '__main__':
    # grav, wvmin, wvmax, FeH, alpha = 4.5, 15600, 15660, 0.1, 0.0
    grav, wvmin, wvmax, FeH, alpha = 4.0, 3000, 30000, 0.0, 0.0
    # temps = [2500, 3000, 3500, 4000, 5000, 6500]
    # temps = [2600, 2800, 3500]
    temps = np.arange(2400, 3001, 200)
    
    out_path = f'{os.getcwd()}/syn_spec/'
    rassine_path = './external/Rassine_public/'

    print('\nChecking spectra templates for')
    for temp in temps:
        
        print(f'   {temp} K\n')
        get_phx_flux_spec(temp, grav, FeH, alpha, out_path)
        # flatten_spec(rassine_path, out_path, temp, grav, FeH, wvmin, wvmax)
        # print('\n Phoenix spectra are ready for use.')



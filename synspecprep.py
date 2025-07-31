"""
File modified from https://github.com/YinanZhao21/SOAP_GPU/blob/main/SOAP_code/phoenix_extraction.py
Download PHX synthetic spectra and flatten them using Rassine

Shih-Yun Tang 20241017
"""
import warnings, logging
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import bz2
import pandas as pd
import configparser
from astropy.io import fits
import astropy.units as u
from scipy import interpolate
from specutils import Spectrum1D
import h5py

from rds3v import rds3v
# from specutils.manipulation import FluxConservingResampler
# from specutils.analysis import gaussian_sigma_width, gaussian_fwhm, fwhm, fwzi
# from specutils import Spectrum1D, SpectralRegion

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

    ########################################################
    # f, ax = plt.subplots(1, 1, figsize=(6, 3), facecolor='white', dpi=200)
    # ax.plot(wavelength_vac, flux_temp, '-', lw=1)
    # # f.savefig(out_path+f'RASSINE_figs/PHX_ACES_spec_{int(temp)}_{grav}_{FeH}_wv{wvmin}_{wvmax}.png', dpi=200)
    # # ax.set_xscale('log')
    # plt.show()

# def get_phx_intensity_spec(temp, logg, FeH, alpha, phx_path, out_path):

#     # file name example:
#     #    ---> lte03900-3.50-0.0.PHOENIX-ACES-AGSS-COND-2011-SpecInt.fits

#     wvmin, wvmax = (3000, 30000)

#     index_logg = np.argmin(np.abs(LOGG_ARRAY - logg))
#     index_FeH = np.argmin(np.abs(FeH_ARRAY - FeH))
#     index_alpha = np.argmin(np.abs(ALPHA_ARRAY - alpha))
#     logg_val = LOGG_ARRAY[index_logg]
#     FeH_val = FeH_ARRAY[index_FeH]

#     if FeH_val > 0.0:
#         alpha_val = 0.0
#     else:
#         alpha_val = ALPHA_ARRAY[index_alpha]

#     index_temp = np.argmin(np.abs(TEMP_ARRAY - temp))
#     if temp > TEMP_ARRAY[index_temp]:
#         temp_low = TEMP_ARRAY[index_temp]
#         temp_high = TEMP_ARRAY[index_temp+1]
#     else:
#         temp_low = TEMP_ARRAY[index_temp-1]
#         temp_high = TEMP_ARRAY[index_temp]

#     sub_dir_low, name_low = PHOENIX_name(
#         temp_low, logg_val, FeH_val, alpha_val, fluxORint='int')
#     sub_dir_high, name_high = PHOENIX_name(
#         temp_high, logg_val, FeH_val, alpha_val, fluxORint='int')
    
#     _, name_use = PHOENIX_name(temp, logg, FeH, alpha)
#     name_use = name_use.replace('.fits', '').replace('/', '')
    
#     name_low_file = Path(f'{phx_path}/SpecInt/{sub_dir_high}/{name_low}')
#     name_high_file = Path(f'{phx_path}/SpecInt/{sub_dir_high}/{name_high}')
    
#     os.makedirs(f'{phx_path}/SpecInt/{sub_dir_low}', exist_ok=True)
#     if not name_low_file.is_file():
#         print('Download spectral data...')
#         os.system(f'curl {PHX_FTP_prefix_Int}/{sub_dir_low}/{name_low} -o {phx_path}/SpecInt/{sub_dir_low}/{name_low}')
#     else:
#         print(f'{name_low} exists...')
#     if not name_high_file.is_file():
#         print('Download spectral data...')
#         os.system(f'curl {PHX_FTP_prefix_Int}/{sub_dir_high}/{name_high} -o {phx_path}/SpecInt/{sub_dir_low}/{name_high}')
#     else:
#         print(f'{name_high} exists...')

#     # wave_data_vac = fits.getdata(phx_syn_path+name_high)
#     with fits.open(f'{phx_path}/SpecInt/{sub_dir_low}/{name_low}') as hdul:
#         wl0 = hdul[0].header['crval1'] # Wavelength at d[:,0] [AA]
#         dwl = hdul[0].header['cdelt1'] # Delta wavelength     [AA]
#         nwl = hdul[0].header['naxis1'] # Number of wl samples
#         wave_data_vac = wl0 + np.arange(nwl) * dwl
#         mu = hdul[1].data
#         spec_low_vac = hdul[0].data

    
#     waveIndex = (wave_data_vac >= (wvmin-500)) & (wave_data_vac <= (wvmax+500))
    
#     wavelength_vac = wave_data_vac[waveIndex]
#     spec_low_vac = fits.getdata(f'{phx_path}/SpecInt/{sub_dir_low}/{name_low}')[:, waveIndex]
#     spec_high_vac = fits.getdata(f'{phx_path}/SpecInt/{sub_dir_low}/{name_high}')[:, waveIndex]
    
#     flux_temp = spec_low_vac + ( (temp - temp_low)/(temp_high - temp_low) )*(spec_high_vac - spec_low_vac)
#     data_name = out_path + f'PHX_ACES_SpecInt_{int(temp)}_{logg}_{FeH}_wv{wvmin}_{wvmax}.p'
#     print(f'   >> Save {data_name}')
#     mu_flux = {f'flux_{mu[i]}': flux_temp[i, :] for i in range(len(mu))}
#     mu_flux['wave'] = wavelength_vac
#     spec_df = pd.DataFrame(mu_flux)

#     # spec_df.to_pickle(data_name)
#     spec_df.to_hdf(data_name, complevel=5, index=False)

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

    
def prep_synmast_intensity_spec(temp, logg, FeH, out_path, syn_marcs):

    # Currently, phx int spectra is only use for photometry.
    # The intensity spectra from PHX has too low resolution in wavelength.
    # Use the low-res (50AA res) int spectra that already dw by ldtk pkg
    # syn_marcs = Path('./data/synmast/LisaGrid_synmast_v1.9_spirou_marcs')
    syn_marcs = Path(syn_marcs)
    syn_files = os.listdir(syn_marcs)
    # clean hidden files and leave file ended with .prf
    syn_files = [f for f in syn_files if f.endswith('.prf') and not f.startswith('.')]

    # make TEMP, LOGG, FeH, array for synmast spec
    # file name ex., mc_5400400+000_b4_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf
    syn_temp_array = np.unique(np.array([float(f[3:7]) for f in syn_files]))
    syn_logg_array = np.unique(np.array([float(f[7:10])/100 for f in syn_files]))
    syn_FeH_array = np.unique(np.array([float(f[10:14])/100 for f in syn_files]))
    
    index_logg = np.argmin(np.abs(syn_logg_array - logg))
    index_FeH = np.argmin(np.abs(syn_FeH_array - FeH))
    logg_val = syn_logg_array[index_logg]
    FeH_val = syn_FeH_array[index_FeH]

    index_temp = np.argmin(np.abs(syn_temp_array - temp))
    if temp > syn_temp_array[index_temp]:
        temp_low = syn_temp_array[index_temp]
        temp_high = syn_temp_array[index_temp+1]
    else:
        temp_low = syn_temp_array[index_temp-1]
        temp_high = syn_temp_array[index_temp]

    name_low = f'mc_{temp_low:1.0f}{logg_val*100:1.0f}{FeH_val*100:+04.0f}_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf'
    name_high = f'mc_{temp_high:1.0f}{logg_val*100:1.0f}{FeH_val*100:+04.0f}_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf'
    name_use = f'mc_{temp:1.0f}{logg*100:1.0f}{FeH*100:+04.0f}_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf'
    
    dir_low_file = str(syn_marcs / name_low)
    dir_high_file = str(syn_marcs / name_high)

    wave, mu, spec_low_vac = rds3v(
        dir_low_file, vsini=0, vrt=2.0, if_flat=False, 
        get_intensity=True)

    wave_data_vac = air2vac(wave)

    _, _, spec_high_vac = rds3v(
        dir_high_file, vsini=0, vrt=2.0, if_flat=False, 
        get_intensity=True)
    
    wavelength_vac = wave_data_vac
    spec_low_vac = spec_low_vac
    spec_high_vac = spec_high_vac

    # dimemnsion (15106, 7)
    flux_temp = spec_low_vac + ( (temp - temp_low)/(temp_high - temp_low) )*(spec_high_vac - spec_low_vac)

    # spec_df = pd.DataFrame({'wave':wavelength_vac, 'flux':flux_temp, 'mu':mu})    
    # spec_df.to_hdf(out_name, key='spec', complevel=5, index=False)

    out_name = f"{out_path}/SimUse/{name_use.replace('.prf', '.h5')}"
    with h5py.File(out_name, 'w') as f:
        f.create_dataset('wave', data=wavelength_vac, compression='gzip', compression_opts=5)
        f.create_dataset('mu', data=mu, compression='gzip', compression_opts=5)
        f.create_dataset('flux', data=flux_temp, compression='gzip', compression_opts=5)

    return out_name

def prep_synmast_flux_spec(temp, logg, FeH, out_path):

    # Currently, phx int spectra is only use for photometry.
    # The intensity spectra from PHX has too low resolution in wavelength.
    # Use the low-res (50AA res) int spectra that already dw by ldtk pkg
    syn_marcs = Path('./data/synmast/LisaGrid_synmast_v1.9_spirou_marcs')

    syn_files = os.listdir(syn_marcs)
    # clean hidden files and leave file ended with .prf
    syn_files = [f for f in syn_files if f.endswith('.prf') and not f.startswith('.')]

    # make TEMP, LOGG, FeH, array for synmast spec
    # file name ex., mc_5400400+000_b4_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf
    syn_temp_array = np.unique(np.array([float(f[3:7]) for f in syn_files]))
    syn_logg_array = np.unique(np.array([float(f[7:10])/100 for f in syn_files]))
    syn_FeH_array = np.unique(np.array([float(f[10:14])/100 for f in syn_files]))
    
    index_logg = np.argmin(np.abs(syn_logg_array - logg))
    index_FeH = np.argmin(np.abs(syn_FeH_array - FeH))
    logg_val = syn_logg_array[index_logg]
    FeH_val = syn_FeH_array[index_FeH]

    index_temp = np.argmin(np.abs(syn_temp_array - temp))
    if temp > syn_temp_array[index_temp]:
        temp_low = syn_temp_array[index_temp]
        temp_high = syn_temp_array[index_temp+1]
    else:
        temp_low = syn_temp_array[index_temp-1]
        temp_high = syn_temp_array[index_temp]

    name_low = f'mc_{temp_low:1.0f}{logg_val*100:1.0f}{FeH_val*100:+04.0f}_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf'
    name_high = f'mc_{temp_high:1.0f}{logg_val*100:1.0f}{FeH_val*100:+04.0f}_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf'
    name_use = f'mc_{temp:1.0f}{logg*100:1.0f}{FeH*100:+04.0f}_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf'
    
    dir_low_file = str(syn_marcs / name_low)
    dir_high_file = str(syn_marcs / name_high)

    wave, spec_low_vac = rds3v(
        dir_low_file, vsini=0, vrt=2.0, if_flat=False, 
        get_intensity=False)

    wave_data_vac = air2vac(wave)

    _, spec_high_vac = rds3v(
        dir_high_file, vsini=0, vrt=2.0, if_flat=False, 
        get_intensity=False)
    
    wavelength_vac = wave_data_vac
    spec_low_vac = spec_low_vac
    spec_high_vac = spec_high_vac

    # dimemnsion (15106, 7)
    flux_temp = spec_low_vac + ( (temp - temp_low)/(temp_high - temp_low) )*(spec_high_vac - spec_low_vac)

    # spec_df = pd.DataFrame({'wave':wavelength_vac, 'flux':flux_temp, 'mu':mu})    
    # spec_df.to_hdf(out_name, key='spec', complevel=5, index=False)

    out_name = f"{out_path}/SimUse/{name_use.replace('.prf', '_flux.h5')}"
    with h5py.File(out_name, 'w') as f:
        f.create_dataset('wave', data=wavelength_vac, compression='gzip', compression_opts=5)
        f.create_dataset('flux', data=flux_temp, compression='gzip', compression_opts=5)

    return out_name



def flatten_spec(rassine_path, out_path, phx_flux_file):

    phx_flux_file = phx_flux_file.replace('.h5', '')

    outname = f'{out_path}/RASSINE_PHX/RASSINE_{phx_flux_file}.h5'
    if os.path.exists(outname): 
        logger.debug(f'RASSINE flattened spectrum already exists: {outname}')
        return outname
    else:
        logger.info(f'Flattening PHX spectrum: {outname}')
    
    # RASSINE take pickle file as input
    pd.read_hdf(f'{out_path}/RASSINE_PHX/{phx_flux_file}.h5').to_pickle(
        f'{out_path}/RASSINE_PHX/{phx_flux_file}.p')

    # cmds = 'python3 '+rassine_path+'Rassine.py '+out_path+'New_Phoenix_'+str(int(temp))+'.p '+out_path+' '+rassine_path
    cmds = f'python3 {rassine_path}/Rassine.py ' + \
            f'{out_path}/RASSINE_PHX/{phx_flux_file}.p ' + \
            f'{out_path}/RASSINE_PHX {rassine_path}/'
    logger.debug('\nUse Rassine to flatten the spectrum...\n')
    logger.debug(f'Running > {cmds}\n')
    os.system(cmds)

    # Only save wave, flux, and continuum_cubic to reduce file size
    rassine_result = pd.read_pickle(
        f'{out_path}/RASSINE_PHX/RASSINE_{phx_flux_file}.p')

    rassine_clean = pd.DataFrame(
        {'wave':rassine_result['wave'],
         'flux':rassine_result['flux'],
         'continuum_cubic':rassine_result['output']['continuum_cubic']}
         )

    rassine_clean.to_hdf(outname, key='spec', complevel=5, index=False)

    # remove all the temperary pickle files
    os.system(f'rm {out_path}/RASSINE_PHX/*.p')
    
    return outname


def final_flux_spec(
        spec_type, spec_dir, sim_spec_path, wvmin, wvmax, Vel_bin_size, plot,
        flatten=True, pars=None):
    # prep for SOAP3 use syn spec

    if spec_type.lower() == 'phx':
        final_spec = pd.read_hdf(spec_dir)
        wave = np.array(final_spec['wave'])
        if flatten:
            flux = np.array(final_spec['flux']/final_spec['continuum_cubic'])
        else:
            flux = np.array(final_spec['flux'])

    elif spec_type.lower() == 'synmast':
        # wave, flux = rds3v(spec_dir, vsini=0, vrt=2.0, if_flat=flatten)
        # wave = air2vac(wave)
        with h5py.File(spec_dir, 'r') as f:
            wave = f['wave'][:]
            flux = f['flux'][:] # (15106, 7)
            flux = np.array(flux)

    elif spec_type.lower() == 'ccf_temp':
        _, _, _, ccf_vsini, ccf_resolution = pars
        wave, flux = rds3v(
            spec_dir, vsini=ccf_vsini, vrt=2.0, instrum=ccf_resolution, 
            if_flat=flatten)
        wave = air2vac(wave)

    # cut to right wave to use for simulation
    waveIndex = (wave >= (wvmin)) & (wave <= (wvmax))
    f = flux[waveIndex]
    w = wave[waveIndex]    

    # print('\n', spec_dir)
    # print('wavelength range:', wvmin, wvmax)
    # print(w)

    # spec = Spectrum1D(
    #             flux = f*u.Unit('erg cm-2 s-1 AA-1'), 
    #             spectral_axis = w*u.AA, 
    #             rest_value = (wvmin+wvmax)/2*u.AA, 
    #             velocity_convention='relativistic' # “doppler_relativistic”, “doppler_optical”, “doppler_radio”
    #             )
    # w, v, f = spec.spectral_axis.value, spec.velocity.value, spec.flux.value
    c_kms = 299792.458
    w_center = (wvmin + wvmax) / 2
    v = c_kms * (w - w_center) / w_center

    tck = interpolate.splrep(v, f, s=0)    
    resolution = Vel_bin_size # km
    v_new = np.arange(
        np.round(np.min(v)), -np.round(np.min(v))+resolution, resolution)
    f_new = interpolate.splev(v_new, tck, der=0)

    v_new = v_new[200:-200]
    f_new = f_new[200:-200]

    simulation_spec = pd.DataFrame({'v':v_new,'f':f_new})

    simulation_spec.to_hdf(
        sim_spec_path, key='spec', complevel=5, index=False)
    
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
    # plt.show()

    # return simulation_spec


def final_intensity_spec(
        model_type, spec_dir, sim_spec_path, wvmin, wvmax, Vel_bin_size, plot,
        flatten=True, pars=None):

    if model_type.lower() == 'phx':
        with h5py.File(spec_dir, 'r') as f:
            wave = f['wave'][:]
            mu = f['mu'][:][::-1] # let mu from larger to small
            fluxes = f['flux'][:].T[:, ::-1]
        if flatten:
            # rise error indicate that furrent PHX int spectra is not support to flatten
            raise ValueError(
                'Flattening PHX intensity spectra is not supported')

    elif model_type.lower() == 'synmast':
        # wave, mu, fluxes = rds3v(
        #     spec_dir, vsini=0, vrt=2.0, if_flat=flatten, 
        #     get_intensity=True)
        # wave = air2vac(wave)
        with h5py.File(spec_dir, 'r') as f:
            wave = f['wave'][:]
            mu = f['mu'][:]
            fluxes = f['flux'][:] # (15106, 7)

        
    elif model_type.lower() == 'ccf_temp':
        _, _, _, ccf_vsini, ccf_resolution = pars
        wave, flux = rds3v(
            spec_dir, vsini=ccf_vsini, vrt=2.0, instrum=ccf_resolution, 
            if_flat=flatten)
        wave = air2vac(wave)

    # cut to right wave to use for simulation
    waveIndex = (wave >= (wvmin)) & (wave <= (wvmax))
    f = fluxes[waveIndex, :]
    w = wave[waveIndex]    

    # print('\n', spec_dir)
    c_kms = 299792.458
    w_center = (wvmin + wvmax) / 2
    v = c_kms * (w - w_center) / w_center
    if model_type.lower() != 'phx':
        resolution = Vel_bin_size # km
    else:
        resolution = np.max(np.diff(v)) # km/s
        # print('resolution:', resolution)

    v_new = np.arange(
            np.round(np.min(v)), -np.round(np.min(v))+resolution, resolution)
    
    f_new = np.zeros((len(v_new), f.shape[1]))
    for i in range(f.shape[1]):
        tck = interpolate.splrep(v, f[:, i], s=0)    
        f_new[:, i] = interpolate.splev(v_new, tck, der=0)

    if 'Photo' not in sim_spec_path:
        v_new = v_new[200:-200]
        f_new = f_new[200:-200, :]

    _pd_dict = {'v':v_new}
    for i in range(f_new.shape[1]):
        _pd_dict[f'mu_{mu[i]}'] = f_new[:, i]
    simulation_spec = pd.DataFrame(_pd_dict)

    simulation_spec.to_hdf(
        sim_spec_path, key='spec', complevel=5, index=False)
    
    ########################################################
    if plot == 1:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(6, 4), facecolor='white', dpi=200)
        
        for i in range(f.shape[1]):
            ax1.plot(w, f[:, i], '-', lw=1, label=f'mu={mu[i]}')
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
    # plt.show()
    # return simulation_spec


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



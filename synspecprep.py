"""
File modified from https://github.com/YinanZhao21/SOAP_GPU/blob/main/SOAP_code/phoenix_extraction.py
Download PHX synthetic spectra and flatten them using Rassine

Shih-Yun Tang 20241017
"""
import warnings
warnings.filterwarnings("ignore")
import os
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

def PHOENIX_name(temp, logg_val, FeH_val, alpha_val):
    if FeH_val <= 0.0:
        prefix_subdir = 'Z-{:.1f}'.format(np.abs(FeH_val))
        if alpha_val == 0.0:
            name = '/lte0{}-{:.2f}-{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(
                int(temp), logg_val, np.abs(FeH_val))
        elif alpha_val > 0.0:
            name = '/lte0{}-{:.2f}-{:.1f}.Alpha=+{:.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(
                int(temp), logg_val, np.abs(FeH_val), alpha_val)
        else:
            name = '/lte0{}-{:.2f}-{:.1f}.Alpha={:.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(
                int(temp), logg_val, np.abs(FeH_val), alpha_val)
    else:
        prefix_subdir = 'Z+{:.1f}'.format(np.abs(FeH_val))
        name = '/lte0{}-{:.2f}+{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(
            int(temp), logg_val, np.abs(FeH_val))
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

PHX_FTP_prefix = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'
PHX_FTP_wave_prefix = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/'
WAVE_FILE = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

def get_phx_spec(temp, grav, FeH, alpha, out_path):

    wvmin, wvmax = (3000, 30000)

    index_logg = np.argmin(np.abs(LOGG_ARRAY - grav))
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
    
    wave_file = Path(out_path+WAVE_FILE)
    name_low_file = Path(out_path+name_low)
    name_high_file = Path(out_path+name_high)
    
    if not wave_file.is_file():
        print('Download wavelength data...')
        os.system(f'curl {PHX_FTP_wave_prefix}{WAVE_FILE} -o {out_path}{WAVE_FILE}')
    else:
        print('Wavelength data exists...')
    if not name_low_file.is_file():
        print('Download spectral data...')
        os.system(f'curl {PHX_FTP_prefix}{sub_dir_low}{name_low} -o {out_path}{name_low}')
    else:
        print(f'{name_low} exists...')
    if not name_high_file.is_file():
        print('Download spectral data...')
        os.system(f'curl {PHX_FTP_prefix}{sub_dir_high}{name_high} -o {out_path}{name_high}')
    else:
        print(f'{name_high} exists...')

    wave_data_vac = fits.getdata(out_path+WAVE_FILE)
    
    waveIndex = (wave_data_vac >= (wvmin-500)) & (wave_data_vac <= (wvmax+500))
    
    wavelength_vac = wave_data_vac[waveIndex]
    spec_low_vac = fits.getdata(out_path+name_low)[waveIndex]
    spec_high_vac = fits.getdata(out_path+name_high)[waveIndex]

    # wavelength_air = vac2air(wavelength_vac)
    # waveIndex_new = (wavelength_air >= wvmin) & (wavelength_air <= wvmax)
    # wavelength_air_new = wavelength_air[waveIndex_new]
    # spec_low_air = spec_low[waveIndex_new]
    # spec_high_air = spec_high[waveIndex_new]
    
    # flux_temp = np.zeros(wavelength_vac.size)
    # for i in np.arange(wavelength_vac.size):
    #     flux_temp[i] = spec_low_vac[i] + ( (temp - temp_low)/(temp_high - temp_low) )*(spec_high_vac[i] - spec_low_vac[i])
    
    flux_temp = spec_low_vac + ( (temp - temp_low)/(temp_high - temp_low) )*(spec_high_vac - spec_low_vac)

    integrated_flux = np.trapz(flux_temp, wavelength_vac)
    data_name = out_path + f'PHX_ACES_spec_{int(temp)}_{grav}_{FeH}_wv{wvmin}_{wvmax}.p'
    # print(f'   >> Save {data_name}')
    spec_df = pd.DataFrame({'wave':wavelength_vac,'flux':flux_temp})
    spec_df.to_pickle(data_name)

    ########################################################
    # f, ax = plt.subplots(1, 1, figsize=(6, 3), facecolor='white', dpi=200)
    # ax.plot(wavelength_vac, flux_temp, '-', lw=1)
    # # f.savefig(out_path+f'RASSINE_figs/PHX_ACES_spec_{int(temp)}_{grav}_{FeH}_wv{wvmin}_{wvmax}.png', dpi=200)
    # # ax.set_xscale('log')
    # plt.show()


def flatten_spec(rassine_path, out_path, temp, grav, FeH):
    outname = f'{out_path}RASSINE_PHX_ACES_spec_{int(temp)}_{grav}_{FeH}_wv3000_30000.p'

    if os.path.exists(outname):
        # print('Already have the spec:', outname)
        return outname

    # cmds = 'python3 '+rassine_path+'Rassine.py '+out_path+'New_Phoenix_'+str(int(temp))+'.p '+out_path+' '+rassine_path
    cmds = f'python3 {rassine_path}Rassine.py {out_path}PHX_ACES_spec_{int(temp)}_{grav}_{FeH}_wv3000_30000.p {out_path} {rassine_path}'
    print('\nUse Rassine to flatten the spectrum...\n')
    print(f'Running > {cmds}\n')
    os.system(cmds)
    # remove the original file
    # os.system(f'rm {out_path}PHX_ACES_spec_{int(temp)}_{grav}_wv{wvmin}_{wvmax}.p')

    # only save wave, flux, and continuum_cubic
    rassine_result = pd.read_pickle(outname)

    rassine_clean = pd.DataFrame(
        {'wave':rassine_result['wave'],
         'flux':rassine_result['flux'],
         'continuum_cubic':rassine_result['output']['continuum_cubic']}
         )
    rassine_clean.to_pickle(outname)
    return outname


def final_spec(
        spec_type, spec_dir, sim_spec_path, wvmin, wvmax, Vel_bin_size,
        flatten=True, pars=None):
    # prep for SOAP3 use syn spec

    if spec_type.lower() == 'phx':
        final_spec = pd.read_pickle(spec_dir)
        wave = np.array(final_spec['wave'])
        if flatten:
            flux = np.array(final_spec['flux']/final_spec['continuum_cubic'])
        else:
            flux = np.array(final_spec['flux'])

    elif spec_type.lower() == 'synmast':
        wave, flux = rds3v(spec_dir, vsini=0, vrt=2.0, if_flat=flatten)
        wave = air2vac(wave)
    
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

    simulation_spec.to_pickle(sim_spec_path)
    
    ########################################################
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
    fig.savefig(sim_spec_path.replace('.p', '_output.png'))    
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
        get_phx_spec(temp, grav, FeH, alpha, out_path)
        # flatten_spec(rassine_path, out_path, temp, grav, FeH, wvmin, wvmax)
        # print('\n Phoenix spectra are ready for use.')



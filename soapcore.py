import sys, os
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

from rebin_jv import rebin_jv


class Star:
    def __init__(
        self, prot, vrot, incl, psi, rad_sun, rad, 
        Temp, Tdiff_spot, Tdiff_faculae, logg, FeH, bands2use
        ):
        """_summary_

        Args:
            prot (float): Rotational period [days]
            vrot (float): Rotational velocity [km/s]
            incl (float): Inclination [deg]
            limba1 (float): linear limb-darkening coeff.
            limba2 (float): quadratic limb-darkening coeff.
            psi (float): Phase
            rad_sun (float): Sun radius [km]
            rad (float): Radius [Rsol]
            Temp (float): Effective temperature of the star [k]
            Temp_diff_spot (float): difference in temperature between 
                the spot temperature and the Teff (<0 for spots, > 1 for plages)
        """
        self.prot           = prot           
        self.vrot           = vrot           
        self.incl           = incl           
        # self.limba1         = limba1         
        # self.limba2         = limba2         
        self.psi            = psi            
        self.rad_sun        = rad_sun        
        self.rad            = rad            
        self.Temp           = Temp           
        self.Tdiff_spot     = Tdiff_spot 
        self.Tdiff_faculae  = Tdiff_faculae
        self.Tspot          = Temp-Tdiff_spot
        self.Tfaculae       = Temp+Tdiff_faculae
        self.logg           = logg
        self.FeH            = FeH
        self.bands2use      = bands2use
    
    def get_quadratic_limbdarken(
            self, temperature, line_centers, l_widths):
        
        if len(line_centers) == 1:
            line_centers = np.array([line_centers])
            l_widths = np.array([l_widths])
        elif len(line_centers) >1:
            line_centers = np.array(line_centers)
            l_widths = np.array(l_widths)

        # print('line_centers:', line_centers)

        # -- Limb darkening coefficients --
        lc_s = line_centers / 10 # AA --> nm
        _width = l_widths / 10 # AA --> nm

        filters = [
            BoxcarFilter(
                f'lc{lc_s[i]//1000}', lc_s[i]-_width[i], lc_s[i]+_width[i]
                ) for i in range(len(lc_s))
                ]
        
        # This decide how many PHX intensity spectra (diff temp, logg, etc) to be dw
        # I cannot find a way to only dw one spectra in ldtk pkg as it is a 
        #    Bayesian base code. Thus, here are min std par that make sure ldtk works.
        sc = LDPSetCreator(
            teff=(temperature,1), logg=(self.logg,0.01), z=(self.FeH,0.01),
            filters=filters, dataset='visir-lowres')

        ps = sc.create_profiles(nsamples=1000)
        qc, _ = ps.coeffs_qd()
        LD_coeff = {
            f"w{int(line_centers[i] // 100)}": qc[i]
            for i in range(len(line_centers))
            }
        
        if 15631.35 in line_centers:
            print('Find FeOH line center...')
            _loc = np.where(line_centers==15631.35)[0][0]
            LD_coeff['EWR_FeOH'] = qc[_loc]
        return LD_coeff


    def find_fluxspec_flatten(
            self, spec_type, Vel_bin_size, plot, syn_spec_path=None):
        
        self.spec_type = spec_type
        self.out_path = f'{os.getcwd()}/syn_spec'
        temp_list = [self.Temp, self.Tspot, self.Tfaculae]

        print('\nChecking flatten spectra templates for')
        print(f'   {temp_list} K\n')
        
        self.dw_phx_flux(temp_list, alpha=0)

        wv_regions = list(self.bands2use.keys())
        for idx, band in enumerate(wv_regions):

            line_cen = self.bands2use[band][0]
            delta_wv = self.bands2use[band][1]
            wvmin, wvmax = (line_cen-delta_wv, line_cen+delta_wv)
            wvmin = int(wvmin); wvmax = int(wvmax)
            
            for tt, nn in zip(temp_list, ['star', 'spot', 'plage']):
                # ---------------------- Read PHX SYN Spectra ----------------------
                if spec_type.lower() == 'phx':
                    
                    ## --- flatten syn spectra ---
                    # 1. flatten PHX spectra at a larger wv range (3000 - 30000 AA)
                    flat_spec_dir = ssp.flatten_spec(
                        syn_spec_path, self.out_path, self.phx_flux_file[tt])
                    
                    # sim_use_spec = f'SimUse_PHX_ACES_{tt}_{self.logg}_{self.FeH}_wv{wvmin}_{wvmax}_flatten.h5'
                    sim_use_spec = f'SimUse_PHX_ACES_{tt}_{self.logg}_{self.FeH}_{band}_flatten.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_use_spec}'
                    
                    # 2. cut the flattened spectra to the region of interest and transfer wavelength to velocity
                    ssp.final_flux_spec(
                        spec_type, flat_spec_dir, sim_spec_path, wvmin, wvmax, 
                        Vel_bin_size, plot, flatten=True)
                    flatten_spec = pd.read_hdf(sim_spec_path)
                
                # ---------------------- Read Synmast Spectra ----------------------
                elif spec_type.lower() == 'synmast':
                    prf_dir = syn_spec_path
                    
                    # mc_2700350+000_b2_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf
                    spec_base_name = f'mc_{tt}{self.logg*100:1.0f}+{self.FeH*100:03.0f}_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf'
                    unflatten_spec_dir = f'{prf_dir}/{spec_base_name}'

                    # Check if the prf file already exists
                    if not os.path.exists(unflatten_spec_dir):
                        print(f'   Don\'t have {unflatten_spec_dir}.')
                        sys.exit('   Please get the spectra from synmast and put it in the syn_spec folder!')

                    sim_use_spec = f'SimUse_Synmast_{tt}_{self.logg}_{self.FeH}_{band}_flatten.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_use_spec}'
                    
                    ssp.final_flux_spec(
                        spec_type, unflatten_spec_dir, sim_spec_path, wvmin, wvmax, 
                        Vel_bin_size, plot, flatten=True)
                    flatten_spec = pd.read_hdf(sim_spec_path) 

                else:
                    print('\n[ERROR] Can only handel PHX and Synmast spec!\n')
                
                # sum total flux for intensity ratio calculation
                # sum_up_flux = self.flux_sum(spec_base_path, tt, wvmin, wvmax)
                # sum_up_flux_box.append(sum_up_flux)

                spec = {
                    f'{band}_v': flatten_spec['v'].to_numpy(), 
                    f'{band}_f': flatten_spec['f'].to_numpy()
                    }
            
                if idx==0:
                    if nn=='star':
                        self.star_spec_flatten = spec
                    elif nn=='spot':
                        self.spot_spec_flatten = spec
                    elif nn=='plage':
                        self.plage_spec_flatten = spec
                    elif nn=='ccf_temp':
                        self.ccf_spec_flatten = spec
                else:
                    if nn=='star':
                        self.star_spec_flatten.update(spec)
                    elif nn=='spot':
                        self.spot_spec_flatten.update(spec)
                    elif nn=='plage':
                        self.plage_spec_flatten.update(spec)
                    elif nn=='ccf_temp':
                        self.ccf_spec_flatten.update(spec)
            
            # self.intensity_ratio[band] = sum_up_flux_box[1]/sum_up_flux_box[0]
            # print(f'\n--> Intensity rato for {band}: {sum_up_flux_box[1]/sum_up_flux_box[0]:1.2f}')
        
    def find_fluxspec(self, spec_type, Vel_bin_size, plot, syn_spec_path=None):
        
        self.spec_type = spec_type          
        temp_list = [self.Temp, self.Tspot, self.Tfaculae]

        print('\nChecking spectra templates for')
        print(f'   {temp_list} K\n')

        wv_regions = list(self.bands2use.keys())
        for idx, band in enumerate(wv_regions):

            line_cen = self.bands2use[band][0]
            delta_wv = self.bands2use[band][1]
            wvmin, wvmax = (line_cen-delta_wv, line_cen+delta_wv)
            wvmin = int(wvmin); wvmax = int(wvmax)

            for tt, nn in zip(temp_list, ['star', 'spot', 'plage']):
                # ---------------------- Read PHX SYN Spectra ----------------------
                if spec_type.lower() == 'phx':
                    
                    sim_use_spec = f'SimUse_PHX_ACES_{tt}_{self.logg}_{self.FeH}_{band}.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_use_spec}'
                    unflattened_spec_dir = f'{self.out_path}/RASSINE_PHX/{self.phx_flux_file[tt]}'
                    ssp.final_flux_spec(
                        spec_type, unflattened_spec_dir, sim_spec_path, wvmin, wvmax, 
                        Vel_bin_size, plot, flatten=False)
                    final_spec = pd.read_hdf(sim_spec_path)

                # ---------------------- Read SYNmast Spectra ----------------------
                elif spec_type.lower() == 'synmast':
                    prf_dir = syn_spec_path
                    
                    # mc_2700350+000_b2_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf
                    spec_base_name = f'mc_{tt}{self.logg*100:1.0f}+{self.FeH*100:03.0f}_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf'
                    unflatten_spec_dir = f'{prf_dir}/{spec_base_name}'

                    # Check if the prf file already exists
                    if not os.path.exists(unflatten_spec_dir):
                        print(f'   Don\'t have {unflatten_spec_dir}.')
                        sys.exit('   Please get the spectra from synmast and put it in the syn_spec folder!')

                    sim_spec_name = f'SimUse_Synmast_{tt}_{self.logg}_{self.FeH}_{band}.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_spec_name}'
                    
                    ssp.final_flux_spec(
                        spec_type, unflatten_spec_dir, sim_spec_path, wvmin, wvmax, 
                        Vel_bin_size, plot, flatten=False)
                    final_spec = pd.read_hdf(sim_spec_path) 

                else:
                    print('\n[ERROR] Can only handel PHX and Synmast spec!\n')
                
                # # sum total flux for intensity ratio calculation
                # sum_up_flux = self.flux_sum(spec_base_path, tt, wvmin, wvmax)
                # sum_up_flux_box.append(sum_up_flux)

                spec = {
                    f'{band}_v': final_spec['v'].to_numpy(), 
                    f'{band}_f': final_spec['f'].to_numpy()
                    }
            
                if idx==0:
                    if nn=='star':
                        self.star_spec = spec
                    elif nn=='spot':
                        self.spot_spec = spec
                    elif nn=='plage':
                        self.plage_spec = spec
                else:
                    if nn=='star':
                        self.star_spec.update(spec)
                    elif nn=='spot':
                        self.spot_spec.update(spec)
                    elif nn=='plage':
                        self.plage_spec.update(spec)
            
            # self.intensity_ratio[band] = 1
            # print(f'\n--> Intensity rato for {band}: {sum_up_flux_box[1]/sum_up_flux_box[0]:1.2f}')

    def find_intensityspec(self, spec_type, Vel_bin_size, plot, syn_spec_path=None):
        
        self.spec_type = spec_type            
        temp_list = [self.Temp, self.Tspot, self.Tfaculae]

        print('\nChecking spectra templates for')
        print(f'   {temp_list} K\n')

        wv_regions = list(self.bands2use.keys())
        for idx, band in enumerate(wv_regions):

            line_cen = self.bands2use[band][0]
            delta_wv = self.bands2use[band][1]
            wvmin, wvmax = (line_cen-delta_wv, line_cen+delta_wv)
            wvmin = int(wvmin); wvmax = int(wvmax)

            for tt, nn in zip(temp_list, ['star', 'spot', 'plage']):
                # ---------------------- Read PHX SYN Spectra ----------------------
                if spec_type.lower() == 'phx':
                    raise NotImplementedError(
                        'Intensity spectra calculation is not implemented yet for PHX.'
                        'Please use the *synmast* spectra instead.'
                    )

                    # self.dw_phx_mu_intensity(temp_list, alpha=0)
                    
                    # sim_spec_name = f'SimUse_PHX_ACES_SpecInt_{tt}_{self.logg}_{self.FeH}_wv{wvmin}_{wvmax}_NOTflatten.p'
                    # sim_spec_path = f'{self.out_path}{sim_spec_name}'
                    
                    # unflatten_spec_dir = spec_base_path.format(tt, self.logg, self.FeH)
                    
                    # ssp.final_spec(
                    #     spec_type, unflatten_spec_dir, sim_spec_path, wvmin, wvmax, 
                    #     Vel_bin_size, flatten=False)
                    # final_spec = pd.read_pickle(sim_spec_path)

                # ---------------------- Read SYNmast Spectra ----------------------
                elif spec_type.lower() == 'synmast':
                    prf_dir = syn_spec_path
                    
                    # mc_2700350+000_b2_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf
                    spec_base_name = f'mc_{tt}{self.logg*100:1.0f}+{self.FeH*100:03.0f}_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf'
                    unflatten_spec_dir = f'{prf_dir}/{spec_base_name}'

                    # Check if the prf file already exists
                    if not os.path.exists(unflatten_spec_dir):
                        print(f'   Don\'t have {unflatten_spec_dir}.')
                        sys.exit('   Please get the spectra from synmast and put it in the syn_spec folder!')

                    sim_spec_name = f'SimUse_Synmast_{tt}_{self.logg}_{self.FeH}_{band}.h5'
                    sim_spec_path = f'{self.out_path}/SimUse/{sim_spec_name}'
                    
                    ssp.final_intensity_spec(
                        spec_type, unflatten_spec_dir, sim_spec_path, wvmin, wvmax, 
                        Vel_bin_size, plot, flatten=False)
                    final_spec = pd.read_hdf(sim_spec_path) 

                else:
                    print('\n[ERROR] Can only handel PHX and Synmast spec!\n')
                
                # # sum total flux for intensity ratio calculation
                # sum_up_flux = self.flux_sum(spec_base_path, tt, wvmin, wvmax)
                # sum_up_flux_box.append(sum_up_flux)

                # mu_s = np.array([0.9636, 0.8864, 0.8018, 0.7071, 0.5976, 0.4629, 0.2673])
                mu_cols = [col for col in final_spec.columns if col.startswith("mu_")]
                df_mu = final_spec[mu_cols]
                spec = {
                    f'{band}_v': final_spec['v'].to_numpy(), 
                    f'{band}_f': df_mu.to_numpy(),
                    }
            
                if idx==0:
                    if nn=='star':
                        self.star_spec = spec
                    elif nn=='spot':
                        self.spot_spec = spec
                    elif nn=='plage':
                        self.plage_spec = spec
                else:
                    if nn=='star':
                        self.star_spec.update(spec)
                    elif nn=='spot':
                        self.spot_spec.update(spec)
                    elif nn=='plage':
                        self.plage_spec.update(spec)
            
            # self.intensity_ratio[band] = 1
            # print(f'\n--> Intensity rato for {band}: {sum_up_flux_box[1]/sum_up_flux_box[0]:1.2f}')


    def dw_phx_flux(self, temp_list, alpha=0):
        self.phx_flux_file = {}
        for tt, nn in zip(temp_list, ['star', 'spot', 'plage']):
            if (tt > 7000) | (tt < 2300) | (tt is None):
        
                raise Exception(
                    f'   [ERROR] Rassine normalization cfg not tuned for {tt}K\n'
                    '   Please check the temperature range in the config file!'
                )
            
            else:
                phx_flux_file = ssp.get_phx_flux_spec(
                    tt, self.logg, self.FeH, alpha, self.phx_path, self.out_path)
                self.phx_flux_file[tt] = phx_flux_file
                
    def dw_phx_mu_intensity(self, temp_list, alpha=0):
        for tt, nn in zip(temp_list, ['star', 'spot', 'plage']):
            if (tt > 7000) | (tt < 2300) | (tt is None):
        
                raise Exception(
                    f'   [ERROR] Rassine normalization cfg not tuned for {tt}K\n'
                    '   Please check the temperature range in the config file!'
                )
            else:
                ssp.get_phx_intensity_spec(
                    tt, self.logg, self.FeH, alpha, self.phx_path, self.out_path)



    # def flux_sum(self, _spec_dir, tt, wvmin, wvmax): 
    #     spec_dir = _spec_dir.format(tt, self.logg, self.FeH)

    #     spec = pd.read_pickle(spec_dir)
    #     wave = spec['wave'].to_numpy()
    #     flux = spec['flux'].to_numpy()

    #     flux = flux[(wave>=wvmin) & (wave<=wvmax)]
    #     wave = wave[(wave>=wvmin) & (wave<=wvmax)]
    #     # integrate the flux

    #     # print(f'\n   ----> {tt}K wv  {wvmin}->{wvmax}: {np.trapz(flux, wave)}')
    #     return np.trapz(flux, wave)

        # if self.Temp in teffs:
        #     self.star_teff_use = self.Temp
        #     self.star_spec = self.read_spec(self.star_teff_use, self.logg)
        #     print(f'\n Using {self.star_teff_use}K template for the stellar photosphere')
        # else:
        #     abs_differ = np.abs(self.Temp - teffs)
        #     _temptab = Table(
        #         data=(teffs, abs_differ), 
        #         names=['teffs', 'abs_differ']
        #         )
        #     _temptab.sort('abs_differ')
        #     temp1 = _temptab['teffs'][0]; diff1 = _temptab['abs_differ'][0]
        #     temp2 = _temptab['teffs'][1]; diff2 = _temptab['abs_differ'][1]

        #     ratio1 = 1 - (diff1/(diff1+diff2))
        #     ratio2 = 1 - (diff2/(diff1+diff2))

        #     spec1 = self.read_spec(temp1, self.logg)
        #     spec2 = self.read_spec(temp2, self.logg)

        #     spec = {'v': spec1['v'], 'f': ratio1*spec1['f']+ratio2*spec2['f']}
        #     self.star_spec = spec

        #     print(f'\nUsing {ratio1:1.1%} of {temp1}K template and {ratio2:1.1%} of {temp2}K template for the stellar photosphere')

        # if self.Temp_spot in teffs:
            
        #     self.spot_teff_use = self.Temp_spot
        #     self.spot_spec = self.read_spec(self.spot_teff_use, self.logg)
        #     print(f'Using {self.spot_teff_use}K template for the spot spectrum')
        # else:
        #     abs_differ = np.abs(self.Temp_spot - teffs)
        #     _temptab = Table(
        #         data=(teffs, abs_differ), 
        #         names=['teffs', 'abs_differ']
        #         )
        #     _temptab.sort('abs_differ')
        #     temp1 = _temptab['teffs'][0]; diff1 = _temptab['abs_differ'][0]
        #     temp2 = _temptab['teffs'][1]; diff2 = _temptab['abs_differ'][1]

        #     ratio1 = 1 - (diff1/(diff1+diff2))
        #     ratio2 = 1 - (diff2/(diff1+diff2))

        #     spec1 = self.read_spec(temp1, self.logg)
        #     spec2 = self.read_spec(temp2, self.logg)

        #     spec = {'v': spec1['v'], 'f': ratio1*spec1['f']+ratio2*spec2['f']}
        #     self.spot_spec = spec

        #     print(f'Using {ratio1:1.1%} of {temp1}K template and {ratio2:1.1%} of {temp2}K template for the spot spectrum')

        # # find the two closest temp to use for interpolate 
        # star_teff_use_loc = np.abs(self.Temp - teffs)
        # spot_teff_use_loc = np.argmin(
        #     np.abs((self.Temp-self.Temp_diff_spot) - teffs)
        #     )
        # self.star_teff_use = teffs[star_teff_use_loc]
        # self.spot_teff_use = teffs[spot_teff_use_loc]

        # self.star_spec = self.read_spec(self.star_teff_use)
        # self.spot_spec = self.read_spec(self.spot_teff_use)

        # print(
        #     '\n Using {}K template for the STAR & {}K template for the SPOT!\n'.format(
        #         self.star_teff_use, self.spot_teff_use)
        #         )

    # def read_spec(self, teff, logg):
    #     file_star = f'./syn_spec/{self.spec_type}_for_SOAP_LineDepthRatio_{teff}_{logg}_0.0.csv'
    #     t = Table.read(file_star, format='csv')

    #     t_dict = {}
    #     # t_dict['w'] = np.array(t['wave[AA]'])
    #     t_dict['v'] = np.array(t['vel[kms]'])
    #     t_dict['f'] = np.array(t['flux'])
    #     return t_dict



class Active_region:
    def __init__(self, long, lat, size, active_region_type, check=False):
        """initiate active regions

        Args:
            long (float): Longitude [deg]
            lat (float): Latitude [deg]
            size (float): spot radius [Rstar]
            active_region_type (int): # 0 if spot, 1 if plage
            check (bool, optional): # 1 if the active region is considered, 
                0 if no active region. Defaults to False.

        Notes:
        For size: Active region's size [Rstar]. 
            If you want a a spot coverage S1 (%) in area of the visible hemisphere, 
            you have to put here sqrt(2*S1), because
            S1 = Area_spot/Area_visible_hemisphere 
               = pi*(size1*Rstar)**2/(2*pi*Rstar**2) 
               = size1**2/2 
            -> size1 = sqrt(2*S1).
            For 0.1% of the visible hemisphere, put 0.045

        """
        self.long = long    
        self.lat  = lat     
        self.s    = size    
        self.active_region_type = active_region_type 
        if self.active_region_type == 0:
            self.active_region_type_str = 'spot' #active region type in string
        else:
            self.active_region_type_str = 'plage'  #active region type in string
        self.check = check                           

    def calc_maps(self, star, grid, nrho):
        """calculate the visibility of the active regions

        Args:
            star (class): star basic info
            grid (int): Grid resolution NxN 
            nrho (int): Resolution for the spot's circumference 
        """
        # Calculates the initial coordinates of the active region circumference
        xyz = stsp.spot_init(self.s, self.long, self.lat, star.incl, nrho) 
        self.xyz = xyz
        
        # Calculates the coordinates of the active region circumference at phase psi
        xyz2 = stsp.spot_phase(xyz, star.incl, nrho, star.psi) 
        self.xyz2 = xyz2

        # Check is the active region is visible at phase psi
        # and return the min and max y and z values for the circumference
        (self.visible, self.iminy, self.iminz, self.imaxy, self.imaxz) =\
            stsp.spot_area(xyz2, nrho, grid) 


class Ccf:
    def __init__(self, rv, intensity, width, step, star):
        """_summary_

        Args:
            rv (float): Radial velocity of each point of the CCF
            intensity (float): Intensity of each point of the CCF
            width (float): Width of the CCF
            step (float): Sampling of the CCF
            star (class): Star class
        """
        self.rv         = rv              
        self.intensity  = intensity       
        self.width      = width           
        self.step       = step            
        self.n          = len(self.rv)   # Number of points of the CCF

        # The CCF has a width of self.width. This CCF has a rotational velocity of 0 because taken in the stellar disk 
        # center. To consider rotation when simulating the emerging spectrum on the limb of the star 
        # (velocity different from 0), We have to extrapolate outside of the range [-self.width;self.width]. This is 
        # done by calculating self.n_v which is the number of point of the CCF required to consider a stellar rotation 
        # of star.vrot. 
        # self.n_v must by an odd integer so that we have as many values on the positive side of the CCF than on the 
        # negative one (because 0 is present).
        
        if np.round(((1.1*star.vrot)/self.step)*2)%2 == 0:
            self.n_v = int(self.n + round(((1.1*star.vrot)/self.step)*2))
        else:
            self.n_v = int(self.n + round(((1.1*star.vrot)/self.step)*2)+1)
        
        # self.v_interval gives the difference in radial velocity between the minimum and maximum values of the CCF, once 
        # the extrapolation is done. 
        # self.n_v gives the number of points of the extrapolated CCF. (self.n_v-1) gives the number of intervals, which 
        # is then multiplied by the sampling of the CCF self.step to give self.v_interval
        self.v_interval = self.step*(self.n_v-1) / 2.

class Spec:
    # def __init__(self, rv, intensity, width, step, star):
    def __init__(self, type, star, band):
        """_summary_

        Args:
            rv (float): Radial velocity of each point of the CCF
            intensity (float): Intensity of each point of the CCF
            width (float): Width of the CCF
            step (float): Sampling of the CCF
            star (class): Star class
        """

        if type == 'photosphere':
            self.rv         = star.star_spec[f'{band}_v']             
            self.intensity  = star.star_spec[f'{band}_f']        
            self.width      = np.max(self.rv)           
            self.step       = self.rv[1]-self.rv[0]            
            self.n          = len(self.rv)   # Number of points of the CCF
        elif type == 'spot':
            self.rv         = star.spot_spec[f'{band}_v']             
            self.intensity  = star.spot_spec[f'{band}_f']        
            self.width      = np.max(self.rv)           
            self.step       = self.rv[1]-self.rv[0]            
            self.n          = len(self.rv)
        elif type == 'plage':
            self.rv         = star.plage_spec[f'{band}_v']
            self.intensity  = star.plage_spec[f'{band}_f']
            self.width      = np.max(self.rv)
            self.step       = self.rv[1]-self.rv[0]
            self.n          = len(self.rv)
        else:
            sys.exit('The type of spectrum is not supported yet!')

        # The CCF has a width of self.width. This CCF has a rotational 
        # velocity of 0 because taken in the stellar disk center. 
        # To consider rotation when simulating the emerging spectrum on 
        # the limb of the star (velocity different from 0). 
        # We have to extrapolate outside of the range [-self.width;self.width]. 
        # This is done by calculating self.n_v which is the number of point 
        # of the CCF required to consider a stellar rotation of star.vrot. 
        # self.n_v must by an odd integer so that we have as many values on 
        # the positive side of the CCF than on the 
        # negative one (because 0 is present).
        
        if np.round(((1.1*star.vrot)/self.step)*2)%2 == 0:
            self.n_v = int(self.n + round(((1.1*star.vrot)/self.step)*2))
        else:
            self.n_v = int(self.n + round(((1.1*star.vrot)/self.step)*2)+1)
        
        # self.v_interval gives the difference in radial velocity between the
        # minimum and maximum values of the CCF, once the extrapolation is done. 
        # self.n_v gives the number of points of the extrapolated CCF. 
        # (self.n_v-1) gives the number of intervals, which is then 
        # multiplied by the sampling of the CCF self.step to give self.v_interval
        self.v_interval = self.step*(self.n_v-1) / 2.

def compute_bis(RV, CCF):
    err = np.ones(len(CCF), dtype=np.float64)
    (mod,c,k,v0,fwhm,sig_c,sig_k,sig_v0,sig_fwhm,span,bis,depth) =\
        stsp.gauss_bis(RV, CCF, err)
    depth=-k
    return depth, bis, span, v0, fwhm


def detect_peaks(
        x, mph=None, mpd=1, threshold=0, edge='rising',
        kpsh=False, valley=False, show=False, ax=None):


    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
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
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        # _plot(x, mph, mpd, threshold, edge, valley, ax, ind)
    return ind

def a0cont(a0vcut, plot=False, poly_order=1, spline_k=1, mpd=100):
    '''
    Performs first-pass fit to blaze shape of telluric standard spectrum.

    Inputs:
    a0wavecut : Wavelength scale of telluric standard spectrum
    a0vcut    : Corresponding flux of telluric standard spectrum
    night     : Date of observation in YYYYMMDD
    order     : Echelle order, as characterized by file index (as opposed to m 
                number; for conversion between the two, see Stahl et al. 2021)
    band      : H or K band

    Outputs:
    continuum : Best fit blaze model
    '''
    x = np.arange(len(a0vcut))

    # mpd: detect peaks that are at least separated by minimum peak distance
    peaks = detect_peaks(a0vcut, mpd=mpd)
    xtimes = 1

    for _ in range(xtimes):
        mask = np.ones(len(peaks), dtype=bool)

        f = np.polyfit(x[peaks], a0vcut[peaks], poly_order)
        q = np.poly1d(f)
        residual = a0vcut[peaks] - q(x[peaks])
        MAD = np.median(np.abs(residual - np.median(residual)))

        mask[(a0vcut[peaks]/np.nanmedian(a0vcut) < .1)] = False
        mask[(a0vcut[peaks] < q(x[peaks])-2*MAD)] = False
        mask[(a0vcut[peaks] > q(x[peaks])+6*MAD)] = False
        peaks = peaks[mask]

    c = 0
    for smoothing in np.arange(1e6, 1e8, 1e6):
        f = interpolate.UnivariateSpline(
            x[peaks], a0vcut[peaks], k=spline_k, s=smoothing)
        continuum = f(x)
        peaks2 = detect_peaks(continuum)
        if len(peaks2) == 1:
            c += 1
        if c == 2:
            break

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

    # if plot:
    #     plt.figure(figsize=(6,2.5), dpi=250)
    #     plt.plot(x,a0vcut,color='black',alpha=.5)
    #     plt.scatter(x[peaks],a0vcut[peaks],s=25,color='blue')
    #     plt.plot(x,f(x),color='orange',alpha=.5)

    return continuum, peaks

def flatten_ccf_model_spec(vel, flux, band, RVandEWR_dir, plot):
    # plot = True

    _mpd = int(len(vel)*0.04)
    cont, peaks = a0cont(
        flux, plot=False, poly_order=2, spline_k=2, mpd=_mpd)
    fluxp_flat = flux / cont
    fluxp_flat = fluxp_flat/np.nanpercentile(fluxp_flat, 99)

    ############################
    if plot == 1:
        fig, ax = plt.subplots(
                1, 1, figsize=(7,3), facecolor='white', dpi=250)
        
        ax.plot(vel, flux, '-k', lw=1, label='Org. model spec', alpha=0.6)
        ax.plot(
            vel, fluxp_flat, '-', c='tab:red', lw=0.8, 
            label='Faltten model spec')
        ax.plot(
            vel[peaks], flux[peaks], 'o', c='tab:blue', ms=4, zorder=5, 
            label='peaks')
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
        ############################
        
    return np.array(fluxp_flat)

def flatten_model_spec(Star, flux_phi, visible_phi, band, RVandEWR_dir, plot):
    # plot = True
    vel = Star.star_spec[f'{band}_v']

    nomral_spec_box = []
    for i in range(len(visible_phi)):
        flux = flux_phi[i]

        _mpd = int(len(vel)*0.04)
        cont, peaks = a0cont(
            flux, plot=False, poly_order=2, spline_k=2, mpd=_mpd)
        fluxp_flat = flux / cont
        fluxp_flat = fluxp_flat/np.nanpercentile(fluxp_flat, 99)

        ############################
        if plot == 1:
            fig, ax = plt.subplots(
                    1, 1, figsize=(7,3), facecolor='white', dpi=250)
            
            ax.plot(vel, flux, '-k', lw=1, label='Org. model spec', alpha=0.6)
            ax.plot(
                vel, fluxp_flat, '-', c='tab:red', lw=0.8, 
                label='Faltten model spec')
            ax.plot(
                vel[peaks], flux[peaks], 'o', c='tab:blue', ms=4, zorder=5, 
                label='peaks')
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
        ############################
        
        nomral_spec_box.append(fluxp_flat)
    return np.array(nomral_spec_box)

def get_equivalent_width(Star, flux_phi, visible_phi, band):

    ew_fe1 = []; ew_fe2 = []; ew_oh0 = []

    ew_cal_regions = [
        [1.56230, 1.56290], 
        [1.56290, 1.56335], 
        [1.56335, 1.56395]]

    band_center_wv = Star.bands2use[band][0] # [15631.35, 30, ['rv', 'photo']]
    vel = Star.star_spec[f'{band}_v']

    c_kms = 299792.458

    # -- vel back to wv --
    del_wv = vel * band_center_wv / c_kms
    wave = (band_center_wv + del_wv) / 1e4
    nor_region = np.where((wave>1.5610) & (wave<1.5650))[0]
    
    for i in range(len(visible_phi)):
        flux = flux_phi[i]
        nor_level = np.percentile(flux[nor_region], 95)
        flux = flux / nor_level
        # --- EW ---
        for r_udx, rr in enumerate(ew_cal_regions):

            use_r = (wave>=rr[0]) & (wave<=rr[1])
            spec = Spectrum1D(
                spectral_axis=wave[use_r]*u.um, 
                flux=flux[use_r]*u.Unit('erg cm-2 s-1 AA-1') 
                )
            ew = equivalent_width(spec)
            # print(ew)

            if r_udx == 0:
                ew_fe1.append(ew.value)
            elif r_udx == 1:
                ew_oh0.append(ew.value)
            elif r_udx == 2:
                ew_fe2.append(ew.value)

    oh_line = np.array(ew_oh0)
    fe_line = (np.array(ew_fe1) + np.array(ew_fe2)) / 2
    line_ratio = oh_line/fe_line

    return line_ratio

def get_line_depth(Star, CCFstar_tot, visible_phi, band):

    depth_d_fe1 = []
    depth_d_fe2 = []
    depth_d_oh0 = []
    depth_l_fe1 = []
    depth_l_fe2 = []
    depth_l_oh0 = []

    vel = Star.star_spec[f'{band}_v']

    for i in range(len(visible_phi)):
        
        CCFstar_tot_flip = 1+ -1*CCFstar_tot[i]

        fe_1 = (vel>-150) & (vel<-50)
        oh_0 = (vel>-50) & (vel<50)
        fe_2 = (vel>50) & (vel<250)

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
    nrho, grid, star, psi, 
    Spec_photo, Spec_spot, Spec_plage,
    active_regions_number, active_region_box,
    inst_reso, cpu_use, int_ratio, RVandEWR_dir, band, 
    savespec, fluxORint
    ):
    """

    Args:
        nrho (int): resolution for the spot's circumference
        grid (int): model grid resolution NxN
        star (class): input star pars
        psi (float): initial phase
        spec (class): quite model spec 
        spec_active_region (class): active model spec 
        ccf (class): quite model ccf 
        ccf_active_region (class): active model ccf 
        active_regions (class): input info. on the spot(s)
        inst_reso (float): instrumental resolution (R)
        cpu_use (int): cpu to use while multiprocessing

    Returns:
        _type_: _description_
    """

    # initiate the quite surface
    Spec_photo.intensity = np.array(
        Spec_photo.intensity, dtype=np.float64)

    if fluxORint == 'flux':
        SPECstar_quiet, quite_surface = stsp.itot_flux(
            star.vrot, star.incl, star.limba1[0], star.limba2[0], 0., 0., 0., grid, 
            Spec_photo.rv, Spec_photo.intensity, Spec_photo.v_interval, 
            Spec_photo.n_v, Spec_photo.n
            )
    if fluxORint == 'int':
        SPECstar_quiet, quite_surface = stsp.itot_int(
            star.vrot, star.incl, star.limba1[0], star.limba2[0], 0., 0., 0., grid, 
            Spec_photo.rv, Spec_photo.intensity, Spec_photo.v_interval, 
            Spec_photo.n_v, Spec_photo.n
            )
    quite_surface_2d = quite_surface.copy()
    # initialization of the variables
    # FLUXstar      = FLUXstar_quiet
    # CCFstar_flux  = CCFstar_quiet
    # CCFstar_bconv = CCFstar_quiet
    # CCFstar_tot   = CCFstar_quiet
    SPECstar_tot  = SPECstar_quiet

    (active_region_checks, active_region_longs, active_region_lats, 
        active_region_sizes, active_region_types) = active_region_box
    
    #Loop over the different active regions
    print('\nCalculate Activate Regions...')
    # for rr, active_region in enumerate(np.arange(active_regions_number)):
    for rr in range(active_regions_number):
        # active_region.check = 1 if the active region is activated, 0 otherwise
        
        if not active_region_checks[rr]: 
            continue 
            
        print(f'\nCalculate Active Region {rr+1}')
        # Calculates the position of the spot initialized at the disc center
        xyz = stsp.spot_init(
            active_region_sizes[rr], active_region_longs[rr], active_region_lats[rr], 
            star.incl, nrho
            )
        
        active_intensity = Spec_spot.intensity if active_region_types[rr] == 0 else Spec_plage.intensity
        limba1 = np.array([star.limba1[0], star.limba1[1]]) if active_region_types[rr] == 0 else np.array([star.limba1[0], star.limba1[2]])
        limba2 = np.array([star.limba2[0], star.limba2[1]]) if active_region_types[rr] == 0 else np.array([star.limba2[0], star.limba2[2]])

        # print(active_region_sizes[rr], active_region_longs[rr], active_region_lats[rr])

        # Scans the yz-area where the spot is for different phases (psi) and
        # returns the spot's "non-contribution" to the total flux and its
        # "non-contribution" to the ccf, for each phase.
        # Thus the result is to be subtracted to the output of the itot() function.
        SPECactive_region_tot, SPECphoto_region_duplicated, visible_phi, active_surface_box, quite_surface_duplicated_box =\
                stsp.spot_scan_npsi(
                    xyz, nrho, psi, len(psi), star.vrot, star.incl, limba1, 
                    limba2, 0., 0., 0., grid, 
                    Spec_photo.rv, Spec_photo.intensity, active_intensity, 
                    Spec_photo.v_interval, Spec_photo.n_v, Spec_photo.n, 
                    active_region_sizes[rr], active_region_longs[rr], active_region_lats[rr], 
                    active_region_types[rr], star.Temp, star.Tdiff_spot, cpu_use, int_ratio, 
                    fluxORint
                    )

        # Calculate the total flux of the star affected by active regions
        # FLUXstar      = FLUXstar-np.array(FLUXactive_region)
        
        # Calculate the CCF of the star affected by the flux effect of active regions
        # CCFstar_flux  = CCFstar_flux-np.array(CCFactive_region_flux)   
        
        # Calculate the CCF of the star affected by the convective blueshift effect of active regions
        # CCFstar_bconv = CCFstar_bconv-np.array(CCFactive_region_bconv) 
        
        # Calculate the CCF of the star affected by the total effect of active regions
        # CCFstar_tot   = CCFstar_tot - np.array(CCFactive_region_tot)  

        # sytang old
        # SPECstar_tot   = SPECstar_tot - np.array(SPECactive_region_tot)  
        # quite_surface_2d = quite_surface_2d - np.array(active_surface_box)

        if savespec:
            istart = (Spec_photo.n_v-len(Spec_photo.rv))//2         
            iend   = istart+len(Spec_photo.rv)
            spec_qs = SPECstar_tot[istart:iend]

            spec_qs_duplicated = np.array(SPECphoto_region_duplicated)
            spec_qs_duplicated = spec_qs_duplicated[:,istart:iend]

            spec_ss = np.array(SPECactive_region_tot)
            spec_ss = spec_ss[:,istart:iend]

            # save to .npy
            np.save(f'./outputs/{RVandEWR_dir}/spec/{band}_qs_{rr+1}.npy', spec_qs)
            np.save(f'./outputs/{RVandEWR_dir}/spec/{band}_ss_{rr+1}.npy', spec_ss)
            np.save(f'./outputs/{RVandEWR_dir}/spec/{band}_qsdup_{rr+1}.npy', spec_qs_duplicated)
            
            vel = np.array(Spec_photo.rv)
            np.save(f'./outputs/{RVandEWR_dir}/spec/{band}_vel_{rr+1}.npy', vel)
        
        SPECstar_tot   = SPECstar_tot - np.array(SPECphoto_region_duplicated) + np.array(SPECactive_region_tot)  
        quite_surface_2d = quite_surface_2d - np.array(quite_surface_duplicated_box) + np.array(active_surface_box)


    # Calculate where the extrapolated CCF corresponds to the boundaries of the non-extrapolated CCF.
    # istart = (Ccf.n_v-len(Ccf.rv))//2   # ccf.n_v is odd given our definition, like len(ccf.rv), therefore 
    #                                     # the difference can be divided by 2. The CCF has been extrapolated 
    #                                     # the same way on each boundary, so dividing the difference between 
    #                                     # the extrapolated and non extrapolated CCF by 2 gives by how many 
    #                                     # points the CCF was extrapolated on each side
    # iend   = istart+len(Ccf.rv)

    # truncate the extrapolated rotating CCF to the same interval as the non-rotating CCF
    # CCFstar_flux  = CCFstar_flux[:,istart:iend]
    # CCFstar_bconv = CCFstar_bconv[:,istart:iend]
    # CCFstar_tot   = CCFstar_tot[:,istart:iend]
    # CCFstar_quiet = CCFstar_quiet[istart:iend]

    # -- Spec ---
    istart = (Spec_photo.n_v-len(Spec_photo.rv))//2         
    iend   = istart+len(Spec_photo.rv)
    SPECstar_tot   = SPECstar_tot[:,istart:iend]
    # SPECstar_quiet = SPECstar_quiet[istart:iend]


    # Normalization, loop through diff phases
    # CCFstar_flux  = np.array([f/np.max(f) for f in CCFstar_flux], dtype=np.float64)
    # CCFstar_bconv = np.array([f/np.max(f) for f in CCFstar_bconv], dtype=np.float64)
    # CCFstar_tot   = np.array([f/np.percentile(f,95) for f in CCFstar_tot], dtype=np.float64)
    # CCFstar_quiet = CCFstar_quiet/np.percentile(CCFstar_quiet, 95)

    
    SPECstar_tot = np.array([f/np.percentile(f,99) for f in SPECstar_tot], dtype=np.float64)
    # SPECstar_quiet = SPECstar_quiet/np.percentile(SPECstar_quiet, 95)

    # Convolution with a Gaussian instrumental profile of a given resolution, 
    # given by the "instrument_reso" variable in the "config.cfg" file
    if inst_reso != 0:

        # CCFstar_tot, CCFstar_quiet = _inst_broaden(inst_reso, Ccf.rv, CCFstar_tot, CCFstar_quiet)
        SPECstar_tot = _inst_broaden(inst_reso, Spec_photo.rv, SPECstar_tot)
    
    return (visible_phi, SPECstar_tot, quite_surface, quite_surface_2d)


def _inst_broaden(inst_reso, rv, star_tot):
    c = 299792458. # speed of light in m/s
    inst_resolution         = inst_reso # resolution R = lambda / Delta(lambda)
    inst_inst_profile_FWHM  = c/inst_resolution/1000. # Resolution = c/Delta_v -> Delta_v = c/R
    inst_inst_profile_sigma = inst_inst_profile_FWHM/(2*np.sqrt(2*np.log(2)))
    Gaussian_low_reso        = np.exp(-rv**2/(2*(inst_inst_profile_sigma)**2))
    
    # star_quiet_tmp = signal.convolve(
    #     -star_quiet+1, Gaussian_low_reso, 'same'
    #     )
    # star_quiet = 1-star_quiet_tmp\
    #     *(1-np.min(star_quiet))/np.max(star_quiet_tmp) # normalization
    
    # Convolution of the CCF with the Gaussian instrumental profile to reduce 
    # the resolution of the CCF
    for i in np.arange(len(star_tot)):
        # CCFstar_flux_tmp  = signal.convolve(-CCFstar_flux[i]+1,Gaussian_low_reso,'same')
        # CCFstar_flux[i]   = 1-CCFstar_flux_tmp*(1-np.min(CCFstar_flux[i]))/np.max(CCFstar_flux_tmp)
        # CCFstar_bconv_tmp = signal.convolve(-CCFstar_bconv[i]+1,Gaussian_low_reso,'same')
        # CCFstar_bconv[i]  = 1-CCFstar_bconv_tmp*(1-np.min(CCFstar_bconv[i]))/np.max(CCFstar_bconv_tmp)
        star_tot_tmp   = signal.convolve(-star_tot[i]+1,Gaussian_low_reso,'same')
        star_tot[i]    = 1-star_tot_tmp*(1-np.min(star_tot[i]))/np.max(star_tot_tmp)

    return star_tot


# # @njit
# def _cc(drvs, CCFstar_tot, vel, tw, tf, visible_phi, ii):
     
#     tf = 1 - tf
#     CCFstar_tot = 1 - CCFstar_tot
#     cc = cc = np.empty(len(drvs))

#     f = CCFstar_tot[ii]
#     # Calculate the cross correlation
#     for i in range(len(drvs)):
#         rv = drvs[i]
#         interp_tf = np.interp(vel, tw+rv, tf)
#         cc[i] = np.sum(f * interp_tf)
    
#     # max_cc = np.max(cc)
#     # cc = max_cc - cc 

#     max_cc = np.max(cc)
#     count_max = 0
#     sum_max = 0.0
#     for i in range(len(drvs)):
#         if cc[i] == max_cc:
#             sum_max += drvs[i]
#             count_max += 1

#     np.save(f'./outputs/CCF/ccf_{visible_phi[ii]:1.3f}', cc)
#     vis.plot_cc(drvs, cc, visible_phi[ii])
#     vis.plot_cc_spec(vel, f, tw, tf, visible_phi[ii], drvs[np.argmax(cc)])
#     ccf_rv_shift = sum_max / count_max
    
#     if -np.abs(ccf_rv_shift) == drvs[0]:
#         ccf_rv_shift = np.nan
#         # vis.plot_cc(drvs, cc, visible_phi[ii])
#         # vis.plot_cc_spec(vel, f, tw, tf, visible_phi[ii], drvs[np.argmax(cc)])
#         print(f'\n     >>>> CCF hit edge for {visible_phi[ii]:1.3f}!')
        
#     return ccf_rv_shift

# @njit
# def _cc(drvs, CCFstar_tot, vel, tw, tf, visible_phi, ii):
     
#     tf = 1 - tf
#     CCFstar_tot = 1 - CCFstar_tot
#     cc = cc = np.empty(len(drvs))

#     f = CCFstar_tot[ii]
#     # Calculate the cross correlation
#     for i in range(len(drvs)):
#         rv = drvs[i]
#         interp_tf = np.interp(vel, tw+rv, tf)
#         cc[i] = np.sum(f * interp_tf)
    
#     # max_cc = np.max(cc)
#     # cc = max_cc - cc 

#     # max_cc = np.max(cc)
#     # count_max = 0
#     # sum_max = 0.0
#     # for i in range(len(drvs)):
#     #     if cc[i] == max_cc:
#     #         sum_max += drvs[i]
#     #         count_max += 1

#     # np.save(f'./outputs/CCF/ccf_{visible_phi[ii]:1.3f}', cc)
#     # vis.plot_cc(drvs, cc, visible_phi[ii])
#     # vis.plot_cc_spec(vel, f, tw, tf, visible_phi[ii], drvs[np.argmax(cc)])
#     # ccf_rv_shift = sum_max / count_max

        
#     return cc

# @njit
# def _cc2(drvs, CCFstar_tot, vel, tw, tf, ii):

#     tf = 1-tf
#     CCFstar_tot = 1 - CCFstar_tot
#     cc = np.empty(len(drvs))
    
#     f = CCFstar_tot[ii]
#     # Calculate the cross correlation
#     for i in range(len(drvs)):
#         rv = drvs[i]
#         cc[i] = np.sum(f * np.interp(vel, tw+rv, tf))
#     cc = np.max(cc) - cc   
    
#     n_min = np.sum( cc == np.min(cc) )
#     if n_min == 1:
#         spot_cc4rv = drvs[np.argmin(cc)]
#     else:
#         drvs_temp = drvs[cc == np.min(cc)]
#         spot_cc4rv = np.mean(drvs_temp)

#     return spot_cc4rv

# # @njit
# def _cc(CCFstar_tot, vel, tw, tf, visible_phi):
#     # the res for the vel is 10m/s, thus moving 
#     # -10km to 10km is moving 1000 pixles
#     # on both sides.
#     # drvs_pixels = np.arange(-400, 401)
#     # drvs = drvs_pixels * 0.025

#     drvs = np.arange(-22, 22, 0.05)
#     # spot_cc4rv = np.zeros(len(CCFstar_tot), dtype=np.float64)
#     # drvs = np.arange(-1, 1, 0.02)
#     spot_cc4rv = np.zeros(len(CCFstar_tot), dtype=np.float64)

#     tf = 1-tf
#     for ii in range(len(CCFstar_tot)):
#         f =  1-CCFstar_tot[ii]

#         # Calculate the cross correlation
#         cc = np.zeros(len(drvs))
#         for i, rv in enumerate(drvs):
#             cc[i] = np.sum(f * np.interp(vel, tw+rv, tf))
#             # tw_shifted = tw+rv
#             # tf_shifted = tf[(tw_shifted >= vel[0]) & (tw_shifted <= vel[-1])]
#             # cc[i] = np.sum(f * tf_shifted )
#         cc = np.max(cc) - cc   

#         vis.plot_cc(drvs, cc, visible_phi[ii])
#         vis.plot_cc_spec(vel, f, tw, tf, visible_phi[ii], drvs[np.argmin(cc)])
        
#         n_min = np.sum( cc == np.min(cc) )
#         if n_min == 1:
#             spot_cc4rv[ii] = drvs[np.argmin(cc)]
#         else:
#             drvs_temp = drvs[cc == np.min(cc)]
#             spot_cc4rv[ii] = np.mean(drvs_temp)

#     return spot_cc4rv

@njit
def _ccf(drvs, mode, w, f, tw, tf, c, weights, meanWl):
    
    # Calculate the cross correlation
    cc = np.zeros(len(drvs), dtype=float)
    df = f - np.nansum(f)/len(f)

    i = 0
    for rv in drvs:
        if mode == "lin":
            # Shift the template linearly
            # fi = sci.interp1d(tw + meanWl * (rv / c), tf)
            # f_shifted = fi(w) * weights
            f_shifted = np.interp(w, tw + meanWl * (rv / c), tf) * weights
            dfs = f_shifted - np.nansum(f_shifted)/len(f_shifted)
        elif mode == "doppler":
            # Apply the Doppler shift
            # fi = sci.interp1d(tw * (1.0 + rv / c), tf)

            # fi = sci.splrep(tw * (1.0 + rv / c), tf, s=0) 
            # f_shifted = sci.splev(w, fi, der=0) * weights
            f_shifted = np.interp(w, tw * (1.0 + rv / c), tf) * weights
            dfs = f_shifted - np.nansum(f_shifted)/len(f_shifted)

            # cc[i] = np.sum(f * sci.splev(w, fi, der=0) * weights)
            # f_new = sci.splev(v_new, tck, der=0)
        i += 1
        # Shifted template evaluated at location of spectrum

        cc[i] = np.nansum(df * dfs * weights)
        # cc /= np.sqrt(np.nansum(df**2) * np.nansum(dfs**2))
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
    flux_obs_nor = flux_obs_nor_box[i]
    cc = ccf_rv(
        wave_obs, flux_obs_nor, wave_tem, flux_tem_nor, 
        ccf_lw, ccf_up, step_size, mode="doppler", 
        skipedge=skipedge)
    # rv[np.argmax(cc)]
    return cc

def get_ccf_peak(ccf_box, drvs, visible_phi, band):
    new_rv_box = []

    for i in range(len(ccf_box)):
        ccf = ccf_box[i]
        peaks, properties = find_peaks(ccf, prominence=0.1) 

        if len(peaks) != 0:
            prominences = properties["prominences"]
            _loc = np.argmax(prominences)
            new_rv = drvs[peaks][_loc]

            if -np.abs(new_rv) == drvs[0]:
                new_rv = np.nan
                print(f'\n     >>>> CCF hit edge {new_rv}km/s at {band} phase {visible_phi[i]:1.3f}!')
        else:
            new_rv = np.nan
            print(f'\n     >>>> CCF cannot find peaks at {band} phase {visible_phi[i]:1.3f}!')
        
        vis.plot_cc(drvs, ccf, new_rv, visible_phi[i], band, properties)
        new_rv_box.append(new_rv)

    return np.array(new_rv_box)
    
    

def cal_ccf_rv(
        star_tot, Star, visible_phi, band, rv_ccf_pars, prf_dir, 
        cpu_use, RVandEWR_dir, Vel_bin_size, plot):
    visible_phi = np.array(visible_phi)
    vel = Star.star_spec[f'{band}_v']
    # vel = np.array([stsp.rndup(i, 4) for i in vel])

    # --- get ccf template ---
    line_cen = Star.bands2use[band][0]
    delta_wv = Star.bands2use[band][1]
    wvmin, wvmax = (line_cen-delta_wv, line_cen+delta_wv)
    wvmin = int(wvmin); wvmax = int(wvmax)

    # ---- trans vel to wave
    c_kms = 299792.458
    w_center = (wvmin + wvmax) / 2
    wave = vel * w_center / c_kms + w_center
    # ----

    ccf_tt, ccf_logg, ccf_FeH, ccf_vsini, ccf_resolution = rv_ccf_pars

    # Use user give par to generate ccf template
    if ccf_tt != 99999:
        # mc_2700350+000_b2_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf
        spec_base_name = f'mc_{ccf_tt:1.0f}{ccf_logg*100:1.0f}+{ccf_FeH*100:03.0f}_b0_water_BT2_synmast_tuned_moon880_spirou_1step_v1.9.prf'
        spec_dir = f'{prf_dir}/{spec_base_name}'
    
        sim_spec_name = f'RVCCFUse_Synmast_spec_{ccf_tt:1.0f}_{ccf_logg:1.2f}_{ccf_FeH:1.2f}_vsini{ccf_vsini:1.02f}_res{ccf_resolution:1.0f}_wv{wvmin}_{wvmax}_flatten.p'
        sim_spec_path = f'{Star.out_path}{sim_spec_name}'

        ssp.final_flux_spec(
            'ccf_temp', spec_dir, sim_spec_path, wvmin, wvmax, 
            Vel_bin_size, flatten=True, pars=rv_ccf_pars)
        _temp = pd.read_pickle(sim_spec_path)
        spec_temp = _temp['f'].to_numpy()
    # Use spot covergae to estimate the teff for spec as ccf template
    else:
        vel_temp = Star.star_spec_flatten[f'{band}_v']
        spec_temp = Star.star_spec_flatten[f'{band}_f']

        wave_temp = vel_temp * w_center / c_kms + w_center
        # do vsini
        wave_temp, spec_temp = rotint(wave_temp, spec_temp, Star.vsini)
        # do instrumental broadening
        hwhm_list = wave_temp / Star.inst_res / 2
        spec_temp = macbro_dyn(wave_temp, spec_temp, hwhm_list)

        vel_temp = c_kms * (wave_temp - w_center) / w_center
        # tck = interpolate.splrep(vel_temp, spec_temp, s=0)    
        # spec_temp = interpolate.splev(vel, tck, der=0)
        # spec_temp = rebin_jv(wave_temp, spec_temp, wave, False)
        spec_temp = flatten_ccf_model_spec(
            wave_temp, spec_temp, band, RVandEWR_dir, plot)

        # save ccf temlplate
        np.save(f'./outputs/{RVandEWR_dir}/ccf_template/{band}_ccf_template.npy', spec_temp)
        np.save(f'./outputs/{RVandEWR_dir}/ccf_template/{band}_ccf_template_wave.npy', wave_temp)


    # exrtra_len = 1000
    # del_vel = stsp.rndup(vel[1] - vel[0], 4)
    # vel_temp = np.arange(
    #     vel[0]-del_vel*exrtra_len, vel[-1]+del_vel*exrtra_len+del_vel, 
    #     del_vel
    #     )
    # vel_temp = np.array([stsp.rndup(i, 4) for i in vel_temp])
    # spec_temp = np.append(np.zeros(exrtra_len)+spec_temp[0], spec_temp)
    # spec_temp = np.append(spec_temp, np.zeros(exrtra_len)+spec_temp[-1])

    ##########################################################################################
    # ccf_rv_shift = _cc(star_tot, vel, vel_temp, spec_temp, visible_phi)
    # ccf_rv_shift = ccf_rv_shift - np.median(ccf_rv_shift)
    # print(f'vel: {vel}, tw: {vel_temp}, tf: {spec_temp}')
    # print(f'vel: {vel.shape}, tw: {vel_temp.shape}, tf: {spec_temp.shape}')

    ##########################################################################################
    # If the number of phases is larger then 10, do trial ccf to get RV ranges
    ccf_step = Vel_bin_size/100
    if len(visible_phi) <= 10:
        _edge = 40 
        clip_obs_edges = _edge / np.diff(vel_temp)[0]
        clip_obs_edges = int(clip_obs_edges*5)

        # func = partial(
        #     _cc, drvs, star_tot, vel, vel_temp, spec_temp, visible_phi)
        # ccf_box = pqdm(np.arange(len(star_tot)), func, n_jobs=cpu_use)
        
        func = partial(
            mc_ccf, wave, star_tot, wave_temp, spec_temp, 
            -_edge, _edge, ccf_step, clip_obs_edges, visible_phi)
        outs = pqdm(np.arange(len(star_tot)), func, n_jobs=cpu_use)

        drvs = np.arange(-_edge, _edge, ccf_step)
        ccf_rv_shift = get_ccf_peak(np.array(outs), drvs, visible_phi, band)
        ccf_rv_shift = ccf_rv_shift - np.nanmin(ccf_rv_shift)
    else:
        # trial CCF on phase closested to 0.1
        _loc0 = np.argmin(np.abs(visible_phi-0.0))
        _loc1 = np.argmin(np.abs(visible_phi-0.15))
        _loc2 = np.argmin(np.abs(visible_phi-0.2))
        _loc3 = np.argmin(np.abs(visible_phi-0.25))
        _loc4 = np.argmin(np.abs(visible_phi-0.3))
        _loc5 = np.argmin(np.abs(visible_phi-0.5))
        trial_locs = np.array([_loc0, _loc1, _loc2, _loc3, _loc4, _loc5], dtype=int)
        # print(trial_locs)
        _edge = 15
        clip_obs_edges = _edge / np.diff(vel_temp)[0]
        clip_obs_edges = int(clip_obs_edges*5)
        # print(f'clip_obs_edges: {clip_obs_edges}')
        
        # func = partial(
        #     _cc, trial_drvs, star_tot[trial_locs], vel, vel_temp, spec_temp, visible_phi)
        # _rv_shift_box = pqdm(np.arange(len(trial_locs)), func, n_jobs=4)

        func = partial(
            mc_ccf, wave, star_tot[trial_locs], wave_temp, spec_temp, 
            -_edge, _edge, ccf_step, clip_obs_edges, visible_phi)
        outs = pqdm(np.arange(len(trial_locs)), func, n_jobs=6)


        drvs = np.arange(-_edge, _edge, ccf_step)
        _rv_shift = get_ccf_peak(
            np.array(outs), drvs, visible_phi[trial_locs], band) 
        print('_rv_shift', _rv_shift)

        ccf_bounds = np.round(_rv_shift, decimals=3)
        # ccf_bound_lw = ccf_bounds[np.nanargmin(ccf_bounds)]
        # ccf_bound_up = ccf_bounds[np.nanargmax(ccf_bounds)]
        # ccf_bound = ccf_bound_up - ccf_bound_lw
        ccf_bound = np.nanmax(np.abs(ccf_bounds))
        # print(f'org ccf_bounds: {ccf_bounds}')
        print(f'org ccf_bound: {ccf_bound}')
        
        if ccf_bound <= 0.02:
            ccf_rv_shift = np.zeros(len(star_tot), dtype=np.float64)
            ccf_rv_shift[:] = np.nan
            return ccf_rv_shift
        
        elif (ccf_bound > 0.02) and (ccf_bound<=0.5):
            ccf_bound = 2
            ccf_step = ccf_step        
        elif (ccf_bound > 0.5) and (ccf_bound<=5):
            ccf_bound = 15
            ccf_step = ccf_step*10
        else:
            ccf_bound = ccf_bound * 4
            ccf_step = ccf_step*100

        print(f'ccf_bound: {ccf_bound}, ccf_step: {ccf_step}')
        # print(f'lower bound: {ccf_bound_lw}, upper bound: {ccf_bound_up}')
        # print(f'ccf_bound: {ccf_bound}')

        clip_obs_edges = (ccf_bound) / np.diff(vel_temp)[0]
        clip_obs_edges = int(clip_obs_edges*5)
            
        # print(f'clip_obs_edges: {clip_obs_edges}')
        
        # func = partial(
        #     _cc, drvs_full, star_tot, vel, vel_temp, spec_temp, visible_phi)
        # ccf_rv_shift_box = pqdm(np.arange(len(star_tot)), func, n_jobs=cpu_use)
        func = partial(
            mc_ccf, wave, star_tot, wave_temp, spec_temp, 
            -ccf_bound, ccf_bound, ccf_step, clip_obs_edges, visible_phi)
        ccf_rv_shift_box = pqdm(np.arange(len(star_tot)), func, n_jobs=cpu_use)

        drvs_full = np.arange(-ccf_bound, ccf_bound, ccf_step)
        ccf_rv_shift = get_ccf_peak(np.array(ccf_rv_shift_box), drvs_full, visible_phi, band)
        ccf_rv_shift = ccf_rv_shift - np.nanmean(ccf_rv_shift)

    return ccf_rv_shift





# def Calculate_activity_signal(
#     nrho, grid, star, psi, ccf, ccf_active_region, active_regions, inst_reso):

#     # Calculates the flux and CCF in each cell of the grid and integrate
#     # over the entire stellar disc to have the integrated flux (FLUXstar) and CCF (CCFstar).
#     # Calculate the CCF (CCFstar) and the total flux (FLUXstar) for the quiet star
#     CCFstar_quiet, FLUXstar_quiet, quite_surface = stsp.itot(
#         star.vrot, star.incl, star.limba1, star.limba2, 0., 0., 0., grid, ccf.rv, 
#         ccf.intensity, ccf.v_interval, ccf.n_v, ccf.n
#         )

#     # initialization of the variables
#     FLUXstar      = FLUXstar_quiet
#     CCFstar_flux  = CCFstar_quiet
#     CCFstar_bconv = CCFstar_quiet
#     CCFstar_tot   = CCFstar_quiet

#     #Loop over the different active regions
#     for idx, active_region in enumerate(active_regions):
#         print(f'\nCalculating each phase for active region {idx+1}...\n')
#         # active_region.check = 1 if the active region is activated, 0 otherwise
#         if not active_region.check: 
#             continue 
        
#         # Calculates the position of the spot initialized at the disc center
#         xyz = stsp.spot_init(
#             active_region.s, active_region.long, active_region.lat, star.incl, nrho
#             )
        
#         # Scans the yz-area where the spot is for different phases (psi) and
#         # returns the spot's "non-contribution" to the total flux and its
#         # "non-contribution" to the ccf, for each phase.
#         # Thus the result is to be subtracted to the output of the itot() function.
#         (CCFactive_region_flux, CCFactive_region_bconv, CCFactive_region_tot, 
#             FLUXactive_region) =\
#                 stsp.spot_scan_npsi(
#                     xyz, nrho, psi, len(psi), star.vrot, star.incl, star.limba1, 
#                     star.limba2, 0., 0., 0., grid, ccf.rv, ccf.intensity, 
#                     ccf_active_region.intensity, ccf.v_interval, ccf.n_v, ccf.n, 
#                     active_region.s, active_region.long, active_region.lat, 
#                     active_region.active_region_type, star.Temp, star.Temp_diff_spot
#                     )

#         # Calculate the total flux of the star affected by active regions
#         FLUXstar      = FLUXstar-np.array(FLUXactive_region)
        
#         # Calculate the CCF of the star affected by the flux effect of active regions
#         CCFstar_flux  = CCFstar_flux-np.array(CCFactive_region_flux)   
        
#         # Calculate the CCF of the star affected by the convective blueshift effect of active regions
#         CCFstar_bconv = CCFstar_bconv-np.array(CCFactive_region_bconv) 
        
#         # Calculate the CCF of the star affected by the total effect of active regions
#         CCFstar_tot   = CCFstar_tot-np.array(CCFactive_region_tot)     

#     # Calculate where the extrapolated CCF corresponds to the boundaries of the non-extrapolated CCF.
#     istart = (ccf.n_v-len(ccf.rv))//2   # ccf.n_v is odd given our definition, like len(ccf.rv), therefore 
#                                         # the difference can be divided by 2. The CCF has been extrapolated 
#                                         # the same way on each boundary, so dividing the difference between 
#                                         # the extrapolated and non extrapolated CCF by 2 gives by how many 
#                                         # points the CCF was extrapolated on each side
#     iend   = istart+len(ccf.rv)

#     # truncate the extrapolated rotating CCF to the same interval as the non-rotating CCF
#     CCFstar_flux  = CCFstar_flux[:,istart:iend]
#     CCFstar_bconv = CCFstar_bconv[:,istart:iend]
#     CCFstar_tot   = CCFstar_tot[:,istart:iend]
#     CCFstar_quiet = CCFstar_quiet[istart:iend]

#     # Normalization, loop through diff phases
#     CCFstar_flux  = np.array([f/np.max(f) for f in CCFstar_flux], dtype=np.float64)
#     CCFstar_bconv = np.array([f/np.max(f) for f in CCFstar_bconv], dtype=np.float64)
#     CCFstar_tot   = np.array([f/np.max(f) for f in CCFstar_tot], dtype=np.float64)
#     CCFstar_quiet = CCFstar_quiet/np.max(CCFstar_quiet)
    
#     #Convolution with a Gaussian instrumental profile of a given resolution, 
#     # given by the "instrument_reso" variable in the "config.cfg" file
#     if inst_reso != 0:
        
#         c = 299792458. # speed of light in m/s
#         HARPS_resolution         = inst_reso # resolution R = lambda / Delta(lambda)
#         HARPS_inst_profile_FWHM  = c/HARPS_resolution/1000. # Resolution = c/Delta_v -> Delta_v = c/R
#         HARPS_inst_profile_sigma = HARPS_inst_profile_FWHM/(2*np.sqrt(2*np.log(2)))
#         Gaussian_low_reso        = np.exp(-ccf.rv**2/(2*(HARPS_inst_profile_sigma)**2))
        
#         CCFstar_quiet_tmp = signal.convolve(
#             -CCFstar_quiet+1, Gaussian_low_reso, 'same'
#             )
#         CCFstar_quiet = 1-CCFstar_quiet_tmp\
#             *(1-np.min(CCFstar_quiet))/np.max(CCFstar_quiet_tmp) # normalization
        
#         # Convolution of the CCF with the Gaussian instrumental profile to reduce 
#         # the resolution of the CCF
#         for i in np.arange(len(CCFstar_flux)):
#             CCFstar_flux_tmp  = signal.convolve(-CCFstar_flux[i]+1,Gaussian_low_reso,'same')
#             CCFstar_flux[i]   = 1-CCFstar_flux_tmp*(1-np.min(CCFstar_flux[i]))/np.max(CCFstar_flux_tmp)
#             CCFstar_bconv_tmp = signal.convolve(-CCFstar_bconv[i]+1,Gaussian_low_reso,'same')
#             CCFstar_bconv[i]  = 1-CCFstar_bconv_tmp*(1-np.min(CCFstar_bconv[i]))/np.max(CCFstar_bconv_tmp)
#             CCFstar_tot_tmp   = signal.convolve(-CCFstar_tot[i]+1,Gaussian_low_reso,'same')
#             CCFstar_tot[i]    = 1-CCFstar_tot_tmp*(1-np.min(CCFstar_tot[i]))/np.max(CCFstar_tot_tmp)

#     return None
    # # Calculate the CCF parameters depth, BIS SPAN, RV and FWHM
    # depth_flux,span_flux,rv_flux,fwhm_flux     = [],[],[],[]
    # depth_bconv,span_bconv,rv_bconv,fwhm_bconv = [],[],[],[]
    # depth_tot,span_tot,rv_tot,fwhm_tot         = [],[],[],[]
    # for i in np.arange(len(CCFstar_tot)):
    #     #Flux effect
    #     CCF_prop_flux = compute_bis(ccf.rv,CCFstar_flux[i])
    #     depth_flux = append(depth_flux,CCF_prop_flux[0])
    #     span_flux = append(span_flux,CCF_prop_flux[2])
    #     rv_flux = append(rv_flux,CCF_prop_flux[3])
    #     fwhm_flux = append(fwhm_flux,abs(CCF_prop_flux[4]))
        
    #     #Convective Blueshift effect
    #     CCF_prop_bconv = compute_bis(ccf.rv,CCFstar_bconv[i])
    #     depth_bconv = append(depth_bconv,CCF_prop_bconv[0])
    #     span_bconv = append(span_bconv,CCF_prop_bconv[2])
    #     rv_bconv = append(rv_bconv,CCF_prop_bconv[3])
    #     fwhm_bconv = append(fwhm_bconv,abs(CCF_prop_bconv[4]))
        
    #     #Total effect
    #     CCF_prop_tot = compute_bis(ccf.rv,CCFstar_tot[i])
    #     depth_tot = append(depth_tot,CCF_prop_tot[0])
    #     span_tot = append(span_tot,CCF_prop_tot[2])
    #     rv_tot = append(rv_tot,CCF_prop_tot[3])
    #     fwhm_tot = append(fwhm_tot,abs(CCF_prop_tot[4]))

    # #Normalize the flux of the star
    # FLUXstar /= FLUXstar_quiet
    
    # #return the variables
    # return FLUXstar_quiet, CCFstar_quiet, FLUXstar, CCFstar_flux, CCFstar_bconv, CCFstar_tot, rv_flux, rv_bconv, rv_tot, span_flux, span_bconv, span_tot, fwhm_flux, fwhm_bconv, fwhm_tot, depth_flux, depth_bconv, depth_tot

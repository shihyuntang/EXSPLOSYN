import os, sys, argparse, socket
import itertools
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
# from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import configparser
from astropy.table import Table
from time import time

import starspot as stsp
import soap2 as soap2
from soap2 import flatten_model_spec
import visualization as vis


_bands = {
    # [line_center, half_wv_width, run models] \AA
    # 'IG_CO': [23157., 25, ['rv', 'photo']],
    'IG_OH': [15631.35, 25, ['rv', 'photo']],
    # 'opt_Gaia': [8665, 25, ['rv', 'photo']],
    # 'MD_9080': [9080, 20, ['rv', 'photo']],
    # 'MD_8220': [8220, 20, ['rv', 'photo']],
    # 'MD_7200': [7200, 20, ['rv', 'photo']],
    # 'MD_6280': [6280, 15, ['rv', 'photo']],
    # 'MD_5580': [5580, 15, ['rv', 'photo']],
    # 'opt_Gaia': [8585, 135, ['rv', 'photo']]
    # 'opt_': [5000.0, 20, ['rv', 'photo']]
}

nir_width = 40
opt_width1 = 20
opt_width2 = 15

# _bands = {
#     # [line_center, half_wv_width, run models] \AA
#     'nir_K3': [23925, nir_width, ['rv', 'photo']],
#     'nir_K4': [23631, nir_width, ['rv', 'photo']],
#     'nir_K5': [23366, nir_width, ['rv', 'photo']],
#     'nir_K6': [23094, nir_width, ['rv', 'photo']],
#     'nir_H6': [17250, nir_width, ['rv', 'photo']],
#     'nir_H14': [16040, nir_width, ['rv', 'photo']],
#     'nir_H16': [15760, nir_width, ['rv', 'photo']],
#     'IG_OH': [15631.35, nir_width, ['rv', 'photo']],
#     'nir_H21': [15111, nir_width, ['rv', 'photo']],
#     'nir_H22': [15006, nir_width, ['rv', 'photo']],
#     #
#     'opt_0': [9950, opt_width1, ['rv', 'photo']],
#     'opt_5': [8680, opt_width1, ['rv', 'photo']],
#     'opt_10': [7700, opt_width1, ['rv', 'photo']],
#     'opt_15': [6920, opt_width2, ['rv', 'photo']],
#     'opt_20': [6280, opt_width2, ['rv', 'photo']],
#     'opt_25': [5750, opt_width2, ['rv', 'photo']],
#     # # 'opt_30': [5320, opt_width2, ['rv', 'photo']],
#     'opt_35': [4940, opt_width2, ['rv', 'photo']],
#     # 'opt_40': [4600, opt_width2, ['rv', 'photo']],
#     # 'opt_45': [4320, opt_width2, ['rv', 'photo']],
#     # 'opt_50': [4070, opt_width2, ['rv', 'photo']],
#     # 'opt_Gaia': [8585, 135, ['rv', 'photo']]
#     # 'opt_': [5000.0, 20, ['rv', 'photo']]
# }

# ------------------------------------------------
# # Define the parameters
# line_centers = np.arange(6000, 24001, 200) # AA, 91 runs
# # line_centers = np.arange(6000, 24001, 200)[-1:] # AA
# # line_centers = np.arange(14000, 24001, 200) # AA
# # line_centers = np.arange(10000, 24001, 200) # AA
# # line_centers = np.arange(14200, 24001, 200) # AA
# _width = 100
# run_models = ['rv', 'photo']

# # Create the dictionary
# _bands = {
#     f"w_{wave_center // 100}": [wave_center, _width, run_models]
#     for wave_center in line_centers
# }

hostname = socket.gethostname() 
hostname = hostname.split('.')[0].split('-')[0] # either Swing or taurus
# RVandEWR_dir = 'RVandEWR_Reiners_3700_2400_width50'
RVandEWR_dir = f'RVandEWR_LkCa4_width100_50cover_Tdel500_{hostname}'
# RVandEWR_dir = f'RVandEWR_res_test_{hostname}_100-1'
# RVandEWR_dir = 'RVandEWR_V830Tau_EWR'

Vel_bin_size = 100/1000 # km/s
# ------------------------------------------------
# _bands = {
#     # [line_center, half_wv_width, run models] \AA
#     # 'w_120': [12000, 50, ['rv', 'photo']],
#     # 'w_128': [12800, 50, ['rv', 'photo']],
#     # 'w_134': [13400, 50, ['rv', 'photo']],
#     # 'w_136': [13600, 50, ['rv', 'photo']],
#     # 'w_138': [13800, 50, ['rv', 'photo']],
# }

if __name__ == '__main__':

    # -- INITIALIZATION --
    config = configparser.ConfigParser(inline_comment_prefixes=';')   
    # config.read("config_v830tau.cfg")
    # config.read("config_Reiners+2010_2.cfg")
    # config.read("config_multi_spot_test.cfg")
    # config.read("config_citau.cfg")
    config.read("config_LkCa4_50pc_Tdel500.cfg")

    parser = argparse.ArgumentParser(
        prog        = 'Run LkCa4_mc in batch',
        description = '''
        NaN
        ''', epilog = 'sytang@rice.edu'
        )
    
    parser.add_argument(
        '-st', dest="st", action="store", help="The start dates id to run.",
        type=int, default=None)
    parser.add_argument(
        '-ed', dest="ed", action="store", help="The end dates id to run.",
        type=int, default=None)
    
    args = parser.parse_args()

    # ------------------------------------------------------------
    line_centers = np.arange(6000, 24001, 200)[args.st:args.ed] # AA, 91 runs
    # line_centers = np.arange(6000, 24001, 200)[-1:] # AA
    # line_centers = np.arange(14000, 24001, 200) # AA
    # line_centers = np.arange(10000, 24001, 200) # AA
    # line_centers = np.arange(14200, 24001, 200) # AA
    _width = 100
    run_models = ['rv', 'photo']

    # Create the dictionary
    _bands = {
        f"w_{wave_center // 100}": [wave_center, _width, run_models]
        for wave_center in line_centers
}
    # ------------------------------------------------------------
    # -- Read the configuration file --
    path_sub = 1 if hostname == 'Swing' else 2
    RASSINE_PATH = config.get('path',f'rassine_path{path_sub}')
    SYNmast_PATH = config.get('path',f'synmast_prf{path_sub}')
    PLOT = int(config.get('output','plot' ))
    SAVESPEC = int(config.get('output','save_spec' ))

    GRID = int(config.get('main','grid' ))
    NRHO = int(config.get('main','nrho' ))
    INST_RESO = int(config.get('main','instrument_reso' ))
    CPU = int(config.get('main','cpu' ))
    CPU = 6 if hostname == 'Swing' else 29
    
    RAD_Sun = int(config.get('star','radius_sun' ))
    RAD = float(config.get('star','radius' )) * RAD_Sun
    PROT = float(config.get('star','prot' ))

    STAR = soap2.Star(
        prot = PROT,
        vrot = (2.*np.pi*RAD)/(PROT*86400.),
        incl = float(config.get('star','I')),
        limba1 = float(config.get('star','limb1')),
        limba2 = float(config.get('star','limb2')),
        psi = float(config.get('star','psi')),
        rad_sun = RAD_Sun,
        rad = RAD,
        Temp = int(config.get('star','Tstar')),
        Tdiff_spot = int(config.get('star','Tdiff_spot')),
        Tdiff_faculae = int(config.get('star','Tdiff_faculae')),
        logg = float(config.get('star', 'logg' )),
        FeH = float(config.get('star', 'FeH' )),
        bands2use = _bands
        )

    STAR.vsini = STAR.vrot * np.sin(np.deg2rad(STAR.incl))
    STAR.inst_res = INST_RESO

    syn_type = config.get('star','syn_spec')
    if syn_type.lower() == 'phx':
        syn_spec_path = RASSINE_PATH
    elif syn_type.lower() == 'synmast':
        syn_spec_path = SYNmast_PATH
    else:
        raise ValueError('Unknown syn_spec type. Please use "phx" or "synmast".')
    
    # -- Setup the synthetic spectra --
    STAR.find_spec(syn_type, Vel_bin_size, syn_spec_path)

    SPEC_photo_box = []
    SPEC_spot_box = []
    SPEC_plage_box = []
    for band in _bands.keys():
        SPEC_photo_box.append(
            soap2.Spec(type = 'photosphere', star = STAR, band = band) )
        SPEC_spot_box.append(
            soap2.Spec(type = 'spot', star = STAR, band = band) )
        SPEC_plage_box.append(
            soap2.Spec(type = 'plage', star = STAR, band = band) )
        
    ######################################################################
    # -- Check how many active regions are there --
    config['active_regions'].items()
    active_regions_number = 0
    # Check for the active_regions section
    for key, value in config['active_regions'].items():
        # Look for keys that start with 'check'
        if (key.startswith('check')) and (value.strip() == '1'):
            active_regions_number += 1

    # Output results
    print(f"\n\nThere are {active_regions_number} active regions found.")

    active_region_checks = [
        int(config.get('active_regions',f'check{i}')) for i in range(1, active_regions_number+1)
        ]
    active_region_longs = [
        float(config.get('active_regions',f'long{i}')) for i in range(1, active_regions_number+1)
        ]
    active_region_lats = [
        float(config.get('active_regions',f'lat{i}')) for i in range(1, active_regions_number+1)
        ]
    active_region_sizes = [
        float(config.get('active_regions',f'size{i}')) for i in range(1, active_regions_number+1)
        ]
    active_region_types = [
        int(config.get('active_regions',f'act_reg_type{i}')) for i in range(1, active_regions_number+1)
        ]
    
    active_region_box = [
        active_region_checks, active_region_longs, active_region_lats, 
        active_region_sizes, active_region_types]

    ######################################################################

    outdir = "./outputs/{}/{}_P={:.2f}_i={:.2f}_T1={:1.0f}_DelT={:1.0f}_logg={:1.1f}_lon={}_lat={}_size={}" .format(
        RVandEWR_dir, STAR.spec_type, STAR.prot, STAR.incl, STAR.Temp,
        STAR.Tdiff_spot, STAR.logg, active_region_longs, active_region_lats, 
        active_region_sizes)
    
    os.makedirs(f"outputs/{RVandEWR_dir}", exist_ok=True)
    os.makedirs(f"outputs/{RVandEWR_dir}/normal_spec", exist_ok=True)
    os.makedirs(f"outputs/{RVandEWR_dir}/spec", exist_ok=True)
    os.makedirs(f"outputs/{RVandEWR_dir}/ccf_template", exist_ok=True)
    os.makedirs(f"outputs/CCF", exist_ok=True)
        
    ######################################################################
    # -- RV CCF use template parameters --
    Use_estimated_teff = int(config.get('ccf','Use_estimated_teff'))
    rv_ccf_pars = np.array([
        float(config.get('ccf','Teff')),
        float(config.get('ccf','logg')),
        float(config.get('ccf','FeH')),
        float(config.get('ccf','vsini')),
        INST_RESO
    ])
    rv_ccf_pars[3] = STAR.vsini if rv_ccf_pars[3] == -1 else rv_ccf_pars[3]
    
    if Use_estimated_teff == 1: 
        rv_ccf_pars[:] = 99999
        STAR.find_spec_flatten(
            syn_type, active_region_sizes, Vel_bin_size, syn_spec_path)
    
    ######################################################################

    # -- Phase at which the signal will be generated --
    phase_step  = float(config.get('output','ph_step'))
    phase_in    = config.get('output','ph_in')

    # Read the phase in the file given in the "phase_in" argument in "config.cfg"
    if phase_in != 'None':
        f   = open(phase_in)
        PSI = np.array([float(l.split()[0]) for l in f.readlines()])
        f.close()
    # Otherwise create a evenly sampled phase between 0 and 1 with the step given
    # by the "ph_step" argument in "config.cfg"
    elif phase_step > 0:
        PSI = np.arange(-0.1, 1.1, phase_step)
        # PSI = np.arange(-0.1, 0.5, phase_step)
    else:
        print("WARNING: put a positive phase step ('ph_step' argument in 'config.cfg') \
or give a file name that contain the phases at which you want to compute the model \
('phase_in' argument in 'config.cfg')")


    for i in range(len(SPEC_photo_box)):
        start_time = time()
        band = list(_bands.keys())[i]

        print('\n\nNow calculating the activity signal for the band: ', band)

        SPEC_photo = SPEC_photo_box[i]
        SPEC_spot = SPEC_spot_box[i]
        SPEC_plage = SPEC_plage_box[i]
        int_ratio = STAR.intensity_ratio[band] 

        visible_phi, SPECstar_tot, quite_surface, quite_surface_2d =\
            soap2.calculate_activity_signal_spec(
            NRHO, GRID, STAR, PSI, 
            SPEC_photo, SPEC_spot, SPEC_plage,
            active_regions_number, active_region_box, INST_RESO, CPU, int_ratio,
            RVandEWR_dir, band, SAVESPEC
            )
        
        # --- flatten the spectra ---
        # np.save(f'./outputs/{RVandEWR_dir}/spec/{band}_non_flat_spec.npy', SPECstar_tot)
        SPECstar_tot = flatten_model_spec(STAR, SPECstar_tot, visible_phi, band, RVandEWR_dir)
        np.save(f'./outputs/{RVandEWR_dir}/spec/{band}_flat_final_spec.npy', SPECstar_tot)

        if band == 'IG_OH':        
            # depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, depth_l_fe2, depth_l_oh0 = soap2.get_line_depth(
            #     STAR, SPECstar_tot, visible_phi, band
            #     )
            ohfe_ratio = soap2.get_equivalent_width(STAR, SPECstar_tot, visible_phi, band)

            # oh_line = np.array(d_oh0)
            # fe_line = (np.array(d_fe1) + np.array(d_fe2)) / 2
            # _ratio = oh_line/fe_line
            # print(f'--> {band} EWR amp. {np.max(ohfe_ratio) - np.min(ohfe_ratio):1.4f}')
        else:
            depth_d_fe1, depth_d_fe2, depth_d_oh0 = [None, None, None]
            ohfe_ratio = np.zeros_like(visible_phi) + np.nan

        # -- Calculate CCF RV --
        print('\nCalculating the RV with CCF...\n')
        ccf_rv_shift = soap2.cal_ccf_rv(
            SPECstar_tot, STAR, visible_phi, band, rv_ccf_pars, SYNmast_PATH, 
            CPU, RVandEWR_dir, Vel_bin_size)
        # print(f'--> {band} RV amp. {np.nanmax(ccf_rv_shift) - np.nanmin(ccf_rv_shift):1.6f} km/s')

        # -- Plot the simulated results --
        text1 =  '{0:12} = {1:<8d}, {2:18} = {3:<8d}\n'.format(   'Grid res.', GRID,          'Circumference res.', NRHO) + \
                '{0:12} = {1:<8d}, {2:18} = {3:<8d}\n'.format(    'Inst. res.', INST_RESO,    'Rsun [km]', RAD_Sun) + \
                '{0:12} = {1:<8.2f}, {2:18} = {3:<8.2f}\n'.format('St Prot [d]', STAR.prot,   'St radius [Rsun]', STAR.rad/RAD_Sun) + \
                '{0:12} = {1:<8.1f}, {2:18} = {3:<8.2f}\n'.format('St incl', STAR.incl,       'St vsini [km/s]', STAR.vsini) + \
                '{0:12} = {1:<8d}, {2:18} = {3:<8d}\n'.format(    'St Teff [K]', STAR.Temp,   'Spot Diff Teff [K]', STAR.Tdiff_spot) + \
                '{0:12} = {1:<8.3f}, {2:18} = {3:<8.3f}\n'.format('LimDark lin', STAR.limba1, 'LimDark quad', STAR.limba2) + \
                '{0:12} = {1:<8}, {2:5} = {3:<4.2f}, {4:5} = {5:<4.2f}'.format('Syn template', STAR.spec_type, 'logg', STAR.logg, '[Fe/H]', 0.0)

        text2 = 'ActR {0}: lon={1:<5.1f}, lat={2:<4.1f}, size = {3:<5.3f} [Rsun] (={4:.2f}%)\n'.format(
            '1',active_region_longs[0],active_region_lats[0],active_region_sizes[0],active_region_sizes[0]**2/2.*100)# +\
                # 'ActR {0}: lon={1:<5.1f}, lat={2:<4.1f}, size = {3:<5.3f} [Rsun] (={4:.2f}%)\n'.format(
            # '2',active_region_longs[1],active_region_lats[1],active_region_sizes[1],active_region_sizes[1]**2/2.*100)
                # 'Act Reg {0}: {1:<6}, lon = {2:<5.1f}, lat = {3:<4.1f}, size = {4:<5.3f} [Rsun] (= {5:.2f}%)\n'.format('3',active_region3.active_region_type_str,active_region3.long,active_region3.lat,active_region3.s,active_region3.s**2/2.*100) +\
                # 'Act Reg {0}: {1:<6}, lon = {2:<5.1f}, lat = {3:<4.1f}, size = {4:<5.3f} [Rsun] (= {5:.2f}%)'.format('4',active_region4.active_region_type_str,active_region4.long,active_region4.lat,active_region4.s,active_region4.s**2/2.*100)

        tab_save = Table(
            data=(visible_phi, ccf_rv_shift, ohfe_ratio), 
            names=('Phase', 'RV', 'EWR'))
        tab_save['Phase'].format = '%.3f'
        tab_save['RV'].format = '%.5f'
        tab_save['EWR'].format = '%.5f'
        tab_save.write(
            f'./outputs/{RVandEWR_dir}/{outdir.split("/")[-1]}_{band}_RV_EWR.csv', 
            format='csv', overwrite=True)

        if PLOT == 1:
            vis.spot_view(
                STAR, GRID, visible_phi, SPECstar_tot, ccf_rv_shift,
                quite_surface_2d, outdir, CPU, ohfe_ratio, text1, text2, band)
        
        end_time = time()
        # print run time in min
        print(f'\n--> {band} run time: {(end_time-start_time)/60:1.2f} min')
    
    ######################################################################
    ## --- Diagnostic plot for RV and EWR
    center_wvs = np.array([_bands[band][0]/1e4 for band in _bands.keys()])

    norm = mpl.colors.Normalize(vmin=center_wvs.min(), vmax=center_wvs.max())
    s_m = mpl.cm.ScalarMappable(cmap=mpl.cm.coolwarm, norm=norm)
    s_m.set_array([])

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.5), dpi=300)    
    
    amp_box = []
    
    for i, ss in zip(
        range(len(SPEC_photo_box)), itertools.cycle('ox>^+d*s')):
        band = list(_bands.keys())[i]
        cen_wv = center_wvs[i]

        tab = Table.read(
            f'./outputs/{RVandEWR_dir}/{outdir.split("/")[-1]}_{band}_RV_EWR.csv', 
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
        f'./outputs/{RVandEWR_dir}/{outdir.split("/")[-1]}_RV_{line_centers[0]}_{line_centers[-1]}.png', 
        bbox_inches='tight')

    ## ---

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=300)    
    ax.plot(center_wvs, amp_box, 'o', ms=3, c='k')

    ax.set_xlabel('Wavelength [um]')
    ax.set_ylabel('RV amplitude [km/s]')
    ax.set_xlim(0, 3.0)
    # ax.set_ylim(0, np.round(np.max(amp_box))+1)

    fig.savefig(
        f'./outputs/{RVandEWR_dir}/{outdir.split("/")[-1]}_RV_amp_{line_centers[0]}_{line_centers[-1]}.png', 
        bbox_inches='tight')


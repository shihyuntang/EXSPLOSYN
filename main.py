import os, sys, argparse, socket, itertools, shutil
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import configparser
from astropy.table import Table
from time import time

import starspot as stsp
import soapcore as soapcore
from soapcore import flatten_model_spec
import visualization as vis

ConfigFile = 'config_LkCa4base_SizeDelT_study.cfg'
# ConfigFile = 'config_PhaseDelay_SizeDelT_study.cfg'
# config.read("config_v830tau.cfg")-8
# config.read("config_Reiners+2010_2.cfg")
# config.read("config_multi_spot_test.cfg")
# config.read("config_citau.cfg")
# config.read("config_LkCa4_50pc_Tdel500.cfg")

SaveDir = f'RVandEWR_SizeDelT_width100'
# SaveDir = f'RVandEWR_mu_intensity_interp_TEST'
# SaveDir = f'RVandEWR_mu_fluxTEST'
# SAVEsub_dir = 'RVandEWR_Reiners_3700_2400_width50'
# SAVEsub_dir = f'RVandEWR_LkCa4_width100_50cover_Tdel500_{hostname}'
# SAVEsub_dir = f'RVandEWR_res_test_{hostname}_100-1'
# SAVEsub_dir = 'RVandEWR_V830Tau_EWR'

# SimSpecLineCenters = np.arange(6000, 24001, 5000) # in AA, test
# SimSpecLineCenters = np.arange(6000, 24001, 200) # AA, 91 runs
# SimSpecLineCenters = np.arange(6000, 24001, 500) # AA, 37 runs

# SimSpecLineCenters = np.arange(10000, 24001, 500) # AA, 29 runs
# SimSpecLineCenters = SimSpecLineCenters[SimSpecLineCenters != 11500]

# only fo CO region
SimSpecLineCenters = np.arange(23000, 24001, 200) # AA, 5 runs
# line_centers = np.arange(6000, 24001, 200)[-1:] # AA
# line_centers = np.arange(14000, 24001, 200) # AA
# line_centers = np.arange(10000, 24001, 200) # AA
# line_centers = np.arange(14200, 24001, 200) # AA

SimSpecWinSize = 100 # in AA
SimSpec_VelBinSize = 100/1000 # in km/s

# ------------------------------------------------
HOSTNAME = socket.gethostname() 
HOSTNAME = HOSTNAME.split('.')[0].split('-')[0] # either Swing or taurus

if __name__ == '__main__':

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
    
    parser.add_argument(
        '-fluxORint', dest="fluxORint", action="store", help="Run model by integrated flux or intensity.",
        type=str, default=None)
    
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
    
    args = parser.parse_args()

    if args.fluxORint is not None:
        if args.fluxORint.lower() not in ['flux', 'int']:
            raise ValueError("Please use 'flux' or 'int' for fluxORint.")
        else:
            print(f'Running model by {args.fluxORint}...')
    else:
        raise ValueError("Please use -fluxORint to specify 'flux' or 'int'.")
    
    # -- INITIALIZATION --
    config = configparser.ConfigParser(inline_comment_prefixes=';')   
    config.read(ConfigFile)
    SaveDir = f'{SaveDir}_{args.out}_{HOSTNAME}'
        
    # ------------------------------------------------------------
    if (args.st is not None) and (args.ed is not None):
        SimSpecLineCenters = SimSpecLineCenters[args.st:args.ed] # AA, test
    run_models = ['rv', 'photo']

    # Create the dictionary
    _bands = {
        f"w{wave_center // 100}": [float(wave_center), SimSpecWinSize, run_models]
        for wave_center in SimSpecLineCenters
    }
    # _bands = {}
    _bands['EWR_FeOH'] = [15631.35, 30, run_models]
    # ------------------------------------------------------------
    # -- Read the configuration file --
    path_sub = 1 if HOSTNAME == 'Swing' else 2
    RASSINE_PATH = config.get('path',f'rassine_path{path_sub}')
    SYNmast_PATH = config.get('path',f'synmast_prf{path_sub}')
    PHX_PATH = config.get('path','PHOENIX_path')

    PLOT = int(config.get('output','plot' ))
    if args.plot is not None:
        PLOT = 1 if 't' in args.plot.lower() else 0
        
    SAVESPEC = int(config.get('output','save_spec' ))

    GRID = int(config.get('main','grid' ))
    NRHO = int(config.get('main','nrho' ))
    INST_RESO = int(config.get('main','instrument_reso' ))
    CPU = int(config.get('main','cpu' ))
    CPU = 6 if HOSTNAME == 'Swing' else 31
    
    RAD_Sun = int(config.get('star','radius_sun' ))
    RAD = float(config.get('star','radius' )) * RAD_Sun
    PROT = float(config.get('star','prot' ))

    STAR = soapcore.Star(
        prot = PROT,
        vrot = (2.*np.pi*RAD)/(PROT*86400.),
        incl = float(config.get('star','I')),
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
    if args.Incl is not None:
        STAR.incl = float(args.Incl)
        
    STAR.vsini = STAR.vrot * np.sin(np.deg2rad(STAR.incl))
    STAR.inst_res = INST_RESO

    syn_type = config.get('star','syn_spec')
    if syn_type.lower() == 'phx':
        syn_spec_path = RASSINE_PATH
        STAR.spec_type = 'phx'
    elif syn_type.lower() == 'synmast':
        syn_spec_path = SYNmast_PATH
        STAR.spec_type = 'synmast'
    else:
        raise ValueError('Unknown syn_spec type. Please use "phx" or "synmast".')
    
    # Currently, we almost always use the PHOENIX spectra for RV CCF
    STAR.phx_path = Path(f'{PHX_PATH}/PHOENIX-ACES-AGSS-COND-2011').expanduser()
    os.makedirs(STAR.phx_path, exist_ok=True)
    os.makedirs(STAR.phx_path / 'SpecInt', exist_ok=True)
    os.makedirs(STAR.phx_path / 'SpecFlux', exist_ok=True)

    # -- Overwrite pars if been args updated --
    # for grid testing
    if args.delT is not None:
        STAR.Tdiff_spot = int(args.delT)
        STAR.Tspot = int(STAR.Temp - STAR.Tdiff_spot)
    if args.out is not None:
        SaveDir = f'RVandEWR_SizeDelT_width100_{args.out}_{HOSTNAME}'
    
        
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
    
    # --- Overwrite pars if been args updated ---
    if args.Scover is not None:
        # arg input is in % for visible DISK, need to transfer into size for soap
        ppc_cover = np.array(args.Scover)/100
        _spot_cover_size = ppc_cover**0.5
        active_region_sizes = [float(np.round(_spot_cover_size, 6))]
    
    if args.Lat is not None:
        active_region_lats = [float(args.Lat)]

    active_region_box = [
        active_region_checks, active_region_longs, active_region_lats, 
        active_region_sizes, active_region_types]

    ######################################################################

    outdir = "./outputs/{}/{}_P={:.2f}_i={:.2f}_T1={:1.0f}_DelT={:1.0f}_logg={:1.1f}_lon={}_lat={}_size={}" .format(
        SaveDir, STAR.spec_type, STAR.prot, STAR.incl, STAR.Temp,
        STAR.Tdiff_spot, STAR.logg, active_region_longs, active_region_lats, 
        active_region_sizes)
    
    os.makedirs(f"syn_spec", exist_ok=True)
    os.makedirs(f"syn_spec/RASSINE_PHX", exist_ok=True)
    os.makedirs(f"syn_spec/SimUse", exist_ok=True)
    os.makedirs(f"outputs/{SaveDir}", exist_ok=True)
    os.makedirs(f"outputs/{SaveDir}/normal_spec", exist_ok=True)
    os.makedirs(f"outputs/{SaveDir}/spec", exist_ok=True)
    os.makedirs(f"outputs/{SaveDir}/ccf_template", exist_ok=True)
    os.makedirs(f"outputs/CCF", exist_ok=True)
        
    ######################################################################
    # -- RV CCF use template parameters --
    UsePhotosphereTeff = int(config.get('ccf','Use_photosphere_teff'))
    rv_ccf_pars = np.array([
        float(config.get('ccf','Teff')),
        float(config.get('ccf','logg')),
        float(config.get('ccf','FeH')),
        float(config.get('ccf','vsini')),
        INST_RESO
    ])
    rv_ccf_pars[3] = STAR.vsini if rv_ccf_pars[3] == -1 else rv_ccf_pars[3]
    
    if UsePhotosphereTeff == 1: 
        rv_ccf_pars[:] = 99999

        # -- Estimated the Teff and use it for the LD Teff --
        # spot_portion = 0 # visible "DISK" portion
        # for ss in active_region_sizes:
        #     spot_portion = spot_portion + ss**2

        # est_teff = self.Temp**4*(1-spot_portion) \
        #     + self.Tspot**4*spot_portion
        # est_teff = int(est_teff**0.25)
        # print(f'\nEstimated Teff for calculating Limb darkening: {est_teff:1.0f} K') 

    # Get flatten synthetic spectra for RV CCF
    STAR.find_fluxspec_flatten(
        syn_type, SimSpec_VelBinSize, PLOT, syn_spec_path)
    

    if args.fluxORint.lower() == 'flux':
        # Flux mode: disk integration by scaling flux based on limb darkening
        # -- Get Limb darkening parameters --
        line_centers = [v[0] for v in _bands.values()]
        l_widths = [v[1] for v in _bands.values()]
        LD_coeff_photo = STAR.get_quadratic_limbdarken(
            STAR.Temp, line_centers, l_widths)
        LD_coeff_spot = STAR.get_quadratic_limbdarken(
            STAR.Tspot, line_centers, l_widths)
        LD_coeff_faculae = STAR.get_quadratic_limbdarken(
            STAR.Tfaculae, line_centers, l_widths)
        
        # -- Setup the synthetic spectra --
        STAR.find_fluxspec(syn_type, SimSpec_VelBinSize, PLOT, syn_spec_path)

    elif args.fluxORint.lower() == 'int':
        STAR.find_intensityspec(syn_type, SimSpec_VelBinSize, PLOT, syn_spec_path)
    
    SPEC_photo_box = []
    SPEC_spot_box = []
    SPEC_plage_box = []
    for band in _bands.keys():
        SPEC_photo_box.append(
            soapcore.Spec(type = 'photosphere', star = STAR, band = band) )
        SPEC_spot_box.append(
            soapcore.Spec(type = 'spot', star = STAR, band = band) )
        SPEC_plage_box.append(
            soapcore.Spec(type = 'plage', star = STAR, band = band) )
        


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
        PSI = np.arange(-0.75, 0.71, phase_step)
        # PSI = np.arange(-0.1, 0.5, phase_step)
    else:
        print("WARNING: put a positive phase step ('ph_step' argument in 'config.cfg') \
or give a file name that contain the phases at which you want to compute the model \
('phase_in' argument in 'config.cfg')")


    for i in range(len(SPEC_photo_box)):
        start_time = time()
        band = list(_bands.keys())[i]

        if args.fluxORint.lower() == 'flux':
            # STAR.limba1_photo = LD_coeff_photo[band][0]
            # STAR.limba2_photo = LD_coeff_photo[band][1]
            # STAR.limba1_spot = LD_coeff_spot[band][0]
            # STAR.limba2_spot = LD_coeff_spot[band][1]
            # STAR.limba1_faculae = LD_coeff_faculae[band][0]
            # STAR.limba2_faculae = LD_coeff_faculae[band][1]

            STAR.limba1 = [
                LD_coeff_photo[band][0],
                LD_coeff_spot[band][0],
                LD_coeff_faculae[band][0]
            ]
            STAR.limba2 = [
                LD_coeff_photo[band][1],
                LD_coeff_spot[band][1],
                LD_coeff_faculae[band][1]
            ]
        elif args.fluxORint.lower() == 'int':
            STAR.limba1 = [0.0, 0.0, 0.0]
            STAR.limba2 = [0.0, 0.0, 0.0]

        print('\n\nNow calculating the activity signal for the band: ', band)

        SPEC_photo = SPEC_photo_box[i]
        SPEC_spot = SPEC_spot_box[i]
        SPEC_plage = SPEC_plage_box[i]
        # int_ratio = STAR.intensity_ratio[band] 
        int_ratio = 1

        visible_phi, SPECstar_tot, quite_surface, quite_surface_2d =\
            soapcore.calculate_activity_signal_spec(
            NRHO, GRID, STAR, PSI, 
            SPEC_photo, SPEC_spot, SPEC_plage,
            active_regions_number, active_region_box, INST_RESO, CPU, int_ratio,
            SaveDir, band, SAVESPEC, args.fluxORint.lower()
            )
        
        # check_if_see_spot = [np.sum(spec_tot - quite_surface) == 0 for spec_tot in quite_surface_2d]

        # print(check_if_see_spot)
        # sys.exit('done!!')
    
        # sys.exit('cool')
        # --- flatten the spectra ---
        # np.save(f'./outputs/{RVandEWR_dir}/spec/{band}_non_flat_spec.npy', SPECstar_tot)
        SPECstar_tot = flatten_model_spec(
            STAR, SPECstar_tot, visible_phi, band, SaveDir, PLOT)
        if SAVESPEC == 1:
            np.save(f'./outputs/{SaveDir}/spec/{band}_flat_final_spec.npy', SPECstar_tot)

        if band == 'EWR_FeOH':        
            # depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, depth_l_fe2, depth_l_oh0 = soap2.get_line_depth(
            #     STAR, SPECstar_tot, visible_phi, band
            #     )
            ohfe_ratio = soapcore.get_equivalent_width(STAR, SPECstar_tot, visible_phi, band)

            # oh_line = np.array(d_oh0)
            # fe_line = (np.array(d_fe1) + np.array(d_fe2)) / 2
            # _ratio = oh_line/fe_line
            # print(f'--> {band} EWR amp. {np.max(ohfe_ratio) - np.min(ohfe_ratio):1.4f}')
        else:
            depth_d_fe1, depth_d_fe2, depth_d_oh0 = [None, None, None]
            ohfe_ratio = np.zeros_like(visible_phi) + np.nan

        # -- Calculate CCF RV --
        print('\nCalculating the RV with CCF...\n')
        ccf_rv_shift = soapcore.cal_ccf_rv(
            SPECstar_tot, STAR, visible_phi, band, rv_ccf_pars, SYNmast_PATH, 
            CPU, SaveDir, SimSpec_VelBinSize, PLOT)
        # print(f'--> {band} RV amp. {np.nanmax(ccf_rv_shift) - np.nanmin(ccf_rv_shift):1.6f} km/s')

        # -- Plot the simulated results --
        text1 =  '{0:12} = {1:<8d}, {2:18} = {3:<8d}\n'.format(   'Grid res.', GRID,          'Circumference res.', NRHO) + \
                '{0:12} = {1:<8d}, {2:18} = {3:<8d}\n'.format(    'Inst. res.', INST_RESO,    'Rsun [km]', RAD_Sun) + \
                '{0:12} = {1:<8.2f}, {2:18} = {3:<8.2f}\n'.format('St Prot [d]', STAR.prot,   'St radius [Rsun]', STAR.rad/RAD_Sun) + \
                '{0:12} = {1:<8.1f}, {2:18} = {3:<8.2f}\n'.format('St incl', STAR.incl,       'St vsini [km/s]', STAR.vsini) + \
                '{0:12} = {1:<8d}, {2:18} = {3:<8d}\n'.format(    'St Teff [K]', STAR.Temp,   'Spot Diff Teff [K]', STAR.Tdiff_spot) + \
                '{:12} = {}\n'.format('LimDark lin', [float(np.round(ll, 3)) for ll in STAR.limba1]) + \
                '{:12} = {}\n'.format('LimDark quad', [float(np.round(ll, 3)) for ll in STAR.limba2]) + \
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
        tab_save['RV'].format = '%.4f'
        tab_save['EWR'].format = '%.5f'
        tab_save.write(
            f'./outputs/{SaveDir}/{outdir.split("/")[-1]}_{band}_RV_EWR.csv', 
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
    if len(SimSpecLineCenters) > 1:
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
                f'./outputs/{SaveDir}/{outdir.split("/")[-1]}_{band}_RV_EWR.csv', 
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
            f'./outputs/{SaveDir}/{outdir.split("/")[-1]}_RV_{SimSpecLineCenters[0]}_{SimSpecLineCenters[-1]}.png', 
            bbox_inches='tight')

        ## ---

        fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=300)    
        ax.plot(center_wvs, amp_box, 'o', ms=3, c='k')

        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel('RV amplitude [km/s]')
        ax.set_xlim(0, 3.0)
        # ax.set_ylim(0, np.round(np.max(amp_box))+1)

        fig.savefig(
            f'./outputs/{SaveDir}/{outdir.split("/")[-1]}_RV_amp_{SimSpecLineCenters[0]}_{SimSpecLineCenters[-1]}.png', 
            bbox_inches='tight')


import os, sys, argparse
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
# from scipy import signal
# import matplotlib.pyplot as plts
# import matplotlib as mpl
import configparser
# from astropy.table import Table
# from numba import njit

import starspot as stsp
import soap2 as soap2
import visualization as vis


if __name__ == '__main__':
    # -- INITIALIZATION --
    config = configparser.ConfigParser(inline_comment_prefixes=';')
    # config.read("config_v830tau.cfg")
    config.read("config_multi_spot_test.cfg")
    # config.read("config_citau.cfg")
    RASSINE_PATH = config.get('path','rassine_path')


    GRID = int(config.get('main','grid' ))
    NRHO = int(config.get('main','nrho' ))
    INST_RESO = int(config.get('main','instrument_reso' ))
    CPU = int(config.get('main','cpu' ))
    band = str(config.get('main','band' ))

    if band == 'H':
        line_cen = (1.56310 + 1.56317) / 2 * 1e4
        wvmin, wvmax = (line_cen-15, line_cen+15)
        wvmin = int(wvmin)
        wvmax = int(wvmax)
    else:
        sys.exit('Only H band is supported now...')

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
        # Tdiff_spot = int(config.get('star','Tdiff_spot')),
        Tdiff_faculae = int(config.get('star','Tdiff_faculae')),
        logg = float(config.get('star', 'logg' )),
        FeH = float(config.get('star', 'FeH' )),
        wvmin = wvmin,
        wvmax = wvmax
        )

    STAR.vsini = STAR.vrot * np.sin(np.deg2rad(STAR.incl))
    STAR.find_spec(config.get('star','syn_type'), RASSINE_PATH)

    # -- Read syn spectra/ccf --
    # For spectra, instead of wavelength, we transfer it into velocity 
    # center from selected lines, e.g., the OH lines in H band.

    rv_spec = STAR.star_spec['v']
    rv_spec_magn_region =  STAR.spot_spec['v']
    rv_spec_sampling = rv_spec[1]-rv_spec[0]

    intensity_spec = STAR.star_spec['f']
    intensity_spec_magn_region = STAR.spot_spec['f']

    SPEC = soap2.Spec(
        rv = rv_spec,
        intensity = intensity_spec,
        width = np.max(rv_spec),
        step = rv_spec_sampling,
        star = STAR
        )
                
    SPEC_active_region = soap2.Spec(
        rv = rv_spec_magn_region,
        intensity = intensity_spec_magn_region,
        width = np.max(rv_spec_magn_region),
        step = rv_spec_sampling,
        star = STAR
        )

    # -------------------------------------------------------------------
    # if STAR.vrot < 10.:
    #     file1 = './solar_CCFs/CCF_solar_spectrum_G2_FTS_reso_not_evenly_sampled_in_freq_higher_res.txt'
    # else:
    #     file1 = './solar_CCFs/CCF_solar_spectrum_G2_FTS_reso_not_evenly_sampled_in_freq_extra_large_low_resolution_higher_res.txt'

    # data_ccf = Table.read(file1, format='ascii')
    # rv_ccf = np.array(data_ccf['vrad'])
    # rv_ccf_magn_region = np.array(data_ccf['vrad'])
    # rv_ccf_sampling = rv_ccf[1]-rv_ccf[0]
    # intensity_ccf = np.array(data_ccf['CCF'])
    # rv_ccf_magn_region = np.array(data_ccf['vrad'])
    # intensity_ccf_magn_region = np.array(data_ccf['CCF_spot'])
                                
    # CCF = soap2.Ccf(
    #     rv = rv_ccf,
    #     intensity = intensity_ccf,
    #     width = max(rv_ccf),
    #     step = rv_ccf_sampling,
    #     star = STAR
    #     )
                
    # CCF_active_region = soap2.Ccf(
    #     rv = rv_ccf_magn_region,
    #     intensity = intensity_ccf_magn_region,
    #     width = max(rv_ccf_magn_region),
    #     step = rv_ccf_sampling,
    #     star = STAR
    #     )
    
    ######################################################################

    active_region1    = soap2.Active_region(
        check              = int(config.get('active_regions','check1')),
        long               = float(config.get('active_regions','long1')),
        lat                = float(config.get('active_regions','lat1')),
        size               = float(config.get('active_regions','size1')),
        active_region_type = int(config.get('active_regions','act_reg_type1'))
        )

    active_region2    = soap2.Active_region(
        check              = int(config.get('active_regions','check2')),
        long               = float(config.get('active_regions','long2')),
        lat                = float(config.get('active_regions','lat2')),
        size               = float(config.get('active_regions','size2')),
        active_region_type = int(config.get('active_regions','act_reg_type2'))
        )

    # active_region3    = soap2.Active_region(
    #             check              = int(config.get('active_regions','check3')),
    #             long               = float(config.get('active_regions','long3')),
    #             lat                = float(config.get('active_regions','lat3')),
    #             size               = float(config.get('active_regions','size3')),
    #             active_region_type = int(config.get('active_regions','act_reg_type3'))
    #             )

    # ACTIVE_REGIONS    = [active_region1, active_region2, active_region3]
    ACTIVE_REGIONS    = [active_region1]

    ######################################################################

    if len(ACTIVE_REGIONS) == 1:
        outdir = "./outputs/OHFeLDR_{}_P={:.2f}_i={:.2f}_T1={:1.0f}_DelT={:1.0f}_logg={:1.1f}_lon={:.1f}_lat={:.1f}_size={:.4f}" .format(
            STAR.spec_type, STAR.prot,STAR.incl,STAR.Temp,STAR.Tdiff_spot,STAR.logg,
            active_region1.long,
            active_region1.lat,
            active_region1.s
            )
    elif len(ACTIVE_REGIONS) == 2:
        outdir = "./outputs/OHFeLineDepth_{}P={:.2f}_i={:.2f}_tefT1f1={:1.0f}DelT={:1.0f}_logg={:1.1f}_lon=({:.1f},{:.1f})_lat=({:.1f},{:.1f})_size=({:.4f},{:.4f})" .format(
            STAR.spec_type, STAR.prot,STAR.incl,STAR.Temp,STAR.Tdiff_spot,STAR.logg,
            active_region1.long,active_region2.long,
            active_region1.lat,active_region2.lat,
            active_region1.s,active_region2.s
            )
    else:
        sys.exit('Only have output name build up to two regions...')

    if not os.path.exists('outputs'):
        os.mkdir('outputs')
        
    # if os.path.exists(outdir):
    #     shutil.rmtree(outdir)
    #     os.mkdir(outdir)
    # else:
    #     os.mkdir(outdir)   

######################################################################

    # calculate the visibility of the active regions
    active_region1.calc_maps(STAR, GRID, NRHO)
    # active_region2.calc_maps(STAR, GRID, NRHO)

    # Phase at which the signal will be generated
    phase_step  = float(config.get('output','ph_step'))
    phase_in    = config.get('output','ph_in')

    # Read the phase in the file given in the "phase_in" argument in "config.cfg"
    if phase_in!='None':
        f   = open(phase_in)
        PSI = np.array([float(l.split()[0]) for l in f.readlines()])
        f.close()
    # Otherwise create a evenly sampled phase between 0 and 1 with the step given
    # by the "ph_step" argument in "config.cfg"
    elif phase_step>0.:
        # PSI = np.arange(-0.2,1.2,phase_step)
        PSI = np.arange(0,1,phase_step)
    else:
        print("WARNING: put a positive phase step ('ph_step' argument in 'config.cfg') \
                or give a file name that contain the phases at which you want to compute the model \
                ('phase_in' argument in 'config.cfg')"
                )

    (visible_phi, SPECstar_tot, SPECstar_quiet, 
        quite_surface, quite_surface_2d) =\
        soap2.calculate_activity_signal_spec(
        NRHO, GRID, STAR, PSI, 
        SPEC, SPEC_active_region,
        ACTIVE_REGIONS, INST_RESO, CPU
        )
    
    depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, depth_l_fe2, depth_l_oh0 = soap2.get_line_depth(
        STAR, SPECstar_tot, visible_phi
        )

    oh_line = np.array(depth_d_oh0)
    fe_line = (np.array(depth_d_fe1) + np.array(depth_d_fe2)) / 2
    line_ratio = oh_line/fe_line

    print(f'LDR amp. {np.max(line_ratio) - np.min(line_ratio):1.4}')


    # use Fe OH lines
    ccf_rv_shift = soap2.cal_ccf_rv(SPECstar_tot, STAR, visible_phi)

    # np.save('test_SPECstar_tot', SPECstar_tot)
    # np.save('test_phase', visible_phi)
    # np.save('test_SPEC_template', SPECstar_tot[np.argmin(np.abs(visible_phi))])
    # np.save('test_v', np.array(STAR.star_spec['v']))
    # sys.exit()


    text1 =   '{0:12} = {1:<8d}, {2:18} = {3:<8d}\n'.format(    'Grid res.', GRID,          'Circumference res.', NRHO) + \
              '{0:12} = {1:<8d}, {2:18} = {3:<8d}\n'.format(    'Inst. res.', INST_RESO,    'Rsun [km]', RAD_Sun) + \
              '{0:12} = {1:<8.2f}, {2:18} = {3:<8.2f}\n'.format('St Prot [d]', STAR.prot,   'St radius [Rsun]', STAR.rad/RAD_Sun) + \
              '{0:12} = {1:<8.1f}, {2:18} = {3:<8.2f}\n'.format('St incl', STAR.incl,       'St vsini [km/s]', STAR.vsini) + \
              '{0:12} = {1:<8d}, {2:18} = {3:<8d}\n'.format(    'St Teff [K]', STAR.Temp,   'Spot Diff Teff [K]', STAR.Tdiff_spot) + \
              '{0:12} = {1:<8.3f}, {2:18} = {3:<8.3f}\n'.format(  'LimDark lin', STAR.limba1, 'LimDark quad', STAR.limba2) + \
              '{0:12} = {1:<8}, {2:5} = {3:<4.2f}, {4:5} = {5:<4.2f}'.format( 'Syn template', STAR.spec_type, 'logg', STAR.logg, '[Fe/H]', 0.0)

    text2 = 'ActR {0}: lon={1:<5.1f}, lat={2:<4.1f}, size = {3:<5.3f} [Rsun] (={4:.2f}%)\n'.format('1',active_region1.long,active_region1.lat,active_region1.s,active_region1.s**2/2.*100) +\
            'ActR {0}: lon={1:<5.1f}, lat={2:<4.1f}, size = {3:<5.3f} [Rsun] (={4:.2f}%)\n'.format('2',active_region2.long,active_region2.lat,active_region2.s,active_region2.s**2/2.*100)
            # 'Act Reg {0}: {1:<6}, lon = {2:<5.1f}, lat = {3:<4.1f}, size = {4:<5.3f} [Rsun] (= {5:.2f}%)\n'.format('3',active_region3.active_region_type_str,active_region3.long,active_region3.lat,active_region3.s,active_region3.s**2/2.*100) +\
            # 'Act Reg {0}: {1:<6}, lon = {2:<5.1f}, lat = {3:<4.1f}, size = {4:<5.3f} [Rsun] (= {5:.2f}%)'.format('4',active_region4.active_region_type_str,active_region4.long,active_region4.lat,active_region4.s,active_region4.s**2/2.*100)

    vis.spot_view(
        STAR, GRID, visible_phi, SPECstar_tot, ccf_rv_shift,
        quite_surface_2d, outdir, CPU, depth_d_fe1, depth_d_fe2, depth_d_oh0, 
        depth_l_fe1, depth_l_fe2, depth_l_oh0, text1, text2
        )
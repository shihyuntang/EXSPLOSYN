import os, sys, argparse
import shutil

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import matplotlib as mpl
import configparser
from astropy.table import Table
from numba import njit

import starspot as stsp
import soap2 as soap2
import visualization as vis

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.path as path
import matplotlib.patches as patches



if __name__ == '__main__':
    ####################################################################################
    #                    INITITIALIZATION                     ##########################
    ####################################################################################

    config = configparser.ConfigParser(inline_comment_prefixes=';')
    config.read("config_v830tau.cfg")

    GRID        = int(config.get('main','grid' ))
    NRHO        = int(config.get('main','nrho' ))
    INST_RESO   = int(config.get('main','instrument_reso' ))
    CPU         = int(config.get('main','cpu' ))
    RAD_Sun     = int(config.get('star','radius_sun' ))
    RAD         = float(config.get('star','radius' )) * RAD_Sun
    PROT        = float(config.get('star','prot' ))
    STAR        = soap2.Star(
                    prot           = PROT,
                    vrot           = (2.*np.pi*RAD)/(PROT*86400.),
                    incl           = float(config.get('star','I')),
                    limba1         = float(config.get('star','limb1')),
                    limba2         = float(config.get('star','limb2')),
                    psi            = float(config.get('star','psi')),
                    rad_sun        = RAD_Sun,
                    rad            = RAD,
                    Temp           = int(config.get('star','Tstar')),
                    Temp_diff_spot = int(config.get('star','Tdiff_spot'))
                    )

    STAR.find_spec(config.get('star','syn_type'))

    ######################################################################
    ###           Read solar CCF       ###################################
    ######################################################################

    # if STAR.vrot < 10.:
    #     file1 = 'solar_CCFs/CCF_solar_spectrum_G2_FTS_reso_not_evenly_sampled_in_freq.rdb'
    # else:
    #     file1 = 'solar_CCFs/CCF_solar_spectrum_G2_FTS_reso_not_evenly_sampled_in_freq_extra_large_low_resolution.rdb'

    # data_ccf = Table.read(file1, format='ascii')
    # rv_ccf = np.array(data_ccf['vrad'])
    # rv_ccf_magn_region = np.array(data_ccf['vrad'])
    # rv_ccf_sampling = rv_ccf[1]-rv_ccf[0]

    # intensity_ccf = np.array(data_ccf['CCF'])
    # rv_ccf_magn_region = np.array(data_ccf['vrad'])
    # intensity_ccf_magn_region = np.array(data_ccf['CCF_spot'])

    # CCF               = soap2.Ccf(rv        = rv_ccf,
    #                             intensity = intensity_ccf,
    #                             width     = np.max(rv_ccf),
    #                             step      = rv_ccf_sampling,
    #                             star      = STAR)
                
    # CCF_active_region = soap2.Ccf(rv        = rv_ccf_magn_region,
    #                             intensity = intensity_ccf_magn_region,
    #                             width     = np.max(rv_ccf_magn_region),
    #                             step      = rv_ccf_sampling,
    #                             star      = STAR)

    ######################################################################
    ###           Read syn spectra       #################################
    ######################################################################

    rv_spec = STAR.star_spec['v']
    rv_spec_magn_region =  STAR.spot_spec['v']
    rv_spec_sampling = rv_spec[1]-rv_spec[0]

    intensity_spec = STAR.star_spec['f']
    intensity_spec_magn_region = STAR.spot_spec['f']

    SPEC               = soap2.Spec(rv        = rv_spec,
                                intensity = intensity_spec,
                                width     = np.max(rv_spec),
                                step      = rv_spec_sampling,
                                star      = STAR)
                
    SPEC_active_region = soap2.Spec(rv        = rv_spec_magn_region,
                                intensity = intensity_spec_magn_region,
                                width     = np.max(rv_spec_magn_region),
                                step      = rv_spec_sampling,
                                star      = STAR)

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


    ACTIVE_REGIONS    = [active_region1]

    # calculate the visibility of the active regions
    active_region1.calc_maps(STAR, GRID, NRHO)
    active_region2.calc_maps(STAR, GRID, NRHO)

    # Phase at which the signal we be generated
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
        PSI = np.arange(0.,1.,phase_step)
    else:
        print("WARNING: put a positive phase step ('ph_step' argument in 'config.cfg') \
                or give a file name that contain the phases at which you want to compute the model \
                ('phase_in' argument in 'config.cfg')"
                )

    visible_phi, xyz2_box, CCFstar_tot, CCFstar_quiet = soap2.calculate_activity_signal_spec(
        NRHO, GRID, STAR, PSI, SPEC, SPEC_active_region, ACTIVE_REGIONS, INST_RESO, CPU
        )
    
    outdir = "./outputs/OHFeLineDepth_Prot={:.2f}_i={:.2f}_lon=({:.1f},{:.1f})_lat=({:.1f},{:.1f})_size=({:.4f},{:.4f})" .format(
        STAR.prot,STAR.incl,
        active_region1.long,active_region2.long,
        active_region1.lat,active_region2.lat,
        active_region1.s,active_region2.s
        )

    if os.path.exists(outdir):
        shutil.rmtree(outdir)
        os.mkdir(outdir)
    else:
        os.mkdir(outdir)   

    vis.spot_view(
        STAR, GRID, active_region1.xyz, visible_phi, xyz2_box, CCFstar_tot, 
        CCFstar_quiet, outdir, CPU
        )
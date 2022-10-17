import sys, os

import numpy as np
from numpy import cos, sin
from numba import njit
from scipy import signal
from astropy.table import Table

import starspot as stsp
import visualization as vis

class Star:
    def __init__(self, prot, vrot, incl, limba1, limba2, psi, rad_sun, rad, Temp, Temp_diff_spot):
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
        self.limba1         = limba1         
        self.limba2         = limba2         
        self.psi            = psi            
        self.rad_sun        = rad_sun        
        self.rad            = rad            
        self.Temp           = Temp           
        self.Temp_diff_spot = Temp_diff_spot 
        self.Temp_spot      = Temp-Temp_diff_spot
    
    def find_spec(self, spec_type):
        self.spec_type = spec_type

        spec_dir = f'./syn_spec'
        spec_list = os.listdir(spec_dir)
        spec_list = [i for i in spec_list if spec_type in i and '.pdf' not in i]

        teffs = np.array([ int(t.split('_')[-3]) for t in spec_list ])

        if self.Temp in teffs:
            self.star_teff_use = self.Temp
            self.star_spec = self.read_spec(self.star_teff_use)
            print(f'\n Using {self.star_teff_use}K template for the stellar photosphere')
        else:
            abs_differ = np.abs(self.Temp - teffs)
            _temptab = Table(
                data=(teffs, abs_differ), 
                names=['teffs', 'abs_differ']
                )
            _temptab.sort('abs_differ')
            temp1 = _temptab['teffs'][0]; diff1 = _temptab['abs_differ'][0]
            temp2 = _temptab['teffs'][1]; diff2 = _temptab['abs_differ'][1]

            ratio1 = 1 - (diff1/(diff1+diff2))
            ratio2 = 1 - (diff2/(diff1+diff2))

            spec1 = self.read_spec(temp1)
            spec2 = self.read_spec(temp2)

            spec = {'v': spec1['v'], 'f': ratio1*spec1['f']+ratio2*spec2['f']}
            self.star_spec = spec

            print(f'\nUsing {ratio1:1.1%} of {temp1}K template and {ratio2:1.1%} of {temp2}K template for the stellar photosphere')

        if self.Temp_spot in teffs:
            self.spot_teff_use = self.Temp_spot
            self.spot_spec = self.read_spec(self.spot_teff_use)
            print(f'Using {self.star_teff_use}K template for the spot spectrum')
        else:
            abs_differ = np.abs(self.Temp_spot - teffs)
            _temptab = Table(
                data=(teffs, abs_differ), 
                names=['teffs', 'abs_differ']
                )
            _temptab.sort('abs_differ')
            temp1 = _temptab['teffs'][0]; diff1 = _temptab['abs_differ'][0]
            temp2 = _temptab['teffs'][1]; diff2 = _temptab['abs_differ'][1]

            ratio1 = 1 - (diff1/(diff1+diff2))
            ratio2 = 1 - (diff2/(diff1+diff2))

            spec1 = self.read_spec(temp1)
            spec2 = self.read_spec(temp2)

            spec = {'v': spec1['v'], 'f': ratio1*spec1['f']+ratio2*spec2['f']}
            self.spot_spec = spec

            print(f'Using {ratio1:1.1%} of {temp1}K template and {ratio2:1.1%} of {temp2}K template for the spot spectrum')

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

    def read_spec(self, teff):
        file_star = f'./syn_spec/{self.spec_type}_for_SOAP_LineDepthRatio_{teff}_4.5_0.0.csv'
        t = Table.read(file_star, format='csv')

        t_dict = {}
        # t_dict['w'] = np.array(t['wave[AA]'])
        t_dict['v'] = np.array(t['vel[kms]'])
        t_dict['f'] = np.array(t['flux'])
        return t_dict



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

def compute_bis(RV, CCF):
    err = np.ones(len(CCF), dtype=np.float64)
    (mod,c,k,v0,fwhm,sig_c,sig_k,sig_v0,sig_fwhm,span,bis,depth) =\
        stsp.gauss_bis(RV, CCF, err)
    depth=-k
    return depth, bis, span, v0, fwhm


def get_line_depth(Star, CCFstar_tot, visible_phi):

    depth_d_fe1 = []
    depth_d_fe2 = []
    depth_d_oh0 = []
    depth_l_fe1 = []
    depth_l_fe2 = []
    depth_l_oh0 = []

    vel = Star.star_spec['v']

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
    Spec, Spec_active_region, Ccf, Ccf_active_region, 
    active_regions, inst_reso, cpu_use
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
    CCFstar_quiet, SPECstar_quiet, quite_surface = stsp.itot(
        star.vrot, star.incl, star.limba1, star.limba2, 0., 0., 0., grid, 
        Spec.rv, Spec.intensity, Spec.v_interval, Spec.n_v, Spec.n,
        Ccf.rv, Ccf.intensity, Ccf.v_interval, Ccf.n_v, Ccf.n
        )

    # initialization of the variables
    # FLUXstar      = FLUXstar_quiet
    # CCFstar_flux  = CCFstar_quiet
    # CCFstar_bconv = CCFstar_quiet
    CCFstar_tot   = CCFstar_quiet
    SPECstar_tot  = SPECstar_quiet

    #Loop over the different active regions
    print('\nCalculate Activate Regions...\n')
    for active_region in active_regions:
        # active_region.check = 1 if the active region is activated, 0 otherwise
        if not active_region.check: 
            continue 
        
        # Calculates the position of the spot initialized at the disc center
        xyz = stsp.spot_init(
            active_region.s, active_region.long, active_region.lat, star.incl, nrho
            )
        
        # Scans the yz-area where the spot is for different phases (psi) and
        # returns the spot's "non-contribution" to the total flux and its
        # "non-contribution" to the ccf, for each phase.
        # Thus the result is to be subtracted to the output of the itot() function.
        CCFactive_region_tot, SPECactive_region_tot, visible_phi, active_surface_box =\
                stsp.spot_scan_npsi(
                    xyz, nrho, psi, len(psi), star.vrot, star.incl, star.limba1, 
                    star.limba2, 0., 0., 0., grid, 
                    Spec.rv, Spec.intensity, Spec_active_region.intensity, 
                    Spec.v_interval, Spec.n_v, Spec.n, 
                    Ccf.rv, Ccf.intensity, Ccf_active_region.intensity, 
                    Ccf.v_interval, Ccf.n_v, Ccf.n, 
                    active_region.s, active_region.long, active_region.lat, 
                    active_region.active_region_type, star.Temp, star.Temp_diff_spot, cpu_use
                    )

        # Calculate the total flux of the star affected by active regions
        # FLUXstar      = FLUXstar-np.array(FLUXactive_region)
        
        # Calculate the CCF of the star affected by the flux effect of active regions
        # CCFstar_flux  = CCFstar_flux-np.array(CCFactive_region_flux)   
        
        # Calculate the CCF of the star affected by the convective blueshift effect of active regions
        # CCFstar_bconv = CCFstar_bconv-np.array(CCFactive_region_bconv) 
        
        # Calculate the CCF of the star affected by the total effect of active regions
        CCFstar_tot   = CCFstar_tot-np.array(CCFactive_region_tot)  
        SPECstar_tot   = SPECstar_tot-np.array(SPECactive_region_tot)  
        quite_surface_2d = quite_surface - np.array(active_surface_box)


    # Calculate where the extrapolated CCF corresponds to the boundaries of the non-extrapolated CCF.
    istart = (Ccf.n_v-len(Ccf.rv))//2   # ccf.n_v is odd given our definition, like len(ccf.rv), therefore 
                                        # the difference can be divided by 2. The CCF has been extrapolated 
                                        # the same way on each boundary, so dividing the difference between 
                                        # the extrapolated and non extrapolated CCF by 2 gives by how many 
                                        # points the CCF was extrapolated on each side
    iend   = istart+len(Ccf.rv)

    # truncate the extrapolated rotating CCF to the same interval as the non-rotating CCF
    # CCFstar_flux  = CCFstar_flux[:,istart:iend]
    # CCFstar_bconv = CCFstar_bconv[:,istart:iend]
    CCFstar_tot   = CCFstar_tot[:,istart:iend]
    CCFstar_quiet = CCFstar_quiet[istart:iend]

    # -- Spec ---
    istart = (Spec.n_v-len(Spec.rv))//2         
    iend   = istart+len(Spec.rv)
    SPECstar_tot   = SPECstar_tot[:,istart:iend]
    SPECstar_quiet = SPECstar_quiet[istart:iend]


    # Normalization, loop through diff phases
    # CCFstar_flux  = np.array([f/np.max(f) for f in CCFstar_flux], dtype=np.float64)
    # CCFstar_bconv = np.array([f/np.max(f) for f in CCFstar_bconv], dtype=np.float64)
    CCFstar_tot   = np.array([f/np.percentile(f,95) for f in CCFstar_tot], dtype=np.float64)
    CCFstar_quiet = CCFstar_quiet/np.percentile(CCFstar_quiet, 95)

    SPECstar_tot   = np.array([f/np.percentile(f,95) for f in SPECstar_tot], dtype=np.float64)
    SPECstar_quiet = SPECstar_quiet/np.percentile(SPECstar_quiet, 95)

    # Convolution with a Gaussian instrumental profile of a given resolution, 
    # given by the "instrument_reso" variable in the "config.cfg" file
    if inst_reso != 0:

        CCFstar_tot, CCFstar_quiet = _inst_broaden(inst_reso, Ccf.rv, CCFstar_tot, CCFstar_quiet)
        SPECstar_tot, SPECstar_quiet = _inst_broaden(inst_reso, Spec.rv, SPECstar_tot, SPECstar_quiet)
    

    return (visible_phi, CCFstar_tot, CCFstar_quiet, SPECstar_tot, SPECstar_quiet, 
                quite_surface, quite_surface_2d)


def _inst_broaden(inst_reso, rv, star_tot, star_quiet):
    c = 299792458. # speed of light in m/s
    inst_resolution         = inst_reso # resolution R = lambda / Delta(lambda)
    inst_inst_profile_FWHM  = c/inst_resolution/1000. # Resolution = c/Delta_v -> Delta_v = c/R
    inst_inst_profile_sigma = inst_inst_profile_FWHM/(2*np.sqrt(2*np.log(2)))
    Gaussian_low_reso        = np.exp(-rv**2/(2*(inst_inst_profile_sigma)**2))
    
    star_quiet_tmp = signal.convolve(
        -star_quiet+1, Gaussian_low_reso, 'same'
        )
    star_quiet = 1-star_quiet_tmp\
        *(1-np.min(star_quiet))/np.max(star_quiet_tmp) # normalization
    
    # Convolution of the CCF with the Gaussian instrumental profile to reduce 
    # the resolution of the CCF
    for i in np.arange(len(star_tot)):
        # CCFstar_flux_tmp  = signal.convolve(-CCFstar_flux[i]+1,Gaussian_low_reso,'same')
        # CCFstar_flux[i]   = 1-CCFstar_flux_tmp*(1-np.min(CCFstar_flux[i]))/np.max(CCFstar_flux_tmp)
        # CCFstar_bconv_tmp = signal.convolve(-CCFstar_bconv[i]+1,Gaussian_low_reso,'same')
        # CCFstar_bconv[i]  = 1-CCFstar_bconv_tmp*(1-np.min(CCFstar_bconv[i]))/np.max(CCFstar_bconv_tmp)
        star_tot_tmp   = signal.convolve(-star_tot[i]+1,Gaussian_low_reso,'same')
        star_tot[i]    = 1-star_tot_tmp*(1-np.min(star_tot[i]))/np.max(star_tot_tmp)

    return star_tot, star_quiet

# @njit
# def _cc(CCFstar_tot, vel, tw):
    
#     drvs = np.arange(-10, 10, 0.01)
#     spot_cc4rv = np.zeros(len(CCFstar_tot), dtype=np.float64)

#     for ii in range(len(CCFstar_tot)):
#         f = 1 - CCFstar_tot[ii]

#         tf = np.zeros(len(tw))
#         tf[tw==0] = np.max(f)

#         # Calculate the cross correlation
#         cc = np.zeros(len(drvs))
#         for i, rv in enumerate(drvs):
#             cc[i] = np.sum(f * np.interp(vel, tw+rv, tf))
            
#         spot_cc4rv[ii] = drvs[np.argmax(cc)]

#     return spot_cc4rv

# def cal_ccf_rv(CCFstar_tot, Ccf):

#     vel = Ccf.rv

#     exrtra_len = len(vel)//2
#     l_extra = np.zeros(exrtra_len) + CCFstar_tot[0][0]
#     r_extra = np.zeros(exrtra_len) + CCFstar_tot[0][-1]

#     # CCF_quiet = np.append(l_extra, CCFstar_quiet)
#     # CCF_quiet = np.append(CCF_quiet, r_extra)

#     del_vel = vel[1] - vel[0]
#     vel_temp = np.arange(
#         vel[0]-del_vel*exrtra_len, vel[-1]+del_vel*exrtra_len+del_vel, 
#         del_vel
#         )

#     del_vel = stsp.rndup(vel[1] - vel[0], 2)
#     vel_temp = np.arange(vel[0]-del_vel*exrtra_len, vel[-1]+del_vel*exrtra_len+del_vel, del_vel)
#     vel_temp = np.array([stsp.rndup(i, 2) for i in vel_temp])

#     ccf_rv_shift = _cc(CCFstar_tot, vel, vel_temp)

#     return ccf_rv_shift

@njit
# @njit(parallel=True)
def _cc(CCFstar_tot, vel, tw, tf, visible_phi):
    # the res for the vel is 25m/s, thus moving 
    # -10km to 10km is moving 400 pixles
    # on both sides.
    # drvs_pixels = np.arange(-400, 401)
    # drvs = drvs_pixels * 0.025

    drvs = np.arange(-5, 5, 0.03)
    spot_cc4rv = np.zeros(len(CCFstar_tot), dtype=np.float64)

    tf =  1-tf
    for ii in range(len(CCFstar_tot)):
        f =  1-CCFstar_tot[ii]

        # Calculate the cross correlation
        cc = np.zeros(len(drvs))
        for i, rv in enumerate(drvs):
            cc[i] = np.sum(f * np.interp(vel, tw+rv, tf))
            # tw_shifted = tw+rv
            # tf_shifted = tf[(tw_shifted >= vel[0]) & (tw_shifted <= vel[-1])]
            # cc[i] = np.sum(f * tf_shifted )
        cc = np.max(cc) - cc   

        # vis.plot_cc(drvs, cc, visible_phi[ii])
        # vis.plot_cc_spec(vel, f, tw, tf, visible_phi[ii], drvs[np.argmin(cc)])
        
        n_min = sum( cc == np.min(cc) )
        if n_min == 1:
            spot_cc4rv[ii] = drvs[np.argmin(cc)]
        else:
            drvs_temp = drvs[cc == np.min(cc)]
            spot_cc4rv[ii] = np.mean(drvs_temp)

    return spot_cc4rv


def cal_ccf_rv(star_tot, Star, visible_phi):

    vel = Star.star_spec['v']

    spec_temp = star_tot[np.argmin(np.abs(visible_phi))]

    print('Using Phase {:1.3f} as CCF template...'.format(
        visible_phi[np.argmin(np.abs(visible_phi))]
    ))


    exrtra_len = 1000
    del_vel = stsp.rndup(vel[1] - vel[0], 3)
    vel_temp = np.arange(
        vel[0]-del_vel*exrtra_len, vel[-1]+del_vel*exrtra_len+del_vel, 
        del_vel
        )
    # vel_temp = np.append(
    #     np.arange(vel[0]-exrtra_len*del_vel, vel[0], del_vel), vel
    #     )
    # vel_temp = np.append(
    #     vel_temp, np.arange(vel[-1], vel[-1]+exrtra_len*del_vel, del_vel)
    #     )
    vel_temp = np.array([stsp.rndup(i, 3) for i in vel_temp])

    spec_temp = np.append(np.zeros(exrtra_len)+spec_temp[0], spec_temp)
    spec_temp = np.append(spec_temp, np.zeros(exrtra_len)+spec_temp[-1])

    ccf_rv_shift = _cc(star_tot, vel, vel_temp, spec_temp, np.array(visible_phi))

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

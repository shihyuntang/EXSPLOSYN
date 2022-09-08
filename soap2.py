import sys, os

import numpy as np
from numpy import cos, sin
from numba import njit
from scipy import signal
from astropy.table import Table

import starspot as stsp


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
    
    def find_spec(self, spec_type):
        self.spec_type = spec_type

        spec_dir = f'./syn_spec'
        spec_list = os.listdir(spec_dir)
        spec_list = [i for i in spec_list if spec_type in i]

        teffs = np.array([ int(t.split('_')[-4]) for t in spec_list ])
        star_teff_use_loc = np.argmin(np.abs(self.Temp - teffs))
        spot_teff_use_loc = np.argmin(
            np.abs((self.Temp-self.Temp_diff_spot) - teffs)
            )
        self.star_teff_use = teffs[star_teff_use_loc]
        self.spot_teff_use = teffs[spot_teff_use_loc]

        self.star_spec = self.read_spec(self.star_teff_use)
        self.spot_spec = self.read_spec(self.spot_teff_use)

        print(
            '\n Using {}K template for the STAR & {}K template for the SPOT!\n'.format(
                self.star_teff_use, self.spot_teff_use)
                )

    def read_spec(self, teff):
        file_star = f'./syn_spec/{self.spec_type}_for_SOAP_LineDepthRatio_{teff}_4.5_0.0_NotEvenlySampled.csv'
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
    nrho, grid, star, psi, ccf, ccf_active_region, active_regions, inst_reso, cpu_use):

    # Calculates the flux and CCF in each cell of the grid and integrate
    # over the entire stellar disc to have the integrated flux (FLUXstar) and CCF (CCFstar).
    # Calculate the CCF (CCFstar) and the total flux (FLUXstar) for the quiet star
    CCFstar_quiet, FLUXstar_quiet, quite_surface = stsp.itot(
        star.vrot, star.incl, star.limba1, star.limba2, 0., 0., 0., grid, ccf.rv, 
        ccf.intensity, ccf.v_interval, ccf.n_v, ccf.n
        )

    # initialization of the variables
    # FLUXstar      = FLUXstar_quiet
    # CCFstar_flux  = CCFstar_quiet
    # CCFstar_bconv = CCFstar_quiet
    CCFstar_tot   = CCFstar_quiet

    #Loop over the different active regions
    print('Calculate Activate Regions...\n')
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
        CCFactive_region_tot, visible_phi, active_surface_box =\
                stsp.spot_scan_npsi(
                    xyz, nrho, psi, len(psi), star.vrot, star.incl, star.limba1, 
                    star.limba2, 0., 0., 0., grid, ccf.rv, ccf.intensity, 
                    ccf_active_region.intensity, ccf.v_interval, ccf.n_v, ccf.n, 
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
        quite_surface_2d = quite_surface - np.array(active_surface_box)


    # Calculate where the extrapolated CCF corresponds to the boundaries of the non-extrapolated CCF.
    istart = (ccf.n_v-len(ccf.rv))//2   # ccf.n_v is odd given our definition, like len(ccf.rv), therefore 
                                        # the difference can be divided by 2. The CCF has been extrapolated 
                                        # the same way on each boundary, so dividing the difference between 
                                        # the extrapolated and non extrapolated CCF by 2 gives by how many 
                                        # points the CCF was extrapolated on each side
    iend   = istart+len(ccf.rv)

    # truncate the extrapolated rotating CCF to the same interval as the non-rotating CCF
    # CCFstar_flux  = CCFstar_flux[:,istart:iend]
    # CCFstar_bconv = CCFstar_bconv[:,istart:iend]
    CCFstar_tot   = CCFstar_tot[:,istart:iend]
    CCFstar_quiet = CCFstar_quiet[istart:iend]

    # Normalization, loop through diff phases
    # CCFstar_flux  = np.array([f/np.max(f) for f in CCFstar_flux], dtype=np.float64)
    # CCFstar_bconv = np.array([f/np.max(f) for f in CCFstar_bconv], dtype=np.float64)
    CCFstar_tot   = np.array([f/np.percentile(f,95) for f in CCFstar_tot], dtype=np.float64)
    CCFstar_quiet = CCFstar_quiet/np.percentile(CCFstar_quiet, 95)
    
    # Convolution with a Gaussian instrumental profile of a given resolution, 
    # given by the "instrument_reso" variable in the "config.cfg" file
    if inst_reso != 0:
        
        c = 299792458. # speed of light in m/s
        HARPS_resolution         = inst_reso # resolution R = lambda / Delta(lambda)
        HARPS_inst_profile_FWHM  = c/HARPS_resolution/1000. # Resolution = c/Delta_v -> Delta_v = c/R
        HARPS_inst_profile_sigma = HARPS_inst_profile_FWHM/(2*np.sqrt(2*np.log(2)))
        Gaussian_low_reso        = np.exp(-ccf.rv**2/(2*(HARPS_inst_profile_sigma)**2))
        
        CCFstar_quiet_tmp = signal.convolve(
            -CCFstar_quiet+1, Gaussian_low_reso, 'same'
            )
        CCFstar_quiet = 1-CCFstar_quiet_tmp\
            *(1-np.min(CCFstar_quiet))/np.max(CCFstar_quiet_tmp) # normalization
        
        # Convolution of the CCF with the Gaussian instrumental profile to reduce 
        # the resolution of the CCF
        for i in np.arange(len(CCFstar_tot)):
            # CCFstar_flux_tmp  = signal.convolve(-CCFstar_flux[i]+1,Gaussian_low_reso,'same')
            # CCFstar_flux[i]   = 1-CCFstar_flux_tmp*(1-np.min(CCFstar_flux[i]))/np.max(CCFstar_flux_tmp)
            # CCFstar_bconv_tmp = signal.convolve(-CCFstar_bconv[i]+1,Gaussian_low_reso,'same')
            # CCFstar_bconv[i]  = 1-CCFstar_bconv_tmp*(1-np.min(CCFstar_bconv[i]))/np.max(CCFstar_bconv_tmp)
            CCFstar_tot_tmp   = signal.convolve(-CCFstar_tot[i]+1,Gaussian_low_reso,'same')
            CCFstar_tot[i]    = 1-CCFstar_tot_tmp*(1-np.min(CCFstar_tot[i]))/np.max(CCFstar_tot_tmp)

    # sys.exit('done')

    
    #return the variables
    return visible_phi, CCFstar_tot, CCFstar_quiet, quite_surface, quite_surface_2d



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

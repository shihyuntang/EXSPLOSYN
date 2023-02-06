"""
This .py is a modification of the origian starspot.c 
from the SOAP2 software

by Shih-Yun Tang 2022/09/04
"""
from functools import partial

import numpy as np
from numpy import cos, sin
from numba import njit
from scipy import signal
from pqdm.processes import pqdm
import pysynphot as pysynphot

import visualization as vis


@njit
def spot_init(s, longitude, latitude, inclination, nrho):
    """Calculate the coordinates (x,y,z) of points of the active region's 
    circumference

    Args:
        s (float): spot radius [Rstar]
        longitude (float): spot longitude [deg]
        latitude (float): spot latitude [deg]
        inclination (float): star inclination angl [deg] 
        nrho (int): resolution for the spot's circumference

    Returns:
        _type_: _description_
    """
    longitude   *= np.pi/180.
    latitude    *= np.pi/180.
    inclination *= np.pi/180.

    # A circular active region has a resolution given by nrho, which implies that we
    # will have a point on the disc circumference every 2*pi/(nrho-1). -1 because
    # there is (nrho-1) intervals
    rho_step = 2.*np.pi/(nrho-1)

    rho = [-np.pi + j*rho_step for j in range(nrho)] # rho goes from -pi to pi
    
    xyz2 = np.ones((3, nrho), dtype=np.float64)
    for j in range(nrho):
        # see Boisse+2012 fig. 1 for coord. setup
        xyz2[0][j] = np.sqrt(1-s*s)   # x=sqrt(r^2-s^2), where r==1
        xyz2[1][j] = s*cos(rho[j]) # projection of the circumference on the y axis
        xyz2[2][j] = s*sin(rho[j]) # projection of the circumference on the x axis

    # to account for the real projection of the spot, we rotate the star and look how the coordinates of the
    # circumference of the spot change
    # Position according to latitude, longitude and inclination. It consists of 
    # three rotations.
    #
    # Conventions :
    # -when inclination=0 the star rotates around z axis
    # -line of sight is along x-axis
    # -and sky plane = yz-plane
    #
    # Be Rx(alpha), Ry(beta), Rz(gamma) the rotations around the x, y and z axis
    # respectively, with angles alpha, beta and gamma (counter-clockwise 
    # direction when looking toward the origin).
    #
    # The rotations to apply are:
    #   Ry(inclination) x Rz(longitude) x Ry(latitude) x X(x,y,z)
    # 
    #         |  cos(b)  0  sin(b) |                       | cos(g) -sin(g) 0 |
    # Ry(b) = |    0     1    0    |               Rz(g) = | sin(g)  cos(g) 0 |
    #         | -sin(b)  0  cos(b) |                       |   0       0    1 |
    #
    #
    # |x'|   |  cos(b2)cos(g)cos(b)-sin(b)sin(b2)  -sin(g)cos(b2)  cos(b2)cos(g)sin(b)+sin(b2)cos(b) |   |x|
    # |y'| = |              sin(g)cos(b)               cos(g)                   sin(g)sin(b)         | x |y|
    # |z'|   |  -sin(b2)cos(g)cos(b)-cos(b2)sin(b)  sin(b2)sin(g) -sin(b2)cos(g)sin(b)+cos(b2)cos(b) |   |z|

    b = -latitude
    g = longitude
    b2 = np.pi/2. - inclination

    R = np.array([
        [  cos(b2)*cos(g)*cos(b)-sin(b)*sin(b2), -sin(g)*cos(b2),  cos(b2)*cos(g)*sin(b)+sin(b2)*cos(b) ], 
        [  sin(g)*cos(b),                         cos(g),          sin(g)*sin(b)                        ],
        [ -sin(b2)*cos(g)*cos(b)-cos(b2)*sin(b),  sin(b2)*sin(g), -sin(b2)*cos(g)*sin(b)+cos(b2)*cos(b) ]
        ])

    return R@xyz2

@njit
def spot_inverse_rotation(xyz, longitude, latitude, inclination, phase):
    """Relocate a point (x,y,z) to the 'initial' configuration
    i.e., when the active region is on the disc center
    Thus it consists of rotating the point, according to latitude, longitude,
    inclination and phase, but in the reverse order.

    Conventions :
    - when inclination=0 the star rotates around z axis (i.e. rotation axis and
      z axis are indistinct), 
    - line of sight is along x-axis
    - and sky plane = yz-plane

    Args:
        xyz (_type_): _description_
        longitude (_type_): _description_
        latitude (_type_): _description_
        inclination (_type_): _description_
        phase (_type_): _description_
        xiyizi (_type_): _description_

    Returns:
        _type_: _description_
    """

    # g2 = --phase*(2*pi);  // inverse phase ([0-1] -> [rad])
    g2 = 2*np.pi*(phase-1)
    # g2 = -(-phase*(2*np.pi))
    i   = inclination * np.pi/180.
    
    b  =  latitude  * np.pi/180.
    g  = -longitude * np.pi/180.
    b2 = -(np.pi/2.-i)

    R1 = np.array([
        [ (1-cos(g2))*cos(i)*cos(i) + cos(g2),  sin(g2)*sin(i),   (1-cos(g2))*cos(i)*sin(i)           ],
        [ -sin(g2)*sin(i),                      cos(g2),           sin(g2)*cos(i)                     ],
        [ (1-cos(g2))*sin(i)*cos(i),           -sin(g2)*cos(i),   (1-cos(g2))*sin(i)*sin(i) + cos(g2) ]
        ])
    
    R2 = np.array([
        [  cos(b)*cos(g)*cos(b2)-sin(b2)*sin(b), -sin(g)*cos(b),   cos(b)*cos(g)*sin(b2)+sin(b)*cos(b2) ],
        [  sin(g)*cos(b2),                        cos(g),          sin(g)*sin(b2)                       ],
        [ -sin(b)*cos(g)*cos(b2)-cos(b)*sin(b2),  sin(b)*sin(g),  -sin(b)*cos(g)*sin(b2)+cos(b)*cos(b2) ]
    ])

    R3 = R2@R1

    return R3@xyz



@njit
def spot_phase(xyz, inclination, nrho, phase):

    # the phase is between 0 and 1, so psi is in radian between -2pi and 0.
    psi = -phase*(2*np.pi) 
    inclination = inclination * np.pi/180.

    axe = np.array([cos(inclination), 0, sin(inclination)])

    R = np.array([
        [(1-cos(psi))*axe[0]*axe[0] + cos(psi), 
         (1-cos(psi))*axe[0]*axe[1] + sin(psi)*axe[2],
         (1-cos(psi))*axe[0]*axe[2] - sin(psi)*axe[1]],
        [(1-cos(psi))*axe[1]*axe[0] - sin(psi)*axe[2],
         (1-cos(psi))*axe[1]*axe[1] + cos(psi),
         (1-cos(psi))*axe[1]*axe[2] + sin(psi)*axe[0]],
        [(1-cos(psi))*axe[2]*axe[0] + sin(psi)*axe[1],
         (1-cos(psi))*axe[2]*axe[1] - sin(psi)*axe[0],
         (1-cos(psi))*axe[2]*axe[2] + cos(psi)]
    ])
    return R@xyz


@njit 
def spot_area(xlylzl, nrho, grid):
    visible = 0
    grid_step = 2./grid # The stellar disc goes from -1 to 1, therefore 2
    miny=1; minz=1; maxy=-1; maxz=-1 # init to 'opposite'-extreme values

    # count how many points of the circumference are visible and how 
    # many are invisible
    counton=0; countoff=0

    for j in range(nrho):
        if xlylzl[0][j]>=0:
            # if x>=0
            counton += 1

            # select the extreme points of the circumference
            if xlylzl[1][j]<miny: miny = xlylzl[1][j]
            if xlylzl[2][j]<minz: minz = xlylzl[2][j] 
            if xlylzl[1][j]>maxy: maxy = xlylzl[1][j]
            if xlylzl[2][j]>maxz: maxz = xlylzl[2][j] 
        else: countoff = 1

    if (counton>0) & (countoff>0):
        # There are both visible and invisible points
        # --> active region is on the edge
        # In this situation there are cases where the yz-area define above is 
        # actually smaller than the real area of the active region on the stellar disk.
        # The minima/maxima are over/under-estimated if the active region is on one 
        # of the axis (y or z). Because if on the y axis, the minimum (or maximum) 
        # won't be on the circumference of the active region. Same for z axis
        if miny*maxy<0:       
            # active region on the z-axis because one point is on the positive side of z, and the other on the negative side of z
            if minz<0: 
                if np.abs(minz)>=np.abs(maxz):
                    minz = -1
                else:
                    maxz = 1
            else: maxz=1
            
            # active region on the bottom-z axis (z<0)
            # active region on the top-z axis (z>=0)
        
        if minz*maxz<0:       
            # active region on the y-axis because one point is on the positive side of y, and the other on the negative side of z
            if miny<0: 
                if np.abs(miny)>=np.abs(maxy):
                    miny = -1
                else:
                    maxy = 1
            else: maxy=1

            # if miny<0: miny=-1
            # else: maxy=1
            # active region on the left hand-y axis (y<0)
            # active region on the right hand-y axis (y>=0)
    visible = 0 if counton==0 else 1


    # Indices of miny, minz,... on the grid
    iminy = np.floor( (1.+miny)/grid_step ) # floor(x) returns the largest integral value that is not greater than x.
                                            # floor of 2.3 is 2.0, floor of 3.8 is 3.0, floor of -2.3 is -3.0, floor of -3.8 is -4.0
    iminz = np.floor( (1.+minz)/grid_step )
    imaxy = np.ceil( (1.+maxy)/grid_step )  # ceil(x) returns the smallest integral value that is not less than x.
                                            # ceil of 2.3 is 3, ceil of 3.8 is 4.0, ceil of -2.3 is -2.0, ceil of -3.8 is -3.0
    imaxz = np.ceil( (1.+maxz)/grid_step )

    return visible, int(iminy), int(iminz), int(imaxy), int(imaxz)

@njit
def rndup(n, nb_decimal):
    """round up float numbers, for nb_decimal of 0 e.g.,
    1.5 --> 2.; 1.4 --> 1.

    Args:
        n (float):
        nb_decimal (ing):

    Returns:
        float: rounded number
    """

    t = n*10**(nb_decimal) - np.floor(n*10**(nb_decimal))
    if t>=0.5:
        n*=10**(nb_decimal)
        n = np.ceil(n)
        return n/10**(nb_decimal)
    else:
        n*=10**(nb_decimal)
        n = np.floor(n)
        return n/10**(nb_decimal) 

@njit
def shifting_CCF(_lambda, CCF, v_shift, n1):

    CCF_derivative = np.zeros(n1, dtype=np.float64)
    CCF_blueshifted = np.zeros(n1, dtype=np.float64)

    # Calculates the CCF derivative
    if (_lambda[1]-_lambda[0]) == 0:
        CCF_derivative[0] = 0
    else:
        CCF_derivative[0] = (CCF[1] - CCF[0])/(_lambda[1]-_lambda[0])

    for i in range(1, n1-1):
        if (_lambda[i+1]-_lambda[i]) == 0:
            CCF_derivative[i] = CCF_derivative[i-1]
        else:
            CCF_derivative[i] = (CCF[i+1] - CCF[i])/(_lambda[i+1]-_lambda[i])   

    CCF_derivative[n1] = 0.
    
    for i in range(n1):
        if (v_shift >= 0) | (i == 0) :
            # dy/dx = derivative => dy = derivative*dx
            CCF_blueshifted[i] = CCF[i] - v_shift * CCF_derivative[i] 
        elif v_shift < 0:
            CCF_blueshifted[i] = CCF[i] - v_shift * CCF_derivative[i-1]

    return CCF_blueshifted


@njit
def itot(v, incl, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte, 
    grid, vrad_spec, intensity_spec, v_interval_spec, n_v_spec, n_spec):
    """Calculates the flux, spec, and CCF in each cell of the grid and integrate
    over the entire stellar disc to have the integrated flux, spec, and CCF
    """

    f_star_spec = np.zeros(n_v_spec, dtype=np.float64)
    # f_star_ccf = np.zeros(n_v_ccf, dtype=np.float64)
    # sum_star = 0

    delta_grid = 2./grid # step of the grid. grid goes from -1 to 1, therefore 2 in total

    i = np.deg2rad(incl)

    quite_surface = np.zeros((grid, grid), dtype=np.float64) # for 2d disk plotting

    # v_interval is from the velocity 0 to the edge of the spectrum taking into 
    # account minimal or maximal rotation (width - v to 0 or 0 to width + v). 
    # n_v is the number of points for all the CCF from minimum rotation to 
    # maximum one (from width - v to width + v).
    # n_v represent therefore the double than v_interval, we therefore have to multiply 
    # v_interval by 2.
    delta_v_spec = 2*v_interval_spec/(n_v_spec-1) # step in speed of the CCF. There is (n_v-1) intervals
    # delta_v_ccf = 2*v_interval_ccf/(n_v_ccf-1) # step in speed of the CCF. There is (n_v-1) intervals

    # Scan of each cell on the grid
    for iy in range(grid+1):
        y = -1. + iy*delta_grid; # y between -1 et 1
        delta = y * v * sin(i) # Give the velocity of the rotation for the given grid 
                                   # cell as a function of y. For y=-1 => Vrot min, 
                                   # for y=1 => Vrot max, and for y=0 => Vrot = 0.
        
        for iz in range(grid+1):
            f_star_once = np.zeros(n_v_spec, dtype=np.float64)

            z = -1. + iz*delta_grid # z between -1 et 1
            if (y*y + z*z)<=1: 
                # projected radius on the sky smaller than 1, which means that 
                # we are on the stellar disc

                # limb-darkening
                # sqrt(r^2-(y^2+z^2)) where r=1. This is equal to 1 at the disc center, 
                # and 0 on the limb. This is often referred in the literature as cos(theta)
                r_cos = np.sqrt( 1 - (y*y+z*z) )  

                # intensity due to limb-darkening (Mandel & Agol 2002)
                limb = 1 - limba1*(1-r_cos) - limba2*(1-r_cos)**2 
                
                ##### modify SOAP2 code from only cal CCF to also cal SPEC #####
                # -------------------------- CCF ------------------------------
                # # The CCF is defined between -20 and +20 km/s with a sampling "delta_v" of 0.1 km/s, 
                # # and with boundaries normalized at 1. To account for stellar rotation in each cell of 
                # # the simulation, the CCF will be shifted depending on the position of the cell on the 
                # # stellar disc. To shift the CCF by a RV of x km/s, we will first shift the CCF by 
                # # "n_v_shifted_quotient" = int(x/sampling) steps, and then interpolate the CCF with 
                # # the remaining velocity "n_v_shifted_remainder" to have the correct shift of x km/s 
                # # at the end.
                
                # # This will be done by shifting the CCF by an integer of steps and then interpolating 
                # # the CCF to match the correct velocity
                # # The sampling of the CCF is every 100 m/s, so 2 km/s corresponds to a shift of 20 steps.
                # n_v_shifted = delta/delta_v_ccf        # by how much steps the CCF is shifted due to rotation
                # n_v_shifted_quotient = rndup(n_v_shifted, 0)
                # n_v_shifted_remainder = (delta - n_v_shifted_quotient*delta_v_ccf)
                # v_shift = n_v_shifted_remainder
                
                # # shifting the CCF with the remainder of n_v_shifted, the quotient will be taken into 
                # # account by shifting all the points of the spectrum
                # intensity_ccf_shift = shifting_CCF(
                #     vrad_ccf, intensity_ccf, v_shift, n_ccf
                #     )
                
                # # difference in number of steps between the CCF without any rotation and the one with rotation
                # diff_CCF_non_v_and_v = (n_v_ccf - n_ccf) / 2.

                # # To take into account the rotation, we increase the width of the CCF. Because the original CCF 
                # # is only defined between -20 and +20 km/s, we have to extrapolate on each side of the boundaries 
                # # of the CCF

                # for j in range(int(diff_CCF_non_v_and_v+n_v_shifted_quotient)):
                #     # extrapolation on the left of the CCF with the value of the left boundary, weighted 
                #     # by the limb-darkening
                #     f_star_ccf[j] += intensity_ccf_shift[0]*limb
                
                # for j in range(
                #     int(diff_CCF_non_v_and_v+n_v_shifted_quotient), 
                #     int(n_ccf+(diff_CCF_non_v_and_v+n_v_shifted_quotient))
                #     ):
                #     # value of the CCF, weighted by the limb-darkening
                #     f_star_ccf[j] += \
                #         intensity_ccf_shift[int(j-(diff_CCF_non_v_and_v+n_v_shifted_quotient))]*limb
                # for j in range(
                #     int(n_ccf+(diff_CCF_non_v_and_v+n_v_shifted_quotient)), 
                #     n_v_ccf
                #     ):
                #     # extrapolation on the right of the CCF with the value of the right boundary, 
                #     # weighted by the limb-darkening
                #     f_star_ccf[j] += intensity_ccf_shift[n_ccf-1]*limb
                
                # -------------------------- SPEC ------------------------------
                n_v_shifted = delta/delta_v_spec        # by how much steps the CCF is shifted due to rotation
                n_v_shifted_quotient = rndup(n_v_shifted, 0)
                n_v_shifted_remainder = (delta - n_v_shifted_quotient*delta_v_spec)
                v_shift = n_v_shifted_remainder
                
                intensity_spec_shift = shifting_CCF(
                    vrad_spec, intensity_spec, v_shift, n_spec
                    )
                
                diff_SPEC_non_v_and_v = (n_v_spec - n_spec) / 2.

                for j in range(int(diff_SPEC_non_v_and_v+n_v_shifted_quotient)):
                    f_star_spec[j] += intensity_spec_shift[0]*limb
                    f_star_once[j] += intensity_spec_shift[0]*limb
                
                for j in range(
                    int(diff_SPEC_non_v_and_v+n_v_shifted_quotient), 
                    int(n_spec+(diff_SPEC_non_v_and_v+n_v_shifted_quotient))
                    ):
                    # value of the CCF, weighted by the limb-darkening
                    f_star_spec[j] += \
                        intensity_spec_shift[int(j-(diff_SPEC_non_v_and_v+n_v_shifted_quotient))]*limb
                    f_star_once[j] += \
                        intensity_spec_shift[int(j-(diff_SPEC_non_v_and_v+n_v_shifted_quotient))]*limb

                for j in range(
                    int(n_spec+(diff_SPEC_non_v_and_v+n_v_shifted_quotient)), 
                    n_v_spec
                    ):
                    # extrapolation on the right of the CCF with the value of the right boundary, 
                    # weighted by the limb-darkening
                    f_star_spec[j] += intensity_spec_shift[n_spec-1]*limb
                    f_star_once[j] += intensity_spec_shift[n_spec-1]*limb
                

                # sum_star += intensity_spec_shift[0]*limb
                quite_surface[iz][iy] = np.sum(f_star_once)

    return f_star_spec, quite_surface

# @njit
# def loi_Planck(lambda0, Temp):
#     c   = 299792458.;     # speed of light m/s
#     h   = 6.62606896e-34; # Planck constant
#     k_b = 1.380e-23;      # Boltzmann constant
    
#     return 2*h*c**2*1./lambda0**5*1./(np.exp((h*c)/(lambda0*k_b*Temp))-1)

@njit
def spot_scan(
    v, i, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte, grid, 
    vrad_spec, intensity_spec, intensity_spec_spot, v_interval_spec, n_v_spec, n_spec, 
    s, longitude, phase, latitude, iminy, iminz, imaxy, imaxz, 
    magn_feature_type, T_star, T_diff_spot, intensity
    ):
    """Scan of the yz-area where the spot is.
    For each grid-point (y,z) we need to check whether it belongs to the spot
    or not. Sadly, we do not know the projected geometry of the spot in its
    actual position. Thus, we have to do an inverse rotation to replace the
    grid point where it would be in the initial configuration. Indeed, in the
    initial configuration, the spot has a well known geometry of a circle
    centered on the x-axis.f
    """

    delta_grid = 2./grid
    # delta_v_ccf = 2.*v_interval_ccf/(n_v_ccf-1)
    delta_v_spec = 2.*v_interval_spec/(n_v_spec-1)

    xayaza = np.ones(3, dtype=np.float64) # actual coordinates
    xiyizi = np.ones(3, dtype=np.float64) # coordinates transformed back to the initial configuration

    # f_spot_ccf_tot = np.zeros(n_v_ccf, dtype=np.float64)
    f_spot_spec_tot = np.zeros(n_v_spec, dtype=np.float64)

    active_surface = np.zeros((grid, grid), dtype=np.float64)

    # the wavelength of the Kitt peak spectrum goes from 3921.2441+6665.5789, the mean being 5293.4115
    # IGRINS H band OH line center 15631.3246 A
    # use_wave_center = 15631.3246e-10
    # loi_Planck_star = loi_Planck(use_wave_center, T_star) 

    # Scan of each cell on the grid
    for iy in range(iminy, imaxy):
        y = -1. + iy*delta_grid; # y between -1 et 1
        delta = y * v * sin(i*np.pi/180.)     # Give the velocity of the rotation for the given grid 
                                   # cell as a function of y. For y=-1 => Vrot min, 
                                   # for y=1 => Vrot max, and for y=0 => Vrot = 0.
        xayaza[1] = y

        for iz in range(iminz, imaxz):

            z = -1. + iz*delta_grid # z between -1 et 1
            if (y*y + z*z)<1: 
                xayaza[0] = np.sqrt(1.-(y*y+z*z)) # sqrt(r^2-(y^2+z^2)) where r=1. This is equal to 1 at the disc center, 
                                                  # and 0 on the limb. This is often referred in the literature as cos(theta)
                xayaza[2] = z

                # xayaza --> xiyizi: Rotate the star so that the spot is on the disc center
                xiyizi = spot_inverse_rotation(xayaza, longitude, latitude, i, phase)    

                # if inside the active region when scanning the grid
                if xiyizi[0]**2 >= (1.-s*s):

                    f_star_once = np.zeros(n_v_spec, dtype=np.float64)
                    # x^2 >= 1-sspot_scan_npsi^2, which means that you are inside the active region      
                    
                    # limb-darkening
                    r_cos = np.sqrt( 1-(y*y+z*z) ) # sqrt(r^2-(y^2+z^2)) where r=1. This is equal to 1 at the disc center, 
                                                  # and 0 on the limb. This is often referred in the literature as cos(theta)
                    limb = 1. - limba1*(1-r_cos) - limba2*(1-r_cos)**2
                    
                    # intensity of the spot (magn_feature_type==0) or the plage (magn_feature_type==1)
                    # if magn_feature_type==0:
                    #     T_spot = T_star-T_diff_spot
                    #     intensity = loi_Planck(use_wave_center,T_spot)/loi_Planck_star
                    # else:
                    #     T_plage = T_star+250.9-407.7*r_cos+190.9*r_cos**2 # plages are brighter on the limb Meunier 2010
                    #     intensity = loi_Planck(use_wave_center,T_plage)/loi_Planck_star
                    
                    ##### modify SOAP2 code from only cal CCF to also cal SPEC #####
                    # -------------------------- CCF ------------------------------
                    # n_v_shifted = delta/delta_v_ccf # by how much steps the CCF is shifted due to rotation
                    # n_v_shifted_quotient = rndup(n_v_shifted, 0) # integer number of steps
                    # n_v_shifted_remainder = (delta - n_v_shifted_quotient*delta_v_ccf)
                    # v_shift = n_v_shifted_remainder

                    # # shifting the CCF with the remainder of n_v_shifted, the quotient will be taken into account by 
                    # # shifting all the points of the spectrum
                    # intensity_ccf_shift = shifting_CCF(
                    #     vrad_ccf, intensity_ccf, v_shift, n_ccf
                    #     )
                    # intensity_ccf_spot_shift = shifting_CCF(
                    #     vrad_ccf, intensity_ccf_spot, v_shift, n_ccf
                    #     )
                    
                    # # difference in number of step between the CCF without any rotation and the one with rotation
                    # diff_CCF_non_v_and_v = (n_v_ccf - n_ccf) / 2.

                    # for j in range(int(diff_CCF_non_v_and_v+n_v_shifted_quotient)):
                    #     # extrapolation on the left of the CCF with the value of the left boundary, weighted by the 
                    #     # limb-darkening and the active region intensity. We also consider limb-darkening for spots 
                    #     # because we also observe them with different stellar depth depending on their position
                    #     # f_spot_flux[j]  += intensity_ccf_shift[0]*limb*(1 - intensity)  # only flux effect
                    #     # f_spot_bconv[j] += intensity_ccf_shift[0]*limb*(1 - 1);         # only convective blueshift effect. 
                    #     #                                                                 # The convective blueshift does not affect the boundaries of the CCF
                    #     f_spot_ccf_tot[j]   += intensity_ccf_shift[0]*limb*(1 - intensity)  # combined effect
                    # for j in range(
                    #     int(diff_CCF_non_v_and_v+n_v_shifted_quotient),
                    #     int(n_ccf+(diff_CCF_non_v_and_v+n_v_shifted_quotient))
                    #     ):
                    #     # value of the CCF, weighted by the limb-darkening and the active region intensity
                    #     # We also consider limb-darkening for spots because we also observe them with different stellar 
                    #     # depth depending on their position
                    #     # f_spot_flux[j] += \
                    #     #     intensity_ccf_shift[int(j-(diff_CCF_non_v_and_v+n_v_shifted_quotient))]*limb \
                    #     #         - intensity * (intensity_ccf_shift[int(j-(diff_CCF_non_v_and_v+n_v_shifted_quotient))])*limb
                    #     # f_spot_bconv[j] += \
                    #     #     intensity_ccf_shift[int(j-(diff_CCF_non_v_and_v+n_v_shifted_quotient))]*limb \
                    #     #         - (intensity_ccf_spot_shift[int(j-(diff_CCF_non_v_and_v+n_v_shifted_quotient))])*limb
                    #     f_spot_ccf_tot[j] += \
                    #         intensity_ccf_shift[int(j-(diff_CCF_non_v_and_v+n_v_shifted_quotient))]*limb \
                    #             - intensity * (intensity_ccf_spot_shift[int(j-(diff_CCF_non_v_and_v+n_v_shifted_quotient))])*limb
 
                    # for j in range(
                    #     int(n_ccf+(diff_CCF_non_v_and_v+n_v_shifted_quotient)),
                    #     n_v_ccf
                    #     ):
                    #     # extrapolation on the right of the CCF with the value of the right boundary, weighted by 
                    #     # the limb-darkening and the active region intensity. We also consider limb-darkening for spots 
                    #     # because we also observe them with different stellar depth depending on their position
                    #     # f_spot_flux[j]  += intensity_ccf_shift[n-1]*limb*(1 - intensity)
                    #     # f_spot_bconv[j] += intensity_ccf_shift[n-1]*limb*(1 - 1)        
                    #     f_spot_ccf_tot[j]   += intensity_ccf_shift[n_ccf-1]*limb * (1-intensity)

                    # -------------------------- SPEC -----------------------------
                    n_v_shifted = delta/delta_v_spec # by how much steps the CCF is shifted due to rotation
                    n_v_shifted_quotient = rndup(n_v_shifted, 0) # integer number of steps
                    n_v_shifted_remainder = (delta - n_v_shifted_quotient*delta_v_spec)
                    v_shift = n_v_shifted_remainder

                    intensity_spec_shift = shifting_CCF(
                        vrad_spec, intensity_spec, v_shift, n_spec
                        )
                    intensity_spec_spot_shift = shifting_CCF(
                        vrad_spec, intensity_spec_spot, v_shift, n_spec
                        )
                    
                    # difference in number of step between the CCF without any rotation and the one with rotation
                    diff_SPEC_non_v_and_v = (n_v_spec - n_spec) / 2.

                    for j in range(int(diff_SPEC_non_v_and_v+n_v_shifted_quotient)):
                        f_spot_spec_tot[j]   += intensity_spec_shift[0]*limb*(1 - intensity)  # combined effect
                        f_star_once[j]   += intensity_spec_shift[0]*limb*(1 - intensity)  # combined effect
                    for j in range(
                        int(diff_SPEC_non_v_and_v+n_v_shifted_quotient),
                        int(n_spec+(diff_SPEC_non_v_and_v+n_v_shifted_quotient))
                        ):
                        f_spot_spec_tot[j] += \
                            intensity_spec_shift[int(j-(diff_SPEC_non_v_and_v+n_v_shifted_quotient))]*limb \
                                - intensity * (intensity_spec_spot_shift[int(j-(diff_SPEC_non_v_and_v+n_v_shifted_quotient))])*limb
                        f_star_once[j] += \
                            intensity_spec_shift[int(j-(diff_SPEC_non_v_and_v+n_v_shifted_quotient))]*limb \
                                - intensity * (intensity_spec_spot_shift[int(j-(diff_SPEC_non_v_and_v+n_v_shifted_quotient))])*limb
                    
                    for j in range(
                        int(n_spec+(diff_SPEC_non_v_and_v+n_v_shifted_quotient)),
                        n_v_spec
                        ):      
                        f_spot_spec_tot[j]   += intensity_spec_shift[n_spec-1]*limb * (1-intensity)
                        f_star_once[j]   += intensity_spec_shift[n_spec-1]*limb * (1-intensity)
                    
                    active_surface[iz][iy] = np.sum(f_star_once)

    return f_spot_spec_tot, active_surface

def spot_scan_npsi_pal(
    xyz, inclination, nrho, psi, grid, v, limba1, limba2, 
    modif_bis_quad, modif_bis_lin, modif_bis_cte, 
    vrad_spec, intensity_spec, intensity_spec_spot, v_interval_spec, n_v_spec, n_spec, 
    s, longitude, latitude, magn_feature_type, T_star, T_diff_spot, intensity, ipsi
    ):

    xyz2 = spot_phase(xyz, inclination, nrho, psi[ipsi])
    visual, iminy, iminz, imaxy, imaxz = spot_area(xyz2, nrho, grid)
    
    if visual==1:
        f_spec_spot_tot, active_surface =\
            spot_scan(
                v, inclination, limba1, limba2, modif_bis_quad, 
                modif_bis_lin, modif_bis_cte, grid, 
                vrad_spec, intensity_spec, intensity_spec_spot, 
                v_interval_spec, n_v_spec, n_spec,  
                s, longitude, psi[ipsi], latitude, 
                iminy, iminz, imaxy, imaxz,
                magn_feature_type, T_star, T_diff_spot, intensity
            )
    else:
        # f_ccf_spot_tot = np.zeros(n_v_ccf, dtype=np.float64)
        f_spec_spot_tot = np.zeros(n_v_spec, dtype=np.float64)
        active_surface = np.zeros((grid, grid), dtype=np.float64)

    return f_spec_spot_tot, psi[ipsi], active_surface

def _get_intensity_ratio(T_star, T_diff_spot):
    # IGRINS H band OH line center 15631.3246 A
    # query phx syn model and intergral \pm 0.02 
    # micron reigon's flux from the OH line
    
    use_wave_center_m = 15631.3246e-10 #m
    use_wave_center_mu = 15631.3246e-10*1e6 #mu
    use_wave_range_mu = [use_wave_center_mu-0.02, use_wave_center_mu+0.02]

    # --- for photosphere (star) ---
    sp = pysynphot.Icat('phoenix', T_star, 0.0, 4.0)
    sp.convert('um')
    
    inter_range = (sp.wave>=use_wave_range_mu[0]) & (sp.wave<=use_wave_range_mu[1])
    inter_wave = sp.wave[inter_range]
    inter_flux = sp.flux[inter_range]
    phx_star_flux = np.trapz(inter_flux, inter_wave)

    # --- for spot ---
    sp = pysynphot.Icat('phoenix', T_star-T_diff_spot, 0.0, 4.0)
    sp.convert('um')
    
    inter_range = (sp.wave>=use_wave_range_mu[0]) & (sp.wave<=use_wave_range_mu[1])
    inter_wave = sp.wave[inter_range]
    inter_flux = sp.flux[inter_range]
    phx_spot_flux = np.trapz(inter_flux, inter_wave)

    return phx_spot_flux/phx_star_flux



def spot_scan_npsi(
    xyz, nrho, psi, npsi, v, inclination, limba1, limba2, modif_bis_quad, 
    modif_bis_lin, modif_bis_cte, grid, 
    vrad_spec, intensity_spec, intensity_spec_spot, v_interval_spec, n_v_spec, n_spec, 
    s, longitude, latitude, magn_feature_type, T_star, T_diff_spot, cpu_use
    ):
    """Scans the yz-area where the spot is for different phases (psi) and        
    returns the spot's "non-contribution" to the total flux and its           
    "non-contribution" to the ccf, for each phase.
    Thus the result is to be subtracted to the output of the itot() function.
    """

    f_ccf_spot_tot_box = []
    f_spec_spot_tot_box = []
    visible_phi_box = []
    active_surface_box = []

    intensity = _get_intensity_ratio(T_star, T_diff_spot)

    func = partial(
        spot_scan_npsi_pal, 
        xyz, inclination, nrho, psi, grid, v, limba1, limba2, 
        modif_bis_quad, modif_bis_lin, modif_bis_cte, 
        vrad_spec, intensity_spec, intensity_spec_spot, v_interval_spec, n_v_spec, n_spec, 
        s, longitude, latitude, magn_feature_type, T_star, T_diff_spot, intensity
        )

    outs = pqdm(np.arange(npsi), func, n_jobs=cpu_use)
    
    for i in range(npsi):
        outsbox = outs[i]

        # f_ccf_spot_tot_box.append(outsbox[0])
        f_spec_spot_tot_box.append(outsbox[0])
        visible_phi_box.append(outsbox[1])
        active_surface_box.append(outsbox[2])

    return f_spec_spot_tot_box, visible_phi_box, active_surface_box


    
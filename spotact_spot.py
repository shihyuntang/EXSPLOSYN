"""
This .py is a modification of the original starspot.c 
from the SOAP2 software

by Shih-Yun Tang 2022/09/04
"""
from functools import partial
import sys

import numpy as np
from numpy import cos, sin
from numba import njit
# from scipy import signal
from pqdm.processes import pqdm
# import pysynphot as pysynphot

import visualization as vis
import spotact_utils as utils


@njit(cache=True)
def spot_location_init(spot_radius, longitude_deg, latitude_deg, inclination_deg, nrho):
    """
    Coordinates (x,y,z) of the active region circumference on the stellar disc
    for the given lat/long/inclination. The circumference is sampled with `nrho`
    points around angle rho ∈ [-π, π).

    Args:
        spot_radius   (float): spot radius in units of R_star (0..1)
        longitude_deg (float): spot longitude [deg]
        latitude_deg  (float): spot latitude  [deg]
        inclination_deg (float): stellar inclination [deg] (0=pole-on, 90=edge-on)
        nrho (int): number of points on the circumference (>=3)

    Returns:
        xyz : (3, nrho) array
    """
    if nrho < 3:
        # minimal safe sampling
        nrho = 3

    lon = np.deg2rad(longitude_deg)
    lat = np.deg2rad(latitude_deg)
    inc = np.deg2rad(inclination_deg)

    # Here, 
    # xyz: real position of the spot on the stellar disc
    # xyz0: position of the spot when it is on the disc center

    # A circular active region has a resolution given by nrho, implies that we
    # will have a point on the disc circumference every 2*pi/(nrho-1). 
    # The -1 is because there is (nrho-1) intervals

    # rho_step = 2*np.pi/(nrho-1)
    # rho = [-np.pi + j*rho_step for j in range(nrho)] # rho goes from -pi to pi
    
    # xyz0 = np.ones((3, nrho), dtype=np.float64)
    # for j in range(nrho):
    #     # see Boisse+2012 fig. 1 for coord. setup
    #     xyz0[0][j] = np.sqrt(1 - s**2)   # x=sqrt(r^2-s^2), where r==1
    #     xyz0[1][j] = s*cos(rho[j]) # projection of the circumference on the y axis
    #     xyz0[2][j] = s*sin(rho[j]) # projection of the circumference on the z axis

    # circumference parameter (avoid duplicating start/end point)
    # rho ∈ [-π, π) without using linspace(endpoint=...)
    step = (2.0 * np.pi) / nrho
    rho = -np.pi + step * np.arange(nrho)

    # circle at disc center: x = sqrt(1 - s^2), y = s cos(rho), z = s sin(rho)
    s = spot_radius
    s2 = s * s
    x0 = np.sqrt(1.0 - s2)
    # allocate once
    xyz0 = np.empty((3, nrho), dtype=np.float64)
    xyz0[0, :] = x0
    xyz0[1, :] = s * np.cos(rho)
    xyz0[2, :] = s * np.sin(rho)


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

    # apply rotations: Ry(inclination) @ Rz(longitude) @ Ry(latitude)
    b  = -lat
    g  =  lon
    b2 = (np.pi / 2.0) - inc

    cb,  sb  = np.cos(b),  np.sin(b)
    cg,  sg  = np.cos(g),  np.sin(g)
    cb2, sb2 = np.cos(b2), np.sin(b2)

    # R = np.array([
    #     [ cb2*cg*cb - sb*sb2,   -sg*cb2,        cb2*cg*sb + sb2*cb ],
    #     [        sg*cb,              cg,               sg*sb       ],
    #     [ -sb2*cg*cb - cb2*sb,   sb2*sg,   -sb2*cg*sb + cb2*cb     ]
    #     ], dtype=np.float64)
    R = np.empty((3, 3), dtype=np.float64)
    R[0,0] =  cb2*cg*cb - sb*sb2
    R[0,1] = -sg*cb2
    R[0,2] =  cb2*cg*sb + sb2*cb
    R[1,0] =  sg*cb
    R[1,1] =  cg
    R[1,2] =  sg*sb
    R[2,0] = -sb2*cg*cb - cb2*sb
    R[2,1] =  sb2*sg
    R[2,2] = -sb2*cg*sb + cb2*cb


    # (3,3) @ (3,nrho) -> (3,nrho)
    xyz = R @ xyz0
    return xyz

@njit(cache=True)
def spot_inverse_rotation(xyz, longitude_deg, latitude_deg, inclination_deg, phase):
    """
    Map point(s) (x,y,z) back to the configuration where the region is at disc center
    by applying the inverse rotation for (phase, longitude, latitude, inclination).

    Args:
        xyz            : (3,) or (3,n) array
        longitude_deg  : [deg]
        latitude_deg   : [deg]
        inclination_deg: [deg]
        phase          : [0..1]

    Returns:
        (3,) or (3,n) array of transformed coordinates
    """
    # inverse phase rotation in radians
    g2 = 2.0 * np.pi * (phase - 1.0)
    i  = np.deg2rad(inclination_deg)
    b  =  np.deg2rad(latitude_deg)
    g  = -np.deg2rad(longitude_deg)
    b2 = -(np.pi / 2.0 - i)

    # R1 = np.array([
    #     [ (1-cos(g2))*cos(i)*cos(i) + cos(g2),  sin(g2)*sin(i),   (1-cos(g2))*cos(i)*sin(i)           ],
    #     [ -sin(g2)*sin(i),                      cos(g2),           sin(g2)*cos(i)                     ],
    #     [ (1-cos(g2))*sin(i)*cos(i),           -sin(g2)*cos(i),   (1-cos(g2))*sin(i)*sin(i) + cos(g2) ]
    #     ])
    # R2 = np.array([
    #     [  cos(b)*cos(g)*cos(b2)-sin(b2)*sin(b), -sin(g)*cos(b),   cos(b)*cos(g)*sin(b2)+sin(b)*cos(b2) ],
    #     [  sin(g)*cos(b2),                        cos(g),          sin(g)*sin(b2)                       ],
    #     [ -sin(b)*cos(g)*cos(b2)-cos(b)*sin(b2),  sin(b)*sin(g),  -sin(b)*cos(g)*sin(b2)+cos(b)*cos(b2) ]
    # ])
    # R3 = R2@R1

    cg,  sg  = np.cos(g),   np.sin(g)
    cg2, sg2 = np.cos(g2),  np.sin(g2)
    ci,  si  = np.cos(i),   np.sin(i)
    cb,  sb  = np.cos(b),   np.sin(b)
    cb2, sb2 = np.cos(b2),  np.sin(b2)

    # R1: rotation by phase around stellar axis projected in our frame
    # written explicitly to avoid pow/allocs
    R1 = np.array([
        [(1.0 - cg2) * ci * ci + cg2,   sg2 * si,            (1.0 - cg2) * ci * si],
        [        -sg2 * si,             cg2,                 sg2 * ci],
        [(1.0 - cg2) * si * ci,        -sg2 * ci,            (1.0 - cg2) * si * si + cg2]
    ], dtype=np.float64)

    # R2: longitude/latitude/inclination block
    R2 = np.array([
        [ cb*cg*cb2 - sb2*sb,    -sg*cb,    cb*cg*sb2 + sb*cb2],
        [ sg*cb2,                 cg,       sg*sb2],
        [-sb*cg*cb2 - cb*sb2,     sb*sg,   -sb*cg*sb2 + cb*cb2]
    ], dtype=np.float64)

    R = R2 @ R1
    return R @ xyz



@njit(cache=True)
def get_spot_phase_coord(xyz, inclination_deg, nrho, phase):
    """
    Rotate the spot circumference coordinates to a given phase.

    Args:
        xyz (np.ndarray): Coordinates of the circumference at disc-center frame; shape (3, nrho).
        inclination_deg (float): Stellar inclination [deg].
        nrho (int): Number of points on the circumference (unused; xyz.shape[1] is used).
        phase (float): Rotation phase in [0, 1].

    Returns:
        np.ndarray: Rotated coordinates; shape (3, nrho).
    """
    # rotation angle (negative = forward rotation from our LOS convention)
    psi = -2.0 * np.pi * phase
    inc = np.deg2rad(inclination_deg)

    # rotation axis in our frame (unit length): (cos i, 0, sin i)
    ax0 = np.cos(inc)
    ax1 = 0.0
    ax2 = np.sin(inc)

    c = np.cos(psi)
    s = np.sin(psi)
    omc = 1.0 - c  # one minus cos

    # Rodrigues rotation matrix around (ax0, ax1, ax2)
    R = np.array([
        [omc*ax0*ax0 + c,         omc*ax0*ax1 + s*ax2,  omc*ax0*ax2 - s*ax1],
        [omc*ax1*ax0 - s*ax2,     omc*ax1*ax1 + c,      omc*ax1*ax2 + s*ax0],
        [omc*ax2*ax0 + s*ax1,     omc*ax2*ax1 - s*ax0,  omc*ax2*ax2 + c    ],
    ], dtype=np.float64)

    return R @ xyz  # (3,3) @ (3,nrho) -> (3,nrho)


@njit 
def spot_area(xlylzl, grid_size):
    """
    Determine a minimal yz bounding box on the stellar disc that contains the visible
    part of the active region’s circumference.

    Args:
        xlylzl (np.ndarray): (3, nrho) array of circumference coordinates (x,y,z).
        grid_size (int): size of the yz grid.

    Returns:
        (visibility_flag, iminy, iminz, imaxy, imaxz)
        visibility_flag: 0 = fully invisible, 1 = at least partly visible
        indices are integer grid indices (inclusive bounds on [min,max])
    """
    # unpack
    x = xlylzl[0, :]
    y = xlylzl[1, :]
    z = xlylzl[2, :]

    # visible points (x >= 0 on the hemisphere we see)
    vis = x >= 0.0
    count_yes = np.sum(vis)
    count_no  = x.shape[0] - count_yes

    if count_yes == 0:
        # Completely invisible; return a trivial box (all zeros) and visibility=0
        return 0, 0, 0, 0, 0

    # init with extrema over the visible subset
    min_y = np.min(y[vis])
    max_y = np.max(y[vis])
    min_z = np.min(z[vis])
    max_z = np.max(z[vis])

    if (count_yes>0) and (count_no>0):
        # There are both visible and invisible points
        # --> active region is on the edge
        # In this situation there are cases where the yz-area define above is 
        # actually smaller than the real area of the active region on the stellar disk.
        # The minima/maxima are over/under-estimated if the active region is on one 
        # of the axis (y or z). Because if on the y axis, the minimum (or maximum) 
        # won't be on the circumference of the active region. Same for z axis
        
        if min_y * max_y<0: # active region on the z-axis because 
                            # one point is on the positive side of z, 
                            # and the other on the negative side of z.

            if min_z < 0: # parts (or all) of the spot is at the southern hemisphere 
                if np.abs(min_z) >= np.abs(max_z): # southern part spot dominate
                    min_z = -1
                else:
                    max_z = 1
            else: max_z = 1
            
            # active region on the bottom-z axis (z<0)
            # active region on the top-z axis (z>=0)
        
        if min_z * max_z<0:       
            if min_y < 0: 
                if np.abs(min_y) >= np.abs(max_y):
                    min_y = -1
                else:
                    max_y = 1
            else: max_y=1
    
    # map from [-1,1] to grid indices with step = 2/grid_size
    grid_step = 2.0 / grid_size

    # Indices of miny, minz,... on the grid
    iminy = np.floor( (1.+min_y)/grid_step ) # floor(x) returns the largest integral value that is not greater than x.
                                            # floor of 2.3 is 2.0, floor of 3.8 is 3.0, floor of -2.3 is -3.0, floor of -3.8 is -4.0
    iminz = np.floor( (1.+min_z)/grid_step )

    imaxy = np.ceil( (1.+max_y)/grid_step )  # ceil(x) returns the smallest integral value that is not less than x.
                                            # ceil of 2.3 is 3, ceil of 3.8 is 4.0, ceil of -2.3 is -2.0, ceil of -3.8 is -3.0
    imaxz = np.ceil( (1.+max_z)/grid_step )

    # clamp to valid index range [0, grid_size]
    if iminy < 0: iminy = 0
    if iminz < 0: iminz = 0
    if imaxy > grid_size: imaxy = grid_size
    if imaxz > grid_size: imaxz = grid_size

    return 1, int(iminy), int(iminz), int(imaxy), int(imaxz)

    
@njit(cache=True)
def rndup(n: float, nb_decimal: int) -> float:
    """
    Round to nearest with ties AWAY from zero at nb_decimal places.
    Examples: 1.5 -> 2.0, 1.4 -> 1.0, -1.5 -> -2.0
    """
    # scale positive, remember sign to avoid fp sign-corner cases
    s = 1.0 if n >= 0.0 else -1.0
    a = abs(n) * (10.0 ** nb_decimal)
    # add 0.5 then floor → ties go up (for positive a)
    r = np.floor(a + 0.5)
    return s * r / (10.0 ** nb_decimal)

# @njit
# def shifting_spectrum(
#     _lambda: np.ndarray, CCF: np.ndarray, v_shift: float, n1: int) -> np.ndarray:
#     """
#     Shift the CCF with a given velocity v_shift using linear interpolation.

#     Args:
#         _lambda (np.ndarray): Wavelength values.
#         CCF (np.ndarray): Cross-Correlation Function values.
#         v_shift (float): Velocity shift.
#         n1 (int): Number of data points.

#     Returns:
#         np.ndarray: The blueshifted CCF values.
#     """
#     # Initialize arrays
#     CCF_derivative = np.zeros(n1, dtype=np.float64)
#     CCF_blueshifted = np.zeros(n1, dtype=np.float64)

#     # Compute the CCF derivative
#     if _lambda[1] - _lambda[0] == 0:
#         CCF_derivative[0] = 0
#     else:
#         CCF_derivative[0] = (CCF[1] - CCF[0]) / (_lambda[1] - _lambda[0])

#     for i in range(1, n1 - 1):
#         if _lambda[i + 1] - _lambda[i] == 0:
#             CCF_derivative[i] = CCF_derivative[i - 1]
#         else:
#             CCF_derivative[i] = (CCF[i + 1] - CCF[i]) / (_lambda[i + 1] - _lambda[i])

#     CCF_derivative[-1] = 0.0  # Last element

#     # Compute the shifted CCF using linear interpolation
#     for i in range(n1):
#         if v_shift >= 0 or i == 0:
#             CCF_blueshifted[i] = CCF[i] - v_shift * CCF_derivative[i]
#         else:  # v_shift < 0 and i > 0
#             CCF_blueshifted[i] = CCF[i] - v_shift * CCF_derivative[i - 1]

#     return CCF_blueshifted

@njit(cache=True)
def shifting_spectrum(v: np.ndarray, y: np.ndarray, v_shift: float, n1: int) -> np.ndarray:
    """
    Fractional shift ONLY (|v_shift| < dv). The integer-bin shift is handled
    by the caller via stop0/start1/stop1. This function just does a
    one-neighbor linear blend on a uniform grid.

    out[i] ≈ y(v[i] - v_shift)

    If v_shift > 0 : sample between y[i] and y[i-1]
        out[i] = (1 - s) * y[i] + s * y[i-1],  s = v_shift/dv in (0,1)
    If v_shift < 0 : sample between y[i] and y[i+1]
        out[i] = (1 - t) * y[i] + t * y[i+1],  t = (-v_shift)/dv in (0,1)

    Edges are clamped (use nearest value).
    """
    out = np.empty(n1, dtype=np.float64)

    dv = v[1] - v[0]
    if dv == 0.0:
        for i in range(n1):
            out[i] = y[i]
        return out

    s = v_shift / dv

    if s >= 0.0:
        # blend with previous bin
        if s >= 1.0:
            s = 1.0  # safety clamp; ideally caller ensures |s|<1
        for i in range(n1):
            y0 = y[i]
            if i == 0:
                y1 = y0
            else:
                y1 = y[i - 1]
            out[i] = (1.0 - s) * y0 + s * y1
    else:
        # s < 0: blend with next bin
        t = -s
        if t >= 1.0:
            t = 1.0
        for i in range(n1):
            y0 = y[i]
            if i == n1 - 1:
                y1 = y0
            else:
                y1 = y[i + 1]
            out[i] = (1.0 - t) * y0 + t * y1

    return out


@njit
def itot_flux(vrot, incl, limba1, limba2, grid_size, vrad_spec, intensity_spec_use, 
              v_interval_spec, n_v_spec, n_spec):
    """Calculates the flux, and spec in each cell of the grid and integrate
    over the entire stellar disc to have the integrated flux, and spec.
    """

    integrated_stellar_spectrum = np.zeros(n_v_spec, dtype=np.float64)

    delta_grid = 2 / grid_size # step of the grid. grid goes from -1 to 1, therefore 2 in total
    delta_v_spec = 2*v_interval_spec / (n_v_spec-1) # step in speed of the Spec. 
                                            # There is (n_v-1) intervals

    sinIncl = sin(incl * np.pi / 180.0)  # hoisted

    # Store summed flux per grid for 2D disk surface plotting
    quite_2d_surface_flux_sum = np.zeros((grid_size, grid_size), dtype=np.float64)

    # v_interval is from velocity 0 to the edge of the spectrum taking into 
    # account minimal (width - v to 0) or maximal rotation (0 to width + v). 
    # n_v is the number of points for all the Spec from minimum rotation to 
    # maximum one (from width - v to width + v).
    # n_v is therefore double than the v_interval, thus, we multiply 
    # v_interval by 2.

    # Scan of each cell on the grid
    for iy in range(grid_size):
        y = -1 + iy*delta_grid # y between -1 to 1
        delta = y * vrot * sinIncl # Give the velocity of the rotation for the given grid 
                                   # cell as a function of y. For y=-1 => Vrot min, 
                                   # for y=1 => Vrot max, and for y=0 => Vrot = 0.
        
        for iz in range(grid_size):
            # cell_integrated_stellar_spectra = np.zeros(n_v_spec, dtype=np.float64)
            # Per-pixel scalar accumulators (replace per-pixel arrays)
            quiet_sum  = 0.0

            z = -1 + iz*delta_grid # z between -1 to 1
            if (y*y + z*z)<=1: 
                # projected radius on the sky smaller than 1, which means that 
                # we are on the stellar disc

                # limb-darkening
                # sqrt(r^2-(y^2+z^2)) where r=1. This is equal to 1 at the disc center, 
                # and 0 on the limb. This is often referred in the literature as cos(theta)
                r_cos = np.sqrt( 1 - (y*y+z*z) )  # mu angle
                
                # intensity due to limb-darkening
                limb = 1 - limba1*(1-r_cos) - limba2*(1-r_cos)**2 
 
                # --- Modify SOAP2 code ---

                # The CCF is defined between -20 and +20 km/s with a sampling "delta_v" of 0.1 km/s, 
                # and with boundaries normalized at 1. To account for stellar rotation in each cell of 
                # the simulation, the CCF will be shifted depending on the position of the cell on the 
                # stellar disc. To shift the CCF by a RV of x km/s, we will first shift the CCF by 
                # "n_v_shifted_quotient" = int(x/sampling) steps, and then interpolate the CCF with 
                # the remaining velocity "n_v_shifted_remainder" to have the correct shift of x km/s 
                # at the end.
                
                # This will be done by shifting the CCF by an integer of steps and then interpolating 
                # the CCF to match the correct velocity
                # The sampling of the CCF is every 100 m/s, so 2 km/s corresponds to a shift of 20 steps.
                                
                # -------------------------- SPEC -----------------------------
                n_v_shifted            = delta / delta_v_spec
                n_v_shifted_quotient   = rndup(n_v_shifted, 0)  # assume returns int
                n_v_shifted_remainder  = delta - n_v_shifted_quotient * delta_v_spec
                v_shift = n_v_shifted_remainder
     
                # shifting the CCF with the remainder of n_v_shifted, the quotient will be taken into 
                # account by shifting all the points of the spectrum
                intensity_spec_shift = shifting_spectrum(
                    vrad_spec, intensity_spec_use, v_shift, n_spec
                    )
                
                # difference in number of steps between the CCF without any rotation and the one with rotation
                # Left/middle/right regions
                stop0 = int((n_v_spec - n_spec) / 2.0 + n_v_shifted_quotient)
                start1 = stop0
                stop1  = int(n_spec + stop0)
                
                # To take into account the rotation, we increase the width of the CCF. Because the original CCF 
                # is only defined between -20 and +20 km/s, we have to extrapolate on each side of the boundaries 
                # of the CCF

                val_q0 = intensity_spec_shift[0]
                for j in range(stop0):
                    integrated_stellar_spectrum[j] += val_q0 * limb
                    quiet_sum += val_q0 * limb
                
                for j in range(start1, stop1):
                    k = j - start1  # 0 .. n_spec-1
                    vq = intensity_spec_shift[k]
                    # value of the CCF, weighted by the limb-darkening
                    integrated_stellar_spectrum[j] += vq * limb
                    quiet_sum += vq * limb

                # Right pad
                val_qR = intensity_spec_shift[n_spec - 1]
                for j in range(stop1, n_v_spec):
                    # extrapolation on the right of the CCF with the value of the right boundary, 
                    # weighted by the limb-darkening
                    integrated_stellar_spectrum[j] += val_qR * limb
                    quiet_sum += val_qR * limb
                
                # sum_star += intensity_spec_shift[0]*limb
                quite_2d_surface_flux_sum[iy, iz] = quiet_sum

    return integrated_stellar_spectrum, quite_2d_surface_flux_sum.T
@njit
def itot_int(vrot, incl, limba1, limba2, grid_size, vrad_spec, intensity_spec, 
             v_interval_spec, n_v_spec, n_spec, mu_s):
    """Calculates the flux, spec, and CCF in each cell of the grid and integrate
    over the entire stellar disc to have the integrated flux, spec, and CCF
    """

    integrated_stellar_spectrum = np.zeros(n_v_spec, dtype=np.float64)

    delta_grid = 2 / grid_size # step of the grid. grid goes from -1 to 1, therefore 2 in total
    delta_v_spec = 2*v_interval_spec / (n_v_spec-1) # step in speed of the Spec. 
                                            # There is (n_v-1) intervals

    sinIncl = sin(incl * np.pi / 180.0)  # hoisted

    quite_2d_surface_flux_sum = np.zeros((grid_size, grid_size), dtype=np.float64)

    # Scan of each cell on the grid
    for iy in range(grid_size):
        y = -1 + iy*delta_grid # y between -1 to 1
        delta = y * vrot * sinIncl # Give the velocity of the rotation for the given grid 
                                   # cell as a function of y. For y=-1 => Vrot min, 
                                   # for y=1 => Vrot max, and for y=0 => Vrot = 0.
        
        for iz in range(grid_size):
            quiet_sum  = 0.0

            z = -1 + iz*delta_grid # z between -1 to 1
            if (y*y + z*z)<=1: 
                r_cos = np.sqrt( 1 - (y*y+z*z) )  # mu angle
                
                # --- Use mu angle spectra instead of scaling flux via limb-darkening ---
                # # index of the closest mu value to r_cos
                # f_loc = int(np.argmin(np.abs(mu_s - r_cos))) 

                # # interpolate on mu spectra
                # if r_cos >= mu_s[0]:
                #     intensity_spec_use = intensity_spec[:, 0]
                # elif r_cos <= mu_s[-1]:
                #     intensity_spec_use = intensity_spec[:, -1]
                # else:
                #     if r_cos > mu_s[f_loc]:
                #         mu1_low = mu_s[f_loc-1]
                #         mu1_high = mu_s[f_loc]
                #         mu1_int_low = intensity_spec[:, f_loc-1]
                #         mu1_int_high = intensity_spec[:, f_loc]
                #     elif r_cos < mu_s[f_loc]:
                #         mu1_low = mu_s[f_loc]
                #         mu1_high = mu_s[f_loc+1]    
                #         mu1_int_low = intensity_spec[:, f_loc]
                #         mu1_int_high = intensity_spec[:, f_loc+1]

                #     intensity_spec_use = mu1_int_low + ( (r_cos - mu1_low)/(mu1_high - mu1_low) )\
                #                     *(mu1_int_high - mu1_int_low)
                f_loc = utils.nearest_mu_index(mu_s, r_cos)
                intensity_spec_use = intensity_spec[:, f_loc]
                
                # -------------------------- SPEC ------------------------------
                n_v_shifted            = delta / delta_v_spec
                n_v_shifted_quotient   = rndup(n_v_shifted, 0)  # assume returns int
                n_v_shifted_remainder  = delta - n_v_shifted_quotient * delta_v_spec
                v_shift = n_v_shifted_remainder
            
                intensity_spec_shift = shifting_spectrum(
                    vrad_spec, intensity_spec_use, v_shift, n_spec
                    )

                # difference in number of steps between the CCF without any rotation and the one with rotation
                # Left/middle/right regions
                stop0 = int((n_v_spec - n_spec) / 2.0 + n_v_shifted_quotient)
                start1 = stop0
                stop1  = int(n_spec + stop0)
                
                # To take into account the rotation, we increase the width of the CCF. Because the original CCF 
                # is only defined between -20 and +20 km/s, we have to extrapolate on each side of the boundaries 
                # of the CCF

                val_q0 = intensity_spec_shift[0]
                for j in range(stop0):
                    integrated_stellar_spectrum[j] += val_q0
                    quiet_sum += val_q0
                
                for j in range(start1, stop1):
                    k = j - start1  # 0 .. n_spec-1
                    vq = intensity_spec_shift[k]
                    # value of the CCF, weighted by the limb-darkening
                    integrated_stellar_spectrum[j] += vq
                    quiet_sum += vq

                # Right pad
                val_qR = intensity_spec_shift[n_spec - 1]
                for j in range(stop1, n_v_spec):
                    # extrapolation on the right of the CCF with the value of the right boundary, 
                    # weighted by the limb-darkening
                    integrated_stellar_spectrum[j] += val_qR
                    quiet_sum += val_qR
                
                # sum_star += intensity_spec_shift[0]*limb
                quite_2d_surface_flux_sum[iy, iz] = quiet_sum

    return integrated_stellar_spectrum, quite_2d_surface_flux_sum.T

@njit(cache=True, fastmath=True)
def spot_scan_flux(
    vrot, incl, Tphoto, grid_size, Spec_dict, star_LD_coeff, 
    vrad_spec, intensity_spec_quiet, 
    v_interval_spec, n_v_spec, n_spec, 
    active_2d_surface_temperature
):
    """
    Integrate flux spectra over active pixels for one phase.
    Returns:
        (integrated_active, integrated_quiet, active_flux_map.T, quiet_flux_map.T)
    """
    # Unpack lookups
    _, ld_coeffs, ld_keys, ld_vals = star_LD_coeff
    _, spec_spectrum, _, sp_keys, sp_vals = Spec_dict

    # Active-pixel indices (nonzero + not photosphere)
    mask = (active_2d_surface_temperature != 0) & (active_2d_surface_temperature != Tphoto)
    act_idxy, act_idxz = np.where(mask)

    delta_grid  = 2.0 / grid_size
    delta_v_spec = (2.0 * v_interval_spec) / (n_v_spec - 1.0)
    sinIncl = np.sin(incl * np.pi / 180.0)

    integrated_stellar = np.zeros(n_v_spec, dtype=np.float64)
    integrated_quiet   = np.zeros(n_v_spec, dtype=np.float64)

    active_flux_map = np.zeros((grid_size, grid_size), dtype=np.float64)
    quiet_flux_map  = np.zeros((grid_size, grid_size), dtype=np.float64)

    # Quiet limb-darkening once
    iq0 = utils.lut_lookup_eq(Tphoto, ld_keys, ld_vals)
    limba1_q = ld_coeffs[iq0, 0]
    limba2_q = ld_coeffs[iq0, 1]

    for ii in range(act_idxy.shape[0]):
        iy = int(act_idxy[ii])
        iz = int(act_idxz[ii])

        y = -1.0 + iy * delta_grid
        z = -1.0 + iz * delta_grid

        # rotation Doppler shift for this pixel
        delta = y * vrot * sinIncl

        active_temp = int(active_2d_surface_temperature[iy, iz])

        # Active spectrum row and LD coeffs
        js = utils.lut_lookup_eq(active_temp, sp_keys, sp_vals)
        intensity_spec_active = spec_spectrum[js]  # shape: (n_spec,)

        iq = utils.lut_lookup_eq(active_temp, ld_keys, ld_vals)
        limba1_a = ld_coeffs[iq, 0]
        limba2_a = ld_coeffs[iq, 1]

        # Limb factor at this pixel
        r_cos = np.sqrt(1.0 - (y * y + z * z))
        limb_quiet  = 1.0 - limba1_q * (1.0 - r_cos) - limba2_q * (1.0 - r_cos) * (1.0 - r_cos)
        limb_active = 1.0 - limba1_a * (1.0 - r_cos) - limba2_a * (1.0 - r_cos) * (1.0 - r_cos)

        # Doppler shift by remainder; quotient handled in index offsets
        n_v_shifted           = delta / delta_v_spec
        n_v_shifted_quotient  = rndup(n_v_shifted, 0)  # external helper; assumed int return
        v_shift               = delta - n_v_shifted_quotient * delta_v_spec

        quiet_shift  = shifting_spectrum(vrad_spec, intensity_spec_quiet, v_shift, n_spec)
        active_shift = shifting_spectrum(vrad_spec, intensity_spec_active, v_shift, n_spec)

        # Window placement (with clamping to avoid overruns)
        stop0  = int((n_v_spec - n_spec) / 2.0 + n_v_shifted_quotient)
        if stop0 < 0:
            stop0 = 0
        elif stop0 > n_v_spec:
            stop0 = n_v_spec

        start1 = stop0
        stop1  = start1 + n_spec
        if stop1 > n_v_spec:
            stop1 = n_v_spec
        mid_len = stop1 - start1  # may be < n_spec near edges

        # Accumulators for maps (scalar)
        a_sum = 0.0
        q_sum = 0.0

        # Left pad
        if stop0 > 0:
            va0 = active_shift[0]
            vq0 = quiet_shift[0]
            for j in range(stop0):
                integrated_stellar[j] += va0 * limb_active
                integrated_quiet[j]   += vq0 * limb_quiet
            a_sum += va0 * limb_active * stop0
            q_sum += vq0 * limb_quiet  * stop0

        # Middle block
        for j in range(mid_len):
            k = j  # 0..mid_len-1
            va = active_shift[k]
            vq = quiet_shift[k]
            jj = start1 + j
            integrated_stellar[jj] += va * limb_active
            integrated_quiet[jj]   += vq * limb_quiet
            a_sum += va * limb_active
            q_sum += vq * limb_quiet

        # Right pad
        if stop1 < n_v_spec:
            vaR = active_shift[n_spec - 1]
            vqR = quiet_shift[n_spec - 1]
            for j in range(stop1, n_v_spec):
                integrated_stellar[j] += vaR * limb_active
                integrated_quiet[j]   += vqR * limb_quiet
            a_sum += vaR * limb_active * (n_v_spec - stop1)
            q_sum += vqR * limb_quiet  * (n_v_spec - stop1)

        active_flux_map[iy, iz] = a_sum
        quiet_flux_map[iy,  iz] = q_sum

    return (integrated_stellar, integrated_quiet,
            active_flux_map.T, quiet_flux_map.T)

@njit(cache=True, fastmath=True)
def spot_scan_int(
    vrot, incl, Tphoto, grid_size, Spec_dict, star_LD_coeff,  # star_LD_coeff not used here
    vrad_spec, intensity_spec_quiet, 
    v_interval_spec, n_v_spec, n_spec, 
    active_2d_surface_temperature
):
    """
    Intensity version: uses precomputed I(mu) grid; no extra LD factor.
    Returns:
        (integrated_active, integrated_quiet, active_flux_map.T, quiet_flux_map.T)
    """
    _, spec_spectrum, mu_s, sp_keys, sp_vals = Spec_dict

    mask = (active_2d_surface_temperature != 0) & (active_2d_surface_temperature != Tphoto)
    act_idxy, act_idxz = np.where(mask)

    delta_grid  = 2.0 / grid_size
    delta_v_spec = (2.0 * v_interval_spec) / (n_v_spec - 1.0)
    sinIncl = np.sin(incl * np.pi / 180.0)

    integrated_stellar = np.zeros(n_v_spec, dtype=np.float64)
    integrated_quiet   = np.zeros(n_v_spec, dtype=np.float64)

    active_flux_map = np.zeros((grid_size, grid_size), dtype=np.float64)
    quiet_flux_map  = np.zeros((grid_size, grid_size), dtype=np.float64)

    for ii in range(act_idxy.shape[0]):
        iy = int(act_idxy[ii])
        iz = int(act_idxz[ii])

        y = -1.0 + iy * delta_grid
        z = -1.0 + iz * delta_grid
        delta = y * vrot * sinIncl

        active_temp = int(active_2d_surface_temperature[iy, iz])
        js = utils.lut_lookup_eq(active_temp, sp_keys, sp_vals)
        intensity_spec_active_full = spec_spectrum[js]  # (n_spec, n_mu)

        # Closest mu to r_cos (intensity grids already encode limb-darkening)
        r_cos = np.sqrt(1.0 - (y * y + z * z))
        f_loc = utils.nearest_mu_index(mu_s, r_cos)

        intensity_quiet_use  = intensity_spec_quiet[:, f_loc]
        intensity_active_use = intensity_spec_active_full[:, f_loc]

        # Doppler shift quotient+remainder
        n_v_shifted          = delta / delta_v_spec
        n_v_shifted_quotient = rndup(n_v_shifted, 0)
        v_shift              = delta - n_v_shifted_quotient * delta_v_spec

        quiet_shift  = shifting_spectrum(vrad_spec, intensity_quiet_use,  v_shift, n_spec)
        active_shift = shifting_spectrum(vrad_spec, intensity_active_use, v_shift, n_spec)

        # Window placement (clamped)
        stop0  = int((n_v_spec - n_spec) / 2.0 + n_v_shifted_quotient)
        if stop0 < 0:
            stop0 = 0
        elif stop0 > n_v_spec:
            stop0 = n_v_spec

        start1 = stop0
        stop1  = start1 + n_spec
        if stop1 > n_v_spec:
            stop1 = n_v_spec
        mid_len = stop1 - start1

        a_sum = 0.0
        q_sum = 0.0

        # Left pad
        if stop0 > 0:
            va0 = active_shift[0]
            vq0 = quiet_shift[0]
            for j in range(stop0):
                integrated_stellar[j] += va0
                integrated_quiet[j]   += vq0
            a_sum += va0 * stop0
            q_sum += vq0 * stop0

        # Middle block
        for j in range(mid_len):
            k = j
            va = active_shift[k]
            vq = quiet_shift[k]
            jj = start1 + j
            integrated_stellar[jj] += va
            integrated_quiet[jj]   += vq
            a_sum += va
            q_sum += vq

        # Right pad
        if stop1 < n_v_spec:
            vaR = active_shift[n_spec - 1]
            vqR = quiet_shift[n_spec - 1]
            for j in range(stop1, n_v_spec):
                integrated_stellar[j] += vaR
                integrated_quiet[j]   += vqR
            a_sum += vaR * (n_v_spec - stop1)
            q_sum += vqR * (n_v_spec - stop1)

        active_flux_map[iy, iz] = a_sum
        quiet_flux_map[iy,  iz] = q_sum

    return (integrated_stellar, integrated_quiet,
            active_flux_map.T, quiet_flux_map.T)

@njit
def spot_scan(
    incl, grid_size, spot_size, longitude, latitude, phase,
    idx_miny, idx_minz, idx_maxy, idx_maxz, active_temp,
    active_2d_surface_temperature, Tphoto
    ):
    """
    Scan of the yz-area where the spot is.
    For each grid-point (y,z) we need to check whether it belongs to the spot
    or not. Sadly, we do not know the projected geometry of the spot in its
    actual position. Thus, we have to do an inverse rotation to replace the
    grid point where it would be in the initial configuration. Indeed, in the
    initial configuration, the spot has a well known geometry of a circle
    centered on the x-axis.
    """

    delta_grid = 2/grid_size
    xyz_actual = np.ones(3, dtype=np.float64) # actual coordinates
    xyz_initial = np.ones(3, dtype=np.float64) # coordinates transformed back to the initial configuration

    # Scan of each cell on the grid
    for iy in range(idx_miny, idx_maxy):
        y = -1 + iy*delta_grid # y between -1 to 1
        xyz_actual[1] = y

        for iz in range(idx_minz, idx_maxz):
            z = -1 + iz*delta_grid # z between -1 to 1

            if (y*y + z*z)<1: 
                xyz_actual[0] = np.sqrt(1-(y*y+z*z)) # sqrt(r^2-(y^2+z^2)) where r=1. 
                                    # This is equal to 1 at the disc center, 
                                    # and 0 on the limb. This is often referred in the literature as cos(theta)
                xyz_actual[2] = z

                # xyz_actual --> xyz_initial: Rotate the star so that the spot 
                # is on the disc center. But, the current spotted grid might not
                # be 
                xyz_initial = spot_inverse_rotation(
                    xyz_actual, longitude, latitude, incl, phase)    

                # If the grid is inside the active region at disk center
                if (xyz_initial[0]**2 >= (1-spot_size**2)) and (xyz_initial[0]>0): # Only need to check x coord.
                                                # As teh star has radius of 1, 1^2-spot_size^2
                                                # is the height (x) of the spot circumference.
                                                # Therefore, if the grid's x_coord^2 >= 1-spot_size^2,
                                                # it is inside the spot circumference.
                    
                    if active_2d_surface_temperature[iy, iz] == Tphoto:
                        # only if the gird is still at the photosphere temperature we update it
                        active_2d_surface_temperature[iy, iz] = active_temp
                    
    return active_2d_surface_temperature

                

def spot_scan_npsi_pal(
    inclination, Tphoto, psi, grid_size, Spec_dict, star_LD_coeff,
    vrot,
    vrad_spec, intensity_spec, v_interval_spec,
    n_v_spec, n_spec, fluxORint,
    active_2d_surface_temperature_psi, ipsi
):
    """
    Worker for a single phase index (ipsi). Returns:
    (intergrated_total_stellar_spectrum, intergrated_total_quiet_spectrum,
     phase_value, active_2d_surface_flux_sum, quiet_2d_surface_flux_sum)
    """
    active_2d_surface_temperature = active_2d_surface_temperature_psi[ipsi]

    if fluxORint == 'flux':
        (intergrated_total_stellar_spectrum,
         intergrated_total_quiet_spectrum,
         active_2d_surface_flux_sum,
         quiet_2d_surface_flux_sum) = spot_scan_flux(
            vrot, inclination, Tphoto, grid_size, Spec_dict, star_LD_coeff,
            vrad_spec, intensity_spec, v_interval_spec, n_v_spec, n_spec,
            active_2d_surface_temperature
        )
    else:  # 'intensity'
        (intergrated_total_stellar_spectrum,
         intergrated_total_quiet_spectrum,
         active_2d_surface_flux_sum,
         quiet_2d_surface_flux_sum) = spot_scan_int(
            vrot, inclination, Tphoto, grid_size, Spec_dict, star_LD_coeff,
            vrad_spec, intensity_spec, v_interval_spec, n_v_spec, n_spec,
            active_2d_surface_temperature
        )

    return (intergrated_total_stellar_spectrum,
            intergrated_total_quiet_spectrum,
            psi[ipsi],
            active_2d_surface_flux_sum,
            quiet_2d_surface_flux_sum)


def spot_scan_npsi(
    psi, npsi, vrot, inclination, Tphoto, grid_size, Spec_dict,
    star_LD_coeff,
    vrad_spec, intensity_spec, v_interval_spec, n_v_spec,
    n_spec, cpu_use, fluxORint,
    active_2d_surface_temperature_psi, silence_pbar
):
    """
    Scans the visible hemisphere across phases (psi) using the
    precomputed temperature cube (active_2d_surface_temperature_psi),
    and returns per-phase integrated spectra and flux maps.

    Returns:
        (intergrated_total_stellar_spectrum_box,   # (npsi, n_v_spec)
         intergrated_total_quiet_spectrum_box,     # (npsi, n_v_spec)
         visible_phi_box,                          # (npsi,)
         active_2d_surface_flux_sum_box,           # (npsi, grid_size, grid_size)
         quiet_2d_surface_flux_sum_box)            # (npsi, grid_size, grid_size)
    """
    # Preallocate outputs with the final shapes/dtypes
    intergrated_total_stellar_spectrum_box = np.empty((npsi, n_v_spec), dtype=np.float64)
    intergrated_total_quiet_spectrum_box   = np.empty((npsi, n_v_spec), dtype=np.float64)
    visible_phi_box                        = np.empty((npsi,), dtype=np.float32)
    active_2d_surface_flux_sum_box         = np.empty((npsi, grid_size, grid_size), dtype=np.float64)
    quiet_2d_surface_flux_sum_box          = np.empty((npsi, grid_size, grid_size), dtype=np.float64)

    # Prepare worker
    worker = partial(
        spot_scan_npsi_pal,
        inclination, Tphoto, psi, grid_size, Spec_dict, star_LD_coeff,
        vrot, vrad_spec, intensity_spec, v_interval_spec, n_v_spec, n_spec,
        fluxORint, active_2d_surface_temperature_psi
    )

    # for i in range(npsi):
    #     print(f'Scanning phase {i+1}/{npsi} ({psi[i]})')
    #     spot_scan_npsi_pal(
    #     xyz, inclination, Tphoto, nrho, psi, grid_size, Spec_dict, star_LD_coeff,
    #     vrot, 
    #     vrad_spec, intensity_spec, v_interval_spec, 
    #     n_v_spec, n_spec, spot_size, longitude, latitude, fluxORint,
    #     active_2d_surface_temperature_psi, i
    #     )

    # Always parallelize
    outs = pqdm(np.arange(npsi, dtype=int), worker, n_jobs=cpu_use, disable=silence_pbar)

    # Fill arrays
    for i, (spec_stellar, spec_quiet, phi, flux_active, flux_quiet) in enumerate(outs):
        intergrated_total_stellar_spectrum_box[i, :] = spec_stellar
        intergrated_total_quiet_spectrum_box[i, :]   = spec_quiet
        visible_phi_box[i]                           = phi
        active_2d_surface_flux_sum_box[i, :, :]      = flux_active
        quiet_2d_surface_flux_sum_box[i, :, :]       = flux_quiet

    return (intergrated_total_stellar_spectrum_box,
            intergrated_total_quiet_spectrum_box,
            visible_phi_box,
            active_2d_surface_flux_sum_box,
            quiet_2d_surface_flux_sum_box)
    
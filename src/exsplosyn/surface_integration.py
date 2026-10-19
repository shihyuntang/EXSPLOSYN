"""Active-region geometry and stellar surface spectral integration.

The spot geometry follows the logic originally ported from SOAP2's starspot.c
by Dumusque et al. 2014. The Python implementation has since been
substantially reworked for EXSPLOSYN, including spectral integration and
phase processing.

Surface arrays are indexed as [iy, iz]: axis 0 is y and axis 1 is z.
Integration flux maps are returned transposed as [iz, iy] for display, with
horizontal y and vertical z. A fixed-y slice is therefore a displayed column.
"""

from functools import partial
from pathlib import Path
from .model_identity import identity_arrays, identity_suffix

import logging
import sys

import numpy as np
from numpy import sin
from numba import njit
from pqdm.processes import pqdm

from .spectral_processing import apply_instrumental_broadening

logger = logging.getLogger(__name__)


@njit(cache=True)
def spot_location_init(spot_radius, longitude_deg, latitude_deg, inclination_angle, nrho):
    """Construct an active region's boundary at the initial rotation phase.

    Start with a circular boundary on the unit stellar sphere, centered along
    the line of sight. Rotate it to the specified latitude and longitude, then
    orient the star by its inclination angle. The x axis points toward the observer;
    y and z span the sky plane. Boundary points may lie on the hidden hemisphere.

    Parameters
    ----------
    spot_radius : float
        Boundary-circle radius in stellar-radius units (0 to 1), not an angle.
    longitude_deg, latitude_deg : float
        Region longitude and latitude in degrees.
    inclination_angle : float
        Stellar inclination angle in degrees: 0 is pole-on, 90 is equator-on.
    nrho : int
        Requested boundary samples; values below 3 are increased to 3.

    Returns
    -------
    xyz : ndarray, shape (3, max(nrho, 3))
        Cartesian boundary coordinates in stellar-radius units, with rows
        ordered x, y, z. Samples are evenly spaced without duplicating the
        closing point. This function does not select visible points.
    """
    if nrho < 3:
        nrho = 3

    lon = np.deg2rad(longitude_deg)
    lat = np.deg2rad(latitude_deg)
    inc = np.deg2rad(inclination_angle)

    # Sample a full circle without repeating the first boundary point.
    step = (2.0 * np.pi) / nrho
    rho = -np.pi + step * np.arange(nrho)

    # On the unit sphere, the boundary circle lies at x = sqrt(1 - radius^2).
    x0 = np.sqrt(1.0 - spot_radius**2)
    xyz0 = np.empty((3, nrho), dtype=np.float64)
    xyz0[0, :] = x0
    xyz0[1, :] = spot_radius * np.cos(rho)
    xyz0[2, :] = spot_radius * np.sin(rho)

    # Place the boundary on the star, then orient it toward the observer.
    # Coordinates are column vectors in a right-handed frame:
    #   x points toward the observer; the sky plane is yz.
    #   The stellar rotation axis is (cos(inc), 0, sin(inc)):
    #   inc=0 is pole-on (axis along x), inc=pi/2 is equator-on (axis along z).
    # Positive rotation angles follow the right-hand rule.
    #
    #             [ cos(b)  0  sin(b)]             [cos(g) -sin(g) 0]
    #   Ry(b) =   [   0     1    0   ]   Rz(g) =   [sin(g)  cos(g) 0]
    #             [-sin(b)  0  cos(b)]             [  0       0    1]
    #
    # Apply the rightmost rotation first:
    #   R = Ry(b2) @ Rz(g) @ Ry(b)
    #   b = -latitude, g = longitude, b2 = pi/2 - inclination_angle (radians).
    b = -lat
    g = lon
    b2 = (np.pi / 2.0) - inc

    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    cb2, sb2 = np.cos(b2), np.sin(b2)

    # With cb=cos(b), sb=sin(b), etc., the product is:
    #     [cb2*cg*cb - sb2*sb,  -cb2*sg,  cb2*cg*sb + sb2*cb]
    # R = [       sg*cb,          cg,            sg*sb       ]
    #     [-sb2*cg*cb - cb2*sb,  sb2*sg, -sb2*cg*sb + cb2*cb]
    #
    # xyz = R @ xyz0 transforms the full boundary into the observer frame.
    R = np.empty((3, 3), dtype=np.float64)
    R[0, 0] = cb2 * cg * cb - sb * sb2
    R[0, 1] = -sg * cb2
    R[0, 2] = cb2 * cg * sb + sb2 * cb
    R[1, 0] = sg * cb
    R[1, 1] = cg
    R[1, 2] = sg * sb
    R[2, 0] = -sb2 * cg * cb - cb2 * sb
    R[2, 1] = sb2 * sg
    R[2, 2] = -sb2 * cg * sb + cb2 * cb

    # Rotate every sampled boundary point into the observer frame.
    xyz = R @ xyz0
    return xyz


@njit(cache=True)
def spot_inverse_rotation(xyz, longitude_deg, latitude_deg, inclination_angle, phase):
    """Transform surface points into the active region's reference frame.

    Undo the stellar phase rotation, then reverse the placement defined by the
    region's longitude, latitude and stellar inclination angle. The region is centered
    along +x, so its circular boundary has constant x. The caller uses these
    coordinates to test whether a surface pixel belongs to the region.

    Parameters
    ----------
    xyz : ndarray, shape (3,) or (3, n_points)
        Observer-frame coordinates in stellar-radius units. The x axis points
        toward the observer; y and z span the sky plane.
    longitude_deg, latitude_deg : float
        Region longitude and latitude in degrees, as in spot_location_init.
    inclination_angle : float
        Stellar inclination angle in degrees: 0 is pole-on, 90 is equator-on.
    phase : float
        Rotation phase in cycles; phases separated by one cycle are equivalent.

    Returns
    -------
    xyz_reference : ndarray, same shape as xyz
        Coordinates in the frame where the region is centered along +x.
    """
    # Undo rotations in reverse order: R_inverse = R_init.T @ R_phase.T.
    # R_init is the placement matrix in spot_location_init; the stellar axis
    # in the observer frame is a = (cos(i), 0, sin(i)).
    # g2 retains the existing phase convention, equivalent to 2*pi*phase
    # modulo one full turn. R1 below rotates by -g2, undoing the phase rotation.
    g2 = 2.0 * np.pi * (phase - 1.0)
    i = np.deg2rad(inclination_angle)
    b = np.deg2rad(latitude_deg)
    g = -np.deg2rad(longitude_deg)
    b2 = -(np.pi / 2.0 - i)

    cg, sg = np.cos(g), np.sin(g)
    cg2, sg2 = np.cos(g2), np.sin(g2)
    ci, si = np.cos(i), np.sin(i)
    cb, sb = np.cos(b), np.sin(b)
    cb2, sb2 = np.cos(b2), np.sin(b2)

    # R1 undoes the phase rotation around the inclined stellar axis.
    # Full trigonometric form (cg2=cos(g2), ci=cos(i), etc. in the code below):
    # R1 = np.array([
    #     [(1-cos(g2))*cos(i)*cos(i) + cos(g2), sin(g2)*sin(i),  (1-cos(g2))*cos(i)*sin(i)],
    #     [-sin(g2)*sin(i),                    cos(g2),         sin(g2)*cos(i)],
    #     [(1-cos(g2))*sin(i)*cos(i),          -sin(g2)*cos(i), (1-cos(g2))*sin(i)*sin(i) + cos(g2)]
    # ])
    R1 = np.array(
        [
            [(1.0 - cg2) * ci * ci + cg2, sg2 * si, (1.0 - cg2) * ci * si],
            [-sg2 * si, cg2, sg2 * ci],
            [(1.0 - cg2) * si * ci, -sg2 * ci, (1.0 - cg2) * si * si + cg2],
        ],
        dtype=np.float64,
    )

    # Inverse placement: reverse the matrix order and negate each angle.
    #   R2 = Ry(latitude) @ Rz(-longitude) @ Ry(inclination_angle - pi/2)
    #      = Ry(b) @ Rz(g) @ Ry(b2) = R_init.T.
    # Full trigonometric form (cb=cos(b), sb=sin(b), etc. in the code below):
    # R2 = np.array([
    #     [cos(b)*cos(g)*cos(b2)-sin(b2)*sin(b), -sin(g)*cos(b), cos(b)*cos(g)*sin(b2)+sin(b)*cos(b2)],
    #     [sin(g)*cos(b2),                       cos(g),        sin(g)*sin(b2)],
    #     [-sin(b)*cos(g)*cos(b2)-cos(b)*sin(b2), sin(b)*sin(g), -sin(b)*cos(g)*sin(b2)+cos(b)*cos(b2)]
    # ])
    R2 = np.array(
        [
            [cb * cg * cb2 - sb2 * sb, -sg * cb, cb * cg * sb2 + sb * cb2],
            [sg * cb2, cg, sg * sb2],
            [-sb * cg * cb2 - cb * sb2, sb * sg, -sb * cg * sb2 + cb * cb2],
        ],
        dtype=np.float64,
    )

    # Undo the phase first, then return to the region-centered frame.
    # Combined transformation: R = R2 @ R1.
    R = R2 @ R1
    return R @ xyz


@njit(cache=True)
def get_spot_phase_coord(xyz, inclination_angle, phase):
    """Rotate an active region's boundary to the requested stellar phase.

    Rotate the initial observer-frame coordinates from spot_location_init
    around the inclined stellar rotation axis. The observer frame stays fixed:
    x points toward the observer, while y and z span the sky plane. Longitude
    and latitude are already included in xyz and are not applied again.

    Parameters
    ----------
    xyz : ndarray, shape (3, n_points)
        Boundary coordinates at the initial phase, in stellar-radius units.
    inclination_angle : float
        Stellar inclination angle in degrees: 0 is pole-on, 90 is equator-on.
    phase : float
        Rotation phase in cycles; increasing phase gives a positive
        right-handed rotation about the stellar axis.

    Returns
    -------
    xyz_phase : ndarray, same shape as xyz
        Rotated boundary coordinates in the observer frame, including any
        points on the hidden hemisphere. Phases one cycle apart are equivalent.
    """
    # This matrix's sine signs make psi=-2*pi*phase a forward phase rotation.
    # For equator-on viewing, a positive quarter cycle rotates +x toward +y.
    psi = -2.0 * np.pi * phase
    inc = np.deg2rad(inclination_angle)

    # Unit rotation axis a = (cos(inc), 0, sin(inc)) in the observer frame.
    # It lies along x for pole-on viewing and along z for equator-on viewing.
    ax0 = np.cos(inc)
    ax1 = 0.0
    ax2 = np.sin(inc)

    c = np.cos(psi)
    s = np.sin(psi)
    omc = 1.0 - c

    # Full trigonometric form for rotation around the inclined stellar axis:
    # R = np.array([
    #     [(1-cos(psi))*cos(inc)*cos(inc) + cos(psi), sin(psi)*sin(inc),  (1-cos(psi))*cos(inc)*sin(inc)],
    #     [-sin(psi)*sin(inc),                       cos(psi),          sin(psi)*cos(inc)],
    #     [(1-cos(psi))*sin(inc)*cos(inc),           -sin(psi)*cos(inc), (1-cos(psi))*sin(inc)*sin(inc) + cos(psi)]
    # ])
    # The code below uses ax0=cos(inc), ax1=0, ax2=sin(inc), c=cos(psi), s=sin(psi).
    R = np.array(
        [
            [omc * ax0 * ax0 + c, omc * ax0 * ax1 + s * ax2, omc * ax0 * ax2 - s * ax1],
            [omc * ax1 * ax0 - s * ax2, omc * ax1 * ax1 + c, omc * ax1 * ax2 + s * ax0],
            [omc * ax2 * ax0 + s * ax1, omc * ax2 * ax1 - s * ax0, omc * ax2 * ax2 + c],
        ],
        dtype=np.float64,
    )

    return R @ xyz


@njit(cache=True)
def spot_area(xyz, grid_size):
    """Estimate the projected grid bounds used to scan an active region.

    Find the y-z extent of visible boundary samples. For a region crossing the
    stellar limb, extend selected bounds toward the disk edge to account for
    surface pixels beyond those samples. The caller then tests pixels within
    this box for actual region membership.

    Parameters
    ----------
    xyz : ndarray, shape (3, n_points)
        Boundary coordinates in stellar-radius units. The x axis points
        toward the observer; points with x >= 0 are considered visible.
    grid_size : int
        Number of pixels along each projected surface axis.

    Returns
    -------
    visibility_flag : int
        1 if any boundary sample is visible, otherwise 0.
    y_start, z_start, y_stop, z_stop : int
        Bounds for range(y_start, y_stop) and range(z_start, z_stop): starts are
        inclusive and stops exclusive. All bounds are zero if no sample is visible.

    Notes
    -----
    Bounds use finite boundary sampling and the existing limb-crossing correction.
    Stops include one extra grid position beyond the rounded-up maximum;
    spot_scan determines actual membership in this conservative scan box.
    """
    x = xyz[0, :]
    y = xyz[1, :]
    z = xyz[2, :]

    # Select samples on the observer-facing hemisphere, including the limb.
    visible_mask = x >= 0.0
    n_visible = np.sum(visible_mask)
    n_hidden = x.shape[0] - n_visible

    if n_visible == 0:
        # No visible boundary samples: return an empty scan box.
        return 0, 0, 0, 0, 0

    # Visible boundary samples provide the initial projected extents.
    min_y = np.min(y[visible_mask])
    max_y = np.max(y[visible_mask])
    min_z = np.min(z[visible_mask])
    max_z = np.max(z[visible_mask])

    if (n_visible > 0) and (n_hidden > 0):
        # When the region crosses the limb, boundary extrema may miss surface
        # pixels near the disk edge. Retain the existing axis-crossing correction.
        if min_y * max_y < 0:
            # The visible boundary crosses y=0; extend the dominant z side.
            if min_z < 0:
                if np.abs(min_z) >= np.abs(max_z):
                    min_z = -1
                else:
                    max_z = 1
            else:
                max_z = 1

        if min_z * max_z < 0:
            # The visible boundary crosses z=0; extend the dominant y side.
            if min_y < 0:
                if np.abs(min_y) >= np.abs(max_y):
                    min_y = -1
                else:
                    max_y = 1
            else:
                max_y = 1

    # Use the same projected coordinate spacing as spot_scan.
    grid_step = 2.0 / grid_size

    # Floor the lower bounds so scanning starts at or before each minimum.
    # Convert projected coordinates to pixel indices: index = (1 + coord) / grid_step.
    y_start = max(0, int(np.floor((1.0 + min_y) / grid_step)))
    z_start = max(0, int(np.floor((1.0 + min_z) / grid_step)))

    # range excludes its stop: include the rounded-up maximum with +1.
    # This extra scan margin also covers small rounding errors at grid boundaries;
    # spot_scan still tests every candidate pixel for actual region membership.
    y_stop = min(grid_size, int(np.ceil((1.0 + max_y) / grid_step)) + 1)
    z_stop = min(grid_size, int(np.ceil((1.0 + max_z) / grid_step)) + 1)

    return 1, y_start, z_start, y_stop, z_stop


@njit(cache=True)
def rndup(value: float, decimal_places: int) -> float:
    """Round to the nearest decimal value, with halfway ties away from zero.

    Used to select the integer-bin part of a spectral Doppler shift. Unlike
    NumPy's ties-to-even rounding, 1.5 becomes 2.0 and -1.5 becomes -2.0.

    Parameters
    ----------
    value : float
        Value to round.
    decimal_places : int
        Number of decimal places; use zero for integer-bin rounding.

    Returns
    -------
    float
        Rounded value, retaining its sign.
    """
    # Round the magnitude, then restore the original sign.
    sign = 1.0 if value >= 0.0 else -1.0
    scaled_magnitude = abs(value) * (10.0**decimal_places)
    # Adding 0.5 before flooring sends halfway ties away from zero.
    rounded_magnitude = np.floor(scaled_magnitude + 0.5)
    return sign * rounded_magnitude / (10.0**decimal_places)


@njit(cache=True)
def shifting_spectrum(
    spectrum_vel_grid: np.ndarray,
    spectrum: np.ndarray,
    vel_shift: float,
    n_spectrum_bins: int,
) -> np.ndarray:
    """Apply the fractional-bin part of a Doppler shift on a uniform velocity grid.

    Approximate spectrum(spectrum_vel_grid[i] - vel_shift) by linearly
    blending each sample with one neighbor. A positive shift uses the previous
    bin; a negative shift uses the next bin. Values beyond either edge are
    held at the nearest endpoint. The caller handles integer-bin placement
    and padding separately.

    Parameters
    ----------
    spectrum_vel_grid : ndarray
        Increasing, uniformly spaced velocity coordinates, with at least two bins.
    spectrum : ndarray
        Spectral values on that grid.
    vel_shift : float
        Residual velocity shift, in the same units as spectrum_vel_grid.
        Its magnitude should be smaller than one bin; larger shifts are clamped
        to one bin rather than treated as a full Doppler shift.
    n_spectrum_bins : int
        Number of spectral bins to shift.

    Returns
    -------
    ndarray
        Shifted spectrum as a new float64 array. Zero grid spacing returns a copy.
    """
    shifted_spectrum = np.empty(n_spectrum_bins, dtype=np.float64)

    vel_step = spectrum_vel_grid[1] - spectrum_vel_grid[0]
    if vel_step == 0.0:
        for i in range(n_spectrum_bins):
            shifted_spectrum[i] = spectrum[i]
        return shifted_spectrum

    shift_fraction = vel_shift / vel_step

    if shift_fraction >= 0.0:
        # Positive shifts sample toward the previous velocity bin.
        if shift_fraction >= 1.0:
            # Clamp defensively; the caller supplies a fractional-bin shift.
            shift_fraction = 1.0
        for i in range(n_spectrum_bins):
            current_value = spectrum[i]
            if i == 0:
                neighbor_value = current_value
            else:
                neighbor_value = spectrum[i - 1]
            shifted_spectrum[i] = (
                1.0 - shift_fraction
            ) * current_value + shift_fraction * neighbor_value
    else:
        # Negative shifts sample toward the next velocity bin.
        blend_fraction = -shift_fraction
        if blend_fraction >= 1.0:
            blend_fraction = 1.0
        for i in range(n_spectrum_bins):
            current_value = spectrum[i]
            if i == n_spectrum_bins - 1:
                neighbor_value = current_value
            else:
                neighbor_value = spectrum[i + 1]
            shifted_spectrum[i] = (
                1.0 - blend_fraction
            ) * current_value + blend_fraction * neighbor_value

    return shifted_spectrum


@njit(cache=True)
def integrate_quiet_flux(
    rotation_vel,
    inclination_angle,
    limb_a1,
    limb_a2,
    grid_size,
    spectrum_vel_grid,
    quiet_spectrum,
    vel_half_range,
    n_padded_bins,
    n_spectrum_bins,
):
    """Integrate a quiet flux spectrum over the projected stellar disk.

    Doppler shift the photospheric spectrum by each pixel's projected rotation
    velocity, apply quadratic limb darkening, and sum the contributions over
    the disk. Also retain each pixel's spectral-bin sum for surface plotting.

    At fixed y, the input spectrum and rotation velocity are unchanged, so
    shift the spectrum once per y index. Limb darkening remains pixel-dependent.
    Pad shifted spectra using their endpoint values.

    Parameters
    ----------
    rotation_vel : float
        Equatorial rotation velocity, in the same units as spectrum_vel_grid.
    inclination_angle : float
        Rotation-axis inclination angle to the line of sight, in degrees.
    limb_a1, limb_a2 : float
        Quadratic limb-darkening coefficients: 1 - a1*(1-mu) - a2*(1-mu)**2.
    grid_size : int
        Number of pixels along each projected surface axis.
    spectrum_vel_grid : ndarray, shape (n_spectrum_bins,)
        Uniform velocity coordinates of the input spectrum.
    quiet_spectrum : ndarray, shape (n_spectrum_bins,)
        Quiet photospheric flux spectrum used for every disk pixel.
    vel_half_range : float
        Half-width of the padded velocity grid.
    n_padded_bins : int
        Number of bins in the integrated, padded spectrum.
    n_spectrum_bins : int
        Number of bins in each input spectrum.

    Returns
    -------
    integrated_stellar_spectrum : ndarray, shape (n_padded_bins,)
        Sum of the shifted, padded spectra over projected disk pixels.
    quiet_flux_map : ndarray, shape (grid_size, grid_size)
        Spectral-bin sums per pixel, returned with axes (z, y). Pixels outside
        the projected disk are zero. These sums are not wavelength integrals.
    """

    integrated_stellar_spectrum = np.zeros(n_padded_bins, dtype=np.float64)

    # Projected coordinates span [-1, 1]; the padded velocity grid spans
    # [-vel_half_range, vel_half_range] with n_padded_bins - 1 intervals.
    grid_step = 2 / grid_size
    vel_step = 2 * vel_half_range / (n_padded_bins - 1)

    sin_inclination_angle = sin(inclination_angle * np.pi / 180.0)

    # Store summed flux per grid for 2D disk surface plotting
    quiet_flux_map = np.zeros((grid_size, grid_size), dtype=np.float64)

    # Scan projected y columns, then z pixels within each column.
    for iy in range(grid_size):
        y = -1 + iy * grid_step
        # Projected rotation velocity depends only on y.
        local_vel = y * rotation_vel * sin_inclination_angle

        # Flux spectra and rotational velocity are constant at fixed y.
        # Shift once per y index; limb darkening and accumulation remain per pixel.
        shift_bins = local_vel / vel_step
        integer_shift_bins = rndup(shift_bins, 0)
        residual_vel = local_vel - integer_shift_bins * vel_step
        vel_shift = residual_vel
        shifted_spectrum = shifting_spectrum(
            spectrum_vel_grid, quiet_spectrum, vel_shift, n_spectrum_bins
        )

        # Preserve the existing integer-bin shift and boundary extrapolation.
        left_pad_stop = int(
            (n_padded_bins - n_spectrum_bins) / 2.0 + integer_shift_bins
        )
        spectrum_start = left_pad_stop
        spectrum_stop = int(n_spectrum_bins + left_pad_stop)

        for iz in range(grid_size):
            quiet_pixel_sum = 0.0

            z = -1 + iz * grid_step
            if (y * y + z * z) <= 1:
                # mu is the viewing-angle cosine: 1 at disk center, 0 at the limb.
                mu = np.sqrt(1 - (y * y + z * z))

                # Quadratic limb darkening scales the flux contribution at this pixel.
                limb_scale = 1 - limb_a1 * (1 - mu) - limb_a2 * (1 - mu) ** 2

                left_edge_value = shifted_spectrum[0]
                for j in range(left_pad_stop):
                    integrated_stellar_spectrum[j] += left_edge_value * limb_scale
                    quiet_pixel_sum += left_edge_value * limb_scale

                for j in range(spectrum_start, spectrum_stop):
                    spectrum_index = j - spectrum_start  # 0 .. n_spectrum_bins-1
                    spectrum_value = shifted_spectrum[spectrum_index]
                    integrated_stellar_spectrum[j] += spectrum_value * limb_scale
                    quiet_pixel_sum += spectrum_value * limb_scale

                # Right pad
                right_edge_value = shifted_spectrum[n_spectrum_bins - 1]
                for j in range(spectrum_stop, n_padded_bins):
                    integrated_stellar_spectrum[j] += right_edge_value * limb_scale
                    quiet_pixel_sum += right_edge_value * limb_scale

                quiet_flux_map[iy, iz] = quiet_pixel_sum

    return integrated_stellar_spectrum, quiet_flux_map.T


@njit(cache=True)
def integrate_quiet_intensity(
    rotation_vel,
    inclination_angle,
    grid_size,
    spectrum_vel_grid,
    quiet_intensity_spectra,
    vel_half_range,
    n_padded_bins,
    n_spectrum_bins,
    mu_grid,
):
    """Integrate quiet specific intensity over the projected stellar disk.

    Select the photospheric spectrum at the nearest mu for each pixel, Doppler
    shift it by the projected rotation velocity, and sum the contributions over
    the disk. Also retain each pixel's spectral-bin sum for surface plotting.
    The mu-dependent spectra already describe limb darkening; no additional
    limb-darkening scaling is applied.

    At fixed y, rotation velocity is unchanged. Reuse the shifted spectrum for
    consecutive pixels selecting the same nearest-mu bin. Mu interpolation
    would require also tracking its weights. Pad spectra using endpoint values.

    Parameters
    ----------
    rotation_vel : float
        Equatorial rotation velocity, in the same units as spectrum_vel_grid.
    inclination_angle : float
        Rotation-axis inclination angle to the line of sight, in degrees.
    grid_size : int
        Number of pixels along each projected surface axis.
    spectrum_vel_grid : ndarray, shape (n_spectrum_bins,)
        Uniform velocity coordinates of the input spectra.
    quiet_intensity_spectra : ndarray, shape (n_spectrum_bins, n_mu)
        Quiet photospheric specific intensity spectra, one column per mu.
    vel_half_range : float
        Half-width of the padded velocity grid.
    n_padded_bins : int
        Number of bins in the integrated, padded spectrum.
    n_spectrum_bins : int
        Number of bins in each input spectrum.
    mu_grid : ndarray, shape (n_mu,)
        Viewing-angle cosines corresponding to the intensity columns.

    Returns
    -------
    integrated_stellar_spectrum : ndarray, shape (n_padded_bins,)
        Sum of the shifted, padded spectra over projected disk pixels.
    quiet_flux_map : ndarray, shape (grid_size, grid_size)
        Spectral-bin sums per pixel, returned with axes (z, y). Pixels outside
        the projected disk are zero. These sums are not wavelength integrals.
    """

    integrated_stellar_spectrum = np.zeros(n_padded_bins, dtype=np.float64)

    # Projected coordinates span [-1, 1]; the padded velocity grid spans
    # [-vel_half_range, vel_half_range] with n_padded_bins - 1 intervals.
    grid_step = 2 / grid_size
    vel_step = 2 * vel_half_range / (n_padded_bins - 1)

    sin_inclination_angle = sin(inclination_angle * np.pi / 180.0)

    quiet_flux_map = np.zeros((grid_size, grid_size), dtype=np.float64)

    shifted_spectrum = np.empty(n_spectrum_bins, dtype=np.float64)

    # Scan projected y columns, then z pixels within each column.
    for iy in range(grid_size):
        y = -1 + iy * grid_step
        # Projected rotation velocity depends only on y.
        local_vel = y * rotation_vel * sin_inclination_angle

        # Rotation and integer-bin padding depend only on y.
        shift_bins = local_vel / vel_step
        integer_shift_bins = rndup(shift_bins, 0)
        residual_vel = local_vel - integer_shift_bins * vel_step
        vel_shift = residual_vel
        left_pad_stop = int(
            (n_padded_bins - n_spectrum_bins) / 2.0 + integer_shift_bins
        )
        spectrum_start = left_pad_stop
        spectrum_stop = int(n_spectrum_bins + left_pad_stop)
        previous_mu_index = -1

        for iz in range(grid_size):
            quiet_pixel_sum = 0.0

            z = -1 + iz * grid_step
            if (y * y + z * z) <= 1:
                mu = np.sqrt(1 - (y * y + z * z))

                # Nearest-mu spectra supply the angular dependence directly.
                mu_index = nearest_mu_index(mu_grid, mu)
                # The same y index and nearest-mu bin select an identical spectrum.
                # Mu interpolation would require including interpolation weights.
                if mu_index != previous_mu_index:
                    quiet_spectrum = quiet_intensity_spectra[:, mu_index]
                    shifted_spectrum = shifting_spectrum(
                        spectrum_vel_grid, quiet_spectrum, vel_shift, n_spectrum_bins
                    )
                    previous_mu_index = mu_index

                # Preserve the original boundary extrapolation and pixel sum order.
                left_edge_value = shifted_spectrum[0]
                for j in range(left_pad_stop):
                    integrated_stellar_spectrum[j] += left_edge_value
                    quiet_pixel_sum += left_edge_value

                for j in range(spectrum_start, spectrum_stop):
                    spectrum_index = j - spectrum_start  # 0 .. n_spectrum_bins-1
                    spectrum_value = shifted_spectrum[spectrum_index]
                    integrated_stellar_spectrum[j] += spectrum_value
                    quiet_pixel_sum += spectrum_value

                # Right pad
                right_edge_value = shifted_spectrum[n_spectrum_bins - 1]
                for j in range(spectrum_stop, n_padded_bins):
                    integrated_stellar_spectrum[j] += right_edge_value
                    quiet_pixel_sum += right_edge_value

                quiet_flux_map[iy, iz] = quiet_pixel_sum

    return integrated_stellar_spectrum, quiet_flux_map.T


@njit(cache=True, fastmath=True)
def spot_scan_flux(
    rotation_vel,
    inclination_angle,
    photosphere_t,
    grid_size,
    spectral_lookup,
    star_LD_coeff,
    spectrum_vel_grid,
    quiet_spectrum,
    vel_half_range,
    n_padded_bins,
    n_spectrum_bins,
    surface_t_map,
):
    """Integrate flux spectra over active pixels for one phase.

    For each active pixel in the [iy, iz] temperature map, select its spectrum
    by temperature and Doppler shift it using the local projected rotation
    velocity. Apply quadratic limb darkening to the shifted spectrum, then
    accumulate the active contribution and the quiet contribution it replaces.
    The caller forms the stellar spectrum as quiet_star + active - replaced_quiet.

    Pixels at fixed y share the same velocity, regardless of z. Reuse the shifted
    quiet spectrum while y is unchanged; active reuse also requires the same
    spectral lookup index. Limb darkening is evaluated separately for each pixel.
    Reused spectra are local to this call, with no persistent spectrum cache.

    Returns
    -------
    integrated_stellar, integrated_quiet : ndarray, shape (n_padded_bins,)
        Summed active and replaced-quiet spectra on the padded velocity grid.
    active_flux_map, quiet_flux_map : ndarray, shape (grid_size, grid_size)
        Per-pixel sums over padded spectral bins, without a bin-width factor.
        Maps are returned as [iz, iy]; unprocessed pixels remain zero.
    """
    _, ld_coeffs, ld_keys, ld_vals = star_LD_coeff
    _, spectra, _, sp_keys, sp_vals = spectral_lookup

    # Integrate only regions that replace the photospheric intensity.
    mask = (surface_t_map != 0) & (surface_t_map != np.float32(photosphere_t))
    act_idx_y, act_idx_z = np.where(mask)

    integrated_stellar = np.zeros(n_padded_bins, np.float64)
    integrated_quiet = np.zeros(n_padded_bins, np.float64)
    active_flux_map = np.zeros((grid_size, grid_size), np.float64)
    quiet_flux_map = np.zeros((grid_size, grid_size), np.float64)

    # Photospheric LD coefficients are shared; the limb factor varies per pixel.
    iq0 = nearest_temperature_index(photosphere_t, ld_keys, ld_vals)
    limba1_q = ld_coeffs[iq0, 0]
    limba2_q = ld_coeffs[iq0, 1]

    previous_quiet_y_index = -1
    quiet_shift = np.empty(n_spectrum_bins, np.float64)
    previous_active_y_index = -1
    previous_active_spectrum = -1
    active_shift = np.empty(n_spectrum_bins, np.float64)

    delta_v_spec = (2.0 * vel_half_range) / (n_padded_bins - 1.0)
    sin_inclination_angle = np.sin(inclination_angle * np.pi / 180.0)

    for i in range(act_idx_y.size):
        iy = int(act_idx_y[i])
        iz = int(act_idx_z[i])

        y = -1.0 + iy * (2.0 / grid_size)
        z = -1.0 + iz * (2.0 / grid_size)

        delta = y * rotation_vel * sin_inclination_angle

        # Active spectrum row and LD coeffs
        sp_idx = nearest_temperature_index(surface_t_map[iy, iz], sp_keys, sp_vals)
        active_spectrum = spectra[sp_idx]  # shape: (n_spectrum_bins,)

        ld_idx = nearest_temperature_index(surface_t_map[iy, iz], ld_keys, ld_vals)
        limba1_a = ld_coeffs[ld_idx, 0]
        limba2_a = ld_coeffs[ld_idx, 1]

        # Limb factor at this pixel
        r_cos = np.sqrt(1.0 - (y * y + z * z))
        limb_quiet = (
            1.0 - limba1_q * (1.0 - r_cos) - limba2_q * (1.0 - r_cos) * (1.0 - r_cos)
        )
        limb_active = (
            1.0 - limba1_a * (1.0 - r_cos) - limba2_a * (1.0 - r_cos) * (1.0 - r_cos)
        )

        # Split the Doppler shift into bin placement and a residual interpolation.
        quotient = rndup(delta / delta_v_spec, 0)
        v_shift = delta - quotient * delta_v_spec

        # At fixed y, both the quiet spectrum and Doppler shift are unchanged.
        # Reuse the shifted spectrum; limb darkening remains pixel-dependent.
        if iy != previous_quiet_y_index:
            quiet_shift = shifting_spectrum(
                spectrum_vel_grid, quiet_spectrum, v_shift, n_spectrum_bins
            )
            previous_quiet_y_index = iy
        # Active temperatures can vary at fixed y, so also check the spectrum.
        # Limb darkening is still applied separately for every pixel.
        if iy != previous_active_y_index or sp_idx != previous_active_spectrum:
            active_shift = shifting_spectrum(
                spectrum_vel_grid, active_spectrum, v_shift, n_spectrum_bins
            )
            previous_active_y_index = iy
            previous_active_spectrum = sp_idx

        # Clamp the shifted window; padding uses the spectrum's edge intensities.
        start = int((n_padded_bins - n_spectrum_bins) / 2.0 + quotient)
        start = max(0, min(start, n_padded_bins))
        stop = min(start + n_spectrum_bins, n_padded_bins)

        # Flux limb factors vary per pixel, so retain per-pixel spectral sums.
        active_pixel_sum = 0.0
        quiet_pixel_sum = 0.0

        # Left pad
        if start > 0:
            active_left_value = active_shift[0]
            quiet_left_value = quiet_shift[0]
            for j in range(start):
                integrated_stellar[j] += active_left_value * limb_active
                integrated_quiet[j] += quiet_left_value * limb_quiet
            active_pixel_sum += active_left_value * limb_active * start
            quiet_pixel_sum += quiet_left_value * limb_quiet * start

        # Middle block
        for j in range(stop - start):
            active_value = active_shift[j]
            quiet_value = quiet_shift[j]
            output_index = start + j
            integrated_stellar[output_index] += active_value * limb_active
            integrated_quiet[output_index] += quiet_value * limb_quiet
            active_pixel_sum += active_value * limb_active
            quiet_pixel_sum += quiet_value * limb_quiet

        # Right pad
        if stop < n_padded_bins:
            active_right_value = active_shift[n_spectrum_bins - 1]
            quiet_right_value = quiet_shift[n_spectrum_bins - 1]
            for j in range(stop, n_padded_bins):
                integrated_stellar[j] += active_right_value * limb_active
                integrated_quiet[j] += quiet_right_value * limb_quiet
            active_pixel_sum += (
                active_right_value * limb_active * (n_padded_bins - stop)
            )
            quiet_pixel_sum += quiet_right_value * limb_quiet * (n_padded_bins - stop)

        active_flux_map[iy, iz] = active_pixel_sum
        quiet_flux_map[iy, iz] = quiet_pixel_sum

    return integrated_stellar, integrated_quiet, active_flux_map.T, quiet_flux_map.T


@njit(cache=True, fastmath=True)
def _sum_padded_intensity(shifted, start, stop, n_padded_bins, n_spectrum_bins):
    """Sum the existing padded spectrum once for an intensity group."""
    value = 0.0
    if start > 0:
        value += shifted[0] * start
    for j in range(stop - start):
        value += shifted[j]
    if stop < n_padded_bins:
        value += shifted[n_spectrum_bins - 1] * (n_padded_bins - stop)
    return value


@njit(cache=True, fastmath=True)
def _accumulate_intensity_group(target, shifted, count, start, stop, n_spectrum_bins):
    """Add a group's intensity to target in place, including edge padding.
    Update each segment directly without creating a padded temporary array.
    """
    if count == 0:
        return
    for j in range(start):
        target[j] += shifted[0] * count
    for j in range(stop - start):
        target[start + j] += shifted[j] * count
    for j in range(stop, target.size):
        target[j] += shifted[n_spectrum_bins - 1] * count


@njit(cache=True, fastmath=True)
def spot_scan_int(
    rotation_vel,
    inclination_angle,
    photosphere_t,
    grid_size,
    spectral_lookup,
    spectrum_vel_grid,
    quiet_spectrum,
    vel_half_range,
    n_padded_bins,
    n_spectrum_bins,
    surface_t_map,
):
    """Integrate intensity spectra over active pixels for one phase.

    For each active pixel in the [iy, iz] temperature map, select its spectrum
    by temperature and nearest mu, then Doppler shift it using the local projected
    rotation velocity. Limb darkening is already included in the mu-dependent
    intensities. Accumulate the active contribution and the quiet contribution
    it replaces. The caller forms the stellar spectrum as
    quiet_star + active - replaced_quiet.

    Pixels at fixed y share the same velocity, regardless of z. Group containing
    pixels sharing y and nearest-mu index; active groups also require the same
    spectral lookup index. Add each group's shifted spectrum times its pixel
    count once, and reuse its spectral sum for the individual pixel maps.
    Reused spectra are local to this call, with no persistent spectrum cache.

    Returns
    -------
    integrated_stellar, integrated_quiet : ndarray, shape (n_padded_bins,)
        Summed active and replaced-quiet spectra on the padded velocity grid.
    active_flux_map, quiet_flux_map : ndarray, shape (grid_size, grid_size)
        Per-pixel sums over padded spectral bins, without a bin-width factor.
        Maps are returned as [iz, iy]; unprocessed pixels remain zero.

    Notes
    -----
    Grouping assumes nearest-mu selection; mu interpolation would require
    accounting for interpolation weights. Grouped sums may differ in roundoff
    from per-pixel accumulation.
    """
    _, spectra, mu_s, sp_keys, sp_vals = spectral_lookup

    # Active-pixel indices (nonzero + not photosphere)
    mask = (surface_t_map != 0) & (surface_t_map != np.float32(photosphere_t))
    act_idx_y, act_idx_z = np.where(mask)

    integrated_stellar = np.zeros(n_padded_bins, np.float64)
    integrated_quiet = np.zeros(n_padded_bins, np.float64)
    active_flux_map = np.zeros((grid_size, grid_size), np.float64)
    quiet_flux_map = np.zeros((grid_size, grid_size), np.float64)

    active_shift = np.empty(n_spectrum_bins, np.float64)
    quiet_shift = np.empty(n_spectrum_bins, np.float64)

    previous_active_y_index = previous_active_mu_index = -1
    previous_active_spectrum = -1
    previous_quiet_y_index = previous_quiet_mu_index = -1

    active_pixel_count = quiet_pixel_count = 0
    active_start = active_stop = quiet_start = quiet_stop = 0

    active_pixel_sum = quiet_pixel_sum = 0.0

    delta_v_spec = 2.0 * vel_half_range / (n_padded_bins - 1.0)
    sin_inclination_angle = np.sin(inclination_angle * np.pi / 180.0)

    for i in range(act_idx_y.size):
        iy = int(act_idx_y[i])
        iz = int(act_idx_z[i])

        y = -1.0 + iy * (2.0 / grid_size)
        z = -1.0 + iz * (2.0 / grid_size)

        delta = y * rotation_vel * sin_inclination_angle

        sp_idx = nearest_temperature_index(surface_t_map[iy, iz], sp_keys, sp_vals)
        mu_index = nearest_mu_index(mu_s, np.sqrt(1.0 - (y * y + z * z)))

        # Split the Doppler shift into bin placement and a residual interpolation.
        quotient = rndup(delta / delta_v_spec, 0)
        v_shift = delta - quotient * delta_v_spec

        # Clamp the shifted window; padding uses the spectrum's edge intensities.
        start = int((n_padded_bins - n_spectrum_bins) / 2.0 + quotient)
        start = max(0, min(start, n_padded_bins))
        stop = min(start + n_spectrum_bins, n_padded_bins)

        # A changed key ends the previous group: add its spectrum times its count.
        # Quiet groups depend on y and mu, independently of active temperature.
        if iy != previous_quiet_y_index or mu_index != previous_quiet_mu_index:
            _accumulate_intensity_group(
                integrated_quiet,
                quiet_shift,
                quiet_pixel_count,
                quiet_start,
                quiet_stop,
                n_spectrum_bins,
            )
            quiet_shift = shifting_spectrum(
                spectrum_vel_grid, quiet_spectrum[:, mu_index], v_shift, n_spectrum_bins
            )
            quiet_start, quiet_stop = start, stop
            quiet_pixel_sum = _sum_padded_intensity(
                quiet_shift, start, stop, n_padded_bins, n_spectrum_bins
            )
            quiet_pixel_count = 0
            previous_quiet_y_index, previous_quiet_mu_index = iy, mu_index
        # Active groups also depend on the spectrum selected by temperature.
        if (
            iy != previous_active_y_index
            or mu_index != previous_active_mu_index
            or sp_idx != previous_active_spectrum
        ):
            _accumulate_intensity_group(
                integrated_stellar,
                active_shift,
                active_pixel_count,
                active_start,
                active_stop,
                n_spectrum_bins,
            )
            active_shift = shifting_spectrum(
                spectrum_vel_grid,
                spectra[sp_idx, :, mu_index],
                v_shift,
                n_spectrum_bins,
            )
            active_start, active_stop = start, stop
            active_pixel_sum = _sum_padded_intensity(
                active_shift, start, stop, n_padded_bins, n_spectrum_bins
            )
            active_pixel_count = 0
            (
                previous_active_y_index,
                previous_active_mu_index,
                previous_active_spectrum,
            ) = (iy, mu_index, sp_idx)

        active_pixel_count += 1
        quiet_pixel_count += 1

        # Keep individual pixel maps, reusing the spectral sum within each group.
        active_flux_map[iy, iz] = active_pixel_sum
        quiet_flux_map[iy, iz] = quiet_pixel_sum

    # No subsequent pixel closes the final groups; zero counts skip accumulation.
    _accumulate_intensity_group(
        integrated_stellar,
        active_shift,
        active_pixel_count,
        active_start,
        active_stop,
        n_spectrum_bins,
    )
    _accumulate_intensity_group(
        integrated_quiet,
        quiet_shift,
        quiet_pixel_count,
        quiet_start,
        quiet_stop,
        n_spectrum_bins,
    )
    return integrated_stellar, integrated_quiet, active_flux_map.T, quiet_flux_map.T


@njit
def spot_scan(
    inclination_angle,
    grid_size,
    spot_size,
    longitude,
    latitude,
    phase,
    idx_miny,
    idx_minz,
    idx_maxy,
    idx_maxz,
    active_t,
    surface_t_map,
    photosphere_t,
):
    """Assign an active region's temperature to visible surface pixels.

    Reconstruct each candidate pixel on the observer-facing unit hemisphere,
    then undo the stellar phase and region placement rotations. In the region's
    reference frame, the spot is centered on +x: membership requires
    x > 0 and x**2 >= 1 - spot_size**2.

    Assign active_t only where the current pixel temperature equals
    photosphere_t. Pixels already assigned a different temperature by an
    earlier region are left unchanged.

    Update surface_t_map directly and return the same array.

    Scan only pixels inside the projected stellar disk (y**2 + z**2 < 1),
    and also excluding pixels exactly on its outer edge.

    Parameters
    ----------
    inclination_angle : float
        Rotation-axis inclination angle to the line of sight, in degrees.
    grid_size : int
        Number of pixels along each projected surface axis.
    spot_size : float
        Boundary-circle radius in stellar-radius units, not an angular radius.
    longitude, latitude : float
        Region coordinates in degrees.
    phase : float
        Stellar rotation phase in cycles.
    idx_miny, idx_minz, idx_maxy, idx_maxz : int
        Candidate scan bounds from spot_area; minima are inclusive and maxima
        exclusive. Actual region membership is tested within these bounds.
    active_t : float
        Temperature assigned to member pixels.
    surface_t_map : ndarray, shape (grid_size, grid_size)
        Surface temperature map indexed as [iy, iz].
    photosphere_t : float
        Quiet temperature; compared using float32 to match the surface map.

    Returns
    -------
    ndarray
        The input surface_t_map, updated in place.
    """

    return _scan_surface_values(
        inclination_angle, grid_size, spot_size, longitude, latitude, phase,
        idx_miny, idx_minz, idx_maxy, idx_maxz, active_t, surface_t_map,
        np.float32(photosphere_t),
    )


@njit(cache=True)
def scan_region_ids(
    inclination_angle, grid_size, spot_size, longitude, latitude, phase,
    idx_miny, idx_minz, idx_maxy, idx_maxz, region_id, surface_region_map,
):
    """Assign a region ID to visible pixels using the shared cap geometry.

    Zero marks unassigned photosphere; positive IDs mark active regions.
    The first region owns overlap pixels, even when its temperature or
    brightness equals the photosphere. The caller initializes off-disk pixels
    to -1. Pixels exactly on the stellar limb retain their initial value.
    Update surface_region_map in place and return the same integer array.
    """
    return _scan_surface_values(
        inclination_angle, grid_size, spot_size, longitude, latitude, phase,
        idx_miny, idx_minz, idx_maxy, idx_maxz, region_id, surface_region_map, 0,
    )


@njit(cache=True, inline="always")
def _scan_surface_values(
    inclination_angle, grid_size, spot_size, longitude, latitude, phase,
    idx_miny, idx_minz, idx_maxy, idx_maxz, active_value, surface_map, quiet_value,
):
    """Apply the projected cap membership test to a caller-selected surface field."""
    delta_grid = 2 / grid_size
    xyz_actual = np.ones(3, dtype=np.float64)  # actual coordinates
    xyz_initial = np.ones(
        3, dtype=np.float64
    )  # coordinates transformed back to the initial configuration

    # Scan y columns, then z pixels within the candidate box.
    for iy in range(idx_miny, idx_maxy):
        y = -1 + iy * delta_grid  # y between -1 to 1
        xyz_actual[1] = y

        for iz in range(idx_minz, idx_maxz):
            z = -1 + iz * delta_grid  # z between -1 to 1

            if (y * y + z * z) < 1:
                # x is the visible hemisphere's viewing-angle cosine (mu).
                xyz_actual[0] = np.sqrt(1 - (y * y + z * z))
                xyz_actual[2] = z

                # Undo phase and placement so the region is centered on +x.
                xyz_initial = spot_inverse_rotation(
                    xyz_actual, longitude, latitude, inclination_angle, phase
                )

                # On the unit sphere, the cap (spot) boundary has x**2 = 1 - radius**2.
                # Require x > 0 to reject the corresponding cap on the opposite side.
                if (xyz_initial[0] ** 2 >= (1 - spot_size**2)) and (xyz_initial[0] > 0):
                    if surface_map[iy, iz] == quiet_value:
                        # Preserve pixels already assigned by earlier regions.
                        surface_map[iy, iz] = active_value

    return surface_map


def integrate_active_phase(
    inclination_angle,
    photosphere_t,
    phases,
    grid_size,
    spectral_lookup,
    star_LD_coeff,
    rotation_vel,
    spectrum_vel_grid,
    quiet_spectrum,
    vel_half_range,
    n_padded_bins,
    n_spectrum_bins,
    spectrum_mode,
    surface_t_cube,
    phase_index,
):
    """Integrate active-pixel spectra for one rotation phase.

    Select surface_t_cube[phase_index] and call the flux or intensity integration
    kernel. Return both the active contribution and the quiet contribution
    over those same pixels; the caller uses the latter to replace quiet flux
    in the full-disk spectrum. This worker does not assemble the final spectrum.

    Parameters
    ----------
    phase_index : int
        Index into phases and surface_t_cube.
    phases : ndarray
        Rotation phases in cycles.
    surface_t_cube : ndarray, shape (n_phases, grid_size, grid_size)
        Precomputed temperature maps with surface axes (y, z).
    spectrum_mode : str
        Spectral mode: 'flux' or 'intensity'.
    star_LD_coeff : tuple
        Packed limb-darkening lookup arrays used in flux mode only.

    Other inputs specify the stellar geometry, spectral lookup and velocity
    grids passed to spot_scan_flux or spot_scan_int.

    Returns
    -------
    active_spectrum, quiet_replacement_spectrum : ndarray, shape (n_padded_bins,)
        Integrated active and replacement quiet contributions.
    phase : float
        phases[phase_index], preserving the input phase value.
    active_flux_map, quiet_replacement_flux_map : ndarray, shape (grid_size, grid_size)
        Spectral-bin sums over active pixels, with display axes (z, y).
    """
    surface_t_map = surface_t_cube[phase_index]

    if spectrum_mode == "flux":
        (
            active_spectrum,
            quiet_replacement_spectrum,
            active_flux_map,
            quiet_replacement_flux_map,
        ) = spot_scan_flux(
            rotation_vel,
            inclination_angle,
            photosphere_t,
            grid_size,
            spectral_lookup,
            star_LD_coeff,
            spectrum_vel_grid,
            quiet_spectrum,
            vel_half_range,
            n_padded_bins,
            n_spectrum_bins,
            surface_t_map,
        )
    else:  # 'intensity'
        (
            active_spectrum,
            quiet_replacement_spectrum,
            active_flux_map,
            quiet_replacement_flux_map,
        ) = spot_scan_int(
            rotation_vel,
            inclination_angle,
            photosphere_t,
            grid_size,
            spectral_lookup,
            spectrum_vel_grid,
            quiet_spectrum,
            vel_half_range,
            n_padded_bins,
            n_spectrum_bins,
            surface_t_map,
        )

    return (
        active_spectrum,
        quiet_replacement_spectrum,
        phases[phase_index],
        active_flux_map,
        quiet_replacement_flux_map,
    )


def integrate_active_phases(
    phases,
    n_phases,
    rotation_vel,
    inclination_angle,
    photosphere_t,
    grid_size,
    spectral_lookup,
    star_LD_coeff,
    spectrum_vel_grid,
    quiet_spectrum,
    vel_half_range,
    n_padded_bins,
    n_spectrum_bins,
    cpu_use,
    spectrum_mode,
    surface_t_cube,
    silence_pbar,
):
    """Integrate active-pixel spectra across the requested rotation phases.

    Bind common inputs to the phase worker, submit one task per phase index,
    and collect results in input phase order. Each task uses its precomputed
    temperature map. With cpu_use=1, pqdm runs directly without subprocesses;
    larger values use a process pool.

    Parameters
    ----------
    phases : ndarray
        Rotation phases in cycles.
    n_phases : int
        Number of phases to process, matching phases and surface_t_cube.
    surface_t_cube : ndarray, shape (n_phases, grid_size, grid_size)
        Precomputed temperature maps with surface axes (y, z).
    cpu_use : int
        Number of phase workers.
    spectrum_mode : str
        Spectral mode: 'flux' or 'intensity'.
    silence_pbar : bool
        Whether to hide the progress display.

    Other inputs specify the stellar geometry, spectral lookup, limb darkening
    and velocity grids passed unchanged to integrate_active_phase.

    Returns
    -------
    active_spectra, quiet_replacement_spectra : ndarray, shape (n_phases, n_padded_bins)
        Active and replacement quiet contributions for each phase.
    phases : ndarray, shape (n_phases,)
        Input phases stored as float32.
    active_flux_maps, quiet_replacement_flux_maps : ndarray, shape (n_phases, grid_size, grid_size)
        Spectral-bin sums over active pixels, with surface axes (z, y).
    """
    # Allocate one result slot per phase; display maps have axes (z, y).
    active_spectra = np.empty(
        (n_phases, n_padded_bins), dtype=np.float64
    )
    quiet_replacement_spectra = np.empty(
        (n_phases, n_padded_bins), dtype=np.float64
    )

    output_phases = np.empty((n_phases,), dtype=np.float32)

    active_flux_maps = np.empty(
        (n_phases, grid_size, grid_size), dtype=np.float64
    )
    quiet_replacement_flux_maps = np.empty(
        (n_phases, grid_size, grid_size), dtype=np.float64
    )

    fixed_inputs = (
        inclination_angle,
        photosphere_t,
        phases,
        grid_size,
        spectral_lookup,
        star_LD_coeff,
        rotation_vel,
        spectrum_vel_grid,
        quiet_spectrum,
        vel_half_range,
        n_padded_bins,
        n_spectrum_bins,
        spectrum_mode,
        surface_t_cube,
    )

    # Bind common inputs so pqdm supplies only the phase index to each task.
    # Process tasks serialize these inputs; partial does not provide shared memory.
    worker = partial(integrate_active_phase, *fixed_inputs)

    try:
        phase_results = pqdm(
            np.arange(n_phases, dtype=int), worker, n_jobs=cpu_use, disable=silence_pbar
        )
    except KeyboardInterrupt:
        logger.warning("Phase integration interrupted by Ctrl+C.")
        # Stop the simulation when phase processing is interrupted.
        sys.exit(1)

    # pqdm returns results in submitted phase order, not completion order.
    for phase_index, (
        active_spectrum,
        quiet_replacement_spectrum,
        phase,
        active_flux_map,
        quiet_replacement_flux_map,
    ) in enumerate(phase_results):
        active_spectra[phase_index, :] = active_spectrum
        quiet_replacement_spectra[phase_index, :] = quiet_replacement_spectrum
        output_phases[phase_index] = phase
        active_flux_maps[phase_index, :, :] = active_flux_map
        quiet_replacement_flux_maps[phase_index, :, :] = quiet_replacement_flux_map

    return (
        active_spectra,
        quiet_replacement_spectra,
        output_phases,
        active_flux_maps,
        quiet_replacement_flux_maps,
    )


# Surface maps and disk-integrated spectra

def build_surface_temperature_cube(
    star, phases, grid_size, nrho, active_regions_number, active_region_box
):
    """Build a projected surface temperature map for each rotation phase.

    Initialize every phase with a quiet photosphere. Place each circular region,
    rotate its boundary to each phase, and use spot_area to limit the candidate
    scan. spot_scan tests actual pixel membership and assigns the region's
    temperature only where the current value equals the photospheric temperature.
    Pixels already assigned a different temperature are left unchanged.

    Region order is supplied by the caller. Black-region markers are stored
    unchanged; this function does not assign spectra or calculate emission.
    The resulting cube can be reused across spectral bands.

    Parameters
    ----------
    star : Star
        Photospheric temperature in Kelvin and stellar inclination angle in degrees.
    phases : array_like
        Rotation phases in cycles.
    grid_size : int
        Number of pixels along each projected surface axis.
    nrho : int
        Requested samples around each active-region boundary.
    active_regions_number : int
        Number of regions to apply from active_region_box.
    active_region_box : tuple of arrays
        Longitudes and latitudes in degrees, boundary-circle radii in
        stellar-radius units, and absolute temperatures or black-region markers.
        Radii are already converted from configured projected area percentages.

    Returns
    -------
    surface_t_cube : ndarray, shape (n_phases, grid_size, grid_size)
        Float32 temperature maps indexed as [phase, iy, iz]. Pixels outside
        the initial disk mask start at zero.
    y, z : ndarray, shape (grid_size,)
        Float64 coordinate axes for the initial disk mask and downstream
        coverage calculations and plots.

    Notes
    -----
    The mask and returned coordinates use -1 + index*2/grid_size, matching
    spot_scan and the spectral integration kernels. The disk mask includes
    the stellar limb; spot_scan assigns regions only inside it.
    """
    # Region arrays share the same order; radii are already in stellar-radius units.
    (
        active_region_longitudes,
        active_region_latitudes,
        active_region_radii,
        active_region_temperatures,
    ) = active_region_box

    # Initialize quiet maps; zero marks pixels outside the initial disk mask.
    n_phases = len(phases)
    temp_quiet = star.photosphere_temperature

    y = -1.0 + np.arange(grid_size) * (2.0 / grid_size)
    z = -1.0 + np.arange(grid_size) * (2.0 / grid_size)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    disk_mask = (Y * Y + Z * Z) <= 1.0

    single_layer = np.where(disk_mask, temp_quiet, 0).astype(np.float32)

    # Store the phase stack as float32 to limit memory use.
    surface_t_cube = np.empty((n_phases, grid_size, grid_size), dtype=np.float32)
    surface_t_cube[...] = single_layer[None, ...]

    logger.info("\nBuilding active-region surface maps...")
    # Construct each placed boundary once, then rotate it for all phases.
    for region_index in range(active_regions_number):
        xyz = spot_location_init(
            active_region_radii[region_index],
            active_region_longitudes[region_index],
            active_region_latitudes[region_index],
            star.inclination_angle,
            nrho,
        )
        for phase_index in range(n_phases):
            phase_boundary = get_spot_phase_coord(xyz, star.inclination_angle, phases[phase_index])
            # Hidden boundaries return an empty scan box. Stops are exclusive.
            _, idx_miny, idx_minz, idx_maxy, idx_maxz = spot_area(
                phase_boundary, grid_size
            )
            surface_t_cube[phase_index] = spot_scan(
                star.inclination_angle,
                grid_size,
                active_region_radii[region_index],
                active_region_longitudes[region_index],
                active_region_latitudes[region_index],
                phases[phase_index],
                idx_miny,
                idx_minz,
                idx_maxy,
                idx_maxz,
                np.round(active_region_temperatures[region_index], 4),
                surface_t_cube[phase_index],
                temp_quiet,
            )

    return surface_t_cube, y, z


def build_surface_region_cube(star, phases, grid_size, nrho, region_geometry):
    """Label projected surface pixels by region idx for each rotation phase.

    Use the same disk coordinates, cap rotations and scan bounds as the
    temperature builder. Store -1 outside the disk, 0 for quiet photosphere
    and IDs 1..N in the supplied region order.
    In overlap pixels, keep the region assigned first. Later regions do not
    overwrite it, even if its temperature or brightness equals the quiet
    photosphere.

    Parameters
    ----------
    star : Star
        Only inclination angle in degrees is required; no temperature is read.
    phases : array_like
        Rotation phases in cycles.
    grid_size : int
        Number of pixels along each projected surface axis.
    nrho : int
        Samples around each active-region boundary.
    region_geometry : tuple of three arrays
        Longitudes and latitudes in degrees, and boundary-circle radii in
        stellar-radius units. Lookup arrays must follow this same region order,
        with quiet photosphere at index 0.

    Returns
    -------
    surface_region_cube : ndarray, shape (n_phases, grid_size, grid_size)
        Int32 IDs indexed as [phase, iy, iz]. Exclude -1 before indexing any
        temperature or brightness lookup array.
    y, z : ndarray
        Projected coordinate axes, using -1 + index*2/grid_size.

    Notes
    -----
    Gaussian integration uses these IDs to select region brightness. IDs retain
    ownership for photospheric-temperature or unit-brightness regions;
    the temperature builder used by PHX/SYNMAST does not retain those identities.
    """
    longitudes, latitudes, radii = (np.asarray(values) for values in region_geometry)
    if any(values.ndim != 1 for values in (longitudes, latitudes, radii)):
        raise ValueError("Region geometry arrays must be one-dimensional.")
    if not (longitudes.size == latitudes.size == radii.size):
        raise ValueError("Region geometry arrays must have matching lengths.")

    y = -1.0 + np.arange(grid_size) * (2.0 / grid_size)
    z = -1.0 + np.arange(grid_size) * (2.0 / grid_size)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    disk = yy * yy + zz * zz <= 1.0

    quiet_ids = np.where(disk, 0, -1).astype(np.int32)
    surface_region_cube = np.empty((len(phases), grid_size, grid_size), dtype=np.int32)
    surface_region_cube[...] = quiet_ids[None, ...]

    for index in range(radii.size):
        boundary = spot_location_init(
            radii[index], longitudes[index], latitudes[index], star.inclination_angle, nrho
        )
        for phase_index, phase in enumerate(phases):
            rotated = get_spot_phase_coord(boundary, star.inclination_angle, phase)

            _, y_start, z_start, y_stop, z_stop = spot_area(rotated, grid_size)

            scan_region_ids(
                star.inclination_angle,
                grid_size,
                radii[index],
                longitudes[index],
                latitudes[index],
                phase,
                y_start,
                z_start,
                y_stop,
                z_stop,
                index + 1,
                surface_region_cube[phase_index],
            )
    return surface_region_cube, y, z


def calculate_activity_signal_spec(
    grid_size,
    star,
    phases,
    Spec,
    instrumental_resolution,
    cpu_use,
    save_dir,
    band,
    save_spectra,
    spectrum_mode,
    silence_pbar,
    *,
    surface_t_cube,
    y,
    z,
    spectral_output: dict | None = None,
):
    """Assemble disk-integrated stellar spectra from quiet and active contributions.

    Integrate the full quiet photosphere once. For each phase, use the supplied
    temperature map to integrate active pixels and the quiet emission those
    pixels would otherwise contribute. Replace that emission using

        stellar_spectrum = quiet_baseline - quiet_on_active_pixels + active_emission.

    Apply the same replacement to surface flux maps. Crop spectral padding,
    then scale non-photometric spectra by their 99th percentile and apply
    instrumental broadening when requested. Photometric spectra retain their
    integrated scale. Continuum fitting is performed separately downstream.

    Parameters
    ----------
    grid_size : int
        Number of pixels along each projected surface axis.
    star : Star
        Stellar parameters and prepared limb-darkening coefficients.
    phases : array_like
        Rotation phases in cycles.
    Spec : dict
        Component temperatures mapped to PreparedSpectrum objects for this band.
    instrumental_resolution : float
        Instrument resolving power; zero disables instrumental broadening.
    cpu_use : int
        Number of workers for phase integration.
    save_dir : str
        Output directory name under outputs.
    band : str
        Configured band name; 'photo' selects the photometric treatment.
    save_spectra : bool
        Whether to collect spectra and their quiet/active contributions for export.
    spectrum_mode : str
        Spectral mode: 'flux' or 'intensity'.
    silence_pbar : bool
        Whether to hide phase-integration progress.
    surface_t_cube : ndarray, shape (len(phases), grid_size, grid_size)
        Precomputed temperature maps indexed as [phase, iy, iz], shared across
        bands and read without modification.
    y, z : ndarray
        Projected coordinate axes returned by build_surface_temperature_cube.
    spectral_output : dict or None
        Optional collection for the caller to add continuum-normalized spectra
        and save one archive. No spectral copies are added when export is disabled.

    Returns
    -------
    phases : ndarray, shape (len(phases),)
        Input phases returned in their original order.
    integrated_stellar_spectrum : ndarray, shape (len(phases), n_spectrum_bins)
        Combined spectra on the original velocity grid after the treatment above.
    quiet_2d_surface_flux_sum : ndarray, shape (len(phases), grid_size, grid_size)
        Quiet spectral-bin sums over active pixels only, with axes (phase, z, y).
    final_2d_surface_flux_sum : ndarray, shape (len(phases), grid_size, grid_size)
        Full-disk spectral-bin sums after active-region replacement, with axes
        (phase, z, y). Maps precede cropping, normalization and broadening.

    Notes
    -----
    Surface-map values are bin sums, not wavelength integrals. Temperature and
    coverage diagnostics are calculated separately from the shared surface maps.
    """
    # --- Step 1: Quiet photosphere baseline (spectrum + surface flux map) ---
    Spec_photosphere = Spec[np.round(star.photosphere_temperature, 4)]
    if spectrum_mode == "flux":
        u1, u2 = star.LD_coeff[f"{band}_{np.round(star.photosphere_temperature, 4)}"]
        integrated_stellar_spectrum, quiet_2d_surface_flux_sum = integrate_quiet_flux(
            star.rotation_vel,
            star.inclination_angle,
            u1,
            u2,
            grid_size,
            Spec_photosphere.spectrum_vel_grid,
            Spec_photosphere.spectrum,
            Spec_photosphere.vel_half_range,
            Spec_photosphere.n_padded_bins,
            Spec_photosphere.n_spectrum_bins,
        )
    else:  # 'intensity'
        integrated_stellar_spectrum, quiet_2d_surface_flux_sum = integrate_quiet_intensity(
            star.rotation_vel,
            star.inclination_angle,
            grid_size,
            Spec_photosphere.spectrum_vel_grid,
            Spec_photosphere.spectrum,
            Spec_photosphere.vel_half_range,
            Spec_photosphere.n_padded_bins,
            Spec_photosphere.n_spectrum_bins,
            Spec_photosphere.mu_grid,
        )

    final_2d_surface_flux_sum = quiet_2d_surface_flux_sum.copy()

    n_phases = len(phases)
    # Pack temperature lookup arrays, then integrate active and replacement quiet emission.
    if spectrum_mode == "flux":
        star_LD_coeff = pack_limb_darkening_lookup(star.LD_coeff, band)
        spectral_lookup = pack_spectral_lookup(Spec, "flux")
    else:
        star_LD_coeff = (None, None)
        spectral_lookup = pack_spectral_lookup(Spec, "intensity")

    (
        active_spectra,
        quiet_replacement_spectra,
        output_phases,
        active_2d_surface_flux_sum,
        quiet_2d_surface_flux_sum,
    ) = integrate_active_phases(
        phases,
        n_phases,
        star.rotation_vel,
        star.inclination_angle,
        star.photosphere_temperature,
        grid_size,
        spectral_lookup,
        star_LD_coeff,
        Spec_photosphere.spectrum_vel_grid,
        Spec_photosphere.spectrum,
        Spec_photosphere.vel_half_range,
        Spec_photosphere.n_padded_bins,
        Spec_photosphere.n_spectrum_bins,
        cpu_use,
        spectrum_mode,
        surface_t_cube,
        silence_pbar,
    )

    # Use the same unpadded interval for saved components and the combined spectrum.
    spectrum_start = (
        Spec_photosphere.n_padded_bins - len(Spec_photosphere.spectrum_vel_grid)
    ) // 2
    spectrum_stop = spectrum_start + len(Spec_photosphere.spectrum_vel_grid)

    # Keep unprocessed contributions on the same unpadded grid.
    if save_spectra:
        arrays = dict(
            quiet_photosphere_flux=integrated_stellar_spectrum[spectrum_start:spectrum_stop],
            active_replacement_quiet_flux=np.asarray(quiet_replacement_spectra)[:, spectrum_start:spectrum_stop],
            active_region_flux=np.asarray(active_spectra)[:, spectrum_start:spectrum_stop],
            phase=np.asarray(output_phases),
            velocity_kms=np.asarray(Spec_photosphere.spectrum_vel_grid),
            wavelength_angstrom=Spec_photosphere.wavelength_grid,
        )
        if spectral_output is not None:
            spectral_output.update(arrays)

    # Replace quiet emission at active pixels in both spectra and surface maps.
    integrated_stellar_spectrum = (
        integrated_stellar_spectrum
        - np.asarray(quiet_replacement_spectra)
        + np.asarray(active_spectra)
    )
    final_2d_surface_flux_sum = (
        final_2d_surface_flux_sum
        - np.asarray(quiet_2d_surface_flux_sum)
        + np.asarray(active_2d_surface_flux_sum)
    )

    # Remove padding after combining the Doppler-shifted contributions.
    integrated_stellar_spectrum = integrated_stellar_spectrum[
        :, spectrum_start:spectrum_stop
    ]

    if save_spectra:
        arrays["spotted_flux"] = integrated_stellar_spectrum

    # Scale spectroscopic outputs; retain absolute integrated scale for photometry.
    if band != "photo":
        integrated_stellar_spectrum = np.array(
            [f / np.percentile(f, 99) for f in integrated_stellar_spectrum],
            dtype=np.float64,
        )

    # Apply instrumental broadening to the combined spectroscopic outputs.
    if (instrumental_resolution != 0) and (band != "photo"):
        integrated_stellar_spectrum = apply_instrumental_broadening(
            instrumental_resolution,
            Spec_photosphere.wavelength_grid,
            integrated_stellar_spectrum,
        )

    if save_spectra:
        if band != "photo":
            arrays["processed_spotted_flux"] = integrated_stellar_spectrum
        if spectral_output is None:
            # Standalone integration can export contributions without continuum fitting.
            directory = Path("outputs") / save_dir / "simulated_spotted_spectra"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez(directory / f"{band}{identity_suffix(star)}.npz", **arrays, **identity_arrays(star))
        else:
            spectral_output.update(arrays)

    return (
        output_phases,
        integrated_stellar_spectrum,
        quiet_2d_surface_flux_sum,
        final_2d_surface_flux_sum,
    )


# Integration lookup and grid helpers

def grid_resolution_check(grid: int, warn_threshold: float = 0.999) -> float:
    """
    Estimate how well a grid of size (grid x grid) approximates the area of a unit disk
    when summing cell areas for cells whose *centers* lie inside the disk.

    Args:
        grid          : number of cells along each axis (grid x grid total).
        warn_threshold: fraction of π below which we log a warning (default 99.9%).

    Returns:
        accuracy (float): (estimated area) / π

    This diagnostic uses cell centers. Surface integration instead samples
    -1 + index*2/grid, so this is not the exact area of its disk mask.
    """
    # Each axis spans [-1, 1] → total length = 2. Each cell has width = step.
    step = 2.0 / grid

    # Build 1D coordinates of **cell centers** along y and z:
    #   centers at: -1 + (i + 0.5) * step   for i=0..grid-1
    centers = -1.0 + (np.arange(grid, dtype=np.float32) + 0.5) * step

    # Internal surface indexing is [iy, iz]: Y varies along axis 0 and Z along
    # axis 1. Displaying its transpose places y horizontally and z vertically.
    Y, Z = np.meshgrid(centers, centers, indexing="ij")

    # Boolean mask: True where the cell center is inside the unit circle (y^2 + z^2 <= 1)
    inside = (Y * Y + Z * Z) <= 1.0
    approx_area = inside.sum(dtype=np.int64) * (step * step)

    # Compare to the analytic area π (unit radius)
    accuracy = approx_area / np.pi

    # Warn if we’re below the requested accuracy threshold
    if accuracy < warn_threshold:
        logger.warning(
            "Grid resolution check WARNING.\n"
            f"   Grid {grid}x{grid} has area/π = {accuracy:1.2%} < {warn_threshold*100:.1f}%.\n"
            "   Consider increasing the grid size to improve conservation."
        )
    return accuracy


def pack_limb_darkening_lookup(star_LD_coeff, band):
    """Pack one band's coefficients in ascending temperature order.

    Keep only entries for `band` (keys like 'FeOH_3600.0').
    Returns:
      temps_tab   : (N,) float64, sorted ascending
      coeffs_tab  : (N,K) float64, aligned with temps_tab
      lut_keys    : (N,) float64 (temperatures)
      lut_vals    : (N,) int32  (row indices 0..N-1)
    """
    # The final underscore separates the temperature; match the full band name.
    sub = {k: v for k, v in star_LD_coeff.items() if k.rsplit("_", 1)[0] == band}
    if not sub:
        raise ValueError(f"No LD entries for band {band!r}")

    temps, coeffs = [], []
    for k, v in sub.items():
        tstr = k.split("_")
        temps.append(np.round(float(tstr[-1]), 4))
        coeffs.append(v)  # e.g., (u1,u2) or (u1,u2,u3)

    temps = np.asarray(temps, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)  # (N,K)

    order = np.argsort(temps)
    temps_tab = temps[order]
    coeffs_tab = coeffs[order]

    # compact LUT (no giant dense array)
    lut_keys = np.round(temps_tab, 4)  # [T1, T2, ...]
    lut_vals = np.arange(temps_tab.size, dtype=np.int32)  # [0, 1, 2, ...]

    return temps_tab, coeffs_tab, lut_keys, lut_vals



@njit
def nearest_temperature_index(temp, lut_keys, lut_vals):
    """Return the spectrum or coefficient row nearest to a temperature.

    Compare float32 map temperatures with float64 lookup keys. Equal-distance
    ties retain the first key; this is a nearest lookup, not an equality test.
    """
    best = 0
    bestd = abs(lut_keys[0] - temp)
    for i in range(1, lut_keys.size):
        d = abs(lut_keys[i] - temp)
        if d < bestd:
            best = i
            bestd = d
    return lut_vals[best]



def pack_spectral_lookup(spec, spectrum_mode):
    """Pack flux or intensity spectra in ascending temperature order.

    spec: dict[temperature -> PreparedSpectrum] with:
          .spectrum   (n_spectrum_bins, n_mu) if 'intensity', or (n_spectrum_bins,) if 'flux'
          .mu_grid    (n_mu,) only when spectrum_mode == 'intensity'
    Returns:
      temps_tab     : (N,) float64, sorted
      spectrum_tab  : (N, n_spectrum_bins[, n_mu]) float64
      mu_s_tab      : (n_mu,) float32 or None
      lut_keys      : (N,) float64
      lut_vals      : (N,) int32
    """
    want_mu = spectrum_mode == "intensity"
    temps, spect_list = [], []
    mu_grid = None

    for T, obj in spec.items():
        temps.append(np.round(T, 4))
        spect_list.append(np.asarray(obj.spectrum, dtype=np.float64))
        if want_mu and mu_grid is None:
            # Use the first spectrum's mu grid for the shared angular lookup.
            # PHX mu varies with temperature; this retains the existing approximation.
            mu_grid = np.asarray(obj.mu_grid, dtype=np.float32)

    temps = np.asarray(temps, dtype=np.float64)
    spectrum_tab = np.stack(spect_list, axis=0)  # (N, ...), float64

    order = np.argsort(temps)
    temps_tab = temps[order]
    spectrum_tab = spectrum_tab[order]

    mu_s_tab = mu_grid if want_mu else None

    lut_keys = np.round(temps_tab, 4)
    lut_vals = np.arange(temps_tab.size, dtype=np.int32)

    return temps_tab, spectrum_tab, mu_s_tab, lut_keys, lut_vals



@njit
def nearest_mu_index(mu_grid, mu):
    """
    Return index of element in mu_grid closest to scalar mu.
    Numba-friendly, no allocations, no argmin.
    """
    best = 0
    bestd = abs(mu_grid[0] - mu)
    for i in range(1, mu_grid.size):
        d = abs(mu_grid[i] - mu)
        if d < bestd:
            best = i
            bestd = d
    return best

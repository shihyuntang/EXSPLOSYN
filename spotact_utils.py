import sys, os, logging
logger = logging.getLogger()

import numpy as np
from numba import njit
from scipy import signal,interpolate
from specutils import Spectrum1D
from specutils.analysis import equivalent_width
import astropy.units as u

def _apply_instrumental_broadening(inst_reso, vel_grid, spec_flux):
    """
    Apply instrumental broadening to a spectrum by convolving it with a Gaussian kernel.

    Args:
        inst_reso (float): Instrumental resolution (R = lambda / delta_lambda).
        vel_grid (np.ndarray): Radial velocity grid in km/s.
        spec_flux (np.ndarray): 2D array of spectra (N_phases x N_vel).

    Returns:
        np.ndarray: Broadened spectra (same shape as input).
    """
    c_kms = 299792.458  # speed of light in km/s

    fwhm = c_kms / inst_reso
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Gaussian kernel on the RV grid
    kernel = np.exp(-vel_grid**2 / (2 * sigma**2))

    broadened_spec = np.empty_like(spec_flux)
    for i in range(len(spec_flux)):
        # Convolve (1 - flux) to preserve line depth, then invert back
        convolved = signal.convolve(1 - spec_flux[i], kernel, mode='same')
        min_val = np.min(spec_flux[i])
        max_val = np.max(convolved)
        broadened_spec[i] = 1 - convolved * (1 - min_val) / max_val

    return broadened_spec

def get_equivalent_width(
        Star, flux_phi, visible_phi, band, intORflux, model_type):
    """
    Calculate the equivalent width ratio (OH/Fe) for a given band and all phases.

    Args:
        Star (Star): Star object.
        flux_phi (array): 2D array of flux for each phase.
        visible_phi (array): Array of phases.
        band (str): Band name.
        intORflux (str): 'flux' or 'intensity'.
        model_type (str): Model type.

    Returns:
        np.ndarray: Array of OH/Fe equivalent width ratios for all phases.
    """
    ew_fe1 = []
    ew_fe2 = []
    ew_oh0 = []

    ew_cal_regions = [
        [1.56230, 1.56290], 
        [1.56290, 1.56335], 
        [1.56335, 1.56395]
    ]

    band_center_wv = Star.bands2use[band][0]

    if intORflux == 'flux':
        vel = Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']
    elif intORflux == 'intensity':
        vel = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']

    c_kms = 299792.458
    del_wv = vel * band_center_wv / c_kms
    wave = (band_center_wv + del_wv) / 1e4  # Convert to microns
    nor_region = np.where((wave > 1.5610) & (wave < 1.5650))[0]
    
    for i in range(len(visible_phi)):
        flux = flux_phi[i]
        nor_level = np.percentile(flux[nor_region], 95)
        flux = flux / nor_level
        # --- EW ---
        for r_udx, rr in enumerate(ew_cal_regions):
            use_r = (wave >= rr[0]) & (wave <= rr[1])
            spec = Spectrum1D(
                spectral_axis=wave[use_r] * u.um, 
                flux=flux[use_r] * u.Unit('erg cm-2 s-1 AA-1')
            )
            ew = equivalent_width(spec)
            if r_udx == 0:
                ew_fe1.append(ew.value)
            elif r_udx == 1:
                ew_oh0.append(ew.value)
            elif r_udx == 2:
                ew_fe2.append(ew.value)

    oh_line = np.array(ew_oh0)
    fe_line = (np.array(ew_fe1) + np.array(ew_fe2)) / 2
    line_ratio = oh_line / fe_line

    return line_ratio

def get_line_depth(
        Star, CCFstar_tot, visible_phi, band, intORflux, model_type):
    """
    Measure the depth and velocity location of Fe and OH lines in the CCF for all phases.

    Args:
        Star (Star): Star object.
        CCFstar_tot (array): 2D array of CCFs for each phase.
        visible_phi (array): Array of phases.
        band (str): Band name.
        intORflux (str): 'flux' or 'intensity'.
        model_type (str): Model type.

    Returns:
        tuple: (depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, depth_l_fe2, depth_l_oh0)
            Each is a list of depths or velocities for Fe1, Fe2, and OH0 lines.
    """
    depth_d_fe1 = []
    depth_d_fe2 = []
    depth_d_oh0 = []
    depth_l_fe1 = []
    depth_l_fe2 = []
    depth_l_oh0 = []

    if intORflux == 'flux':
        vel = Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']
    elif intORflux == 'intensity':
        vel = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']

    for i in range(len(visible_phi)):
        CCFstar_tot_flip = 1 + -1 * CCFstar_tot[i]

        fe_1 = (vel > -150) & (vel < -50)
        oh_0 = (vel > -50) & (vel < 50)
        fe_2 = (vel > 50) & (vel < 250)

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


def grid_resolution_check(grid):
    """
    Check the accuracy of the grid area approximation for the stellar disk.

    The final spectra are approximated by summing all flux in each grid cell,
    so a finer grid better conserves the flux scale. This function checks how
    close the summed area of the grid cells inside the disk is to pi.

    Args:
        grid (int): Grid resolution (number of cells per axis).

    Returns:
        None. Prints a warning if the area is less than 99.9% of pi.
    """
    grid_side_length = 2. / grid  # Grid goes from -1 to 1, so total length is 2

    # Scan each cell on the grid (y-z plane)
    _area = 0.0
    for iy in range(grid):
        y = -1 + iy * grid_side_length  # y from -1 to 1
        for iz in range(grid):
            z = -1. + iz * grid_side_length  # z from -1 to 1
            if (y * y + z * z) <= 1:
                # If the cell is inside the disk, add its area
                _area += grid_side_length ** 2
    accuracy = _area / np.pi
    if accuracy < 0.999:
        logger.warning(
            f'Grid resolution check WARNING.\n'
            f'   Grid {grid}x{grid} has an area/pi={accuracy:1.2%} < 99.9%.\n'
            '   Consider increasing the number of grid to improve accuracy.'
        )


def detect_peaks(
        x, mph=None, mpd=1, threshold=0, edge='rising',
        kpsh=False, valley=False, show=False, ax=None):
    """
    Detect peaks in a 1D array.

    Args:
        x (array): 1D data array.
        mph (float, optional): Minimum peak height.
        mpd (int, optional): Minimum peak distance.
        threshold (float, optional): Minimum threshold between peak and neighbors.
        edge (str, optional): 'rising', 'falling', 'both', or None.
        kpsh (bool, optional): Keep peaks with same height even if closer than mpd.
        valley (bool, optional): If True, detect valleys (local minima) instead of peaks.
        show (bool, optional): If True, plot the result.
        ax (matplotlib axis, optional): Axis to plot on.

    Returns:
        ind (np.ndarray): Indices of detected peaks.
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # Find indices of all peaks
    dx = x[1:] - x[:-1]
    # Handle NaN's
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
    # Remove peaks near NaNs
    if ind.size and indnan.size:
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # First and last values cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # Remove peaks below minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # Remove peaks with neighbors too close in height
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # Remove small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # Keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # Remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        # _plot(x, mph, mpd, threshold, edge, valley, ax, ind)
    return ind

def a0cont(a0vcut, plot=False, poly_order=1, spline_k=1, mpd=100):
    """
    Performs first-pass fit to blaze shape of telluric standard spectrum.

    Args:
        a0vcut (array): Flux of telluric standard spectrum.
        plot (bool): If True, plot the result.
        poly_order (int): Polynomial order for initial fit.
        spline_k (int): Spline degree.
        mpd (int): Minimum peak distance for peak detection.

    Returns:
        continuum (np.ndarray): Best fit blaze model.
        peaks (np.ndarray): Indices of detected peaks.
    """
    x = np.arange(len(a0vcut))

    # Detect peaks that are at least separated by minimum peak distance
    peaks = detect_peaks(a0vcut, mpd=mpd)
    xtimes = 1

    for _ in range(xtimes):
        mask = np.ones(len(peaks), dtype=bool)

        # Fit a polynomial to the detected peaks
        f = np.polyfit(x[peaks], a0vcut[peaks], poly_order)
        q = np.poly1d(f)
        residual = a0vcut[peaks] - q(x[peaks])
        MAD = np.median(np.abs(residual - np.median(residual)))

        # Mask out outliers and low points
        mask[(a0vcut[peaks]/np.nanmedian(a0vcut) < .1)] = False
        mask[(a0vcut[peaks] < q(x[peaks])-2*MAD)] = False
        mask[(a0vcut[peaks] > q(x[peaks])+6*MAD)] = False
        peaks = peaks[mask]

    c = 0
    for smoothing in np.arange(1e6, 1e8, 1e6):
        # Fit a spline to the filtered peaks
        f = interpolate.UnivariateSpline(
            x[peaks], a0vcut[peaks], k=spline_k, s=smoothing)
        continuum = f(x)
        peaks2 = detect_peaks(continuum)
        if len(peaks2) == 1:
            c += 1
        if c == 2:
            break

    # If not converged, try larger smoothing
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

    return continuum, peaks


import matplotlib.pyplot as plt

def save_phase_pngs(
    cube, y, z, outdir="phase_maps",
    sentinel_offdisk=0, cmap="inferno",
    add_colorbar=True, phase_values=None,
):
    import os, numpy as np, matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    Y, Z = np.meshgrid(y, z, indexing="ij")           # Y,Z shaped (G,G): axis0=y, axis1=z
    disk_mask = (Y*Y + Z*Z) <= 1.0

    vals = cube[:, disk_mask].astype(np.float32)
    vals = vals[vals != sentinel_offdisk]
    vmin = np.percentile(vals, 1); vmax = np.percentile(vals, 99)
    if vmin == vmax: vmax = vmin + 1

    Npsi, G, _ = cube.shape
    extent = (-1, 1, -1, 1)                           # ★ x=y in [-1,1], y=z in [-1,1]

    for i in range(Npsi):
        frame = cube[i].astype(np.float32)
        mask2d = ~disk_mask | (frame == sentinel_offdisk)

        # ★ Transpose so horizontal axis = y, vertical axis = z
        frame_plot = np.ma.array(frame.T,  mask=mask2d.T)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300, constrained_layout=True)
        im = ax.imshow(
            frame_plot,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            aspect="equal",
        )

        # ★ Labels now match your diagram
        ax.set_xlabel("y (stellar radii)")
        ax.set_ylabel("z (stellar radii)")
        title = f"Surface Temperature — phase {i:03d}"
        if phase_values is not None:
            title += f" (ψ={phase_values[i]:.3f})"
        ax.set_title(title)

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
            cbar.set_label("Temperature (int16 units)")

        fig.savefig(os.path.join(outdir, f"phase_{i:03d}.png"))
        plt.close(fig)


def coverage_and_teff(
    cube,                   # shape (Npsi, G, G), int16 or float
    y, z,                   # grid coords in [-1, 1], len = G
    T_phot,                 # quiet photosphere temperature (same units as cube)
    sentinel_offdisk=0,
    scale=1.0,              # if you stored temps scaled (e.g., ×10), give that here
    weight="area",          # "area" (projected area), "mu" (cosθ), or a callable(Y,Z)->weights
):
    """
    Returns:
      f_spot:   total non-photosphere fraction per phase (Npsi,)
      f_cool:   fraction with T < T_phot per phase (Npsi,)
      f_hot:    fraction with T > T_phot per phase (Npsi,)
      Teff:     effective temperature per phase from <T^4>^(1/4) (Npsi,)
    """
    cube = cube.astype(np.float64) / scale
    G = cube.shape[1]
    Y, Z = np.meshgrid(y, z, indexing="ij")
    disk = (Y*Y + Z*Z) <= 1.0

    # weights
    if weight == "area":
        W = np.ones_like(Y, dtype=np.float64)
    elif weight == "mu":
        W = np.sqrt(np.clip(1.0 - (Y*Y + Z*Z), 0.0, 1.0))  # cosθ
    elif callable(weight):
        W = np.asarray(weight(Y, Z), dtype=np.float64)
    else:
        raise ValueError("weight must be 'area', 'mu', or a callable(Y,Z).")

    # mask off-disk & sentinel
    on = disk & (cube[0] != sentinel_offdisk)  # any phase works for sentinel/off-disk
    W = np.where(on, W, 0.0)
    Wsum = W.sum()
    if Wsum == 0:
        raise ValueError("No on-disk pixels found. Check grid/sentinel.")

    # Broadcast weights for all phases
    W3 = W[None, ...]  # (1,G,G) to (Npsi,G,G)

    # Classify pixels
    is_phot = (cube == T_phot)
    is_cool = (cube <  T_phot)
    is_hot  = (cube >  T_phot)

    # Fractions (weighted by chosen W)
    f_spot = ( (W3 * (~is_phot)).sum(axis=(1,2)) / Wsum )
    f_cool = ( (W3 * is_cool   ).sum(axis=(1,2)) / Wsum )
    f_hot  = ( (W3 * is_hot    ).sum(axis=(1,2)) / Wsum )

    # Effective temperature from area-weighted <T^4>^(1/4)
    T4_mean = (W3 * (cube**4)).sum(axis=(1,2)) / Wsum
    Teff = np.power(np.clip(T4_mean, 0.0, None), 0.25)

    return f_spot, f_cool, f_hot, Teff

import numpy as np

def teff_obs_rotavg(
    cube, y, z, sentinel_offdisk=0, scale=1.0, limb_darkening=None
):
    """
    Rotation-averaged observer Teff from the phase cube.

    Parameters
    ----------
    cube : (Npsi, G, G) array, temps in K*scale
    y,z  : grid coords in [-1,1], len=G
    sentinel_offdisk : value used off-disk
    scale : if you stored temps as int(T*scale)
    limb_darkening : None or callable(mu)->I(mu) (bolometric intensity law)

    Returns
    -------
    Teff_obs : scalar (K)
    """
    cube = cube.astype(np.float64) / scale
    Y, Z = np.meshgrid(y, z, indexing='ij')
    r2 = Y*Y + Z*Z
    disk = r2 <= 1.0
    mu = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

    # Weights for flux integration
    if limb_darkening is None:
        W = mu  # foreshortening only
    else:
        W = limb_darkening(mu)

    W = np.where(disk, W, 0.0)
    denom = W.sum()

    # Ignore off-disk sentinels
    mask_off = (cube == sentinel_offdisk)
    T4 = np.where(mask_off, 0.0, cube**4)

    # Per-phase flux proxy and rotation-average
    num = (T4 * W[None, ...]).sum(axis=(1, 2))
    T4_mean = num.mean() / denom
    return T4_mean**0.25


def fspot_global_from_cube(
    cube, y, z, T_phot, sentinel_offdisk=0, scale=1.0, mu_min=0.05
):
    """
    Estimate *global* (surface-area) spot filling factor from the phase cube,
    independent of spot temperature. Also returns projected (on-disk) f per phase.

    Idea: spots are pixels where T != T_phot. Convert projected pixels to
    surface-area weights with 1/mu (mu = cosθ), clip near the limb with mu_min,
    compute per-phase surface-area fractions, then average over phases.

    Parameters
    ----------
    cube : (Npsi, G, G) temps in K*scale
    y,z  : grid coords
    T_phot : quiet-photosphere temperature (in K)
    sentinel_offdisk : value used off-disk
    scale : if you stored temps as int(T*scale)
    mu_min : limb cutoff to keep 1/mu finite (typ. 0.03–0.1)

    Returns
    -------
    f_global : scalar, estimated global surface-area spot fraction
    f_proj_per_phase : (Npsi,) projected on-disk spot fraction per phase
    """
    cube = cube.astype(np.float64) / scale
    Y, Z = np.meshgrid(y, z, indexing='ij')
    r2 = Y*Y + Z*Z
    disk = r2 <= 1.0
    mu = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

    # On-disk indicator that also excludes sentinel
    on = disk & (cube[0] != sentinel_offdisk)

    # Temperature-agnostic spot mask: anything not equal to quiet T
    is_spot = (cube != T_phot) & on[None, ...]

    # --- Projected (on-disk) fractions per phase (good sanity check)
    W_proj = np.where(on, 1.0, 0.0)            # projected pixels weight equally
    denom_proj = W_proj.sum()
    f_proj_per_phase = (is_spot * W_proj[None, ...]).sum(axis=(1, 2)) / denom_proj

    # --- Deprojected surface-area estimate per phase using 1/mu
    # (avoid limb singularity with a small cutoff)
    A = np.where(on & (mu >= mu_min), 1.0 / mu, 0.0)
    denom_surf = A.sum()
    f_surf_per_phase = (is_spot * A[None, ...]).sum(axis=(1, 2)) / denom_surf

    # Rotation-average to estimate global
    f_global = f_surf_per_phase.mean()

    return f_global, f_proj_per_phase


def LD_dict2array(star_LD_coeff, band):
    """
    Keep only entries for `band` (keys like 'R_4300').
    Returns:
      temps_tab   : (N,) int16, sorted ascending
      coeffs_tab  : (N,K) float32, aligned with temps_tab
      lut_keys    : (N,) int32  (temperatures)
      lut_vals    : (N,) int32  (row indices 0..N-1)
    """
    sub = {k: v for k, v in star_LD_coeff.items() if k.startswith(f"{band}_")}
    if not sub:
        raise ValueError(f"No LD entries for band {band!r}")

    temps, coeffs = [], []
    for k, v in sub.items():
        _, tstr = k.split("_", 1)
        temps.append(int(tstr))
        coeffs.append(v)  # e.g., (u1,u2) or (u1,u2,u3)

    temps = np.asarray(temps, dtype=np.int32)
    coeffs = np.asarray(coeffs, dtype=np.float32)  # (N,K)

    order = np.argsort(temps)
    temps_tab  = temps[order].astype(np.int16)
    coeffs_tab = coeffs[order]

    # compact LUT (no giant dense array)
    lut_keys = temps_tab.astype(np.int32)                       # [T1, T2, ...]
    lut_vals = np.arange(temps_tab.size, dtype=np.int32)        # [0, 1, 2, ...]

    return temps_tab, coeffs_tab, lut_keys, lut_vals

@njit
def lut_lookup_eq(temp, lut_keys, lut_vals):
    """
    Compact O(N) lookup by equality. N is small, so this is cheap.
    Returns row index (>=0) or -1 if not found.
    """
    for i in range(lut_keys.size):
        if lut_keys[i] == temp:
            return lut_vals[i]
    return -1



def Spec_intensity_dict2array(spec, fluxORint):
    """
    spec: dict[int -> Spec] with:
          .intensity  (n_spec, n_mu) if 'intensity', or (n_spec,) if 'flux'
          .mu         (n_mu,) only when fluxORint == 'intensity'
    Returns:
      temps_tab     : (N,) int16, sorted
      spectrum_tab  : (N, n_spec[, n_mu]) float32
      mu_s_tab      : (n_mu,) float32 or None
      lut_keys      : (N,) int32
      lut_vals      : (N,) int32
    """
    want_mu = (fluxORint == "intensity")
    temps, spect_list = [], []
    mu_grid = None

    for T, obj in spec.items():
        temps.append(int(T))
        spect_list.append(np.asarray(obj.intensity, dtype=np.float32))
        if want_mu and mu_grid is None:
            # as miu are the same for all T, thus we only need one array, unlike spectrum
            mu_grid = np.asarray(obj.mu, dtype=np.float32)


    temps = np.asarray(temps, dtype=np.int32)
    spectrum_tab = np.stack(spect_list, axis=0)  # (N, ...), float32

    order = np.argsort(temps)
    temps_tab    = temps[order].astype(np.int16)
    spectrum_tab = spectrum_tab[order]

    mu_s_tab = mu_grid if want_mu else None

    lut_keys = temps_tab.astype(np.int32)
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
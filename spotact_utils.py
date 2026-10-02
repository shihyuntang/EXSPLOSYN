import sys, os, logging
logger = logging.getLogger()

import numpy as np
from numba import njit
from scipy import signal,interpolate
from specutils import Spectrum1D
from specutils.analysis import equivalent_width
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as const
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


import numpy as np

def get_equivalent_width(Star, flux_phi, visible_phi, band, intORflux, model_type):
    """
    Fast EW(OH)/EW(Fe) for all phases using pure NumPy.

    Equivalent width is computed as:
        EW = ∫ (1 - F_norm) dλ
    with F_norm = F / continuum, where the continuum level is the 95th percentile
    within 1.561–1.565 µm for each phase.

    Returns
    -------
    np.ndarray, shape (n_phase,)
        OH/Fe equivalent-width ratio per phase.
    """
    # --- Resolve velocity grid for this band at the photosphere temp ---
    key_v = f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v'
    if intORflux == 'flux':
        vel = Star.flux_spec[key_v]
    else:
        vel = Star.int_spec[key_v]
    vel = np.asarray(vel, dtype=np.float64, order='C')

    # --- Build wavelength array (µm), centered on band center like your code ---
    band_center_wv = float(Star.bands2use[band][0])  # in Angstrom
    c_kms = 299_792.458
    wave_A = band_center_wv + vel * (band_center_wv / c_kms)   # Å
    wave_um = wave_A / 1e4                                     # µm
    wave_um = np.asarray(wave_um, dtype=np.float64, order='C')

    # --- Masks: normalization region and EW sub-ranges (in µm) ---
    norm_mask = (wave_um > 1.5610) & (wave_um < 1.5650)

    # Fe, OH, Fe windows (µm)
    ew_windows = [
        (1.56230, 1.56290),  # Fe 1
        (1.56290, 1.56335),  # OH
        (1.56335, 1.56395),  # Fe 2
    ]
    masks = [(wave_um >= a) & (wave_um <= b) for (a, b) in ew_windows]

    # Pre-extract x arrays for integration (saves boolean indexing in the loop)
    x_fe1 = wave_um[masks[0]]
    x_oh  = wave_um[masks[1]]
    x_fe2 = wave_um[masks[2]]
    x_norm = wave_um[norm_mask]

    # --- Loop phases (need per-phase percentile) ---
    nphi = len(visible_phi)
    ew_fe1 = np.empty(nphi, dtype=np.float64)
    ew_fe2 = np.empty(nphi, dtype=np.float64)
    ew_oh  = np.empty(nphi, dtype=np.float64)

    for i in range(nphi):
        # ensure contiguous float64
        flux = np.asarray(flux_phi[i], dtype=np.float64, order='C')

        # Continuum estimate (95th percentile) in the normalization region
        # guard against empty or NaN-only slice
        y_norm = flux[norm_mask]
        nor_level = np.nanpercentile(y_norm, 95.0)
        
        if not np.isfinite(nor_level) or nor_level <= 0:
            # hopeless phase → set EWs to NaN
            ew_fe1[i] = ew_fe2[i] = ew_oh[i] = np.nan
            continue

        f_norm = flux / nor_level

        # 1 - normalized flux in each window
        y_fe1 = 1.0 - f_norm[masks[0]]
        y_oh  = 1.0 - f_norm[masks[1]]
        y_fe2 = 1.0 - f_norm[masks[2]]

        # Integrate with trapezoid in µm (units cancel in the ratio)
        # If a mask is empty (shouldn’t be), return NaN for that EW.
        ew_fe1[i] = np.trapz(y_fe1, x_fe1) if y_fe1.size else np.nan
        ew_oh[i]  = np.trapz(y_oh,  x_oh)  if y_oh.size  else np.nan
        ew_fe2[i] = np.trapz(y_fe2, x_fe2) if y_fe2.size else np.nan

    # Average the two Fe EWs, then ratio
    fe_mean = 0.5 * (ew_fe1 + ew_fe2)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = ew_oh / fe_mean
    # Avoid infinities from ~zero Fe mean
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio

# def get_line_depth(
#         Star, CCFstar_tot, visible_phi, band, intORflux, model_type):
#     """
#     Measure the depth and velocity location of Fe and OH lines in the CCF for all phases.

#     Args:
#         Star (Star): Star object.
#         CCFstar_tot (array): 2D array of CCFs for each phase.
#         visible_phi (array): Array of phases.
#         band (str): Band name.
#         intORflux (str): 'flux' or 'intensity'.
#         model_type (str): Model type.

#     Returns:
#         tuple: (depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, depth_l_fe2, depth_l_oh0)
#             Each is a list of depths or velocities for Fe1, Fe2, and OH0 lines.
#     """
#     depth_d_fe1 = []
#     depth_d_fe2 = []
#     depth_d_oh0 = []
#     depth_l_fe1 = []
#     depth_l_fe2 = []
#     depth_l_oh0 = []

#     if intORflux == 'flux':
#         vel = Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v']
#     elif intORflux == 'intensity':
#         vel = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v']

#     for i in range(len(visible_phi)):
#         CCFstar_tot_flip = 1 + -1 * CCFstar_tot[i]

#         fe_1 = (vel > -150) & (vel < -50)
#         oh_0 = (vel > -50) & (vel < 50)
#         fe_2 = (vel > 50) & (vel < 250)

#         fe_1_loc = np.argmax(CCFstar_tot_flip[fe_1])
#         oh_0_loc = np.argmax(CCFstar_tot_flip[oh_0])
#         fe_2_loc = np.argmax(CCFstar_tot_flip[fe_2])

#         depth_d_fe1.append(CCFstar_tot_flip[fe_1][fe_1_loc])
#         depth_d_fe2.append(CCFstar_tot_flip[fe_2][fe_2_loc])
#         depth_d_oh0.append(CCFstar_tot_flip[oh_0][oh_0_loc])
        
#         depth_l_fe1.append(vel[fe_1][fe_1_loc])
#         depth_l_fe2.append(vel[fe_2][fe_2_loc])
#         depth_l_oh0.append(vel[oh_0][oh_0_loc])

#     return depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, depth_l_fe2, depth_l_oh0




def grid_resolution_check(grid: int, warn_threshold: float = 0.999) -> float:
    """
    Estimate how well a grid of size (grid x grid) approximates the area of a unit disk
    when summing cell areas for cells whose *centers* lie inside the disk.

    Args:
        grid          : number of cells along each axis (grid x grid total).
        warn_threshold: fraction of π below which we log a warning (default 99.9%).

    Returns:
        accuracy (float): (estimated area) / π
    """
    # Each axis spans [-1, 1] → total length = 2. Each cell has width = step.
    step = 2.0 / grid

    # Build 1D coordinates of **cell centers** along y and z:
    #   centers at: -1 + (i + 0.5) * step   for i=0..grid-1
    centers = -1.0 + (np.arange(grid, dtype=np.float32) + 0.5) * step

    # Make a 2D grid of centers. indexing="ij" → Y varies with the first index (rows),
    # Z with the second (columns). Shapes: (grid, grid) each.
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
    y = np.asarray(x, dtype=np.float64)
    n = y.size
    if n < 3:
        return np.empty(0, dtype=int)

    if valley:
        y = -y

    # Finite mask & convenient view that ignores NaNs
    finite = np.isfinite(y)
    if not finite.all():
        # Replace NaNs with -inf so they cannot be peaks,
        # and we can compute differences without special-casing
        y = y.copy()
        y[~finite] = -np.inf

    # First/last cannot be peaks
    dy = y[1:] - y[:-1]

    # Candidate edges
    # Rising edge: dy switches + or 0 right before, and - after
    # Falling edge: dy switches - before, and + or 0 after
    # (We emulate your <= / >= conventions)
    left = np.hstack((dy, 0.0))
    right = np.hstack((0.0, dy))

    cand = np.empty(0, dtype=int)
    if edge is None:
        # strictly interior local maxima
        cand = np.flatnonzero((left < 0.0) & (right > 0.0))
    else:
        e = edge.lower()
        idx = []
        if e in ('rising', 'both'):
            idx.append(np.flatnonzero((left <= 0.0) & (right > 0.0)))
        if e in ('falling', 'both'):
            idx.append(np.flatnonzero((left < 0.0) & (right >= 0.0)))
        if idx:
            cand = np.unique(np.concatenate(idx))  # sorted union

    if cand.size == 0:
        return cand

    # Remove boundary points (already implied by construction, but keep explicit)
    cand = cand[(cand > 0) & (cand < n - 1)]

    # Drop anything touching NaN neighbors
    if not finite.all():
        bad = ~(finite[cand] & finite[cand-1] & finite[cand+1])
        if bad.any():
            cand = cand[~bad]
            if cand.size == 0:
                return cand

    # Minimum peak height
    if mph is not None:
        cand = cand[y[cand] >= mph]
        if cand.size == 0:
            return cand

    # Threshold vs neighbors
    if threshold > 0.0:
        left_diff = y[cand] - y[cand - 1]
        right_diff = y[cand] - y[cand + 1]
        mindiff = np.minimum(left_diff, right_diff)
        cand = cand[mindiff >= threshold]
        if cand.size == 0:
            return cand

    # Minimum peak distance (greedy keep largest first)
    if mpd > 1:
        order = np.argsort(y[cand])[::-1]  # tallest → shortest
        cand_sorted = cand[order]
        keep = np.ones(cand_sorted.size, dtype=bool)

        # O(k^2) worst-case, but k << n typically; simple & robust
        for i in range(cand_sorted.size):
            if not keep[i]:
                continue
            lo = cand_sorted[i] - mpd
            hi = cand_sorted[i] + mpd
            # If kpsh=False, remove all within window.
            # If kpsh=True, keep equal-height ties.
            too_close = (cand_sorted >= lo) & (cand_sorted <= hi)
            if kpsh:
                # do not delete those with the exact same height
                same_height = (y[cand_sorted] == y[cand_sorted[i]])
                too_close &= ~same_height
            keep |= False  # no-op to keep type checker calm
            keep[too_close] = False
            keep[i] = True  # keep the current best

        cand = np.sort(cand_sorted[keep])

    # Optional plotting hook preserved (no-op here)
    if show and ax is not None:
        pass
    return cand


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
    x = np.arange(a0vcut.size)
    # initial peak pick with min peak distance
    peaks = detect_peaks(a0vcut, mpd=mpd)

    # robust prune peaks against poly fit
    if peaks.size >= (poly_order + 1):
        f = np.polyfit(x[peaks], a0vcut[peaks], poly_order)
        q = np.poly1d(f)
        resid = a0vcut[peaks] - q(x[peaks])
        MAD = np.median(np.abs(resid - np.median(resid)))

        mask = np.ones(peaks.size, dtype=bool)
        # low S/N or deep lines
        mask &= (a0vcut[peaks] / np.nanmedian(a0vcut) >= 0.1)
        # reject strong devs from the poly trend
        mask &= (a0vcut[peaks] >= q(x[peaks]) - 2 * MAD)
        mask &= (a0vcut[peaks] <= q(x[peaks]) + 6 * MAD)
        peaks = peaks[mask]

    # Fallback: if we lost too many peaks, repopulate with coarse Mpd
    if peaks.size < max(3, spline_k + 1):
        peaks = detect_peaks(a0vcut, mpd=max(1, mpd // 2))
        if peaks.size < max(3, spline_k + 1):
            # last resort: take a coarse uniform subset to avoid spline crash
            idx = np.linspace(0, a0vcut.size - 1, num=max(3, spline_k + 1)).astype(int)
            peaks = idx

    # Grow smoothing geometrically; stop once the spline envelope yields ~one broad bump twice
    s = 1e6
    hits = 0
    continuum = None
    for _ in range(10):  # at most 10 spline fits per tier
        f = interpolate.UnivariateSpline(x[peaks], a0vcut[peaks], k=spline_k, s=s)
        continuum = f(x)
        # count envelopes' peaks
        pk2 = detect_peaks(continuum)
        if pk2.size <= 1:
            hits += 1
            if hits >= 2:
                break
        else:
            hits = 0
        s *= 10.0  # geometric growth

    # If somehow didn't converge, ensure continuum is at least smooth-ish
    if continuum is None:
        f = interpolate.UnivariateSpline(x[peaks], a0vcut[peaks], k=spline_k, s=1e8)
        continuum = f(x)

    return continuum, peaks




def save_phase_pngs(
    cube, y, z, outdir="phase_maps",
    sentinel_offdisk=0, cmap="inferno",
    add_colorbar=True, phase_values=None,
):
    import os, numpy as np, matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    Y, Z = np.meshgrid(y, z, indexing="ij")           # Y,Z shaped (G,G): axis0=y, axis1=z
    disk_mask = (Y*Y + Z*Z) <= 1.0

    vals = cube[:, disk_mask].astype(np.float64)
    vals = vals[vals != sentinel_offdisk]
    vmin = np.percentile(vals, 1); vmax = np.percentile(vals, 99)
    if vmin == vmax: vmax = vmin + 1

    Npsi, G, _ = cube.shape
    extent = (-1, 1, -1, 1)                           # ★ x=y in [-1,1], y=z in [-1,1]

    for i in range(Npsi):
        frame = cube[i].astype(np.float64)
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


def ld_law_quadratic(mu, u1, u2):
    """
    Quadratic limb darkening shape L(mu) = 1 - u1*(1-mu) - u2*(1-mu)^2.
    Returns non-negative values (clipped at 0).
    """
    L = 1.0 - u1*(1.0 - mu) - u2*(1.0 - mu)**2
    return np.clip(L, 0.0, None)

# def _get_u1u2_for_T(ld_coeff_dict, band, T):
#     """
#     Fetch (u1,u2) for a given band and temperature T.
#     Falls back to nearest available integer temperature if exact not found.
#     """
#     key = f"{band}_{np.round(T, 4)}"
#     if key in ld_coeff_dict:
#         return ld_coeff_dict[key]
#     else:
#         print(ld_coeff_dict)
#         raise KeyError(f"No LD coeffs found for band '{band}' at T={np.round(T, 4)}K.")
    

#     # # Nearest-neighbor fallback across available temps for this band
#     # # Expect keys like "<band>_<Tint>"
#     # temps = []
#     # coeffs = []
#     # prefix = f"{band}_"
#     # for k, v in ld_coeff_dict.items():
#     #     if k.startswith(prefix):
#     #         try:
#     #             temps.append(int(k.split("_")[-1]))
#     #             coeffs.append(v)
#     #         except Exception:
#     #             pass
#     # if not temps:
#     #     raise KeyError(f"No LD coeffs found for band '{band}'.")
#     # temps = np.asarray(temps)
#     # idx = np.argmin(np.abs(temps - float(T)))
#     # return coeffs[idx]

def _get_u1u2_for_T(ld_coeff_dict, band, T):
    """
    Fetch (u1,u2) for a given band and temperature T.
    Falls back to nearest available temperature if exact not found
    due to float32 vs float64 precision mismatches.
    """
    # 1. Try exact 4-decimal match first
    key = f"{band}_{np.round(T, 4)}"
    if key in ld_coeff_dict:
        return ld_coeff_dict[key]
    
    # 2. Fallback: Nearest-neighbor search for float32 precision drift
    temps = []
    coeffs = []
    prefix = f"{band}_"
    for k, v in ld_coeff_dict.items():
        if k.startswith(prefix):
            try:
                # Extract the float temperature from the key
                temps.append(float(k.split("_")[-1]))
                coeffs.append(v)
            except Exception:
                pass
                
    if not temps:
        raise KeyError(f"No LD coeffs found for band '{band}'.")
        
    temps = np.asarray(temps)
    idx = np.argmin(np.abs(temps - float(T)))
    
    # Safety check: ensure it is just a tiny float rounding error (< 0.1 K), 
    # not a completely missing temperature grid
    if np.abs(temps[idx] - float(T)) > 0.1:
        raise KeyError(f"No LD coeffs found for band '{band}' at T={T}K. Closest is {temps[idx]}K.")
        
    return coeffs[idx]


def coverage_and_teff(
    cube_geometry,                   # (Npsi, G, G), int or float; may be scaled
    y, z,                   # grid coords in [-1, 1], len = G
    T_phot,                 # photosphere temperature (same units as 'cube' after scale)
    ld_coeff_dict,          # dict: f"{band}_{int(T)}" -> (u1, u2)
    band,                   
    sentinel_offdisk=0,
    scale=1.0,              # if stored temps scaled (e.g., ×10), give that here
    ld_by_local_temp=True,  # if True, LD varies with local T in 'cube'
    T_ref_for_ld=None,      # if ld_by_local_temp=False, use this T for LD (default: T_phot)
    black_spot_sentinel=None,
):
    """
    Returns:
      f_spot:   geometric disk fraction with T != T_phot per phase (Npsi,)
      f_cool:   geometric disk fraction with T <  T_phot per phase (Npsi,)
      f_hot:    geometric disk fraction with T >  T_phot per phase (Npsi,)
      Teff:     flux-equivalent Teff per phase from <T^4>_W^(1/4), W = mu * L(mu) (Npsi,)
    Notes:
      - Coverage fractions are **pure projected-area** fractions over the visible disk.
      - Teff uses **physical weights** W = μ × L(μ) with quadratic limb darkening.
      - Sentinel/off-disk handled per phase.
    """
    # Temperatures in physical units
    cube_geometry = cube_geometry.astype(np.float64) / scale              # (Npsi, G, G)
    cube_flux = cube_geometry.copy()
    cube_ld = cube_geometry.copy()
    Npsi, G, G2 = cube_geometry.shape
    assert G == G2, "grid must be square"

    # Separate cube used only for the bolometric-flux experiment.
    # cube_flux = cube_geometry.copy()       # <<<<< for DARK spot test
    # black_temperature_threshold = 2600.0   # <<<<< for DARK spot test

    # Geometry
    Y, Z = np.meshgrid(y, z, indexing="ij")
    r2   = Y*Y + Z*Z
    disk = (r2 <= 1.0)
    mu2d = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))         # cos(theta) on image plane

    # Per-phase validity mask (visible & not sentinel)
    on3 = disk[None, ...] & (cube_geometry != sentinel_offdisk)  # (Npsi, G, G)

    if black_spot_sentinel is None:
        black_mask = np.zeros_like(on3, dtype=bool)
    else:
        black_mask = (
            on3
            & np.isclose(
                cube_geometry,
                float(black_spot_sentinel),
                atol=0.02,
                rtol=0.0,
            )
        )

    # Black pixels remain geometrically present but emit zero flux.
    cube_flux[black_mask] = 0.0

    # Use photospheric LD only as a geometric weighting proxy.
    # This prevents a lookup for nonexistent 9999.99 K coefficients.
    cube_ld[black_mask] = float(T_phot)


    # black_mask = (on3 & (cube_geometry < black_temperature_threshold))  # <<<<< for DARK spot test
    # cube_flux[black_mask] = 0.0 # <<<<< for DARK spot test

    # ---------- Geometric (projected-disk) coverage weights ----------
    W2_area = np.ones_like(Y, dtype=np.float64)         # (G, G)
    W3_area = np.where(on3, W2_area[None, ...], 0.0)    # (Npsi, G, G)
    Wsum_phase_area = W3_area.sum(axis=(1, 2))          # (Npsi,)
    if np.any(Wsum_phase_area == 0):
        bad = np.where(Wsum_phase_area == 0)[0]
        raise ValueError(f"No valid on-disk pixels for phases: {bad}")

    # Classify temperatures
    # atol = 0.0 if (cube_geometry.dtype.kind in "iu") else 1e-6
    # The surface cube originates as float32, so allow its expected
    # sub-millikelvin representation difference.
    temperature_atol = 0.1
    # is_phot = np.isclose(cube_geometry, T_phot, atol=atol)
    # is_cool = (cube_geometry <  T_phot) & (~is_phot)
    # is_hot  = (cube_geometry >  T_phot) & (~is_phot)

    is_phot = (
        np.isclose(cube_geometry, T_phot, atol=temperature_atol, rtol=0.0,)
        & (~black_mask)
        & on3
        )
    
    is_cool = (on3 & (~is_phot) & ((cube_geometry < T_phot) | black_mask)
               )

    is_hot = (on3 & (~is_phot) & (cube_geometry > T_phot) & (~black_mask)
              )

    is_spot = on3 & (~is_phot)

    # Geometric fractions (projected disk)
    # f_spot = (W3_area * (~is_phot)).sum(axis=(1, 2)) / Wsum_phase_area
    f_spot = (W3_area * ( is_spot)).sum(axis=(1, 2)) / Wsum_phase_area
    f_cool = (W3_area * ( is_cool)).sum(axis=(1, 2)) / Wsum_phase_area
    f_hot  = (W3_area * (  is_hot)).sum(axis=(1, 2)) / Wsum_phase_area

    # ---------- weights for Teff: W = LD(μ) ----------
    if ld_by_local_temp:
        mu3 = mu2d[None, ...]                           # (1, G, G), broadcasts
        # Map unique 1.4f temps to LD coeffs (only where valid)
        # Ts = np.unique(cube_geometry[on3].astype(float))
        Ts = np.unique(cube_ld[on3].astype(float))
        u1_map, u2_map = {}, {}
        for TT in Ts:
            u1, u2 = _get_u1u2_for_T(ld_coeff_dict, band, TT)
            u1_map[TT] = float(u1)
            u2_map[TT] = float(u2)
        # Build per-pixel u1,u2
        U1 = np.zeros_like(cube_geometry, dtype=np.float64)
        U2 = np.zeros_like(cube_geometry, dtype=np.float64)
        # T_all = np.round(cube_geometry, 4)
        T_all = np.round(cube_ld, 4)
        for TT in Ts:
            sel = (T_all == TT) & on3
            if np.any(sel):
                U1[sel] = u1_map[TT]
                U2[sel] = u2_map[TT]
        L3 = ld_law_quadratic(mu3, U1, U2)              # (Npsi, G, G)
        W3_phy = np.where(on3, L3, 0.0)

    else:
        T_ref = T_phot if (T_ref_for_ld is None) else T_ref_for_ld
        u1, u2 = _get_u1u2_for_T(ld_coeff_dict, band, T_ref)
        L2 = ld_law_quadratic(mu2d, u1, u2)             # (G, G)
        W3_phy = np.where(on3, L2[None, ...], 0.0)

    Wsum_phase_phy = W3_phy.sum(axis=(1, 2))
    if np.any(Wsum_phase_phy <= 0):
        bad = np.where(Wsum_phase_phy <= 0)[0]
        raise ValueError(f"No on-disk pixels (or zero physical weight) for phases: {bad}")

    # ---------- Teff per phase from physically-weighted <T^4> ----------
    # T4_mean_phase = (W3_phy * (cube_geometry**4)).sum(axis=(1, 2)) / Wsum_phase_phy
    T4_mean_phase = (W3_phy * cube_flux**4).sum(axis=(1, 2)) / Wsum_phase_phy
    # T4_mean_phase = (W3_phy * (cube_flux**4)).sum(axis=(1, 2)) / Wsum_phase_phy     # <<<<< for DARK spot test
    Teff_rotavg = (T4_mean_phase.mean())**0.25

    # ---------- Effective Spot Temperature per phase ----------
    # Isolate the physical weights for only the spotted regions
    # W3_spot_phy = W3_phy * (~is_phot)
    W3_spot_phy = W3_phy * is_spot
    Wsum_spot_phy = W3_spot_phy.sum(axis=(1, 2))

    # Initialize array with NaNs (so phases with 0 spots don't plot as 0 Kelvin)
    T_spot_eff_phase = np.full(Npsi, np.nan, dtype=np.float64)
    
    # Only calculate where spots are actually visible on the disk
    valid_spots = (Wsum_spot_phy > 0)
    
    if np.any(valid_spots):
        # T4_spot_mean = (W3_spot_phy[valid_spots] * (cube_geometry[valid_spots]**4)).sum(axis=(1, 2)) / Wsum_spot_phy[valid_spots]
        T4_spot_mean = (W3_spot_phy[valid_spots] * (cube_flux[valid_spots]**4)).sum(axis=(1, 2)) / Wsum_spot_phy[valid_spots]
        T_spot_eff_phase[valid_spots] = T4_spot_mean ** 0.25

    return f_spot, f_cool, f_hot, T4_mean_phase, Teff_rotavg, T_spot_eff_phase

    # return f_spot, f_cool, f_hot, T4_mean_phase, Teff_rotavg


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
        tstr = k.split("_")
        temps.append(np.round(float(tstr[-1]), 4))
        coeffs.append(v)  # e.g., (u1,u2) or (u1,u2,u3)

    temps = np.asarray(temps, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)  # (N,K)

    order = np.argsort(temps)
    temps_tab  = temps[order]
    coeffs_tab = coeffs[order]

    # compact LUT (no giant dense array)
    lut_keys = np.round(temps_tab, 4)                      # [T1, T2, ...]
    lut_vals = np.arange(temps_tab.size, dtype=np.int32)        # [0, 1, 2, ...]

    return temps_tab, coeffs_tab, lut_keys, lut_vals

@njit
def lut_lookup_eq(temp, lut_keys, lut_vals):
    """
    Nearest-neighbor lookup to bridge float32 (cube) and float64 (keys).
    """
    best = 0
    bestd = abs(lut_keys[0] - temp)
    for i in range(1, lut_keys.size):
        d = abs(lut_keys[i] - temp)
        if d < bestd:
            best = i
            bestd = d
    return lut_vals[best]



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
        temps.append(np.round(T, 4))
        spect_list.append(np.asarray(obj.intensity, dtype=np.float64))
        if want_mu and mu_grid is None:
            # as miu are the same for all T, thus we only need one array, unlike spectrum
            mu_grid = np.asarray(obj.mu, dtype=np.float32)


    temps = np.asarray(temps, dtype=np.float64)
    spectrum_tab = np.stack(spect_list, axis=0)  # (N, ...), float32

    order = np.argsort(temps)
    temps_tab    = temps[order]
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


def compute_template_bolometric_intensity(wave_vac_A, intensity_erg_mu, temp):
    """
    Compute the total bolometric specific intensity for a given stellar template at a specific mu.
    
    Integrates the empirical synthetic intensity spectrum via the trapezoidal rule 
    and appends a numerical Planck function tail for wavelengths beyond 
    the template limits (e.g., lambda > 5.5 um).
    
    Parameters
    ----------
    wave_vac_A : numpy.ndarray
        Wavelength array in vacuum Angstroms.
    intensity_erg_mu : numpy.ndarray
        Specific intensity array at a specific mu angle in erg s^-1 cm^-2 sr^-1 A^-1.
    temp : float
        Effective temperature of the specific model region [K].
        
    Returns
    -------
    I_bol : float
        Total integrated specific intensity in erg s^-1 cm^-2 sr^-1.
    """
    # -------------------------------------------------------------------------
    # PHOENIX native units are erg / s / cm^2 / cm / sr. 
    # We must multiply by 1e-8 to convert to erg / s / cm^2 / Angstrom / sr 
    # -------------------------------------------------------------------------
    intensity_erg_mu_A = intensity_erg_mu * 1e-8

    # 1. Integrate the available empirical template range
    I_model = np.trapezoid(intensity_erg_mu_A, wave_vac_A)
    
    # 2. Calculate the Blackbody tail
    max_wave_m = np.max(wave_vac_A) * 1e-10
    wave_tail_m = np.logspace(np.log10(max_wave_m), np.log10(1e-2), 2000)
    
    h = const.h.cgs.value
    c = const.c.cgs.value
    k = const.k_B.cgs.value
    
    wave_tail_cm = wave_tail_m * 100.0
    
    # Planck function B_lambda gives Specific Intensity (erg s^-1 cm^-2 cm^-1 sr^-1)
    bb_intensity_cm = (2.0 * h * c**2 / (wave_tail_cm**5)) / (np.exp((h * c) / (wave_tail_cm * k * temp)) - 1.0)
    
    # Convert to Angstrom grid for consistent integration
    wave_tail_A = wave_tail_cm * 1e8
    bb_intensity_A = bb_intensity_cm * 1e-8  # erg s^-1 cm^-2 sr^-1 A^-1
    
    # Integrate the specific intensity tail
    I_tail = np.trapezoid(bb_intensity_A, wave_tail_A)
    
    return I_model + I_tail

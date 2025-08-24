import sys, os, logging
logger = logging.getLogger()

import numpy as np
from numba import njit
from scipy import signal,interpolate
from specutils import Spectrum1D
from specutils.analysis import equivalent_width
import astropy.units as u
import matplotlib.pyplot as plt


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
    key_v = f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v'
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
#         vel = Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']
#     elif intORflux == 'intensity':
#         vel = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.0f}_v']

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
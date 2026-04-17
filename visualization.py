
import subprocess, os, shutil
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.path as path
import matplotlib.patches as patches

import numpy as np
from numba import njit
from tqdm import tqdm
from pqdm.processes import pqdm

mpl.rcParams['figure.dpi'] = 250
mpl.rcParams['figure.facecolor'] = 'white'

mpl.rcParams['font.family'] = 'sans-serif' # ['sans-serif', 'monospace', 'serif', 'cursive']
mpl.rcParams['font.style'] = 'normal'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.direction'] = 'in'

def RAxisAngle(axis=[0, 1, 0], theta=0):
    """
    Rotate an arbitrary point by an axis and an angle.
    from Starry: https://github.com/rodluger/paparazzi/blob/119743ec26bbca8b3240c38be36b0c47cb2c9880/src/figures/linalg.py
    """
    cost = np.cos(theta)
    sint = np.sin(theta)

    return np.reshape(
        [
            cost + axis[0] * axis[0] * (1 - cost),
            axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
            axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
            axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
            cost + axis[1] * axis[1] * (1 - cost),
            axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
            axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
            axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
            cost + axis[2] * axis[2] * (1 - cost),
        ],
        [3, 3],
    )

def get_ortho_latitude_lines(
    radius, inc=np.pi / 2, obl=0, dlat=np.pi / 6, npts=1000):
    """
    Return the lines of constant latitude on an orthographic projection.
    from Starry: https://github.com/rodluger/paparazzi/blob/119743ec26bbca8b3240c38be36b0c47cb2c9880/src/figures/linalg.py
    """
    # Angular quantities
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(obl)
    so = np.sin(obl)

    # Latitude lines
    res = []
    latlines = np.arange(-np.pi / 2, np.pi / 2, dlat)[1:]
    for lat in latlines:

        # Figure out the equation of the ellipse
        y0 = np.sin(lat) * si
        a = np.cos(lat)
        b = a * ci
        x = np.linspace(-a, a, npts)
        y1 = y0 - b * np.sqrt(1 - (x / a) ** 2)
        y2 = y0 + b * np.sqrt(1 - (x / a) ** 2)

        # Mask lines on the backside
        if si != 0:
            if inc > np.pi / 2:
                ymax = y1[np.argmax(x ** 2 + y1 ** 2)]
                y1[y1 < ymax] = np.nan
                ymax = y2[np.argmax(x ** 2 + y2 ** 2)]
                y2[y2 < ymax] = np.nan
            else:
                ymax = y1[np.argmax(x ** 2 + y1 ** 2)]
                y1[y1 > ymax] = np.nan
                ymax = y2[np.argmax(x ** 2 + y2 ** 2)]
                y2[y2 > ymax] = np.nan

        # Rotate them
        for y in (y1, y2):
            xr = -x * co + y * so
            yr = x * so + y * co
            res.append((xr*radius, y*radius))

    return res


def get_ortho_longitude_lines(
    radius, inc=np.pi / 2, obl=0, theta=0, dlon=np.pi / 6, npts=1000
):
    """
    Return the lines of constant longitude on an orthographic projection.
    from Starry: https://github.com/rodluger/paparazzi/blob/119743ec26bbca8b3240c38be36b0c47cb2c9880/src/figures/linalg.py
    """

    # Angular quantities
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(obl)
    so = np.sin(obl)

    # Are we (essentially) equator-on?
    equator_on = (inc > 88 * np.pi / 180) and (inc < 92 * np.pi / 180)

    # Longitude grid lines
    res = []
    if equator_on:
        offsets = np.arange(-np.pi / 2, np.pi / 2, dlon)
    else:
        offsets = np.arange(0, 2 * np.pi, dlon)

    for offset in offsets:

        # Super hacky, sorry. This can probably
        # be coded up more intelligently.
        if equator_on:
            sgns = [1]
            if np.cos(theta + offset) >= 0:
                bsgn = 1
            else:
                bsgn = -1
        else:
            bsgn = 1
            if np.cos(theta + offset) >= 0:
                sgns = np.array([1, -1])
            else:
                sgns = np.array([-1, 1])

        for lon, sgn in zip([0, np.pi], sgns):

            # Viewed at i = 90
            y = np.linspace(-1, 1, npts)
            b = bsgn * np.sin(lon - theta - offset)
            x = b * np.sqrt(1 - y ** 2)
            z = sgn * np.sqrt(np.abs(1 - x ** 2 - y ** 2))

            if equator_on:

                pass

            else:

                # Rotate by the inclination
                R = RAxisAngle([1, 0, 0], np.pi / 2 - inc)
                v = np.vstack(
                    (x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1))
                )
                x, y, _ = np.dot(R, v)

                # Mask lines on the backside
                if si != 0:
                    if inc < np.pi / 2:
                        imax = np.argmax(x ** 2 + y ** 2)
                        y[: imax + 1] = np.nan
                    else:
                        imax = np.argmax(x ** 2 + y ** 2)
                        y[imax:] = np.nan

            # Rotate by the obliquity
            xr = -x * co + y * so 
            yr = x * so + y * co
            res.append((xr*radius, yr*radius))

    return res

def map_add_grid(axins, incl, radius, phase):
    theta = phase*(2*np.pi) 
    # theta = 0
    # x = np.linspace(-radius, radius, 10000)
    # y = np.sqrt(radius**2 - x**2)
    # axins.plot(x, y, "k-", lw=1, zorder=10)
    # axins.plot(x, -y, "k-", lw=1, zorder=10)

    lat_lines = get_ortho_latitude_lines(radius, inc=incl * np.pi / 180)
    for x, y in lat_lines:
        axins.plot(x, y, "#aaaaaa", lw=0.3, alpha=0.3, zorder=11)
    
    lon_lines = get_ortho_longitude_lines(
        radius, inc=incl * np.pi / 180, 
        theta=np.pi + theta # * np.pi / 180
    )
    for n, l in enumerate(lon_lines):
        axins.plot(l[0], l[1], "#aaaaaa", lw=0.3, alpha=0.3, zorder=11)
    return axins

def spot_view_pel(
    Star, grid, visible_phi, SPECstar_tot, ccf_rv_shift, quite_surface_2d, 
    outdir, ohfe_ratio, text1, text2, band, intORflux, model_type, j):
    
    _teff = Star._phase_teff[j]
    _fspot = Star._phase_fspot[j] * 100
    _fcool = Star._phase_fcool[j] * 100
    _fhot = Star._phase_fhot[j] * 100

    # -------------------------------------------------
    fig = plt.figure(figsize = (7, 3.5), dpi=300 )

    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.0, right=0.35, bottom=0.3, top=1.)#, wspace=0.0, hspace=0.16)
    ax1 = plt.subplot(gs1[0])

    gs2 = gridspec.GridSpec(2, 1)
    gs2.update(left=0.46, right=1., bottom=0. , top=1., wspace=0.0, hspace=0.25)
    ax2 = plt.subplot(gs2[0])
    ax3 = plt.subplot(gs2[1])

    radius = grid//2

    st_spot_surface = quite_surface_2d[j]/np.max(quite_surface_2d)
    st_spot_surface[st_spot_surface == 0.0] = np.nan
    ax1.imshow(
        st_spot_surface, extent=[-radius, radius, -radius, radius], 
        cmap='Oranges_r', vmin=0, vmax=1, origin='lower')
    ax1 = map_add_grid(ax1, Star.incl, radius, visible_phi[j])
    
    ax1.text(
        0.5, -0.09, fr'Phase ${visible_phi[j]:.3f}$', va='center', ha='center', size='small',
        transform=ax1.transAxes)
    ax1.text(
        0.5, -0.15,
        fr'$T_{{\rm eff}}={_teff:4.0f}$ K, '
        fr'$f_{{spot}}={_fspot:5.2f}$%'  # <-- percent outside $
        fr' ($f_{{cool}}={_fcool:5.2f}$%'
        fr', $f_{{hot}}={_fhot:5.2f}$%)',
        va='center', ha='center', size='xx-small', transform=ax1.transAxes
    )
    
    c_kms = 299792.458
    w_center = Star.bands2use[band][0]  # [15631.35, 30, ['rv', 'photo']]
    if intORflux == 'intensity':
        _v = Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v']
        _wave =( _v * w_center / c_kms) + w_center
        ax2.plot(_wave, SPECstar_tot[j], '-k', lw=1)
    elif intORflux == 'flux':
        _v = Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v']
        _wave =( _v * w_center / c_kms) + w_center
        ax2.plot(_wave, SPECstar_tot[j], '-k', lw=1)

    ax2.set_ylim(np.min(SPECstar_tot)-0.05, 1.05)
    ax2.set_xlim(_wave[0], _wave[-1])
    # if intORflux == 'intensity':
    #     ax2.set_xlim(
    #         Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v'][0], 
    #         Star.int_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v'][-1]
    #         )
    # elif intORflux == 'flux':
    #     ax2.set_xlim(
    #         Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v'][0], 
    #         Star.flux_spec[f'{band}_{model_type}_{intORflux}_{Star.Temp:1.4f}_v'][-1]
    #         )
    # ax2.set_xlabel(r'Velocity [km s$^{-1}$]',  size='small')
    ax2.set_xlabel(r'Wavelength [$\AA$]',  size='small')
    ax2.set_ylabel(f'Normalized Flux',  size='small')
    
    if 'ewr' in band:

        ax3.plot(visible_phi[:j], ohfe_ratio[:j], '-o', c='tab:grey', ms=2, mew=2)
        ax3.plot(visible_phi[j], ohfe_ratio[j], 'o', c='k', ms=2, mew=2)

        ax3.set_ylim(np.min(ohfe_ratio)-0.1, np.max(ohfe_ratio)+0.1)
    # ax3.set_xlim(-0.5, 1.5)
    ax3.set_xlim(-1, 1.0)
    ax3.set_xlabel(r'Phase',  size='small')
    ax3.set_ylabel('Equivalent Width Ratio  (OH/Fe)',  size='small')

    # --- ccf rv from normal soap2 procedure ---
    ax3right = ax3.twinx()
    ax3right.plot(visible_phi[:j], ccf_rv_shift[:j], '-o', c='skyblue', ms=2, mew=2)
    ax3right.plot(visible_phi[j], ccf_rv_shift[j], 'o', c='tab:blue', ms=2, mew=2)
    
    rv_std = np.nanstd(ccf_rv_shift)
    ax3right.set_ylim(
        np.nanmin(ccf_rv_shift)-3*rv_std, np.nanmax(ccf_rv_shift)+3*rv_std)
    ax3right.set_ylabel(
        r'Spot induced RV [km s$^{-1}$]', size='small', color='tab:blue', 
        rotation=270, labelpad=10
        )

    ax3.tick_params(axis='x', which='both', labelsize='small', 
                    top=True, direction='in', width=.7)
    ax3.tick_params(axis='y', which='both', labelsize='small', 
                    direction='in', width=.7)
    ax3right.tick_params(axis='y', which='both', labelsize='small', 
                    direction='in', width=.7, colors='tab:blue')

    ax2.tick_params(axis='both', which='both', labelsize='small', right=True, 
            top=True, direction='in', width=.7)
            
    for axes in [ax2, ax3, ax3right]:
        axes.xaxis.set_minor_locator(AutoMinorLocator(5))
        axes.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.axis('off')

    ax1.text(
        0.05, -0.20, text1, family='monospace', size=5, 
        transform=ax1.transAxes, ha='left', va='top')
    ax1.text(
        0.01, -0.40, text2, family='monospace', size=4, 
        transform=ax1.transAxes, ha='left', va='top')

    out_dir_pre = f'./outputs/{outdir.split("/")[2]}'
    fig.savefig(
        f'{out_dir_pre}/gif_tmp/phase{visible_phi[j]:1.3f}_{band}.png', 
        format='png', bbox_inches='tight')
    plt.close(fig)

def spot_view(
    Star, grid_size, visible_phi, integrated_stellar_spectrum, ccf_rv_shift, 
    final_2d_surface_flux_sum, outdir, cpu_use, ohfe_ratio, text1, text2, 
    band, intORflux, model_type
    ):

    # ------------------------------------------
    print('\nPlotting...\n')

    out_dir_pre = f'./outputs/{outdir.split("/")[2]}'
    out_dir_pre_tmp = f"{out_dir_pre}/gif_tmp"
    os.makedirs(out_dir_pre_tmp, exist_ok=True)
    func = partial(
        spot_view_pel, Star, grid_size, visible_phi, 
        integrated_stellar_spectrum, ccf_rv_shift, final_2d_surface_flux_sum, outdir, ohfe_ratio, 
        text1, text2, band, intORflux, model_type
        )
    
    # for ii in np.arange(len(visible_phi)):
    #     spot_view_pel( Star, grid_size, visible_phi, 
    #     integrated_stellar_spectrum, ccf_rv_shift, final_2d_surface_flux_sum, outdir, ohfe_ratio, 
    #     text1, text2, band, intORflux, model_type, ii)
    
    pqdm(np.arange(len(visible_phi)), func, n_jobs=cpu_use)
    
    status, _ = subprocess.getstatusoutput('convert') #magick
    if status <= 1:
        print('\nNow making .gif\n')
        working_dir = os.getcwd()
        os.chdir(out_dir_pre_tmp)
        os.system('convert -delay 20 +dither $(ls *.png | sort -k 1.6g) temp.gif')
        os.system('convert temp.gif \( +clone -set delay 300 \) +swap +delete temp.gif')
        os.system(f'mv temp.gif "../{outdir.split("/")[-1]}_{band}.gif"')
        os.chdir('../')
        os.system(f'rm -r gif_tmp')
        os.chdir(working_dir)
    else:
        print('\n"magick or convert" function not found, skip making .gif\n')

def plot_cc(drvs, cc, new_rv, phase, band, properties, outdir):
    # drvs_temp = drvs[cc == np.max(cc)]

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    ax.plot(drvs, cc, 'k-', lw=.7)

    # ax.axvline(
        # np.mean(drvs_temp), ls='-', lw=1, alpha=0.5, c='tab:blue'
        # )
    if new_rv is not None:
        ax.axvline(
            new_rv, ls='--', lw=1, alpha=0.5, c='tab:red'
        )

        ax.text(0.05, 0.95, f'RV: {new_rv:1.4f} km/s', transform=ax.transAxes, size='x-small')
    ax.text(0.05, 0.85, f'Prominences: {properties["prominences"]}', transform=ax.transAxes, size='x-small')
    ax.text(0.05, 0.75, f'left_bases: {properties["left_bases"]}', transform=ax.transAxes, size='x-small')
    ax.text(0.05, 0.65, f'right_bases: {properties["right_bases"]}', transform=ax.transAxes, size='x-small')

    ax.set_xlabel(r'Velocity [km/s]',  size='small')
    ax.set_ylabel('CCF',  size='small')
    ax.tick_params(
        axis='both', which='both', labelsize='small', right=True, 
        top=True, direction='in', width=.7)

    out_dir_pre = f'./outputs/{outdir.split("/")[2]}'
    os.makedirs(f'{out_dir_pre}/CCF', exist_ok=True)
    fig.savefig(
        f'{out_dir_pre}/CCF/{band}_phase_{phase:1.3f}.png', format='png', 
        bbox_inches='tight'
        )
    plt.close(fig)


def plot_cc_spec(vel, f, tw, tf, phase, delrv):

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    ax.plot(vel, f, 'k-', lw=.7)

    tf = np.interp(vel, tw+delrv, tf)
    ax.plot(vel, tf, '-', c='tab:red', alpha=0.5, lw=1)
    
    ax.set_xlabel(r'Velocity [km/s]',  size='small')
    ax.set_ylabel('Flux',  size='small')
    ax.tick_params(
        axis='both', which='both', labelsize='small', right=True, 
        top=True, direction='in', width=.7)
    
    fig.savefig(
        f'./outputs/CCF/phase{phase:1.3f}_spec.png', format='png', 
        bbox_inches='tight'
        )
    plt.close(fig)

    

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


def spot_view_pel(
    Star, grid, visible_phi, SPECstar_tot, ccf_rv_shift, quite_surface_2d, 
    outdir, depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, depth_l_fe2, 
    depth_l_oh0, text1, text2, j
    ):
    f = plt.figure(figsize=(7, 3.5))

    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.0, right=0.35, bottom=0.3, top=1.)#, wspace=0.0, hspace=0.16)
    ax1 = plt.subplot(gs1[0])

    gs2 = gridspec.GridSpec(2, 1)
    gs2.update(left=0.46, right=1., bottom=0. , top=1., wspace=0.0, hspace=0.25)
    ax2 = plt.subplot(gs2[0])
    ax3 = plt.subplot(gs2[1])

    radius = grid//2

    st_spot_surface = quite_surface_2d[j]/np.max(quite_surface_2d[j])
    st_spot_surface[st_spot_surface == 0.0] = np.nan
    ax1.imshow(
        st_spot_surface, extent=[-radius, radius, -radius, radius], 
        cmap='Oranges_r', vmin=0, vmax=1, origin='lower')
    ax1.text(
        0.5, -0.1, f'Phase {visible_phi[j]:.3f}', va='center', ha='center', size='small',
        transform=ax1.transAxes)
    
    ax2.plot(Star.star_spec['v'], SPECstar_tot[j], '-k', lw=1)
    ax2.plot(
        [depth_l_fe1[j], depth_l_fe2[j], depth_l_oh0[j]],
        (np.array([depth_d_fe1[j], depth_d_fe2[j], depth_d_oh0[j]])-1)/-1,
        's', ms=.6, mew=.6, color='tab:grey')

    ax2.set_ylim(np.min(SPECstar_tot)-0.05, 1.05)
    ax2.set_xlim(-300, 300)
    ax2.set_xlabel(fr'Velocity [km s$^{-1}$]',  size='small')
    ax2.set_ylabel(f'Relative Flux',  size='small')

    oh_line = np.array(depth_d_oh0)
    fe_line = (np.array(depth_d_fe1) + np.array(depth_d_fe2)) / 2
    line_ratio = oh_line/fe_line

    ax3.plot(visible_phi[:j], line_ratio[:j], 'o', c='tab:grey', ms=3, mew=3)
    ax3.plot(visible_phi[j], line_ratio[j], 'o', c='k', ms=3, mew=3)

    ax3.set_ylim(np.min(line_ratio)-0.1, np.max(line_ratio)+0.1)
    ax3.set_xlim(-0.3, 1.3)
    ax3.set_xlabel(r'Phase',  size='small')
    ax3.set_ylabel('Line depth ratio (OH/Fe)',  size='small')

    ax3right = ax3.twinx()
    ax3right.plot(visible_phi[:j], ccf_rv_shift[:j], 'o', c='skyblue', ms=3, mew=3)
    ax3right.plot(visible_phi[j], ccf_rv_shift[j], 'o', c='tab:blue', ms=3, mew=3)

    ax3right.set_ylim(np.min(ccf_rv_shift)-1, np.max(ccf_rv_shift)+1)
    ax3right.set_ylabel(
        'Spot induced RV [km/s]', size='small', color='tab:blue', rotation=270, labelpad=10)

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
        0.05, -0.18, text1, family='monospace', size=5, 
        transform=ax1.transAxes, ha='left', va='top')
    ax1.text(
        0.05, -0.45, text2, family='monospace', size=5, 
        transform=ax1.transAxes, ha='left', va='top')


    f.savefig(
        f'{outdir}/phase{visible_phi[j]:1.3f}.png', format='png', bbox_inches='tight')
    plt.close()

def spot_view(
    Star, grid, visible_phi, SPECstar_tot, ccf_rv_shift, quite_surface_2d, 
    outdir, cpu_use, depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, 
    depth_l_fe2, depth_l_oh0, text1, text2
    ):

    # ------------------------------------------
    print('\nPlotting...\n')

    func = partial(
        spot_view_pel, Star, grid, visible_phi, 
        SPECstar_tot, ccf_rv_shift, quite_surface_2d, outdir, depth_d_fe1, depth_d_fe2, 
        depth_d_oh0, depth_l_fe1, depth_l_fe2, depth_l_oh0, text1, text2
        )
    
    # for ii in np.arange(len(visible_phi)):
    #     spot_view_pel(Star, grid, visible_phi, 
    #     SPECstar_tot, ccf_rv_shift, quite_surface_2d, outdir, depth_d_fe1, depth_d_fe2, 
    #     depth_d_oh0, depth_l_fe1, depth_l_fe2, depth_l_oh0, text1, text2, ii)
    pqdm(np.arange(len(visible_phi)), func, n_jobs=cpu_use)
   
    status, _ = subprocess.getstatusoutput('convert')
    if status <= 1:
        print('\nNow making .gif\n')
        os.chdir(outdir)
        os.system('convert -delay 20 +dither $(ls *.png | sort -k 1.6g) temp.gif')
        os.system('convert temp.gif \( +clone -set delay 300 \) +swap +delete temp.gif')
        os.system(f'mv temp.gif "{outdir.split("/")[-1]}.gif"')
    else:
        print('\n"convert" function not found, skip making .gif\n')
    # return xyz2_box, spot_surface, st_surface


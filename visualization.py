
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


@njit
def limb_dark(grid, radius, limba1, limba2):

    st_surface_limb = np.ones((grid, grid), dtype=np.float64)

    for i in np.arange(-radius, radius):
        for j in np.arange(-radius, radius):
            i_nor, j_nor = i/radius, j/radius

            r_cos = np.sqrt( 1 - (i_nor**2+j_nor**2) )  

            # intensity due to limb-darkening (Mandel & Agol 2002)
            limb = 1 - limba1*(1-r_cos) - limba2*(1-r_cos)**2 
            st_surface_limb[i+radius,j+radius] = limb
    
    return st_surface_limb


@njit
def init_surface(grid, limba1, limba2):
    
    grid_use = grid+1 if grid%2==0 else grid
    st_surface = np.ones((grid_use, grid_use), dtype=np.float64)

    radius = grid_use//2

    # for i in np.arange(-radius, radius):
    #     for j in np.arange(-radius, radius):

    #         if np.sqrt(i**2+j**2)>=radius:
    #             st_surface[i+radius,j+radius] = 0

    st_surface_limb = limb_dark(grid_use, radius, limba1, limba2)

    return st_surface*st_surface_limb, radius



def spot_patch_selec(vis_yes, vis_no, xyz2_box, j, visible, radius, size, phi):    
    if (vis_yes==1) & (vis_no==0):
        y_vis_coord = xyz2_box[j][1,:][visible]*radius 
        z_vis_coord = xyz2_box[j][2,:][visible]*radius

        boundpoints = np.concatenate([ [y_vis_coord], [z_vis_coord] ]).T
        bound = path.Path(boundpoints)

        patch = patches.PathPatch(bound, facecolor='tab:red', lw=0.5, edgecolor='tab:red')
        # ax1.add_patch(patch)
    else:
        y_vis_coord = xyz2_box[j][1,:]*radius 
        z_vis_coord = xyz2_box[j][2,:]*radius

        x_coord = xyz2_box[j][0, :].copy()
        asign = np.sign(x_coord)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(bool)
        edge_loc = np.arange(len(signchange))[signchange]
        if np.sign(x_coord[0]) == 1:
            edge_loc[0] = edge_loc[0]-1
        else:
            edge_loc[1] = edge_loc[1]-1
        
        y_edge = y_vis_coord[edge_loc]
        z_edge = z_vis_coord[edge_loc]

        slope = np.diff(z_edge)/np.diff(y_edge)
        slope = -np.abs(slope) if phi<=0.5 else np.abs(slope)
        movey = radius/2 if phi<=0.5 else -radius/2
        movez = movey * -1/slope

        extended_y_edge = y_edge+movey
        extended_z_edge = z_edge+movez

        y_vis_coord = y_vis_coord[visible]
        z_vis_coord = z_vis_coord[visible]
        if np.sign(x_coord[0]) == 1:
            try:
                y_vis_coord = np.insert(y_vis_coord, edge_loc[0]+1, extended_y_edge)
                z_vis_coord = np.insert(z_vis_coord, edge_loc[0]+1, extended_z_edge)
            except:
                pass
        else:
            y_vis_coord = np.insert(extended_y_edge, 1, y_vis_coord)
            z_vis_coord = np.insert(extended_z_edge, 1, z_vis_coord)

        boundpoints = np.concatenate([ [y_vis_coord], [z_vis_coord] ]).T
        bound = path.Path(boundpoints)
        patch = patches.PathPatch(bound, facecolor='tab:red', lw=0.5, edgecolor='tab:red')
        # ax1.add_patch(patch)

        # ax1.plot(y_edge, z_edge, 'x', ms=5)
        # ax1.plot(extended_y_edge, extended_z_edge, 'x', ms=5)

    coord_xx = []; coord_yy = []
    for xx in np.arange(radius, -radius-1, -1):
        for yy in np.arange(-radius, radius+1):
            coord_xx.append(yy)
            coord_yy.append(xx)

    points1 = np.concatenate([[coord_xx], [coord_yy]]).T
    spot_mask = bound.contains_points(points1)
    return spot_mask, patch

def spot_view_pel(
    Star, grid, xyz, visible_phi, xyz2_box, CCFstar_tot, CCFstar_quiet, 
    outdir, depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, depth_l_fe2, 
    depth_l_oh0, text1, text2, j
    ):
    st_surface, radius = init_surface(grid, Star.limba1, Star.limba2)
    f = plt.figure(figsize=(7, 3.5))

    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.0, right=0.35, bottom=0.3, top=1.)#, wspace=0.0, hspace=0.16)
    ax1 = plt.subplot(gs1[0])

    gs2 = gridspec.GridSpec(2, 1)
    gs2.update(left=0.46, right=1., bottom=0. , top=1., wspace=0.0, hspace=0.25)
    ax2 = plt.subplot(gs2[0])
    ax3 = plt.subplot(gs2[1])


    vis_yes, vis_no = any(xyz2_box[j][0, :]>0), any(xyz2_box[j][0, :]<0)
    visible = xyz2_box[j][0,:]>0

    spot_mask, patch = spot_patch_selec(
        vis_yes, vis_no, xyz2_box, j, visible, radius, st_surface.shape[0], visible_phi[j]
        )
    spot_mask = spot_mask.reshape([int(st_surface.shape[0]), int(st_surface.shape[0])])
    spot_surface = ~spot_mask*np.ones([int(st_surface.shape[0]), int(st_surface.shape[0])])*0.15

    st_spot_surface = st_surface + spot_surface

    ax1.imshow(st_spot_surface, extent=[-radius, radius, -radius, radius], cmap='Oranges_r')
    ax1.text(
        0.5, -0.1, f'Phase {visible_phi[j]:.2f}', va='center', ha='center', size='small',
        transform=ax1.transAxes)
    
    ax2.plot(Star.star_spec['v'], CCFstar_tot[j], '-k', lw=1)
    ax2.plot(
        [depth_l_fe1[j], depth_l_fe2[j], depth_l_oh0[j]],
        (np.array([depth_d_fe1[j], depth_d_fe2[j], depth_d_oh0[j]])-1)/-1,
        's', ms=1, mew=1, color='tab:grey')

    ax2.set_ylim(np.min(CCFstar_tot)-0.05, 1.05)
    ax2.set_xlim(-300, 300)
    ax2.set_xlabel(fr'Velocity [km s$^{-1}$]',  size='small')
    ax2.set_ylabel(f'Relative Flux',  size='small')

    oh_line = np.array(depth_d_oh0)
    fe_line = (np.array(depth_d_fe1) + np.array(depth_d_fe2)) / 2
    line_ratio = oh_line/fe_line

    ax3.plot(visible_phi[:j], line_ratio[:j], 'o', c='tab:grey', ms=3, mew=3)
    ax3.plot(visible_phi[j], line_ratio[j], 'o', c='k', ms=3, mew=3)

    ax3.set_ylim(np.min(line_ratio)-0.1, np.max(line_ratio)+0.1)
    ax3.set_xlim(-0.2, 1.2)
    ax3.set_xlabel(r'Phase',  size='small')
    ax3.set_ylabel('Line depth ratio (OH/Fe)',  size='small')

    for axes in [ax2, ax3]:
        axes.tick_params(axis='both', which='both', labelsize='small', right=True, 
                    top=True, direction='in', width=.7)

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
        f'{outdir}/phase{visible_phi[j]:1.2f}.png', format='png', bbox_inches='tight')
    plt.close()

def spot_view(
    Star, grid, xyz, visible_phi, xyz2_box, CCFstar_tot, CCFstar_quiet, 
    outdir, cpu_use, depth_d_fe1, depth_d_fe2, depth_d_oh0, depth_l_fe1, 
    depth_l_fe2, depth_l_oh0, text1, text2
    ):

    # ------------------------------------------
    print('\nPlotting...\n')

    func = partial(
        spot_view_pel, Star, grid, xyz, visible_phi, xyz2_box, 
        CCFstar_tot, CCFstar_quiet, outdir, depth_d_fe1, depth_d_fe2, 
        depth_d_oh0, depth_l_fe1, depth_l_fe2, depth_l_oh0, text1, text2
        )
    pqdm(np.arange(len(visible_phi)), func, n_jobs=cpu_use)
   
    status, _ = subprocess.getstatusoutput('convert')
    if status <= 1:
        print('\nNow making .gif\n')
        os.chdir(outdir)
        os.system('convert -delay 20 +dither $(ls *.png | sort -V) temp.gif')
        os.system('convert temp.gif \( +clone -set delay 300 \) +swap +delete temp.gif')
        os.system(f'mv temp.gif "{outdir.split("/")[-1]}.gif"')
    else:
        print('\n"convert" function not found, skip making .gif\n')
    # return xyz2_box, spot_surface, st_surface


def debug_plot(grid, phase, iminy, iminz, imaxy, imaxz, xyz2):
    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    st_surface, radius = init_surface(grid, 0.1797, 0.3463)
    ax.imshow(
        st_surface, extent=[-radius, radius, -radius, radius], cmap='Oranges_r')

    ax.axvline(iminy-radius, lw=1, color='tab:red')
    ax.axvline(imaxy-radius, lw=1, color='tab:blue')
    ax.axhline(iminz-radius, lw=1, color='tab:green')
    ax.axhline(imaxz-radius, lw=1, color='k')

    visible = xyz2[0, :] > 0
    ax.plot(xyz2[1, :][visible]*radius, xyz2[2, :][visible]*radius, '-', lw=1, c='k')

    if os.path.exists('./debug'):
        shutil.rmtree('./debug')
        os.mkdir('./debug')
    else:
        os.mkdir('./debug')   
    f.savefig(
            f'./debug/phase{phase:1.2f}.png', format='png', bbox_inches='tight')


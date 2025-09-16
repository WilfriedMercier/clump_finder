#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Visualization routine of the ouput of substructure detection.
"""

import regions
import copy
import os.path                             as     opath
import astropy.units                       as     u
import numpy                               as     np
import matplotlib.pyplot                   as     plt
import matplotlib.colors                   as     mcol
from   numpy.typing                        import NDArray
from   matplotlib.gridspec                 import GridSpec
from   astropy.visualization.mpl_normalize import simple_norm

def plot_im_residuals(
        image:        NDArray[np.floating],
        res:          NDArray[np.floating], 
        mask:         NDArray[np.bool],
        segmap:       NDArray[np.integer],
        band:         str,
        idd:          int,
        z:            float,
        bg_mask:      NDArray[np.bool] | None = None,
        cmap:         str | mcol.Colormap     = 'plasma',
        plotPath:     str                     = '',
        zoom_bounds:  tuple[int, int, int, int] | None = None,
        bulge_params: list[float]             = []
    ) -> None:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>
    
    Produce a plot with image, model, and residuals, showing the substructure segmentation map.
    
    :param image: image of the galaxy in the given band
    :type image: `NDArray`_
    :param res: residual image 
    :type res: `NDArray`_
    :param mask: segmentation mask with :python:`True` corresponding to the galaxy and :python:`False` outside of the galaxy
    :type mask: `NDArray`_
    :param segmap: substructure segmentation map
    :type segmap: `NDArray`_
    :param band: band for this image
    :type band: :python:`str`
    :param idd: ID of the galaxy
    :type idd: :python:`int`
    :param z: redshift of the galaxy
    :type z: :python:`float`
    :param bg_mask: mask with :python:`True` for the background pixels and :python:`False` otherwise
    :type bg_mask: `NDArray`_
    :param cmap: colormap used for the image and model
    :type cmap: :python:`str` or `Colormap`_
    :param plotPath: output path for the figure
    :type plotPath: :python:`str`
    :param zoom_bounds: list of bounds (ymin, ymax, xmin, xmax) to zoom-in on the plots. If :python:`None`, there is no zoom.
    :type zoom_bounds: :python:`(int, int, int, int)` or :python:`None`
    :param bulge_params: list of parameters to plot the bulge mask on top of the image. Parameters are: [x position, y position, radius in :math:`\rm pixels`]
    :type bulge_params: :python:`list`
    '''
    
    if zoom_bounds is None: zoom_bounds = (0, res.shape[1], 0, res.shape[0])
    
    # Masked versions of the residuals
    res_msk         = copy.deepcopy(res)
    res_msk[~mask]  = np.nan
    
    # Outer regions of the mask
    res_nmsk        = copy.deepcopy(res)
    res_nmsk[~bg_mask if bg_mask is not None else mask] = np.nan
    
    # Masked version of the image
    im_msk          = copy.deepcopy(image)
    im_msk[~mask]   = np.nan
    
    # Outer regions of the image
    im_nmsk         = copy.deepcopy(image)
    im_nmsk[mask]   = np.nan
    
    # Create a bulge artist to add to the images
    center      = regions.core.PixCoord(bulge_params[0] - zoom_bounds[0], bulge_params[1] - zoom_bounds[2])
    mask_bulge  = regions.EllipsePixelRegion(center = center, 
                                             width  = 2 * bulge_params[2], 
                                             height = 2 * bulge_params[2],
                                             angle  = 0 * u.deg #type: ignore
                                            )
    
    bulge_artist1 = mask_bulge.as_artist()
    bulge_artist2 = mask_bulge.as_artist()
    bulge_artist0 = mask_bulge.as_artist()
        
    f    = plt.figure(figsize=(9, 4))
    gs   = GridSpec(2, 3, wspace=0.03, hspace=0.5, height_ratios=[0.97, 0.03])
    
    ax0  = f.add_subplot(gs[0, 0])
    ax1  = f.add_subplot(gs[0, 1])
    ax2  = f.add_subplot(gs[0, 2])
    
    axc0 = f.add_subplot(gs[1, 0])
    axc1 = f.add_subplot(gs[1, 1])
    axc2 = f.add_subplot(gs[1, 2])
        
    ax0.set_title('Image',          size=14)
    ax1.set_title('Residuals',      size=14)
    ax2.set_title('Clump seg. map', size=14)
    
    if   not np.all(np.isnan(im_msk)):  norm = simple_norm(im_msk,  stretch='sqrt')  
    elif not np.all(np.isnan(im_nmsk)): norm = simple_norm(im_nmsk, stretch='sqrt')
    else:
        print(f'NaN everywhere in galaxy {idd}.')
        return
    
    ret0 = ax0.imshow(im_msk[zoom_bounds[0]:zoom_bounds[1], zoom_bounds[2]:zoom_bounds[3]],
                        origin = 'lower',
                        cmap   = cmap,
                        norm   = norm #type: ignore
                        )
    
    art = ax0.add_artist(bulge_artist0)

    art.set_color('magenta') #type: ignore
    art.set_linestyle('--')  #type: ignore
    art.set_zorder(100)
    
    _    = ax0.imshow(im_nmsk[zoom_bounds[0]:zoom_bounds[1], zoom_bounds[2]:zoom_bounds[3]],
                        origin = 'lower',
                        cmap   = cmap,
                        norm   = norm, #type: ignore
                        alpha  = 0.3
                        )
    
    lim  = 2 * np.nanmax([np.abs(np.nanstd(res_msk)), np.abs(np.nanstd(res_msk))])
    norm = mcol.Normalize(vmin=-lim, vmax=lim)
    
    ret1 = ax1.imshow(res_msk[zoom_bounds[0]:zoom_bounds[1], zoom_bounds[2]:zoom_bounds[3]],
                        origin = 'lower', 
                        cmap   = 'bwr', 
                        norm   = norm
                        )
    
    art = ax1.add_artist(bulge_artist1)
    art.set_color('magenta') #type: ignore
    art.set_linestyle('--')  #type: ignore
    art.set_zorder(100)
    
    _    = ax1.imshow(res_nmsk[zoom_bounds[0]:zoom_bounds[1], zoom_bounds[2]:zoom_bounds[3]],
                        origin = 'lower', 
                        cmap   = 'bwr', 
                        norm   = norm,
                        )
    
    # Symmetric min/max bound
    lim      = np.nanmax([-np.nanmin(segmap), np.nanmax(segmap)])
    lim      = lim if lim > 0 else 1
    
    # Cmap used to created the segmented colormap
    cmap     = plt.cm.get_cmap('rainbow')
    nb       = 2*lim + 1
    
    # New segmented custom colormap (0 = white, others are taken from the continous cmap)
    colors        = cmap(np.linspace(0, 1, nb))
    colors[nb//2] = (1, 1, 1, 1)
    cmap          = mcol.LinearSegmentedColormap.from_list('customClump', colors, N=nb) #type: ignore
    
    m0       = segmap == 0
    sNan     = segmap.astype(float)
    sNan[m0] = np.nan
    sNan     = segmap
    
    ret2 = ax2.imshow(sNan[zoom_bounds[0]:zoom_bounds[1], zoom_bounds[2]:zoom_bounds[3]],
                        origin = 'lower', 
                        cmap   = cmap, 
                        norm   = mcol.Normalize(vmin=-lim, vmax=lim) #type: ignore
                        )
    
    art = ax2.add_artist(bulge_artist2)
    art.set_color('magenta') #type: ignore
    art.set_linestyle('--')  #type: ignore
    art.set_zorder(100)
    
    # Show contour to distinguish the mask
    for ax in (ax0, ax1, ax2):
        
        ax.contour(mask[zoom_bounds[0]:zoom_bounds[1], zoom_bounds[2]:zoom_bounds[3]], 
                    colors=['k']
                    )
        
        tmp_cont = m0[zoom_bounds[0]:zoom_bounds[1], zoom_bounds[2]:zoom_bounds[3]]
        
        if np.any(~tmp_cont):
            ax.contour(tmp_cont,
                        colors     = ['k'], 
                        linewidths = [0.5]
                        )
    
    ax0.set_xlabel(' ')
    ax1.set_xlabel(' ', size=15)
    ax2.set_xlabel(' ')
    ax0.set_ylabel(' ', size=15)
    
    for ax in (ax0, ax1, ax2): ax.tick_params(labelsize=10, direction='in')
    for ax in (ax1, ax2):
        ax.set_ylabel(' ')
        ax.set_yticklabels([])
    
    # Add a title
    plt.figtext(0.5, 1, f'{band} - ID {idd} - ' + r'$z = $' + f'{z:.1f}' , fontsize=15, horizontalalignment='center')
    
    # Add colourbars
    col1 = plt.colorbar(ret1, cax=axc1, orientation='horizontal')

    col1.ax.tick_params(labelsize=18)
    col1.set_label(r'MJy$\,$sr$^{-1}$', fontsize=15)
    
    col2 = plt.colorbar(ret2, cax=axc2, orientation='horizontal')

    col2.ax.tick_params(labelsize=18)
    col2.set_label(r'Sub-structure', fontsize=15)
    col2.ax.set_xticklabels([])
    
    col0 = plt.colorbar(ret0, cax=axc0, orientation='horizontal')

    col0.ax.tick_params(labelsize=18)
    col0.set_label(r'MJy$\,$sr$^{-1}$', fontsize=15)
    
    plt.savefig(opath.join(plotPath, f'{idd}_{band}.pdf'), 
                bbox_inches = 'tight', 
                transparent = True
                )
    
    plt.close()
    
    return
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Miscellaneous functions.
"""

import regions
import logging
import astropy.units       as u
import astropy.cosmology   as cosmology
import numpy               as np
from   numpy.typing        import NDArray

def generate_bulge_mask(
        ra_pix     : int,
        dec_pix    : int,
        z          : float,
        shape      : tuple[int, int],
        cosmo      : cosmology.Cosmology,
        radius_kpc : int | float = 1,
        pix_size   : int | float = 0.03
    ) -> NDArray[np.bool]:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

    Generate a mask encompassing a circular bulge of a given size with :python:`True` for pixels inside the bulge and :python:`False` for those outside.

    :param int ra_pix: position along the RA axis in pixels
    :param int dec_pix: position along the Dec axis in pixels
    :param float z: redshift of the source used to convert physical sizes to angles
    :param shape: shape of the returned mask
    :type shape: (int, int)
    :param cosmo: cosmological model used to convert physical sizes to angles
    :type cosmo: astropy.cosmology.Cosmology

    Keyword parameters
    ------------------

    :param radius_kpc: radius of the bulge in kpc
    :type radius_kpc: int or float
    :param pix_size: size of a pixel in arcsec
    :type pix_size: int or float

    :returns: boolean mask delimiting the bulge
    :rtype: numpy.ndarray[bool]
    '''
    
    # New bulge mask is simply 1kpc radius to avoid problematic bulge models
    center      = regions.core.PixCoord(ra_pix, dec_pix)
    radius      = np.rad2deg((u.Quantity(radius_kpc, 'kpc') / cosmo.angular_diameter_distance(z)).to('').value) * 3600 / pix_size #type: ignore

    mask_bulge  = regions.EllipsePixelRegion(center = center, 
                                             width  = 2 * radius, 
                                             height = 2 * radius,
                                             angle  = 0 * u.deg, #type: ignore
                                            )
    
    mask_bulge  = mask_bulge.to_mask().to_image(shape).astype(bool) #type: ignore

    return mask_bulge

def find_neighbours(pixel       : tuple[int, int], 
                    pixels      : list[tuple[int, int]], 
                    max0        : int,
                    max1        : int,
                    noTypeCheck : bool = False,
                   ) -> list[tuple[int, int]]:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>
    
    Given a pixel position and a list of all pixel positions, find neighbours that are in the list.
    
    .. note::
        
        Neighbours means adjacent pixels to the left, right, bottom, and top only.
        
    :param pixel: (Y, X) position of the given pixel
    :type pixel: (int, int)
    :param pixels: (Y, X) position of all pixels
    :type pixels: list[(int, int)]
    :param int max0: maximum position along axis 0
    :param int max1: maximum position along axis 1

    Keyword parameters
    ------------------
    
    :param bool noTypeCheck: whether not to perform type conversion. Type conversion slows down the code but ensures data are properly cast to the right format to find neighbours.
    
    :returns: list of the (Y, X) positions of the adjacent pixels in the input pixel list
    :rtype: :python:`list[(int, int)]`
    '''
    
    # Perform type conversion to avoid type issues when comparing
    if not noTypeCheck:
        pixel  = list(pixel) #type: ignore
        pixels = [list(i) for i in pixels] #type: ignore
    
    tmp_pix    = []
    
    # Check if pixel to the left belongs to the list
    pix_next   = [pixel[0] - 1, pixel[1]]
    
    if pixel[0] > 0 and pix_next in pixels: tmp_pix.append(pix_next)
        
    # Check if pixel to the right belongs to the list
    pix_next   = [pixel[0] + 1, pixel[1]]
    
    if pixel[0] < max0 and pix_next in pixels: tmp_pix.append(pix_next)
        
    # Check if pixel above belongs to the list
    pix_next   = [pixel[0], pixel[1] - 1]
    
    if pixel[1] > 0 and pix_next in pixels: tmp_pix.append(pix_next)
        
    # Check if pixel below belongs to the list
    pix_next   = [pixel[0], pixel[1] + 1]

    if pixel[1] < max1 and pix_next in pixels: tmp_pix.append(pix_next)
        
    return tmp_pix

def compute_surface_mask(segmap  : NDArray[np.integer], 
                         value   : int = 1,
                         surface : int = 0
                        ) -> np.ndarray | None:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>
    
    Compute a mask that keeps pixels that satisfy the surface threshold described below. Each independent structure will have a unique integer associated to it.
    
    .. important::
        
        Pixels are selected by the mask if they form a contiguous structure whose surface is larger than the given minimum surface. Contiguity is defined here as adjacent pixels along the left, right, bottom, and top directions (i.e. no diagonal).
    
    .. note::
        
        If the given :python:`value` is not in the segmentation map, the mask is returned as :python:`None`.
    
    :param segmap: segmentation map used to select the pixels forming contiguous structures
    :type segmap: numpy.ndarray

    Keyword parameters
    ------------------

    :param int value: value in the segmentation mask that is used to select the pixels. In other words, the mask will only be computed for pixels whose value is equal to :python:`value`.
    :param int surface: minimum surface in pixels that a contiguous structure must have in order to be kept in the output mask
    
    :returns: mask (with a shape equal to :python:`segmap.shape`) with one integer per structure that matches the surface criterion. A value of 0 means no structure (i.e. background pixels).
    :rtype: 
        - :python:`None` if :python:`value not in segmap` else
        - numpy.ndarray[int]
    
    :raises:
        * :python:`ValueError` if :python:`segmap.ndim != 2 or surface < 0`
        * :python:`TypeError` if :python:`not isinstance(surface, int)`
    '''

    logger = logging.getLogger(__name__)
    
    try:
        if segmap.ndim != 2:
            raise ValueError(f'segmap has {segmap.ndim} dimensions but it must have only two (i.e. an image).')
    except ValueError as e:
        logger.error(str(e))
        raise

    try:
        if not isinstance(surface, int): 
            raise TypeError(f'surface has type {type(surface)} but it must be an integer.')
    except TypeError as e:
        logger.error(str(e))
        raise

    try:
        if surface < 0:
            raise ValueError(f'surface has value {surface} but it must be larger than or equal to 0.')
    except ValueError as e:
        logger.error(str(e))
        raise

    # If no value corresponds in the segmap, no need to derive a mask
    if value not in segmap: return
        
    # Output binary mask (by default no pixel is associated to the mask, i.e. False everywhere)
    out      = np.full(segmap.shape, 0, dtype=int)
    
    # Pixels that the surface mask must be applied onto: it corresponds to a collection of (Y, X) coordinates.
    pixels   = list(list(i) for i in np.asarray(np.where(segmap == value)).T)
    
    # Set of pixels that have already been tested
    pix_list = []
    
    shp0     = segmap.shape[0] - 1
    shp1     = segmap.shape[1] - 1
    
    # Counter that associates a number to each new structure found
    cnt      = 1
    
    # Loop through each pixel and check its neighbours and the neighbours of their neighbours (and so on...)
    for pixel in pixels:
        
        # Temporary set of pixels that belong to the current structure being tested
        tmp_pix = [pixel]
        
        # Do not add pixel if already tested, i.e. if it belongs to pix_list, i.e. if it has already been associated to a structure
        if pixel in pix_list: continue
        
        #################################################
        #         Find neighbours of neighbours         #
        #################################################
        
        for tpix in tmp_pix:
            
            # Neighbours of the current pixel
            positions = find_neighbours(tpix, pixels, shp0, shp1) #type: ignore
            
            # Go through new neighbours and only add them if they are not already in the temporary set of pixels being tested
            for position in positions:
                if position not in tmp_pix:
                    tmp_pix.append(position) #type: ignore
            
        #######################################################################
        #           Add structure to the list if sufficiently large           #
        #######################################################################
            
        if len(tmp_pix) > surface:
            
            #print(np.asarray(tmp_pix))
            coord      = tuple(np.asarray(tmp_pix).T)
            
            # Update the output mask
            out[coord] = cnt
            cnt       += 1
            
        # Save those pixels to not search again for a new structure when running into them later on
        pix_list.extend(tmp_pix)
            
    return out
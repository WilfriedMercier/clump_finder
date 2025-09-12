#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Miscellaneous functions.
"""

import regions
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
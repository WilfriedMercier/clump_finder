#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Detection curves used for the intrinsic detection method.

See `Mercier et al. (2025) <https://arxiv.org/abs/2506.13881>`_.
"""

import astropy.cosmology as     cosmology
import numpy             as     np
from   numpy.typing      import NDArray
from   typing            import overload

@overload
def mag_detection_curve(z: NDArray, cosmo : cosmology.Cosmology) -> NDArray: ...

@overload
def mag_detection_curve(z: float | int, cosmo : cosmology.Cosmology) -> float: ...

def mag_detection_curve(
        z     : float | int | NDArray, 
        cosmo : cosmology.Cosmology
    ) -> float | NDArray:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>
    
    Compute the AB magnitude detection curve for a substructure evaluated at redshift z.
    
    Note:
        To get the detection threshold on the pixel level, one must divide the corresponding flux by 20 since the smallest substructures must be made of 20 pixels.

    :param z: redshift of the galaxy
    :type z: :python:`float`, :python:`int`, or :python:`numpy.ndarray`
    :param cosmo: cosmology used to convert to intrinsic physical quantities
    :type cosmo: astropy.cosmomology.Cosmology

    :rtype: :python:`float` or :python:`numpy.ndarray`
    '''
    
    dl = cosmo.luminosity_distance(z).to('Mpc').value #type: ignore
    
    return 8.33 - 2.5*np.log10((1+z) / dl / dl)

@overload
def flux_detection_curve_pixel_level(z: NDArray, cosmo : cosmology.Cosmology) -> NDArray: ...

@overload
def flux_detection_curve_pixel_level(z: float | int, cosmo : cosmology.Cosmology) -> float: ...

def flux_detection_curve_pixel_level(
        z     : float | int | NDArray, 
        cosmo : cosmology.Cosmology
    ) -> float | NDArray:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>
    
    Compute the flux detection curve in MJy/sr for a pixel in a galaxy at redshift z.
    
    :param z: redshift of the galaxy
    :type z: :python:`float`, :python:`int`, or :python:`numpy.ndarray`
    :param cosmo: cosmology used to convert to intrinsic physical quantities
    :type cosmo: astropy.cosmomology.Cosmology

    :rtype: :python:`float` or :python:`numpy.ndarray`
    '''

    mag_threshold = mag_detection_curve(z, cosmo)

    # Convert to a pixel-level flux threshold in MJy/sr by
    # 1. Converting AB mag to Jy
    # 2. Converting Jy to MJy
    # 3. Converting MJy to MJy/sr by dividing by the area of a substructure used to define the mag threshold
    return 10**((8.9 - mag_threshold) / 2.5) / 1e6 / (20 * 2.11590909090909E-14)

@overload
def surface_detection_curve(z: NDArray, cosmo : cosmology.Cosmology) -> NDArray: ...

@overload
def surface_detection_curve(z: float | int, cosmo : cosmology.Cosmology) -> float: ...

def surface_detection_curve(
        z     : float | int | NDArray, 
        cosmo : cosmology.Cosmology
    ) -> float | NDArray:
    '''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>
    
    Compute the surface detection curve evaluated at redshift z in arcsec^2.

    :param z: redshift of the galaxy
    :type z: :python:`float`, :python:`int`, or :python:`numpy.ndarray`
    :param cosmo: cosmology used to convert to intrinsic physical quantities
    :type cosmo: astropy.cosmomology.Cosmology

    :rtype: :python:`float` or :python:`numpy.ndarray`
    '''
    
    da = cosmo.angular_diameter_distance(z).to('Gpc').value #type: ignore
    
    return 0.054 / da / da
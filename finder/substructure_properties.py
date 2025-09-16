#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Functions useful to extract the flux and area of substructures after detection.
"""

import numpy        as     np
from   numpy.typing import NDArray

def compute_property_substructure(
        idc        : int,
        image      : NDArray[np.floating], 
        sub_segmap : NDArray[np.integer]
    ) -> tuple[float, int]:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

    Determine the flux and area of a given substructure in a galaxy.

    .. note::

        The flux is given in the same unit as :python:`image` (i.e. sum of pixels) and the area in :math:`\rm pixels`. 

    :param idc: ID of the substructure. It must be a value present in the substructure segmentation map.
    :type idc: :python:`int`
    :param image: residual map used to estimate the flux of the substructure
    :type image: `NDArray`_
    :param sub_segmap: segmentation map of the detected substructures
    :type sub_segmap: `NDArray`_

    :raises: :python:`ValueError` if :python:`idc not in sub_segmap`

    :returns: Flux and area of the substructure.
    :rtype: :python:`(float, int)`
    '''

    # Checking that ID is correct
    if idc not in sub_segmap:
        raise ValueError(f'Substructure ID is {idc} but only IDs {np.unique(sub_segmap)} are in the segmentation map.')

    # Total flux of detected substructure
    flux = np.nansum(res_masked := image[sub_segmap == idc])

    # Area in pixel of detected substructure
    area = res_masked.size

    return flux, area #type: ignore

def compute_properties_substructures(
        image      : NDArray[np.floating], 
        sub_segmap : NDArray[np.integer]
    ) -> tuple[list[int], list[float], list[int]]:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

    Determine the flux and area of each substructure in the given galaxy.
    
    .. note::

        Flux values are given in the same unit as :python:`image` (i.e. sum of pixels) and areas are in :math:`\rm pixels`. 

    :param image: image used to estimate the flux in substructures
    :type image: `NDArray`_
    :param sub_segmap: segmentation map of the detected substructures
    :type sub_segmap: `NDArray`_

    :returns: IDs of substructures, their flux, and their area in pixels.
    :rtype: :python:`(list[int], list[float], list[int])`
    '''

    flux_sub = []
    area_sub = []

    # List of substructure IDs
    id_sub   = list(np.sort(np.unique(sub_segmap)))
    
    # Looping through clumps
    for idc in id_sub:

        flux, area = compute_property_substructure(idc, image, sub_segmap)

        flux_sub.append(flux)
        area_sub.append(area) 

    return id_sub, flux_sub, area_sub
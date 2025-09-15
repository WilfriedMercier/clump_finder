#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Functions useful to extract the flux and area of substructures after detection.
"""

import logging
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

    :param int idc: ID of the substructure ID. It must be a value in the substructure segmentation map.
    :param image: residual map used to estimate the flux of the substructure
    :type image: numpy.ndarray
    :param sub_segmap: segmentation map of the detected substructures
    :type sub_segmap: numpy.ndarray

    :raises: ValueError if :python:`idc not in sub_segmap`

    :returns: substructure flux (same unit as :python:`image`) and its area in pixels
    :rtype: :python:`(float, int)`
    '''

    logger = logging.getLogger(__name__)

    logger.debug(f'Computing properties of substructure ID {idc}.')

    # Checking that ID is correct
    try:
        if idc not in sub_segmap:
            raise ValueError(f'Substructure ID is {idc} but only IDs {np.unique(sub_segmap)} are in the segmentation map.')
    except ValueError as e:
        logger.error(str(e))
        raise

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

    :param image: image used to estimate the flux in substructures
    :type image: numpy.ndarray
    :param sub_segmap: segmentation map of the detected substructures
    :type sub_segmap: numpy.ndarray

    :returns: IDs of substructures, their flux (same unit as :python:`image`), and their area in pixels
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
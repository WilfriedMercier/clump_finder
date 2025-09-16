#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Functions used to determine the the background threshold in an image.
"""

import numpy             as     np
from   numpy.typing      import NDArray

from .finder             import ClumpFinder

def bg_threshold_n_sigma(image    : NDArray[np.float32], 
                         model    : NDArray[np.float32] | float, 
                         mask_bg  : NDArray[np.bool   ] | bool,
                         n_sigma  : int | float = 2
                        ) -> float:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>
    
    Given an image with background pixels, measure n times the standard deviation of the background in the residuals.
    
    :param image: image of the galaxy
    :type image: `NDArray`_
    :param model: 2D model of the galaxy. If :python:`image` corresponds to the residuals, use :python:`model = 0`.
    :type model: `NDArray`_
    :param mask_bg: mask with :python:`True` for pixels that belong to the background
    :type mask_bg: `NDArray`_
    :param n_sigma: number of standard deviations of the background to estimate
    :type n_sigma: :python:`int` or :python:`float`
    
    :returns: n times the standard deviation of the background signal in the residuals.
    :rtype: :python:`float`
    '''

    return n_sigma * np.nanstd((image - model)[mask_bg]) #type: ignore

def find_bg_threshold(image     : NDArray[np.float32], 
                      model     : NDArray[np.float32] | float, 
                      mask_bg   : NDArray[np.bool   ] | bool,
                      surface   : int,
                      positive  : bool = True,
                      precision : float = 0.1
                     ) -> float:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>
    
    Find the lowest background threshold that does not detect substructures in the background pixels of a residual image.
    
    :param image: image of the galaxy
    :type image: `NDArray`_
    :param model: 2D model of the galaxy. If :python:`image` corresponds to the residuals, use :python:`model = 0`.
    :type model: `NDArray`_
    :param mask_bg: mask with :python:`True` for pixels that belong to the background
    :type mask_bg: `NDArray`_
    :param surface: minimum surface in pixels that a structure must have in order to be detected
    :type surface: :python:`int`
    :param positive: whether to find the best threshold for substructures with 

        - positive fluxes (:python:`positive = True`) or
        - negative fluxes (:python:`positive = False`)

    :type positive: :python:`bool`

    :param precision: precision on the flux threshold used to stop the dichotomy
    :type precision: :python:`float`
    
    :returns: Number of standard deviations in the background regions of the residuals.
    :rtype: :python:`float`
    '''
    
    def cond(clump_map: np.ndarray, positive: bool):
        
        if positive: return np.any(clump_map[~np.isnan(clump_map)] > 0) #type: ignore
        else:        return np.any(clump_map[~np.isnan(clump_map)] < 0) #type: ignore

    # Define a new clump finder that only acts on the background signal
    bg_clump_finder = ClumpFinder(image, 
                                  mask    = mask_bg,
                                  mask_bg = mask_bg
                                 )
    
    # Starting point is threshold = the median value of the background
    n_sigma          = 0.0
    n_sigma_up       = 10.0
    bg_clump_map     = bg_clump_finder.detect(model, n_sigma_up, surface=surface)
    
    # Need to find min and max bounds where we know that clumps are detected at the min bound
    # But not at the max bound, i.e. best solution is in between
    while cond(bg_clump_map, positive):
        
        n_sigma      = n_sigma_up
        n_sigma_up  *= 2
        bg_clump_map = bg_clump_finder.detect(model, n_sigma_up, surface=surface)
        
    # Loop until we reach the required precision
    while n_sigma_up - n_sigma > precision:
        
        # Search for clumps at the middle of the range
        n_sigma_mid  = (n_sigma_up + n_sigma)/2
        bg_clump_map = bg_clump_finder.detect(model, n_sigma_mid, surface=surface)
        
        # If clumps are present, we know that the threshold is in the range [mid, max]
        # Otherwise it is in the range [min, mid]
        if cond(bg_clump_map, positive):
            n_sigma    = n_sigma_mid
        else:
            n_sigma_up = n_sigma_mid
            
    # Ensure that n_sigma gets the most precise estimate if the loop ended with n_sigma_up being updated
    return n_sigma_mid #type: ignore
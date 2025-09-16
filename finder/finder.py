#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Utilities for substructure detection.
"""

import numpy        as     np
from   numpy.typing import NDArray

from   .misc        import compute_surface_mask

class ClumpFinder:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>
    
    Class used to find substructures in residual images of galaxies using the technique described in `Mercier et al. (2025) <https://arxiv.org/abs/2506.13881>`_.

    .. important::

        One can provide the residual image instead of the galaxy image. 
        In such a case, call :meth:`~clump_finder.finder.ClumpFinder.detect` with :python:`model_im = 0` to detect substructures in the residuals.
    
    :param image: image of the galaxy in the given band
    :type image: `NDArray`_
    
    :param mask: segmentation mask for the galaxy with :python:`True` for pixels belonging to the galaxy. If :python:`None`, the mask is set to :python:`True` (i.e. all pixels are considered to belong to the galaxy).
    :type mask: `NDArray`_ or :python:`bool` or :python:`None`
    :param mask_bg: segmentation mask for the background only with :python:`True` for pixels belonging to the background. If :python:`None`, the background cannot be estimated.
    :type mask_bg: `NDArray`_ or :python:`bool` or :python:`None`
    :param mask_bulge: mask that hides parts of the galaxy that are in the bulge. :python:`True` for pixels in the bulge zone and :python:`False` for pixels outside.
    :type mask_bulge: `NDArray`_ or :python:`bool` or :python:`None`
    '''
    
    def __init__(
            self, 
            image      : NDArray[np.floating], 
            mask       : bool | NDArray[np.bool] | None = None,
            mask_bg    : bool | NDArray[np.bool] | None = None,
            mask_bulge : bool | NDArray[np.bool] | None = None
        ) -> None:
        
        # Storing parameters
        self.image      : NDArray[np.floating]           = image
        self.mask       : NDArray[np.bool] | bool        = mask if mask is not None else True
        self.mask_bg    : NDArray[np.bool] | bool | None = mask_bg
        self.mask_bulge : NDArray[np.bool] | bool | None = mask_bulge

    @property
    def mask_bulge(self) -> NDArray[np.bool] | bool: # type: ignore
        return self._mask_bulge
    
    @mask_bulge.setter
    def mask_bulge(self, mask: NDArray[np.bool] | bool | None): # type: ignore

        if mask is not None:
            self._mask_bulge = mask
        else:
            self._mask_bulge = False
    
    @property
    def mask_bg(self) -> NDArray[np.bool] | bool | None: #type: ignore
        return self._mask_bg

    @mask_bg.setter
    def mask_bg(self, mask: NDArray[np.bool] | bool | None): #type: ignore

        self._mask_bg : NDArray[np.bool] | bool | None = mask

        # Save value of the median background signal in the image when setting the background mask
        if mask is not None:
            self.bg : float | None = np.nanmedian(self.image[mask])
        else:
            self.bg : float | None = None

    def detect(
            self, 
            model_im       : NDArray[np.floating] | float, 
            flux_threshold : float,
            surface        : int
        ) -> NDArray[np.integer]:
        r'''
        .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>
        
        Detect substructures in a residual image given flux and surface thresholds.
        
        The substructures are found by looking at the residuals after subtracting the galaxy model from the image.
        If the image is already the residuals, one can provide :python:`model_im = 0`.
            
        .. note::
            
            This technique identifies both under- and over-densities above a given threshold in the image.
        
        :param model_im: 2D model of the galaxy
        :type model_im: `NDArray`_ or :python:`float`
        :param flux_threshold: threshold used to determine whether a pixel is bright enough or not
        :type flux_threshold: :python:`float`
        :param surface: surface criterion in :python:`pixels` to decide whether a structure is sufficiently extended or not
        :type surface: :python:`int`
        
        :returns: clump detection map with one value per substructure with
        
            - negative values for under-dense pixels
            - positive values for over-dense pixels (i.e. substructures)
            - 0 for the background
            
        :rtype: `NDArray`_ with the same shape as :python:`model_im`
        
        :raises: :python:`ValueError` if 
            
            - there is no background mask given at initialization
            - :python:`n_sigma < 0`
        '''
        
        if self.mask_bg is None:
            raise ValueError('No background mask provided. Please provide a mask at initialisation before calling this method.')
        
        if surface < 0:
            raise ValueError(f'Surface is equal to {surface} but it must be larger than or equal to 0.')
    
        # Residuals 
        res : NDArray[np.floating] = self.image - model_im # type: ignore
        
        # Mask that keeps pixels above the threshold. This includes positive AND negative pixels.
        mask_thr = np.asarray(np.abs(res) < flux_threshold)
        
        # Mask that selects postive pixels
        mask_pos = np.array(res > 0)
        
        # Mask that selects negative pixels
        mask_neg = np.array(res < 0)
        
        # Transform the residual map into an array of -1, 0, and 1 values
        res : NDArray[np.integer] = np.full(res.shape, 0, dtype=int)
        
        # Transform into -1 and 1 values for negative and positive "clumps"
        res[mask_neg]        = -1
        res[mask_pos]        =  1
        
        # Put 0 outside of the galaxy and for pixels with values below the threshold
        res[~self.mask]      = 0
        res[mask_thr]        = 0

        # Mask pixels which are within the bulge mask
        res[self.mask_bulge] = 0
        
        ###################################################
        #             Apply surface criterion             #
        ###################################################
        
        # Compute a surface mask for the under-dense regions (negative clumps)
        neg_res        = compute_surface_mask(res, value=-1, surface=surface) #type: ignore
        #neg_res = None
        
        # Compute a surface mask for the over-dense regions (positive clumps)
        pos_res        = compute_surface_mask(res, value=1,  surface=surface) #type: ignore
        
        res           *= 0
        
        if neg_res is not None:
            mask       = np.array(neg_res > 0)
            res[mask]  = -neg_res[mask] # Negative clumps are associated to negative values
        
        if pos_res is not None:
            mask       = np.array(pos_res > 0)
            res[mask]  = pos_res[mask]

        return res
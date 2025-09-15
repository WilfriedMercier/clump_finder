#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr

from .finder           import ClumpFinder

from .background       import (
    bg_threshold_n_sigma, 
    find_bg_threshold
)

from .detection_curves import (
    mag_detection_curve,
    flux_detection_curve_pixel_level,
    surface_detection_curve
)

from .substructure_properties import (
    compute_property_substructure,
    compute_properties_substructures
)

from .misc import (
    DETECTION_TYPE,
    compute_surface_mask
)
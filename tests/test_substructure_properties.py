#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Test the detection curve functions.
"""

import finder
import pytest
import copy
import numpy        as     np
from   numpy.typing import NDArray

@pytest.fixture
def image_empty(): return np.full((100, 100), 0.0)

@pytest.fixture
def simple_segmap(): 

    segmap           = np.full((100, 100), 0)
    segmap[:10, :10] = 1

    return segmap

@pytest.fixture
def multiple_data():

    image  = np.full((100, 100), 0.0)
    segmap = np.full((100, 100), 0)

    for idd in range(1, 8):

        image[ (idd-1)*10:idd*10, (idd-1)*10:idd*10] = idd * 5
        segmap[(idd-1)*10:idd*10, (idd-1)*10:idd*10] = idd

    return image, segmap

class Test_compute_property_substructure:

    # Test that a ValueError is returned if the ID is not in the segmap
    @pytest.mark.xfail(raises=ValueError)
    def test_id_not_in_segmap(self, 
            image_empty   : NDArray,
            simple_segmap : NDArray[np.int_]
        ):

        flux, area = finder.compute_property_substructure(
            2, 
            image_empty, 
            simple_segmap
        )

    # Test that the flux and area are properly computed for an empty image
    def test_empty_image_with_segmap(self, 
            image_empty   : NDArray,
            simple_segmap : NDArray[np.int_]
        ):


        flux, area = finder.compute_property_substructure(
            1, 
            image_empty, 
            simple_segmap
        )

        assert flux == 0 and area == 100

    # Test that the flux and area are properly computed for an image
    def test_non_empty_image_with_segmap(self, 
            image_empty   : NDArray,
            simple_segmap : NDArray[np.int_]
        ):

        image       = copy.deepcopy(image_empty)
        mask        = simple_segmap == 1
        image[mask] = 5

        flux, area = finder.compute_property_substructure(
            1, 
            image, 
            simple_segmap
        )

        assert flux == 500 and area == 100

class Test_compute_properties_substructures:

    # Test that the IDs of the substructures matches the entire 
    # substructure population
    def test_id_list(self, 
            multiple_data: tuple[NDArray, NDArray[np.int_]]
        ):

        ids, _, _ = finder.compute_properties_substructures(*multiple_data)

        for idd in ids: assert(idd in multiple_data[1])

    # Test that the computed fluxes are all correct
    def test_flux_list(self, 
            multiple_data: tuple[NDArray, NDArray[np.int_]]
        ):

        ids, fluxes, _ = finder.compute_properties_substructures(*multiple_data)

        for idd, flux in zip(ids, fluxes): 
            assert(flux == np.nansum(multiple_data[0][multiple_data[1] == idd]))

    # Test that the computed areas are all correct
    def test_area_list(self, 
            multiple_data: tuple[NDArray, NDArray[np.int_]]
        ):

        ids, _, areas = finder.compute_properties_substructures(*multiple_data)

        for idd, area in zip(ids, areas): 
            assert(area == len(multiple_data[0][multiple_data[1] == idd]))
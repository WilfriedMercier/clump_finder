#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Test the misc submodule.
"""

from   numpy.typing      import NDArray
from   finder.misc       import misc
import astropy.cosmology as     cosmology
import numpy             as     np
import astropy.units     as     u
import pytest
import finder

@pytest.fixture
def cosmology_flat(): return cosmology.FlatLambdaCDM(70, 0.3) # type: ignore

@pytest.fixture
def cosmology_flat_alternative(): return cosmology.FlatLambdaCDM(65, 0.5) # type: ignore

class Test_generate_bulge_mask():
    r'''Test suite for the bulge mask generation function.'''

    # Testing that the mask is properly generated with True and False values
    def test_basic_mask(self, 
            cosmology_flat : cosmology.Cosmology
        ):

        mask = misc.generate_bulge_mask(
            51, 51, 1.0, (100, 100),
            cosmology_flat,
            radius_kpc = 1,
            pix_size   = 0.03
        )

        assert True in mask and False in mask

    # Test that mask generation is robust against alternative cosmology
    def test_basic_mask_alternative_cosmology(self, 
            cosmology_flat_alternative : cosmology.Cosmology
        ):

        mask = misc.generate_bulge_mask(
            51, 51, 1.0, (100, 100),
            cosmology_flat_alternative,
            radius_kpc = 1,
            pix_size   = 0.03
        )

        assert True in mask and False in mask

    # Test that all pixels are True for a big bulge
    def test_too_big_mask(self,
            cosmology_flat : cosmology.Cosmology
        ):

        pix_size   = 0.03

        # This should make a bulge mask bigger than the entire image
        radius_kpc = (cosmology_flat.angular_diameter_distance(1.0).to('kpc') * 75 * pix_size * u.arcsec.to('rad')).value # type: ignore

        mask = misc.generate_bulge_mask(
            51, 51, 1.0, (100, 100),
            cosmology_flat,
            radius_kpc = radius_kpc,
            pix_size   = pix_size
        )

        assert np.all(mask)

    # Test that a bulge mask with bigger pixels but on an image of the same pixel size
    # hides fewer pixels than with small pixels
    def test_size_pixel(self,
            cosmology_flat : cosmology.Cosmology
        ):

        pix_size_small = 0.03
        pix_size_big   = 1

        mask_small_pix = misc.generate_bulge_mask(
            51, 51, 1.0, (100, 100),
            cosmology_flat,
            radius_kpc = 2,
            pix_size   = pix_size_small
        )
        
        mask_big_pix   = misc.generate_bulge_mask(
            51, 51, 1.0, (100, 100),
            cosmology_flat,
            radius_kpc = 2,
            pix_size   = pix_size_big
        )

        assert len(mask_small_pix[mask_small_pix]) > len(mask_big_pix[mask_big_pix])

@pytest.fixture
def mock_segmentation_map():
    
    # Create a mock segmentation map
    segmap               = np.full((100, 100), 0)
    segmap[:10, :10]     = 1
    segmap[35:40, 25:35] = 2
    segmap[75:80, 60:80] = 3
    segmap[7:9, 12:14]   = 4

    return segmap

class Test_compute_surface_mask:

    # Test that the call fails when not providing a 2D array for the segmap
    @pytest.mark.xfail(raises=ValueError)
    def test_not_2D(self):
        
        finder.misc.compute_surface_mask(
            np.array([1]),
            value   = 1,
            surface = 1
        )

    # Test that the call fails when not providing an integer value for the surface
    @pytest.mark.xfail(raises=TypeError)
    def test_surface_not_int(self):

        res = finder.misc.compute_surface_mask(
            np.full((2, 2), 0),
            value   = 1,
            surface = 1.1 # type: ignore
        )

    # Test that the calls fails when providing a negative surface
    @pytest.mark.xfail(raises=ValueError)
    def test_surface_negative(self):

        res = finder.misc.compute_surface_mask(
            np.full((2, 2), 0),
            value   = 1,
            surface = -1
        )

    # Test that the result is None when computing a surface mask 
    # for a value that is not in the segmentation map
    def test_value_not_in_segmap(self):

        res = finder.misc.compute_surface_mask(
            np.full((2, 2), 0),
            value   = 1,
            surface = 1
        )

        assert res is None

    # Test that the output mask is ok when there is a substructure
    # that is large enough to be detected
    def test_output_validity_large_substructure(self,
            mock_segmentation_map : NDArray[np.integer]
        ):

        res = finder.misc.compute_surface_mask(
            mock_segmentation_map,
            value   = 2,
            surface = 20
        )

        assert np.all(res == (mock_segmentation_map == 2))
    
    # Test that the output mask is ok when there is a substructure
    # that is not large enough to be detected
    def test_output_validity_small_substructure(self,
            mock_segmentation_map : NDArray[np.integer]
        ):

        res = finder.misc.compute_surface_mask(
            mock_segmentation_map,
            value   = 4,
            surface = 20
        )

        assert res is not None and np.all(~res)

class Test_find_neighbours:

    # Test that the finder finds no neighbours when there are none
    def test_no_neighbours(self):

        res = finder.misc.find_neighbours(
            (5, 5),
            [(5, 5), (0, 1), (10, 3)],
            20,
            20
        )

        assert res == []

    # Test that the finder finds a single neighbour when there is one
    def test_one_neighbour(self):

        res = finder.misc.find_neighbours(
            (5, 5),
            [(5, 5), (5, 6), (10, 3)],
            20,
            20
        )

        assert res == [[5, 6]]

    # Test that the finder fins a single neighbour when there is one
    # within the limits and one outside them
    def test_one_neighbour_in_one_out(self):

        res = finder.misc.find_neighbours(
            (20, 5),
            [(5, 5), (20, 6), (21, 5)],
            20,
            20
        )

        assert res == [[20, 6]]

    # Test that all neighbours are found when present
    def test_all_neighbours(self):

        neighbours = [(19, 5), (20, 6), (20, 4), (21, 5)]
        
        res = finder.misc.find_neighbours(
            (20, 5),
            neighbours + [(5, 5), (0, 12)],
            100,
            100
        )

        # First check that there are exactly four neighbours found
        assert len(res) == 4

        # For each real neighbour, check that it appears in the result
        for neighbour in neighbours: assert list(neighbour) in res
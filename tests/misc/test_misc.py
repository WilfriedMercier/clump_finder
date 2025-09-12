#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Test the misc submodule.
"""

from   finder.misc       import misc
import astropy.cosmology as     cosmology
import numpy             as     np
import astropy.units     as     u
import pytest

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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Test the background functions.
"""

import finder
import numpy             as     np
import astropy.units     as     u

class Test_bg_threshold_n_sigma():
    r'''Test suite for the bg_threshold_n_sigma function.'''

    # Test that the std of a flat image is 0
    def test_flat_image(self):

        sigma =  finder.bg_threshold_n_sigma(
            np.full((100, 100), 10),
            0, 
            np.full((100, 100), True),
            n_sigma = 1
        )

        assert sigma == 0

    # Test that the std is correct for an image with half of its pixels with a given value
    # and the other half with a different value
    def test_semi_flat_image(self):

        image         = np.full((100, 100), 10)
        image[:50, :] = 0

        sigma =  finder.bg_threshold_n_sigma(
            image,
            0, 
            np.full((100, 100), True),
            n_sigma = 1
        )

        assert sigma == 5

    # Testing that the sigma value is ok for a basic noise image without a model
    # Since the image is randomly generated, we tolerate an error of 1%
    def test_noise_image(self):

        image = np.random.default_rng(seed=666).normal(loc=0, scale=1, size=(100, 100))

        sigma =  finder.bg_threshold_n_sigma(
            image,  # type: ignore
            0, 
            np.full(image.shape, True),
            n_sigma = 1
        )

        assert np.abs(sigma - 1) < 1e-2

    # Test that a twice higher n gives a twice higher std
    def test_double_n(self):

        image         = np.full((100, 100), 10)
        image[:50, :] = 0
        mask          = np.full((100, 100), True)

        sigma_1 =  finder.bg_threshold_n_sigma(
            image,
            0, 
            mask,
            n_sigma = 1
        )

        sigma_2 =  finder.bg_threshold_n_sigma(
            image,
            0, 
            mask,
            n_sigma = 2
        )

        assert sigma_2 == 2*sigma_1

    # Test that removing the same model as the image leads to a null std
    def test_model_subtraction(self):

        image         = np.full((100, 100), 10)
        image[:50, :] = 0

        sigma =  finder.bg_threshold_n_sigma(
            image,
            image, 
            np.full((100, 100), True),
            n_sigma = 1
        )

        assert sigma == 0

    # Test that masking works as expected for a semi-flat image
    def test_masking(self):

        image         = np.full((100, 100), 10)
        image[:50, :] = 0

        sigma =  finder.bg_threshold_n_sigma(
            image,
            0, 
            image == 0,
            n_sigma = 1
        )

        assert sigma == 0
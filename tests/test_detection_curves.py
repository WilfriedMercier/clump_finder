#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Test the detection curve functions.
"""

import finder
import pytest
import numpy             as     np
import astropy.cosmology as     cosmology
from   numpy.typing      import NDArray

@pytest.fixture
def z_values(): return np.linspace(0.1, 10, 10)

@pytest.fixture
def z_values_around_16(): return np.array([0.1, 0.5, 1, 1.6, 1.7, 2, 10])

@pytest.fixture
def cosmo(): return cosmology.FlatLambdaCDM(70, 0.3) # type: ignore

class Test_mag_detection_curve:

    # Test that the magnitude threshold for higher values of z
    # is deeper than for lower values
    def test_variation_with_z(self, 
            z_values : NDArray, 
            cosmo    : cosmology.Cosmology
        ):

        mag = finder.mag_detection_curve(z_values, cosmo)

        assert np.all(mag[1:] > mag[:-1])

class Test_flux_detection_curve_pixel_level:

    # Test that the flux threshold at the pixel level for
    # higher values of z is deeper than for lower values
    def test_variation_with_z(self, 
            z_values : NDArray, 
            cosmo    : cosmology.Cosmology
        ):

        mag = finder.flux_detection_curve_pixel_level(z_values, cosmo)

        assert np.all(mag[1:] < mag[:-1])

class Test_surface_detection_curve:

    # Test that the surface threshold reaches a minimum at z ~ 1.6
    # which is the peak of the angular diameter distance with redshift
    def test_variation_with_z(self, 
            z_values_around_16 : NDArray, 
            cosmo              : cosmology.Cosmology
        ):

        surface = finder.surface_detection_curve(z_values_around_16, cosmo)

        assert np.all(surface[:3] > surface[3]) and np.all(surface[4:] > surface[3])
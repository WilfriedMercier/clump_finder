#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Test the ClumpFinder class.
"""

import finder
import pytest
import pathlib
import astropy.io.fits   as     fits
import astropy.cosmology as     cosmology
import numpy             as     np
from   numpy.typing      import NDArray

@pytest.fixture
def data_99() -> tuple[NDArray, NDArray[np.integer]]:

    # Open residual image
    with fits.open(pathlib.Path('tests/test_data/99_residual.fits')) as hdul:
        image = hdul[0].data # type: ignore
    
    # Open segmentation map
    with fits.open(pathlib.Path('tests/test_data/99_segmentation.fits')) as hdul:
        segmentation = hdul[0].data # type: ignore

    return image, segmentation

@pytest.fixture
def clump_segmap_intrinsic_99() -> NDArray[np.integer]:

    with fits.open(pathlib.Path('tests/test_data/99_csegmap_int.fits')) as hdul:
        return np.unique(hdul[0].data) # type: ignore
    
@pytest.fixture
def clump_segmap_optimal_99() -> NDArray[np.integer]:

    with fits.open(pathlib.Path('tests/test_data/99_csegmap_opt.fits')) as hdul:
        return np.unique(hdul[0].data) # type: ignore

class Test_ClumpFinder:

    # Test that init values of data are properly stored
    def test_init_values_without_bulge_mask(self, 
            data_99 : tuple[NDArray, NDArray[np.integer]]
        ):

        cf = finder.ClumpFinder(
            data_99[0],
            mask       = (m := data_99[1] == 2086),
            mask_bg    = (m_bg := data_99[1] == 0),
            mask_bulge = None
        )

        assert np.all(cf.image == data_99[0])
        assert np.all(cf.mask  == m)
        assert np.all(cf.mask_bg == m_bg)
        assert cf.bg == np.nanmedian(data_99[0][m_bg])
        assert not cf.mask_bulge

    # Test init value of bulge mask when provided
    def test_init_value_bulge_mask(self, 
            data_99 : tuple[NDArray, NDArray[np.integer]]
        ):

        cf = finder.ClumpFinder(
            data_99[0],
            mask       = (m := data_99[1] == 2086),
            mask_bg    = data_99[1] == 0,
            mask_bulge = m
        )

        assert np.all(cf.mask_bulge == m)

    # Test that the setter for the bulge mask works properly
    def test_set_bulge_mask(self,
            data_99 : tuple[NDArray, NDArray[np.integer]]
        ):

        cf = finder.ClumpFinder(
            data_99[0],
            mask       = (m := data_99[1] == 2086),
            mask_bg    = data_99[1] == 0,
            mask_bulge = m
        )

        cf.mask_bulge = None

        assert not cf.mask_bulge

    # Test that the setter for the background mask works properly,
    # including recomputing the background fluctuations
    def test_set_bg_mask(self,
            data_99 : tuple[NDArray, NDArray[np.integer]]
        ):

        cf = finder.ClumpFinder(
            data_99[0],
            mask       = (m := data_99[1] == 2086),
            mask_bg    = data_99[1] == 0,
            mask_bulge = m
        )

        cf.mask_bg = m

        assert np.all(cf.mask_bg == m)
        assert cf.bg == np.nanmedian(data_99[0][m])

    # Test that the setter for the background mask works properly,
    # when setting it to None
    def test_set_bg_mask_to_None(self,
            data_99 : tuple[NDArray, NDArray[np.integer]]
        ):

        cf = finder.ClumpFinder(
            data_99[0],
            mask       = (m := data_99[1] == 2086),
            mask_bg    = data_99[1] == 0,
            mask_bulge = m
        )

        cf.mask_bg = None

        assert cf.mask_bg is None
        assert cf.bg is None

    @pytest.mark.xfail(raises=ValueError)
    def test_detect_no_bg_mask(self,
            data_99 : tuple[NDArray, NDArray[np.integer]]
        ):

        cf = finder.ClumpFinder(
            data_99[0],
            mask       = data_99[1] == 2086,
            mask_bg    = None,
            mask_bulge = None
        )

        cf.detect(0.0, 1.0, 20)

    @pytest.mark.xfail(raises=ValueError)
    def test_detect_negative_surface(self,
            data_99 : tuple[NDArray, NDArray[np.integer]]
        ):

        cf = finder.ClumpFinder(
            data_99[0],
            mask       = data_99[1] == 2086,
            mask_bg    = data_99[1] == 0,
            mask_bulge = None
        )

        cf.detect(0.0, 1.0, -20)

    # Test that the optimal detection works on a real case
    # Because the parameters might have been slightly different
    # We only test that we retrieve the correct unique values
    def test_detect_optimal(self,
            data_99 : tuple[NDArray, NDArray[np.integer]],
            clump_segmap_optimal_99 : NDArray[np.integer]
        ):

        cosmo      = cosmology.FlatLambdaCDM(70, 0.3) # type: ignore

        # Define a bulge mask similar to what was done in the paper
        mask_bulge = finder.generate_bulge_mask(
            data_99[0].shape[0] // 2,
            data_99[0].shape[1] // 2,
            1.069,
            data_99[0].shape, # type: ignore
            cosmo,
            radius_kpc = 1,
            pix_size   = 0.03
        )

        # Setup the clump finder
        cf = finder.ClumpFinder(
            data_99[0],
            mask       = data_99[1] == 2086,
            mask_bg    = (m_bg := data_99[1] == 0),
            mask_bulge = mask_bulge
        )

        # Compute the 2sigma flux threshold used for the optimal detection
        flux_threshold = finder.bg_threshold_n_sigma(
            data_99[0], 
            0.0,
            m_bg,
            n_sigma = 2
        )

        res_unique = np.unique(cf.detect(0.0, flux_threshold, 20))

        for value in res_unique: assert value in clump_segmap_optimal_99
        for value in clump_segmap_optimal_99: assert value in res_unique

    # Test that the intrinsic detection works on a real case
    # Because the parameters might have been slightly different
    # We only test that we retrieve the correct unique values
    def test_detect_intrinsic(self,
            data_99 : tuple[NDArray, NDArray[np.integer]],
            clump_segmap_intrinsic_99 : NDArray[np.integer]
        ):

        cosmo      = cosmology.FlatLambdaCDM(70, 0.3) # type: ignore

        # Define a bulge mask similar to what was done in the paper
        mask_bulge = finder.generate_bulge_mask(
            data_99[0].shape[0] // 2,
            data_99[0].shape[1] // 2,
            z := 1.069,
            data_99[0].shape, # type: ignore
            cosmo,
            radius_kpc = 1,
            pix_size   = (pix_size := 0.03)
        )

        # Setup the clump finder
        cf = finder.ClumpFinder(
            data_99[0],
            mask       = data_99[1] == 2086,
            mask_bg    = (m_bg := data_99[1] == 0),
            mask_bulge = mask_bulge
        )

        # Compute the flux and surface thresholds for the intrinsic detection
        flux_threshold    = finder.mag_detection_curve(z, cosmo)
        surface_criterion = int(finder.surface_detection_curve(z, cosmo) / pix_size / pix_size)

        res_unique        = np.unique(cf.detect(0.0, flux_threshold, surface_criterion))

        for value in res_unique: assert value in clump_segmap_intrinsic_99
        for value in clump_segmap_intrinsic_99: assert value in res_unique

    # Test that the intrinsic detection detects fewer substructures than
    # the optimal one
    def test_detect_intrinsic_fewer_than_optimal(self,
            data_99 : tuple[NDArray, NDArray[np.integer]]
        ):

        cosmo      = cosmology.FlatLambdaCDM(70, 0.3) # type: ignore

        # Define a bulge mask similar to what was done in the paper
        mask_bulge = finder.generate_bulge_mask(
            data_99[0].shape[0] // 2,
            data_99[0].shape[1] // 2,
            z := 1.069,
            data_99[0].shape, # type: ignore
            cosmo,
            radius_kpc = 1,
            pix_size   = (pix_size := 0.03)
        )

        # Setup the clump finder
        cf = finder.ClumpFinder(
            data_99[0],
            mask       = data_99[1] == 2086,
            mask_bg    = (m_bg := data_99[1] == 0),
            mask_bulge = mask_bulge
        )

        # Compute the flux and surface thresholds for the intrinsic detection
        flux_threshold    = finder.mag_detection_curve(z, cosmo)
        surface_criterion = int(finder.surface_detection_curve(z, cosmo) / pix_size / pix_size)

        res_unique_int    = np.unique(cf.detect(0.0, flux_threshold, surface_criterion))

        # Compute the 2sigma flux threshold used for the optimal detection
        flux_threshold = finder.bg_threshold_n_sigma(
            data_99[0], 
            0.0,
            m_bg,
            n_sigma = 2
        )

        res_unique_opt    = np.unique(cf.detect(0.0, flux_threshold, 20))

        assert len(res_unique_int) < len(res_unique_opt)
        
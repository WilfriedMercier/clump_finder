import marimo

__generated_with = "0.15.2"
app = marimo.App(width="columns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We provide an comparison of the intrinsic and optimal detections (as defined in [Mercier et al., 2015]([https://ui.adsabs.harvard.edu/abs/2015ApJ...805...15M/abstract](https://ui.adsabs.harvard.edu/abs/2025arXiv250613881M/abstract))) using four galaxies studied in the paper.""")
    return


@app.cell
def _():
    from   astropy.visualization.mpl_normalize import simple_norm
    from   astropy.cosmology                   import FlatLambdaCDM

    import marimo            as mo
    import matplotlib.pyplot as plt
    import astropy.io.fits   as fits
    import numpy             as np
    import pathlib
    import finder

    plt.style.use('default')

    path = pathlib.Path('data')
    return FlatLambdaCDM, finder, fits, mo, np, path, plt, simple_norm


@app.cell
def _(fits, path):
    ids       = [3, 56, 96, 678]
    redshifts = [1.6833, 1.4373, 1.7058, 2.2707]
    images    = []
    models    = []
    segmaps   = []

    for _idd in ids:

        # Open image
        with fits.open(path / 'images' / f'{_idd}_f444w.fits') as hdul:
            images.append(hdul[0].data)

        # Open model
        with fits.open(path / 'models' / f'{_idd}_f444w.fits') as hdul:
            models.append(hdul[0].data)

        # Open segmentation map
        with fits.open(path / 'segmentation' / f'{_idd}.fits') as hdul:
            segmaps.append(hdul[0].data)
    return ids, images, models, redshifts, segmaps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Compute the local $2\sigma$ flux threshold for each galaxy""")
    return


@app.cell
def _(finder, images, models, segmaps):
    thr = []

    for _image, _model, _segmap in zip(images, models, segmaps):

        thr.append(
            finder.bg_threshold_n_sigma(
                _image,
                _model,
                _segmap == 0,
                n_sigma = 2
            )
        )
    return (thr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Produce a $1\,\rm kpc$ mask to hide the bulge""")
    return


@app.cell
def _(FlatLambdaCDM, finder, images, np, redshifts):
    cosmo       = FlatLambdaCDM(70, 0.3)
    mask_bulges = []

    for _image, _z in zip(images, redshifts):

        _center  = np.array(_image.shape) // 2

        mask_bulges.append(
            finder.generate_bulge_mask(
                _center[0], _center[1], _z,
                _image.shape,
                cosmo,
                radius_kpc = 1,
                pix_size   = 0.03
            )
        )
    return cosmo, mask_bulges


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Perform the detections in the residuals""")
    return


@app.cell
def _(cosmo, finder, images, mask_bulges, models, np, redshifts, segmaps, thr):
    sub_segmaps_opt = []
    sub_segmaps_int = []

    for _image, _model, _segmap, _mask_bulge, _thr, _z in zip(
        images, models, segmaps, mask_bulges, thr, redshifts
        ):

        # Recover the value in the segmentation map corresponding to the galaxy
        _center  = np.array(_image.shape) // 2
        _gal_val = _segmap[_center[0], _center[1]]
    
        # Setup the substructure finder
        _cf = finder.ClumpFinder(
            _image,
            mask = _segmap == _gal_val,
            mask_bg    = _segmap == 0,
            mask_bulge = _mask_bulge
        )

        # Optimal detection
        sub_segmaps_opt.append(
            _cf.detect(_model, _thr, 20)
        )

        # Intrinsic detection:
        # 1. We compute the flux and surface thresholds
        # 2. We perform the detection 
        thr_int = finder.flux_detection_curve_pixel_level(
            _z, cosmo
        )

        surf_int = finder.surface_detection_curve(
            _z, cosmo
        )

        # We cast to an int the surface threshold by going from arcsec2 to pixels
        sub_segmaps_int.append(
            _cf.detect(_model, thr_int, int(surf_int/0.03/0.03))
        )
    return sub_segmaps_int, sub_segmaps_opt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note that, by default the algorithm detects positive and negative substructures. In the paper, we only considered positive ones, that is over-densities with respect to the model. 
    In what follows, we discard negative substructures.
    """
    )
    return


@app.cell
def _(sub_segmaps_int, sub_segmaps_opt):
    sub_segmaps_opt_pos = []
    sub_segmaps_int_pos = []

    for _sint, _sopt in zip(sub_segmaps_int, sub_segmaps_opt):

        _sint[_sint < 0] = 0
        sub_segmaps_int_pos.append(_sint)
    
        _sopt[_sopt < 0] = 0
        sub_segmaps_opt_pos.append(_sopt)
    return sub_segmaps_int_pos, sub_segmaps_opt_pos


@app.cell
def _(plt):
    def show_im(
            f,
            ax_loc, 
            im, 
            seg,
            cmap       = 'plasma',
            norm       = None,
            title      = '',
            mask_bulge = None,
            colors     = 'k'
        ):

        # Showing the galaxy
        ax = f.add_subplot(*ax_loc)

        plt.imshow(
            im, 
            origin = 'lower', 
            cmap   = cmap,
            norm   = norm
        )

        plt.contour(seg, origin='lower', colors=colors)

        if mask_bulge is not None:
            plt.contour(mask_bulge, origin='lower', colors=colors, linestyles='--')

        ax.set_title(title)

        return ax
    return (show_im,)


@app.cell
def _(show_im, simple_norm):
    def figure_galaxy(
            f, 
            col,
            idd,
            image,
            model,
            segmap,
            sub_segmap,
            mask_bulge
        ):
    
        axs = []

        norm = simple_norm(image, stretch='sqrt', percent=99.5)

        # For each plot, we add the bulge mask
        axs.append(show_im(f, (4, 4, 0 + 4*col + 1), image, segmap, norm = norm, title=f'ID {idd}', mask_bulge = mask_bulge, colors='w'))
        axs.append(show_im(f,  (4, 4, 0 + 4*col + 2), model, segmap, norm = norm, title='Model', mask_bulge = mask_bulge, colors='w'))
        axs.append(show_im(f,  (4, 4, 0 + 4*col + 3), image - model, segmap, norm = norm, title='Residuals', mask_bulge = mask_bulge, colors='w'))
        axs.append(show_im(f, (4, 4, 0 + 4*col + 4), sub_segmap,    segmap, cmap = 'gnuplot2', title='Substructures', mask_bulge = mask_bulge, colors='w'))
    
        for ax in axs:
            ax.tick_params(direction='in', right=True, top=True, labelleft=False, labelbottom=False)

    return (figure_galaxy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Showing results for the intrinsic detection""")
    return


@app.cell
def _(
    figure_galaxy,
    ids,
    images,
    mask_bulges,
    mo,
    models,
    plt,
    segmaps,
    sub_segmaps_int_pos,
):
    _f   = plt.figure(figsize=(8, 8), dpi=300)

    for _pos in range(len(images)):

        figure_galaxy(
            _f, 
            _pos, 
            ids[_pos],
            images[_pos], 
            models[_pos],
            segmaps[_pos],
            sub_segmaps_int_pos[_pos],
            mask_bulges[_pos]
        )

    mo.vstack([_f])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Showing results for the optimal detection""")
    return


@app.cell
def _(
    figure_galaxy,
    ids,
    images,
    mask_bulges,
    mo,
    models,
    plt,
    segmaps,
    sub_segmaps_opt_pos,
):
    _f   = plt.figure(figsize=(8, 8), dpi=300)

    for _pos in range(len(images)):

        figure_galaxy(
            _f, 
            _pos, 
            ids[_pos],
            images[_pos], 
            models[_pos],
            segmaps[_pos],
            sub_segmaps_opt_pos[_pos],
            mask_bulges[_pos]
        )

    mo.vstack([_f])
    return


if __name__ == "__main__":
    app.run()

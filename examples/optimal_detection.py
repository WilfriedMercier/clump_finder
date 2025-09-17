import marimo

__generated_with = "0.15.2"
app = marimo.App(width="columns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We provide an example of optimal detection (as defined in [Mercier et al., 2015]([https://ui.adsabs.harvard.edu/abs/2015ApJ...805...15M/abstract](https://ui.adsabs.harvard.edu/abs/2025arXiv250613881M/abstract))) using one of the galaxies studied in the paper.""")
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
    idd      = 56
    redshift = 1.6833

    # Open image
    with fits.open(path / 'images' / f'{idd}_f444w.fits') as hdul:
        image = hdul[0].data

    # Open model
    with fits.open(path / 'models' / f'{idd}_f444w.fits') as hdul:
        model = hdul[0].data

    # Open segmentation map
    with fits.open(path / 'segmentation' / f'{idd}.fits') as hdul:
        segmap = hdul[0].data
    return image, model, redshift, segmap


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
        ax = f.add_subplot(ax_loc)
    
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Visualising the image, model, and residuals""")
    return


@app.cell
def _(image, model, plt, segmap, show_im, simple_norm):
    f   = plt.figure(figsize=(15, 5))
    axs = []

    # Showing the galaxy, model, and residuals
    norm = simple_norm(image, stretch='sqrt', percent=99.9)

    axs.append(show_im(f, 131, image,         segmap, norm = norm, title='Cutout'))
    axs.append(show_im(f, 132, model,         segmap, norm = norm, title='Model'))
    axs.append(show_im(f, 133, image - model, segmap, norm = norm, title='Residuals'))

    for ax in axs:
        ax.tick_params(direction='in', right=True, top=True, labelleft=False, labelbottom=False)

    plt.show()
    return (norm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Compute the local $2\sigma$ flux threhsold""")
    return


@app.cell
def _(finder, image, model, segmap):
    thr = finder.bg_threshold_n_sigma(
        image,
        model,
        segmap == 0,
        n_sigma = 2
    )

    print(thr)
    return (thr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Produce a $1\,\rm kpc$ mask to hide the bulge""")
    return


@app.cell
def _(FlatLambdaCDM, finder, image, np, redshift):
    center     = np.array(image.shape) // 2

    mask_bulge = finder.generate_bulge_mask(
        center[0], center[1], redshift,
        image.shape,
        FlatLambdaCDM(70, 0.3),
        radius_kpc = 1,
        pix_size   = 0.03
    )
    return center, mask_bulge


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Perform the optimal detection in the residuals""")
    return


@app.cell
def _(center, finder, image, mask_bulge, segmap):
    # Recover the value in the segmentation map corresponding to the galaxy
    gal_val = segmap[center[0], center[1]]

    # Setup the substructure finder
    cf = finder.ClumpFinder(
        image,
        mask = segmap == gal_val,
        mask_bg    = segmap == 0,
        mask_bulge = mask_bulge
    )
    return (cf,)


@app.cell
def _(cf, model, thr):
    # Run the substructure finder
    sub_segmap = cf.detect(model, thr, 20)
    return (sub_segmap,)


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
def _(sub_segmap):
    sub_segmap[sub_segmap < 0] = 0
    return


@app.cell
def _(image, mask_bulge, mo, model, norm, plt, segmap, show_im, sub_segmap):
    f2   = plt.figure(figsize=(8, 8))
    axs2 = []

    # For each plot, we add the bulge mask
    axs2.append(show_im(f2, 221, image,         segmap, norm = norm, title='Cutout', mask_bulge = mask_bulge, colors='w'))
    axs2.append(show_im(f2, 222, model,         segmap, norm = norm, title='Model', mask_bulge = mask_bulge, colors='w'))
    axs2.append(show_im(f2, 223, image - model, segmap, norm = norm, title='Residuals', mask_bulge = mask_bulge, colors='w'))
    axs2.append(show_im(f2, 224, sub_segmap,    segmap, cmap = 'gnuplot2', title='Substructures', mask_bulge = mask_bulge, colors='w'))

    for ax2 in axs2:
        ax2.tick_params(direction='in', right=True, top=True, labelleft=False, labelbottom=False)

    mo.vstack([f2])
    return


if __name__ == "__main__":
    app.run()

[![pages-build-deployment](https://github.com/WilfriedMercier/clump_finder/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/WilfriedMercier/clump_finder/actions/workflows/pages/pages-build-deployment)

The goal of the algorithm is to find contiguous structures of pixels in an image given that:

1. each pixel is brighther than a given threshold and
2. the contiguous structure is more extended than a given area.

This algorithm can be used to detect substructures in residual images as follows

```python
from finder import ClumpFinder

cf = ClumpFinder(
    image, mask, mask_bg, mask_bulge
)

cf.detect(model, flux_threshold, surface_threshold)
```

For more details, please refer to the documentation and check the [examples directory](https://github.com/WilfriedMercier/clump_finder/tree/main/examples). Examples are provided as [marimo notebooks](<https://marimo.io/>)but can also be run as python scripts.
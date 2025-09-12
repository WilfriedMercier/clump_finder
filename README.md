# Information about the clump pipeline package

The full clump detection pipeline is done with the following code

```python
import find_clumps

find_clumps.process_galaxies(config)
```

where `config` is the name of the configuration file. All the details about the clump detection are stored in the configuration file.

## Details about package structure

|          File      |     Description    |
|--------------------|--------------------|
| clump_utils.py | Set of utility functions and classes used for clump detection                   |
| configuration.yaml | An example of configuration file |
| cutout_utils.py | Set of utility functions used to create cutouts necessary for clump detection                   |
| detection_curves.py | Functions containing the detection curves used by the intrinsic detection method                   |
| find_clumps.py | Routines used to perform clump detection in the pipeline |
| load_routines.py | Routines used to load the configuration file and the input files |
| misc.py | Miscellaneous functions |
| visualization.py | Functions used to create recap files to visualize clump detection |

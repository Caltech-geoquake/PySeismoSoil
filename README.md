# PySeismoSoil

PySeismoSoil is a Python library for performing 1D seismic site response analysis.


## Copyright and license

Copyright (c) 2019, California Institute of Technology, based on research supported by the National Science Foundation (NSF) Cooperative Agreement EAR-1033462 and the U.S. Geological Survey (USGS) Cooperative Agreement G12AC20038. All rights reserved.

Please carefully read the license [here](https://github.com/jsh9/PySeismoSoil/blob/master/LICENSE) for the terms and conditions of using this library.


## Authors

The authors of this library are the current and past members of the [Geoquake Research Group](http://asimaki.caltech.edu/) of the California Institute of Technology: Jian Shi, Domniki Asimaki, and Wei Li.


## Installation

Install most recent stable version:

```bash
pip install git+https://github.com/jsh9/PySeismoSoil@v0.3.4
```

Or, install with latest changes (may contain features ahead of the stable version):

```bash
pip install git+https://github.com/jsh9/PySeismoSoil
```


## Dependencies

* Python: 3.6+
* matplotlib: 2.0.0+
* numpy: 1.11.0+
* scipy: 1.1.0+
* numba: 0.38.0+


## API Documentation

Stable version: https://pyseismosoil.readthedocs.io/en/stable/

(Latest features: https://pyseismosoil.readthedocs.io/en/latest/)

## Examples

Go to the "examples" folder from the root directory. Those examples help you quickly get familiar with the usage of this library.


## Knowledge base

The models and algorithms used in this library mainly come from these research papers:

1. J. Shi (2019) "Improving Site Response Analysis for Earthquake Ground Motion Modeling." Dissertation (Ph.D.), California Institute of Technology. doi:10.7907/X5NZ-DQ21. [[URL](http://resolver.caltech.edu/CaltechTHESIS:05302019-150220368)]

2. J. Shi, D. Asimaki (2018) "A Generic Velocity Profile for Basin Sediments in California Conditioned on Vs30." Seismological Research Letters, 89 (4), 1397-1409. [[URL](http://resolver.caltech.edu/CaltechAUTHORS:20180523-153705346)]

3. J. Shi, D. Asimaki (2017) "From stiffness to strength: Formulation and validation of a hybrid hyperbolic nonlinear soil model for site-response analyses." Bulletin of the Seismological Society of America, 107 (3), 1336-1355. [[URL](http://resolver.caltech.edu/CaltechAUTHORS:20170404-150827374)]

4. W. Li, D. Assimaki (2010) "Site- and motion-dependent parametric uncertainty of site-response analyses in earthquake simulations." Bulletin of the Seismological Society of America 100 (3), 954-968. [[URL](http://resolver.caltech.edu/CaltechAUTHORS:20140904-160952252)]

5. D. Asimaki, W. Li, J. Steidl, J. Schmedes (2008) "Quantifying nonlinearity susceptibility via site-response modeling uncertainty at three sites in the Los Angeles Basin." Bulletin of the Seismological Society of America 98 (5), 2364-2390. [[URL](http://resolver.caltech.edu/CaltechAUTHORS:20140828-163417572)]


## Report issues

To report bugs and submit suggestions, please use the ["Issues" section](https://github.com/jsh9/PySeismoSoil/issues) of this GitHub repository.

## How to cite this library

To cite this library, please include this DOI in your publication: [![DOI](https://zenodo.org/badge/169386936.svg)](https://zenodo.org/badge/latestdoi/169386936).

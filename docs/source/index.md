# PySeismoSoil

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [1. Functionalities](#1-functionalities)
  - [1.1. Site-response simulations](#11-site-response-simulations)
  - [1.2. Sediment velocity model (SVM)](#12-sediment-velocity-model-svm)
  - [1.3. Site amplification factors](#13-site-amplification-factors)
  - [1.4. Useful utilities for ground motion analysis](#14-useful-utilities-for-ground-motion-analysis)
- [2. Copyright and license](#2-copyright-and-license)
- [3. Authors](#3-authors)
- [4. Installation](#4-installation)
- [5. API Documentation](#5-api-documentation)
- [6. Examples](#6-examples)
- [7. Knowledge base](#7-knowledge-base)
- [8. Report issues](#8-report-issues)
- [9. How to cite this library](#9-how-to-cite-this-library)
- [10. How to contribute to this library](#10-how-to-contribute-to-this-library)

______________________________________________________________________

<!--TOC-->

[![PyPI](https://img.shields.io/pypi/v/pyseismosoil)](https://pypi.org/project/PySeismoSoil/)
[![Docs](https://github.com/Caltech-geoquake/PySeismoSoil/actions/workflows/build-and-deploy-docs.yml/badge.svg?branch=main)](https://github.com/Caltech-geoquake/PySeismoSoil/actions/workflows/build-and-deploy-docs.yml)
[![Downloads](https://static.pepy.tech/badge/pyseismosoil)](https://pepy.tech/project/pyseismosoil)
[![Downloads](https://static.pepy.tech/badge/pyseismosoil/month)](https://pepy.tech/project/pyseismosoil)
[![Downloads](https://static.pepy.tech/badge/pyseismosoil/week)](https://pepy.tech/project/pyseismosoil)

PySeismoSoil is a Python library for performing 1D seismic site response
analysis.

## 1. Functionalities

### 1.1. Site-response simulations

- **Linear simulations,** which assumes viscoelastic wave propagation through
  1D soil profile. Suitable for very weak motions.
- **Equivalent linear simulations,** which updates the shear modulus and the
  damping to the strain the motion produces (using G/G<sub>max</sub> and
  damping curves as "look up tables"), and iterates until the response and the
  soil properties converge. Suitable for moderate ground motions.
- **Nonlinear simulations,** which utilizes advanced constitutive models and
  propagates waves in the time domain. PySeismoSoil implements the hybrid
  hyperbolic (HH) model
  ([Shi & Asimaki, 2017](https://authors.library.caltech.edu/records/4naa9-fbd36)),
  which describes both the small-strain stiffness and the large-strain strength
  from a velocity profile alone, or the MKZ (modified
  [Kondner and Zelasko, 1963](https://www.scirp.org/reference/referencespapers?referenceid=2740984))
  model.

### 1.2. Sediment velocity model (SVM)

Oftentimes, only the _V<sub>S30</sub>_ of a site is known, rather than its
detailed _V<sub>S</sub>_ profile.

The sediment velocity model (SVM) by
[Shi & Asimaki (2018)](https://authors.library.caltech.edu/records/as8rk-jjb29)
turns _V<sub>S30</sub>_, and optionally the depth to _V<sub>S</sub>_=1,000 m/s
soil, into a one-dimensional profile for California basin sediments. It also
draws randomized profiles that stay consistent with that _V<sub>S30</sub>_ and
basin depth, for calculations that need a suite of profiles rather than one.

### 1.3. Site amplification factors

A database of site amplification factors (both amplitude and phase shift),
conditioned on on _V<sub>S30</sub>_, basin depth (_z_<sub>1</sub>), and PGA of
the input ground motion. They were pre-calculated using nonlinear simulation
with the HH (hybrid hyperbolic) constitutive model by
[Shi & Asimaki (2017)](https://authors.library.caltech.edu/records/4naa9-fbd36).

PySeismoSoil offers functions to apply these amplification factors to input
motions.

### 1.4. Useful utilities for ground motion analysis

- **Ground motions.** Read an accelerogram, baseline-correct it, filter it, and
  compute its response spectrum.
- **Fourier spectra.** Work with the amplitude and phase of a motion, including
  smoothing and the fundamental frequency.
- **_V<sub>S</sub>_ profiles.** Plot a profile, query _V<sub>S</sub>_ at a
  depth, and compute _z_<sub>1</sub> and the linear transfer function.
- **Modulus and damping curves.** Store G/G<sub>max</sub> and damping versus
  strain, including curves calibrated from a _V<sub>S</sub>_ profile for the HH
  model.
- **Goodness-of-fit scores.** Compare a simulated motion with a recording.

## 2. Copyright and license

Copyright (c) 2024, California Institute of Technology, based on research
supported by the National Science Foundation (NSF) Cooperative Agreement
EAR-1033462 and the U.S. Geological Survey (USGS) Cooperative Agreement
G12AC20038. All rights reserved.

Please carefully read the license
[here](https://github.com/Caltech-geoquake/PySeismoSoil/blob/main/LICENSE) for
the terms and conditions of using this library.

## 3. Authors

The authors of this library are the current and past members of the
[Geoquake Research Group](http://asimaki.caltech.edu/) of the
[California Institute of Technology](https://www.caltech.edu/): Jian Shi,
Domniki Asimaki, Wei Li, and Flora Xia.

## 4. Installation

Install most recent stable version:

```bash
pip install PySeismoSoil
```

If you already have an older version installed and want to upgrade to the
newest version, use `pip install --upgrade PySeismoSoil`.

## 5. API Documentation

The HTML documentation is published at
<https://caltech-geoquake.github.io/PySeismoSoil/>.

- [Utility classes](https://caltech-geoquake.github.io/PySeismoSoil/classes.html)
- [Helper modules](https://caltech-geoquake.github.io/PySeismoSoil/modules.html)

## 6. Examples

The notebooks in
[`examples/`](https://github.com/Caltech-geoquake/PySeismoSoil/tree/main/examples)
walk through each piece of the library: ground motions, Vs profiles, curves,
the HH calibration, linear, equivalent-linear, and nonlinear runs, batch
simulations, the sediment velocity model, site factors, site-effect adjustment,
and goodness-of-fit scores.

## 7. Knowledge base

The models and algorithms used in this library mainly come from these research
papers:

1. J. Shi (2019) "Improving Site Response Analysis for Earthquake Ground Motion
   Modeling." _Dissertation (Ph.D.), California Institute of Technology_.
   doi:10.7907/X5NZ-DQ21. \[[URL](https://thesis.caltech.edu/11571/)\]

2. J. Shi, D. Asimaki (2018) "A Generic Velocity Profile for Basin Sediments in
   California Conditioned on Vs30." _Seismological Research Letters_, 89 (4),
   1397-1409.
   \[[URL](https://authors.library.caltech.edu/records/as8rk-jjb29)\]

3. J. Shi, D. Asimaki (2017) "From stiffness to strength: Formulation and
   validation of a hybrid hyperbolic nonlinear soil model for site-response
   analyses." _Bulletin of the Seismological Society of America_, 107 (3),
   1336-1355.
   \[[URL](https://authors.library.caltech.edu/records/4naa9-fbd36)\]

4. W. Li, D. Assimaki (2010) "Site- and motion-dependent parametric uncertainty
   of site-response analyses in earthquake simulations." _Bulletin of the
   Seismological Society of America_, 100 (3), 954-968.
   \[[URL](https://authors.library.caltech.edu/records/fet2d-ecs26)\]

5. D. Asimaki, W. Li, J. Steidl, J. Schmedes (2008) "Quantifying nonlinearity
   susceptibility via site-response modeling uncertainty at three sites in the
   Los Angeles Basin." _Bulletin of the Seismological Society of America_, 98
   (5), 2364-2390.
   \[[URL](https://authors.library.caltech.edu/records/eftgh-f2s74)\]

## 8. Report issues

To report bugs and submit suggestions, please use the
["Issues" section](https://github.com/jsh9/PySeismoSoil/issues) of this GitHub
repository.

## 9. How to cite this library

To cite this library, please include this DOI in your publication:
[![DOI](https://zenodo.org/badge/169386936.svg)](https://zenodo.org/badge/latestdoi/169386936).

## 10. How to contribute to this library

Please read the
[contributing instructions](https://caltech-geoquake.github.io/PySeismoSoil/blob/master/CONTRIBUTING.md)
to get started.

```{toctree}
:hidden:
:maxdepth: 2

classes
modules
```

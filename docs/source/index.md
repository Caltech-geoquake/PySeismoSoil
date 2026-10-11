# PySeismoSoil

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [1. What PySeismoSoil does](#1-what-pyseismosoil-does)
  - [1.1. Site-response simulations](#11-site-response-simulations)
  - [1.2. Sediment velocity model](#12-sediment-velocity-model)
  - [1.3. Site amplification factors](#13-site-amplification-factors)
  - [1.4. Profiles, motions, and goodness of fit](#14-profiles-motions-and-goodness-of-fit)
- [2. Copyright and license](#2-copyright-and-license)
- [3. Authors](#3-authors)
- [4. Installation](#4-installation)
- [5. Supported Python versions](#5-supported-python-versions)
- [6. API Documentation](#6-api-documentation)
- [7. Examples](#7-examples)
- [8. Knowledge base](#8-knowledge-base)
- [9. Report issues](#9-report-issues)
- [10. How to cite this library](#10-how-to-cite-this-library)
- [11. How to contribute to this library](#11-how-to-contribute-to-this-library)

______________________________________________________________________

<!--TOC-->

[![PyPI](https://img.shields.io/pypi/v/pyseismosoil)](https://pypi.org/project/PySeismoSoil/)
[![Docs](https://github.com/Caltech-geoquake/PySeismoSoil/actions/workflows/build-and-deploy-docs.yml/badge.svg?branch=main)](https://github.com/Caltech-geoquake/PySeismoSoil/actions/workflows/build-and-deploy-docs.yml)
[![Downloads](https://static.pepy.tech/badge/pyseismosoil)](https://pepy.tech/project/pyseismosoil)
[![Downloads](https://static.pepy.tech/badge/pyseismosoil/month)](https://pepy.tech/project/pyseismosoil)
[![Downloads](https://static.pepy.tech/badge/pyseismosoil/week)](https://pepy.tech/project/pyseismosoil)

PySeismoSoil propagates earthquake shaking through a layered soil column. Give
it a shear-wave velocity profile and an input ground motion, and it returns the
motion the soil delivers: the change in amplitude, frequency content, and
timing that near-surface sediments impose on the shaking.

## 1. What PySeismoSoil does

A site-response study asks how a soil column filters an incoming wave. The
library covers that calculation at three levels of soil behavior, and the steps
that usually sit around it: building a profile when the only site parameter you
have is Vs30, applying amplification factors when a full wave-propagation run
is more than the calculation needs, and scoring a simulation against a
recording.

### 1.1. Site-response simulations

`Linear_Simulation`, `Equiv_Linear_Simulation`, and `Nonlinear_Simulation` take
the same two inputs, a `Vs_Profile` and a `Ground_Motion`, and return the same
kind of result. The base of the column can be elastic (a rock outcrop motion,
waves free to leave the profile) or rigid (a borehole motion, the downgoing
wave reflected back into the soil).

- **Linear.** Viscoelastic wave propagation through the profile. This is the
  model to use when strains stay small. It produces a transfer function and, on
  request, the motion in every layer, and it can deconvolve a surface recording
  down to the base of the profile.
- **Equivalent linear.** Updates the shear modulus and the damping to the
  strain the motion produces, using G/Gmax and damping curves, and iterates
  until the response and the soil properties agree. This is the practical model
  for moderate shaking.
- **Nonlinear.** Integrates the stress-strain response in the time domain. The
  soil model is the hybrid hyperbolic (HH) formulation of
  [Shi & Asimaki (2017)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=o5_Mbh8AAAAJ&citation_for_view=o5_Mbh8AAAAJ:NtGNdKbuCngC),
  which describes both the small-strain stiffness and the large-strain strength
  from a velocity profile alone, or the MKZ model. `HH_Calibration` builds the
  HH parameters from a `Vs_Profile`.

`Batch_Simulation` runs a set of these analyses together.

### 1.2. Sediment velocity model

A measured velocity profile is often the input you do not have. The sediment
velocity model (SVM) of
[Shi & Asimaki (2018)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=o5_Mbh8AAAAJ&citation_for_view=o5_Mbh8AAAAJ:AFXcoJnoRH0C)
turns Vs30, and optionally the depth to 1,000 m/s material, into a
one-dimensional profile for California basin sediments. It also draws
randomized profiles that stay consistent with that Vs30 and basin depth, for
calculations that need a suite of profiles rather than one.

### 1.3. Site amplification factors

`Site_Factors` returns frequency-dependent amplification and phase that already
include nonlinear soil response. The factors are conditioned on Vs30, basin
depth (z1), and the peak acceleration of the input motion. They were proposed
by Shi, Asimaki, and Graves (2019); the development is in
[Shi (2019)](https://scholar.google.com/scholar?hl=en&q=Shi+2019+%22Improving+Site+Response+Analysis+for+Earthquake+Ground+Motion+Modeling%22).
`Site_Effect_Adjustment` applies a chosen factor to a rock-outcrop motion,
which is how a simulated rock time series picks up site response without a new
wave-propagation run.

### 1.4. Profiles, motions, and goodness of fit

The objects the simulations use are useful on their own. The library reads and
plots ground motions and Vs profiles, stores G/Gmax and damping curves, and
compares a simulation with a recording through the goodness-of-fit scores
described in
[Shi & Asimaki (2017)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=o5_Mbh8AAAAJ&citation_for_view=o5_Mbh8AAAAJ:NtGNdKbuCngC).

## 2. Copyright and license

Copyright (c) 2024, California Institute of Technology, based on research
supported by the National Science Foundation (NSF) Cooperative Agreement
EAR-1033462 and the U.S. Geological Survey (USGS) Cooperative Agreement
G12AC20038. All rights reserved.

Please carefully read the license
[here](https://caltech-geoquake.github.io/PySeismoSoil/blob/master/LICENSE) for
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

## 5. Supported Python versions

PySeismoSoil currently support these Python versions:

- 3.11
- 3.12
- 3.13

## 6. API Documentation

The HTML documentation is published at
<https://caltech-geoquake.github.io/PySeismoSoil/>.

- [Utility classes](https://caltech-geoquake.github.io/PySeismoSoil/classes.html)
- [Helper modules](https://caltech-geoquake.github.io/PySeismoSoil/modules.html)

## 7. Examples

The notebooks in
[`examples/`](https://github.com/Caltech-geoquake/PySeismoSoil/tree/main/examples)
walk through each piece of the library: ground motions, Vs profiles, curves,
the HH calibration, linear, equivalent-linear, and nonlinear runs, batch
simulations, the sediment velocity model, site factors, site-effect adjustment,
and goodness-of-fit scores.

## 8. Knowledge base

The models and algorithms used in this library mainly come from these research
papers:

1. J. Shi (2019) "Improving Site Response Analysis for Earthquake Ground Motion
   Modeling." _Dissertation (Ph.D.), California Institute of Technology_.
   doi:10.7907/X5NZ-DQ21.
   \[[URL](https://scholar.google.com/scholar?hl=en&q=Shi+2019+%22Improving+Site+Response+Analysis+for+Earthquake+Ground+Motion+Modeling%22)\]

2. J. Shi, D. Asimaki (2018) "A Generic Velocity Profile for Basin Sediments in
   California Conditioned on Vs30." _Seismological Research Letters_, 89 (4),
   1397-1409.
   \[[URL](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=o5_Mbh8AAAAJ&citation_for_view=o5_Mbh8AAAAJ:AFXcoJnoRH0C)\]

3. J. Shi, D. Asimaki (2017) "From stiffness to strength: Formulation and
   validation of a hybrid hyperbolic nonlinear soil model for site-response
   analyses." _Bulletin of the Seismological Society of America_, 107 (3),
   1336-1355.
   \[[URL](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=o5_Mbh8AAAAJ&citation_for_view=o5_Mbh8AAAAJ:NtGNdKbuCngC)\]

4. W. Li, D. Assimaki (2010) "Site- and motion-dependent parametric uncertainty
   of site-response analyses in earthquake simulations." _Bulletin of the
   Seismological Society of America_, 100 (3), 954-968.
   \[[URL](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=o5_Mbh8AAAAJ&citation_for_view=o5_Mbh8AAAAJ:_FxGoFyzp5QC)\]

5. D. Asimaki, W. Li, J. Steidl, J. Schmedes (2008) "Quantifying nonlinearity
   susceptibility via site-response modeling uncertainty at three sites in the
   Los Angeles Basin." _Bulletin of the Seismological Society of America_, 98
   (5), 2364-2390.
   \[[URL](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=o5_Mbh8AAAAJ&citation_for_view=o5_Mbh8AAAAJ:W7OEmFMy1HYC)\]

## 9. Report issues

To report bugs and submit suggestions, please use the
["Issues" section](https://github.com/jsh9/PySeismoSoil/issues) of this GitHub
repository.

## 10. How to cite this library

To cite this library, please include this DOI in your publication:
[![DOI](https://zenodo.org/badge/169386936.svg)](https://zenodo.org/badge/latestdoi/169386936).

## 11. How to contribute to this library

Please read the
[contributing instructions](https://caltech-geoquake.github.io/PySeismoSoil/blob/master/CONTRIBUTING.md)
to get started.

```{toctree}
:hidden:
:maxdepth: 2

classes
modules
```

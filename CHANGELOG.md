# Change Log

## [Unreleased]

- Changed
  - Modernized the CI pipeline: replaced `isort`, `cercis`, and `flake8` with
    `muff-format`, `pydoclint`, and an updated set of pre-commit hooks (same as
    in [bootstrap2](https://github.com/jsh9/bootstrap2))
  - Auto-formatted code and docstrings with the new formatters
  - Minimum numpy version is now 2.4.0
- Removed
  - Python 3.10 support (minimum version is now 3.11)
- Fixed
  - `Vs_Profile.query_Vs_at_depth()` raising `TypeError` for a scalar depth
    with numpy 2.4+
  - `d_89()` and `d_10()` in `helper_gof_scores.py` raising `TypeError` when
    `fmin` or `fmax` is `None`
  - `GOF_Scores.calc_scores()` failing with numpy 2.4+ when computing the Arias
    intensity / energy integral scores (d1-d4) or the spectral scores (d8-d9)
    (#38)
  - `sine_smooth()` corrupting the first and last bins of the smoothed spectrum
    (the window was folded back in about the wrong points at both ends). This
    skewed the Fourier spectra score (d9), whose default frequency range
    reaches the Nyquist frequency
- Added
  - Tests for the goodness-of-fit scores and for `sine_smooth()`
  - A `notebooks` tox env and CI job that execute every example notebook on
    every PR (#40)
  - A first cell in every example notebook that prints when the notebook was
    last run (in Pacific Time)
  - A pre-commit hook, `check-notebooks-were-run`, that checks that every
    example notebook was re-run (in order, starting from cell 1) in the current
    branch
  - A script to re-run all example notebooks and save their outputs
    (`scripts/run_notebooks.py`), and a `run-notebooks` tox env that runs it
- Maintenance
  - Re-ran all example notebooks with the current code and saved their outputs
    (#40)

## [0.6.3] - 2025-10-15

- Added
  - Python 3.12 and 3.13 support
  - Development requirements in requirements.dev
- Changed
  - Updated docstrings throughout the codebase
  - Auto-formatted code for consistency
  - Migrated from setup.cfg to pyproject.toml
  - Removed 760m/s boundary on mu estimation formula when generating G/Gmax
    curve parameters for the hybrid hyperbolic model
- Removed
  - Python 3.8 support (minimum version is now 3.9)
- Fixed
  - Python 3.12 pipeline issues
  - GitHub Pages deployment workflow permissions by adding environment
    configuration

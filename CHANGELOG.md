# Change Log

## [Unreleased]

- Changed
  - `save_txt()` of `HH_Param_Multi_Layer` and `MKZ_Param_Multi_Layer` now
    writes 6 significant digits by default (was 5), the same as the files that
    `get_all_HH_x_params()` and `get_all_H4_x_params()` save
  - `helper_mkz_model.deserialize_array_to_params()` now raises a `ValueError`
    with a message when the array does not have 4 values, instead of a bare
    `AssertionError` (which `python -O` skips)
  - `MKZ_Param_Multi_Layer.serialize_to_2D_array()` now returns the layout of
    the H4 parameter files that the class reads (rows gamma_ref, 0, s, beta),
    instead of the rows gamma_ref, s, beta, Gmax. Code that reads rows of its
    result by position needs to change. Like `save_txt()`, it warns when it
    drops a Gmax other than 1.0 (#79)
- Fixed
  - `Damping_Calibration.get_H4_x_param()` returning HH parameters
    (`HH_Param_Multi_Layer`) instead of H4 (MKZ) parameters
    (`MKZ_Param_Multi_Layer`) (#60)
  - `MKZ_Param_Multi_Layer` reading H4 parameter files (and 2D arrays) in the
    wrong row order, which made `s` 0 and the G/Gmax and damping curves flat.
    Its `save_txt()` and `get_all_H4_x_params(save_txt=True)` now also write
    the file format (gamma_ref, 0, s, beta). This format has no Gmax, so
    `save_txt()` warns (once, naming the layers) when it drops a Gmax other
    than 1.0. A file or 2D array that is not valid H4 parameters now raises a
    `ValueError` that names the layer: its 2nd row ("b") is not 0, it has NaN
    (for example, from a trailing delimiter), or it does not have 4 rows. H4
    files written by 0.7.1 or earlier have the rows gamma_ref, s, beta, Gmax,
    so they raise this error. To convert one, delete its 4th row and insert a
    row of zeros as its 2nd row, e.g.,
    `np.insert(np.genfromtxt(old_file, ndmin=2)[:3], 1, 0, axis=0)` (#79)
  - `HH_Param_Multi_Layer` and `MKZ_Param_Multi_Layer` failing to load a
    one-layer parameter file, including one written by `save_txt()` (#86)
  - `str()` of a `GOF_Scores` object raising `ZeroDivisionError` when no score
    has been calculated (e.g., before `calc_scores()` is run). It now shows the
    average score as `nan` (#65)
  - `helper_gof_scores.S_()` changing the caller's arrays (it clipped the
    values below machine epsilon in place) (#45)
  - `sine_smooth()` raising `IndexError` for spectra with more than about 4490
    frequencies whose spacing is coarser than `window_span / 1.854305` (about
    0.16 Hz at the default `window_span=0.3`). It now uses a convolution
    instead of Python loops, which is also much faster on long spectra (such as
    those that `GOF_Scores.calc_scores()` smooths), with the same results to
    within rounding (#44)

## [0.7.1] - 2026-10-11

- Added
  - An introduction in the README to the main capabilities (linear,
    equivalent-linear, and nonlinear site response, the sediment velocity
    model, site amplification factors, and goodness-of-fit scores)
  - A pre-commit hook, `copy_readme`, that copies `README.md` into the
    documentation homepage (`docs/source/index.md`) so the published docs stay
    in step with the README
- Changed
  - `PySeismoSoil.__version__` is read from the installed package metadata
    instead of a hard-coded string
  - `muff-check` and `muff-format` now run only as pre-commit hooks (removed
    the duplicate `tox` envs, which CI ran separately). CI now also
    format-checks `scripts/`, `docs/`, and the example notebooks. When a hook
    modifies a file in CI, the log shows the diff, and the lint rules that
    `muff-check` fixed
  - The formatters (`muff-format`, `format-docstring`, and the other formatting
    hooks) now also run on `tests/`, and the test files are formatted. Only the
    data files in `tests/files/` are still excluded from all hooks.
  - `muff-check` now also lints `tests/`, and the lint violations there are
    fixed. Tests use plain `assert` statements and `pytest.raises()` instead of
    the `unittest`-style assertion methods, and read the data files via
    `pathlib`
  - The tests are now plain `pytest` functions instead of methods of
    `unittest.TestCase` classes. Shared setup is now in `pytest` fixtures or
    module-level constants
  - Test functions that differ only in their inputs and expected results are
    now combined into parametrized tests (`pytest.mark.parametrize`). Each case
    still runs (and is reported) as a separate test
- Full diff
  - https://github.com/Caltech-geoquake/PySeismoSoil/compare/0.7.0...0.7.1

## [0.7.0] - 2026-10-06

- Changed
  - Modernized the CI pipeline: replaced `isort`, `cercis`, and `flake8` with
    `muff-format`, `pydoclint`, and an updated set of pre-commit hooks (same as
    in [bootstrap2](https://github.com/jsh9/bootstrap2))
  - Auto-formatted code and docstrings with the new formatters
  - Added the `muff-check` linter (with auto-fix) to pre-commit, `tox`, and CI
    (`tests/` is excluded), and fixed the lint violations in the library, the
    example notebooks, `scripts/`, and `docs/`
  - Boolean arguments (e.g., `show_fig`, `verbose`, `parallel`) of functions
    and methods are now keyword-only, so they must be passed by name (this also
    makes the arguments that follow them keyword-only)
  - `Batch_Simulation.run()` writes to `batch_sim_<time>` (instead of
    `./batch_sim_<time>`) by default, which is the same location
  - `get_current_time()` returns a timezone-aware local time (same format)
  - Minimum numpy version is now 2.4.0
  - `SVM.get_randomized_profile()` no longer calls `np.random.seed()`, so it
    leaves numpy's global random state alone, with or without a `seed`. Code
    that relied on it to seed later `np.random` calls needs to call
    `np.random.seed()` itself
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
  - `SVM.get_randomized_profile()` with `seed=None` using the current time in
    seconds as the seed. It could only give 60 different profiles, and the
    search for a compliant profile (`vs30_z1_compliance=True`) repeated the
    same profile until the next second. A given `seed` still gives the same
    profile as before
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
  - Made `test_get_randomized_profile` use a fixed seed, so that it no longer
    fails randomly (#48)
- Full diff
  - https://github.com/Caltech-geoquake/PySeismoSoil/compare/0.6.3...0.7.0

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

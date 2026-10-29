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
  - `raw_data` of `Curve`, `GGmax_Curve`, `Damping_Curve` and `Stress_Curve` is
    now a copy of the input array (not the array itself), in the same units as
    the other attributes: strain and damping in %, and stress in kPa. With
    `interpolate=True`, it used to keep the units of the input (e.g., strain in
    unit 1 for `strain_unit='1'`), although `plot(plot_interpolated=False)`
    plots it on a "Strain [%]" axis and the damping fits take it in %. The
    other attributes (`strain`, `values`, `GGmax`, `damping`, `stress`) are no
    longer views of `raw_data` (they were with `interpolate=False`).
    `raw_data`, `strain` and `values` are now float arrays, even for integer
    input (#85)
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
  - `Curve`, `GGmax_Curve`, `Damping_Curve` and `Stress_Curve` converting units
    inside the caller's array: `strain_unit='1'` (the default of
    `Stress_Curve`), `damping_unit='1'`, and a `stress_unit` other than 'kPa'.
    `Stress_Curve` also no longer raises a `TypeError` for an integer array
    with a `stress_unit` other than 'kPa' (#85)
  - `Damping_Curve.get_HH_x_param()`/`get_H4_x_param()` and
    `Multiple_Damping_Curves.get_all_HH_x_params()`/`get_all_H4_x_params()`
    fitting strain (or damping) values 100 times too small for curves created
    with `interpolate=True` and `strain_unit='1'` (or `damping_unit='1'`) (#85)
  - `Curve`, `GGmax_Curve` and `Damping_Curve` with `interpolate=True` and
    `strain_unit='1'` reading `min_strain` and `max_strain` in unit 1 instead
    of %, so the default range (1e-4 to 10) gave a strain array from 0.01% to
    1000%, and the curve was cut off at small strains and flat at large
    strains. The data is now converted to % before it is interpolated (related
    to #85)
  - `helper_hh_model.fit_HH_x_single_layer()` and
    `helper_mkz_model.fit_H4_x_single_layer()` subtracting the initial damping
    inside the caller's array. Through `Damping_Curve.get_HH_x_param()`/
    `get_H4_x_param()` and `Multiple_Damping_Curves.get_all_HH_x_params()`/
    `get_all_H4_x_params()` (with `parallel=False`), this shifted the curves
    down by their initial damping: `raw_data` (the caller's array) and, for
    curves that are not interpolated, also `damping` and `get_curve_matrix()`.
    So `show_fig=True` of `get_all_HH_x_params()`/`get_all_H4_x_params()` also
    plotted the shifted data, unlike with `parallel=True`. The figure of
    `fit_HH_x_single_layer(show_fig=True)` and
    `fit_H4_x_single_layer(show_fig=True)` (and so of
    `Damping_Curve.get_HH_x_param()`/`get_H4_x_param()`) now also shows the
    damping data as given (it showed the data shifted to start at 0), like
    `get_all_HH_x_params()`/`get_all_H4_x_params()` (#84)

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

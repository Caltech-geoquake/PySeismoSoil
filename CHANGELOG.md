# Change Log

## [Unreleased]

- Changed
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

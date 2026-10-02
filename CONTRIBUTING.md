# Contributing Instructions

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [1. General guideline](#1-general-guideline)
- [2. Install the library for development](#2-install-the-library-for-development)
- [3. Running local tests and linting](#3-running-local-tests-and-linting)
- [4. Re-run the example notebooks in every PR](#4-re-run-the-example-notebooks-in-every-pr)
- [5. Update the documentations](#5-update-the-documentations)

______________________________________________________________________

<!--TOC-->

## 1. General guideline

In general, contributors should make code changes on a branch, and then
[create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
to have the changes merged.

## 2. Install the library for development

- It is strongly recommended that contributors work on code changes in an
  isolated Python environment
- Use `pip install -e .` to install this library locally, so that any local
  code changes are reflected immediately in your current Python environment

## 3. Running local tests and linting

You can run tests with the `tox` command.

And you can run auto-formatting with the command `pre-commit run -a`. The
pre-commit hooks include `muff-format` (the code formatter, configured in
`muff.toml`) and `format-docstring`.

Docstrings use the NumPy style and are checked with `pydoclint`
(`tox -e pydoclint`). To check formatting without modifying any files, run
`tox -e muff-format`.

## 4. Re-run the example notebooks in every PR

Every PR needs to re-run all the example notebooks in `examples/` and commit
them, so that their outputs always come from the current code. To do this, run
this command from the root directory:

```
tox -e run-notebooks
```

It installs this library and the packages needed to run the notebooks in a
separate environment (managed by `tox`), runs each notebook from top to bottom
in a fresh kernel, and saves the notebooks that run without errors. This takes
a few minutes. Options after `--` are passed to the script
(`scripts/run_notebooks.py`):

- To run only some notebooks, pass their paths (e.g.,
  `tox -e run-notebooks -- examples/Demo_01_Ground_Motion.ipynb`)
- To run several notebooks at a time, use `-j` (e.g.,
  `tox -e run-notebooks -- -j 4`), but the timings that some notebooks print
  are then less accurate

If you already have a development environment with this library installed
(`pip install -e .`) and the packages in `requirements.dev`, you can also run
`python scripts/run_notebooks.py` directly.

The first cell of each notebook prints when the notebook was last run (in
Pacific Time). The pre-commit hook `check-notebooks-were-run` fails if:

- this time is not later than the commit on `main` that your branch started
  from (or the latest commit on `main` that you merged into your branch), or
- the code cells were not run in order starting from 1, any cell was skipped,
  or any cell has an error output.

Locally, the hook only checks the notebooks in each commit. CI checks all the
notebooks. If you run a notebook in Jupyter instead, use "Restart Kernel and
Run All Cells". New notebooks need the same first cell (copy it from an
existing notebook).

Separately, `tox -e notebooks` checks that every notebook runs, without
modifying them. CI runs it on every PR.

## 5. Update the documentations

If you would like to make changes to the documentations of this library, you
need to install the dependencies for building documentations with the following
command (from the root directory):

```
pip install -r docs/requirements.txt
```

To build the documentation HTML pages locally, navigate to the `docs` folder,
and run `make clean html`. To view the generated HTML documentation, open the
file `docs/build/html/index.html` in the browser.

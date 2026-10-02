# Contributing Instructions

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [1. General guideline](#1-general-guideline)
- [2. Install the library for development](#2-install-the-library-for-development)
- [3. Running local tests and linting](#3-running-local-tests-and-linting)
- [4. Update the documentations](#4-update-the-documentations)

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

To check that every example notebook in `examples/` still runs, use
`tox -e notebooks` (this takes a few minutes; CI runs it on every PR). It does
not modify the notebooks. To also save the fresh outputs into the notebooks,
run `tox -e notebooks -- --overwrite`, and then `pre-commit run -a`.

## 4. Update the documentations

If you would like to make changes to the documentations of this library, you
need to install the dependencies for building documentations with the following
command (from the root directory):

```
pip install -r docs/requirements.txt
```

To build the documentation HTML pages locally, navigate to the `docs` folder,
and run `make clean html`. To view the generated HTML documentation, open the
file `docs/build/html/index.html` in the browser.

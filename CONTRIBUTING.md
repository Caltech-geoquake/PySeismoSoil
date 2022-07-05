# Contributing Instructions

## 1. General guideline

In general, contributors should make code changes on a branch, and then [create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) to have the changes merged.

## 2. Install the library for development

- It is strongly recommended that contributors work on code changes in an isolated Python environment
- Use `pip install -e .` to install this library locally, so that any local code changes are reflected immediately in your current Python environment

## 3. Running local tests and linting

Make sure to run the following checks on your local computer, before pushing any code to GitHub:
- Code style: run `./run_linting.sh`
    + You might want to run `chmod +x run_linting.sh` to make `run_linting.sh` executable
- Unit tests: run `./run_tests.sh`
    + You might want to run `chmod +x run_tests.sh` to make `run_tests.sh` executable

(Even if you forget to run the checking above on your local computer, unit tests and code styles are checked on every push at GitHub.)

## 4. Update the documentations

If you would like to make changes to the documentations of this library, you need to install the dependencies for building documentations with the following command (from the root directory):

```
pip install -r docs/requirements.txt
```

To build the documentation HTML pages locally, navigate to the `docs` folder, and run `make clean html`.  To view the generated HTML documentation, open the file `docs/build/html/index.html` in the browser.

# Contributing Instructions

## 1. General guideline

In general, contributors should make code changes on a branch, and then [create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) to have the changes merged.

## 2. Install the library for development

- It is strongly recommended that contributors work on code changes in an isolated Python environment
- Use `pip install -e .[dev]` to install this library locally, so that any local code changes are reflected immediately in your current Python environment
    + Explanation of `.[dev]`:
        - `.` means to run the `setup.py` script in the root folder (`.`)
        - `[dev]` means to also install the extra dependencies that are only needed for developers (see `requirements.dev` for what they are)
            + Note: if you are using zsh (Z shell), such as on macOS, you need to except the square brackets: `pip install -e .\[dev\]`

## 3. Before pushing code to GitHub

Make sure to run the following checks on your local computer, before pushing any code to GitHub:
- Code style: run `./run_linting.sh`
    + You might want to run `chmod +x run_linting.sh` to make `run_linting.sh` executable
- Unit tests: run `./run_tests.sh`
    + You might want to run `chmod +x run_tests.sh` to make `run_tests.sh` executable

(Even if you forget to run the checking above on your local computer, unit tests and code styles are checked on every push at GitHub.)

## 4.

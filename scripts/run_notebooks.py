"""
Re-run the example notebooks and save their outputs.

Usage (from the root directory of this repository)::

    tox -e run-notebooks  # all notebooks in examples/
    tox -e run-notebooks -- -j 4  # run 4 notebooks at a time
    tox -e run-notebooks -- examples/Demo_01_Ground_Motion.ipynb

Each notebook runs from top to bottom in a fresh kernel, from the folder that
it is in. A notebook is saved only if all its cells run without errors.

The ``run-notebooks`` tox env sets up the environment that this script needs.
To run the script directly instead (``python scripts/run_notebooks.py``), it
needs the packages in ``requirements.dev``, and PySeismoSoil installed in the
current Python environment (``pip install -e .``).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / 'examples'


def run_notebook(path: Path, timeout: int) -> str | None:
    """
    Run one notebook, and save it if all its cells run without errors.

    Parameters
    ----------
    path : Path
        The path to the notebook.
    timeout : int
        The maximum time (in seconds) that each cell is allowed to run.

    Returns
    -------
    str | None
        The error message if the notebook fails to run; otherwise None.
    """
    notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None

            # Remove cell timing info that other tools may have recorded
            cell.metadata.pop('execution', None)

    client = NotebookClient(
        notebook,
        timeout=timeout,
        record_timing=False,
        resources={'metadata': {'path': str(path.parent)}},
    )
    try:
        client.execute()
    except Exception as exc:  # noqa: BLE001
        return f'{type(exc).__name__}: {exc}'

    nbformat.write(notebook, path)
    return None


def main(argv: list[str] | None = None) -> int:
    """
    Re-run the notebooks passed on the command line (or all of them).

    Parameters
    ----------
    argv : list[str] | None, default=None
        The command-line arguments. If None, ``sys.argv[1:]`` is used.

    Returns
    -------
    int
        The exit code: 0 if all notebooks run without errors, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description='Re-run the example notebooks and save their outputs.'
    )
    parser.add_argument(
        'notebooks',
        nargs='*',
        type=Path,
        help='The notebooks to run (default: all notebooks in examples/)',
    )
    parser.add_argument(
        '-j',
        '--jobs',
        type=int,
        default=1,
        help='How many notebooks to run at the same time (default: 1)',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='The maximum time in seconds for each cell (default: 600)',
    )
    args = parser.parse_args(argv)

    notebooks = args.notebooks or sorted(EXAMPLES_DIR.glob('*.ipynb'))
    n_total = len(notebooks)
    print(f'Running {n_total} notebook(s) with {args.jobs} job(s)...')

    failed = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_notebook, path, args.timeout): path
            for path in notebooks
        }
        for k, (future, path) in enumerate(futures.items(), start=1):
            error = future.result()
            status = 'saved' if error is None else 'FAILED (not saved)'
            print(f'[{k}/{n_total}] {path.name}: {status}', flush=True)
            if error is not None:
                failed.append(path)
                print(error)

    print(f'\nFinished in {time.monotonic() - start:.0f} seconds.')
    if failed:
        print(f'{len(failed)} notebook(s) failed:')
        for path in failed:
            print(f'  - {path}')

        return 1

    print('All notebooks ran without errors. You can now commit them.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

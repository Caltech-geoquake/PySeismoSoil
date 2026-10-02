"""
Check that the example notebooks were re-run in the current branch.

This is a pre-commit hook. For each notebook passed on the command line, it
checks that:

1. The notebook was run from top to bottom in a fresh kernel: the k-th code
   cell has the execution count k (no skipped or un-run cells), and no cell
   has an error output.
2. The first code cell prints when the notebook was last run (in Pacific
   Time), and that time is later than the latest commit from ``main`` in the
   current branch (i.e., the merge base of the current branch and ``main``).
   In other words, the notebook was re-run in the current branch.

The check is skipped on ``main`` itself.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE_BRANCH = 'main'

PACIFIC_TIME = ZoneInfo('America/Los_Angeles')

LAST_RUN_PATTERN = re.compile(
    r'^Last run: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (PDT|PST)$',
    flags=re.MULTILINE,
)

PACIFIC_TIME_OFFSETS = {
    'PDT': timezone(timedelta(hours=-7)),
    'PST': timezone(timedelta(hours=-8)),
}

HOW_TO_FIX = (
    'To fix this, re-run all the notebooks with '
    '`python scripts/run_notebooks.py`, and commit them.'
)


def _git(*args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ['git', *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_base_commit() -> tuple[str, datetime] | None:
    """
    Get the latest commit from ``main`` in the current branch.

    Returns
    -------
    tuple[str, datetime] | None
        The short hash and the commit time of the merge base of ``HEAD`` and
        ``origin/main`` (or ``main``, if there is no ``origin/main``). None if
        the current branch is ``main`` itself.

    Raises
    ------
    RuntimeError
        When there is no ``main`` branch to compare with, or when the merge
        base cannot be found (e.g., in a shallow clone).
    """
    try:
        current_branch = _git('symbolic-ref', '--quiet', '--short', 'HEAD')
    except subprocess.CalledProcessError:  # detached HEAD
        current_branch = None

    if current_branch == BASE_BRANCH:
        return None

    for base_ref in (f'origin/{BASE_BRANCH}', BASE_BRANCH):
        try:
            _git('rev-parse', '--verify', '--quiet', base_ref)
            break
        except subprocess.CalledProcessError:
            continue
    else:
        raise RuntimeError(
            f'Cannot find the `origin/{BASE_BRANCH}` or `{BASE_BRANCH}`'
            ' branch to compare with.'
        )

    try:
        merge_base = _git('merge-base', 'HEAD', base_ref)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f'Cannot find the latest commit from `{base_ref}` in the current'
            ' branch. If this is a shallow clone, fetch the full history'
            ' first (e.g., `git fetch --unshallow`).'
        ) from exc

    short_hash, timestamp = _git(
        'show', '--no-patch', '--format=%h %ct', merge_base
    ).split()
    return short_hash, datetime.fromtimestamp(int(timestamp), tz=UTC)


def _get_output_text(cell: dict) -> str:
    texts = []
    for output in cell.get('outputs', []):
        text = output.get('text') or output.get('data', {}).get('text/plain')
        if text:
            texts.append(''.join(text) if isinstance(text, list) else text)

    return '\n'.join(texts)


def check_notebook(
        filename: str,
        base_commit: tuple[str, datetime],
) -> list[str]:
    """
    Check one notebook.

    Parameters
    ----------
    filename : str
        The path to the notebook.
    base_commit : tuple[str, datetime]
        The short hash and the commit time of the latest commit from ``main``
        in the current branch.

    Returns
    -------
    list[str]
        The problems found. Empty if the notebook passes the check.
    """
    with Path(filename).open(encoding='utf-8') as fp:
        notebook = json.load(fp)

    code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    if not code_cells:
        return []

    problems = []

    for k, cell in enumerate(code_cells, start=1):
        execution_count = cell.get('execution_count')
        if execution_count != k:
            found = (
                'has not been run'
                if execution_count is None
                else f'has the execution count [{execution_count}]'
            )
            problems.append(
                f'Code cell #{k} {found} (expected [{k}]). The notebook needs'
                ' to be run from top to bottom in a fresh kernel.'
            )
            break

    for k, cell in enumerate(code_cells, start=1):
        if any(o['output_type'] == 'error' for o in cell.get('outputs', [])):
            problems.append(f'Code cell #{k} has an error output.')
            break

    match = LAST_RUN_PATTERN.search(_get_output_text(code_cells[0]))
    if match is None:
        problems.append(
            'The first code cell does not print when the notebook was last'
            ' run (e.g., "Last run: 2026-10-02 09:30:00 PDT").'
        )
    else:
        time_str, time_zone = match.groups()
        last_run = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').replace(
            tzinfo=PACIFIC_TIME_OFFSETS[time_zone]
        )
        base_hash, base_time = base_commit
        if last_run <= base_time:
            base_time_pacific = base_time.astimezone(PACIFIC_TIME)
            problems.append(
                f'It was last run at {time_str} {time_zone}, which is not'
                f' later than the latest commit from `{BASE_BRANCH}` in this'
                f' branch (commit {base_hash}, made at'
                f' {base_time_pacific:%Y-%m-%d %H:%M:%S %Z}).'
            )

    return problems


def main(argv: list[str] | None = None) -> int:
    """
    Run the check on the notebooks passed on the command line.

    Parameters
    ----------
    argv : list[str] | None, default=None
        The command-line arguments. If None, ``sys.argv[1:]`` is used.

    Returns
    -------
    int
        The exit code: 0 if all notebooks pass, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description='Check that the notebooks were re-run in this branch.'
    )
    parser.add_argument('filenames', nargs='*', help='The notebooks to check')
    args = parser.parse_args(argv)

    try:
        base_commit = get_base_commit()
    except RuntimeError as exc:
        print(f'Error: {exc}')
        return 1

    if base_commit is None:  # on `main` itself
        return 0

    exit_code = 0
    for filename in args.filenames:
        problems = check_notebook(filename, base_commit)
        if problems:
            exit_code = 1
            print(f'{filename}:')
            for problem in problems:
                print(f'  - {problem}')

    if exit_code:
        print(f'\n{HOW_TO_FIX}')

    return exit_code


if __name__ == '__main__':
    sys.exit(main())

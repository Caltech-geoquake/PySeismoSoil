"""Reading, checking, and writing the data files that the GUI works with."""

from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np

# Unit of acceleration --> the number that converts m/s/s to that unit (i.e.,
# the "division factor" that converts that unit to m/s/s), as in MATLAB
# SeismoSoil
G_ACCEL = 9.81
ACCEL_UNITS: dict[str, float] = {'m/s^2': 1.0, 'gal': 100.0, 'g': 1 / G_ACCEL}

# Number of columns or rows of the data files
NUM_PROFILE_COLUMNS = 5
NUM_MOTION_COLUMNS = 2
NUM_THICKNESS_VS_COLUMNS = 2
NUM_CURVE_COLUMNS_PER_MATERIAL = 4
NUM_MKZ_PARAM_ROWS = 4
NUM_HH_PARAM_ROWS = 9

# Precision (significant digits) of the output files, as in MATLAB SeismoSoil
PRECISION = 6
PRECISION_MOTION = 7

# SMC files (https://escweb.wr.usgs.gov/nsmp-data/smcfmt.html)
_SMC_LINE_NPTS = 13  # 0-indexed line number of the number of points
_SMC_LINE_SAMPLING_RATE = 17  # 0-indexed line number of the sampling rate
_SMC_FIRST_COMMENT_LINE = 27
_SMC_FIELD_WIDTH = 10
_PEER_NUM_HEADER_LINES = 4
_PEER_DT_POSITION = 1  # the 2nd number on the line "NPTS, DT" line

DataKind = Literal[
    'vs_profile',
    'vs_profile_2_col',
    'curve',
    'motion',
    'mkz_param',
    'hh_param',
    'tau_max',
]


class InputError(ValueError):
    """An input file or value is invalid."""


def load_matrix(path: str | pathlib.Path) -> np.ndarray:
    """
    Load a numeric data file (like MATLAB's ``importdata()``).

    The numbers can be separated by spaces, tabs, or commas. Lines that are not
    numeric (e.g., header lines) are skipped.

    Parameters
    ----------
    path : str | pathlib.Path
        Path of the file.

    Returns
    -------
    np.ndarray
        A 2D array. Rows that are shorter than the others are padded with NaN.

    Raises
    ------
    InputError
        When the file does not contain any numbers
    """
    text = pathlib.Path(path).read_text(encoding='utf-8', errors='replace')
    rows: list[list[float]] = []
    for line in text.splitlines():
        tokens = line.replace(',', ' ').split()
        if not tokens:
            continue

        try:
            rows.append([float(tok) for tok in tokens])
        except ValueError:
            continue  # a header line, or a line with text

    if not rows:
        raise InputError(f'No numeric data in {pathlib.Path(path).name}.')

    n_col = max(len(row) for row in rows)
    data = np.full((len(rows), n_col), np.nan)
    for i, row in enumerate(rows):
        data[i, : len(row)] = row

    return data


def save_matrix(
        path: str | pathlib.Path,
        data: np.ndarray,
        precision: int = PRECISION,
) -> None:
    """
    Save a numeric array as a tab-delimited text file.

    Parameters
    ----------
    path : str | pathlib.Path
        Path of the file.
    data : np.ndarray
        The data (1D or 2D).
    precision : int, default=PRECISION
        Number of significant digits.
    """
    np.savetxt(
        str(path),
        np.atleast_1d(data),
        fmt=f'%.{precision}g',
        delimiter='\t',
    )


def site_code_from_filename(filename: str) -> str:
    """
    Get a site code (the default profile name) from a file name.

    For example, the site code of "profile_CE.11023.txt" is "CE.11023".

    Parameters
    ----------
    filename : str
        File name (with or without directories).

    Returns
    -------
    str
        The site code.
    """
    stem = pathlib.Path(filename).stem
    if 'profile_' in stem:
        return stem.split('profile_', 1)[1] or stem

    return stem


def unique_path(path: str | pathlib.Path) -> pathlib.Path:
    """
    Append "_" to a file name until no file has this name.

    Parameters
    ----------
    path : str | pathlib.Path
        The desired path.

    Returns
    -------
    pathlib.Path
        A path that does not exist yet.
    """
    path = pathlib.Path(path)
    while path.exists():
        path = path.with_name(f'{path.stem}_{path.suffix}')

    return path


def split_name(filename: str) -> tuple[str, str]:
    """
    Split a file name into the name without extension, and the extension.

    Parameters
    ----------
    filename : str
        File name.

    Returns
    -------
    tuple[str, str]
        The name without extension, and the extension (e.g., ".txt").
    """
    path = pathlib.Path(filename)
    return path.stem, path.suffix


def read_peer_at2(path: str | pathlib.Path) -> tuple[float, np.ndarray]:
    """
    Read a PEER (NGA) strong motion file (".AT2").

    Parameters
    ----------
    path : str | pathlib.Path
        Path of the file.

    Returns
    -------
    tuple[float, np.ndarray]
        The time step (s), and the acceleration (unit: g).

    Raises
    ------
    InputError
        When the file format is not recognized
    """
    lines = (
        pathlib
        .Path(path)
        .read_text(encoding='utf-8', errors='replace')
        .splitlines()
    )
    if len(lines) <= _PEER_NUM_HEADER_LINES:
        raise InputError(f'{pathlib.Path(path).name} is not a PEER file.')

    info = lines[_PEER_NUM_HEADER_LINES - 1].upper()
    try:
        if 'DT=' in info:
            dt_text = info.split('DT=', 1)[1].split()[0].rstrip(',')
        else:  # another format: "8200    0.0050    NPTS, DT"
            dt_text = info.replace(',', ' ').split()[_PEER_DT_POSITION]

        dt = float(dt_text)
    except (IndexError, ValueError) as err:
        raise InputError(
            f'Cannot find the time step in {pathlib.Path(path).name}.'
        ) from err

    values = []
    for line in lines[_PEER_NUM_HEADER_LINES:]:
        values.extend(_parse_floats(line))

    return dt, np.array(values)


def read_smc(path: str | pathlib.Path) -> tuple[float, np.ndarray]:
    """
    Read an SMC (USGS "strong motion CD") acceleration file.

    Parameters
    ----------
    path : str | pathlib.Path
        Path of the file.

    Returns
    -------
    tuple[float, np.ndarray]
        The time step (s), and the acceleration (unit: cm/s/s).

    Raises
    ------
    InputError
        When the file format is not recognized
    """
    name = pathlib.Path(path).name
    lines = (
        pathlib
        .Path(path)
        .read_text(encoding='utf-8', errors='replace')
        .splitlines()
    )
    try:
        npts = int(float(lines[_SMC_LINE_NPTS].split()[0]))
        sampling_rate = float(lines[_SMC_LINE_SAMPLING_RATE].split()[1])
    except (IndexError, ValueError) as err:
        raise InputError(f'{name} is not a valid SMC file.') from err

    if npts <= 0 or sampling_rate <= 0:
        raise InputError(f'{name} is not a valid SMC file.')

    i_data = _SMC_FIRST_COMMENT_LINE
    while i_data < len(lines) and lines[i_data].startswith('|'):
        i_data += 1

    values: list[float] = []
    width = _SMC_FIELD_WIDTH
    for line in lines[i_data:]:
        if line and line[0] not in ' -':  # leading space removed by an editor
            line = ' ' + line  # noqa: PLW2901

        for k in range(0, len(line.rstrip()), width):
            field = line[k : k + width].strip()
            if field:
                try:
                    values.append(float(field))
                except ValueError as err:
                    raise InputError(
                        f'Invalid number in {name}: {field}'
                    ) from err

    return 1.0 / sampling_rate, np.array(values[:npts])


def _parse_floats(line: str) -> list[float]:
    """Parse the numbers on a line, ignoring anything else."""
    out = []
    for tok in line.replace(',', ' ').split():
        try:
            out.append(float(tok))
        except ValueError:
            continue

    return out


def check_data(  # noqa: C901
        data: np.ndarray,
        kind: DataKind,
        name: str | None = None,
) -> None:
    """
    Check the format of the data from an input file.

    This is a port of ``checkInputs()`` of MATLAB SeismoSoil.

    Parameters
    ----------
    data : np.ndarray
        The data to check.
    kind : DataKind
        The kind of data.
    name : str | None, default=None
        Name of the data, to show in the error message.

    Raises
    ------
    InputError
        When the data are invalid
    """
    msg = None
    if not np.all(np.isfinite(data)):
        msg = 'Data has NaN.'
    elif kind == 'motion':
        msg = _check_motion(data)
    elif np.any(data < 0):
        msg = 'Data has negative values.'
    elif kind == 'vs_profile':
        msg = _check_vs_profile(data)
    elif kind == 'vs_profile_2_col':
        if data.shape[1] not in {
            NUM_THICKNESS_VS_COLUMNS,
            NUM_PROFILE_COLUMNS,
        }:
            msg = 'Data has incorrect number of columns. (Should be 2 or 5.)'
        elif np.any(data[:-1, 0] <= 0):
            msg = (
                'Soil thickness has non-positive numbers between first and '
                'second-to-last layers.'
            )
        elif np.any(data[:, 1] <= 0):
            msg = 'Vs has non-positive numbers.'
    elif kind == 'curve':
        if data.shape[1] % NUM_CURVE_COLUMNS_PER_MATERIAL != 0:
            msg = (
                'Data has incorrect number of columns. (Should be a multiple '
                'of 4.)'
            )
    elif kind == 'mkz_param':
        if data.shape[0] != NUM_MKZ_PARAM_ROWS:
            msg = 'Data has incorrect number of rows. (Should be 4.)'
        elif np.any(data[1, :] != 0):
            msg = 'The second row should be all zeros.'
    elif kind == 'hh_param':
        if data.shape[0] != NUM_HH_PARAM_ROWS:
            msg = 'Data has incorrect number of rows. (Should be 9.)'
    elif kind == 'tau_max' and min(data.shape) != 1:
        msg = 'Data should have only one row or one column.'

    if msg is not None:
        if name:
            msg = (
                f'{name} {msg[5:]}'
                if msg.startswith('Data ')
                else f'{name}: {msg}'
            )

        raise InputError(msg)


def _check_vs_profile(data: np.ndarray) -> str | None:
    """Check a Vs profile with 5 columns; return the error message."""
    if data.shape[1] != NUM_PROFILE_COLUMNS:
        return 'Data has incorrect number of columns. (Should be 5.)'

    if np.any(data[:-1, 0] <= 0):
        return (
            'Soil thickness has non-positive numbers between first and '
            'second-to-last layers.'
        )

    if np.any(data[:, 1:4] <= 0):
        return 'Vs, damping, or mass density has non-positive numbers.'

    if np.any(data[:-1, 4] <= 0):
        return (
            'Material number has non-positive numbers between first and '
            'second-to-last layers.'
        )

    if np.any(data[:, 2] > 1):
        return 'Unit of damping ratio wrong: should be 1, not percent.'

    return None


def _check_motion(data: np.ndarray) -> str | None:
    """Check a two-column motion; return the error message."""
    if data.shape[1] != NUM_MOTION_COLUMNS:
        return 'Data has incorrect number of columns. (Should be 2.)'

    if np.any(data[:, 0] < 0):
        return 'Data has negative time values.'

    if data.shape[0] < NUM_MOTION_COLUMNS or np.any(np.diff(data[:, 0]) <= 0):
        return 'Data has time column not monotonically increasing.'

    return None


def check_consistency(
        vs_profile: np.ndarray,
        *,
        curve: np.ndarray | None = None,
        G_param: np.ndarray | None = None,
        xi_param: np.ndarray | None = None,
        param_name: str = 'HH',
) -> None:
    """
    Check that the input files of an analysis are consistent with each other.

    This is a port of ``checkAllInputs()`` of MATLAB SeismoSoil.

    Parameters
    ----------
    vs_profile : np.ndarray
        The Vs profile.
    curve : np.ndarray | None, default=None
        The G/Gmax and damping curves.
    G_param : np.ndarray | None, default=None
        The parameters of the G/Gmax curves (or the H2 parameters).
    xi_param : np.ndarray | None, default=None
        The parameters of the damping curves.
    param_name : str, default='HH'
        Name of the parameters (e.g., "HH", "H4", or "H2"), for the messages.

    Raises
    ------
    InputError
        When the inputs are inconsistent
    """
    n_mat = int(np.max(vs_profile[:-1, 4]))
    if (
        G_param is not None
        and xi_param is not None
        and G_param.shape[1] != xi_param.shape[1]
    ):
        raise InputError(
            f'"{param_name}_G" and "{param_name}_x" dimensions do not match.'
        )

    if curve is not None:
        if curve.shape[1] < n_mat * NUM_CURVE_COLUMNS_PER_MATERIAL:
            raise InputError(
                '"Curve" file does not have enough material properties.'
            )

        if (
            G_param is not None
            and curve.shape[1]
            != G_param.shape[1] * NUM_CURVE_COLUMNS_PER_MATERIAL
        ):
            raise InputError(
                f'Number of columns in "curve" should be 4 times that of '
                f'the "{param_name}" parameters.',
            )

    for param in (G_param, xi_param):
        if param is not None and param.shape[1] < n_mat:
            raise InputError(
                f'The "{param_name}" parameters do not have enough materials.'
            )


def format_duration(seconds: float) -> str:
    """
    Format a duration (like ``sec2datetime()`` of MATLAB SeismoSoil).

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        For example, "1 min 5.2 sec".
    """
    minutes, sec = divmod(max(seconds, 0.0), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    parts = []
    if hours:
        parts.append(f'{hours} hr')

    if hours or minutes:
        parts.append(f'{minutes} min')

    parts.append(f'{sec:.1f} sec')
    return ' '.join(parts)

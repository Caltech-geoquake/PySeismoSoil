import pathlib

import numpy as np
import pytest

from PySeismoSoil.gui import data_io as io

f_dir = pathlib.Path(__file__).parent / 'files'

PEER_NEW_FORMAT = """PEER NGA STRONG MOTION DATABASE RECORD
Chi-Chi Taiwan-04, 9/20/1999, CHY102, E
ACCELERATION TIME SERIES IN UNITS OF G
NPTS=      7, DT=   .0050 SEC,
  -.2537828E-05  -.2525256E-05  -.2492238E-05  -.2471364E-05  -.2491540E-05
  -.2587946E-05  -.2817938E-05
"""

PEER_OLD_FORMAT = """PEER NGA STRONG MOTION DATABASE RECORD
CHICHI AFTERSHOCK 09/20/99 2146, KAU048, N, (CWB)
ACCELERATION TIME HISTORY IN UNITS OF G
6    0.0100    NPTS, DT
  -0.131447E-05  -0.133225E-05  -0.134986E-05  -0.136655E-05  -0.138179E-05
  -0.139576E-05
"""


def _smc_number(value: float) -> str:
    """Format a number as in SMC files (e.g., "-1.7415E+1")."""
    exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
    return f'{value / 10**exponent:.4f}E{exponent:+d}'.rjust(10)


def _smc_text(values: list[float], sampling_rate: int = 200) -> str:
    """Create the text of an SMC file."""
    lines = ['2 CORRECTED ACCELEROGRAM', '*', '5231A'] + ['*'] * 8
    lines += ['    -32768      2005       163        15        41        50']
    lines += ['         1         3    -32768      3103        90       360']
    lines += [f'{len(values):10d}    -32768    -32768    -32768    -32768']
    lines += ['    -32768'] * 3
    lines += [f'  1.7000000E+38  {sampling_rate:.7E}  3.3533001E+01']
    lines += ['  1.7000000E+38'] * 9
    lines += ['| Digitized 2005/07/11', '| Converted to SMC']
    fields = [_smc_number(v) for v in values]
    lines += [''.join(fields[i : i + 8]) for i in range(0, len(fields), 8)]
    return '\n'.join(lines) + '\n'


def test_load_matrix__delimiters_and_headers(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'data.txt'
    path.write_text('time accel\n0, 1.5\n0.01\t-2\n\n0.02 3e-1\n')
    data = io.load_matrix(path)
    assert np.allclose(data, [[0, 1.5], [0.01, -2], [0.02, 0.3]])


def test_load_matrix__ragged_rows(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'data.txt'
    path.write_text('1 2 3\n4 5\n')
    data = io.load_matrix(path)
    assert data.shape == (2, 3)
    assert np.isnan(data[1, 2])
    with pytest.raises(io.InputError, match='NaN'):
        io.check_data(data, 'curve')


def test_load_matrix__no_numbers(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'empty.txt'
    path.write_text('nothing here\n')
    with pytest.raises(io.InputError, match='No numeric data'):
        io.load_matrix(path)


def test_save_matrix(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'out.txt'
    io.save_matrix(path, np.array([[1.23456789, 2], [3, 4]]), precision=4)
    assert path.read_text().splitlines()[0] == '1.235\t2'


def test_check_data__valid_sample_files() -> None:
    io.check_data(io.load_matrix(f_dir / 'profile_FKSH14.txt'), 'vs_profile')
    io.check_data(io.load_matrix(f_dir / 'curve_FKSH14.txt'), 'curve')
    io.check_data(io.load_matrix(f_dir / 'H4_G_IWTH04.txt'), 'mkz_param')
    io.check_data(io.load_matrix(f_dir / 'HH_G_FKSH14.txt'), 'hh_param')
    io.check_data(io.load_matrix(f_dir / 'sample_accel.txt'), 'motion')


@pytest.mark.parametrize(
    ('data', 'kind', 'message'),
    [
        (np.ones((3, 4)), 'vs_profile', 'number of columns'),
        (
            np.array([[2, 100, 2, 1800, 1], [0, 300, 0.02, 2000, 0]]),
            'vs_profile',
            'Unit of damping',
        ),
        (
            np.array([[2, 100, 0.02, 1800, 0], [0, 300, 0.02, 2000, 0]]),
            'vs_profile',
            'Material number',
        ),
        (
            np.array([[-2, 100, 0.02, 1800, 1], [0, 300, 0.02, 2000, 0]]),
            'vs_profile',
            'negative',
        ),
        (np.ones((3, 5)), 'curve', 'multiple of 4'),
        (np.array([[0, 1], [0, 2]]), 'motion', 'monotonically'),
        (np.array([[-1, 1], [0, 2]]), 'motion', 'negative time'),
        (np.ones((3, 3)), 'mkz_param', 'Should be 4'),
        (np.ones((4, 3)), 'mkz_param', 'second row'),
        (np.ones((4, 3)), 'hh_param', 'Should be 9'),
        (np.ones((2, 3)), 'tau_max', 'one row or one column'),
    ],
)
def test_check_data__invalid(
        data: np.ndarray, kind: str, message: str
) -> None:
    with pytest.raises(io.InputError, match=message):
        io.check_data(np.array(data, dtype=float), kind)


def test_check_data__name_in_message() -> None:
    with pytest.raises(io.InputError, match=r'^Motion #2 has NaN\.$'):
        io.check_data(np.array([[0, np.nan]]), 'motion', 'Motion #2')


def test_check_consistency() -> None:
    profile = io.load_matrix(f_dir / 'profile_FKSH14.txt')
    curve = io.load_matrix(f_dir / 'curve_FKSH14.txt')
    hh_g = io.load_matrix(f_dir / 'HH_G_FKSH14.txt')
    hh_x = io.load_matrix(f_dir / 'HH_X_FKSH14.txt')
    io.check_consistency(profile, curve=curve, G_param=hh_g, xi_param=hh_x)
    with pytest.raises(io.InputError, match='dimensions do not match'):
        io.check_consistency(profile, G_param=hh_g, xi_param=hh_x[:, :3])

    with pytest.raises(io.InputError, match='not have enough material'):
        io.check_consistency(profile, curve=curve[:, :8])

    with pytest.raises(io.InputError, match='4 times'):
        io.check_consistency(
            profile, curve=curve, G_param=np.hstack((hh_g, hh_g))
        )


@pytest.mark.parametrize(
    ('filename', 'code'),
    [
        ('profile_CE.11023.txt', 'CE.11023'),
        ('/some/dir/profile_FKSH14.dat', 'FKSH14'),
        ('my_site.txt', 'my_site'),
    ],
)
def test_site_code_from_filename(filename: str, code: str) -> None:
    assert io.site_code_from_filename(filename) == code


def test_unique_path(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'curve_A.txt'
    assert io.unique_path(path) == path
    path.write_text('')
    assert io.unique_path(path) == tmp_path / 'curve_A_.txt'


def test_read_peer_at2(tmp_path: pathlib.Path) -> None:
    for text, dt, n_pts in [
        (PEER_NEW_FORMAT, 0.005, 7),
        (PEER_OLD_FORMAT, 0.01, 6),
    ]:
        path = tmp_path / 'record.AT2'
        path.write_text(text)
        dt_, accel = io.read_peer_at2(path)
        assert np.isclose(dt_, dt)
        assert accel.shape == (n_pts,)

    path.write_text('too short\n')
    with pytest.raises(io.InputError, match='not a PEER file'):
        io.read_peer_at2(path)


def test_read_smc(tmp_path: pathlib.Path) -> None:
    values = list(np.linspace(-20, 20, 19))
    path = tmp_path / 'record.smc'
    path.write_text(_smc_text(values, sampling_rate=200))
    dt, accel = io.read_smc(path)
    assert np.isclose(dt, 0.005)
    assert np.allclose(accel, values, atol=1e-3)
    path.write_text('not an SMC file\n')
    with pytest.raises(io.InputError, match='not a valid SMC file'):
        io.read_smc(path)


@pytest.mark.parametrize(
    ('seconds', 'text'),
    [(5.25, '5.2 sec'), (65, '1 min 5.0 sec'), (3725, '1 hr 2 min 5.0 sec')],
)
def test_format_duration(seconds: float, text: str) -> None:
    assert io.format_duration(seconds) == text

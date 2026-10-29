import re
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import PySeismoSoil.helper_generic as hlp
from PySeismoSoil.class_Vs_profile import Vs_Profile

f_dir = Path(__file__).resolve().parent / 'files'


@pytest.fixture
def prof() -> Vs_Profile:
    data, _ = hlp.read_two_column_stuff(
        str(f_dir / 'two_column_data_example.txt')
    )
    data[:, 0] *= 10
    data[:, 1] *= 100
    return Vs_Profile(data)


DATA_2_COLUMNS = np.array(
    [[10, 20, 30, 0], [100, 120, 160, 190]],
    dtype=float,
).T
DATA_5_COLUMNS = np.array(
    [
        [10, 20, 30, 0],
        [100, 120, 160, 190],
        [0.01, 0.01, 0.01, 0.01],
        [1600, 1600, 1600, 1600],
        [1, 2, 3, 0],
    ],
    dtype=float,
).T


def _replace(
        data: np.ndarray, index: tuple[int, int], value: float
) -> np.ndarray:
    """Return a copy of ``data`` with ``data[index]`` set to ``value``."""
    data_ = data.copy()
    data_[index] = value
    return data_


@pytest.mark.parametrize(
    ('data', 'exception', 'match'),
    [
        pytest.param(
            None,
            TypeError,
            'must be a file name or a numpy array',
            id='None_as_data',
        ),
        pytest.param(
            3.6,
            TypeError,
            'must be a file name or a numpy array',
            id='other_type_as_data',
        ),
        pytest.param(
            _replace(DATA_2_COLUMNS, (2, 1), np.nan),
            ValueError,
            'should contain no NaN values',
            id='NaN_values',
        ),
        pytest.param(
            _replace(DATA_2_COLUMNS, (2, 0), 0),
            ValueError,
            'should be all positive, except',
            id='non_positive_thickness',
        ),
        pytest.param(
            _replace(DATA_2_COLUMNS, (-1, 0), -1),
            ValueError,
            'last layer thickness should be',
            id='negative_last_layer_thickness',
        ),
        pytest.param(
            np.array([[[1, 2, 3, 0], [1, 2, 3, 4]]]).T,  # one more dimension
            ValueError,
            'should be a 2D numpy array',
            id='incorrect_number_of_dimensions',
        ),
        pytest.param(
            _replace(DATA_2_COLUMNS, (2, 1), -1),
            ValueError,
            re.escape('Vs column should be all positive.'),
            id='negative_Vs',
        ),
        pytest.param(
            _replace(DATA_5_COLUMNS, (2, 3), 0),
            ValueError,
            'damping and density columns',
            id='non_positive_density',
        ),
        pytest.param(
            _replace(DATA_5_COLUMNS, (1, -1), 2.2),
            ValueError,
            'should be all integers',
            id='material_number_not_integer',
        ),
        pytest.param(
            _replace(DATA_5_COLUMNS, (1, -1), 0),
            ValueError,
            'should be all positive',
            id='material_number_not_positive',
        ),
        pytest.param(
            _replace(DATA_5_COLUMNS, (-1, -1), -1),
            ValueError,
            'last layer should be non-negative',
            id='material_number_of_last_layer_negative',
        ),
        pytest.param(
            DATA_5_COLUMNS[:, 0:-1].copy(),  # one fewer column
            ValueError,
            'either 2 or 5 columns',
            id='incorrect_number_of_columns',
        ),
    ],
)
def test_Vs_profile_format(
        data: np.ndarray | float | None,
        exception: type[Exception],
        match: str,
) -> None:
    with pytest.raises(exception, match=match):
        Vs_Profile(data)


# The profiles are transposed in the test.
@pytest.mark.parametrize(
    ('data', 'units', 'damping', 'density'),
    [
        pytest.param(
            [[10, 0], [200, 500], [5, 3], [1.8, 2.0], [1, 0]],
            {'damping_unit': '%', 'density_unit': 'g/cm^3'},
            [0.05, 0.03],
            [1800, 2000],
            id='float_array',
        ),
        pytest.param(
            [[10, 0], [200, 500], [5, 3], [1800, 2000], [1, 0]],
            {'damping_unit': '%'},
            [0.05, 0.03],
            [1800, 2000],
            id='integer_array_in_kg_per_m3',
        ),
        pytest.param(
            [[10, 0], [200, 500], [5, 3], [2, 2], [1, 0]],
            {'damping_unit': '%', 'density_unit': 'g/cm^3'},
            [0.05, 0.03],
            [2000, 2000],
            id='integer_array_in_g_per_cm3',
        ),
    ],
)
def test_init__unit_conversion_does_not_change_the_input_array(
        data: list[list[float]],
        units: dict[str, str],
        damping: list[float],
        density: list[float],
) -> None:
    data_ = np.array(data).T
    data_before = data_.copy()
    prof_1 = Vs_Profile(data_, **units)
    prof_2 = Vs_Profile(data_, **units)  # from the same array again

    # The units are converted in a copy, so the input array is not changed,
    # and the second profile is the same as the first one
    assert np.array_equal(data_, data_before)
    for prof in (prof_1, prof_2):
        assert np.allclose(prof.vs_profile[:, 2], damping)
        assert np.allclose(prof.vs_profile[:, 3], density)


@pytest.mark.parametrize(
    'data',
    [
        pytest.param(DATA_2_COLUMNS, id='2_columns'),
        pytest.param(DATA_5_COLUMNS, id='5_columns'),
    ],
)
def test_init__changing_the_input_array_does_not_change_the_profile(
        data: np.ndarray,
) -> None:
    data_ = data.copy()
    prof = Vs_Profile(data_)
    vs_profile = prof.vs_profile.copy()
    data_[:] = 999.0

    # The private arrays are the columns of `vs_profile`, not views of `data_`
    assert np.array_equal(prof.vs_profile, vs_profile)
    private_arrays = ['_thk', '_vs', '_xi', '_rho', '_material_number']
    for j, name in enumerate(private_arrays):
        assert np.array_equal(getattr(prof, name), vs_profile[:, j])

    # The thicknesses are 10, 20, 30, and 0 m
    assert np.allclose(prof.get_depth_array(), [0, 10, 30, 60])


def test_plot(prof: Vs_Profile) -> None:
    prof.plot(c='r', ls='--')

    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes()
    prof.plot(fig=fig, ax=ax, label='profile')
    ax.legend(loc='best')


@pytest.mark.parametrize(
    ('data', 'n_layer', 'material_number', 'has_halfspace'),
    [
        pytest.param(
            np.genfromtxt(f_dir / 'sample_profile.txt'),
            12,
            list(range(1, 14)),
            True,
            id='already_a_half_space',
        ),
        pytest.param(
            np.genfromtxt(f_dir / 'two_column_data_example.txt'),
            15,
            [*range(1, 16), 0],
            False,
            id='no_half_space',
        ),
        pytest.param(
            np.array([
                [10, 200, 0.05, 1800, 1],
                [20, 300, 0.05, 1800, 0],  # finite, with material number 0
            ]),
            2,
            [1, 2, 0],  # the old last layer gets a new material number
            False,
            id='no_half_space__last_layer_with_material_number_0',
        ),
    ],
)
def test_add_halfspace(
        data: np.ndarray,
        n_layer: int,
        material_number: list[int],
        *,
        has_halfspace: bool,
) -> None:
    data_before = data.copy()
    prof_1 = Vs_Profile(data, add_halfspace=False)
    prof_2 = Vs_Profile(data, add_halfspace=True)

    assert (prof_1._thk[-1] == 0) == has_halfspace
    assert prof_2._thk[-1] == 0
    assert prof_1._thk[-2] != 0
    assert prof_2._thk[-2] != 0  # assert only one "halfspace"
    assert prof_1.n_layer == n_layer
    assert prof_1.n_layer == prof_2.n_layer
    assert np.array_equal(prof_2.vs_profile[:, 4], material_number)
    assert np.array_equal(prof_2._material_number, material_number)
    assert np.array_equal(data, data_before)

    # The new profile is a valid Vs profile
    prof_3 = Vs_Profile(prof_2.vs_profile)
    assert np.array_equal(prof_3.vs_profile, prof_2.vs_profile)


def test_vs30(prof: Vs_Profile) -> None:
    assert prof.vs30 == pytest.approx(276.9231, abs=1e-4)


def test_get_amplif_function() -> None:
    profile_FKSH14 = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    af_RO = profile_FKSH14.get_ampl_function(freq_resolution=0.5, fmax=15)[0]
    af_benchmark = np.array(
        [
            [0.50, 1.20218233839345],  # from MATLAB
            [1, 2.40276180417506],
            [1.50, 3.35891308492276],
            [2, 1.52759821088595],
            [2.50, 1.23929961393844],
            [3, 1.44564547138629],
            [3.50, 2.43924932880498],
            [4, 4.01661301316906],
            [4.50, 2.32960159664501],
            [5, 1.79841404983353],
            [5.50, 1.96256021192571],
            [6, 3.12817017367637],
            [6.50, 3.92494425374814],
            [7, 2.10815322297781],
            [7.50, 1.66638537272089],
            [8, 1.95562752738785],
            [8.50, 3.37394970215842],
            [9, 2.59724801539598],
            [9.50, 1.57980154466212],
            [10, 1.42540110327715],
            [10.5, 1.82180321630950],
            [11, 3.04707962007382],
            [11.5, 2.60349869620899],
            [12, 1.84273534851058],
            [12.5, 1.79995928341286],
            [13, 2.35928076072069],
            [13.5, 3.59881564870728],
            [14, 3.31112261403936],
            [14.5, 2.61283127927210],
            [15, 2.69868407060282],
        ],
    )
    assert np.allclose(af_RO.spectrum_2col, af_benchmark, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize(
    ('get_f0', 'f0_benchmark'),
    [
        pytest.param(Vs_Profile.get_f0_BH, 1.05, id='BH'),
        pytest.param(Vs_Profile.get_f0_RO, 1.10, id='RO'),
    ],
)
def test_f0_BH_or_f0_RO(
        prof: Vs_Profile,
        get_f0: Callable[[Vs_Profile], float],
        f0_benchmark: float,
) -> None:
    assert get_f0(prof) == pytest.approx(f0_benchmark, abs=1e-2)


# The profiles and the benchmarks are transposed in the test.
@pytest.mark.parametrize(
    ('data', 'depth', 'benchmark'),
    [
        pytest.param(
            [[5, 4, 3, 2, 1], [200, 500, 700, 1000, 1200]],
            8,
            [[5, 3, 0], [200, 500, 2000]],
            id='in_the_middle_of_a_layer',
        ),
        pytest.param(
            [[5, 4, 3, 2, 1], [200, 500, 700, 1000, 1200]],
            9,
            [[5, 4, 0], [200, 500, 2000]],
            id='on_the_boundary_of_a_layer',
        ),
        pytest.param(
            [[5, 4, 3, 2, 1], [200, 500, 700, 1000, 1200]],
            30,
            [[5, 4, 3, 2, 16, 0], [200, 500, 700, 1000, 1200, 2000]],
            id='beyond_the_total_depth',
        ),
        pytest.param(
            [[5, 4, 3, 2, 1, 0], [200, 500, 700, 1000, 1200, 1500]],
            30,
            [[5, 4, 3, 2, 1, 15, 0], [200, 500, 700, 1000, 1200, 1500, 2000]],
            id='beyond_the_total_depth__with_a_half_space',
        ),
        pytest.param(
            [[5, 4, 3, 2, 1, 0], [200, 500, 700, 1000, 1200, 1200]],
            30,
            [[5, 4, 3, 2, 1, 15, 0], [200, 500, 700, 1000, 1200, 1200, 2000]],
            id='beyond_the_total_depth__with_a_half_space_of_the_same_Vs',
        ),
    ],
)
def test_truncate(
        data: list[list[float]], depth: float, benchmark: list[list[float]]
) -> None:
    prof = Vs_Profile(np.array(data).T)
    vs_profile = prof.vs_profile.copy()
    z_max = prof.z_max
    new_prof = prof.truncate(depth=depth, Vs=2000)
    assert np.allclose(new_prof.vs_profile[:, :2], np.array(benchmark).T)

    # The original profile is not changed
    assert np.array_equal(prof.vs_profile, vs_profile)
    assert np.array_equal(prof._thk, vs_profile[:, 0])
    assert prof.z_max == z_max


def test_query_Vs_at_depth__query_numpy_array() -> None:
    prof = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))

    # (1) Test ground surface
    assert prof.query_Vs_at_depth(0.0, as_profile=False) == pytest.approx(
        120.0, abs=1e-7
    )

    # (2) Test depth within a layer
    assert prof.query_Vs_at_depth(1.0, as_profile=False) == pytest.approx(
        120.0, abs=1e-7
    )

    # (3) Test depth at layer interface
    assert prof.query_Vs_at_depth(105, as_profile=False) == pytest.approx(
        1030.0, abs=1e-7
    )
    assert prof.query_Vs_at_depth(106, as_profile=False) == pytest.approx(
        1210.0, abs=1e-7
    )
    assert prof.query_Vs_at_depth(107, as_profile=False) == pytest.approx(
        1210.0, abs=1e-7
    )

    # (4) Test infinite depth
    assert prof.query_Vs_at_depth(1e9, as_profile=False) == pytest.approx(
        1210.0, abs=1e-7
    )

    # (5) Test depth at layer interface -- input is an array
    result = prof.query_Vs_at_depth(np.array([7, 8, 9]), as_profile=False)
    is_all_close = np.allclose(result, [190, 280, 280])
    assert is_all_close

    # (6) Test depth at ground sufrace and interface
    result = prof.query_Vs_at_depth(np.array([0, 1, 2, 3]), as_profile=False)
    is_all_close = np.allclose(result, [120, 120, 190, 190])
    assert is_all_close

    # (7) Test invalid input: list
    with pytest.raises(TypeError, match='needs to be a single number'):
        prof.query_Vs_at_depth([1, 2])

    with pytest.raises(TypeError, match='needs to be a single number'):
        prof.query_Vs_at_depth({1, 2})

    with pytest.raises(TypeError, match='needs to be a single number'):
        prof.query_Vs_at_depth((1, 2))

    # (8) Test invalid input: negative values
    with pytest.raises(ValueError, match='Please provide non-negative'):
        prof.query_Vs_at_depth(-2)

    with pytest.raises(ValueError, match='Please provide non-negative'):
        prof.query_Vs_at_depth(np.array([-2, 1]))


def test_query_Vs_at_depth__query_Vs_Profile_objects() -> None:
    prof = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))

    # (1) Test invalid input: non-increasing array
    with pytest.raises(
        ValueError, match='needs to be monotonically increasing'
    ):
        prof.query_Vs_at_depth(np.array([1, 2, 5, 4, 6]), as_profile=True)

    # (2) Test invalid input: repeated values
    with pytest.raises(
        ValueError, match='should not contain duplicate values'
    ):
        prof.query_Vs_at_depth(np.array([1, 2, 4, 4, 6]), as_profile=True)

    # (3) Test a scalar input
    result = prof.query_Vs_at_depth(1.0, as_profile=True)
    benchmark = Vs_Profile(np.array([[1, 0], [120, 120]]).T)
    compare = np.allclose(result.vs_profile, benchmark.vs_profile)
    assert compare

    # (4) Test a general case
    result = prof.query_Vs_at_depth(np.array([0, 1, 2, 3, 9]), as_profile=True)
    benchmark = Vs_Profile(
        np.array([[1, 1, 1, 6, 0], [120, 120, 190, 190, 280]]).T
    )
    compare = np.allclose(result.vs_profile, benchmark.vs_profile)
    assert compare


def test_query_Vs_given_thk__using_a_scalar_as_thk() -> None:
    prof = Vs_Profile(str(f_dir / 'sample_profile.txt'))

    # (1a) Test trivial case: top of layer
    result = prof.query_Vs_given_thk(9, n_layers=1, at_midpoint=False)
    is_all_close = np.allclose(result, [10])
    assert is_all_close

    # (1b) Test trivial case: mid point of layer
    result = prof.query_Vs_given_thk(9, n_layers=1, at_midpoint=True)
    is_all_close = np.allclose(result, [50])
    assert is_all_close

    # (2a) Test normal case: top of layer
    result = prof.query_Vs_given_thk(1, n_layers=5, at_midpoint=False)
    is_all_close = np.allclose(result, [10, 20, 30, 40, 50])
    assert is_all_close

    # (2a) Test normal case: mid point of layer
    result = prof.query_Vs_given_thk(1, n_layers=14, at_midpoint=True)
    is_all_close = np.allclose(
        result,
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 120, 120],
    )
    assert is_all_close

    # (3a) Test invalid input
    with pytest.raises(ValueError, match='should be positive'):
        result = prof.query_Vs_given_thk(1, n_layers=0)

    # (3b) Test invalid input
    with pytest.raises(TypeError, match='needs to be a scalar or a numpy'):
        result = prof.query_Vs_given_thk([1, 2], n_layers=0)

    # (4a) Test large thickness: top of layer
    result = prof.query_Vs_given_thk(100, n_layers=4, at_midpoint=False)
    is_all_close = np.allclose(result, [10, 120, 120, 120])
    assert is_all_close

    # (4b) Test large thickness: mid point of layer (version 1)
    result = prof.query_Vs_given_thk(100, n_layers=4, at_midpoint=True)
    is_all_close = np.allclose(result, [120, 120, 120, 120])
    assert is_all_close

    # (4c) Test large thickness: mid point of layer (version 2)
    result = prof.query_Vs_given_thk(17, n_layers=3, at_midpoint=True)
    is_all_close = np.allclose(result, [90, 120, 120])
    assert is_all_close

    # (4d) Test large thickness: mid point of layer (version 3)
    result = prof.query_Vs_given_thk(18, n_layers=3, at_midpoint=True)
    is_all_close = np.allclose(result, [100, 120, 120])
    assert is_all_close

    # (5a) Test returning Vs_Profile object: one layer, on top of layers
    result = prof.query_Vs_given_thk(
        9,
        n_layers=1,
        as_profile=True,
        at_midpoint=False,
        add_halfspace=True,
    )
    benchmark = Vs_Profile(np.array([[9, 10], [0, 10]]))
    compare = np.allclose(result.vs_profile, benchmark.vs_profile)
    assert compare

    # (5b) Test returning Vs_Profile object: one layer, mid point of layers
    result = prof.query_Vs_given_thk(
        9,
        n_layers=1,
        as_profile=True,
        at_midpoint=True,
        add_halfspace=True,
    )
    benchmark = Vs_Profile(np.array([[9, 50], [0, 50]]))
    compare = np.allclose(result.vs_profile, benchmark.vs_profile)
    assert compare

    # (5c) Test returning Vs_Profile object: one layer, mid point of layers
    result = prof.query_Vs_given_thk(
        9,
        n_layers=1,
        as_profile=True,
        at_midpoint=True,
        add_halfspace=False,
    )
    benchmark = Vs_Profile(np.array([[9, 50]]))
    compare = np.allclose(result.vs_profile, benchmark.vs_profile)
    assert compare

    # (6a) Test returning Vs_Profile object: multiple layers, top of layers
    result = prof.query_Vs_given_thk(
        3,
        n_layers=5,
        as_profile=True,
        at_midpoint=False,
        add_halfspace=True,
        show_fig=True,
    )
    benchmark = Vs_Profile(
        np.array([[3, 3, 3, 3, 3, 0], [10, 40, 70, 100, 120, 120]]).T,
    )
    compare = np.allclose(result.vs_profile, benchmark.vs_profile)
    assert compare

    # (6b) Test returning Vs_Profile object: multiple layers, mid of layers
    result = prof.query_Vs_given_thk(
        3,
        n_layers=5,
        as_profile=True,
        at_midpoint=True,
        add_halfspace=True,
        show_fig=True,
    )
    benchmark = Vs_Profile(
        np.array([[3, 3, 3, 3, 3, 0], [20, 50, 80, 110, 120, 120]]).T,
    )
    compare = np.allclose(result.vs_profile, benchmark.vs_profile)
    assert compare


@pytest.mark.parametrize(
    ('precision_kwargs', 'first_line'),
    [
        pytest.param({}, '10.00\t100.00\t0.01\t1600\t1', id='default'),
        pytest.param(
            {'precision': ('%.1f', '%.1f', '%.3f', '%.1f', '%d')},
            '10.0\t100.0\t0.010\t1600.0\t1',
            id='tuple',
        ),
        pytest.param(
            {'precision': ['%.1f', '%.1f', '%.3f', '%.1f', '%d']},
            '10.0\t100.0\t0.010\t1600.0\t1',
            id='list',
        ),
    ],
)
def test_to_txt(
        tmp_path: Path,
        precision_kwargs: dict[str, Sequence[str]],
        first_line: str,
) -> None:
    prof = Vs_Profile(DATA_5_COLUMNS)
    file_name = tmp_path / 'profile.txt'
    prof.to_txt(str(file_name), **precision_kwargs)

    assert file_name.read_text(encoding='utf-8').splitlines()[0] == first_line
    prof_from_file = Vs_Profile(str(file_name))
    assert np.allclose(prof_from_file.vs_profile, prof.vs_profile)


@pytest.mark.parametrize(
    ('precision', 'exception', 'match'),
    [
        pytest.param('%.2f', TypeError, 'must be a tuple or a list', id='str'),
        pytest.param(
            ('%.2f', '%.2f', '%.4g', '%.5g'),
            ValueError,
            'Length of precision must be 5',
            id='4_specifiers',
        ),
    ],
)
def test_to_txt__invalid_precision(
        tmp_path: Path,
        precision: str | tuple[str, ...],
        exception: type[Exception],
        match: str,
) -> None:
    prof = Vs_Profile(DATA_5_COLUMNS)
    with pytest.raises(exception, match=match):
        prof.to_txt(str(tmp_path / 'profile.txt'), precision=precision)

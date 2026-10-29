from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from PySeismoSoil.class_curves import (
    Curve,
    Damping_Curve,
    GGmax_Curve,
    Multiple_Damping_Curves,
    Multiple_GGmax_Curves,
    Multiple_GGmax_Damping_Curves,
    Stress_Curve,
)
from PySeismoSoil.class_parameters import (
    HH_Param,
    HH_Param_Multi_Layer,
    MKZ_Param,
    MKZ_Param_Multi_Layer,
)

f_dir = Path(__file__).resolve().parent / 'files'


def test_init() -> None:
    data = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    curve = Curve(data[:, 2:4])
    damping_data = curve.raw_data[:, 1]
    damping_bench = [
        1.6683,
        1.8386,
        2.4095,
        3.8574,
        7.4976,
        12.686,
        18.102,
        21.005,
        21.783,
        21.052,
    ]
    assert np.allclose(damping_data, damping_bench)


CurveInUnit1Case = tuple[type[Curve], dict[str, str], slice, list[float]]


# A curve class, its arguments for data in unit 1, the columns of
# "curve_FKSH14.txt" (data in %) to use, and the factors from unit 1 to %
@pytest.fixture(
    params=[
        pytest.param(
            (Curve, {'strain_unit': '1'}, slice(0, 2), [100, 1]), id='Curve'
        ),
        pytest.param(
            (GGmax_Curve, {'strain_unit': '1'}, slice(0, 2), [100, 1]),
            id='GGmax_Curve',
        ),
        pytest.param(
            (
                Damping_Curve,
                {'strain_unit': '1', 'damping_unit': '1'},
                slice(2, 4),
                [100, 100],
            ),
            id='Damping_Curve',
        ),
    ],
)
def curve_in_unit_1_case(request: pytest.FixtureRequest) -> CurveInUnit1Case:
    return request.param


@pytest.mark.parametrize(
    'interpolate', [False, True], ids=['not_interpolated', 'interpolated']
)
def test_init__units_converted_in_a_copy_of_the_input(
        curve_in_unit_1_case: CurveInUnit1Case,
        *,
        interpolate: bool,
) -> None:
    curve_class, unit_kwargs, columns, unit_factors = curve_in_unit_1_case

    # G/Gmax or damping data in %, passed in unit 1
    data_in_pct = np.genfromtxt(f_dir / 'curve_FKSH14.txt')[:, columns]
    user_array = data_in_pct / unit_factors
    user_array_before = user_array.copy()
    curve = curve_class(user_array, interpolate=interpolate, **unit_kwargs)

    # The caller's array is not changed, and the object does not share it
    assert np.array_equal(user_array, user_array_before)
    for attribute in [curve.raw_data, curve.strain, curve.values]:
        assert not np.shares_memory(attribute, user_array)

    # `raw_data` holds the data points (not interpolated) in %, like `strain`
    assert np.allclose(curve.raw_data, data_in_pct)


def test_init__interpolated_curve_same_in_unit_1_and_pct(
        curve_in_unit_1_case: CurveInUnit1Case,
) -> None:
    curve_class, unit_kwargs, columns, unit_factors = curve_in_unit_1_case
    data_in_pct = np.genfromtxt(f_dir / 'curve_FKSH14.txt')[:, columns]
    curve_in_1 = curve_class(
        data_in_pct / unit_factors, interpolate=True, **unit_kwargs
    )
    curve_in_pct = curve_class(data_in_pct, interpolate=True)

    # `min_strain` and `max_strain` (default: 1e-4 and 10) are in %, whatever
    # the unit of the data
    assert np.allclose(curve_in_1.strain, np.geomspace(1e-4, 10, num=50))
    assert np.allclose(curve_in_1.strain, curve_in_pct.strain)
    assert np.allclose(curve_in_1.values, curve_in_pct.values)


@pytest.mark.parametrize(
    ('stress_unit', 'kPa_per_unit'),
    [('Pa', 1e-3), ('kPa', 1.0), ('MPa', 1e3), ('GPa', 1e6)],
)
def test_stress_curve__units_converted_in_a_copy_of_the_input(
        stress_unit: str, kPa_per_unit: float
) -> None:
    strain_in_pct = np.geomspace(1e-4, 10, num=20)
    stress_in_kPa = 50 * strain_in_pct / (1 + strain_in_pct)

    # The default `strain_unit` of `Stress_Curve` is '1'
    user_array = np.column_stack((
        strain_in_pct / 100,
        stress_in_kPa / kPa_per_unit,
    ))
    user_array_before = user_array.copy()
    curve = Stress_Curve(user_array, stress_unit=stress_unit)

    # The caller's array is not changed, and the object does not share it
    assert np.array_equal(user_array, user_array_before)
    for attribute in [curve.raw_data, curve.strain, curve.stress]:
        assert not np.shares_memory(attribute, user_array)

    # `raw_data`, `strain` and `stress` are in % and kPa
    expected = np.column_stack((strain_in_pct, stress_in_kPa))
    assert np.allclose(curve.raw_data, expected)
    assert np.allclose(np.column_stack((curve.strain, curve.stress)), expected)


def test_stress_curve__integer_input_array() -> None:
    # The conversion from Pa to kPa gives floats, even for integer input
    user_array = np.array([[1, 1500], [2, 2500], [3, 3000]])
    curve = Stress_Curve(user_array, strain_unit='%', stress_unit='Pa')
    assert np.allclose(curve.stress, [1.5, 2.5, 3.0])
    assert np.array_equal(user_array, [[1, 1500], [2, 2500], [3, 3000]])


def test_plot() -> None:
    filename = str(f_dir / 'curve_FKSH14.txt')
    data = np.genfromtxt(filename)
    curve = Curve(data[:, 2:4])
    curve.plot(marker='.')

    curves = Multiple_Damping_Curves(filename)
    curves.plot()


HH_X_PARAM_NAMES = {
    'gamma_t',
    'a',
    'gamma_ref',
    'beta',
    's',
    'Gmax',
    'mu',
    'Tmax',
    'd',
}
H4_X_PARAM_NAMES = {'gamma_ref', 's', 'beta', 'Gmax'}


@pytest.mark.parametrize(
    ('get_param', 'param_names'),
    [
        pytest.param(
            Damping_Curve.get_HH_x_param, HH_X_PARAM_NAMES, id='HH_x'
        ),
        pytest.param(
            Damping_Curve.get_H4_x_param, H4_X_PARAM_NAMES, id='H4_x'
        ),
    ],
)
def test_HH_x_or_H4_x_fit_single_layer(
        get_param: Callable[..., HH_Param | MKZ_Param], param_names: set[str]
) -> None:
    data = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    curve = Damping_Curve(data[:, 2:4])

    try:
        param = get_param(
            curve,
            pop_size=1,
            n_gen=1,
            show_fig=True,
            use_scipy=False,
        )
        assert len(param) == len(param_names)
        assert param.keys() == param_names
    except ImportError:  # DEAP library may not be installed
        pass

    param = get_param(
        curve, pop_size=1, n_gen=1, show_fig=True, use_scipy=True
    )
    assert len(param) == len(param_names)
    assert param.keys() == param_names


@pytest.mark.parametrize(
    'interpolate', [False, True], ids=['not_interpolated', 'interpolated']
)
@pytest.mark.parametrize(
    'get_param',
    [
        pytest.param(Damping_Curve.get_HH_x_param, id='HH_x'),
        pytest.param(Damping_Curve.get_H4_x_param, id='H4_x'),
    ],
)
def test_HH_x_or_H4_x_fit_single_layer__same_result_in_unit_1_and_pct(
        get_param: Callable[..., HH_Param | MKZ_Param],
        *,
        interpolate: bool,
) -> None:
    data_in_1 = np.genfromtxt(f_dir / 'curve_FKSH14.txt')[:, 2:4] / 100.0
    curve_in_1 = Damping_Curve(
        data_in_1,
        strain_unit='1',
        damping_unit='1',
        interpolate=interpolate,
    )
    curve_in_pct = Damping_Curve(data_in_1 * 100, interpolate=interpolate)

    # Both curves fit the same data points, in %
    assert np.array_equal(curve_in_1.raw_data, curve_in_pct.raw_data)
    param_in_1 = get_param(curve_in_1, pop_size=1, n_gen=1)
    param_in_pct = get_param(curve_in_pct, pop_size=1, n_gen=1)
    assert dict(param_in_1) == pytest.approx(dict(param_in_pct))


@pytest.mark.parametrize(
    'get_param',
    [
        pytest.param(Damping_Curve.get_HH_x_param, id='HH_x'),
        pytest.param(Damping_Curve.get_H4_x_param, id='H4_x'),
    ],
)
def test_HH_x_or_H4_x_fit_single_layer__curve_unchanged(
        get_param: Callable[..., HH_Param | MKZ_Param],
) -> None:
    user_array = np.genfromtxt(f_dir / 'curve_FKSH14.txt')[:, 2:4]
    user_array_before = user_array.copy()
    curve = Damping_Curve(user_array)
    get_param(curve, pop_size=1, n_gen=1)

    # The fit subtracts the initial damping from a copy of the data
    assert np.array_equal(curve.damping, user_array_before[:, 1])
    assert np.array_equal(curve.raw_data, user_array_before)
    assert np.array_equal(user_array, user_array_before)


def test_value_check() -> None:
    data = np.genfromtxt(f_dir / 'curve_FKSH14.txt')[:, 2:4]
    with pytest.raises(ValueError, match='G/Gmax values must be between'):
        GGmax_Curve(data)

    with pytest.raises(ValueError, match='damping values must be between'):
        Damping_Curve(data * 100.0)

    with pytest.raises(ValueError, match='should have all non-negative'):
        Stress_Curve(data * -1)


def test_multiple_damping_curves() -> None:
    mdc = Multiple_Damping_Curves(str(f_dir / 'curve_FKSH14.txt'))

    # Test __len__
    assert len(mdc) == 5

    # Test __getitem__
    strain_bench = [
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1,
        3,
    ]
    damping_bench = [
        1.6683,
        1.8386,
        2.4095,
        3.8574,
        7.4976,
        12.686,
        18.102,
        21.005,
        21.783,
        21.052,
    ]
    layer_0_bench = np.column_stack((strain_bench, damping_bench))
    assert np.allclose(mdc[0].raw_data, layer_0_bench)

    # Test __setitem__
    with pytest.raises(TypeError, match='new `item` must be of type'):
        mdc[2] = 2.5  # use an incorrect type

    mdc[4] = Damping_Curve(layer_0_bench)
    assert np.allclose(mdc[0].raw_data, mdc[4].raw_data)

    # Test __delitem__
    mdc_2 = mdc[2]
    mdc_3 = mdc[3]
    del mdc[2]
    assert len(mdc) == 4
    assert mdc.n_layer == 4

    # Test __contains__
    assert mdc_2 not in mdc
    assert mdc_3 in mdc

    # Test slicing
    mdc_slice = mdc[:2]
    assert len(mdc_slice) == 2
    assert isinstance(mdc_slice, Multiple_Damping_Curves)
    assert isinstance(mdc_slice[0], Damping_Curve)

    # Test append
    with pytest.raises(TypeError, match='`curve` should be a numpy array'):
        # use an incorrect type
        mdc[2] = GGmax_Curve(str(f_dir / 'curve_FKSH14.txt'))

    assert len(mdc) == 4
    mdc.append(mdc_3)
    assert len(mdc) == 5
    assert mdc.n_layer == 5


@pytest.mark.parametrize(
    'use_scipy',
    [
        pytest.param(True, id='differential_evolution'),
        pytest.param(False, id='DEAP'),
    ],
)
@pytest.mark.parametrize(
    ('get_all_params', 'param_names'),
    [
        pytest.param(
            Multiple_Damping_Curves.get_all_HH_x_params,
            HH_X_PARAM_NAMES,
            id='HH_x',
        ),
        pytest.param(
            Multiple_Damping_Curves.get_all_H4_x_params,
            H4_X_PARAM_NAMES,
            id='H4_x',
        ),
    ],
)
def test_HH_x_or_H4_x_fit_multi_layer(
        get_all_params: Callable[
            ..., HH_Param_Multi_Layer | MKZ_Param_Multi_Layer
        ],
        param_names: set[str],
        *,
        use_scipy: bool,
) -> None:
    mdc = Multiple_Damping_Curves(str(f_dir / 'curve_FKSH14.txt'))
    mdc_ = mdc[:2]
    try:
        params = get_all_params(
            mdc_,
            pop_size=1,
            n_gen=1,
            save_txt=False,
            use_scipy=use_scipy,
        )
    except ImportError:
        if use_scipy:
            raise

        return  # DEAP library may not be installed

    assert len(params) == 2
    assert isinstance(params[0].data, dict)
    assert params[0].keys() == param_names


@pytest.mark.parametrize(
    'interpolate', [False, True], ids=['not_interpolated', 'interpolated']
)
@pytest.mark.parametrize(
    'get_all_params',
    [
        pytest.param(Multiple_Damping_Curves.get_all_HH_x_params, id='HH_x'),
        pytest.param(Multiple_Damping_Curves.get_all_H4_x_params, id='H4_x'),
    ],
)
def test_HH_x_or_H4_x_fit_multi_layer__same_result_in_unit_1_and_pct(
        get_all_params: Callable[
            ..., HH_Param_Multi_Layer | MKZ_Param_Multi_Layer
        ],
        *,
        interpolate: bool,
) -> None:
    curve_data = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    damping_data_in_1 = [
        curve_data[:, 2:4] / 100.0,
        curve_data[:, 6:8] / 100.0,
    ]
    mdc_in_1 = Multiple_Damping_Curves([
        Damping_Curve(
            data,
            strain_unit='1',
            damping_unit='1',
            interpolate=interpolate,
        )
        for data in damping_data_in_1
    ])
    mdc_in_pct = Multiple_Damping_Curves([
        Damping_Curve(data * 100, interpolate=interpolate)
        for data in damping_data_in_1
    ])

    params_in_1 = get_all_params(mdc_in_1, pop_size=1, n_gen=1)
    params_in_pct = get_all_params(mdc_in_pct, pop_size=1, n_gen=1)
    for param_in_1, param_in_pct in zip(
        params_in_1.param_list, params_in_pct.param_list, strict=True
    ):
        assert dict(param_in_1) == pytest.approx(dict(param_in_pct))


@pytest.mark.parametrize(
    'get_all_params',
    [
        pytest.param(Multiple_Damping_Curves.get_all_HH_x_params, id='HH_x'),
        pytest.param(Multiple_Damping_Curves.get_all_H4_x_params, id='H4_x'),
    ],
)
def test_HH_x_or_H4_x_fit_multi_layer__curves_unchanged(
        get_all_params: Callable[
            ..., HH_Param_Multi_Layer | MKZ_Param_Multi_Layer
        ],
) -> None:
    mdc = Multiple_Damping_Curves(str(f_dir / 'curve_FKSH14.txt'))[:2]
    raw_data_before = [curve.raw_data.copy() for curve in mdc.curves]
    damping_before = [curve.damping.copy() for curve in mdc.curves]
    curve_matrix_before = mdc.get_curve_matrix()

    # With `parallel=False`, the fit gets the curves' own arrays (the worker
    # processes of `parallel=True` get copies)
    get_all_params(mdc, pop_size=1, n_gen=1, parallel=False)
    for curve, raw_data, damping in zip(
        mdc.curves, raw_data_before, damping_before, strict=True
    ):
        assert np.array_equal(curve.raw_data, raw_data)
        assert np.array_equal(curve.damping, damping)

    assert np.array_equal(mdc.get_curve_matrix(), curve_matrix_before)


def test_get_all_H4_x_params__saved_file_loads_back(tmp_path: Path) -> None:
    mdc = Multiple_Damping_Curves(str(f_dir / 'curve_FKSH14.txt'))[:2]
    txt_filename = str(tmp_path / 'H4_x_FKSH14.txt')
    params = mdc.get_all_H4_x_params(
        pop_size=1, n_gen=1, save_txt=True, txt_filename=txt_filename
    )

    # Same format as the "H4_x_*.txt" files: one column per layer, with the
    # rows gamma_ref, 0, s, beta
    saved = np.genfromtxt(txt_filename)
    assert saved.shape == (4, 2)
    for j, param in enumerate(params.param_list):
        assert np.allclose(
            saved[:, j],
            [param['gamma_ref'], 0, param['s'], param['beta']],
            rtol=1e-5,
        )

    # The file has no Gmax, and loads back with Gmax = 1.0, which is also the
    # Gmax of the fitted parameters
    params_loaded = MKZ_Param_Multi_Layer(txt_filename)
    assert params_loaded.n_layer == 2
    for j, param in enumerate(params.param_list):
        assert dict(params_loaded[j]) == pytest.approx(dict(param), rel=1e-5)

    # `save_txt()` writes the same file (same layout and precision)
    txt_filename_2 = tmp_path / 'H4_x_FKSH14_2.txt'
    params.save_txt(str(txt_filename_2))
    saved_text = Path(txt_filename).read_text(encoding='utf-8')
    assert txt_filename_2.read_text(encoding='utf-8') == saved_text


@pytest.mark.parametrize(
    ('curves_class', 'get_curve_matrix', 'filler_value', 'filled_column'),
    [
        # The damping curves are lost, so the damping columns (index 3 of
        # every 4 columns) are filled with a dummy value
        pytest.param(
            Multiple_GGmax_Curves,
            lambda mgc, value: mgc.get_curve_matrix(
                damping_filler_value=value
            ),
            1.23,
            3,
            id='multiple_GGmax_curves',
        ),
        # The G/Gmax curves are lost, so the G/Gmax columns (index 1 of every
        # 4 columns) are filled with a dummy value
        pytest.param(
            Multiple_Damping_Curves,
            lambda mdc, value: mdc.get_curve_matrix(GGmax_filler_value=value),
            0.76,
            1,
            id='multiple_damping_curves',
        ),
    ],
)
def test_get_curve_matrix(
        curves_class: type[Multiple_GGmax_Curves | Multiple_Damping_Curves],
        get_curve_matrix: Callable[[Any, float], np.ndarray],
        filler_value: float,
        filled_column: int,
) -> None:
    curves = curves_class(str(f_dir / 'curve_FKSH14.txt'))
    curve = get_curve_matrix(curves, filler_value)

    curve_benchmark = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    for j in range(curve_benchmark.shape[1]):
        # the original info is lost; use the same dummy value
        if j % 4 == filled_column:
            curve_benchmark[:, j] = filler_value

    assert np.allclose(curve, curve_benchmark, rtol=1e-5, atol=0.0)


def test_init_multiple_GGmax_damping_curves() -> None:
    # Case 1: with MGC and MDC
    mgc = Multiple_GGmax_Curves(str(f_dir / 'curve_FKSH14.txt'))
    mdc = Multiple_Damping_Curves(str(f_dir / 'curve_FKSH14.txt'))
    with pytest.raises(ValueError, match='Both parameters are `None`'):
        Multiple_GGmax_Damping_Curves()

    with pytest.raises(ValueError, match='one and only one input parameter'):
        Multiple_GGmax_Damping_Curves(mgc_and_mdc=(mgc, mdc), data=2.6)

    with pytest.raises(TypeError, match='needs to be of type'):
        Multiple_GGmax_Damping_Curves(mgc_and_mdc=(mdc, mgc))

    mgc_ = Multiple_GGmax_Curves(str(f_dir / 'curve_FKSH14.txt'))
    del mgc_[-1]
    with pytest.raises(ValueError, match='same number of soil layers'):
        Multiple_GGmax_Damping_Curves(mgc_and_mdc=(mgc_, mdc))

    mgdc = Multiple_GGmax_Damping_Curves(mgc_and_mdc=(mgc, mdc))
    matrix = mgdc.get_curve_matrix()
    benchmark = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    assert np.allclose(matrix, benchmark)

    # Case 2: with a numpy array
    array = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    array_ = np.column_stack((array, array[:, -1]))
    with pytest.raises(ValueError, match='needs to be a multiple of 4'):
        Multiple_GGmax_Damping_Curves(data=array_)

    mgdc = Multiple_GGmax_Damping_Curves(data=array)
    mgc_, mdc_ = mgdc.get_MGC_MDC_objects()
    assert np.allclose(mgc_.get_curve_matrix(), mgc.get_curve_matrix())
    assert np.allclose(mdc_.get_curve_matrix(), mdc.get_curve_matrix())

    # Case 3: with a file name
    with pytest.raises(
        TypeError, match='must be a 2D numpy array or a file name'
    ):
        Multiple_GGmax_Damping_Curves(data=3.5)

    mgdc = Multiple_GGmax_Damping_Curves(data=str(f_dir / 'curve_FKSH14.txt'))
    mgc_, mdc_ = mgdc.get_MGC_MDC_objects()
    assert np.allclose(mgc_.get_curve_matrix(), mgc.get_curve_matrix())
    assert np.allclose(mdc_.get_curve_matrix(), mdc.get_curve_matrix())

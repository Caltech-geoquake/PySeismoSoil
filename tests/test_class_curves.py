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

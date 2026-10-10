from pathlib import Path

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


def test_HH_x_fit_single_layer() -> None:
    data = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    curve = Damping_Curve(data[:, 2:4])

    try:
        hhx = curve.get_HH_x_param(
            pop_size=1,
            n_gen=1,
            show_fig=True,
            use_scipy=False,
        )
        assert len(hhx) == 9
        assert hhx.keys() == {
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
    except ImportError:  # DEAP library may not be installed
        pass

    hhx = curve.get_HH_x_param(
        pop_size=1, n_gen=1, show_fig=True, use_scipy=True
    )
    assert len(hhx) == 9
    assert hhx.keys() == {
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


def test_H4_x_fit_single_layer() -> None:
    data = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    curve = Damping_Curve(data[:, 2:4])

    try:
        h4x = curve.get_H4_x_param(
            pop_size=1,
            n_gen=1,
            show_fig=True,
            use_scipy=False,
        )
        assert len(h4x) == 4
        assert h4x.keys() == {'gamma_ref', 's', 'beta', 'Gmax'}
    except ImportError:  # DEAP library may not be installed
        pass

    h4x = curve.get_H4_x_param(
        pop_size=1,
        n_gen=1,
        show_fig=True,
        use_scipy=True,
    )
    assert len(h4x) == 4
    assert h4x.keys() == {'gamma_ref', 's', 'beta', 'Gmax'}


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


def test_HH_x_fit_multi_layer__differential_evolution_algorithm() -> None:
    mdc = Multiple_Damping_Curves(str(f_dir / 'curve_FKSH14.txt'))
    mdc_ = mdc[:2]
    hhx = mdc_.get_all_HH_x_params(
        pop_size=1,
        n_gen=1,
        save_txt=False,
        use_scipy=True,
    )
    assert len(hhx) == 2
    assert isinstance(hhx[0].data, dict)
    assert hhx[0].keys() == {
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


def test_HH_x_fit_multi_layer__DEAP_algorithm() -> None:
    mdc = Multiple_Damping_Curves(str(f_dir / 'curve_FKSH14.txt'))
    mdc_ = mdc[:2]
    try:
        hhx = mdc_.get_all_HH_x_params(
            pop_size=1,
            n_gen=1,
            save_txt=False,
            use_scipy=False,
        )
        assert len(hhx) == 2
        assert isinstance(hhx[0].data, dict)
        assert hhx[0].keys() == {
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
    except ImportError:  # DEAP library may not be installed
        pass


def test_H4_x_fit_multi_layer__differential_evolution_algorithm() -> None:
    mdc = Multiple_Damping_Curves(str(f_dir / 'curve_FKSH14.txt'))
    mdc_ = mdc[:2]
    h4x = mdc_.get_all_H4_x_params(
        pop_size=1,
        n_gen=1,
        save_txt=False,
        use_scipy=True,
    )
    assert len(h4x) == 2
    assert isinstance(h4x[0].data, dict)
    assert h4x[0].keys() == {'gamma_ref', 's', 'beta', 'Gmax'}


def test_H4_x_fit_multi_layer__DEAP_algorithm() -> None:
    mdc = Multiple_Damping_Curves(str(f_dir / 'curve_FKSH14.txt'))
    mdc_ = mdc[:2]
    try:
        h4x = mdc_.get_all_H4_x_params(
            pop_size=1,
            n_gen=1,
            save_txt=False,
            use_scipy=False,
        )
        assert len(h4x) == 2
        assert isinstance(h4x[0].data, dict)
        assert h4x[0].keys() == {'gamma_ref', 's', 'beta', 'Gmax'}
    except ImportError:  # DEAP library may not be installed
        pass


def test_multiple_GGmax_curve_get_curve_matrix() -> None:
    damping = 1.23  # choose a dummy value

    mgc = Multiple_GGmax_Curves(str(f_dir / 'curve_FKSH14.txt'))
    curve = mgc.get_curve_matrix(damping_filler_value=damping)

    curve_benchmark = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    for j in range(curve_benchmark.shape[1]):
        # original damping info is lost; use same dummy value
        if j % 4 == 3:
            curve_benchmark[:, j] = damping

    assert np.allclose(curve, curve_benchmark, rtol=1e-5, atol=0.0)


def test_multiple_damping_curve_get_curve_matrix() -> None:
    GGmax = 0.76  # choose a dummy value

    mdc = Multiple_Damping_Curves(str(f_dir / 'curve_FKSH14.txt'))
    curve = mdc.get_curve_matrix(GGmax_filler_value=GGmax)

    curve_benchmark = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    for j in range(curve_benchmark.shape[1]):
        # original damping info is lost; use same dummy value
        if j % 4 == 1:
            curve_benchmark[:, j] = GGmax

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

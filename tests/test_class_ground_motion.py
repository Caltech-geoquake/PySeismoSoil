import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy as sp

from PySeismoSoil.class_frequency_spectrum import Frequency_Spectrum
from PySeismoSoil.class_ground_motion import Ground_Motion as GM
from PySeismoSoil.class_Vs_profile import Vs_Profile

f_dir = Path(__file__).resolve().parent / 'files'


def test_loading_data__two_columns_from_file() -> None:
    # Two columns from file
    gm = GM(str(f_dir / 'sample_accel.txt'), unit='gal')

    PGA_benchmark = 294.30  # unit: cm/s/s
    PGV_benchmark = 31.46  # unit: cm/s
    PGD_benchmark = 38.77  # unit: cm
    tol = 1e-2

    assert gm.pga_in_gal == pytest.approx(PGA_benchmark, abs=tol)
    assert gm.pgv_in_cm_s == pytest.approx(PGV_benchmark, abs=tol)
    assert gm.pgd_in_cm == pytest.approx(PGD_benchmark, abs=tol)
    assert gm.peak_Arias_Intensity == pytest.approx(1.524, abs=tol)
    assert gm.rms_accel == pytest.approx(0.4645, abs=tol)


def test_loading_data__two_columns_from_numpy_array() -> None:
    # Two columns from numpy array
    gm = GM(np.array([[0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4]]).T, unit='m/s/s')
    assert gm.pga == pytest.approx(4, abs=1e-7)


def test_loading_data__one_column_from_file() -> None:
    # One column from file
    gm = GM(str(f_dir / 'one_column_data_example.txt'), unit='g', dt=0.2)
    assert gm.pga_in_g == pytest.approx(12.0, abs=1e-7)


def test_loading_data__one_column_from_numpy_array() -> None:
    # One column from numpy array
    gm = GM(np.array([1, 2, 3, 4, 5]), unit='gal', dt=0.1)
    assert gm.pga_in_gal == pytest.approx(5.0, abs=1e-7)


def test_loading_data__one_column_without_specifying_dt() -> None:
    # One column without specifying dt
    error_msg = 'is needed for one-column `data`.'
    with pytest.raises(ValueError, match=error_msg):
        GM(np.array([1, 2, 3, 4, 5]), unit='gal')


def test_loading_data__test_invalid_unit_names() -> None:
    # Test invalid unit names
    with pytest.raises(ValueError, match=re.escape('Invalid `unit` name.')):
        GM(np.array([1, 2, 3, 4, 5]), unit='test', dt=0.1)

    with pytest.raises(ValueError, match=r"use '/s/s' instead of 's\^2'"):
        GM(np.array([1, 2, 3, 4, 5]), unit='m/s^2', dt=0.1)


def test_differentiation() -> None:
    veloc = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [1, 3, 7, -1, -3, 5],
    ]).T
    gm = GM(veloc, unit='m', motion_type='veloc')
    accel_benchmark = np.array(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 20, 40, -80, -20, 80]],
    ).T
    assert np.allclose(gm.accel, accel_benchmark)


def test_integration__artificial_example() -> None:
    gm = GM(str(f_dir / 'two_column_data_example.txt'), unit='m/s/s')
    v_bench = np.array(
        [
            [0.1000, 0.1000],  # from MATLAB
            [0.2000, 0.3000],
            [0.3000, 0.6000],
            [0.4000, 1.0000],
            [0.5000, 1.5000],
            [0.6000, 1.7000],
            [0.7000, 2.0000],
            [0.8000, 2.4000],
            [0.9000, 2.9000],
            [1.0000, 3.5000],
            [1.1000, 3.8000],
            [1.2000, 4.2000],
            [1.3000, 4.7000],
            [1.4000, 5.3000],
            [1.5000, 6.0000],
        ],
    )
    u_bench = np.array(
        [
            [0.1000, 0.0100],  # from MATLAB
            [0.2000, 0.0400],
            [0.3000, 0.1000],
            [0.4000, 0.2000],
            [0.5000, 0.3500],
            [0.6000, 0.5200],
            [0.7000, 0.7200],
            [0.8000, 0.9600],
            [0.9000, 1.2500],
            [1.0000, 1.6000],
            [1.1000, 1.9800],
            [1.2000, 2.4000],
            [1.3000, 2.8700],
            [1.4000, 3.4000],
            [1.5000, 4.0000],
        ],
    )
    assert np.allclose(gm.veloc, v_bench)
    assert np.allclose(gm.displ, u_bench)


def test_integration__real_world_example() -> None:
    # Note: In this test, the result by cumulative trapezoidal numerical
    #       integration is used as the benchmark. Since it is infeasible to
    #       achieve perfect "alignment" between the two time histories,
    #       we check the correlation coefficient instead of element-wise
    #       check.
    veloc_ = np.genfromtxt(f_dir / 'sample_accel.txt')
    gm = GM(veloc_, unit='m/s', motion_type='veloc')
    displ = gm.displ[:, 1]
    displ_cumtrapz = np.append(
        0, sp.integrate.cumulative_trapezoid(veloc_[:, 1], dx=gm.dt)
    )
    r = np.corrcoef(displ_cumtrapz, displ)[1, 1]  # cross-correlation
    assert r >= 0.999


def test_fourier_transform() -> None:
    gm = GM(str(f_dir / 'two_column_data_example.txt'), unit='m/s/s')
    freq, spec = gm.get_Fourier_spectrum(real_val=False).raw_data.T

    freq_bench = [
        0.6667,
        1.3333,
        2.0000,
        2.6667,
        3.3333,
        4.0000,
        4.6667,
        5.3333,
    ]
    FS_bench = [
        60.0000 + 0.0000j,
        -1.5000 + 7.0569j,
        -1.5000 + 3.3691j,
        -7.5000 + 10.3229j,
        -1.5000 + 1.3506j,
        -1.5000 + 0.8660j,
        -7.5000 + 2.4369j,
        -1.5000 + 0.1577j,
    ]
    assert np.allclose(freq, freq_bench, atol=0.0001, rtol=0.0)
    assert np.allclose(spec, FS_bench, atol=0.0001, rtol=0.0)


def test_baseline_correction() -> None:
    gm = GM(str(f_dir / 'sample_accel.txt'), unit='m/s/s')
    corrected = gm.baseline_correct(show_fig=True)
    assert isinstance(corrected, GM)


def test_high_pass_filter() -> None:
    gm = GM(str(f_dir / 'sample_accel.txt'), unit='m')
    hp = gm.highpass(cutoff_freq=1.0, show_fig=True)
    assert isinstance(hp, GM)


def test_low_pass_filter() -> None:
    gm = GM(str(f_dir / 'sample_accel.txt'), unit='m')
    lp = gm.lowpass(cutoff_freq=1.0, show_fig=True)
    assert isinstance(lp, GM)


def test_band_pass_filter() -> None:
    gm = GM(str(f_dir / 'sample_accel.txt'), unit='m')
    bp = gm.bandpass(cutoff_freq=[0.5, 8], show_fig=True)
    assert isinstance(bp, GM)


def test_band_stop_filter() -> None:
    gm = GM(str(f_dir / 'sample_accel.txt'), unit='m')
    bs = gm.bandstop(cutoff_freq=[0.5, 8], show_fig=True)
    assert isinstance(bs, GM)


def test_amplify_via_profile() -> None:
    gm = GM(str(f_dir / 'sample_accel.txt'), unit='m')
    vs_prof = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    output_motion = gm.amplify(vs_prof, boundary='elastic')
    assert isinstance(output_motion, GM)


def test_deconvolution() -> None:
    # Assert `deconvolve()` & `amplify()` are reverse operations to each
    # other.
    gm = GM(str(f_dir / 'sample_accel.txt'), unit='m')
    vs_prof = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))

    for boundary in ['elastic', 'rigid']:
        deconv_motion = gm.deconvolve(vs_prof, boundary=boundary)
        output_motion = deconv_motion.amplify(vs_prof, boundary=boundary)
        assert nearly_identical(gm.accel, output_motion.accel)

        amplified_motion = gm.amplify(vs_prof, boundary=boundary)
        output_motion = amplified_motion.deconvolve(vs_prof, boundary=boundary)
        assert nearly_identical(gm.accel, output_motion.accel)


def test_plot() -> None:
    filename = str(f_dir / 'sample_accel.txt')
    gm = GM(filename, unit='m')

    _, axes = gm.plot()  # automatically generate fig/ax objects
    assert isinstance(axes, tuple)
    assert len(axes) == 3
    assert axes[0].title.get_text() == os.path.split(filename)[1]

    fig2 = plt.figure(figsize=(8, 8))
    fig2_, axes = gm.plot(fig=fig2)  # feed an external figure object
    assert np.allclose(fig2_.get_size_inches(), (8, 8))


def test_unit_convert() -> None:
    data = np.array([1, 3, 7, -2, -10, 0])
    gm = GM(data, unit='m', dt=0.1)
    accel = gm.accel[:, 1]
    accel_in_m = gm._unit_convert(unit='m/s/s')[:, 1]
    accel_in_gal = gm._unit_convert(unit='gal')[:, 1]
    accel_in_g = gm._unit_convert(unit='g')[:, 1]
    assert np.allclose(accel_in_m, accel)
    assert np.allclose(accel_in_gal, accel * 100)
    assert np.allclose(accel_in_g, accel / 9.81)


def test_scale_motion() -> None:
    data = np.array([1, 3, 7, -2, -10, 0])
    gm = GM(data, unit='g', dt=0.1)
    gm_scaled_1 = gm.scale_motion(factor=2.0)  # scale by 2.0
    gm_scaled_2 = gm.scale_motion(target_PGA_in_g=5.0)  # scale by 0.5
    assert np.allclose(gm.accel[:, 1] * 2, gm_scaled_1.accel[:, 1])
    assert np.allclose(gm.accel[:, 1] * 0.5, gm_scaled_2.accel[:, 1])


def test_amplify_by_tf__case_1_an_artificial_transfer_function() -> None:
    gm = GM(str(f_dir / 'sample_accel.txt'), unit='gal')
    ratio_benchmark = 2.76
    freq = np.arange(0.01, 50, step=0.01)
    tf = ratio_benchmark * np.ones_like(freq)
    transfer_function = Frequency_Spectrum(np.column_stack((freq, tf)))
    new_gm = gm.amplify_by_tf(transfer_function, show_fig=False)[0]
    ratio = new_gm.accel[:, 1] / gm.accel[:, 1]
    assert np.allclose(ratio, ratio_benchmark)


def test_amplify_by_tf__case_2_a_transfer_function_from_a_Vs_profile() -> None:
    gm = GM(str(f_dir / 'sample_accel.txt'), unit='gal')
    vs_prof = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    tf_RO, tf_BH, _ = vs_prof.get_transfer_function()
    gm_with_tf_RO = gm.amplify_by_tf(tf_RO)[0]
    gm_with_tf_BH = gm.amplify_by_tf(tf_BH)[0]

    gm_with_tf_RO_ = gm.amplify(vs_prof, boundary='elastic')
    gm_with_tf_BH_ = gm.amplify(vs_prof, boundary='rigid')

    # Assert that `amplify_by_tf()` and `amplify()` can generate
    # nearly identical results
    assert nearly_identical(gm_with_tf_RO.accel, gm_with_tf_RO_.accel)
    assert nearly_identical(gm_with_tf_BH.accel, gm_with_tf_BH_.accel)


def nearly_identical(
        motion_1: np.ndarray, motion_2: np.ndarray, thres: float = 0.99
) -> bool:
    """
    Assert that two ground motions are nearly identical, by checking the
    correlation coefficient between two time series.

    Parameters
    ----------
    motion_1 : np.ndarray
        Two-column array (time, acceleration).
    motion_2 : np.ndarray
        Two-column array (time, acceleration).
    thres : float, default=0.99
        The threshold that the correlation coefficient must be above (or equal
        to).

    Returns
    -------
    result : bool
        Whether the motions are nearly identical
    """
    if not np.allclose(motion_1[:, 0], motion_2[:, 0], rtol=0.001, atol=0.0):
        return False

    r = np.corrcoef(motion_1[:, 1], motion_2[:, 1])
    return not r[1, 0] < thres

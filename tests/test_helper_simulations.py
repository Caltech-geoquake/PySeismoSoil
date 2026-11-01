from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import PySeismoSoil.helper_simulations as sim
import PySeismoSoil.helper_site_response as sr
from PySeismoSoil.class_curves import Multiple_GGmax_Damping_Curves
from PySeismoSoil.class_parameters import HH_Param_Multi_Layer
from PySeismoSoil.class_Vs_profile import Vs_Profile

f_dir = Path(__file__).resolve().parent / 'files'


def test_check_layer_count() -> None:
    # Case 1(a): normal case, with parameters
    vs_profile = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    HH_G = HH_Param_Multi_Layer(str(f_dir / 'HH_G_FKSH14.txt'))
    HH_x = HH_Param_Multi_Layer(str(f_dir / 'HH_X_FKSH14.txt'))
    sim.check_layer_count(vs_profile, G_param=HH_G, xi_param=HH_x)

    # Case 1(b): normal case, with curves
    curves_data = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    mgdc = Multiple_GGmax_Damping_Curves(data=curves_data)
    sim.check_layer_count(vs_profile, GGmax_and_damping_curves=mgdc)

    # Case 2(a): abnormal case, with parameters
    del HH_G[-1]
    with pytest.raises(ValueError, match='Not enough sets of parameters'):
        sim.check_layer_count(vs_profile, G_param=HH_G)

    # Case 2(b): abnormal case, with curves
    curves_data_ = curves_data[:, :-4]
    mgdc_ = Multiple_GGmax_Damping_Curves(data=curves_data_)
    with pytest.raises(ValueError, match='Not enough sets of curves'):
        sim.check_layer_count(vs_profile, GGmax_and_damping_curves=mgdc_)


@pytest.mark.parametrize(
    ('boundary', 'min_correlation'),
    [
        pytest.param('elastic', 0.99, id='elastic'),
        # rigid cases can lead to higher errors
        pytest.param('rigid', 0.97, id='rigid'),
    ],
)
def test_linear(boundary: str, min_correlation: float) -> None:
    """
    Test that ``helper_simulations.linear()`` produces identical results to
    ``helper_site_response.linear_site_resp()``.
    """
    vs_profile = np.genfromtxt(f_dir / 'profile_FKSH14.txt')
    accel_in = np.genfromtxt(f_dir / 'sample_accel.txt')

    result = sim.linear(vs_profile, accel_in, boundary=boundary)[3]
    result_ = sr.linear_site_resp(vs_profile, accel_in, boundary=boundary)[0]

    # Time arrays need to match well
    assert np.allclose(result[:, 0], result_[:, 0], rtol=0.0001, atol=0.0)

    # Only check correlation (more lenient). Because `sim.linear()`
    # re-discretizes soil profiles into finer layers, so numerical errors
    # may accumulate over the additional layers.
    r = np.corrcoef(result[:, 1], result_[:, 1])
    assert r[0, 1] >= min_correlation

    plt.figure()
    plt.plot(result[:, 0], result[:, 1], label='every layer', alpha=0.6)
    plt.plot(result_[:, 0], result_[:, 1], label='surface only', alpha=0.6)
    plt.legend()
    plt.xlabel('Time [sec]')
    plt.ylabel('Acceleration')
    plt.grid(ls=':', lw=0.5)
    plt.title(f'{boundary.capitalize()} boundary')


def _scaled_motion(pga: float) -> np.ndarray:
    accel = np.genfromtxt(f_dir / 'sample_accel.txt')[:2000]
    accel[:, 1] *= pga / np.max(np.abs(accel[:, 1]))
    return accel


@pytest.mark.parametrize('boundary', ['elastic', 'rigid'])
def test_equiv_linear_freq_dependent__weak_motion_is_linear(
        boundary: str,
) -> None:
    profile = np.genfromtxt(f_dir / 'profile_FKSH14.txt')
    curve = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    motion = _scaled_motion(0.001)
    linear = sim.linear(profile, motion, boundary=boundary)
    fd_eql = sim.equiv_linear_freq_dependent(
        profile, motion, curve, boundary=boundary, verbose=False
    )
    assert len(fd_eql) == 11
    surface_lin = linear[3][:, 1]
    surface_eql = fd_eql[3][:, 1]
    assert surface_eql.shape == surface_lin.shape
    assert np.corrcoef(surface_lin, surface_eql)[0, 1] > 0.999
    assert np.isclose(
        np.max(np.abs(surface_eql)), np.max(np.abs(surface_lin)), rtol=0.02
    )


def test_equiv_linear_freq_dependent__strong_motion() -> None:
    profile = np.genfromtxt(f_dir / 'profile_FKSH14.txt')
    curve = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
    motion = _scaled_motion(3.0)
    eql = sim.equiv_linear(profile, motion, curve, verbose=False)
    fd_eql = sim.equiv_linear_freq_dependent(
        profile, motion, curve, verbose=False
    )
    for result in fd_eql:
        assert np.all(np.isfinite(result))

    # The frequency-dependent method keeps more high-frequency amplification
    # than the classic equivalent linear method
    def high_freq_tf(results: tuple) -> float:
        freq, tf = results[1], np.abs(results[2])
        return float(np.mean(tf[(freq > 10) & (freq < 20)]))

    assert high_freq_tf(fd_eql) > high_freq_tf(eql)
    # ... and the peak strains are of the same order of magnitude
    ratio = fd_eql[10][:, 1] / eql[10][:, 1]
    assert np.all((ratio > 0.1) & (ratio < 10))


def test_ka02_strain_shape() -> None:
    freq = np.linspace(0, 25, 1001)
    spectrum = np.exp(-freq / 3.0) + 1e-3
    shape = sim._ka02_strain_shape(freq, spectrum)
    assert shape.shape == freq.shape
    assert np.all(shape <= 1)
    assert np.all(shape > 0)
    assert np.isclose(shape[1], 1.0)  # flat below the average frequency
    assert shape[-1] < 0.1  # decays above it
    assert np.all(np.diff(shape[freq > 5]) <= 1e-12)
    assert np.all(sim._ka02_strain_shape(freq, np.zeros_like(freq)) == 1)

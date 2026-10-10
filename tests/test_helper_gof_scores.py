from pathlib import Path

import numpy as np
import pytest

import PySeismoSoil.helper_gof_scores as gof

f_dir = Path(__file__).resolve().parent / 'files'


def test_calc_AriasIntensity__constant_accel() -> None:
    # Ia(t) = pi / (2g) * integral of a^2 dt, so for a = 1 m/s/s lasting
    # 1 second, the peak Arias intensity is pi / (2g)
    t = np.linspace(0, 1, 101)
    accel = np.column_stack((t, np.ones_like(t)))
    Ia, Ia_peak = gof.calc_AriasIntensity(accel)

    assert isinstance(Ia_peak, float)
    assert Ia_peak == pytest.approx(np.pi / (2 * 9.81), abs=1e-7)
    assert Ia.shape == (101, 2)
    assert np.allclose(Ia[:, 0], t)
    assert Ia[-1, 1] == pytest.approx(Ia_peak, abs=1e-7)


def test_d_89__default_fmin_and_fmax() -> None:
    meas = np.genfromtxt(f_dir / 'sample_accel.txt')
    d8, d9 = gof.d_89(meas, meas.copy())  # fmin and fmax are None
    assert d8 == pytest.approx(0.0, abs=1e-7)
    assert d9 == pytest.approx(0.0, abs=1e-7)


def test_d_10__default_fmin_and_fmax() -> None:
    meas = np.genfromtxt(f_dir / 'sample_accel.txt')
    d10 = gof.d_10(meas, meas.copy())  # fmin and fmax are None
    assert d10 == pytest.approx(0.1, abs=1e-7)

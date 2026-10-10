from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import PySeismoSoil.helper_generic as hlp
import PySeismoSoil.helper_signal_processing as sig

f_dir = Path(__file__).resolve().parent / 'files'


def test_fourier_transform() -> None:
    accel, _ = hlp.read_two_column_stuff(
        str(f_dir / 'two_column_data_example.txt'),
    )
    freq, FS = sig.fourier_transform(accel, real_val=False).T

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
    assert np.allclose(FS, FS_bench, atol=0.0001, rtol=0.0)


def test_calc_transfer_function() -> None:
    input_accel = np.genfromtxt(f_dir / 'sample_accel.txt')
    output_accel = input_accel.copy()
    output_accel[:, 1] *= 2.3
    transfer_func = sig.calc_transfer_function(input_accel, output_accel)
    assert np.allclose(transfer_func[:, 1], 2.3)


def test_lin_smooth() -> None:
    raw_signal = sig.fourier_transform(
        np.genfromtxt(f_dir / 'sample_accel.txt'),
    )
    freq = raw_signal[:, 0]
    log_smoothed = sig.log_smooth(raw_signal[:, 1], lin_space=False)
    lin_smoothed = sig.lin_smooth(raw_signal[:, 1])

    alpha = 0.75
    plt.figure()
    plt.semilogx(freq, raw_signal[:, 1], alpha=alpha, label='raw')
    plt.semilogx(freq, lin_smoothed, alpha=alpha, label='lin smoothed')
    plt.semilogx(freq, log_smoothed, alpha=alpha, label='log smoothed')
    plt.grid(ls=':')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Signal value')
    plt.legend(loc='best')


def test_sine_smooth__constant_spectrum() -> None:
    freq = np.linspace(0.1, 50, 2000)
    spectrum = np.column_stack((freq, np.full_like(freq, 2.0)))
    smoothed = sig.sine_smooth(spectrum)

    assert smoothed.shape == freq.shape

    # A constant spectrum stays constant everywhere, including at both
    # ends (where the smoothing window is folded back in by mirroring)
    assert np.allclose(smoothed, smoothed[0])

    assert np.allclose(smoothed, 2.0, rtol=0.01)

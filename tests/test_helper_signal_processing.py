from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

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


@pytest.mark.parametrize(
    'df',
    [
        pytest.param(0.2, id='window_spans_1_bin_each_side'),
        pytest.param(1.0, id='window_spans_0_bins_each_side'),
    ],
)
def test_sine_smooth__long_coarse_spectrum(df: float) -> None:
    # With a frequency spacing this coarse (relative to the default window
    # span of 0.3 Hz), a spectrum of more than about 4490 bins used to raise
    # IndexError
    freq = df * np.arange(1, 5001)
    spectrum = np.column_stack((freq, np.full_like(freq, 2.0)))
    smoothed = sig.sine_smooth(spectrum)

    assert smoothed.shape == freq.shape
    assert np.all(np.isfinite(smoothed))
    assert np.allclose(smoothed, smoothed[0])


def test_sine_smooth__sample_accel() -> None:
    # Values from the implementation before #44 (which used loops instead of a
    # convolution). The window spans 14 bins on each side here, so the parts
    # beyond the first and the last bins are folded back in to bins 1-14 and
    # 2192-2205. The values are checked at the ends of these ranges, at the
    # bins next to them, and in the middle.
    accel = np.genfromtxt(f_dir / 'sample_accel.txt')
    smoothed = sig.sine_smooth(sig.fourier_transform(accel))
    expected = {
        0: 3091.3729624401317,
        1: 3213.388694077549,
        14: 8961.687053541178,
        15: 8949.681863763217,
        1000: 356.9908478511785,
        2191: 17.082260661834315,
        2192: 16.739084914173507,
        2205: 13.245827994033528,
        2206: 13.210226204992246,
    }

    assert smoothed.shape == (2207,)
    assert np.allclose(
        smoothed[list(expected)],
        list(expected.values()),
        rtol=1e-10,
        atol=0.0,
    )


def test_sine_smooth__window_wider_than_spectrum() -> None:
    # The window spans 6 bins on each side, more than the 5 bins of the
    # spectrum, so the parts of the window that are folded back in at one end
    # reach beyond the other end (and are dropped). Values from the
    # implementation before #44.
    freq = 0.05 * np.arange(1, 6)
    spectrum = np.column_stack((freq, [1.0, 2.0, 3.0, 4.0, 5.0]))
    smoothed = sig.sine_smooth(spectrum)
    expected = [
        2.5008400082948876,
        2.7300851645515656,
        3.2311812398099264,
        3.669964679436411,
        3.8393909391275804,
    ]

    assert np.allclose(smoothed, expected, rtol=1e-10, atol=0.0)

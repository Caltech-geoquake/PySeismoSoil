import unittest
import numpy as np
import matplotlib.pyplot as plt

import PySeismoSoil.helper_generic as hlp
import PySeismoSoil.helper_signal_processing as sig

import os
from os.path import join as _join


f_dir = _join(os.path.dirname(os.path.realpath(__file__)), 'files')


class Test_Helper_Signal_Processing(unittest.TestCase):
    def test_fourier_transform(self):
        accel, _ = hlp.read_two_column_stuff(
            _join(f_dir, 'two_column_data_example.txt'),
        )
        freq, FS = sig.fourier_transform(accel, real_val=False).T

        freq_bench = [0.6667, 1.3333, 2.0000, 2.6667, 3.3333, 4.0000, 4.6667, 5.3333]
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

        self.assertTrue(np.allclose(freq, freq_bench, atol=0.0001, rtol=0.0))
        self.assertTrue(np.allclose(FS, FS_bench, atol=0.0001, rtol=0.0))

    def test_calc_transfer_function(self):
        input_accel = np.genfromtxt(_join(f_dir, 'sample_accel.txt'))
        output_accel = input_accel.copy()
        output_accel[:, 1] *= 2.3
        transfer_func = sig.calc_transfer_function(input_accel, output_accel)
        self.assertTrue(np.allclose(transfer_func[:, 1], 2.3))

    def test_lin_smooth(self):
        raw_signal = sig.fourier_transform(
            np.genfromtxt(_join(f_dir, 'sample_accel.txt')),
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


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Helper_Signal_Processing)
    unittest.TextTestRunner(verbosity=2).run(SUITE)

# Author: Jian Shi

import unittest
import numpy as np

import PySeismoSoil.helper_generic as hlp
import PySeismoSoil.helper_signal_processing as sig

class Test_Helper_Signal_Processing(unittest.TestCase):
    '''
    Unit test for helper functions in helper_signal_processing.py
    '''

    def test_fourier_transform(self):

        accel, _ = hlp.read_two_column_stuff('two_column_data_example.txt')
        freq, FS = sig.fourier_transform(accel, real_val=False).T

        freq_bench = [0.6667, 1.3333, 2.0000, 2.6667, 3.3333, 4.0000, 4.6667,
                      5.3333]
        FS_bench = [60.0000 + 0.0000j, -1.5000 + 7.0569j, -1.5000 + 3.3691j,
                    -7.5000 +10.3229j, -1.5000 + 1.3506j, -1.5000 + 0.8660j,
                    -7.5000 + 2.4369j,   -1.5000 + 0.1577j]

        self.assertTrue(np.allclose(freq, freq_bench, atol=0.0001))
        self.assertTrue(np.allclose(FS, FS_bench, atol=0.0001))


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Helper_Signal_Processing)
    unittest.TextTestRunner(verbosity=2).run(SUITE)

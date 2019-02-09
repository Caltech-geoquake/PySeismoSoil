# Author: Jian Shi

import unittest
import numpy as np

import PySeismoSoil.helper_generic as hlp

from PySeismoSoil.class_frequency_spectrum import Frequency_Spectrum as FS

class Test_Class_Frequency_Spectrum(unittest.TestCase):
    '''
    Unit test for Frequency_Spectrum class
    '''

    def test_load_data(self):

        txt_filename = 'two_column_data_example.txt'

        fs_bench, df_bench = hlp.read_two_column_stuff(txt_filename)
        fs = FS(txt_filename, fmin=0.1, fmax=2.5, n_pts=20, log_scale=False)

        self.assertAlmostEqual(fs.raw_df, df_bench)
        self.assertTrue(np.allclose(fs.raw_data_2col, fs_bench))
        self.assertAlmostEqual(fs.spectrum_1col[0], 1)
        self.assertAlmostEqual(fs.spectrum_1col[-1], 7)

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Frequency_Spectrum)
    unittest.TextTestRunner(verbosity=2).run(SUITE)

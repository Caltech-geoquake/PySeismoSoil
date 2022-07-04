import unittest
import numpy as np

import PySeismoSoil.helper_generic as hlp
from PySeismoSoil.class_frequency_spectrum import Frequency_Spectrum as FS

import os
from os.path import join as _join


f_dir = _join(os.path.dirname(os.path.realpath(__file__)), 'files')


class Test_Class_Frequency_Spectrum(unittest.TestCase):
    def test_load_data(self):
        txt_filename = _join(f_dir, 'two_column_data_example.txt')

        fs_bench, df_bench = hlp.read_two_column_stuff(txt_filename)
        fs = FS(txt_filename, fmin=0.1, fmax=2.5, n_pts=20, log_scale=False)

        self.assertAlmostEqual(fs.raw_df, df_bench)
        self.assertTrue(np.allclose(fs.raw_data, fs_bench))
        self.assertAlmostEqual(fs.spectrum[0], 1)
        self.assertAlmostEqual(fs.spectrum[-1], 7)

    def test_plot(self):
        txt_filename = _join(f_dir, 'two_column_data_example.txt')
        fs = FS(txt_filename, fmin=0.1, fmax=2.5, n_pts=20, log_scale=False)
        fig, ax = fs.plot()
        self.assertEqual(ax.title.get_text(), os.path.split(txt_filename)[1])


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Frequency_Spectrum)
    unittest.TextTestRunner(verbosity=2).run(SUITE)

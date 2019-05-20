# Author: Jian Shi

import os
import unittest
import numpy as np
import matplotlib.pyplot as plt

from PySeismoSoil.class_ground_motion import Ground_Motion as GM
from PySeismoSoil.class_Vs_profile import Vs_Profile

class Test_Class_Ground_Motion(unittest.TestCase):
    '''
    Unit test for Ground_Motion class
    '''

    def test_loading_data(self):
        # Two columns from file
        gm = GM('./files/sample_accel.txt', unit='gal')

        PGA_benchmark = 294.30  # unit: cm/s/s
        PGV_benchmark = 31.46   # unit: cm/s
        PGD_benchmark = 38.77   # unit: cm
        tol = 1e-2

        self.assertAlmostEqual(gm.pga_in_gal, PGA_benchmark, delta=tol)
        self.assertAlmostEqual(gm.pgv_in_cm_s, PGV_benchmark, delta=tol)
        self.assertAlmostEqual(gm.pgd_in_cm, PGD_benchmark, delta=tol)
        self.assertAlmostEqual(gm.peak_Arias_Intensity, 1.524, delta=tol)
        self.assertAlmostEqual(gm.rms_accel, 0.4645, delta=tol)

        # Two columns from numpy array
        gm = GM(np.array([[0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4]]).T, unit='m/s/s')
        self.assertAlmostEqual(gm.pga, 4)

        # One column from file
        gm = GM('./files/one_column_data_example.txt', unit='g', dt=0.2)
        self.assertAlmostEqual(gm.pga_in_g, 12.0)

        # One column from numpy array
        gm = GM(np.array([1, 2, 3, 4, 5]), unit='gal', dt=0.1)
        self.assertAlmostEqual(gm.pga_in_gal, 5.0)

        # One column without specifying dt
        error_msg = 'is needed for one-column `data`.'
        with self.assertRaisesRegex(ValueError, error_msg):
            gm = GM(np.array([1, 2, 3, 4, 5]), unit='gal')

    def test_fourier_transform(self):
        gm = GM('./files/two_column_data_example.txt', unit='m/s/s')
        freq, spec = gm.get_Fourier_spectrum(real_val=False).raw_data.T

        freq_bench = [0.6667, 1.3333, 2.0000, 2.6667, 3.3333, 4.0000, 4.6667,
                      5.3333]
        FS_bench = [60.0000 + 0.0000j, -1.5000 + 7.0569j, -1.5000 + 3.3691j,
                    -7.5000 +10.3229j, -1.5000 + 1.3506j, -1.5000 + 0.8660j,
                    -7.5000 + 2.4369j, -1.5000 + 0.1577j]
        self.assertTrue(np.allclose(freq, freq_bench, atol=0.0001))
        self.assertTrue(np.allclose(spec, FS_bench, atol=0.0001))

    def test_baseline_correction(self):
        gm = GM('./files/sample_accel.txt', unit='m/s/s')
        corrected = gm.baseline_correct(show_fig=True)
        self.assertTrue(isinstance(corrected, GM))

    def test_filters(self):
        gm = GM('./files/sample_accel.txt', unit='m')
        hp = gm.highpass(cutoff_freq=1.0, show_fig=True)
        lp = gm.lowpass(cutoff_freq=1.0, show_fig=True)
        bp = gm.bandpass(cutoff_freq=[0.5, 8], show_fig=True)
        bs = gm.bandstop(cutoff_freq=[0.5, 8], show_fig=True)
        self.assertTrue(isinstance(hp, GM))
        self.assertTrue(isinstance(lp, GM))
        self.assertTrue(isinstance(bp, GM))
        self.assertTrue(isinstance(bs, GM))

    def test_deconvolution(self):
        gm = GM('./files/sample_accel.txt', unit='m')
        vs_prof = Vs_Profile('./files/profile_FKSH14.txt')
        deconv_motion = gm.deconvolve(vs_prof, show_fig=True)
        self.assertTrue(isinstance(deconv_motion, GM))

    def test_num_integration(self):
        gm = GM('./files/two_column_data_example.txt', unit='m/s/s')
        v_bench = np.array([[0.1000, 0.1000],
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
                            [1.5000, 6.0000]])
        u_bench = np.array([[0.1000, 0.0100],
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
                            [1.5000, 4.0000]])
        self.assertTrue(np.allclose(gm.veloc, v_bench))
        self.assertTrue(np.allclose(gm.displ, u_bench))

    def test_plot(self):
        filename = './files/sample_accel.txt'
        gm = GM(filename, unit='m')

        fig, axes = gm.plot()  # automatically generate fig/ax objects
        self.assertTrue(isinstance(axes, tuple))
        self.assertEqual(len(axes), 3)
        self.assertEqual(axes[0].title.get_text(), os.path.split(filename)[1])

        fig2 = plt.figure(figsize=(8, 8))
        fig2_, axes = gm.plot(fig=fig2)  # feed an external figure object
        self.assertTrue(np.allclose(fig2_.get_size_inches(), (8, 8)))

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Ground_Motion)
    unittest.TextTestRunner(verbosity=2).run(SUITE)


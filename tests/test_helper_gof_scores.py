import os
import unittest
from os.path import join as _join

import numpy as np

import PySeismoSoil.helper_gof_scores as gof

f_dir = _join(os.path.dirname(os.path.realpath(__file__)), 'files')


class Test_Helper_GOF_Scores(unittest.TestCase):
    def test_calc_AriasIntensity__constant_accel(self):
        # Ia(t) = pi / (2g) * integral of a^2 dt, so for a = 1 m/s/s lasting
        # 1 second, the peak Arias intensity is pi / (2g)
        t = np.linspace(0, 1, 101)
        accel = np.column_stack((t, np.ones_like(t)))
        Ia, Ia_peak = gof.calc_AriasIntensity(accel)

        self.assertIsInstance(Ia_peak, float)
        self.assertAlmostEqual(Ia_peak, np.pi / (2 * 9.81))
        self.assertEqual(Ia.shape, (101, 2))
        self.assertTrue(np.allclose(Ia[:, 0], t))
        self.assertAlmostEqual(Ia[-1, 1], Ia_peak)

    def test_d_89__default_fmin_and_fmax(self):
        meas = np.genfromtxt(_join(f_dir, 'sample_accel.txt'))
        d8, d9 = gof.d_89(meas, meas.copy())  # fmin and fmax are None
        self.assertAlmostEqual(d8, 0.0)
        self.assertAlmostEqual(d9, 0.0)

    def test_d_10__default_fmin_and_fmax(self):
        meas = np.genfromtxt(_join(f_dir, 'sample_accel.txt'))
        d10 = gof.d_10(meas, meas.copy())  # fmin and fmax are None
        self.assertAlmostEqual(d10, 0.1)


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Helper_GOF_Scores)
    unittest.TextTestRunner(verbosity=2).run(SUITE)

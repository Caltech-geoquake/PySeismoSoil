import os
import unittest
from os.path import join as _join

import numpy as np
from scipy.special import erf

from PySeismoSoil.class_gof_scores import GOF_Scores

f_dir = _join(os.path.dirname(os.path.realpath(__file__)), 'files')

ALL_SCORES = {
    'score_arias': True,
    'score_rms': True,
    'score_spectra': True,
    'score_cross_correlation': True,
}


class Test_Class_GOF_Scores(unittest.TestCase):
    def setUp(self):
        self.meas = np.genfromtxt(_join(f_dir, 'sample_accel.txt'))

    def test_calc_scores__identical_signals(self):
        gof_scores = GOF_Scores(self.meas, self.meas.copy())
        scores = gof_scores.calc_scores(**ALL_SCORES, verbose=False)

        self.assertEqual(len(scores), 10)
        self.assertTrue(np.allclose(scores[:9], 0.0))  # no difference at all
        self.assertAlmostEqual(scores[9], 0.1)  # perfect cross correlation
        self.assertTrue(np.allclose(gof_scores.get_scores(), scores))

    def test_calc_scores__scaled_simulation(self):
        # Scaling the simulation by 0.8 scales the amplitude-based metrics by
        # 0.8 (and the Arias intensity / energy integral by 0.8^2), but does
        # not change the normalized time histories or the correlation.
        simu = np.column_stack((self.meas[:, 0], 0.8 * self.meas[:, 1]))
        gof_scores = GOF_Scores(self.meas, simu)
        scores = gof_scores.calc_scores(**ALL_SCORES, verbose=False)

        score_for_ratio_0p8 = 10 * erf(0.8 - 1)
        score_for_ratio_0p64 = 10 * erf(0.8**2 - 1)

        self.assertTrue(np.allclose(scores[0:2], 0.0, atol=1e-6))  # d1, d2
        self.assertTrue(
            np.allclose(scores[2:4], score_for_ratio_0p64, atol=1e-3)
        )  # d3, d4
        self.assertTrue(
            np.allclose(scores[4:9], score_for_ratio_0p8, atol=1e-2)
        )  # d5 to d9
        self.assertAlmostEqual(scores[9], 0.1)  # d10


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_GOF_Scores)
    unittest.TextTestRunner(verbosity=2).run(SUITE)

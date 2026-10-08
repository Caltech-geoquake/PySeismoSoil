import unittest
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

from PySeismoSoil.class_gof_scores import GOF_Scores

f_dir = Path(__file__).resolve().parent / 'files'

ALL_SCORES = {
    'score_arias': True,
    'score_rms': True,
    'score_spectra': True,
    'score_cross_correlation': True,
}


class Test_Class_GOF_Scores(unittest.TestCase):
    def setUp(self) -> None:
        self.meas = np.genfromtxt(f_dir / 'sample_accel.txt')

    def test_calc_scores__identical_signals(self) -> None:
        gof_scores = GOF_Scores(self.meas, self.meas.copy())
        scores = gof_scores.calc_scores(**ALL_SCORES, verbose=False)

        assert len(scores) == 10
        assert np.allclose(scores[:9], 0.0)  # no difference at all
        assert scores[9] == pytest.approx(
            0.1, abs=1e-7
        )  # perfect cross correlation
        assert np.allclose(gof_scores.get_scores(), scores)

    def test_calc_scores__scaled_simulation(self) -> None:
        # Scaling the simulation by 0.8 scales the amplitude-based metrics by
        # 0.8 (and the Arias intensity / energy integral by 0.8^2), but does
        # not change the normalized time histories or the correlation.
        simu = np.column_stack((self.meas[:, 0], 0.8 * self.meas[:, 1]))
        gof_scores = GOF_Scores(self.meas, simu)
        scores = gof_scores.calc_scores(**ALL_SCORES, verbose=False)

        score_for_ratio_0p8 = 10 * erf(0.8 - 1)
        score_for_ratio_0p64 = 10 * erf(0.8**2 - 1)

        assert np.allclose(scores[0:2], 0.0, atol=1e-06)  # d1, d2
        assert np.allclose(
            scores[2:4], score_for_ratio_0p64, atol=0.001
        )  # d3, d4
        assert np.allclose(
            scores[4:9], score_for_ratio_0p8, atol=0.01
        )  # d5 to d9
        assert scores[9] == pytest.approx(0.1, abs=1e-7)  # d10


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_GOF_Scores)
    unittest.TextTestRunner(verbosity=2).run(SUITE)

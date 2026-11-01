import multiprocessing.pool
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

from PySeismoSoil.class_gof_scores import Batch_GOF_Scores, GOF_Scores

f_dir = Path(__file__).resolve().parent / 'files'

ALL_SCORES = {
    'score_arias': True,
    'score_rms': True,
    'score_spectra': True,
    'score_cross_correlation': True,
}


@pytest.fixture
def meas() -> np.ndarray:
    return np.genfromtxt(f_dir / 'sample_accel.txt')


def test_calc_scores__identical_signals(meas: np.ndarray) -> None:
    gof_scores = GOF_Scores(meas, meas.copy())
    scores = gof_scores.calc_scores(**ALL_SCORES, verbose=False)

    assert len(scores) == 10
    assert np.allclose(scores[:9], 0.0)  # no difference at all
    assert scores[9] == pytest.approx(
        0.1, abs=1e-7
    )  # perfect cross correlation
    assert np.allclose(gof_scores.get_scores(), scores)


def test_calc_scores__scaled_simulation(meas: np.ndarray) -> None:
    # Scaling the simulation by 0.8 scales the amplitude-based metrics by
    # 0.8 (and the Arias intensity / energy integral by 0.8^2), but does
    # not change the normalized time histories or the correlation.
    simu = np.column_stack((meas[:, 0], 0.8 * meas[:, 1]))
    gof_scores = GOF_Scores(meas, simu)
    scores = gof_scores.calc_scores(**ALL_SCORES, verbose=False)

    score_for_ratio_0p8 = 10 * erf(0.8 - 1)
    score_for_ratio_0p64 = 10 * erf(0.8**2 - 1)

    assert np.allclose(scores[0:2], 0.0, atol=1e-6)  # d1, d2
    assert np.allclose(scores[2:4], score_for_ratio_0p64, atol=1e-3)  # d3, d4
    assert np.allclose(scores[4:9], score_for_ratio_0p8, atol=1e-2)  # d5 to d9
    assert scores[9] == pytest.approx(0.1, abs=1e-7)  # d10


def test_batch_gof_scores__parallel_closes_pool(
        meas: np.ndarray,
        created_pools: list[multiprocessing.pool.Pool],
) -> None:
    simu = np.column_stack((meas[:, 0], 0.8 * meas[:, 1]))
    batch = Batch_GOF_Scores([
        GOF_Scores(meas, meas.copy()),
        GOF_Scores(meas, simu),
    ])
    results = batch.run(parallel=True, n_cores=2, options=ALL_SCORES)

    assert len(created_pools) == 1
    created_pools[0].join()  # raises `ValueError` if the pool is still running

    # Same scores as without parallel computing
    results_serial = batch.run(options=ALL_SCORES)
    assert len(results) == 2
    for result, result_serial in zip(results, results_serial, strict=True):
        assert np.allclose(result.get_scores(), result_serial.get_scores())

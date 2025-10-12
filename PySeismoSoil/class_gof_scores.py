import itertools
import multiprocessing as mp
from typing import Any

import numpy as np

from PySeismoSoil import helper_generic as hlp
from PySeismoSoil import helper_gof_scores as gof


class GOF_Scores:
    """
    Class implementation of the goodness-of-fit scoring.

    Parameters
    ----------
    measurement : np.ndarray
        A 2D numpy array with 2 columns. The 0th column contains the time in
        seconds, and the 1st column contains an acceleration time series.
    simulation : np.ndarray
        A 2D numpy array with 2 columns. The 0th column contains the time in
        seconds, and the 1st column contains an acceleration time series.

    Attributes
    ----------
    measurement : np.ndarray
        Same as `measurement` parameter.
    simulation : np.ndarray
        Same as `simulation` parameter.
    scores : np.ndarray
        When `calc_scores()` has been run, holds the computed goodness-of-fit
        scores.

    Raises
    ------
    TypeError
        When measurement and simulation time series are not the same length.
    """

    measurement: np.ndarray
    simulation: np.ndarray
    scores: np.ndarray

    def __init__(
            self,
            measurement: np.ndarray,
            simulation: np.ndarray,
    ) -> None:
        hlp.check_two_column_format(measurement, name='`measurement`')
        hlp.check_two_column_format(simulation, name='`simulation`')

        if measurement.shape[0] != simulation.shape[0]:
            raise TypeError(
                'Length of measurement and simulation must be the same.'
            )

        self.measurement = measurement
        self.simulation = simulation

        self.scores = np.full(10, None, dtype=np.float32)

    def __str__(self) -> str:
        """Define a string representation of calculated scores."""
        sn = [
            'Normalized Arias Intensity (S1)',
            'Normalized Energy Integral (S2)',
            'Peak Arias Intensity (S3)',
            'Peak Energy Integral (S4)',
            'RMS Acceleration (S5)',
            'RMS Velocity (S6)',
            'RMS Displacement (S7)',
            'Spectral Acceleration (S8)',
            'Fourier Spectra (S9)',
            'Cross Correlation (S10)',
        ]

        sum = 0
        count = 0

        text = '\nGoodness of Fit Scores\n'
        text += '---------------------------------------\n'
        for ix, sc in enumerate(self.scores):
            if not np.isnan(sc):
                text += f'{sn[ix]:>31}: {sc: .3f}\n'
                sum += sc
                count += 1

        text += '---------------------------------------\n'
        text += f'Average Score: {sum / count:.3f}\n'

        return text

    def get_meas(self) -> np.ndarray:
        """
        Return two-column measurement array, where the first column is time.
        """
        return self.measurement

    def get_simu(self) -> np.ndarray:
        """
        Return two-column simulation array, where the first column is time.
        """
        return self.simulation

    def get_scores(self) -> np.ndarray:
        """
        Return entire score array, with 'None' for scores that haven't been
        calculated.
        """
        return self.scores

    def calc_scores(
            self,
            fmin: float | None = None,
            fmax: float | None = None,
            *,
            score_arias: bool = True,
            score_rms: bool = True,
            score_spectra: bool = True,
            score_cross_correlation: bool = False,
            baseline: bool = True,
            verbose: bool = False,
            show_fig: bool = False,
    ) -> np.ndarray:
        """
        Calculate the goodness-of-fit scores with the given measurement and
        simulation time series.

        Parameters
        ----------
        fmin : float | None
            Minimum frequency to be considered, in units of Hz. Default is
            (sampling frequency)/(length of time series).
        fmax : float | None
            Maximum frequency to be considered, in units of Hz. Default is
            (sampling frequency)/2.0.
        score_arias : bool
            Whether or not to compute the arias intensity and energy integral
            group of scores.
        score_rms : bool
            Whether or not to compute the RMS group of scores.
        score_spectra : bool
            Whether or not to compute the FAS and spectral acceleration group
            of scores.
        score_cross_correlation : bool
            Whether or not to compute the cross-correlation score. Only
            recommended if the measurement and simulation time series being
            compared are identical in start time.
        baseline : bool
            Whether or not to perform baseline correction of the time series.
        verbose : bool
            Whether or not to print output information.
        show_fig : bool
            Whether or not to plot.

        Returns
        -------
        scores : np.ndarray
            A vector containing the goodness-of-fit scores, in the following
            order: [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]. If the score
            hasn't been calculated, it will be omitted from the returned array.

        Raises
        ------
        ValueError
            If fmax is larger than fmin.
        """
        t = self.measurement[:, 0]
        n = self.measurement.shape[0]
        dt = t[1] - t[0]
        fs = 1.0 / dt  # sampling frequency [hz]

        if fmax is None:
            fmax = fs / 2.0

        if fmin is None:
            fmin = fs / n

        if fmin >= fmax:
            raise ValueError('fmax must be larger than fmin.')

        scores = np.array([])
        if score_arias:
            scores0 = gof.d_1234(
                self.measurement,
                self.simulation,
                fmin,
                fmax,
                baseline,
                show_fig,
            )
            for ind, sc in enumerate(scores0):
                scores = np.append(scores, sc)
                self.scores[0 + ind] = sc

        if score_rms:
            scores1 = gof.d_567(
                self.measurement,
                self.simulation,
                fmin,
                fmax,
                baseline,
                show_fig,
            )
            for ind, sc in enumerate(scores1):
                scores = np.append(scores, sc)
                self.scores[4 + ind] = sc

        if score_spectra:
            scores2 = gof.d_89(
                self.measurement,
                self.simulation,
                fmin,
                fmax,
                baseline,
                show_fig,
            )
            for ind, sc in enumerate(scores2):
                scores = np.append(scores, sc)
                self.scores[7 + ind] = sc

        if score_cross_correlation:
            scores3 = gof.d_10(
                self.measurement,
                self.simulation,
                fmin,
                fmax,
                baseline,
                show_fig,
            )
            scores = np.append(scores, scores3)
            self.scores[9] = scores3

        if verbose:
            sn = [
                'Normalized Arias Intensity (S1)',
                'Normalized Energy Integral (S2)',
                'Peak Arias Intensity (S3)',
                'Peak Energy Integral (S4)',
                'RMS Acceleration (S5)',
                'RMS Velocity (S6)',
                'RMS Displacement (S7)',
                'Spectral Acceleration (S8)',
                'Fourier Spectra (S9)',
                'Cross Correlation (S10)',
            ]

            print('Goodness of Fit Scores')
            print('---------------------------------------')
            ind = 0
            if score_arias:
                [
                    print(f'{sn[ix]:>31}: {s: .3f}')
                    for ix, s in enumerate(scores[ind : ind + 4])
                ]

                ind += 4

            if score_rms:
                [
                    print(f'{sn[ix + 4]:>31}: {s: .3f}')
                    for ix, s in enumerate(scores[ind : ind + 3])
                ]

                ind += 3

            if score_spectra:
                [
                    print(f'{sn[ix + 7]:>31}: {s: .3f}')
                    for ix, s in enumerate(scores[ind : ind + 2])
                ]

                ind += 2

            if score_cross_correlation:
                print(f'{sn[9]:>31}: {self.scores[9]: .3f}')

            print('---------------------------------------')
            print(f'Average Score: {np.mean(scores):.3f}')

        return scores


class Batch_GOF_Scores:
    """Run goodness-of-fit scoring in batch."""

    def __init__(self, list_of_scores: list[GOF_Scores]) -> None:
        if not isinstance(list_of_scores, list):
            raise TypeError('`list_of_gof` should be a list.')

        if len(list_of_scores) == 0:
            raise ValueError('`list_of_gof` should have at least one element.')

        n_scores = len(list_of_scores)

        self.list_of_scores = list_of_scores
        self.n_scores = n_scores

    def run(
            self,
            parallel: bool = False,
            n_cores: int | None = 1,
            options: dict[str, Any] | None = None,
    ) -> list[GOF_Scores]:
        """
        Run gof scoring in batch.

        Parameters
        ----------
        parallel : bool
            Whether to use multiple CPU cores to run simulations.
        n_cores : int | None
            Number of CPU cores to be used. If ``None``, all CPU cores will be
            used.
        options : dict[str, Any] | None
            Options to be passed to the ``run()`` method for the GoF scoring.

        Returns
        -------
        score_results : list[GOF_Scores]
            A list of the score objects containing computed scores.
        """
        options = {} if options is None else options

        N = self.n_scores

        score_results = []
        if not parallel:
            for i in range(self.n_scores):
                score_results.append(self._run_single_score([i, options]))
            # END FOR
        else:
            p = mp.Pool(n_cores)
            score_results = p.map(
                self._run_single_score,
                itertools.product(range(N), [options]),
            )
        # END IF

        return score_results

    def _run_single_score(self, all_params: list[Any]) -> GOF_Scores:
        """
        Run a single simulation.

        Parameters
        ----------
        all_params : list[Any]
            All the parameters needed for running the simulation. It should
            have the following structure: [i, [catch_errors, options]], where:
                - ``i`` is the index of the current simulation in the batch.
                - ``options``: same as in the ``run()`` method

        Returns
        -------
        score_result : GOF_Scores
            Score results for a single GOF score object.
        """
        i, options = all_params  # unpack

        score_obj = self.list_of_scores[i]
        _ = score_obj.calc_scores(**options)

        return score_obj

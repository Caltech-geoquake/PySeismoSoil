import numpy as np
from matplotlib import pyplot as plt

from PySeismoSoil import helper_generic as hlp
from PySeismoSoil import helper_gof_scores as gof

"""
TODO: Cross correlation score
TODO: Proper commenting and documentation
TODO: Batch gof score class
"""

class GOF_Scores:
    """
    Class implementation of the goodness-of-fit scoring.

    Parameters
    ----------
    measurement : np.ndarray
        A 2D numpy array with 2 columns. The 0th column contains the time
        in seconds, and the 1st column contains an acceleration time series.
    simulation : np.ndarray
        A 2D numpy array with 2 columns. The 0th column contains the time
        in seconds, and the 1st column contains an acceleration time series.

    Attributes
    ----------
    measurement : np.ndarray
        [TODO]
    simulation : np.ndarray
        [TODO]

    Raises
    ------
    [RAISES]
    """

    def __init__(
            self,
            measurement: np.ndarray,
            simulation: np.ndarray,
    ) -> None:
        hlp.check_two_column_format(measurement, name='`measurement`')
        hlp.check_two_column_format(simulation, name='`simulation`')

        if measurement.shape[0] != simulation.shape[0]:
            raise TypeError('Length of measurement and simulation must be the same.')

        self.measurement = measurement
        self.simulation = simulation

    def score(
        self,
        fmin: float | None = None,
        fmax: float | None = None,
        *,
        score_groups: tuple[bool, bool, bool, bool] = (True, True, True, False),
        verbose: bool = True,
        show_fig: bool = True,
    ) -> np.ndarray:
        """
        Calculates the goodness-of-fit scores with the given measurement
        and simulation time series.

        Parameters
        ----------
        fmin : float | None
            Minimum frequency to be considered, in units of Hz.
            Default is (sampling frequency)/(length of time series).
        fmax : float | None
            Maximumimum frequency to be considered, in units of Hz.
            Default is (sampling frequency)/2.0.
        show_fig : bool
            Whether or not to plot.

        Returns
        -------
        scores : np.ndarray
            A vector containing the goodness-of-fit scores, in the following order:
            [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]
        """
        t = self.measurement[:, 0]
        n = self.measurement.shape[0]
        dt = t[1]-t[0]
        fs = 1.0/dt # sampling frequency [hz]
        
        if fmax == None:
            fmax = fs/2.0
        if fmin == None:
            fmin = fs/n
            
        if fmin >= fmax:
            raise ValueError('fmax must be larger than fmin.')
        
        # TODO: one fig for each score group that is getting show_fig

        scores = np.array([])
        if score_groups[0]:
            scores0 = gof.d_1234(self.measurement,self.simulation,fmin,fmax,show_fig)
            for sc in scores0:
                scores = np.append(scores, sc)
        if score_groups[1]:
            scores1 = gof.d_567(self.measurement,self.simulation,fmin,fmax,show_fig)
            for sc in scores1:
                scores = np.append(scores, sc)
        if score_groups[2]:
            scores2 = gof.d_89(self.measurement, self.simulation,fmin,fmax,show_fig)
            for sc in scores2:
                scores = np.append(scores, sc)
        if score_groups[3]:
            # TODO: gof.d_10() for cross correlation
            0
        
        if verbose:
            sn = ['Normalized Arias Intensity', 
                  'Normalized Energy Integral',
                  'Peak Arias Intensity',
                  'Peak Energy Integral',
                  'RMS Acceleration',
                  'RMS Velocity',
                  'RMS Displacement',
                  'Spectral Acceleration',
                  'Fourier Spectra',
                  'Cross Correlation']

            print('Goodness of Fit Scores')
            print('----------------------------------')
            ind = 0
            if score_groups[0]:
                [print(f'{sn[ix]:>26}: {s: .3f}') for ix, s in enumerate(scores[ind:ind+4])]
                ind += 4
            if score_groups[1]:
                [print(f'{sn[ix+4]:>26}: {s: .3f}') for ix, s in enumerate(scores[ind:ind+3])]
                ind += 3
            if score_groups[2]:
                [print(f'{sn[ix+7]:>26}: {s: .3f}') for ix, s in enumerate(scores[ind:ind+2])]
                ind += 2
            if score_groups[3]:
                [print(f'{sn[ix+9]:>26}: {s: .3f}') for ix, s in enumerate(scores[ind])]
            print('----------------------------------')
            print(f'Average Score: {np.mean(scores):.3f}')
        
        return scores

"""
### pgain_12.273_vs30_475

Python vs Matlab:
c1: 0.8587 0.8587
c2: 0.8004 0.8014
c3: 1.3485 1.3485 
c4: 1.1072 1.1072
c5: 0.6681 0.6681
c6: 0.5420 0.5420
c7: 1.0253 0.8836 (updated python: 0.9595)
c8: 0.9545 1.0016
c9: -2.2074 -2.5694
avg score: 0.5664 0.5157

### pgain_15.026_vs30_650

Python vs Matlab:
c1: 0.9168 0.9168
c2: 0.9487 0.9487
c3: -2.6997 -2.6997
c4: -2.9303 -2.9303
c5: -1.4648 -1.4648
c6: -1.6042 -1.6042
c7: -1.2755 -1.1838 (updated python: -1.2591)
c8: -1.9881 -2.1110
c9: -4.2612 -4.5906
avg score: -1.5954 -1.6354
"""

class Batch_GOF_Scores:
    """
    Run goodness-of-fit scoring in batch.
    """
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
    fmin : float
        Minimum frequency to be considered, in units of Hz.
    fmax : float
        Maximumimum frequency to be considered, in units of Hz.

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
        raise Exception('Length of measurement and simulation must be the same.', 
        f'(measurement={measurement.shape[0]}, simulation={simulation.shape[0]})')

        self.measurement = measurement
        self.simulation = simulation
        self.values = values

    def score(
        self,
        fmin: float | None = None,
        fmax: float | None = None,
        show_fig: bool = False,
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
            raise Exception(f'fmax must be larger than fmin. (fmin={fmin}, fmax={fmax})')



def gofScores(measurement, simulation, 
              fmin=None, 
              fmax=None, 
              plot_options=[False, False, False, False], 
              baseline_options=[True, True],
              filter_order=4,
              exact_resp_spect=True):
    """
    USAGE: [scores,average,weighted_avg,fhs] = ...
    gofScores(measurement,simulation,fmin,fmax,...
    plot_options,baseline_options,filter_order,exact_resp_spect)
    Calculate goodness-of-fit scores, as defined in Shi & Asimaki (2017).
    [INPUTS]
       measurement: Measured time series (in two columns: time in sec,
                    acceleration)
       simulation: Simulated time series (in two columns, time and accel.)
       fmin: Minimum frequency to be considered (Hz)
       fmax: Maximum frequency to be considered (Hz)
       plot_option: Whether or not to plot comparison figures for measurement
                    and simulation. A 4-element vector, whose each element is
                    either 0 or 1, which corresponds to whether: (1) scores
                    1-4, (2) scores 5-7, (3) scores 8-9, and (4) final
                    summary, will be displayed respectively. Default is [0,
                    0, 0, 0]
       baseline_options: Whether or not to perform baseline correction for
                         scores 5-7 and scores 8-9. This should be a 
                         3-element vector. Default is [1, 1].
                         Please do not change this default unless you really
                         know what you are doing.
       filter_order: The order of band pass filter to be performed to both
                     measurement and simulation. Default is 4. Please do not
                     change this default unless you really know what you are
                     doing.
       exact_resp_spect: Whether or not to use exact response spectra
                         calculater (slower) as opposed to approximate
                         response spectra calculator (faster). Default: 'y'.
    [OUTPUTS]
       scores: A row vector whose 1st to 9th element are the 9 scores
       average: The plain average score of the 9 scores
       weighted_avg: Weighted average of the 9 scores, which combines S1 and
                     S2, S3 and S4 (respectively).
       fhs: figure handles of the four possible figures, i.e., [fh1234,fh567,fh89,fhstft]
    (c) Jian Shi, 7/21/2013
    [UPDATE LOG]
      In February 2015: A new scoring scheme (from -10 to 10) is used, which
                        eventually goes into Shi & Asimaki (2017).
      On 12/29/2015, "exact_resp_spect" option is added.
    """
    
    hlp.check_two_column_format(measurement, name='`measurement`')
    hlp.check_two_column_format(simulation, name='`simulation`')
    
    if measurement.shape[0] != simulation.shape[0]:
        raise Exception('Length of measurement and simulation must be the same.', 
        f'(measurement={measurement.shape[0]}, simulation={simulation.shape[0]})')
        
    t = measurement[:, 0]
    n = measurement.shape[0]
    dt = t[1]-t[0]
    fs = 1.0/dt # sampling frequency [hz]
    
    if fmax == None:
        fmax = fs/2.0
    if fmin == None:
        fmin = fs/n
        
    if fmin >= fmax:
        raise Exception(f'fmax must be larger than fmin. (fmin={fmin}, fmax={fmax})')
        
    c1,c2,c3,c4 = gof.d_1234(measurement,simulation,fmin,fmax,plot_options[0],filter_order)
    c5,c6,c7 = gof.d_567(measurement,simulation,fmin,fmax,plot_options[1],baseline_options[0],filter_order=4)
    c8, c9 = gof.d_89(measurement,simulation,fmin,fmax,plot_options[1],baseline_options[0])
    
    scores = (c1, c2, c3, c4, c5, c6, c7, c8, c9)
    
    [print(f'c{ix+1}: {s:.4f}') for ix, s in enumerate(scores)]
    print(f'avg score: {np.mean(scores):.4f}')
    
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
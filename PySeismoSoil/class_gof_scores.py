import numpy as np
import scipy
from matplotlib import pyplot as plt

from PySeismoSoil import helper_generic as hlp
from PySeismoSoil import helper_gof_scores as gof

# import pywt
# from modwt import modwt, modwtmra

import os

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
        raise Exception(f'Length of measurement and simulation must be the same. (measurement={measurement.shape[0]}, simulation={simulation.shape[0]})')
        
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

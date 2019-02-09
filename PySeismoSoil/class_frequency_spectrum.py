# Author: Jian Shi

import os
import numpy as np
import matplotlib.pyplot as plt

from . import helper_generic as hlp
from . import helper_site_response as sr
from . import helper_signal_processing as sig

class Frequency_Spectrum():
    '''
    Class implementation of a frequency spectrum object. The user-supplied
    frequency spectrum is internally interpolated onto a reference frequency
    array. (If frequency range implied in `data` and/or `df` goes beyond `fmin`
    and/or `fmax`, then the interpolation algorithm automatically use the 0th
    and the last point in the extrapolated parts.)

    Parameters
    ----------
    data : str or numpy.ndarray
        If str: the full file name on the hard drive containing the data.
        If np.ndarray: the numpy array containing the data.
        The data can have one column (which contains the motion) or two
        columns (0st column: freq; 1st column: spectrum). If only one column
        is supplied, another input parameter `df` must also be supplied.
    df : float
        Frequency interval. Not necessary if `data` has two columns (with the
        0th column being the frequency information). If `data` has one column,
        it is assumed that the values in `data` correspond to a linear
        frequency array.
    fmin : float
        Minimum frequency of the automatically constructed frequency array
        for interpolation
    fmax : float
        Maximum frequency of the automatically constructed frequency array
        for interpolation
    npts : int
        Number of points in the automatically constructed frequency array
        for interpolation
    log_scale : bool
        Whether the automatically constructed frequency array is in log scale
        (or linear scale)
    sep : str
        Delimiter identifier, only useful if `data` is a file name

    Attributes
    ----------
    raw_df : float
        Original frequency interval as entered
    raw_data_2col : numpy.array
        Raw frequency spectrum (before interpolation) that the user provided
    npts : int
        Same as the input parameter
    freq : numpy.array
        The reference frequency array for interpolation
    fmin : float
        Same as the input parameter
    fmax : float
        Same as the input parameter
    spectrum : numpy.array
        A two-column numpy array. The reference frequency array and the
        interpolated spectrum values
    spectrum_1col : numpy.array
        Just the interpolated spectrum values
    amplitude_1col : numpy.array
        The amplitude (or "magnitude") of `spectrum_1col`. Note that
        `spectrum_1col` can already be all real numbers.
    iscomplex : bool
        Is `spectrum_1col` complex or already real?
    '''

    def __init__(self, data, df=None, fmin=0.1, fmax=30, npts=1000,
                 log_scale=True, sep='\t'):

        data_, df = hlp.read_two_column_stuff(data, df, sep)
        if isinstance(data, str):  # is a file name
            self.__path_name, self.__file_name = os.path.split(data)

        if log_scale:
            ref_freq = np.logspace(np.log10(fmin), np.log10(fmax), npts)
        else:
            ref_freq = np.linspace(fmin, fmax, npts)

        ref_spect = np.interp(ref_freq, np.real_if_close(data_[:,0]), data_[:,1])

        self.raw_df = df
        self.raw_data_2col = data_
        self.npts = npts
        self.freq = ref_freq
        self.fmin = min(ref_freq)
        self.fmax = max(ref_freq)
        self.spectrum = np.column_stack((ref_freq, ref_spect))
        self.spectrum_1col = ref_spect
        self.amplitude_1col = np.abs(ref_spect)
        self.iscomplex = np.iscomplex(self.spectrum_1col).any()

    #--------------------------------------------------------------------------
    def __repr__(self):

        text = 'df = %.2f Hz, npts = %d, f_min = %.2f Hz, f_max = %.2f Hz' \
               % (self.df, self.npts, self.fmin, self.fmax)
        return text

    #--------------------------------------------------------------------------
    def plot(self, logx=True, logy=False, **kwargs_plot):
        '''
        Plot the shape of the interpolated spectrum.

        Parameters
        ----------
        logx : bool
            Whether to show x scale as log
        logy : bool
            Whether to show y scale as log
        **kwargs_plot :
            Extra keyword arguments are passed to matplotlib.pyplot.plot()

        Returns
        -------
        fig, ax :
            Objects of matplotlib figure and axes
        '''

        fig = plt.figure()
        ax = plt.axes()
        ax.plot(self.freq, self.amplitude_1col, **kwargs_plot)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        ax.grid(ls=':')
        if logx: ax.set_xscale('log')
        if logy: ax.set_yscale('log')
        if self.__file_name: ax.set_title(self.__file_name)

        return fig, ax

    #--------------------------------------------------------------------------
    def get_smoothed(self, win_len=15, inplace=False, show_fig=False, **kwargs):
        '''
        Smooth the spectrum by calculating the convolution of the raw
        signal and the smoothing window.

        Parameters
        ----------
        win_len : int
            Length of the smoothing window. Larget numbers means more smoothing.
        inplace : bool
            Whether to use the smoothed signal to replace the raw signal
            internally
        show_fig : bool
            Whether to show a before/after figure
        **kwargs :
            Extra keyword arguments get passed to the function
            helper_signal_processing.log_smooth()

        Returns
        -------
        sm : numpy.array (optional, only if `inplace`)
            The smoothed signal. 1D numpy array
        fig, ax :
            matplotlib objects of the figure and axes
        '''

        sm = sig.log_smooth(self.spectrum_1col, win_len=win_len, fmin=self.fmin,
                            fmax=self.fmax, **kwargs)

        if show_fig:
            fig = plt.figure()
            ax = plt.axes()
            ax.semilogx(self.freq, self.spectrum_1col, color='gray', label='original')
            ax.semilogx(self.freq, sm, color='m', label='smoothed')
            ax.grid(ls=':')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Spectrum')
            ax.legend(loc='best')
            if self.__file_name: ax.set_title(self.__file_name)

        if inplace:
            self.spectrum_1col = sm
            self.spectrum = np.column_stack((self.freq, sm))
            return fig, ax
        else:
            return sm, fig, ax

#%%----------------------------------------------------------------------------
class Amplification_Function(Frequency_Spectrum):
    '''
    Amplification function, which is the magnitude of a complex-valued transfer
    function.
    '''

    def get_f0(self):
        '''
        Get the "fundamental frequency" of an amplification function, which is
        the frequency of the first amplitude peak.

        Returns
        -------
        f0 : float
            The fundamental frequency
        '''
        return sr.find_f0(self.spectrum)

#%%----------------------------------------------------------------------------
class Transfer_Function(Frequency_Spectrum):
    '''
    Complex-valued transfer function.
    '''

    def get_amplitude(self):
        '''
        Returns
        -------
        amplitude : numpy.array
            2D numpy array with two columns. Amplitude spectrum with the
            accompanying frequency array
        '''
        return np.column_stack((self.freq, self.amplitude_1col))

    def get_phase(self, wrapped=True):
        '''
        Return the phase shift angle (unit: rad) of the transfer function.

        Parameter
        ---------
        wrapped : bool
            Whether to return the wrapped (i.e., between [-2 * pi, 2 * pi])
            phase angle or not. If False, numpy.unwrap() will be used to unwrap
            the phase.

        Returns
        -------
        phase : numpy.array
            2D numpy array with two columns. Phase angle with the accompanying
            frequency array.
        '''
        if not self.iscomplex:
            print('Warning: the frequency spectrum is not a complex array...')
        if wrapped:
            return np.column_stack((self.freq, np.angle(self.spectrum_1col)))
        else:
            return np.column_stack((self.freq,
                                    np.unwrap(np.angle(self.spectrum_1col))))




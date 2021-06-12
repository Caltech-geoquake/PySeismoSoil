import os
import numpy as np
import matplotlib.pyplot as plt

from . import helper_generic as hlp
from . import helper_site_response as sr
from . import helper_signal_processing as sig


class Frequency_Spectrum:
    """
    Class implementation of a frequency spectrum object. The user-supplied
    frequency spectrum is internally interpolated onto a reference frequency
    array. (If frequency range implied in ``data`` and/or ``df`` goes beyond ``fmin``
    and/or ``fmax``, then the interpolation algorithm automatically use the 0th
    and the last point in the extrapolated parts.)

    Parameters
    ----------
    data : str or numpy.ndarray
        If str: the full file name on the hard drive containing the data.
        If np.ndarray: the numpy array containing the data.
        The data can have one column (which contains the spectrum) or two
        columns (0st column: freq; 1st column: spectrum). If only one column
        is supplied, another input parameter ``df`` must also be supplied.
    df : float
        Frequency interval. Not necessary if ``data`` has two columns (with the
        0th column being the frequency information). If ``data`` has one column,
        it is assumed that the values in ``data`` correspond to a linear
        frequency array.
    interpolate : bool
        Whether to use the interpolated spectra in place of the raw data.
    fmin : float
        Minimum frequency of the manuall constructed frequency array for
        interpolation. It has no effect if ``interpolate`` is ``False``.
    fmax : float
        Maximum frequency of the manually constructed frequency array for
        interpolation. It has no effect if ``interpolate`` is ``False``.
    n_pts : int
        Number of points in the manualy constructed frequency array for
        interpolation. It has no effect if ``interpolate`` is ``False``.
    log_scale : bool
        Whether the manually constructed frequency (for interpolation) array
        is in log scale (or linear scale). It has no effect if ``interpolate``
        is False.
    sep : str
        Delimiter identifier, only useful if ``data`` is a file name.

    Attributes
    ----------
    raw_df : float
        Original frequency interval as entered.
    raw_data : numpy.ndarray
        Raw frequency spectrum (before interpolation) that the user provided.
    n_pts : int
        Same as the input parameter.
    freq : numpy.ndarray
        The reference frequency array for interpolation.
    fmin : float
        Same as the input parameter.
    fmax : float
        Same as the input parameter.
    spectrum_2col : numpy.ndarray
        A two-column numpy array (frequency and spectrum).
    spectrum : numpy.ndarray
        Just the spectrum values.
    amplitude : numpy.ndarray
        The amplitude (or "magnitude") of ``spectrum``. Note that
        ``spectrum`` can already be all real numbers.
    amplitude_2col: numpy.ndarray
        A two-column numpy array (frequency and amplitude).
    phase : numpy.ndarray
        The phase angle of ``spectrum``. If ``spectrum`` has all real values,
        ``phase`` is all zeros.
    phase_2col : numpy.ndarray
        A two-column numpy array (frequency and phase).
    iscomplex : bool
        Is ``spectrum`` complex or already real?
    """
    def __init__(
            self, data, *, df=None, interpolate=False, fmin=0.1, fmax=30,
            n_pts=1000, log_scale=True, sep='\t',
    ):
        data_, df = hlp.read_two_column_stuff(data, df, sep)
        if isinstance(data, str):  # is a file name
            self._path_name, self._file_name = os.path.split(data)
        else:
            self._path_name = None
            self._file_name = None

        if not interpolate:
            fmin = df
            n_pts = data_.shape[0]
            fmax = df * n_pts
            freq = np.real_if_close(data_[:, 0])
            spect = data_[:, 1]
        else:
            freq, spect = hlp.interpolate(
                fmin, fmax, n_pts, np.real_if_close(data_[:, 0]),
                data_[:, 1], log_scale=log_scale,
            )
        self.raw_df = df
        self.raw_data = data_
        self.n_pts = n_pts
        self.freq = freq
        self.fmin = min(freq)
        self.fmax = max(freq)
        self.spectrum_2col = np.column_stack((freq, spect))
        self.spectrum = spect
        self.amplitude = np.abs(spect)
        self.amplitude_2col = np.column_stack((freq, self.amplitude))
        self.phase = np.angle(spect)
        self.phase_2col = np.column_stack((freq, self.phase))
        self.iscomplex = np.iscomplex(self.spectrum).any()

    def __repr__(self):
        text = 'df = %.2f Hz, n_pts = %d, f_min = %.2f Hz, f_max = %.2f Hz' \
               % (self.raw_df, self.n_pts, self.fmin, self.fmax)
        return text

    def plot(
            self, fig=None, ax=None, figsize=None, dpi=100,
            logx=True, logy=False, plot_abs=False, **kwargs_plot,
    ):
        """
        Plot the shape of the interpolated spectrum.

        Parameters
        ----------
        fig : matplotlib.figure.Figure or ``None``
            Figure object. If None, a new figure will be created.
        ax : matplotlib.axes._subplots.AxesSubplot or ``None``
            Axes object. If None, a new axes will be created.
        figsize: (float, float)
            Figure size in inches, as a tuple of two numbers. The figure
            size of ``fig`` (if not ``None``) will override this parameter.
        dpi : float
            Figure resolution. The dpi of ``fig`` (if not ``None``) will override
            this parameter.
        logx : bool
            Whether to show x scale as log.
        logy : bool
            Whether to show y scale as log.
        plot_abs : bool
            Whether to plot the absolute values of the spectrum.
        **kwargs_plot :
            Extra keyword arguments are passed to ``matplotlib.pyplot.plot()``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object being created or being passed into this function.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object being created or being passed into this function.
        """
        fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize=figsize, dpi=dpi)

        if plot_abs:
            ax.plot(self.freq, self.amplitude, **kwargs_plot)
        else:
            ax.plot(self.freq, self.spectrum, **kwargs_plot)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude or phase')
        ax.grid(ls=':')
        if logx: ax.set_xscale('log')
        if logy: ax.set_yscale('log')
        if self._file_name: ax.set_title(self._file_name)

        return fig, ax

    def get_smoothed(self, win_len=15, show_fig=False, *, log_scale, **kwargs):
        """
        Smooth the spectrum by calculating the convolution of the raw
        signal and the smoothing window.

        Parameters
        ----------
        win_len : int
            Length of the smoothing window. Larget numbers means more smoothing.
        show_fig : bool
            Whether to show a before/after figure.
        log_scale : bool
            Whether the frequency spacing of this frequency spectrum is in log
            scale (otherwise, linear scale). If this argument is incorrect, it
            could lead to strange smoothing results.
        **kwargs :
            Extra keyword arguments get passed to the function
            ``helper_signal_processing.log_smooth()``.

        Returns
        -------
        sm : numpy.ndarray (optional, only if ``inplace``)
            The smoothed signal. 1D numpy array.
        fig : matplotlib.figure.Figure
            The figure object being created or being passed into this function.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object being created or being passed into this function.
        """
        sm = sig.log_smooth(
            self.spectrum, win_len=win_len, fmin=self.fmin,
            fmax=self.fmax, lin_space=not log_scale, **kwargs,
        )
        if show_fig:
            fig = plt.figure()
            ax = plt.axes()
            ax.semilogx(self.freq, self.spectrum, color='gray', label='original')
            ax.semilogx(self.freq, sm, color='m', label='smoothed')
            ax.grid(ls=':')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Spectrum')
            ax.legend(loc='best')

        return sm, fig, ax

    def get_f0(self):
        """
        Get the "fundamental frequency" of the amplitude spectrum, which is
        the frequency of the first amplitude peak.

        Returns
        -------
        f0 : float
            The fundamental frequency.
        """
        return sr.find_f0(self.amplitude_2col)

    def get_unwrapped_phase(self, robust=True):
        """
        Unwrpped the phase component of the spectrum.

        Parameters
        ----------
        robust : bool
            When unwrapping, whether to use the robust adjustment or not.
            Turning this option on can help mitigate some issues associated
            with incomplete unwrapping due to discretization errors.

        Returns
        -------
        unwrapped : Frequency_Spectrum
            A frequency spectrum with unwrapped phase component.
        """
        if robust:
            unwrapped_phase = sr.robust_unwrap(self.phase)
        else:
            unwrapped_phase = np.unwrap(self.phase)

        data_1col = self.amplitude * np.exp(1j * unwrapped_phase)
        unwrapped = Frequency_Spectrum(data_1col, df=self.raw_df, interpolate=False)
        return unwrapped

# Author: Jian Shi

import numpy as np
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt

from . import helper_generic as hlp
from . import helper_site_response as sr

#%%############################################################################
def lowpass(orig_signal,cutoff_freq,show_fig=False,filter_order=4,padlen=None):
    '''
    IIR low pass filter. Zero phase distorsion.

    Parameters
    ----------
    orig_signal : numpy.ndarray
        The signal to be filtered(2 columns)
    cutoff_freq : float
        Cut-off frequency
    filter_order : int (default = 4)
        Filter order.
    padlen : int
        pad length

    Returns
    -------
    filtered : numpy.ndarray
        Filtered signal, in two columns, where the 0th is the time and the 1st
        is the signal value.
    '''

    return _filter_kernel(orig_signal, cutoff_freq, 'lowpass', show_fig=show_fig,
                          filter_order=filter_order, padlen=padlen)

#%%############################################################################
def highpass(orig_signal,cutoff_freq,show_fig=False,filter_order=4,padlen=None):
    '''
    IIR high pass filter. Zero phase distorsion.

    Parameters
    ----------
    orig_signal : numpy.ndarray
        The signal to be filtered(2 columns)
    cutoff_freq : float
        Cut-off frequency
    filter_order : int (default = 4)
        Filter order.
    padlen : int
        pad length

    Returns
    -------
    filtered : numpy.ndarray
        Filtered signal, in two columns, where the 0th is the time and the 1st
        is the signal value.
    '''

    return _filter_kernel(orig_signal, cutoff_freq, 'highpass', show_fig=show_fig,
                          filter_order=filter_order, padlen=padlen)

#%%############################################################################
def bandpass(orig_signal,cutoff_freq,show_fig=False,filter_order=4,padlen=None):
    '''
    IIR band pass filter. Zero phase distorsion.

    Parameters
    ----------
    orig_signal : numpy.ndarray
        The signal to be filtered(2 columns)
    cutoff_freq : [float, float]
        Cut-off frequencies, from low to high
    filter_order : int (default = 4)
        Filter order.
    padlen : int
        pad length

    Returns
    -------
    filtered : numpy.ndarray
        Filtered signal, in two columns, where the 0th is the time and the 1st
        is the signal value.
    '''

    return _filter_kernel(orig_signal, cutoff_freq, 'bandpass', show_fig=show_fig,
                          filter_order=filter_order, padlen=padlen)

#%%############################################################################
def bandstop(orig_signal,cutoff_freq,show_fig=False,filter_order=4,padlen=None):
    '''
    IIR band stop filter. Zero phase distorsion.

    Parameters
    ----------
    orig_signal : numpy.ndarray
        The signal to be filtered(2 columns)
    cutoff_freq : [float, float]
        Cut-off frequencies, from low to high
    filter_order : int (default = 4)
        Filter order.
    padlen : int
        pad length

    Returns
    -------
    filtered : numpy.ndarray
        Filtered signal, in two columns, where the 0th is the time and the 1st
        is the signal value.
    '''

    return _filter_kernel(orig_signal, cutoff_freq, 'bandstop', show_fig=show_fig,
                          filter_order=filter_order, padlen=padlen)

#%%############################################################################
def _filter_kernel(orig_signal, cutoff_freq, filter_type, show_fig=False,
                   filter_order=4, padlen=None):
    '''
    Common helper function to the four filtering functions.
    '''

    if filter_type in ['bandpass', 'bandstop']:
        fmin, fmax = cutoff_freq
        if not isinstance(cutoff_freq, (list, tuple, np.ndarray)):
            raise TypeError('`cutoff_freq` must be a list, tuple, or numpy array.')
        if len(cutoff_freq) != 2:
            raise ValueError('`cutoff_freq` must have length 2.')
        if cutoff_freq[1] <= cutoff_freq[0]:
            raise ValueError('`cutoff_freq` must be two values from smaller to larger.')
    elif filter_type in ['highpass', 'lowpass']:
        if not isinstance(cutoff_freq, (float, int, np.number)):
            raise TypeError('`cutoff_freq` must be float, int, or numpy.number.')
    else:
        raise ValueError("`filter_type` must be in {'highpass', 'lowpass', "
                         "'bandpass', 'bandstop'}.")

    hlp.check_two_column_format(orig_signal, name='`orig_signal`')

    x = orig_signal[:,1]
    time = orig_signal[:,0]
    dt = time[1] - time[0]

    sampling_rate = 1.0 / dt
    f_nyquist = sampling_rate / 2.0
    df = 1.0 / (len(x) * dt)
    Wn = np.array(cutoff_freq) / f_nyquist

    if filter_type == 'highpass' and cutoff_freq <= 0:
        return orig_signal
    if filter_type == 'lowpass' and cutoff_freq >= f_nyquist:
        return orig_signal

    if filter_type == 'bandpass':
        if fmax >= f_nyquist * 0.9999:
            return _filter_kernel(orig_signal, fmin, 'highpass', show_fig=show_fig,
                                  filter_order=filter_order, padlen=padlen)
        if fmin <= df:
            return _filter_kernel(orig_signal, fmax, 'lowpass', show_fig=show_fig,
                                  filter_order=filter_order, padlen=padlen)
    if filter_type == 'bandstop':
        if fmax >= f_nyquist * 0.9999:
            return _filter_kernel(orig_signal, fmin, 'lowpass', show_fig=show_fig,
                                  filter_order=filter_order, padlen=padlen)
        if fmin <= df:
            return _filter_kernel(orig_signal, fmax, 'highpass', show_fig=show_fig,
                                  filter_order=filter_order, padlen=padlen)

    b, a = scipy.signal.butter(filter_order, Wn, btype=filter_type)
    y = scipy.signal.filtfilt(b, a, x, padlen=padlen)

    if show_fig == True:
        plt.figure()
        plt.subplot(221)
        plt.plot(time,x)
        plt.title('Original')
        plt.grid(ls=':')

        plt.subplot(223)
        plt.plot(time,y)
        plt.title('After filtering')
        plt.xlabel('Time [sec]')
        plt.grid(ls=':')

        ax = plt.subplot(222)
        freq_orig, spec_orig = fourier_transform(orig_signal).T
        plt.plot(freq_orig,spec_orig)
        ylim = ax.get_ylim()
        if isinstance(cutoff_freq, (list, np.ndarray)) and len(cutoff_freq) >= 1:
            plt.plot([cutoff_freq[0]] * 2, ylim, c='orange')
            plt.plot([cutoff_freq[1]] * 2, ylim, c='orange')
        else:
            plt.plot([cutoff_freq] * 2, ylim, c='orange')
        plt.xscale('log')
        plt.xlim(0.01)
        plt.ylim(0,np.max(spec_orig[1:]))
        plt.grid(ls=':')
        plt.xlabel('Frequency [Hz]')

        ax = plt.subplot(224)
        freq, spec = fourier_transform(np.column_stack((time, y))).T
        plt.plot(freq,spec)
        ylim = ax.get_ylim()
        if isinstance(cutoff_freq, (list, np.ndarray)) and len(cutoff_freq) >= 1:
            plt.plot([cutoff_freq[0]] * 2, ylim, c='orange')
            plt.plot([cutoff_freq[1]] * 2, ylim, c='orange')
        else:
            plt.plot([cutoff_freq] * 2, ylim, c='orange')
        plt.xscale('log')
        plt.xlim(0.01)
        plt.ylim(0,np.max(spec[1:]))
        plt.grid(ls=':')
        plt.xlabel('Frequency [Hz]')

        plt.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.5)

    return np.column_stack((time, y))

#%%############################################################################
def baseline(orig_signal, show_fig=False, cutoff_freq=0.20):
    '''
    Baseline correction of a time-domain signal using high pass filter.

    orig_signal should be two-columned.

    Returns three variables: [t,y], t, and y
    '''

    hlp.check_two_column_format(orig_signal, name='`orig_signal`')

    a = orig_signal[:,1]
    time = orig_signal[:,0]
    dt = time[1] - time[0]
    n0 = len(a)

    #---------- Remove pre-event mean -----------------------------------------
    pre_mean = (a[0] + a[1] + a[2] + a[3] + a[4])/5.0
    a = a - pre_mean

    #---------- Obtain first and last zero crossing ---------------------------
    cross_bound_left = 0
    cross_bound_right = len(a)

    if a[0] >= 0:
        flag1 = 1
    else:
        flag1 = -1

    for i in range(1, len(a)):
        if a[i] >= 0:
            flag2 = 1
        else:
            flag2 = -1
        if flag1 * flag2 < 0:
            cross_bound_left = i
            break

    if a[-1] >= 0:
        glaf1 = 1
    else:
        glaf1 = -1

    for j in range(len(a)-2, -1, -1):
        if a[j] >= 0:
            glaf2 = 1
        else:
            glaf2 = -1

        if glaf1 * glaf2 < 0:
            cross_bound_right = j
            break

    #---------- Pad zeros on both ends ----------------------------------------
    a_cut = a[cross_bound_left : cross_bound_right + 1]

    filter_order = 2
    t_zpad = 1.5 * filter_order / cutoff_freq  # time span of zeros
    nr_zpad = int(np.max([1000, np.round(t_zpad/dt), cross_bound_left + 1])) # number of zeros added

    a_cut = np.append(np.zeros(nr_zpad), np.append(a_cut, np.zeros(nr_zpad)))
    t_cut = np.linspace(dt, len(a_cut) * dt, len(a_cut), endpoint=True)

    #----------- Step 4: High-pass filter -------------------------------------
    a_new = highpass(np.column_stack((t_cut, a_cut)), cutoff_freq)#, padlen=len(x)//10)
    a_new = a_new[:, 1]

    #----------- Shift a_new in time to match original signal -----------------
    a_new = a_new[nr_zpad - cross_bound_left - 1 : ]

    if len(a_new) >= n0:
        a_new = a_new[:n0]
    else:
        a_new = np.append(a_new, np.zeros(n0 - len(a_new)))

    a_new = np.column_stack((time, a_new))

    #---------- Remove trend (assumed straight light) in displacement ---------
    _, u_new = sr.num_int(a_new)
    u_new2 = _remove_linear_trend(u_new)

    v_new2 = sr.num_diff(u_new2)
    a_new2 = sr.num_diff(v_new2)

    #----------- Show plots ---------------------------------------------------
    if show_fig:
        v, u = sr.num_int(orig_signal)
        v_, u_ = sr.num_int(a_new2)

        plt.figure(figsize=(12,7))

        ax = plt.subplot(331)
        ax.plot(time, a, c='b')
        ax.set_title('Uncorrected')
        ax.set_ylabel('Acceleration')
        ax.grid(ls=':')

        ax = plt.subplot(334)
        ax.plot(v[:,0], v[:,1], c='b')
        ax.set_ylabel('Velocity')
        ax.grid(ls=':')

        ax = plt.subplot(337)
        ax.plot(u[:,0], u[:,1], c='b')
        ax.set_ylabel('Displacement')
        ax.grid(ls=':')

        ax = plt.subplot(332)
        ax.plot(a_new2[:, 0], a_new2[:, 1], c='m')
        ax.set_title('Baseline corrected')
        ax.set_ylabel('Acceleration')
        ax.grid(ls=':')

        ax = plt.subplot(335)
        ax.plot(v_[:,0], v_[:,1], c='m')
        ax.set_ylabel('Velocity')
        ax.grid(ls=':')

        ax = plt.subplot(338)
        ax.plot(u_[:,0], u_[:,1], c='m')
        ax.set_ylabel('Displacement')
        ax.grid(ls=':')

        freq_o, spec_o = fourier_transform(orig_signal).T
        freq_n, spec_n = fourier_transform(a_new2).T

        ax = plt.subplot(333)
        ax.plot(freq_o, spec_o, c='b', lw=1.75)
        ax.plot(freq_n, spec_n, c='m', lw=1.15)
        ax.set_xscale('log')
        ax.set_xlim(0.01)
        ax.set_ylim(0, np.max(spec_o[1:]))
        ax.set_xlabel('Frequency [Hz]')
        ax.set_title('Fourier spectra')
        ax.grid(ls=':')

        plt.tight_layout(pad=0.3)

    return a_new2

#%%############################################################################
def _remove_linear_trend(signal):

    t = signal[:, 0]
    x = signal[:, 1]

    slope, intercept = np.polyfit(t, x, 1)
    baseline = slope * t + intercept

    return np.column_stack((t, x - baseline))

#%%############################################################################
def fourier_transform(signal_2_col, real_val=True, double_sided=False,
                      show_fig=False):
    '''
    Fourier transform using FFT.

    Parameters
    ----------
    signal_2_col : numpy.ndarray
        Signal in two columns (time array and signal array)
    real_val : bool
        Whether to return the amplitude (or "magnitude") of the complex numbers
    double_sided : bool
        Whether to return the second half of the spectrum (i.e. beyond the
        Nyquist frequency)
    show_fig : bool
        Whether to show figures of the spectrum

    Returns
    -------
    A two-column array containing [freq_array,spectrum]
    '''

    hlp.check_two_column_format(signal_2_col, '`signal_2_col`')

    signal_ = signal_2_col

    time_array = signal_[:,0]
    x = signal_[:,1]
    N = len(time_array)

    dt = float(time_array[1] - time_array[0])

    X = scipy.fftpack.fft(x)

    if double_sided == False:
        freq_array = np.arange(1, int(np.ceil(N/2.0)) + 1, 1)/(N*dt)
        if real_val:  # absolute Fourier spectra
            X = abs(X[0 : int(np.ceil(N/2.0))])
            spectrum = X
        else:  # complex Fourier spectra
            X = X[0 : int(np.ceil(N/2.0))]
            spectrum = X
    else:
        freq_array = np.arange(1, N+1, 1)/(N*dt)
        if real_val:  # absolute Fourier spectra
            X = abs(X)
            spectrum = X
        else:  # complex Fourier spectra
            spectrum = X

    if show_fig:
        plt.figure(1)

        plt.subplot(211)
        plt.plot(time_array,x)
        plt.xlabel('Time [sec]')
        plt.ylabel('Acceleration [gal]')
        plt.grid(ls=':')

        plt.subplot(212)
        plt.plot(freq_array,abs(spectrum))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.ylim(0,np.max(spectrum[1:]))
        plt.xlim(0.01)
        plt.grid(ls=':')
        if double_sided is False:
            plt.xscale('log')
        elif double_sided is True:
            pass

        plt.show()

    return np.column_stack((freq_array, spectrum))

#%%############################################################################
def log_smooth(signal, win_len=15, window='hanning', lin_space=True, fmin=None,
               fmax=None, n_pts=None, fix_ends=True, beta1=0.9, beta2=0.9):
    '''
    Smooth a frequency spectrum with constant window size in logarithmic space.

    Parameters
    ----------
    signal : numpy.ndarray
        The 1D signal to be smoothed.
    win_len : int
        The length of the convolution window
    window : str
        The name of the window. Valid values:
            ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    lin_space : bool
        Whether or not the points of the signal is uniformly spaced linearly.
        If False, the signal is treated as uniformaly spaced logarithmically.
    fmin, fmax : float
        Minimum and maximum frequencies (in Hz) that the signal is spaced within.
        Only needed when lin_space is True.
    n_pts : int
        The number of points of the logarithmically interpolated the signal.
        Only needed when lin_space is True.
    fix_ends : bool
        Whether or not to fix the two ends of the smoothed signal, so that
        the "boundary effect" from convolution can be corrected. If True, the
        first and last n points will be adjusted using the exponentially
        weighted averaging method. (n is half of win_len.)
    beta1, beta2 : float
        The "strength" of exponentially weighted averaging. For the head and
        the tail ends, respectively. Values should be within [0, 1].

    Returns
    -------
    smoothed_signal : numpy.ndarray
        The smoothed signal which has the same dimension as the original signal.
    '''

    if not isinstance(signal, np.ndarray):
        raise TypeError('signal must be a numpy array.')
    if signal.ndim > 1 and max(signal.shape) > 1:
        raise TypeError('signal must be a 1D numpy array.')
    if signal.size < win_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if win_len < 3:
        return signal
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("'Window' should be 'flat', 'hanning', 'hamming', 'bartlett', or 'blackman'")

    if lin_space and (fmin == None or fmax == None):
        raise ValueError('If lin_space is True, fmin and fmax must be specified.')

    if n_pts == None:
        n_pts = len(signal)

    if lin_space:
        lin_freq = np.linspace(fmin, fmax, n_pts)
        log_freq = np.logspace(np.log10(fmin), np.log10(fmax), n_pts)
        x = np.interp(log_freq, lin_freq, signal)  # interpolate into log space
    else:
        x = np.copy(signal)

    y = smooth(x, win_len, window)

    if fix_ends:
        n = win_len // 2

        if beta1 < 0 or beta1 > 1:
            raise ValueError('beta1 should be within [0, 1].')
        if beta2 < 0 or beta2 > 1:
            raise ValueError('beta2 should be within [0, 1].')

        y[0] = np.mean(x[:n])
        for j in range(1,n):  # exponentially weigted average
            y[j] = beta1 * y[j-1] + (1 - beta1) * x[j]

        y[-1] = np.mean(x[len(y)-n:])
        for j in range(len(y)-2, len(y)-n-1, -1):
            y[j] = beta2 * y[j+1] + (1 - beta2) * x[j]

    smoothed_signal = y
    return smoothed_signal

#%%############################################################################
def smooth(x, window_len=15, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TO-DO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

    [Copied from: http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html]
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts one-dimensional arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("'Window' should be 'flat', 'hanning', 'hamming', 'bartlett', or 'blackman'")

    if window == 'flat': # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), x, mode='same')
    return y

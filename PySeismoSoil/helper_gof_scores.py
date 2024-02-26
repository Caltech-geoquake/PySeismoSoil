import numpy as np
import scipy
from matplotlib import pyplot as plt

from PySeismoSoil import helper_generic as hlp
from PySeismoSoil import helper_signal_processing as sp
from PySeismoSoil import helper_site_response as sr

import os

# Modwt imports
import pdb
import pywt
from scipy.ndimage import convolve1d

def S_(meas, simu):
    if isinstance(meas, (float, np.float64)):
        meas = np.max((meas, np.finfo(np.float64).eps))
        simu = np.max((simu, np.finfo(np.float64).eps))
    else:
        meas[meas < np.finfo(np.float64).eps] = np.finfo(np.float64).eps
        simu[simu < np.finfo(np.float64).eps] = np.finfo(np.float64).eps
    
    rela_diff = (simu-meas)/meas
    score = scipy.special.erf(rela_diff*1)*10
    return score

def d_1234(measurement,simulation,fmin,fmax=None,plot_option=True,filter_order=4):
    """
    Calculates the first four goodness-of-fit scores in the GoF scheme
    described in Shi & Asimaki (2017)*.
    
    [Inputs]
      measurement: Measured time history. Must be two-columned.
      simulation:  Simulated time history. Must be two-columned.
      fmin: Minimum frequency to be examined, unit: Hz.
      fmax: Maximum frequency to be examined, unit: Hz.
      plot_option: If 1, show comparison figures; if 0, don't show.
      filter_order: Order of band pass filter. Default is 4.
      
    [Outputs]
      d1, d2, d3, d4: The four scores.
      fh: The figure handle, when plot_option is set to 1.
      
    * J. Shi, and D. Asimaki. (2017) "From stiffness to strength: Formulation
      and validation of a hybrid hyperbolic nonlinear soil model for site-
      response analyses." Bulletin of the Seismological Society of America. 
      Vol. 107, No. 3, 1336-1355.
      
    (c) Jian Shi, 2/17/2015
    Adapted to Python - fx 02/2024
    """
    
    q = 15
    
    a_m = measurement[q-1:-q,:]
    a_s = simulation[q-1:-q,:]
    
    t1 = a_m[:,0]
    t2 = a_s[:,0]

    a_m = sp.baseline(a_m)
    a_s = sp.baseline(a_s)
    
    if fmax == None:
        a_m = sp.highpass(a_m, fmin, filter_order=filter_order)
        a_s = sp.highpass(a_s, fmin, filter_order=filter_order)
    else:
        a_m = sp.bandpass(a_m, [fmin, fmax], filter_order=filter_order)
        a_s = sp.bandpass(a_s, [fmin, fmax], filter_order=filter_order)
    
    Ia_m, Ia_m_peak = calc_AriasIntensity(a_m)
    Ia_s, Ia_s_peak = calc_AriasIntensity(a_s)
    
    v_m = sr.num_int(a_m)[0]
    v_s = sr.num_int(a_s)[0]
    
    Ie_m, Ie_m_peak = calc_AriasIntensity(v_m)
    Ie_s, Ie_s_peak = calc_AriasIntensity(v_s)
    
    N_Ia_m = Ia_m[:,1]/Ia_m_peak # normalized Ia duration
    N_Ia_s = Ia_s[:,1]/Ia_s_peak # normalized Ia duration
    
    N_Ie_m = Ie_m[:,1]/Ie_m_peak # normalized Ie duration
    N_Ie_s = Ie_s[:,1]/Ie_s_peak # normalized Ie duration
    
    n_m = N_Ia_m.shape[0]
    n_s = N_Ia_s.shape[0]
    
    # If mismatched in time spacing, interpolate to grid with more points
    # (assuming both time series start at 0 and and stop at same time)
    if n_m < n_s: 
        N_Ia_m = hlp.interpolate(t2[0], t2[-1], n_s, N_Ia_m, t1, log_scale=False)
        N_Ie_m = hlp.interpolate(t2[0], t2[-1], n_s, N_Ie_m, t1, log_scale=False)
        
        N_Ia_m = np.nan_to_num(N_Ia_m)
        N_Ie_m = np.nan_to_num(N_Ie_m)
        
        tt_array = t2
    elif n_m > n_s:
        N_Ia_s = hlp.interpolate(t1[0], t1[-1], n_m, N_Ia_s, t2, log_scale=False)
        N_Ie_s = hlp.interpolate(t1[0], t1[-1], n_m, N_Ie_s, t2, log_scale=False)
        
        N_Ia_s = np.nan_to_num(N_Ia_s)
        N_Ie_s = np.nan_to_num(N_Ie_s)
        
        tt_array = t1
    else:
        tt_array = t1
        
    # Calculate scores
    d1 = np.mean( S_(N_Ia_m[1:], N_Ia_s[1:]) ) # because first element of meas is 0
    d2 = np.mean( S_(N_Ie_m[1:], N_Ie_s[1:]) )
    d3 = S_(Ia_m_peak, Ia_s_peak)
    d4 = S_(Ie_m_peak, Ie_s_peak)
    
    return d1, d2, d3, d4

def calc_AriasIntensity(accel_in_SI_unit):
    """
    USAGE: [Ia,Ia_peak] = calcAriasIntensity(accel_in_SI_unit)

    [Input] accel_in_SI_unit: Acceleration in SI unit (i.e., m/s^2). It needs
                                to be in two columns, with the first column
                                being time array, and the second column being
                                acceleration.
    [Outputs] Ia: Arias intensity duration as a function of time
                    (unit: m/s^2) It also has two columns.
                Ia_peak: The maximun value of the Ia duration, which is also
                        the value of the last element in the Ia duration array.
    (c) Jian Shi, 7/3/2013
    Adapted to Python - fx 02/2024
    """
    g = 9.81

    t = accel_in_SI_unit[:, 0]
    a = accel_in_SI_unit[:, 1]
    n = a.shape[0]

    dt = t[1] - t[0]
    Ia = np.zeros((n, 1))
    a_sq = a**2.0

    for ix in range(1, n):
        Ia[ix] = Ia[ix-1] + np.pi/(2*g)*a_sq[ix-1]*dt
        
    Ia_peak = float(Ia[-1])
    Ia = np.hstack((t.reshape(n,1), Ia))

    return Ia, Ia_peak

def d_567(measurement,simulation,fmin,fmax,plot_option=True,baseline_option=True,filter_order=4):
    """
    Calculates the 5th, 6th, and 7th goodness-of-fit scores in the GoF scheme
    described in Shi & Asimaki (2017)*.
    
    [Inputs]
      measurement: Measured time history. Must be two-columned.
      simulation:  Simulated time history. Must be two-columned.
      fmin: Minimum frequency to be examined, unit: Hz.
      fmax: Maximum frequency to be examined, unit: Hz.
      plot_option: If 1, show comparison figures; if 0, don't show.
      baseline_option: Whether or not to perform baseline correction to the
                       input time histories. Default is 1 (YES), because time
                       histories with strong baseline bias could lead to
                       unrealistic peak acceleration/velocity/displacement.
      filter_order: Order of band pass filter. Default is 4.
      
    [Outputs]
      d5, d6, d7: The scores.
      fh: The figure handle, when plot_option is set to 1.
      
    * J. Shi, and D. Asimaki. (2017) "From stiffness to strength: Formulation
      and validation of a hybrid hyperbolic nonlinear soil model for site-
      response analyses." Bulletin of the Seismological Society of America. 
      Vol. 107, No. 3, 1336-1355.
      
    (c) Jian Shi, 2/17/2015
    """
    
    q = 15
    
    a_m = measurement[q-1:-q,:]
    a_s = simulation[q-1:-q,:]
    
    a_m0 = a_m
    a_s0 = a_s
    
    t1 = a_m[:,0]
    t2 = a_s[:,0]
    n = t1.shape[0]
    
    if fmax == None:
        a_m = sp.highpass(a_m, fmin, filter_order=filter_order)
        a_s = sp.highpass(a_s, fmin, filter_order=filter_order)
    else:
        a_m = sp.bandpass(a_m, [fmin, fmax], filter_order=filter_order)
        a_s = sp.bandpass(a_s, [fmin, fmax], filter_order=filter_order)
    
    rms_a_m = calc_rms(a_m)
    rms_a_s = calc_rms(a_s)
    
    if baseline_option:
        a_m = sp.baseline(a_m)
        a_s = sp.baseline(a_s)
    
    v_m, u_m = sr.num_int(a_m)
    v_s, u_s = sr.num_int(a_s)
    
    rms_v_m = calc_rms(v_m)
    rms_v_s = calc_rms(v_s)
    
    u_m = baseline_wavelet(u_m)
    u_s = baseline_wavelet(u_s)
    
    rms_u_m = calc_rms(u_m)
    rms_u_s = calc_rms(u_s)
    
    # Calculate scores
    d5 = S_(rms_a_m,rms_a_s)
    d6 = S_(rms_v_m,rms_v_s)
    d7 = S_(rms_u_m,rms_u_s)
    
    return d5, d6, d7

def baseline_wavelet(signal, wavelet_level=6, wavelet_name='dmey', show_fig=False):
    """
    Perform baseline correction using wavelet decomposition. This function
    first decomposes the signal into different wavelets, and then removes the
    low-frequency trend (i.e., the baseline) from the wavelets, and then
    reconstruct the remaining wavelets into a new signal.
    
    While baseline correction using high pass filtering is more suitable for
    correcting acceleration time histories, this wavelet decomposition method
    is maybe more suitable for correcting displacement time histories. 
       
    (c) Jian Shi, 5/25/2013
    """
    t = signal[:, 0]
    x = signal[:, 1]
    
    # coeffs = pywt.wavedec(x, wavelet=wavelet_name, level=wavelet_level, mode='symmetric')
    # coeffs[0] = np.zeros_like(coeffs[0])
    # y = pywt.waverec(coeffs, wavelet=wavelet_name)
    # y = np.column_stack((t, y[:-1]))
    
    coeffs = modwt(x, filters=wavelet_name, level=wavelet_level)
    mra = modwtmra(coeffs, filters=wavelet_name)
    
    # for mm in mra:
    #     plt.figure()
    #     plt.plot(mm)
    
    mra[-1,:] = np.zeros_like(mra[-1,:])
        
    y = np.sum(mra, axis=0)
    
    plt.figure()
    plt.plot(y)
    
    y = np.column_stack((t, y))
    
    return y

def calc_rms(x):
    rms = np.sqrt(np.mean(x[:, 1] ** 2.0))
    return rms

def d_89(measurement,simulation,fmin=None,fmax=None,plot_option=True,baseline_option=True):
    """
    Calculates the last two goodness-of-fit scores in the GoF scheme
    described in Shi & Asimaki (2017)*.
    
    [Inputs]
      measurement: Measured time history. Must be two-columned.
      simulation:  Simulated time history. Must be two-columned.
      fmin: Minimum frequency to be examined, unit: Hz.
      fmax: Maximum frequency to be examined, unit: Hz.
      plot_option: If 1, show comparison figures; if 0, don't show.
      baseline_option: Whether or not to perform baseline correction to the
                       input time histories. Default is 1 (YES), because time
                       histories with strong baseline bias could lead to
                       undesirable artifacts.
      exact_resp_spect: Whether or not to compare the exact response spectra
                        of the measurement and the simulation, or just
                        compare the approximate response spectra. The exact
                        spectra are much slower to calculate. Default is YES.
                        
    [Outputs]
      d8, d9: The scores.
      fh: The figure handle, when plot_option is set to 1.
      
    * J. Shi, and D. Asimaki. (2017) "From stiffness to strength: Formulation
      and validation of a hybrid hyperbolic nonlinear soil model for site-
      response analyses." Bulletin of the Seismological Society of America. 
      Vol. 107, No. 3, 1336-1355.
      
    (c) Jian Shi, 2/17/2015
    """
    
    t1 = measurement[:, 0]
    t2 = simulation[:, 0]
    
    dt1 = t1[1]-t1[0]
    fs1 = 1.0/dt1
    n1 = measurement.shape[0]
    
    dt2 = t2[1]-t2[0]
    fs2 = 1.0/dt2
    n2 = simulation.shape[0]
    
    if fmin == None:
        fmin = np.max(fs1/n1, fs2/n2)
    if fmax == None:
        fmax = np.min(fs1, fs2)/2.0
    if fmin >= fmax:
        raise Exception(f'Error: fmax must be larger than fmin. (fmax={fmax}, fmin={fmin})')
    
    if baseline_option:
        measurement = sp.baseline(measurement)
        simulation = sp.baseline(simulation)
        
    # response spectra
    a1 = measurement[:,1]
    a2 = simulation[:,1]
    
    Tmax = 1.0/fmin
    Tmin = 1.0/fmax
    
    Tn_m, SA_m, _, _, _, _, _ = sr.response_spectra(measurement, T_max=np.max((Tmax, 5.0)), n_pts=100)
    Tn_s, SA_s, _, _, _, _, _ = sr.response_spectra(simulation, T_max=np.max((Tmax, 5.0)), n_pts=100)
    
    idx1_RS = np.min(( np.where(Tn_m >= Tmin)[0] ))
    idx2_RS = np.max(( np.where(Tn_m <= Tmax)[0] ))
        
    # fourier transform
    ft_m = sp.fourier_transform(measurement)
    ft_s = sp.fourier_transform(simulation)
    
    farray_m = ft_m[:,0]
    farray_s = ft_s[:,0]
    
    df_m = farray_m[1]-farray_m[0]
    df_s = farray_s[1]-farray_s[0]
    
    FS_m = sp.lin_smooth(ft_m[:,1], window_len=round(0.3/df_m))
    FS_s = sp.lin_smooth(ft_s[:,1], window_len=round(0.3/df_s))
    
    idx1_FS = np.max(( np.min(np.where(farray_m >= fmin)[0]), np.min(np.where(farray_s >= fmin)[0]) ))
    idx2_FS = np.min(( np.max(np.where(farray_m <= fmax)[0]), np.max(np.where(farray_s <= fmax)[0]) ))
    
    # calculate scores
    d8 = np.mean( S_(SA_m[idx1_RS:idx2_RS], SA_s[idx1_RS:idx2_RS]) )
    d9 = np.mean( S_(FS_m[idx1_FS:idx2_FS], FS_s[idx1_FS:idx2_FS]) )
    
    return d8, d9

## MODWT FUNCTIONS ##
# Code from: https://github.com/pistonly/modwtpy

def upArrow_op(li, j):
    if j == 0:
        return [1]
    N = len(li)
    li_n = np.zeros(2**(j - 1) * (N - 1) + 1)
    for i in range(N):
        li_n[2**(j - 1) * i] = li[i]
    return li_n


def period_list(li, N):
    n = len(li)
    # append [0 0 ...]
    n_app = N - np.mod(n, N)
    li = list(li)
    li = li + [0] * n_app
    if len(li) < 2 * N:
        return np.array(li)
    else:
        li = np.array(li)
        li = np.reshape(li, [-1, N])
        li = np.sum(li, axis=0)
        return li


def circular_convolve_mra(h_j_o, w_j):
    ''' calculate the mra D_j'''
    return convolve1d(w_j,
                      np.flip(h_j_o),
                      mode="wrap",
                      origin=(len(h_j_o) - 1) // 2)


def circular_convolve_d(h_t, v_j_1, j):
    '''
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j)
    '''
    N = len(v_j_1)
    w_j = np.zeros(N)
    ker = np.zeros(len(h_t) * 2**(j - 1))

    # make kernel
    for i, h in enumerate(h_t):
        ker[i * 2**(j - 1)] = h

    w_j = convolve1d(v_j_1, ker, mode="wrap", origin=-len(ker) // 2)
    return w_j

def modwt(x, filters, level):
    '''
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return np.vstack(wavecoeff)

def modwtmra(w, filters):
    ''' Multiresolution analysis based on MODWT'''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    # D
    level, N = w.shape
    level = level - 1
    D = []
    g_j_part = [1]
    for j in range(level):
        # g_j_part
        g_j_up = upArrow_op(g, j)
        g_j_part = np.convolve(g_j_part, g_j_up)
        # h_j_o
        h_j_up = upArrow_op(h, j + 1)
        h_j = np.convolve(g_j_part, h_j_up)
        h_j_t = h_j / (2**((j + 1) / 2.))
        if j == 0: h_j_t = h / np.sqrt(2)
        h_j_t_o = period_list(h_j_t, N)
        D.append(circular_convolve_mra(h_j_t_o, w[j]))
    # S
    j = level - 1
    g_j_up = upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2**((j + 1) / 2.))
    g_j_t_o = period_list(g_j_t, N)
    S = circular_convolve_mra(g_j_t_o, w[-1])
    D.append(S)
    return np.vstack(D)
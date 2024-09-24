import numpy as np
import pywt
import scipy
from matplotlib import pyplot as plt
from scipy.ndimage import convolve1d

from PySeismoSoil import helper_generic as hlp
from PySeismoSoil import helper_signal_processing as sp
from PySeismoSoil import helper_site_response as sr


def S_(
        meas: float | np.ndarray, simu: float | np.ndarray
) -> float | np.ndarray:
    """
    Score the provided elements as described in Shi & Asimaki (2017).

    Parameters
    ----------
    meas : float | np.ndarray
        The measurement value to compare against.
    simu : float | np.ndarray
        The simulation value to compare with the measurement (truth).

    Returns
    -------
    score : float | np.ndarray
        The computed score between `meas` and `simu`.

    References
    ----------
    1. J. Shi, and D. Asimaki. (2017) "From stiffness to strength: Formulation
       and validation of a hybrid hyperbolic nonlinear soil model for site-
       response analyses." Bulletin of the Seismological Society of America.
       Vol. 107, No. 3, 1336-1355.

    Notes
    -----
    Original Matlab code (c) Jian Shi, 2/17/2015
    Ported to Python by Flora Xia, 02/2024
    """
    eps2 = 1e-12
    if isinstance(meas, (float, np.float64)):
        meas = np.max((meas, np.finfo(np.float64).eps))
        simu = np.max((simu, np.finfo(np.float64).eps))

        if meas < eps2:
            rela_diff = simu - meas
        else:
            rela_diff = (simu - meas) / meas
    else:
        meas[meas < np.finfo(np.float64).eps] = np.finfo(np.float64).eps
        simu[simu < np.finfo(np.float64).eps] = np.finfo(np.float64).eps

        rela_diff = (simu - meas) / meas
        rela_diff[meas < eps2] = simu[meas < eps2] - meas[meas < eps2]

    score = scipy.special.erf(rela_diff * 1) * 10
    return score


def d_1234(
        measurement: np.ndarray,
        simulation: np.ndarray,
        fmin: float | None = None,
        fmax: float | None = None,
        baseline: bool = True,
        show_fig: bool = False,
) -> tuple[float, float, float, float]:
    """
    Calculate the first four goodness-of-fit scores in the GoF scheme
    described in Shi & Asimaki (2017).

    Parameters
    ----------
    measurement : np.ndarray
        Measured time history. Must be two-columned.
    simulation : np.ndarray
        Simulated time history. Must be two-columned.
    fmin : float | None
        Minimum frequency to be considered, in units of Hz.
        Default is (sampling frequency)/(length of time series).
    fmax : float | None
        Maximum frequency to be considered, in units of Hz.
        Default is (sampling frequency)/2.0.
    baseline : bool
        Whether or not to perform baseline correction of the time series.
    show_fig : bool
        Whether or not to plot.

    Returns
    -------
    d1 : float
        Normalized arias intensity score.
    d2 : float
        Normalized energy integral score.
    d3 : float
        Peak arias intensity score.
    d4 : float
        Peak energy integral score.

    References
    ----------
    1. J. Shi, and D. Asimaki. (2017) "From stiffness to strength: Formulation
       and validation of a hybrid hyperbolic nonlinear soil model for site-
       response analyses." Bulletin of the Seismological Society of America.
       Vol. 107, No. 3, 1336-1355.

    Notes
    -----
    Original Matlab code (c) Jian Shi, 2/17/2015
    Ported to Python by Flora Xia, 02/2024
    """
    filter_order = 4
    q = 15

    a_m = measurement[q - 1 : -q, :]
    a_s = simulation[q - 1 : -q, :]

    t1 = a_m[:, 0]
    t2 = a_s[:, 0]

    if baseline:
        a_m = sp.baseline(a_m)
        a_s = sp.baseline(a_s)

    if fmax is None:
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

    N_Ia_m = Ia_m[:, 1] / Ia_m_peak  # normalized Ia duration
    N_Ia_s = Ia_s[:, 1] / Ia_s_peak  # normalized Ia duration

    N_Ie_m = Ie_m[:, 1] / Ie_m_peak  # normalized Ie duration
    N_Ie_s = Ie_s[:, 1] / Ie_s_peak  # normalized Ie duration

    n_m = N_Ia_m.shape[0]
    n_s = N_Ia_s.shape[0]

    # If mismatched in time spacing, interpolate to grid with more points
    # (assuming both time series start at 0 and and stop at same time)
    if n_m < n_s:
        N_Ia_m = hlp.interpolate(
            t2[0], t2[-1], n_s, N_Ia_m, t1, log_scale=False
        )
        N_Ie_m = hlp.interpolate(
            t2[0], t2[-1], n_s, N_Ie_m, t1, log_scale=False
        )

        N_Ia_m = np.nan_to_num(N_Ia_m)
        N_Ie_m = np.nan_to_num(N_Ie_m)

        tt_array = t2
    elif n_m > n_s:
        N_Ia_s = hlp.interpolate(
            t1[0], t1[-1], n_m, N_Ia_s, t2, log_scale=False
        )
        N_Ie_s = hlp.interpolate(
            t1[0], t1[-1], n_m, N_Ie_s, t2, log_scale=False
        )

        N_Ia_s = np.nan_to_num(N_Ia_s)
        N_Ie_s = np.nan_to_num(N_Ie_s)

        tt_array = t1
    else:
        tt_array = t1

    # Calculate scores
    d1 = np.mean(S_(N_Ia_m[1:], N_Ia_s[1:]))  # because first element of meas is 0
    d2 = np.mean(S_(N_Ie_m[1:], N_Ie_s[1:]))
    d3 = S_(Ia_m_peak, Ia_s_peak)
    d4 = S_(Ie_m_peak, Ie_s_peak)

    # Plotting
    if show_fig:
        fig, ax = plt.subplots(2, 2, figsize=(10, 7), dpi=100)

        ax[0, 0].plot(tt_array, N_Ia_m, label='Measurement', color='tab:blue')
        ax[0, 0].plot(
            tt_array,
            N_Ia_s,
            label='Simulation',
            linestyle='--',
            color='tab:orange',
        )
        ax[0, 0].set_title(f'Norm. Arias Intensity (S1): {d1:.2f}')
        ax[0, 0].set_xlabel('Time (s)')
        ax[0, 0].set_ylabel(
            r' Normalized Arias Intensity, I$_a$ / I$_{a, peak}$'
        )
        ax[0, 0].set_xlim((0, np.max(Ia_m[:, 0])))
        ax[0, 0].set_ylim((-0.1, 1.1))
        ax[0, 0].legend()
        ax[0, 0].grid(alpha=0.5)

        ax[0, 1].plot(tt_array, N_Ie_m, label='Measurement', color='tab:blue')
        ax[0, 1].plot(tt_array, N_Ie_s, label='Simulation', color='tab:orange')
        ax[0, 1].set_title(f'Norm. Energy Integral (S2): {d2:.2f}')
        ax[0, 1].set_xlabel('Time (s)')
        ax[0, 1].set_ylabel(
            r'Normalized Energy Integral, I$_e$ / I$_{e, peak}$'
        )
        ax[0, 1].set_xlim((0, np.max(Ie_m[:, 0])))
        ax[0, 1].set_ylim((-0.1, 1.1))
        ax[0, 1].grid(alpha=0.5)

        ax[1, 0].plot(
            tt_array, Ia_m[:, 1], label='Measurement', color='tab:blue'
        )
        ax[1, 0].plot(
            tt_array, Ia_s[:, 1], label='Simulation', color='tab:orange'
        )
        ax[1, 0].set_title(f'Peak Arias Intensity (S3): {d3:.2f}')
        ax[1, 0].set_xlabel('Time (s)')
        ax[1, 0].set_ylabel(r'Arias Intensity, I$_a$')
        ax[1, 0].set_xlim((0, np.max(Ia_m[:, 0])))
        ax[1, 0].grid(alpha=0.5)

        ax[1, 1].plot(
            tt_array, Ie_m[:, 1], label='Measurement', color='tab:blue'
        )
        ax[1, 1].plot(
            tt_array, Ie_s[:, 1], label='Simulation', color='tab:orange'
        )
        ax[1, 1].set_title(f'Peak Energy Integral (S4): {d4:.2f}')
        ax[1, 1].set_xlabel('Time (s)')
        ax[1, 1].set_ylabel(r'Energy Integral, I$_e$')
        ax[1, 1].set_xlim((0, np.max(Ie_m[:, 0])))
        ax[1, 1].grid(alpha=0.5)

        fig.suptitle(
            r'Arias Intensity I$_a$ and Energy Integral I$_e$', fontsize=16
        )

        plt.tight_layout()

    return (d1, d2, d3, d4)


def calc_AriasIntensity(
        accel_in_SI_unit: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compute Arias intensity for scoring.

    Parameters
    ----------
    accel_in_SI_unit : np.ndarray
        Acceleration in SI unit (i.e., m/s^2). It needs
        to be in two columns, with the first column
        being time array, and the second column being
        acceleration.

    Returns
    -------
    Ia : np.ndarray
        Arias intensity duration as a function of time
                    (unit: m/s^2) It also has two columns.
    Ia_peak : float
        The maximum value of the Ia duration, which is also
        the value of the last element in the Ia duration array.

    Notes
    -----
    Original Matlab code (c) Jian Shi, 2/17/2015
    Ported to Python by Flora Xia, 02/2024
    """
    g = 9.81

    t = accel_in_SI_unit[:, 0]
    a = accel_in_SI_unit[:, 1]
    n = a.shape[0]

    dt = t[1] - t[0]
    Ia = np.zeros((n, 1))
    a_sq = a**2.0

    for ix in range(1, n):
        Ia[ix] = Ia[ix - 1] + np.pi / (2 * g) * a_sq[ix - 1] * dt

    Ia_peak = float(Ia[-1])
    Ia = np.hstack((t.reshape(n, 1), Ia))

    return Ia, Ia_peak


def d_567(
        measurement: np.ndarray,
        simulation: np.ndarray,
        fmin: float | None = None,
        fmax: float | None = None,
        baseline: bool = True,
        show_fig: bool = False,
) -> tuple[float, float, float]:
    """
    Calculate the 5th, 6th, and 7th goodness-of-fit scores in the GoF scheme
    described in Shi & Asimaki (2017).

    Parameters
    ----------
    measurement : np.ndarray
        Measured time history. Must be two-columned.
    simulation : np.ndarray
        Simulated time history. Must be two-columned.
    fmin : float | None
        Minimum frequency to be considered, in units of Hz.
        Default is (sampling frequency)/(length of time series).
    fmax : float | None
        Maximum frequency to be considered, in units of Hz.
        Default is (sampling frequency)/2.0.
    baseline : bool
        Whether or not to perform baseline correction of the time series.
    show_fig : bool
        Whether or not to plot.

    Returns
    -------
    d5 : float
        RMS Acceleration score.
    d6 : float
        RMS velocity score.
    d7 : float
        RMS displacement score.

    References
    ----------
    1. J. Shi, and D. Asimaki. (2017) "From stiffness to strength: Formulation
       and validation of a hybrid hyperbolic nonlinear soil model for site-
       response analyses." Bulletin of the Seismological Society of America.
       Vol. 107, No. 3, 1336-1355.

    Notes
    -----
    Original Matlab code (c) Jian Shi, 2/17/2015
    Ported to Python by Flora Xia, 02/2024
    """
    filter_order = 4
    q = 15

    a_m = measurement[q - 1 : -q, :]
    a_s = simulation[q - 1 : -q, :]

    t1 = a_m[:, 0]
    t2 = a_s[:, 0]

    if baseline:
        if fmax is None:
            a_m = sp.highpass(a_m, fmin, filter_order=filter_order)
            a_s = sp.highpass(a_s, fmin, filter_order=filter_order)
        else:
            a_m = sp.bandpass(a_m, [fmin, fmax], filter_order=filter_order)
            a_s = sp.bandpass(a_s, [fmin, fmax], filter_order=filter_order)

    pga_m = getAbsPeak(a_m)
    pga_s = getAbsPeak(a_s)

    rms_a_m = calc_rms(a_m)
    rms_a_s = calc_rms(a_s)

    if baseline:
        a_m = sp.baseline(a_m)
        a_s = sp.baseline(a_s)

    v_m, u_m = sr.num_int(a_m)
    v_s, u_s = sr.num_int(a_s)

    rms_v_m = calc_rms(v_m)
    rms_v_s = calc_rms(v_s)

    pgv_m = getAbsPeak(v_m)
    pgv_s = getAbsPeak(v_s)

    if baseline:
        u_m = baseline_wavelet(u_m)
        u_s = baseline_wavelet(u_s)

    pgd_m = getAbsPeak(u_m)
    pgd_s = getAbsPeak(u_s)

    rms_u_m = calc_rms(u_m)
    rms_u_s = calc_rms(u_s)

    # Calculate scores
    d5 = S_(rms_a_m, rms_a_s)
    d6 = S_(rms_v_m, rms_v_s)
    d7 = S_(rms_u_m, rms_u_s)

    # Plotting
    if show_fig:
        fig, ax = plt.subplots(3, 1, figsize=(10, 7), dpi=100, sharex=True)

        if pga_m >= pga_s:
            ax[0].plot(t1, a_m[:, 1], label='Measurement', linewidth=0.75)
            ax[0].plot(t2, a_s[:, 1], label='Simulation', linewidth=0.75)
        else:
            ax[0].plot(
                t2,
                a_s[:, 1],
                label='Simulation',
                linewidth=0.75,
                color='tab:orange',
            )

            ax[0].plot(
                t1,
                a_m[:, 1],
                label='Measurement',
                linewidth=0.75,
                color='tab:blue',
            )

        ax[0].set_ylabel('Acceleration')
        ax[0].set_title(
            f'RMS Acceleration (S5): {d5:.2f}, RMS Velocity (S6): {d6:.2f}, RMS Displacement (S7): {d7:.2f}'
        )
        ax[0].set_xlim((0, np.max([np.max(t1), np.max(t2)])))
        ax[0].legend()
        ax[0].grid(alpha=0.5)

        if pgv_m >= pgv_s:
            ax[1].plot(t1, v_m[:, 1], label='Measurement', linewidth=0.75)
            ax[1].plot(t2, v_s[:, 1], label='Simulation', linewidth=0.75)
        else:
            ax[1].plot(
                t2,
                v_s[:, 1],
                label='Simulation',
                linewidth=0.75,
                color='tab:orange',
            )

            ax[1].plot(
                t1,
                v_m[:, 1],
                label='Measurement',
                linewidth=0.75,
                color='tab:blue',
            )

        ax[1].set_ylabel('Velocity')
        ax[1].set_xlim((0, np.max([np.max(t1), np.max(t2)])))
        ax[1].grid(alpha=0.5)

        if pgd_m >= pgd_s:
            ax[2].plot(t1, u_m[:, 1], label='Measurement', linewidth=0.75)
            ax[2].plot(t2, u_s[:, 1], label='Simulation', linewidth=0.75)
        else:
            ax[2].plot(
                t2,
                u_s[:, 1],
                label='Simulation',
                linewidth=0.75,
                color='tab:orange',
            )

            ax[2].plot(
                t1,
                u_m[:, 1],
                label='Measurement',
                linewidth=0.75,
                color='tab:blue',
            )

        ax[2].set_ylabel('Displacement')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_xlim((0, np.max([np.max(t1), np.max(t2)])))
        ax[2].grid(alpha=0.5)

        fig.suptitle(r'Time Histories', fontsize=16)

        plt.tight_layout()

    return (d5, d6, d7)


def baseline_wavelet(
        signal: np.ndarray, wavelet_level: int = 6, wavelet_name: str = 'dmey'
) -> np.ndarray:
    """
    Perform baseline correction using wavelet decomposition. This function
    first decomposes the signal into different wavelets, and then removes the
    low-frequency trend (i.e., the baseline) from the wavelets, and then
    reconstruct the remaining wavelets into a new signal.

    While baseline correction using high pass filtering is more suitable for
    correcting acceleration time histories, this wavelet decomposition method
    is maybe more suitable for correcting displacement time histories.

    The python version of this function uses a Python version of modwt, the
    functions of which come from https://github.com/pistonly/modwtpy and can be
    found included at the bottom of this file.

    Parameters
    ----------
    signal: np.ndarray
        The signal to be corrected. Must have two columns, with the first
        being time and the second containing the data.
    wavelet_level: int
        Wavelet level.
    wavelet_name: str
        Type of wavelet to use.

    Returns
    -------
    y : np.ndarray
        The baseline corrected signal. Also has two columns.

    Notes
    -----
    Original Matlab code (c) Jian Shi, 2/17/2015
    Ported to Python by Flora Xia, 02/2024
    """
    t = signal[:, 0]
    x = signal[:, 1]

    coeffs = modwt(x, filters=wavelet_name, level=wavelet_level)
    mra = modwtmra(coeffs, filters=wavelet_name)
    mra[-1, :] = np.zeros_like(mra[-1, :])

    y = np.sum(mra, axis=0)
    y = np.column_stack((t, y))

    return y


def calc_rms(x: np.ndarray) -> float:
    """
    Compute the RMS of the provided signal `x`.

    Parameters
    ----------
    x : np.ndarray
        Assumed to be a two-column array, with the time in the first
        column and the data in the second.

    Returns
    -------
    rms : float
        The RMS value of `x`.
    """
    rms = np.sqrt(np.mean(x[:, 1] ** 2.0))
    return rms


def getAbsPeak(x: np.ndarray) -> float:
    """
    Get the peak value of the absolute value of a signal `x`.

    Parameters
    ----------
    x : np.ndarray
        Assumed to be a two-column array, with the time in the first
        column and the data in the second.

    Returns
    -------
    peak : float
        The peak value of the absolute value of `x`.

    Raises
    ------
    TypeError
        If the size of the second dimension of `x` is not one or two
    """
    if x.shape[1] == 1:
        peak = np.max(np.abs(x))
    elif x.shape[1] == 2:
        peak = np.max(np.abs(x[:, 1]))
    else:
        raise TypeError('Dimension error.')

    return peak


def d_89(
        measurement: np.ndarray,
        simulation: np.ndarray,
        fmin: float | None = None,
        fmax: float | None = None,
        baseline: bool = True,
        show_fig: bool = False,
) -> tuple[float, float]:
    """
    Calculate the last two goodness-of-fit scores in the GoF scheme
    described in Shi & Asimaki (2017).

    Parameters
    ----------
    measurement : np.ndarray
        Measured time history. Must be two-columned.
    simulation : np.ndarray
        Simulated time history. Must be two-columned.
    fmin : float | None
        Minimum frequency to be considered, in units of Hz.
        Default is (sampling frequency)/(length of time series).
    fmax : float | None
        Maximum frequency to be considered, in units of Hz.
        Default is (sampling frequency)/2.0.
    baseline : bool
        Whether or not to perform baseline correction of the time series.
    show_fig : bool
        Whether or not to plot.

    Returns
    -------
    d8 : float
        Spectral acceleration score.
    d9 : float
        Fourier spectra score.

    Raises
    ------
    ValueError
        If fmin is greater than fmax

    References
    ----------
    1. J. Shi, and D. Asimaki. (2017) "From stiffness to strength: Formulation
       and validation of a hybrid hyperbolic nonlinear soil model for site-
       response analyses." Bulletin of the Seismological Society of America.
       Vol. 107, No. 3, 1336-1355.

    Notes
    -----
    Original Matlab code (c) Jian Shi, 2/17/2015
    Ported to Python by Flora Xia, 02/2024
    """
    if baseline:
        measurement = sp.baseline(measurement)
        simulation = sp.baseline(simulation)

    t1 = measurement[:, 0]
    t2 = simulation[:, 0]

    dt1 = t1[1] - t1[0]
    fs1 = 1.0 / dt1
    n1 = measurement.shape[0]

    dt2 = t2[1] - t2[0]
    fs2 = 1.0 / dt2
    n2 = simulation.shape[0]

    if fmin is None:
        fmin = np.max(fs1 / n1, fs2 / n2)

    if fmax is None:
        fmax = np.min(fs1, fs2) / 2.0

    if fmin >= fmax:
        raise ValueError(
            f'Error: fmax must be larger than fmin. (fmax={fmax}, fmin={fmin})'
        )

    if baseline:
        measurement = sp.baseline(measurement)
        simulation = sp.baseline(simulation)

    # response spectra
    Tmax = 1.0 / fmin
    Tmin = 1.0 / fmax

    Tn_m, SA_m, _, _, _, _, _ = sr.response_spectra(
        measurement,
        T_min=np.min((Tmin, 0.01)),
        T_max=np.max((Tmax, 5.0)),
        n_pts=100,
    )
    Tn_s, SA_s, _, _, _, _, _ = sr.response_spectra(
        simulation,
        T_min=np.min((Tmin, 0.01)),
        T_max=np.max((Tmax, 5.0)),
        n_pts=100,
    )

    idx1_RS = np.min(np.where(Tn_m >= Tmin)[0])
    idx2_RS = np.max(np.where(Tn_m <= Tmax)[0]) + 1

    # fourier transform
    ft_m = sp.fourier_transform(measurement)
    ft_s = sp.fourier_transform(simulation)

    farray_m = ft_m[:, 0]
    farray_s = ft_s[:, 0]

    FS_m = sp.sine_smooth(ft_m)
    FS_s = sp.sine_smooth(ft_s)

    idx1_FS = np.max((
        np.min(np.where(farray_m >= fmin)[0]),
        np.min(np.where(farray_s >= fmin)[0]),
    ))
    idx2_FS = (
        np.min((
            np.max(np.where(farray_m <= fmax)[0]),
            np.max(np.where(farray_s <= fmax)[0]),
        ))
        + 1
    )

    # calculate scores
    d8 = np.mean(S_(SA_m[idx1_RS:idx2_RS], SA_s[idx1_RS:idx2_RS]))
    d9 = np.mean(S_(FS_m[idx1_FS:idx2_FS], FS_s[idx1_FS:idx2_FS]))

    # Plotting
    if show_fig:
        fig, ax = plt.subplots(1, 2, figsize=(10, 7), dpi=100)

        # Spectral acceleration
        ax[0].fill_between(
            Tn_m[idx1_RS:idx2_RS],
            np.maximum(SA_m[idx1_RS:idx2_RS], SA_s[idx1_RS:idx2_RS]),
            alpha=0.5,
            color='silver',
            label='Scored Range',
        )
        ax[0].plot(
            Tn_m, SA_m, label='Measurement', color='tab:blue', linewidth=1.25
        )
        ax[0].plot(
            Tn_s, SA_s, label='Simulation', color='tab:orange', linewidth=1.25
        )
        ax[0].set_title(f'Spectral Acceleration (S8): {d8:.2f}')
        ax[0].set_xscale('log')
        ax[0].set_xlim((np.min(Tn_m), np.max(Tn_m)))
        ax[0].set_ylim(bottom=0.0)
        ax[0].set_xlabel('Period (s)')
        ax[0].set_ylabel('Spectral Acceleration')
        ax[0].grid(alpha=0.5)
        ax[0].grid(axis='x', which='minor', alpha=0.35, linestyle='--')
        ax[0].legend()

        # Fourier amplitude spectrum
        ax[1].fill_between(
            farray_m[idx1_FS:idx2_FS],
            np.maximum(FS_m[idx1_FS:idx2_FS], FS_s[idx1_FS:idx2_FS]),
            alpha=0.5,
            color='silver',
            label='Scored Range',
        )
        ax[1].plot(
            farray_m,
            FS_m,
            label='Measurement',
            color='tab:blue',
            linewidth=1.25,
        )
        ax[1].plot(
            farray_s,
            FS_s,
            label='Simulation',
            color='tab:orange',
            linewidth=1.25,
        )
        ax[1].set_title(f'Fourier Spectra (S9): {d9:.2f}')
        ax[1].set_xscale('log')
        ax[1].set_xlim((np.min(farray_m), np.max(farray_m)))
        ax[1].set_ylim(bottom=0.0)
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Fourier Amplitude Spectrum')
        ax[1].grid(alpha=0.5)
        ax[1].grid(axis='x', which='minor', alpha=0.35, linestyle='--')

        fig.suptitle(r'Frequency Content', fontsize=16)

        plt.tight_layout()

    return (d8, d9)


def d_10(
        measurement: np.ndarray,
        simulation: np.ndarray,
        fmin: float | None = None,
        fmax: float | None = None,
        baseline: bool = True,
        show_fig: bool = False,
) -> float:
    """
    Cross-correlation measure of goodness-of-fit, as described in:
    Anderson (2004).

    Parameters
    ----------
    measurement : np.ndarray
        Measured time history. Must be two-columned.
    simulation : np.ndarray
        Simulated time history. Must be two-columned.
    fmin : float | None
        Minimum frequency to be considered, in units of Hz.
        Default is (sampling frequency)/(length of time series).
    fmax : float | None
        Maximum frequency to be considered, in units of Hz.
        Default is (sampling frequency)/2.0.
    baseline : bool
        Whether or not to perform baseline correction of the time series.
    show_fig : bool
        Whether or not to plot.

    Returns
    -------
    d10 : float
        Cross correlation score.

    Raises
    ------
    ValueError
        If fmin is greater than fmax
    TypeError
        If measurement is not the same length as simulation

    References
    ----------
    1. Anderson, J. G. (2004, August). Quantitative measure of the
       goodness-of-fit of synthetic seismograms. In Proceedings of the
       13th world conference on earthquake engineering (Vol. 243, p. 243).
       Vancouver, BC, Canada: International Association for Earthquake
       Engineering.
    """
    if baseline:
        measurement = sp.baseline(measurement)
        simulation = sp.baseline(simulation)

    t1 = measurement[:, 0]
    a1 = measurement[:, 1]

    t2 = simulation[:, 0]
    a2 = simulation[:, 1]

    dt1 = t1[1] - t1[0]
    fs1 = 1.0 / dt1
    n1 = measurement.shape[0]

    dt2 = t2[1] - t2[0]
    fs2 = 1.0 / dt2
    n2 = simulation.shape[0]

    if fmin is None:
        fmin = np.max(fs1 / n1, fs2 / n2)

    if fmax is None:
        fmax = np.min(fs1, fs2) / 2.0

    if fmin >= fmax:
        raise ValueError(
            f'Error: fmax must be larger than fmin. (fmax={fmax}, fmin={fmin})'
        )

    if n1 != n2:
        raise TypeError(
            'Length of measurement and simulation must be the same.'
        )

    numerator = np.sum(a1 * a2) * dt1

    a1_sqr_int = np.sum(a1 * a1) * dt1
    a2_sqr_int = np.sum(a2 * a2) * dt2
    denominator = np.sqrt(a1_sqr_int) * np.sqrt(a2_sqr_int)

    d10 = 1 / (10 * np.abs(numerator / denominator))

    # Plotting
    if show_fig:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=100, sharex=True)

        ax.plot(
            t2,
            simulation[:, 1],
            label='Simulation',
            linewidth=0.75,
            color='tab:orange',
        )
        ax.plot(
            t1,
            measurement[:, 1],
            label='Measurement',
            linewidth=0.75,
            color='tab:blue',
        )
        ax.set_ylabel('Acceleration')
        ax.set_xlim((0, np.max([np.max(t1), np.max(t2)])))
        ax.legend()
        ax.grid(alpha=0.5)

        fig.suptitle(r'Time Histories', fontsize=16)

        plt.tight_layout()

    return d10


def upArrow_op(li, j):
    """Code from: https://github.com/pistonly/modwtpy"""
    if j == 0:
        return [1]

    N = len(li)
    li_n = np.zeros(2 ** (j - 1) * (N - 1) + 1)
    for i in range(N):
        li_n[2 ** (j - 1) * i] = li[i]

    return li_n


def period_list(li, N):
    """Code from: https://github.com/pistonly/modwtpy"""
    n = len(li)
    # append [0 0 ...]
    n_app = N - np.mod(n, N)
    li = list(li)
    li = li + [0] * n_app
    if len(li) < 2 * N:
        li_result = np.array(li)
    else:
        li = np.array(li)
        li = np.reshape(li, [-1, N])
        li = np.sum(li, axis=0)
        li_result = li

    return li_result


def circular_convolve_mra(h_j_o, w_j):
    """Calculate the mra D_j. Code from: https://github.com/pistonly/modwtpy"""
    return convolve1d(
        w_j, np.flip(h_j_o), mode='wrap', origin=(len(h_j_o) - 1) // 2
    )


def circular_convolve_d(h_t, v_j_1, j):
    """Code from: https://github.com/pistonly/modwtpy"""
    N = len(v_j_1)
    w_j = np.zeros(N)
    ker = np.zeros(len(h_t) * 2 ** (j - 1))

    # make kernel
    for i, h in enumerate(h_t):
        ker[i * 2 ** (j - 1)] = h

    w_j = convolve1d(v_j_1, ker, mode='wrap', origin=-len(ker) // 2)
    return w_j


def modwt(x, filters, level):
    """
    Code from: https://github.com/pistonly/modwtpy
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    """
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
    """
    Multiresolution analysis based on MODWT
    Code from: https://github.com/pistonly/modwtpy
    """
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
        h_j_t = h_j / (2 ** ((j + 1) / 2.0))
        if j == 0:
            h_j_t = h / np.sqrt(2)

        h_j_t_o = period_list(h_j_t, N)
        D.append(circular_convolve_mra(h_j_t_o, w[j]))
    # S
    j = level - 1
    g_j_up = upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2 ** ((j + 1) / 2.0))
    g_j_t_o = period_list(g_j_t, N)
    S = circular_convolve_mra(g_j_t_o, w[-1])
    D.append(S)
    return np.vstack(D)

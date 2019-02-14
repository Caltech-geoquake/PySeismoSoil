# Author: Jian Shi

import numpy as np
import matplotlib.pyplot as plt

from . import helper_generic as hlp
from . import helper_site_response as sr
from . import helper_signal_processing as sig

from PySeismoSoil.class_frequency_spectrum import Frequency_Spectrum as FS

class Ground_Motion:
    '''
    Class implementation of an earthquake ground motion.

    Parameters
    ----------
    data : str or numpy.ndarray
        If str: the full file name on the hard drive containing the data.
        If np.ndarray: the numpy array containing the motion data.

        The motion data can be acceleration, velocity, or displacement.

        The data can have one column (which contains the motion) or two
        columns (1st column: time; 2nd column: motion). If only one column
        is supplied, another input parameter "dt" must also be supplied.
    unit : str
        Valid values include:
            ['m', 'cm', 'ft', 'in',
             'm/s', 'cm/s', 'ft/s', 'in/s',
             'm/s/s', 'cm/s/s', 'ft/s/s', 'in/s/s', 'gal', 'g']
    motion_type : {'accel', 'veloc', 'displ'}
        Specifying what type of motion "data" contains. It needs to be
        consistent with "unit". For example, if motion_type is "accel" and
        unit is "m/s", an exception will be raised.
    dt : float
        Recording time interval of the ground motion. If `data` has only one
        column, this parameter must be supplied. If `data` has two columns,
        this parameter is ignored.
    sep : str
        Delimiter character for reading the text file. If `data` is supplied as
        a numpy array, this parameter is ignored.
    **kwargs_to_genfromtxt :
        Any extra keyword arguments will be passed to numpy.genfromtxt()
        function for loading the data from the hard drive (if applicable).

    Attributes
    ----------
    dt : float
        Recording time interval of the motion
    time : numpy.ndarray
        1D numpy array: the time points in seconds
    accel : numpy.ndarray
        A numpy array of two columns, whose first column is identical to "time",
        and second column is the acceleration in SI unit
    veloc : numpy.ndarray
        A numpy array of two columns, whose first column is identical to "time",
        and second column is the velocity in SI unit
    displ : numpy.ndarray
        A numpy array of two columns, whose first column is identical to "time",
        and second column is the displacement in SI unit
    pga, pgv, pgd : float
        Peak ground acceleration, velocity, and displacement in SI unit
    pga_in_gal, pga_in_g, pgv_in_cm_s, pgd_in_cm : <float>
        PGA, PGV, and PGD in other common units
    Arias_Intensity : numpy.ndarray
        A numpy array of two columns, whose first column is identical to "time",
        and second column is the Arias intensity
    Arias_Intensity_normalized : numpy.ndarray
        A numpy array of two columns, whose first column is identical to "time",
        and second column is the normalized Arias intensity
    peak_Arias_Intensity : float
        The last element of the second column of Arias_Intensity
    T5_95 : float
        The time interval (in seconds) between 5% of peak Arias intensity
        to 95% of peak Arias intensity
    rms_accel, rms_veloc, rms_displ : float
        Root-mean-square acceleration, velocity, and displacement of the motion
    '''

    #--------------------------------------------------------------------------
    def __init__(self, data, unit, motion_type='accel', dt=None, sep='\t',
                 **kwargs_to_genfromtxt):

        data_, dt = hlp.read_two_column_stuff(data, delta=dt, sep=sep)

        if unit not in ['m', 'cm', 'ft', 'in',
                        'm/s', 'cm/s', 'ft/s', 'in/s',
                        'm/s/s', 'cm/s/s', 'ft/s/s', 'in/s/s', 'gal', 'g']:
            raise ValueError("Invalid nane for `unit`.")

        if motion_type not in ['accel','veloc','displ']:
            raise ValueError("motion_type must be: ['accel','veloc','displ']")

        if (unit == 'g' or unit == 'gal') and motion_type != 'accel':
            raise ValueError("If unit is g or gal, then motion_type must be `accel`.")

        if motion_type == 'veloc':  # convert data into acceleration first
            data_ = sr.num_diff(data_)
        elif motion_type == 'displ':
            data_ = sr.num_diff(sr.num_diff(data_))

        if unit in ['cm', 'cm/s', 'cm/s/s', 'gal']:
            data_[:, 1] = data_[:, 1] / 100.0  # cm --> m
        elif unit in ['ft', 'ft/s', 'ft/s/s']:
            data_[:, 1] = data_[:, 1] * 0.3048  # ft --> m
        elif unit in ['in', 'in/s', 'in/s/s']:
            data_[:, 1] = data_[:, 1] * 0.0254  # in --> m
        elif unit == 'g':
            data_[:, 1] = data_[:, 1] * 9.81  # g --> m/s/s

        self.dt = float(dt)  # float; unit: sec
        self.__npts = len(data_[:,0])  # int; how many time points
        self.time = np.linspace(0, self.dt*(self.__npts-1), num=self.__npts)

        self.accel = data_  # numpy array, with length unit 'm'
        self.veloc = self.get_veloc()  # numpy array, with length unit 'm'
        self.displ = self.get_displ()  # numpy array, with length unit 'm'

        self.pga = float(np.max(np.abs(data_[:,1])))
        self.pgv = float(np.max(np.abs(self.veloc[:,1])))
        self.pgd = float(np.max(np.abs(self.displ[:,1])))

        self.pga_in_gal = self.pga * 100.0
        self.pga_in_g   = self.pga / 9.81
        self.pgv_in_cm_s = self.pgv * 100.0
        self.pgd_in_cm = self.pgd * 100.0

        arias_result = self.__calc_Arias()
        self.Arias_Intensity = arias_result[0]
        self.Arias_Intensity_normalized = arias_result[1]
        self.peak_Arias_Intensity = arias_result[2]
        self.T5_95 = arias_result[3]
        self.rms_accel, self.rms_veloc, self.rms_displ = self.__calc_RMS()

    #--------------------------------------------------------------------------
    def __repr__(self):
        '''
        Basic information of a ground motion.
        '''
        text = 'Npts=%d, dt=%.4gs, PGA=%.3gg=%.3ggal, PGV=%.3gcm/s, PGD=%.3gcm, T5_95=%.3gs'\
               % (self.__npts, self.dt, self.pga_in_g, self.pga_in_gal,
                  self.pgv_in_cm_s, self.pgd_in_cm, self.T5_95)

        return text

    #--------------------------------------------------------------------------
    def summary(self):
        '''
        Shows a brief summary of the ground motion.
        '''
        print(self)
        self.plot_waveforms()

    #--------------------------------------------------------------------------
    def get_Fourier_spectrum(self, real_val=True, double_sided=False,
                             show_fig=False):
        '''
        Get Fourier spectrum of the ground motion.

        Parameters
        ----------
        real_val : bool
            Whether to return the amplitude (or "magnitude") of the complex numbers
        double_sided : bool
            Whether to return the second half of the spectrum (i.e. beyond the
            Nyquist frequency)
        show_fig : bool
            Whether to show figures of the spectrum

        Return
        ------
        fs : PySeismoSoil.class_frequency_spectrym.Frequency_Spectrum
            A frequency spectrum object
        '''

        x = sig.fourier_transform(self.accel, real_val, double_sided, show_fig)
        fs = FS(x)

        return fs

    #--------------------------------------------------------------------------
    def get_response_spectra(self, T_min=0.01, T_max=10, n_pts=60,
                             damping=0.05, show_fig=True, parallel=False,
                             n_cores=None, subsample_interval=1):
        '''
        Get response spectra of the ground motion.

        Parameters
        ----------
        T_min : float
            Minimum period value to calculate the response spectra
        T_max : float
            Maximum period value to calculate the response spectra
        n_pts : int
            Number of points you want for the response spectra. A high number
            increases computation time.
        damping : float
            Damping of the dash pots. Do not use "percent" as unit. Use 1-based.
        show_fig : bool
            Whether to show a figure of the response spectra
        parallel : bool
            Whether to perform the calculation in parallel
        n_cores : int or None
            Number of cores to use in parallel. Not necessary if not `parallel`.
        subsample_interval : int
            The interval at which to subsample the input acceleration. A higher
            number saves computation time.

        Returns
        -------
        (Tn, SA, PSA, SV, PSV, SD, fn) : a tuple of 1D numpy.ndarray
            Periods, spectral acceleration, pseudo spectral acceleration,
            spectral velocity, pseudo spectral velocity, spectral displacement,
            and frequencies, respectively.
        '''

        rs = sr.response_spectra(self.accel, damping=damping, T_min=T_min,
                                 T_max=T_max, n_pts=n_pts, show_fig=show_fig,
                                 parallel=parallel, n_cores=n_cores,
                                 subsample_interval=subsample_interval)

        return rs

    #--------------------------------------------------------------------------
    def plot_waveforms(self, show_as_unit='m', figsize=(5,6), dpi=100):
        '''
        Plots acceleration, velocity, and displacement waveforms together.

        Returns the figure object.
        '''
        if self.file_name:
            title = self.file_name
        else:
            title = ''

        if show_as_unit == 'm':
            accel_ = self.accel
        elif show_as_unit == 'cm':
            accel_ = self.unit_convert(unit='cm/s/s')
        else:
            raise ValueError('"show_as_unit" can only be "m" or "cm".')

        fig = sr.plot_motion(accel_, unit=show_as_unit, title=title,
                             figsize=figsize, dpi=dpi)

        return fig

    #--------------------------------------------------------------------------
    def unit_convert(self, unit='m/s/s'):
        '''
        Convert the unit of acceleration. "In-place" conversion is not allowed,
        because ground motions are always stored in SI units internally.
        '''
        data_ = self.accel.copy()

        if unit == 'm/s/s':
            pass
        elif unit in ['cm/s/s', 'gal']:
            data_[:,1] = data_[:,1]*100  # m/s/s --> cm/s/s
        elif unit == 'g':
            data_[:,1] = data_[:,1]/9.81  # m/s/s --> g
        elif unit == 'ft/s/s':
            data_[:,1] = data_[:,1]/0.3048  # m --> ft
        elif unit == 'in/s/s':
            data_[:,1] = data_[:,1]/0.0254  # m --> in
        else:
            raise ValueError('Unrecognized unit. Must be an acceleration unit.')

        return data_

    #--------------------------------------------------------------------------
    def get_accel(self, unit='m/s/s'):
        '''
        Returns the acceleration time history.
        '''
        return self.unit_convert(unit)

    #--------------------------------------------------------------------------
    def get_veloc(self, unit='m/s'):
        '''
        Returns the velocity time history.
        '''
        if unit == 'm/s':
            return sr.num_int(self.accel)[0]
        elif unit in ['cm/s','ft/s','in/s']:
            return sr.num_int(self.unit_convert(unit + '/s'))[0]
        else:
            raise ValueError('Unrecognized unit.')

    #--------------------------------------------------------------------------
    def get_displ(self, unit='m'):
        '''
        Returns the displacement time history.
        '''
        if unit == 'm':
            return sr.num_int(self.accel)[1]
        elif unit in ['cm','ft','in']:
            return sr.num_int(self.unit_convert(unit + '/s/s'))[1]
        else:
            raise ValueError('Unrecognized unit.')

    #--------------------------------------------------------------------------
    def __calc_RMS(self):
        '''
        Private method.

        Returns RMS acceleration, velocity, and displacement. Unit: SI.
        '''

        acc = self.accel
        vel, dis = sr.num_int(acc)
        rms_accel = np.sqrt(np.mean(acc[:,1]**2.0))
        rms_veloc = np.sqrt(np.mean(vel[:,1]**2.0))
        rms_displ = np.sqrt(np.mean(dis[:,1]**2.0))

        return rms_accel, rms_veloc, rms_displ

    #--------------------------------------------------------------------------
    def __arias_time_bounds(self, t, Ia_normalized, low_lim, high_lim):
        '''
        Private method.

        Calculate lower and upper time bounds corresponding to two given
        normalized Arias intensity percentages (e.g., [0.05, 0.95])
        '''

        if low_lim >= high_lim:
            raise ValueError('low_lim must be smaller than high_lim.')

        if t is None:
            t = self.accel[:,0]
        if Ia_normalized is None:
            Ia_normalized = self.Arias_Intensity_normalized[:,1]

        if len(t) != len(Ia_normalized):
            raise ValueError('Ia_normalized and t must have the same length.')

        n = len(t)
        t_low = 0.0  # initialize this variable, in case low_lim <= 0 seconds
        t_high = t[-1]  # initialize t_high, in case high_lim >= max(time)

        prev = Ia_normalized[0]
        for i in range(n):
            if Ia_normalized[i] >= low_lim and prev < low_lim:
                t_low = t[i]
            if Ia_normalized[i] >= high_lim and prev < high_lim:
                t_high = t[i]
            prev = Ia_normalized[i]

        return t_low, t_high

    #--------------------------------------------------------------------------
    def __calc_Arias(self, motion='accel', show_fig=False):
        '''
        Private method.

        Calculate Arias intensity. Returns the intensity time series, peak
        intensity, and T5_95 (time interval from 5% Arias intensity to 95%
        Arias intensity).
        '''

        g = 9.81

        if motion == 'accel':
            t = self.accel[:,0]
            a = self.accel[:,1]
        elif motion == 'veloc':
            t = self.veloc[:,0]
            a = self.veloc[:,1]
        elif motion == 'displ':
            t = self.displ[:,0]
            a = self.displ[:,1]

        n = len(a)

        dt = t[1] - t[0]
        Ia_1col = np.zeros(n)
        a_sq = a**2.

        for i in range(1,n):
            Ia_1col[i] = Ia_1col[i-1] + np.pi/(2*g)*a_sq[i-1]*dt

        Ia_peak = float(Ia_1col[-1])
        Ia = np.column_stack((t,Ia_1col))
        Ia_norm_1col = Ia_1col / Ia_peak  # normalized
        Ia_norm = np.column_stack((t,Ia_norm_1col))

        t_low, t_high = self.__arias_time_bounds(t, Ia_norm_1col, 0.05, 0.95)
        T5_95 = t_high - t_low

        if show_fig:
            plt.figure()
            ax = plt.axes()
            ax.plot(t, Ia)
            ax.grid(ls=':')
            ax.set_xlabel('Time [sec]')
            ax.set_ylabel('Arias intensity')

            y_low, y_high = ax.get_ylim()
            plt.plot([t_low,t_low],[y_low,y_high],lw=0.75,ls='--',c='r')
            plt.plot([t_high,t_high],[y_low,y_high],lw=0.75,ls='--',c='r')

        return Ia, Ia_norm, Ia_peak, T5_95

    #--------------------------------------------------------------------------
    def scale_motion(self, factor=1, target_PGA=None, inplace=False):
        '''
        Scale ground motion, either by specifying a factor, or specifying a
        target PGA level.
        '''

        if factor != None and target_PGA != None:
            raise ValueError('At least one of "factor" and "target_PGA" should be None.')

        if target_PGA != None:
            factor = target_PGA / self.pga
        else:  # factor != None, and target_PGA is None
            pass

        if not inplace:
            time = self.accel[:,0]
            acc = self.accel[:,1]
            acc_scaled = acc * factor

            return np.column_stack((time, acc_scaled))
        else:
            self.accel[:,1] = self.accel[:,1] * factor

    #--------------------------------------------------------------------------
    def truncate(self, limit, arias=True, extend=[0,0], inplace=False,
                 show_fig=False):
        '''
        Truncate ground motion, removing data points in the head and/or tail.

        Parameters
        ----------
        limit : tuple or list of two elements
            The lower/upper bounds of time (e.g., [2, 95]) or normalized Arias
            intensity (e.g., [0.05, 0.95])
        arias : <bool>
            Whether or not "limit" means the normalized Arias intensity
        extend : tuple or list of two elements
            How many seconds to extend before and after the original truncated
            time limits. For example, if extend is [5, 5] sec, and the original
            time limits are [3, 50] sec, then the actual time limits are
            [0, 55] sec. (3 - 5 = -2 smaller than 0, so truncated at 0.)
        inplace : <bool>
            Whether or not to perform the truncation in-place.
        show_fig : <bool>
            Whether or not to show the waveforms before and after truncation.

        Returns
        -------
        If not inplace:
            - if show_fig: returns (truncated_accel, figure_object)
            - if not show_fig: returns truncated_accel
        If inplace:
            - if show_fig: returns figure_object
            - if not show_fig: returns None
        '''

        if not isinstance(limit, (tuple, list)):
            raise TypeError('"limit" must be a list/tuple of  two elements.')
        if len(limit) != 2:
            raise ValueError('Length of "limit" must be 2.')
        if not isinstance(extend, (tuple, list)):
            raise TypeError('"extend" must be a list/tuple of  two elements.')
        if len(extend) != 2:
            raise ValueError('Length of "extend" must be 2.')

        if extend[0] < 0 or extend[1] < 0:
            raise ValueError('extend should be non negative.')

        lim1, lim2 = limit
        if lim1 >= lim2:
            raise ValueError('"limit" must be in ascending order.')

        if not arias:  # "limit" represents actual time limits
            t1, t2 = lim1, lim2
        else:  # "limit" represents bounds of normalized Arias instensity
            t1, t2 = self.__arias_time_bounds(None, None, lim1, lim2)

        t1 -= extend[0]
        t2 += extend[1]

        n1 = int(t1 / self.dt)
        n2 = int(t2 / self.dt)

        if n1 < 0:           n1 = 0
        if n2 > self.__npts: n2 = self.__npts

        time_trunc = self.accel[:n2-n1, 0]
        accel_trunc = self.accel[n1:n2, 1]
        truncated = np.column_stack((time_trunc, accel_trunc))

        if show_fig:
            fig = plt.figure(figsize=(5,6))
            fig.subplots_adjust(left=0.2)

            plt.subplot(3,1,1)
            plt.plot(self.time, self.accel[:,1], 'gray', lw=1.75, label='original')
            plt.plot(self.time[n1:n2], truncated[:,1], 'm', lw=1., label='truncated')
            plt.grid(ls=':')
            plt.ylabel('Accel.')
            plt.legend(loc='best')

            plt.subplot(3,1,2)
            plt.plot(self.time, self.veloc[:,1], 'gray', lw=1.75)
            plt.plot(self.time[n1:n2], sr.numInt(truncated)[0][:,1], 'm', lw=1.)
            plt.grid(ls=':')
            plt.ylabel('Veloc.')

            plt.subplot(3,1,3)
            plt.plot(self.time, self.displ[:,1], 'gray', lw=1.75)
            plt.plot(self.time[n1:n2], sr.numInt(truncated)[1][:,1], 'm', lw=1.)
            plt.grid(ls=':')
            plt.ylabel('Displ.')
            plt.xlabel('Time [sec]')

            plt.tight_layout(pad=0.3)

        if not inplace:
            if show_fig:
                return truncated, fig
            else:
                return truncated
        else:
            self.time = time_trunc
            self.accel = truncated
            if show_fig:
                return fig
            else:
                return None

    #--------------------------------------------------------------------------
    def baseline_correct(self, cutoff_freq=0.20, show_fig=False, inplace=False):
        '''
        Baseline-correct the acceleration (via zero-phase-shift high-pass method)
        '''

        accel_ = sig.baseline(self.accel, show_fig=show_fig,
                              cutoff_freq=cutoff_freq)[0]

        if inplace:
            self.accel = accel_
            return None
        else:
            return accel_

    #--------------------------------------------------------------------------
    def lowpass(self, cutoff_freq, show_fig=False, inplace=False,
                filter_order=4, padlen=150):
        '''
        Zero-phase-shift low-pass.
        '''

        accel_ = sig.lowpass(self.accel, cutoff_freq, show_fig=show_fig,
                             filter_order=filter_order, padlen=padlen)[0]

        if inplace:
            self.accel = accel_
            return None
        else:
            return accel_

    #--------------------------------------------------------------------------
    def highpass(self, cutoff_freq, show_fig=False, inplace=False,
                filter_order=4, padlen=150):
        '''
        Zero-phase-shift low-pass.
        '''

        accel_ = sig.highpass(self.accel, cutoff_freq, show_fig=show_fig,
                              filter_order=filter_order, padlen=padlen)[0]

        if inplace:
            self.accel = accel_
            return None
        else:
            return accel_

    #--------------------------------------------------------------------------
    def save_accel(self, fname, sep='\t', t_prec='%.5g', motion_prec='%.5g',
                   unit='m/s/s'):
        '''
        Saves the acceleration as a text file.
        '''

        fmt = [t_prec, motion_prec]
        data = self.accel

        if unit == 'm/s/s':
            pass
        elif unit == 'g':
            data[:,1] = data[:,1] / 9.81
        elif unit in ['gal', 'cm/s/s']:
            data[:,1] = data[:,1] * 100.0

        np.savetxt(fname, data, fmt=fmt, delimiter=sep)

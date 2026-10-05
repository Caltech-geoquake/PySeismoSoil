"""Ground motion class."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from PySeismoSoil import helper_generic as hlp
from PySeismoSoil import helper_signal_processing as sig
from PySeismoSoil import helper_site_response as sr
from PySeismoSoil.class_frequency_spectrum import Frequency_Spectrum
from PySeismoSoil.class_Vs_profile import Vs_Profile

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# Number of elements that the ``limit`` argument (lower and upper bounds) and
# the ``extend`` argument (amounts to extend before and after) of
# ``Ground_Motion.truncate()`` must have
NUM_ELEMENTS_IN_LIMIT_ARG = 2
NUM_ELEMENTS_IN_EXTEND_ARG = 2


class Ground_Motion:
    r"""
    Class implementation of an earthquake ground motion.

    Parameters
    ----------
    data : str | np.ndarray
        - If str: the full file name on the hard drive containing the data.
        - If np.ndarray: the numpy array containing the motion data.

        The motion data can be acceleration, velocity, or displacement.

        The data can have one column (which contains the motion) or two columns
        (1st column: time; 2nd column: motion). If only one column is supplied,
        another input parameter ``dt`` must also be supplied.
    unit : str
        Valid values include: ['m', 'cm', 'm/s', 'cm/s', 'm/s/s', 'cm/s/s',
        'gal', 'g']
    motion_type : Literal['accel', 'veloc', 'displ'], default='accel'
        Specifying what type of motion "data" contains. It needs to be
        consistent with "unit". For example, if motion_type is "accel" and unit
        is "m/s", an exception will be raised.
    dt : float | None, default=None
        Recording time interval of the ground motion. If ``data`` has only one
        column, this parameter must be supplied. If ``data`` has two columns,
        this parameter is ignored.
    sep : str, default='\t'
        Delimiter character for reading the text file. If ``data`` is supplied
        as a numpy array, this parameter is ignored.
    **kwargs_to_genfromtxt : dict[Any, Any]
        Any extra keyword arguments will be passed to ``numpy.genfromtxt()``
        function for loading the data from the hard drive (if applicable).

    Attributes
    ----------
    dt : float
        Recording time interval of the motion.
    time : np.ndarray
        1D numpy array: the time points in seconds.
    accel : np.ndarray
        A numpy array of two columns, whose first column is identical to
        "time", and second column is the acceleration in SI unit.
    veloc : np.ndarray
        A numpy array of two columns, whose first column is identical to
        "time", and second column is the velocity in SI unit.
    displ : np.ndarray
        A numpy array of two columns, whose first column is identical to
        "time", and second column is the displacement in SI unit.
    npts : int
        Number of time points in the motion.
    pga : float
        Peak ground acceleration in SI unit.
    pgv : float
        Peak ground velocity in SI unit.
    pgd : float
        Peak ground displacement in SI unit.
    pga_in_gal : float
        Peak ground acceleration in gal units.
    pga_in_g : float
        Peak ground acceleration in g units.
    pgv_in_cm_s : float
        Peak ground velocity in cm/s units.
    pgd_in_cm : float
        Peak ground displacement in cm units.
    Arias_Intensity : np.ndarray
        A numpy array of two columns, whose first column is identical to
        "time", and second column is the Arias intensity.
    Arias_Intensity_normalized : np.ndarray
        A numpy array of two columns, whose first column is identical to
        "time", and second column is the normalized Arias intensity.
    peak_Arias_Intensity : float
        The last element of the second column of Arias_Intensity.
    T5_95 : float
        The time interval (in seconds) between 5% of peak Arias intensity to
        95% of peak Arias intensity.
    rms_accel : float
        Root-mean-square acceleration of the ground motion.
    rms_veloc : float
        Root-mean-square velocity of the ground motion.
    rms_displ : float
        Root-mean-square displacement of the ground motion.

    Raises
    ------
    ValueError
        When ``motion_type`` has invalid values
    """

    dt: float
    time: np.ndarray
    accel: np.ndarray
    veloc: np.ndarray
    displ: np.ndarray
    npts: int
    pga: float
    pgv: float
    pgd: float
    pga_in_gal: float
    pga_in_g: float
    pgv_in_cm_s: float
    pgd_in_cm: float
    Arias_Intensity: np.ndarray
    Arias_Intensity_normalized: np.ndarray
    peak_Arias_Intensity: float
    T5_95: float
    rms_accel: float
    rms_veloc: float
    rms_displ: float

    def __init__(
            self,
            data: str | np.ndarray,
            *,
            unit: str,
            motion_type: Literal['accel', 'veloc', 'displ'] = 'accel',
            dt: float | None = None,
            sep: str = '\t',
            **kwargs_to_genfromtxt: dict[Any, Any],  # noqa: ARG002
    ) -> None:
        if isinstance(data, str):  # a file name
            self._path_name, self._file_name = os.path.split(data)
        else:
            self._path_name, self._file_name = None, None

        data_, dt = hlp.read_two_column_stuff(data, delta=dt, sep=sep)

        valid_unit_name = [
            'm',
            'cm',
            'm/s',
            'cm/s',
            'm/s/s',
            'cm/s/s',
            'gal',
            'g',
        ]
        if unit not in valid_unit_name:
            if 's^2' in unit:
                raise ValueError(
                    "Please use '/s/s' instead of 's^2' in `unit`."
                )

            raise ValueError(
                f'Invalid `unit` name. Valid names are: {valid_unit_name}',
            )

        if motion_type not in {'accel', 'veloc', 'displ'}:
            raise ValueError(
                "`motion_type` must be in {'accel', 'veloc', 'displ'}"
            )

        if (unit in {'g', 'gal'}) and motion_type != 'accel':
            raise ValueError(
                "If unit is 'g' or 'gal', then `motion_type` must be 'accel'.",
            )

        # Keep the plain assignment instead of `/=` or `*=` because `data_` can
        # be an integer array, which this assignment casts back to integers,
        # whereas the in-place operators would raise a `TypeError`
        if unit in {'cm', 'cm/s', 'cm/s/s', 'gal'}:
            data_[:, 1] = data_[:, 1] / 100.0  # noqa: PLR6104  # cm --> m
        elif unit == 'g':
            data_[:, 1] = data_[:, 1] * 9.81  # noqa: PLR6104  # g --> m/s/s

        self.dt = float(dt)  # float; unit: sec
        self.npts = len(data_[:, 0])  # int; how many time points
        self.time = np.linspace(0, self.dt * (self.npts - 1), num=self.npts)

        if motion_type == 'accel':
            self.accel = data_  # numpy array, with length unit 'm'
            self.veloc, self.displ = sr.num_int(self.accel)
        elif motion_type == 'veloc':
            self.accel = sr.num_diff(data_)
            self.veloc = data_
            self.displ = sr.num_int(data_)[0]
        else:  # displ
            self.veloc = sr.num_diff(data_)
            self.accel = sr.num_diff(self.veloc)
            self.displ = data_

        self.pga = float(np.max(np.abs(self.accel[:, 1])))
        self.pgv = float(np.max(np.abs(self.veloc[:, 1])))
        self.pgd = float(np.max(np.abs(self.displ[:, 1])))

        self.pga_in_gal = self.pga * 100.0
        self.pga_in_g = self.pga / 9.81
        self.pgv_in_cm_s = self.pgv * 100.0
        self.pgd_in_cm = self.pgd * 100.0

        arias_result = self.__calc_Arias()
        self.Arias_Intensity = arias_result[0]
        self.Arias_Intensity_normalized = arias_result[1]
        self.peak_Arias_Intensity = arias_result[2]
        self.T5_95 = arias_result[3]
        self.rms_accel, self.rms_veloc, self.rms_displ = self.__calc_RMS()

    def __repr__(self) -> str:
        """Return basic information of a ground motion."""
        return (
            f'n_pts={self.npts:d}, dt={self.dt:.4g}s,'
            f' PGA={self.pga_in_g:.3g}g={self.pga_in_gal:.3g}gal,'
            f' PGV={self.pgv_in_cm_s:.3g}cm/s, PGD={self.pgd_in_cm:.3g}cm,'
            f' T5_95={self.T5_95:.3g}s'
        )

    def summary(self) -> None:
        """Show a brief summary of the ground motion."""
        print(self)
        self.plot()

    def get_Fourier_spectrum(
            self,
            *,
            real_val: bool = True,
            double_sided: bool = False,
            show_fig: bool = False,
    ) -> Frequency_Spectrum:
        """
        Get Fourier spectrum of the ground motion.

        Parameters
        ----------
        real_val : bool, default=True
            Whether to return the amplitude (or "magnitude") of the complex
            numbers.
        double_sided : bool, default=False
            Whether to return the second half of the spectrum (i.e. beyond the
            Nyquist frequency).
        show_fig : bool, default=False
            Whether to show figures of the spectrum.

        Returns
        -------
        Frequency_Spectrum
            A frequency spectrum object.
        """
        x = sig.fourier_transform(
            self.accel,
            real_val=real_val,
            double_sided=double_sided,
            show_fig=show_fig,
        )
        return Frequency_Spectrum(x)

    def get_response_spectra(
            self,
            T_min: float = 0.01,
            T_max: float = 10,
            n_pts: int = 60,
            damping: float = 0.05,
            *,
            show_fig: bool = True,
            parallel: bool = False,
            n_cores: int | None = None,
            subsample_interval: int = 1,
    ) -> tuple[np.ndarray, ...]:
        """
        Get elastic response spectra of the ground motion.

        This uses the "exact" solution to the equation of motion (Section 5.2,
        Dynamics of Structures, Second Edition, by Anil K. Chopra).

        Parameters
        ----------
        T_min : float, default=0.01
            Minimum period value to calculate the response spectra. Unit: sec.
        T_max : float, default=10
            Maximum period value to calculate the response spectra. Unit: sec.
        n_pts : int, default=60
            Number of points you want for the response spectra. A high number
            increases computation time.
        damping : float, default=0.05
            Damping of the dash pots. Do not use "percent" as unit. Unit: 1
            (i.e., not percent).
        show_fig : bool, default=True
            Whether to show a figure of the response spectra.
        parallel : bool, default=False
            Whether to perform the calculation in parallel.
        n_cores : int | None, default=None
            Number of cores to use in parallel. Not necessary if not
            ``parallel``.
        subsample_interval : int, default=1
            The interval at which to subsample the input acceleration in the
            time domain. A higher number reduces computation time, but could
            lead to less accurate results.

        Returns
        -------
        (Tn, SA, PSA, SV, PSV, SD, fn) : tuple[np.ndarray, ...]
            Periods, spectral acceleration, pseudo spectral acceleration,
            spectral velocity, pseudo spectral velocity, spectral displacement,
            and frequencies, respectively. Units: SI.
        """
        return sr.response_spectra(
            self.accel,
            damping=damping,
            T_min=T_min,
            T_max=T_max,
            n_pts=n_pts,
            show_fig=show_fig,
            parallel=parallel,
            n_cores=n_cores,
            subsample_interval=subsample_interval,
        )

    def plot(
            self,
            show_as_unit: str = 'm',
            fig: Figure | None = None,
            ax: Axes | None = None,
            figsize: tuple[float, float] = (5, 6),
            dpi: float = 100,
    ) -> tuple[Figure, Axes]:
        """
        Plot acceleration, velocity, and displacement waveforms together.

        Parameters
        ----------
        show_as_unit : str, default='m'
            What unit to convert the ground motion into, when plotting.
        fig : Figure | None, default=None
            Figure object. If None, a new figure will be created.
        ax : Axes | None, default=None
            Axes object. If None, a new axes will be created.
        figsize : tuple[float, float], default=(5, 6)
            Figure size in inches, as a tuple of two numbers. The figure size
            of ``fig`` (if not ``None``) will override this parameter.
        dpi : float, default=100
            Figure resolution. The dpi of ``fig`` (if not ``None``) will
            override this parameter.

        Returns
        -------
        fig : Figure
            The figure object being created or being passed into this function.
        ax : Axes
            The axes object being created or being passed into this function.

        Raises
        ------
        ValueError
            When the value of ``show_as_unit`` is invalid
        """
        title = self._file_name or ''

        if show_as_unit == 'm':
            accel_ = self.accel
        elif show_as_unit == 'cm':
            accel_ = self._unit_convert(unit='cm/s/s')
        elif show_as_unit == 'g':
            accel_ = self._unit_convert(unit='g')
        else:
            raise ValueError("`show_as_unit` can only be 'm', 'cm', or 'g'.")

        fig, ax = sr.plot_motion(
            accel_,
            unit=show_as_unit,
            title=title,
            fig=fig,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
        )
        return fig, ax

    def _unit_convert(
            self, unit: Literal['m/s/s', 'cm/s/s', 'gal', 'g'] = 'm/s/s'
    ) -> np.ndarray:
        """
        Convert the unit of acceleration.

        "In-place" conversion is not allowed, because ground motions are always
        stored in SI units internally.

        Parameters
        ----------
        unit : Literal['m/s/s', 'cm/s/s', 'gal', 'g'], default='m/s/s'
            What unit to convert the acceleration into.

        Returns
        -------
        accel : np.ndarray
            Acceleration time history with the desired unit. It is a 2D numpy
            array wity two columns (time and acceleration).

        Raises
        ------
        ValueError
            When the value of ``unit`` is not valid
        """
        accel = self.accel.copy()

        if unit == 'm/s/s':
            pass
        elif unit in {'cm/s/s', 'gal'}:
            accel[:, 1] *= 100  # m/s/s --> cm/s/s
        elif unit == 'g':
            accel[:, 1] /= 9.81  # m/s/s --> g
        else:
            raise ValueError(
                'Unrecognized `unit`. Must be an acceleration unit.'
            )

        return accel

    def __calc_RMS(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return RMS acceleration, velocity, and displacement. Unit: SI."""
        acc = self.accel
        vel, dis = sr.num_int(acc)
        rms_accel = np.sqrt(np.mean(acc[:, 1] ** 2.0))
        rms_veloc = np.sqrt(np.mean(vel[:, 1] ** 2.0))
        rms_displ = np.sqrt(np.mean(dis[:, 1] ** 2.0))

        return rms_accel, rms_veloc, rms_displ

    def __arias_time_bounds(
            self,
            t: np.ndarray | None,
            Ia_normalized: np.ndarray | None,
            low_lim: float,
            high_lim: float,
    ) -> tuple[float, float]:
        """
        Calculate lower and upper time bounds for given Arias intensities.

        The bounds correspond to two given normalized Arias intensity
        percentages (e.g., [0.05, 0.95]).
        """
        if low_lim >= high_lim:
            raise ValueError('low_lim must be smaller than high_lim.')

        if t is None:
            t = self.accel[:, 0]

        if Ia_normalized is None:
            Ia_normalized = self.Arias_Intensity_normalized[:, 1]

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

    def __calc_Arias(
            self,
            motion: str = 'accel',
            *,
            show_fig: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Calculate Arias intensity.

        Returns the intensity time series, peak intensity, and T5_95 (time
        interval from 5% Arias intensity to 95% Arias intensity).
        """
        g = 9.81

        if motion == 'accel':
            t = self.accel[:, 0]
            a = self.accel[:, 1]
        elif motion == 'veloc':
            t = self.veloc[:, 0]
            a = self.veloc[:, 1]
        elif motion == 'displ':
            t = self.displ[:, 0]
            a = self.displ[:, 1]

        n = len(a)

        dt = t[1] - t[0]
        Ia_1col = np.zeros(n)
        a_sq = a**2.0

        for i in range(1, n):
            Ia_1col[i] = Ia_1col[i - 1] + np.pi / (2 * g) * a_sq[i - 1] * dt

        Ia_peak = float(Ia_1col[-1])
        Ia = np.column_stack((t, Ia_1col))
        Ia_norm_1col = Ia_1col / Ia_peak  # normalized
        Ia_norm = np.column_stack((t, Ia_norm_1col))

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
            plt.plot([t_low, t_low], [y_low, y_high], lw=0.75, ls='--', c='r')
            plt.plot(
                [t_high, t_high], [y_low, y_high], lw=0.75, ls='--', c='r'
            )

        return Ia, Ia_norm, Ia_peak, T5_95

    def scale_motion(
            self,
            factor: float = 1.0,
            target_PGA_in_g: float | None = None,
    ) -> Ground_Motion:
        """
        Scale ground motion.

        This can be done either by specifying a factor, or specifying a target
        PGA level.

        Parameters
        ----------
        factor : float, default=1.0
            The factor to multiply to the original acceleration (with the unit
            of m/s/s)
        target_PGA_in_g : float | None, default=None
            The target PGA (in g). If it is not None, it overrides ``factor``.

        Returns
        -------
        scaled_motion : Ground_Motion
            The scaled motion
        """
        if target_PGA_in_g is not None:
            factor = target_PGA_in_g / self.pga_in_g
        else:  # factor != None, and target_PGA_in_g is None
            pass

        time = self.accel[:, 0]
        acc = self.accel[:, 1]
        acc_scaled = acc * factor
        return Ground_Motion(np.column_stack((time, acc_scaled)), unit='m')

    def truncate(  # noqa: PLR0915
            self,
            limit: tuple[float, float],
            *,
            arias: bool = True,
            extend: tuple[float, float] = (0, 0),
            show_fig: bool = False,
    ) -> tuple[Ground_Motion, Figure | None, Axes | None, tuple[int, int]]:
        """
        Truncate ground motion, removing data points in the head and/or tail.

        Parameters
        ----------
        limit : tuple[float, float]
            The lower/upper bounds of time (e.g., [2, 95]) or normalized Arias
            intensity (e.g., [0.05, 0.95]).
        arias : bool, default=True
            If ``True``, ``limit`` means the normalized Arias intensity.
            Otherwise, ``limit`` means the actual time.
        extend : tuple[float, float], default=(0, 0)
            How many seconds to extend before and after the original truncated
            time limits. For example, if extend is [5, 5] sec, and the original
            time limits are [3, 50] sec, then the actual time limits are [0,
            55] sec. (3 - 5 = -2 smaller than 0, so truncated at 0.)
        show_fig : bool, default=False
            Whether to show the waveforms before and after truncation.

        Returns
        -------
        truncated_accel : Ground_Motion
            Truncated ground motion.
        fig : Figure | None
            The figure object being created or being passed into this function.
        ax : Axes | None
            The axes object being created or being passed into this function.
        (n1, n2) : tuple[int, int]
            The indices at which signal is truncated. In other words,
            truncated_accel = original_accel[n1 : n2].

        Raises
        ------
        TypeError
            When the type of ``limit`` or ``extend`` is invalid
        ValueError
            When the value or dimension of ``limit`` or ``extend`` is invalid
        """
        if not isinstance(limit, (tuple, list)):
            raise TypeError('`limit` must be a list/tuple of  two elements.')

        if len(limit) != NUM_ELEMENTS_IN_LIMIT_ARG:
            raise ValueError(
                f'Length of `limit` must be {NUM_ELEMENTS_IN_LIMIT_ARG}.'
            )

        if not isinstance(extend, (tuple, list)):
            raise TypeError('`extend` must be a list/tuple of  two elements.')

        if len(extend) != NUM_ELEMENTS_IN_EXTEND_ARG:
            raise ValueError(
                f'Length of `extend` must be {NUM_ELEMENTS_IN_EXTEND_ARG}.'
            )

        if extend[0] < 0 or extend[1] < 0:
            raise ValueError('`extend` should be non negative.')

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

        n1 = max(n1, 0)

        n2 = min(n2, self.npts)

        time_trunc = self.accel[: n2 - n1, 0]
        accel_trunc = self.accel[n1:n2, 1]
        truncated = np.column_stack((time_trunc, accel_trunc))

        if show_fig:
            ax = [None] * 3
            fig = plt.figure(figsize=(5, 6))
            fig.subplots_adjust(left=0.2)

            ax[0] = fig.add_subplot(3, 1, 1)
            ax[0].plot(
                self.time, self.accel[:, 1], 'gray', lw=1.75, label='original'
            )
            ax[0].plot(
                self.time[n1:n2],
                truncated[:, 1],
                'm',
                lw=1.0,
                label='truncated',
            )
            ax[0].grid(ls=':')
            ax[0].set_ylabel('Accel. [m/s/s]')
            ax[0].legend(loc='best')

            ax[1] = fig.add_subplot(3, 1, 2)
            ax[1].plot(self.time, self.veloc[:, 1], 'gray', lw=1.75)
            ax[1].plot(
                self.time[n1:n2], sr.num_int(truncated)[0][:, 1], 'm', lw=1.0
            )
            ax[1].grid(ls=':')
            ax[1].set_ylabel('Veloc. [m/s]')

            ax[2] = fig.add_subplot(3, 1, 3)
            ax[2].plot(self.time, self.displ[:, 1], 'gray', lw=1.75)
            ax[2].plot(
                self.time[n1:n2], sr.num_int(truncated)[1][:, 1], 'm', lw=1.0
            )
            ax[2].grid(ls=':')
            ax[2].set_ylabel('Displ. [m]')
            ax[2].set_xlabel('Time [sec]')

            fig.tight_layout(pad=0.3)
        else:
            fig, ax = None, None

        return Ground_Motion(truncated, unit='m'), fig, ax, (n1, n2)

    def amplify_by_tf(
            self,
            transfer_function: Frequency_Spectrum,
            *,
            taper: bool = False,
            extrap_tf: bool = True,
            deconv: bool = False,
            show_fig: bool = False,
            dpi: float = 100,
            return_fig_obj: bool = False,
    ) -> tuple[Ground_Motion, Figure | None, Axes | None]:
        """
        Amplify (or de-amplify) ground motions in the frequency domain.

        The mathematical process behind this function is as follows:

            (1) INPUT = fft(input)
            (2) OUTPUT = INPUT * TRANS_FUNC
            (3) output = ifft(OUTPUT)

        Parameters
        ----------
        transfer_function : Frequency_Spectrum
            The transfer function to apply to the ground motion. It only needs
            to be "single-sided" (see notes below).
        taper : bool, default=False
            Whether to taper the input acceleration (using Tukey taper)
        extrap_tf : bool, default=True
            Whether to extrapolate the transfer function if its frequency range
            does not reach the frequency range implied by the input motion
        deconv : bool, default=False
            If ``False``, a regular amplification is performed; otherwise, the
            transfer function is "deducted" from the input motion
            ("deconvolution").
        show_fig : bool, default=False
            Whether to show an illustration of how the calculation is carried
            out.
        dpi : float, default=100
            Desired DPI for the figures; only effective when ``show_fig`` is
            ``True``.
        return_fig_obj : bool, default=False
            Whether to return figure and axis objects to the caller.

        Returns
        -------
        output_motion : Ground_Motion
            The resultant ground motion in time domain
        fig : Figure | None
            The figure object being created or being passed into this function.
        ax : Axes | None
            The axes object being created or being passed into this function.

        Raises
        ------
        TypeError
            When the type of ``transfer_function`` is not valid

        Notes
        -----
        "Single sided": For example, the sampling time interval of
        ``input_motion`` is 0.01 sec, then the Nyquist frequency is 50 Hz.
        Therefore, the transfer function needs to contain information at least
        up to the Nyquist frequency, i.e., at least 0-50 Hz, and anything above
        50 Hz will not affect the input motion at all.
        """
        if not isinstance(transfer_function, Frequency_Spectrum):
            raise TypeError(
                '`transfer_function` needs to be of type '
                '`Frequency_Spectrum` (or its subclass).',
            )

        freq = transfer_function.freq
        tf_1col = transfer_function.spectrum
        transfer_function_single_sided = (freq, tf_1col)
        result = sr.amplify_motion(
            self.accel,
            transfer_function_single_sided,
            taper=taper,
            extrap_tf=extrap_tf,
            deconv=deconv,
            show_fig=show_fig,
            dpi=dpi,
            return_fig_obj=return_fig_obj,
        )
        if return_fig_obj:
            output_accel, fig, ax = result
            return Ground_Motion(output_accel, unit='m'), fig, ax

        output_accel = result[0]
        return Ground_Motion(output_accel, unit='m'), None, None

    def amplify(
            self,
            soil_profile: Vs_Profile,
            boundary: Literal['elastic', 'rigid'] = 'elastic',
            *,
            show_fig: bool = False,
    ) -> Ground_Motion:
        """
        Amplify the ground motion via a 1D soil profile.

        This uses the linear site amplification method.

        Parameters
        ----------
        soil_profile : Vs_Profile
            The soil profile through which to deconvolve the gound motion.
        boundary : Literal['elastic', 'rigid'], default='elastic'
            The type of boundary of the bottom of the soil profile.
        show_fig : bool, default=False
            Whether to show a figure that illustrates the deconvolution
            process.

        Returns
        -------
        Ground_Motion
            The amplified ground motion.

        Raises
        ------
        TypeError
            When the type of ``soil_profile`` is not valid
        """
        if not isinstance(soil_profile, Vs_Profile):
            raise TypeError('`soil_profile` must be of type `Vs_Profile`.')

        vs_profile = soil_profile.vs_profile
        surface_motion = self.accel  # note: unit is SI
        response = sr.linear_site_resp(
            vs_profile,
            surface_motion,
            deconv=False,
            boundary=boundary,
            show_fig=show_fig,
        )[0]
        return Ground_Motion(response, unit='m')

    def compare(
            self,
            another_ground_motion: Ground_Motion,
            *,
            this_ground_motion_as_input: bool = True,
            smooth: bool = True,
            input_accel_label: str = 'Input',
            output_accel_label: str = 'Output',
    ) -> tuple[Figure, Axes]:
        """
        Compare with another ground motion.

        This plots comparison figures showing two time histories and the
        transfer function between them.

        Parameters
        ----------
        another_ground_motion : Ground_Motion
            Another ground motion object.
        this_ground_motion_as_input : bool, default=True
            If ``True``, this ground motion is treated as the input ground
            motion. Otherwise, the other ground motion is treated as the input.
        smooth : bool, default=True
            In the comparison plot, whether to also show the smoothed
            amplification factor.
        input_accel_label : str, default='Input'
            The text label for the input acceleration in the figure legend.
        output_accel_label : str, default='Output'
            The text label for the output acceleration in the figure legend.

        Returns
        -------
        fig : Figure
            The figure object created in this function.
        ax : Axes
            The axes object created in this function.

        Raises
        ------
        TypeError
            When the type of ``another_ground_motion`` is not valid
        """
        if not isinstance(another_ground_motion, Ground_Motion):
            raise TypeError(
                '`another_ground_motion` must be a `Ground_Motion`.'
            )

        if this_ground_motion_as_input:
            accel_in = self.accel
            accel_out = another_ground_motion.accel
        else:
            accel_in = another_ground_motion.accel
            accel_out = self.accel

        amp_ylabel = (
            f'Amplification\n({input_accel_label} ➡ {output_accel_label})'
        )
        phs_ylabel = (
            f'Phase shift [rad]\n({input_accel_label} ➡ {output_accel_label})'
        )

        fig, ax = sr.compare_two_accel(
            accel_in,
            accel_out,
            smooth=smooth,
            input_accel_label=input_accel_label,
            output_accel_label=output_accel_label,
            amplification_ylabel=amp_ylabel,
            phase_shift_ylabel=phs_ylabel,
        )
        return fig, ax

    def deconvolve(
            self,
            soil_profile: Vs_Profile,
            boundary: Literal['elastic', 'rigid'] = 'elastic',
            *,
            show_fig: bool = False,
    ) -> Ground_Motion:
        """
        Deconvolve the ground motion.

        That is, propagate the motion downwards to get the borehole motion
        (rigid boundary) or the "rock outcrop" motion (elastic boundary).

        Parameters
        ----------
        soil_profile : Vs_Profile
            The soil profile through which to deconvolve the gound motion.
        boundary : Literal['elastic', 'rigid'], default='elastic'
            The type of boundary of the bottom of the soil profile.
        show_fig : bool, default=False
            Whether to show a figure that illustrates the deconvolution
            process.

        Returns
        -------
        Ground_Motion
            The deconvolved motion on the rock outcrop or in a borehole.

        Raises
        ------
        TypeError
            When the type of ``soil_profile`` is not valid
        """
        if not isinstance(soil_profile, Vs_Profile):
            raise TypeError('`soil_profile` must be of type `Vs_Profile`.')

        vs_profile = soil_profile.vs_profile
        surface_motion = self.accel  # note: unit is SI
        response = sr.linear_site_resp(
            vs_profile,
            surface_motion,
            deconv=True,
            boundary=boundary,
            show_fig=show_fig,
        )[0]
        return Ground_Motion(response, unit='m')

    def baseline_correct(
            self,
            cutoff_freq: float = 0.20,
            *,
            show_fig: bool = False,
    ) -> Ground_Motion:
        """
        Baseline-correct the acceleration.

        This uses the zero-phase-shift high-pass method.

        Parameters
        ----------
        cutoff_freq : float, default=0.20
            The frequency (unit: Hz) for high passing. Energies below this
            frequency are filtered out.
        show_fig : bool, default=False
            Whether to show figures comparing before and after.

        Returns
        -------
        corrected : Ground_Motion
            The baseline-corrected ground motion, with SI units.
        """
        accel_ = sig.baseline(
            self.accel, show_fig=show_fig, cutoff_freq=cutoff_freq
        )
        return Ground_Motion(accel_, unit='m')

    def lowpass(
            self,
            cutoff_freq: float,
            *,
            show_fig: bool = False,
            filter_order: int = 4,
            padlen: int = 150,
    ) -> Ground_Motion:
        """
        Zero-phase-shift low-pass filtering.

        Parameters
        ----------
        cutoff_freq : float
            Cut-off frequency (unit: Hz).
        show_fig : bool, default=False
            Whether to show a figure of "before vs after"
        filter_order : int, default=4
            Filter order.
        padlen : int, default=150
            Pad length (the number of elements by which to extend x at both
            ends of axis before applying the filter). If None, use the default
            value
            (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html).

        Returns
        -------
        filtered : Ground_Motion
            Filtered signal.
        """
        accel_ = sig.lowpass(
            self.accel,
            cutoff_freq,
            show_fig=show_fig,
            filter_order=filter_order,
            padlen=padlen,
        )
        return Ground_Motion(accel_, unit='m')

    def highpass(
            self,
            cutoff_freq: float,
            *,
            show_fig: bool = False,
            filter_order: int = 4,
            padlen: int = 150,
    ) -> Ground_Motion:
        """
        Zero-phase-shift high-pass filtering.

        Parameters
        ----------
        cutoff_freq : float
            Cut-off frequency (unit: Hz).
        show_fig : bool, default=False
            Whether to show a figure of "before vs after"
        filter_order : int, default=4
            Filter order.
        padlen : int, default=150
            Pad length (the number of elements by which to extend x at both
            ends of axis before applying the filter). If None, use the default
            value
            (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html).

        Returns
        -------
        filtered : Ground_Motion
            Filtered signal.
        """
        accel_ = sig.highpass(
            self.accel,
            cutoff_freq,
            show_fig=show_fig,
            filter_order=filter_order,
            padlen=padlen,
        )
        return Ground_Motion(accel_, unit='m')

    def bandpass(
            self,
            cutoff_freq: tuple[float, float],
            *,
            show_fig: bool = False,
            filter_order: int = 4,
            padlen: int = 150,
    ) -> Ground_Motion:
        """
        Zero-phase-shift band-pass filtering.

        Parameters
        ----------
        cutoff_freq : tuple[float, float]
            Cut-off frequencies (in Hz), from low to high.
        show_fig : bool, default=False
            Whether to show a figure of "before vs after"
        filter_order : int, default=4
            Filter order.
        padlen : int, default=150
            Pad length (the number of elements by which to extend x at both
            ends of axis before applying the filter). If None, use the default
            value
            (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html).

        Returns
        -------
        filtered : Ground_Motion
            Filtered signal
        """
        accel_ = sig.bandpass(
            self.accel,
            cutoff_freq,
            show_fig=show_fig,
            filter_order=filter_order,
            padlen=padlen,
        )
        return Ground_Motion(accel_, unit='m')

    def bandstop(
            self,
            cutoff_freq: tuple[float, float],
            *,
            show_fig: bool = False,
            filter_order: int = 4,
            padlen: int = 150,
    ) -> Ground_Motion:
        """
        Zero-phase-shift band-stop filtering.

        Parameters
        ----------
        cutoff_freq : tuple[float, float]
            Cut-off frequencies (in Hz), from low to high.
        show_fig : bool, default=False
            Whether to show a figure of "before vs after"
        filter_order : int, default=4
            Filter order.
        padlen : int, default=150
            Pad length (the number of elements by which to extend x at both
            ends of axis before applying the filter). If None, use the default
            value
            (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html).

        Returns
        -------
        filtered : Ground_Motion
            Filtered signal
        """
        accel_ = sig.bandstop(
            self.accel,
            cutoff_freq,
            show_fig=show_fig,
            filter_order=filter_order,
            padlen=padlen,
        )
        return Ground_Motion(accel_, unit='m')

    def save_accel(
            self,
            fname: str,
            sep: str = '\t',
            t_prec: str = '%.5g',
            motion_prec: str = '%.5g',
            unit: str = 'm/s/s',
    ) -> None:
        r"""
        Save the acceleration as a text file.

        Parameters
        ----------
        fname : str
            File name (including path).
        sep : str, default='\t'
            Delimiter.
        t_prec : str, default='%.5g'
            The precision specifier for the "time" column.
        motion_prec : str, default='%.5g'
            The precision specifier for the "motion" column.
        unit : str, default='m/s/s'
            What unit shall the exported acceleration be in.
        """
        fmt = [t_prec, motion_prec]
        data = self.accel

        # Keep the plain assignment instead of `/=` or `*=` because `data` can
        # be an integer array, which this assignment casts back to integers,
        # whereas the in-place operators would raise a `TypeError`
        if unit == 'm/s/s':
            pass
        elif unit == 'g':
            data[:, 1] = data[:, 1] / 9.81  # noqa: PLR6104
        elif unit in {'gal', 'cm/s/s'}:
            data[:, 1] = data[:, 1] * 100.0  # noqa: PLR6104

        np.savetxt(fname, data, fmt=fmt, delimiter=sep)

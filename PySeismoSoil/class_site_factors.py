import os
import itertools
import numpy as np
import pkg_resources
from scipy.interpolate import griddata

from PySeismoSoil.class_frequency_spectrum import Frequency_Spectrum


class Site_Factors:
    """
    Class implementation of site response factors proposed by Shi, Asimaki, and
    Graves (2019).

    Parameters
    ----------
    Vs30_in_meter_per_sec : float
        Vs30 values in SI unit.
    z1_in_m : float
        z1 (basin depth) in meters.
    PGA_in_g : float
        PGA in g.
    lenient : bool
        Whether to ensure the given Vs30, z1, and PGA values are within the
        valid range. If False and the given values fall outside the valid
        range, the given values (e.g., Vs30 = 170 m/s) will be treated as
        the closest boundary values (e.g., Vs30 = 175 m/s).

    Attributes
    ----------
    Attributes same as the inputs
    """

    Vs30_array = [
        175,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        800,
        850,
        900,
        950,
    ]
    z1_array = [8, 16, 24, 36, 75, 150, 300, 450, 600, 900]
    PGA_array = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5]

    def __init__(
            self,
            Vs30_in_meter_per_sec,
            z1_in_m,
            PGA_in_g,
            lenient=False,
    ):
        self.dir_amplif = pkg_resources.resource_filename(
            'PySeismoSoil',
            'data/amplification/',
        )
        self.dir_phase = pkg_resources.resource_filename(
            'PySeismoSoil',
            'data/phase/',
        )
        status = Site_Factors._range_check(
            Vs30_in_meter_per_sec,
            z1_in_m,
            PGA_in_g,
        )
        if 'Vs30 out of range' in status:
            if not lenient:
                raise ValueError('Vs30 should be between [175, 950] m/s')
            else:
                Vs30_in_meter_per_sec = 175 if Vs30_in_meter_per_sec < 175 else 950
        if 'z1 out of range' in status:
            if not lenient:
                raise ValueError('z1_in_m should be between [8, 900] m')
            else:
                z1_in_m = 8 if z1_in_m < 8 else 900
        if 'PGA out of range' in status:
            if not lenient:
                raise ValueError('PGA should be between [0.01g, 1.5g]')
            else:
                PGA_in_g = 0.01 if PGA_in_g < 0.01 else 1.5
        if (
            'Invalid Vs30-z1 combination' in status
        ):  # TODO: think about whether to add leniency
            raise ValueError(
                'Vs30 and z1 combination not valid. (The `lenient` '
                'option does not apply to this type of issue.)',
            )

        self.Vs30 = Vs30_in_meter_per_sec
        self.z1 = z1_in_m
        self.PGA = PGA_in_g

    def get_amplification(
            self,
            method='nl_hh',
            Fourier=True,
            show_interp_plots=False,
    ):
        """
        Get site amplification factors.

        Parameters
        ----------
        method : {'nl_hh', 'eq_hh'}
            Which site response simulation method was used to calculate the
            amplification factors. 'nl_hh' uses the results from nonlinear site
            response simulation, which is recommended.
        Fourier : bool
            Whether to return Fourier-spectra-based amplification
            factors (True) or response-spectra based factors (``False``).
        show_interp_plots : bool
            Whether to plot interpolated curve together with the "reference
            curves".

        Returns
        -------
        amplif : PySeismoSoil.class_frequency_spectrum.Frequency_Spectrum
            Amplification factors as a function of frequency.
            (Note: Even if ``Fourier`` is set to ``False``, i.e., the user is
            querying response spectral amplification, the returned result
            is still (freq, amplif). The user can take the reciprocal of
            frequency to get period.)
        """
        if method not in {'nl_hh', 'eq_hh'}:
            raise ValueError("Currently, only 'nl_hh' and 'eq_hh' are valid.")

        period_or_freq, amplif = self._get_results(
            'amplif',
            self.dir_amplif,
            method=method,
            Fourier=Fourier,
            show_interp_plots=show_interp_plots,
        )
        if Fourier:
            freq = period_or_freq
            result = np.column_stack((freq, amplif))
        else:  # response spectra
            freq = 1.0 / period_or_freq
            result = np.column_stack((freq, amplif))[::-1, :]  # so that freq increases
        return Frequency_Spectrum(result)

    def get_phase_shift(self, method='eq_hh', show_interp_plots=False):
        """
        Get site amplification factors

        Parameters
        ----------
        method : {'eq_hh'}
            Which site response simulation method was used to calculate the
            amplification factors. Currently, only 'eq_hh' is valid.
        show_interp_plots : bool
            Whether to plot interpolated curve together with the "reference
            curves".

        Returns
        -------
        phase : PySeismoSoil.class_frequency_spectrum.Frequency_Spectrum
            Phase shift as a function of frequency.
        """
        if method not in {'eq_hh'}:
            raise ValueError("Currently, only 'eq_hh' is valid.")

        freq, phase_shift = self._get_results(
            'phase',
            self.dir_phase,
            method=method,
            Fourier=True,
            show_interp_plots=show_interp_plots,
        )
        return Frequency_Spectrum(np.column_stack((freq, phase_shift)))

    def get_both_amplf_and_phase(self, method='nl_hh', show_interp_plots=False):
        """
        Get both amplification and phase-shift factors

        Parameters
        ----------
        method : {'nl_hh', 'eq_hh'}
            Which site response simulation method was used to calculate the
            amplification factors. 'nl_hh' is recommended.
        show_interp_plots : bool
            Whether to plot interpolated curve together with the "reference
            curves".

        Returns
        -------
        amplif, phase : PySeismoSoil.class_frequency_spectrum.Frequency_Spectrum
            Amplification and phase-shift as functions of frequency.
        """
        amplif = self.get_amplification(
            method=method,
            Fourier=True,
            show_interp_plots=show_interp_plots,
        )
        phase = self.get_phase_shift(
            method='eq_hh',  # always use eq_hh
            show_interp_plots=show_interp_plots,
        )
        return amplif, phase

    def _get_results(
            self,
            amplif_or_phase,
            data_dir,
            method='nl_hh',
            Fourier=True,
            show_interp_plots=False,
    ):
        """
        Get amplification or phase results.

        Parameters
        ----------
        amplif_or_phase : {'amplif', 'phase'}
            Specifies what to query: amplification or phase.
        data_dir : str
            Directory where the csv data files are stored.
        method : {'nl_hh', 'eq_hh', 'eq_kz'}
            Which site response simulation method was used to calculate the
            amplification factors. 'nl_hh' is recommended.
        Fourier : bool
            Whether to return Fourier-spectra-based amplification
            factors (True) or response-spectra based factors (``False``).
        show_interp_plots : bool
            Whether to plot interpolated curve together with the "reference
            curves".

        Returns
        -------
        x : numpy.ndarray
            Frequency or period array.
        y_interp : numpy.ndarray
            Amplification or phase shift, interpolated.
        """
        Vs30 = self.Vs30
        z1 = self.z1
        PGA = self.PGA

        combinations = self._locate_grids()

        points = []  # to hold reference (Vs30, z1, PGA) points
        y_list = []  # to hold values at these reference points
        for Vs30_i, z1_i, PGA_i in combinations:
            Vs30_grid = Site_Factors.Vs30_array[Vs30_i]
            z1_grid = Site_Factors.z1_array[z1_i]
            PGA_grid = Site_Factors.PGA_array[PGA_i]
            x, y = Site_Factors._query(
                amplif_or_phase,
                Vs30_grid,
                z1_grid,
                PGA_grid,
                method=method,
                Fourier=Fourier,
                data_dir=data_dir,
            )
            points.append((Vs30_grid, z1_grid, PGA_grid))
            y_list.append(y)

        y_interp = Site_Factors._interpolate(points, y_list, (Vs30, z1, PGA))

        if Fourier:
            index_trunc = 139  # truncate at frequency = 20 Hz
            x = x[: index_trunc + 1]
            y_interp = y_interp[: index_trunc + 1]
            for ii in range(len(y_list)):
                y_list[ii] = y_list[ii][: index_trunc + 1]

        if show_interp_plots:
            Site_Factors._plot_interp(
                points,
                (Vs30, z1, PGA),
                x,
                y_list,
                y_interp,
                Fourier=Fourier,
            )

        return x, y_interp

    @staticmethod
    def _query(
            amplif_or_phase,
            Vs30,
            z1,
            PGA,
            Fourier=True,
            method='nl_hh',
            data_dir=None,
    ):
        """
        Query amplification or phase factors from pre-computed .csv files. The
        given Vs30, z1_in_m, and PGA_in_g values need to match the
        pre-defined values (see `Vs30_array`, `z1_array`, and `PGA_array`
        at the top of this file).

        Parameters
        ----------
        amplif_or_phase : {'amplif', 'phase'}
            Specifies what to query: amplification or phase.
        Vs30 : scalar
            Vs30 value. Unit: m/s.
        z1 : scalar
            Basin depth (i.e., depth to Vs = 1000 m/s). Unit: m.
        PGA : scalar
            Peak ground acceleration. Unit: g.
        Fourier : bool
            Whether to return Fourier-spectra-based amplification
            factors or response-spectra based factors.
        method : {'nl_hh', 'eq_hh', 'eq_kz'}
            Which site response simulation method was used to calculate the
            amplification factors. 'nl_hh' is recommended.
        data_dir : str
            Directory where the csv data files are stored.

        Returns
        -------
        x : numpy.ndarray
            Period array (for response-spectra-based) or frequency array (for
            Fourier-spectra-based).
        y_values_at_given_PGA : numpy.ndarray
            Amplificatino or phase shift corresponding to each period (or
            frequency).
        """
        if Vs30 not in Site_Factors.Vs30_array:
            raise ValueError('`Vs30` should be in %s.' % Site_Factors.Vs30_array)
        if z1 not in Site_Factors.z1_array:
            raise ValueError('`z1` should be in %s.' % Site_Factors.z1_array)
        if PGA not in Site_Factors.PGA_array:
            raise ValueError('`PGA` should be in %s.' % Site_Factors.PGA_array)

        if method not in ['nl_hh', 'eq_kz', 'eq_hh']:
            raise ValueError("`method` must be within {'nl_hh', 'eq_kz', 'eq_hh'}")

        if amplif_or_phase == 'amplif':
            if Fourier:
                y_filename = '%d_%03d_af_fs_%s_avg.csv' % (Vs30, z1, method)
                x_filename = '%d_%03d_freq.csv' % (Vs30, z1)
            else:  # response spectra
                y_filename = '%d_%03d_af_rs_%s_avg.csv' % (Vs30, z1, method)
                x_filename = '%d_%03d_period.csv' % (Vs30, z1)
        else:  # phase shift
            y_filename = '%d_%03d_phase_shift_%s_avg.csv' % (Vs30, z1, method)
            x_filename = '%d_%03d_freq.csv' % (Vs30, z1)

        y = np.genfromtxt(os.path.join(data_dir, y_filename), delimiter=',')
        x = np.genfromtxt(os.path.join(data_dir, x_filename), delimiter=',')
        PGA_index = np.argwhere(np.array(Site_Factors.PGA_array) == PGA)[0][0]
        y_values_at_given_PGA = y[PGA_index, :]

        return x, y_values_at_given_PGA

    def _locate_grids(self):
        """
        Locates the "reference grids", i.e., rereference Vs30, z1, and PGA
        values (in terms of the indices, not actual values).

        Returns all possible combinations of Vs30, z1, and PGA values.
        """
        Vs30_loc, z1_loc, PGA_loc = Site_Factors._find_neighbors(
            self.Vs30, self.z1, self.PGA,
        )

        combinations = list(itertools.product(Vs30_loc, z1_loc, PGA_loc))
        assert len(list(combinations)) == 8

        return combinations

    @staticmethod
    def _find_neighbors(Vs30_in_mps, z1_in_m, PGA_in_g):
        """
        Find the indices of Vs30, z1, and PGA that surround the provided values.
        If the provided values fall onto the "reference" Vs30, z1, or PGA values,
        two indices are still returned.

        The three inputs need to already within the correct range.
        """
        Vs30_loc = Site_Factors._search_sorted(Vs30_in_mps, Site_Factors.Vs30_array)
        z1_loc = Site_Factors._search_sorted(z1_in_m, Site_Factors.z1_array)
        PGA_loc = Site_Factors._search_sorted(PGA_in_g, Site_Factors.PGA_array)

        return Vs30_loc, z1_loc, PGA_loc

    @staticmethod
    def _search_sorted(value, array):
        """
        Search for the location of `value` within `array`.

        Example behaviors:
            In: _search_sorted(3, [0, 1, 2, 3, 4, 5])
            Out: [2, 3]

            In: _search_sorted(1, [0, 1, 2, 3, 4, 5])
            Out: [0, 1]

            In: _search_sorted(0, [0, 1, 2, 3, 4, 5])
            Out: [0, 1]
        """
        if value < array[0] or value > array[-1]:
            raise ValueError(
                'You have encountered an internal bug. Please '
                'copy the whole error message, and contact '
                'the author of this library for help.',
            )
        if value == array[0]:
            return [0, 1]
        if value == array[-1]:
            return [len(array) - 2, len(array) - 1]

        i = np.searchsorted(array, value, side='left')
        return [i - 1, i]

    @staticmethod
    def _interpolate(ref_points, values, interp_points, method='linear'):
        """
        High-dimensional interpolation.

        Parameters
        ----------
        ref_points : list of tuples
            Coordinates of reference points at which the values are given by
            `values`. Each element of ``ref_points`` is the coordinate of a
            point as a tuple.
        values : list of lists
            Values of interest corresponding to each reference point. There can be
            different versions of values at the reference points (for example, at
            different frequencies, the reference points take on different voltages).

            So the structure of ``values`` shall look like this::

             values =
                 [ [1, 2, 3, 4, ...]  # reference point No.1
                   [2, 3, 4, 5, ...]  # reference point No.2
                   [3, 4, 5, 6, ...]  # reference point No.3
                   ...
                   [9, 10, 11, 12, ...]  # reference point No.X
                 ]   # Each vertical slice is a version of values at the ref. points

        interp_points : list or tuple
            Point at which you want to know the value. Only one point is allow at
            a time.
        method : {'linear', 'nearest', 'cubic'}
            Method of interpolation. See documentation of
            ``scipy.interpolate.griddata``.

        Returns
        -------
        interp_result : numpy.ndarray
            The interpolation result having the same length as the number of
            "versions" in ``values``.
        """
        assert type(ref_points) == list
        assert type(values) == list
        assert isinstance(interp_points, (tuple, list))
        assert len(ref_points) == 8
        assert len(ref_points) == len(values)
        assert len(interp_points) == 3  # 3D coordinate

        values = np.array(values)

        if isinstance(interp_points, list):
            interp_points = tuple(interp_points)

        n = len(values[0])
        interp_result = []
        for i in range(n):
            res = griddata(ref_points, values[:, i], interp_points, method=method)
            interp_result.append(res.flatten()[0])

        return np.array(interp_result)

    @staticmethod
    def _plot_interp(
            ref_points,
            query_point,
            T_or_freq,
            amps,
            amp_interp,
            phases=None,
            phase_interp=None,
            Fourier=True,
    ):
        """
        Show a plot of the amplification and/or phase shift factors at the
        reference (Vs30, z1, PGA) points, as well as the interpolated factors.

        Parameters
        ----------
        ref_points : list<tuples>
            List of tuples of (Vs30, z1, PGA), which are the reference points.
        query_point : tuple
            A tuple of (Vs30, z1, PGA) at which you want to query the factors.
        T_or_freq : numpy.ndarray
            Period or frequency array.
        amps : list of numpy.ndarray
            A list of amplification factors at the reference points. Must have
            the same length as ``ref_points``.
        amp_interp : numpy.ndarray
            Interpolated amplification factor at ``query_point``.
        phases : list of numpy.ndarray (optional)
            A list of phase shift factors at the reference points. Must have
            the same length as ``ref_points``.
        phase_interp : numpy.ndarray
            Interpolated phase shift factor at ``query_point``.
        Fourier : bool
            Whether the amplification factors passed in are the
            Fourier-based factors.

        Return
        ------
        fig, ax1, ax2 OR fig, ax:
            If the user also passes in the phase factors, then two subplots are
            produced, and ``ax1`` and ``ax2`` are the axes objects of the two
            subplots.
        """
        import matplotlib.pyplot as plt

        if phases is not None and phase_interp is not None:
            phase_flag = True
            figsize = (7, 3)
        else:
            phase_flag = False
            figsize = (4, 3)

        alpha = 0.8

        fig = plt.figure(figsize=figsize, dpi=200)
        if phase_flag:
            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)
        else:
            ax = plt.axes()
        for j, ref_point in enumerate(ref_points):
            label = '%d m/s, %d m, %.2gg' % ref_point
            if phase_flag:
                ax1.semilogx(T_or_freq, amps[j], alpha=alpha)
                ax2.semilogx(T_or_freq, phases[j], alpha=alpha, label=label)
            else:
                ax.semilogx(T_or_freq, amps[j], alpha=alpha, label=label)

        if phase_flag:
            ax1.semilogx(T_or_freq, amp_interp, 'k--', lw=2.5)
            ax1.grid(ls=':')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Amplification')

            ax2.plot(T_or_freq, phase_interp, 'k--', lw=2.5, label='Interpolated')
            ax2.grid(ls=':')
            ax2.set_xlabel('Frequency [Hz]')
            ax2.set_ylabel('Phase shift')
        else:
            ax.semilogx(T_or_freq, amp_interp, 'k--', lw=2.5, label='Interpolated')
            ax.grid(ls=':')
            if Fourier:
                ax.set_xlabel('Frequency [Hz]')
            else:
                ax.set_xlabel('Period [sec]')
            ax.set_ylabel('Amplification or phase shift')

        if phase_flag:
            fig.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.3, rect=[0, 0.03, 1, 0.94])
        fig.suptitle('$V_{S30}$ = %d m/s, $z_1$ = %d m, PGA = %.2g$g$' % query_point)

        bbox_anchor_loc = (1.0, 0.02, 1.0, 1.02)
        plt.legend(bbox_to_anchor=bbox_anchor_loc, loc='center left')

        if phase_flag:
            return fig, ax1, ax2
        else:
            return fig, ax

    @staticmethod
    def _range_check(Vs30_in_mps, z1_in_m, PGA_in_g):
        """
        Check if the provided Vs30, z1_in_m, and PGA_in_g values are within
        the pre-computed range.

        The return value (``status``) indicates the kind(s) of errors
        associated with the given input parameters.
        """
        if not isinstance(Vs30_in_mps, (float, int, np.number)):
            raise TypeError('Vs30 must be int, float, or numpy.number.')
        if not isinstance(z1_in_m, (float, int, np.number)):
            raise TypeError('z1_in_m must be int, float, or numpy.number.')
        if not isinstance(PGA_in_g, (float, int, np.number)):
            raise TypeError('PGA_in_g must be int, float, or numpy.number.')

        status = []

        if Vs30_in_mps < 175 or Vs30_in_mps > 950:
            status.append('Vs30 out of range')
        if z1_in_m < 8 or z1_in_m > 900:
            status.append('z1 out of range')
        if PGA_in_g < 0.01 or PGA_in_g > 1.5:
            status.append('PGA out of range')

        if Vs30_in_mps > 400 and z1_in_m > 750:
            status.append('Invalid Vs30-z1 combination')
        elif Vs30_in_mps > 450 and z1_in_m > 600:
            status.append('Invalid Vs30-z1 combination')
        elif Vs30_in_mps > 550 and z1_in_m > 450:
            status.append('Invalid Vs30-z1 combination')
        elif Vs30_in_mps > 600 and z1_in_m > 300:
            status.append('Invalid Vs30-z1 combination')
        elif Vs30_in_mps > 650 and z1_in_m > 150:
            status.append('Invalid Vs30-z1 combination')
        elif Vs30_in_mps > 750 and z1_in_m > 75:
            status.append('Invalid Vs30-z1 combination')
        elif Vs30_in_mps > 800 and z1_in_m > 36:
            status.append('Invalid Vs30-z1 combination')
        elif Vs30_in_mps > 850 and z1_in_m > 16:
            status.append('Invalid Vs30-z1 combination')
        else:
            pass

        return status

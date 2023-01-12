import os
import numpy as np
import matplotlib.pyplot as plt

from PySeismoSoil import helper_generic as hlp
from PySeismoSoil import helper_hh_model as hh
from PySeismoSoil import helper_mkz_model as mkz
from PySeismoSoil import helper_site_response as sr


class Curve:
    """
    Class implementation of a strain-dependent curve. It can be a stress-strain
    curve, a G/Gmax curve as a function of strain, or a damping curve as
    a function of strain.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array with 2 columns. Its 0th column contains the strain
        array, and the 1st column contains the accompanying values (such as
        stress, or G/Gmax).
    strain_unit : {'1', '%'}
        The unit of the strain.
    interpolate : bool
        Whether to interpolate the input curve or not. If ``False``, the
        following several parameters (``min_strain``, ``max_strain``,
        ``n_pts``, ``log_scale``) have no effects.
    min_strain : float
        Minimum strain value of the strain array. If ``interpolate`` is ``True``,
        the raw ``data`` will be internally interpolated at a strain array
        defined by ``min_strain``, ``max_strain``, and ``n_pts``.
    max_strain : float
        Maximum strain value of the strain array. Only effective when
        ``interpolate`` is set to ``True``.
    n_pts : int
        Number of points of the desired strain array to do the interpolation.
        Only effective when ``interpolate`` is set to ``True``.
    log_scale : bool
        Whether the strain array for interpolation is in log scale (or linear
        scale). Only effective when ``interpolate`` is set to ``True``.
    check_values : bool
        Whether to ensure that all values in ``data`` >= 0 when a class object
        is being constructed.

    Attributes
    ----------
    raw_data : numpy.ndarray
        The raw data that the user passed in
    strain : numpy.ndarray
        The strain array at which interpolation happens, a 1D numpy array of
        shape (``n_pts``, ). The unit is percent (unit conversion happens
        internally if applicable).
    values : numpy.ndarray
        The interpolated values; same shape as ``strain``
    """

    def __init__(
            self,
            data,
            *,
            strain_unit='%',
            interpolate=False,
            min_strain=0.0001,
            max_strain=10.0,
            n_pts=50,
            log_scale=True,
            check_values=True,
    ):

        hlp.check_two_column_format(data, '`curve`', ensure_non_negative=check_values)

        if interpolate:
            strain, values = hlp.interpolate(
                min_strain,
                max_strain,
                n_pts,
                data[:, 0],
                data[:, 1],
                log_scale=log_scale,
            )
        else:
            strain, values = data[:, 0], data[:, 1]

        if strain_unit not in ['1', '%']:
            raise ValueError("`strain_unit` must be '1' or '%'.")

        if strain_unit == '1':
            strain *= 100  # strain values are internally stored in unit of %

        self.raw_data = data
        self.strain = strain
        self.values = values

    def __repr__(self):
        return '{} object:\n{}'.format(self.__class__, str(self.raw_data))

    def plot(
            self,
            plot_interpolated=True,
            fig=None,
            ax=None,
            title=None,
            xlabel='Strain [%]',
            ylabel=None,
            figsize=(3, 3),
            dpi=100,
            **kwargs_to_matplotlib,
    ):
        """
        Plot the curve (y axis: values, x axis: strain)

        Parameters
        ----------
        plot_interpolated : bool
            Whether to plot the interpolated curve or the raw data.
        fig : matplotlib.figure.Figure or ``None``
            Figure object. If None, a new figure will be created.
        ax : matplotlib.axes._subplots.AxesSubplot or ``None``
            Axes object. If None, a new axes will be created.
        title : str
            Title of plot.
        xlabel : str
            X label of plot.
        ylabel : str
            Y label of plot.
        figsize: (float, float)
            Figure size in inches, as a tuple of two numbers. The figure
            size of ``fig`` (if not ``None``) will override this parameter.
        dpi : float
            Figure resolution. The dpi of ``fig`` (if not ``None``) will
            override this parameter.
        **kwargs_to_matplotlib :
            Keyword arguments to be passed to ``matplotlib.pyplot.plot()``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object being created or being passed into this function.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object being created or being passed into this function.
        """
        fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize=figsize, dpi=dpi)
        if plot_interpolated:
            ax.semilogx(self.strain, self.values, **kwargs_to_matplotlib)
        else:
            ax.semilogx(
                self.raw_data[:, 0],
                self.raw_data[:, 1],
                **kwargs_to_matplotlib,
            )
        ax.grid(ls=':')
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        return fig, ax


class GGmax_Curve(Curve):
    """
    Class implementation of a G/Gmax curve, as a function of shear strain.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array with 2 columns. Its 0th column contains the strain
        array, and the 1st column contains the G/Gmax values.
    strain_unit : {'1', '%'}
        The unit of the strain.
    interpolate : bool
        Whether to interpolate the input curve or not. If ``False``, the
        following several parameters (``min_strain``, ``max_strain``,
        ``n_pts``, ``log_scale``) have no effects.
    min_strain : float
        Minimum strain value of the strain array. If ``interpolate`` is ``True``,
        the raw ``data`` will be internally interpolated at a strain array
        defined by ``min_strain``, ``max_strain``, and ``n_pts``.
    max_strain : float
        Maximum strain value of the strain array. Only effective when
        ``interpolate`` is set to ``True``.
    n_pts : int
        Number of points of the desired strain array to do the interpolation.
        Only effective when ``interpolate`` is set to ``True``.
    log_scale : bool
        Whether the strain array for interpolation is in log scale (or linear
        scale). Only effective when ``interpolate`` is set to ``True``.
    check_values : bool
        Whether to automatically check the validity of the G/Gmax values (i.e.,
        between 0 and 1).

    Attributes
    ----------
    raw_data : numpy.ndarray
        The raw data that the user passed in.
    strain : numpy.ndarray
        The strain array at which interpolation happens, a 1D numpy array of
        shape (``n_pts``, ). The unit is percent (unit conversion happens
        internally if applicable).
    GGmax : numpy.ndarray
        The interpolated G/Gmax values; same shape as ``strain``.
    """

    def __init__(
            self,
            data,
            *,
            strain_unit='%',
            interpolate=False,
            min_strain=0.0001,
            max_strain=10.0,
            n_pts=50,
            log_scale=True,
            check_values=True,
    ):

        super().__init__(
            data,
            strain_unit=strain_unit,
            interpolate=interpolate,
            min_strain=min_strain,
            max_strain=max_strain,
            n_pts=n_pts,
            log_scale=log_scale,
            check_values=check_values,
        )
        self.GGmax = self.values

        if check_values and (np.any(self.GGmax > 1) or np.any(self.GGmax < 0)):
            raise ValueError('The provided G/Gmax values must be between [0, 1].')


class Damping_Curve(Curve):
    """
    Class implementation of a damping curve, as a function of shear strain.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array with 2 columns. Its 0th column contains the strain
        array, and the 1st column contains the damping values.
    strain_unit : {'1', '%'}
        The unit of the strain.
    damping_unit : {'1', '%'}
        The unit of damping.
    interpolate : bool
        Whether to interpolate the input curve or not. If ``False``, the
        following several parameters (``min_strain``, ``max_strain``,
        ``n_pts``, ``log_scale``) have no effects.
    min_strain : float
        Minimum strain value of the strain array. If ``interpolate`` is ``True``,
        the raw ``data`` will be internally interpolated at a strain array
        defined by ``min_strain``, ``max_strain``, and ``n_pts``.
    max_strain : float
        Maximum strain value of the strain array. Only effective when
        ``interpolate`` is set to ``True``.
    n_pts : int
        Number of points of the desired strain array to do the interpolation.
        Only effective when ``interpolate`` is set to ``True``.
    log_scale : bool
        Whether the strain array for interpolation is in log scale (or linear
        scale). Only effective when ``interpolate`` is set to ``True``.
    check_values : bool
        Whether to automatically check the validity of the damping values (i.e.,
        between 0 and 1).

    Attributes
    ----------
    raw_data : numpy.ndarray
        The raw data that the user passed in.
    strain : numpy.ndarray
        The strain array at which interpolation happens, a 1D numpy array of
        shape (``n_pts``, ). The unit is percent (unit conversion happens
        internally if applicable).
    damping : numpy.ndarray
        The interpolated damping values; same shape as ``strain``. The unit is
        percent (unit conversion happens internally if applicable).
    """

    def __init__(
            self,
            data,
            *,
            strain_unit='%',
            damping_unit='%',
            interpolate=False,
            min_strain=0.0001,
            max_strain=10.0,
            n_pts=50,
            log_scale=True,
            check_values=True,
    ):
        super().__init__(
            data,
            strain_unit=strain_unit,
            interpolate=interpolate,
            min_strain=min_strain,
            max_strain=max_strain,
            n_pts=n_pts,
            log_scale=log_scale,
            check_values=check_values,
        )
        self.damping = self.values

        if damping_unit not in ['1', '%']:
            raise ValueError("`damping_unit` must be '1' or '%'.")

        if damping_unit == '1':
            self.damping *= 100  # unit: 1 --> %

        if check_values and (np.any(self.damping > 100) or np.any(self.damping < 0)):
            raise ValueError('The provided damping values must be between [0, 100].')

    def get_HH_x_param(
            self,
            use_scipy=True,
            pop_size=800,
            n_gen=100,
            lower_bound_power=-4,
            upper_bound_power=6,
            eta=0.1,
            seed=0,
            show_fig=False,
            verbose=False,
            parallel=False,
            n_cores=None,
    ):
        """
        Obtain the HH_x parameters from the damping curve data, using the
        genetic algorithm provided in DEAP.

        Parameters
        ----------
        use_scipy : bool
            Whether to use the "differential_evolution" algorithm implemented
            in scipy
            (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
            to perform optimization. If ``False``, use the algorithm in the
            DEAP package.
        pop_size : int
            The number of individuals in a generation.
        n_gen : int
            Number of generations that the evolution lasts.
        lower_bound_power : float
            The 10-based power of the lower bound of all the 9 parameters. For
            example, if your desired lower bound is 0.26, then set this parameter
            to be numpy.log10(0.26).
        upper_bound_power : float
            The 10-based power of the upper bound of all the 9 parameters.
        eta : float
            Crowding degree of the mutation or crossover. A high ``eta`` will produce
            children resembling to their parents, while a low ``eta`` will produce
            solutions much more different.
        seed : int
            Seed value for the random number generator.
        show_fig : bool
            Whether to show the curve fitting results as a figure.
        verbose : bool
            Whether to display information (statistics of the loss in each
            generation) on the console.
        parallel : bool
            Whether to use parallel computing to simultaneously evaluate different
            individuals in a population. Note that different generations still
            evolve one after another. Only effective for the differential evolution
            for now. Also note that if using parallelization in differential
            evolution, you may need more generations to achieve the same
            optimization loss, because the best solution is being updated only once
            per generation.
        n_cores : int
            Number of CPU cores to use. If ``None``, all cores are used. No
            effects if ``parallel`` is set to ``False``.

        Return
        ------
        HH_x_param : PySeismoSoil.class_parameters.HH_Param
            The best parameters found in the optimization.
        """
        from PySeismoSoil.class_parameters import HH_Param

        HH_x_param = hh.fit_HH_x_single_layer(
            self.raw_data,
            use_scipy=use_scipy,
            pop_size=pop_size,
            n_gen=n_gen,
            lower_bound_power=lower_bound_power,
            upper_bound_power=upper_bound_power,
            eta=eta,
            seed=seed,
            show_fig=show_fig,
            verbose=verbose,
            parallel=parallel,
            n_cores=n_cores,
        )

        self.HH_x_param = HH_Param(HH_x_param)
        return self.HH_x_param

    def get_H4_x_param(
            self,
            use_scipy=True,
            pop_size=800,
            n_gen=100,
            lower_bound_power=-4,
            upper_bound_power=6,
            eta=0.1,
            seed=0,
            show_fig=False,
            verbose=False,
            parallel=False,
            n_cores=None,
    ):
        """
        Obtain the HH_x parameters from the damping curve data, using the
        genetic algorithm provided in DEAP.

        Parameters
        ----------
        use_scipy : bool
            Whether to use the "differential_evolution" algorithm implemented
            in scipy
            (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
            to perform optimization. If ``False``, use the algorithm in the
            DEAP package.
        pop_size : int
            The number of individuals in a generation
        n_gen : int
            Number of generations that the evolution lasts
        lower_bound_power : float
            The 10-based power of the lower bound of all the 9 parameters. For
            example, if your desired lower bound is 0.26, then set this parameter
            to be numpy.log10(0.26)
        upper_bound_power : float
            The 10-based power of the upper bound of all the 9 parameters.
        eta : float
            Crowding degree of the mutation or crossover. A high ``eta`` will produce
            children resembling to their parents, while a low ``eta`` will produce
            solutions much more different.
        seed : int
            Seed value for the random number generator.
        show_fig : bool
            Whether to show the curve fitting results as a figure.
        verbose : bool
            Whether to display information (statistics of the loss in each
            generation) on the console.
        parallel : bool
            Whether to use parallel computing to simultaneously evaluate different
            individuals in a population. Note that different generations still
            evolve one after another. Only effective for the differential evolution
            for now. Also note that if using parallelization in differential
            evolution, you may need more generations to achieve the same
            optimization loss, because the best solution is being updated only once
            per generation.
        n_cores : int
            Number of CPU cores to use. If ``None``, all cores are used. No
            effects if ``parallel`` is set to ``False``.

        Return
        ------
        H4_x_param : PySeismoSoil.class_parameters.MKZ_Param
            The best parameters found in the optimization.
        """
        from PySeismoSoil.class_parameters import MKZ_Param

        H4_x_param = mkz.fit_H4_x_single_layer(
            self.raw_data,
            use_scipy=use_scipy,
            pop_size=pop_size,
            n_gen=n_gen,
            lower_bound_power=lower_bound_power,
            upper_bound_power=upper_bound_power,
            eta=eta,
            seed=seed,
            show_fig=show_fig,
            verbose=verbose,
            parallel=parallel,
            n_cores=n_cores,
        )

        self.H4_x_param = MKZ_Param(H4_x_param)
        return self.H4_x_param


class Stress_Curve(Curve):
    """
    Class implementation of a stress curve, as a function of shear strain.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array with 2 columns. Its 0th column contains the strain
        array, and the 1st column contains the G/Gmax values.
    strain_unit : {'1', '%'}
        The unit of the strain.
    stress_unit : {'Pa', 'kPa', 'MPa', 'GPa'}
        The unit of the stress.
    min_strain : float
        Minimum strain value of the strain array. The raw ``data`` is
        internally interpolated at a strain array defined by ``min_strain``,
        ``max_strain``, and ``n_pts``.
    max_strain : float
        Maximum strain value of the strain array.
    n_pts : int
        Number of points of the desired strain array to do the interpolation.
    log_scale : bool
        Whether the strain array for interpolation is in log scale (or linear
        scale).
    check_values : bool
        Whether to assert that all the stress values are non negative.

    Attributes
    ----------
    raw_data : numpy.ndarray
        The raw data that the user passed in
    strain : numpy.ndarray
        The strain array at which interpolation happens, a 1D numpy array of
        shape (``n_pts``, ). The unit is percent (unit conversion happens
        internally if applicable).
    stress : numpy.ndarray
        The interpolated damping values; same shape as ``strain``. The unit is
        always 'kPa'.
    """

    def __init__(
            self,
            data,
            *,
            strain_unit='1',
            stress_unit='kPa',
            min_strain=0.0001,
            max_strain=10.0,
            n_pts=50,
            log_scale=True,
            check_values=True,
    ):
        super().__init__(
            data,
            strain_unit=strain_unit,
            min_strain=min_strain,
            max_strain=max_strain,
            n_pts=n_pts,
            log_scale=log_scale,
            check_values=check_values,
        )
        self.stress = self.values

        if stress_unit not in ['Pa', 'kPa', 'MPa', 'GPa']:
            raise ValueError("`stress_unit` must be {'Pa', 'kPa', 'MPa', 'GPa'}.")

        if stress_unit == 'Pa':
            self.stress /= 1e3
        elif stress_unit == 'kPa':
            pass
        elif stress_unit == 'MPa':
            self.stress *= 1e3
        else:  # GPa
            self.stress *= 1e6


class Multiple_Curves:
    """
    Class implementation of multiple curves.

    Its behavior is similar to a list, but with a more stringent requirement:
    all elements are of the same data type, i.e., ``element_class``.

    The list-like behaviors available are:
        - indexing: foo[3]
        - slicing: foo[:4]
        - setting values: foo[2] = ...
        - length: len(foo)
        - deleting item: del foo[2]
        - checking existance: bar in foo
        - appending: foo.append(bar)

    Parameters
    ----------
    list_of_curves : list<numpy.ndarray> or list<Curve>
        List of 2-column numpy arrays, which are in (strain [%], curve_value)
        format. Or list of a valid Curve-like type (such as ``GGmax_Curve``).
    element_class : PySeismoSoil.class_curves.Curve or its subclass
        A class name. Each element of ``list_of_curve`` will be used to
        initialize an object of ``element_class``.

    Attributes
    ----------
    element_class : PySeismoSoil.class_curves.Curve or its subclass
        Same as the input parameter.
    curves : list<``element_class``>
        A list of curve objects whose type is specified by the user.
    n_layer : int
        The number of soil layers (i.e., the length of the list).
    """

    def __init__(self, list_of_curves, *, element_class=Curve):
        curves = []
        for curve in list_of_curves:
            if isinstance(curve, np.ndarray):
                curves.append(element_class(curve))
            elif isinstance(curve, element_class):
                curves.append(curve)
            else:
                raise TypeError('An element in `list_of_curves` has invalid type.')

        self.element_class = element_class
        self.curves = curves
        self.n_layer = len(curves)

    def __repr__(self):
        return 'n_layers = %d, type: %s' % (self.n_layer, type(self.curves[0]))

    def __contains__(self, item):
        return item in self.curves

    def __len__(self):
        return self.n_layer

    def __setitem__(self, i, item):
        if not isinstance(item, self.element_class):
            raise TypeError('The new `item` must be of type %s.' % self.element_class)
        self.curves[i] = item

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.curves[i]
        if isinstance(i, slice):  # return an object of the same class
            return self.__class__(self.curves[i])  # filled with the sliced data
        raise TypeError('Indices must be integers or slices, not %s' % type(i))

    def __delitem__(self, i):
        del self.curves[i]
        self.n_layer -= 1

    def append(self, item):
        """Append another curve item to the curves."""
        if not isinstance(item, self.element_class):
            raise TypeError('The new `item` must be of type %s.' % self.element_class)
        self.curves.append(item)
        self.n_layer += 1

    def plot(
            self,
            plot_interpolated=True,
            fig=None,
            ax=None,
            title=None,
            xlabel='Strain [%]',
            ylabel=None,
            figsize=(3, 3),
            dpi=100,
            **kwargs_to_matplotlib,
    ):
        """
        Plot multiple curves together on one figure.

        Parameters
        ----------
        plot_interpolated : bool
            Whether to plot the interpolated curve or the raw data
        fig : matplotlib.figure.Figure or ``None``
            Figure object. If None, a new figure will be created.
        ax : matplotlib.axes._subplots.AxesSubplot or ``None``
            Axes object. If None, a new axes will be created.
        title : str
            Title of plot.
        xlabel : str
            X label of plot.
        ylabel : str
            Y label of plot.
        figsize: (float, float)
            Figure size in inches, as a tuple of two numbers. The figure
            size of ``fig`` (if not ``None``) will override this parameter.
        dpi : float
            Figure resolution. The dpi of ``fig`` (if not ``None``) will override
            this parameter.
        **kwargs_to_matplotlib :
            Keyword arguments to be passed to ``matplotlib.pyplot.plot()``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object being created or being passed into this function.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object being created or being passed into this function.
        """
        fig = plt.figure()
        ax = plt.axes()
        for curve in self.curves:
            curve.plot(
                plot_interpolated=plot_interpolated,
                fig=fig,
                ax=ax,
                figsize=figsize,
                dpi=dpi,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
            )

        return fig, ax


class Multiple_Damping_Curves(Multiple_Curves):
    """
    Class implementation of multiple damping curves.

    Its behavior is similar to a list,
    but with a more stringent requirement: all elements are of the same data
    type, i.e., Damping_Curve.

    The list-like behaviors available are:
        - indexing: foo[3]
        - slicing: foo[:4]
        - setting values: foo[2] = ...
        - length: len(foo)
        - deleting item: del foo[2]
        - checking existance: bar in foo
        - appending: foo.append(bar)

    Parameters
    ----------
    filename_or_list_of_curves : str or list<numpy.ndarray>
        A file name of a validly formatted "curve file", or a list of 2-column
        numpy arrays, which are in (strain [%], damping [%]) format.
    sep : str
        Delimiter of the file to be imported. If ``filename_or_list_of_curves``
        is a list, this parameter has no effect.

    Attributes
    ----------
    curves : list<Damping_Curve>
        A list of Damping_Curve objects.
    n_layer : int
        The number of soil layers (i.e., the length of the list).
    """

    def __init__(self, filename_or_list_of_curves, *, sep='\t'):
        if isinstance(filename_or_list_of_curves, str):  # file name
            curves = np.genfromtxt(filename_or_list_of_curves, delimiter=sep)
            _, list_of_damping_curves = hlp.extract_from_curve_format(curves)
            self._filename = filename_or_list_of_curves
        elif isinstance(filename_or_list_of_curves, list):
            list_of_damping_curves = filename_or_list_of_curves
            self._filename = None
        else:
            raise TypeError('Unrecognized type for `filename_or_list_of_curves`.')

        self._sep = sep

        super().__init__(
            list_of_damping_curves,
            element_class=Damping_Curve,
        )

    def plot(
            self,
            plot_interpolated=True,
            fig=None,
            ax=None,
            title=None,
            xlabel='Strain [%]',
            ylabel='Damping [%]',
            figsize=(3, 3),
            dpi=100,
            **kwargs_to_matplotlib,
    ):
        """
        Plot multiple curves together on one figure.

        Parameters
        ----------
        plot_interpolated : bool
            Whether to plot the interpolated curve or the raw data.
        fig : matplotlib.figure.Figure or ``None``
            Figure object. If None, a new figure will be created.
        ax : matplotlib.axes._subplots.AxesSubplot or ``None``
            Axes object. If None, a new axes will be created.
        title : str
            Title of plot.
        xlabel : str
            X label of plot.
        ylabel : str
            Y label of plot.
        figsize: (float, float)
            Figure size in inches, as a tuple of two numbers. The figure
            size of ``fig`` (if not ``None``) will override this parameter.
        dpi : float
            Figure resolution. The dpi of ``fig`` (if not ``None``) will override
            this parameter.
        **kwargs_to_matplotlib :
            Keyword arguments to be passed to ``matplotlib.pyplot.plot()``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object being created or being passed into this function.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object being created or being passed into this function.
        """
        fig, ax = super().plot(
            plot_interpolated=plot_interpolated,
            fig=fig,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            dpi=dpi,
            **kwargs_to_matplotlib,
        )

        return fig, ax

    def get_curve_matrix(
            self,
            GGmax_filler_value=1.0,
            save_to_file=False,
            full_file_name=None,
    ):
        """
        Produce a full "curve matrix" based on the damping data defined in
        objects of this class.

        The "curve matrix" will be in the following format:
            +------------+--------+------------+-------------+-------------+--------+-----+
            | strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ... |
            +============+========+============+=============+=============+========+=====+
            |    ...     |  ...   |    ...     |    ...      |    ...      |  ...   | ... |
            +------------+--------+------------+-------------+-------------+--------+-----+

        Since this class only defines damping curves, not G/Gmax curves,
        G/Gmax will be filled with some dummy values.

        Parameters
        ----------
        GGmax_filler_value : float
            A dummy value to fill all the G/Gmax curves.
        save_to_file : bool
            Whether to save the "curve matrix" as a text file.
        full_file_name : str or ``None``
            Full file name to save to the hard drive. It can be ``None`` if
            ``save_to_file`` is set to ``False``.

        Returns
        -------
        curve_matrix : numpy.ndarray
            A matrix containing damping curves in the above-mentioned format.
        """
        lengths = []  # lengths of strain array of each layer
        for curve_ in self.curves:
            lengths.append(len(curve_.strain))

        max_length = np.max(lengths)

        curve_matrix = None
        for curve_ in self.curves:
            strain = curve_.strain
            if len(curve_.strain) == max_length:
                strain_ = strain  # we can use the original strain array
                damping_ = curve_.damping
            else:  # otherwise we need a new strain array to match `max_length`
                strain_ = np.geomspace(np.min(strain), np.max(strain), max_length)
                damping_ = np.interp(strain_, strain, curve_.damping)
            # END IF
            GGmax = np.ones_like(strain_) * GGmax_filler_value
            tmp_matrix = np.column_stack((strain_, GGmax, strain_, damping_))
            if curve_matrix is None:
                curve_matrix = tmp_matrix
            else:
                curve_matrix = np.column_stack((curve_matrix, tmp_matrix))
            # END IF
        # END FOR
        return curve_matrix

    def get_all_HH_x_params(
            self,
            use_scipy=True,
            pop_size=800,
            n_gen=100,
            lower_bound_power=-4,
            upper_bound_power=6,
            eta=0.1,
            seed=0,
            show_fig=False,
            verbose=False,
            parallel=False,
            n_cores=None,
            save_txt=False,
            txt_filename=None,
            sep=None,
            save_fig=False,
            fig_filename=None,
            dpi=100,
    ):
        """
        Obtain the HH_x parameters from the damping curve data (for all the
        curves), using the genetic algorithm provided in DEAP.

        Parameters
        ----------
        use_scipy : bool
            Whether to use the "differential_evolution" algorithm implemented
            in scipy
            (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
            to perform optimization. If ``False``, use the algorithm in the
            DEAP package.
        pop_size : int
            The number of individuals in a generation.
        n_gen : int
            Number of generations that the evolution lasts.
        lower_bound_power : float
            The 10-based power of the lower bound of all the 9 parameters. For
            example, if your desired lower bound is 0.26, then set this parameter
            to be numpy.log10(0.26).
        upper_bound_power : float
            The 10-based power of the upper bound of all the 9 parameters.
        eta : float
            Crowding degree of the mutation or crossover. A high ``eta`` will produce
            children resembling to their parents, while a low ``eta`` will produce
            solutions much more different.
        seed : int
            Seed value for the random number generator.
        show_fig : bool
            Whether to show the curve fitting results as a figure.
        verbose : bool
            Whether to display information (statistics of the loss in each
            generation) on the console.
        parallel : bool
            Whether to use parallel computing across layers, i.e., calculate
            multiple layers simultaneously.
        n_cores : int
            Number of CPU cores to use. If None, all cores are used. No effects
            if the parallelization options are set to ``False``.
        save_txt : bool
            Whether to save the results as a "HH_x_STATION_NAME.txt" file.
        txt_filename : str
            File name of the text file to save HH parameters. If the object is
            created via a "curve" text file, then `txt_filename` can be None
            and the output filename will be determined automatically.
        sep : str
            Delimiter to separate columns of data in the output file.
        save_fig : bool
            Whether to save damping fitting figures to hard drive.
        fig_filename : str
            Full file name of the figure. If the object is created via a
            "curve" text file, then `fig_filename` can be None, and the
            output figure name will be determined automatically.
        dpi : float
            Figure resolution.

        Return
        ------
        HH_x_param : PySeismoSoil.class_parameters.HH_Param_Multi_Layer
            The best parameters for each soil layer found in the optimization.
        """
        from PySeismoSoil.class_parameters import HH_Param_Multi_Layer

        if save_fig and fig_filename is None:
            fig_filename = self._produce_output_file_name('HH', 'png')

        if save_txt:
            if txt_filename is None:
                txt_filename = self._produce_output_file_name('HH', 'txt')
            if sep is None:
                sep = self._sep

        list_of_np_array = [_.raw_data for _ in self.curves]
        params = sr.fit_all_damping_curves(
            list_of_np_array,
            hh.fit_HH_x_single_layer,
            hh.tau_HH,
            use_scipy=use_scipy,
            pop_size=pop_size,
            n_gen=n_gen,
            lower_bound_power=lower_bound_power,
            upper_bound_power=upper_bound_power,
            eta=eta,
            seed=seed,
            show_fig=show_fig,
            verbose=verbose,
            parallel=parallel,
            n_cores=n_cores,
            save_fig=save_fig,
            fig_filename=fig_filename,
            dpi=dpi,
            save_txt=save_txt,
            txt_filename=txt_filename,
            sep=sep,
            func_serialize=hh.serialize_params_to_array,
        )

        return HH_Param_Multi_Layer(params)

    def get_all_H4_x_params(
            self,
            use_scipy=True,
            pop_size=800,
            n_gen=100,
            lower_bound_power=-4,
            upper_bound_power=6,
            eta=0.1,
            seed=0,
            show_fig=False,
            verbose=False,
            parallel=False,
            n_cores=None,
            save_txt=False,
            txt_filename=None,
            sep=None,
            save_fig=False,
            fig_filename=None,
            dpi=100,
    ):
        """
        Obtain the H4_x parameters from the damping curve data (for all the
        curves), using the genetic algorithm provided in DEAP.

        Parameters
        ----------
        use_scipy : bool
            Whether to use the "differential_evolution" algorithm implemented
            in scipy
            (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
            to perform optimization. If ``False``, use the algorithm in the
            DEAP package.
        pop_size : int
            The number of individuals in a generation.
        n_gen : int
            Number of generations that the evolution lasts.
        lower_bound_power : float
            The 10-based power of the lower bound of all the 9 parameters. For
            example, if your desired lower bound is 0.26, then set this parameter
            to be numpy.log10(0.26).
        upper_bound_power : float
            The 10-based power of the upper bound of all the 9 parameters.
        eta : float
            Crowding degree of the mutation or crossover. A high ``eta`` will produce
            children resembling to their parents, while a low ``eta`` will produce
            solutions much more different.
        seed : int
            Seed value for the random number generator.
        show_fig : bool
            Whether to show the curve fitting results as a figure.
        verbose : bool
            Whether to display information (statistics of the loss in each
            generation) on the console.
        parallel : bool
            Whether to use parallel computing across layers, i.e., calculate
            multiple layers simultaneously.
        n_cores : int
            Number of CPU cores to use. If None, all cores are used. No effects
            if the parallelization options are set to ``False``.
        save_txt : bool
            Whether to save the results as a "HH_x_STATION_NAME.txt" file.
        txt_filename : str
            File name of the text file to save HH parameters. If the object is
            created via a "curve" text file, then `txt_filename` can be ``None``
            and the output filename will be determined automatically.
        sep : str
            Delimiter to separate columns of data in the output file.
        save_fig : bool
            Whether to save damping fitting figures to hard drive.
        fig_filename : str
            Full file name of the figure. If the object is created via a
            "curve" text file, then ``fig_filename`` can be None, and the
            output figure name will be determined automatically.
        dpi : float
            Figure resolution

        Return
        ------
        H4_x_param : PySeismoSoil.class_parameters.H4_Param_Multi_Layer
            The best parameters for each soil layer found in the optimization.
        """
        from PySeismoSoil.class_parameters import MKZ_Param_Multi_Layer

        if save_fig and fig_filename is None:
            fig_filename = self._produce_output_file_name('H4', 'png')

        if save_txt:
            if txt_filename is None:
                txt_filename = self._produce_output_file_name('H4', 'txt')
            if sep is None:
                sep = self._sep

        list_of_np_array = [_.raw_data for _ in self.curves]
        params = sr.fit_all_damping_curves(
            list_of_np_array,
            mkz.fit_H4_x_single_layer,
            mkz.tau_MKZ,
            use_scipy=use_scipy,
            pop_size=pop_size,
            n_gen=n_gen,
            lower_bound_power=lower_bound_power,
            upper_bound_power=upper_bound_power,
            eta=eta,
            seed=seed,
            show_fig=show_fig,
            verbose=verbose,
            parallel=parallel,
            n_cores=n_cores,
            save_fig=save_fig,
            fig_filename=fig_filename,
            dpi=dpi,
            save_txt=save_txt,
            txt_filename=txt_filename,
            sep=sep,
            func_serialize=mkz.serialize_params_to_array,
        )

        return MKZ_Param_Multi_Layer(params)

    def _produce_output_file_name(self, prefix, extension):
        """
        Produce the output file name.

        Parameters
        ----------
        prefix : {'HH', 'H4'} or str
            Prefix of file name.
        extension : {'png', 'txt'} or str
            File extension (without the dot).

        Returns
        -------
        new_file_name : str
            The new file name based on the input "curve" file name.
        """
        if self._filename is None:
            raise ValueError(
                'Please make sure to create this object from '
                'a text file, so that there is an original file '
                'name to work with.',
            )

        path_name, file_name = os.path.split(self._filename)
        file_name_, _ = os.path.splitext(file_name)
        if 'curve_' in file_name_:
            site_name = file_name_[6:]
        else:
            site_name = file_name_

        new_file_name = '{}_x_{}.{}'.format(prefix, site_name, extension)

        return new_file_name


class Multiple_GGmax_Curves(Multiple_Curves):
    """
    Class implementation of multiple G/Gmax curves.

    Its behavior is similar to a list,
    but with a more stringent requirement: all elements are of the same data
    type, i.e., GGmax_Curve.

    The list-like behaviors available are:
        - indexing: foo[3]
        - slicing: foo[:4]
        - setting values: foo[2] = ...
        - length: len(foo)
        - deleting item: del foo[2]
        - checking existance: bar in foo
        - appending: foo.append(bar)

    Parameters
    ----------
    filename_or_list_of_curves : str or list<numpy.ndarray>
        A file name of a validly formatted "curve file", or a list of 2-column
        numpy arrays, which are in (strain [%], G/Gmax) format.
    sep : str
        Delimiter of the file to be imported. If ``filename_or_list_of_curves``
        is a list, this parameter has no effect.

    Attributes
    ----------
    curves : list<GGmax_Curve>
        A list of GGmax_Curve objects.
    n_layer : int
        The number of soil layers (i.e., the length of the list).
    """

    def __init__(self, filename_or_list_of_curves, *, sep='\t'):
        if isinstance(filename_or_list_of_curves, str):  # file name
            curves = np.genfromtxt(filename_or_list_of_curves, delimiter=sep)
            list_of_GGmax_curves, _ = hlp.extract_from_curve_format(curves)
            self._filename = filename_or_list_of_curves
        elif isinstance(filename_or_list_of_curves, list):
            list_of_GGmax_curves = filename_or_list_of_curves
            self._filename = None
        else:
            raise TypeError('Unrecognized type for `filename_or_list_of_curves`.')

        self._sep = sep

        super().__init__(
            list_of_GGmax_curves,
            element_class=GGmax_Curve,
        )

    def plot(
            self,
            plot_interpolated=True,
            fig=None,
            ax=None,
            title=None,
            xlabel='Strain [%]',
            ylabel='G/Gmax',
            figsize=(3, 3),
            dpi=100,
            **kwargs_to_matplotlib,
    ):
        """
        Plot multiple curves together on one figure.

        Parameters
        ----------
        plot_interpolated : bool
            Whether to plot the interpolated curve or the raw data.
        fig : matplotlib.figure.Figure or ``None``
            Figure object. If None, a new figure will be created.
        ax : matplotlib.axes._subplots.AxesSubplot or ``None``
            Axes object. If None, a new axes will be created.
        title : str
            Title of plot.
        xlabel : str
            X label of plot.
        ylabel : str
            Y label of plot.
        figsize: (float, float)
            Figure size in inches, as a tuple of two numbers. The figure
            size of ``fig`` (if not ``None``) will override this parameter.
        dpi : float
            Figure resolution. The dpi of ``fig`` (if not ``None``) will override
            this parameter.
        **kwargs_to_matplotlib :
            Keyword arguments to be passed to ``matplotlib.pyplot.plot()``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object being created or being passed into this function.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object being created or being passed into this function.
        """
        fig, ax = super().plot(
            plot_interpolated=plot_interpolated,
            fig=fig,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            dpi=dpi,
            **kwargs_to_matplotlib,
        )

        return fig, ax

    def get_curve_matrix(
            self,
            damping_filler_value=1.0,
            save_to_file=False,
            full_file_name=None,
    ):
        """
        Produce a full "curve matrix" based on the G/Gmax data defined in
        objects of this class.

        The full "curve matrix" will be in the following format:
            +------------+--------+------------+-------------+-------------+--------+-----+
            | strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ... |
            +============+========+============+=============+=============+========+=====+
            |    ...     |  ...   |    ...     |    ...      |    ...      |  ...   | ... |
            +------------+--------+------------+-------------+-------------+--------+-----+

        Since this class only defines G/Gmax curves, not damping curves,
        damping values will be filled with some dummy values.

        Parameters
        ----------
        damping_filler_value : float
            A dummy value to fill all the damping curves.
        save_to_file : bool
            Whether to save the "curve matrix" as a text file.
        full_file_name : str or ``None``
            Full file name to save to the hard drive. It can be ``None`` if
            ``save_to_file`` is set to ``False``.

        Returns
        -------
        curve_matrix : numpy.ndarray
            A matrix containing G/Gmax curves in the above-mentioned format.
        """
        lengths = []  # lengths of strain array of each layer
        for curve_ in self.curves:
            lengths.append(len(curve_.strain))

        max_length = np.max(lengths)

        curve_matrix = None
        for curve_ in self.curves:
            strain = curve_.strain
            if len(curve_.strain) == max_length:
                strain_ = strain  # we can use the original strain array
                GGmax_ = curve_.GGmax
            else:  # otherwise we need a new strain array to match `max_length`
                strain_ = np.geomspace(np.min(strain), np.max(strain), max_length)
                GGmax_ = np.interp(strain_, strain, curve_.GGmax)
            # END IF
            damping = np.ones_like(strain_) * damping_filler_value
            tmp_matrix = np.column_stack((strain_, GGmax_, strain_, damping))
            if curve_matrix is None:
                curve_matrix = tmp_matrix
            else:
                curve_matrix = np.column_stack((curve_matrix, tmp_matrix))
            # END IF
        # END FOR
        return curve_matrix


class Multiple_GGmax_Damping_Curves:
    """
    A "parent" class that holds both G/Gmax curves and damping curves
    information. The user can EITHER initialize this class by providing
    instances of ``Multiple_GGmax_Curves`` and ``Multiple_Damping_Curves``
    classes, OR by providing a numpy array containing the curves. (The user
    can provide one and only one input parameter, and leave the other parameter
    to ``None``.)

    Parameters
    ----------
    mgc_and_mdc : (Multiple_GGmax_Curves, Multiple_Damping_Curves) or ``None``
        A tuple of two elements, which are the G/Gmax curve information and
        the damping curve information, respectively. The two objects needs to
        have the same ``n_layer`` attribute.
    data : numpy.ndarray, str, or ``None``
        A 2D numpy array of the following format:
            +------------+--------+------------+-------------+-------------+--------+-----+
            | strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ... |
            +============+========+============+=============+=============+========+=====+
            |    ...     |  ...   |    ...     |    ...      |    ...      |  ...   | ... |
            +------------+--------+------------+-------------+-------------+--------+-----+

        Or a full name of a text file containing the 2D array.

    Attributes
    ----------
    mgc : Multiple_GGmax_Curves
        Object containing information of G/Gmax curves. It will be ``None`` if
        ``mgc_and_mdc`` is not provided.
    mdc : Multiple_Damping_Curves
        Object containing information of damping curves. It will be ``None`` if
        ``mgc_and_mdc`` is not provided.
    data : numpy.ndarray
        2D numpy array containing the curve information in the format shown
        above. It will be ``None`` if the ``data`` is not provided.
    n_layer : int
        Number of soil layers.
    """

    def __init__(self, *, mgc_and_mdc=None, data=None):
        if mgc_and_mdc is None and data is None:
            raise ValueError(
                'Both parameters are `None`. Please provide '
                'one and only one input parameter.',
            )
        if mgc_and_mdc is not None and data is not None:
            raise ValueError(
                'Both parameters are not `None`. Please provide '
                'one and only one input parameter.',
            )
        if mgc_and_mdc is not None:
            if not isinstance(mgc_and_mdc, tuple):
                raise TypeError('`mgc_and_mdc` needs to be a tuple.')
            if len(mgc_and_mdc) != 2:
                raise ValueError('Length of `mgc_and_mdc` needs to be 2.')
            if not isinstance(mgc_and_mdc[0], Multiple_GGmax_Curves):
                raise TypeError(
                    'The 0th element of `mgc_and_mdc` needs to '
                    'be of type `Multiple_GGmax_Curves`.',
                )
            if not isinstance(mgc_and_mdc[-1], Multiple_Damping_Curves):
                raise TypeError(
                    'The last element of `mgc_and_mdc` needs to '
                    'be of type `Multiple_Damping_Curves`.',
                )
            self.mgc = mgc_and_mdc[0]
            self.mdc = mgc_and_mdc[-1]
            self.data = None
            if self.mgc.n_layer != self.mdc.n_layer:
                raise ValueError(
                    'The ``Multiple_GGmax_Curves`` instance and '
                    'the ``Multiple_Damping_Curves`` instance '
                    'need to have the same number of soil layers.',
                )
            self.n_layer = self.mgc.n_layer
        else:  # `data` is not `None`
            if not isinstance(data, (np.ndarray, str)):
                raise TypeError('`data` must be a 2D numpy array or a file name.')
            if isinstance(data, str):
                data = np.genfromtxt(data)
            self.mgc = None
            self.mdc = None
            hlp.assert_2D_numpy_array(data, name='`data`')
            if data.shape[1] % 4 != 0:
                raise ValueError(
                    'The number of columns of `data` needs '
                    'to be a multiple of 4. However, your '
                    '`data` has %d columns.' % data.shape[1],
                )
            self.data = data
            self.n_layer = data.shape[1] // 4

    def get_MGC_MDC_objects(self):
        """
        Get ``Multiple_GGmax_Curves`` and ``Multiple_Damping_Curves`` objects.

        Returns
        -------
        mgc : Multiple_GGmax_Curves
            Object containing information of G/Gmax curves.
        mdc : Multiple_Damping_Curves
            Object containing information of damping curves.
        """
        if self.mgc is not None:
            return self.mgc, self.mdc
        else:  # the user provides a matrix containing curve information
            GGmax_curve_list, damping_curves_list = hlp.extract_from_curve_format(
                self.data,
                ensure_non_negative=False,
            )
            mgc = Multiple_GGmax_Curves(GGmax_curve_list)
            mdc = Multiple_Damping_Curves(damping_curves_list)
            return mgc, mdc

    def get_curve_matrix(self):
        """
        Get a "curve matrix" containing both G/Gmax and damping information.

        Returns
        -------
        curve_matrix : numpy.ndarray
            A 2D numpy array with the following format::
                +------------+--------+------------+-------------+-------------+--------+-----+
                | strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ... |
                +============+========+============+=============+=============+========+=====+
                |    ...     |  ...   |    ...     |    ...      |    ...      |  ...   | ... |
                +------------+--------+------------+-------------+-------------+--------+-----+
        """
        if self.data is not None:
            return self.data
        else:
            mgc_matrix = self.mgc.get_curve_matrix()
            mdc_matrix = self.mdc.get_curve_matrix()
            return hlp.merge_curve_matrices(mgc_matrix, mdc_matrix)

    def plot(self):
        """Plot the G/Gmax and damping curves."""
        mgc, mdc = self.get_MGC_MDC_objects()
        mgc.plot(ylabel=r'$G/G_{\max}$')
        mdc.plot(ylabel='Damping [%]')

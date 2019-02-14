# Author: Jian Shi

import os
import numpy as np
import matplotlib.pyplot as plt

from . import helper_generic as hlp
from . import helper_hh_model as hh

#%%============================================================================
class Curve():
    '''
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
        The unit of the strain
    min_strain, max_strain : float
        Minimum and maximum strain value of the strain array. The raw `data` is
        internally interpolated at a strain array defined by `min_strain`,
        `max_strain`, and `n_pts`.
    n_pts : int
        Number of points of the desired strain array to do the interpolation
    log_scale : bool
        Whether the strain array for interpolation is in log scale (or linear
        scale)
    ensure_non_negative : bool
        Whether to ensure that all values in `data` >= 0 when a class object
        is being constructed

    Attributes
    ----------
    raw_data : numpy.ndarray
        The raw data that the user passed in
    strain : numpy.array
        The strain array at which interpolation happens, a 1D numpy array of
        shape (n_pts, ). The unit is percent (unit conversion happens internally
        if applicable).
    values : numpy.array
        The interpolated values; same shape as `strain`
    '''
    def __init__(self, data, strain_unit='%', min_strain=0.0001, max_strain=10.,
                 n_pts=50, log_scale=True, ensure_non_negative=True):

        hlp.check_two_column_format(data, '`curve`',
                                    ensure_non_negative=ensure_non_negative)

        strain, values = hlp.interpolate(min_strain, max_strain, n_pts,
                                         data[:, 0], data[:, 1],
                                         log_scale=log_scale)

        if strain_unit not in ['1', '%']:
            raise ValueError("`strain_unit` must be '1' or '%'.")

        if strain_unit == '1':
            strain *= 100  # strain values are internally stored in unit of %

        self.raw_data = data
        self.strain = strain
        self.values = values

    def __repr__(self):
        return '%s object:\n%s' % (self.__class__, str(self.raw_data))

    def plot(self, plot_interpolated=True, title=None, xlabel='Strain [%]',
             ylabel=None, figsize=None, dpi=100, **kwargs_to_matplotlib):
        '''
        Plot the curve (y axis: values, x axis: strain)

        Parameter
        ---------
        plot_interpolated : bool
            Whether to plot the interpolated curve or the raw data
        title : str
            Title of plot
        xlabel : str
            X label of plot
        ylabel : str
            Y label of plot
        figsize : tuple
            Figure size
        dpi : int
            DPI of plot
        **kwargs_to_matplotlib :
            Keyword arguments to be passed to matplotlib.pyplot.plot()

        Returns
        -------
        fig, ax :
            matplotlib objects of the figure and the axes
        '''
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes()
        if plot_interpolated:
            ax.semilogx(self.strain, self.values, **kwargs_to_matplotlib)
        else:
            ax.semilogx(self.raw_data[:, 0], self.raw_data[:, 1],
                        **kwargs_to_matplotlib)
        ax.grid(ls=':')
        ax.set_xlabel('Strain [%]')
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        return fig, ax

#%%============================================================================
class GGmax_Curve(Curve):
    '''
    Class implementation of a G/Gmax curve, as a function of shear strain.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array with 2 columns. Its 0th column contains the strain
        array, and the 1st column contains the G/Gmax values.
    strain_unit : {'1', '%'}
        The unit of the strain
    min_strain, max_strain : float
        Minimum and maximum strain value of the strain array. The raw `data` is
        internally interpolated at a strain array defined by `min_strain`,
        `max_strain`, and `n_pts`.
    n_pts : int
        Number of points of the desired strain array to do the interpolation
    log_scale : bool
        Whether the strain array for interpolation is in log scale (or linear
        scale)
    check_values : bool
        Whether to automatically check the validity of the G/Gmax values (i.e.,
        between 0 and 1)

    Attributes
    ----------
    raw_data : numpy.ndarray
        The raw data that the user passed in
    strain : numpy.array
        The strain array at which interpolation happens, a 1D numpy array of
        shape (n_pts, ). The unit is percent (unit conversion happens internally
        if applicable).
    GGmax : numpy.array
        The interpolated G/Gmax values; same shape as `strain`
    '''
    def __init__(self, data, strain_unit='%', min_strain=0.0001, max_strain=10.,
                 n_pts=50, log_scale=True, check_values=True):

        super(GGmax_Curve, self).__init__(data, strain_unit=strain_unit,
                                          min_strain=min_strain,
                                          max_strain=max_strain,
                                          n_pts=n_pts, log_scale=log_scale)
        self.GGmax = self.values
        del self.values

        if check_values and np.any(self.GGmax > 1) or np.any(self.GGmax < 0):
            raise ValueError('The provided G/Gmax values must be between [0, 1].')

#%%============================================================================
class Damping_Curve(Curve):
    '''
    Class implementation of a damping curve, as a function of shear strain.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array with 2 columns. Its 0th column contains the strain
        array, and the 1st column contains the G/Gmax values.
    strain_unit : {'1', '%'}
        The unit of the strain
    damping_unit : {'1', '%'}
        The unit of damping
    min_strain, max_strain : float
        Minimum and maximum strain value of the strain array. The raw `data` is
        internally interpolated at a strain array defined by `min_strain`,
        `max_strain`, and `n_pts`.
    n_pts : int
        Number of points of the desired strain array to do the interpolation
    log_scale : bool
        Whether the strain array for interpolation is in log scale (or linear
        scale)
    check_values : bool
        Whether to automatically check the validity of the damping values (i.e.,
        between 0 and 1)

    Attributes
    ----------
    raw_data : numpy.ndarray
        The raw data that the user passed in
    strain : numpy.array
        The strain array at which interpolation happens, a 1D numpy array of
        shape (n_pts, ). The unit is percent (unit conversion happens internally
        if applicable).
    damping : numpy.array
        The interpolated damping values; same shape as `strain`. The unit is
        percent (unit conversion happens internally if applicable).
    '''

    def __init__(self, data, strain_unit='%', damping_unit='%',
                 min_strain=0.0001, max_strain=10., n_pts=50, log_scale=True,
                 check_values=True):

        super(Damping_Curve, self).__init__(data, strain_unit=strain_unit,
                                            min_strain=min_strain,
                                            max_strain=max_strain,
                                            n_pts=n_pts, log_scale=log_scale)
        self.damping = self.values
        del self.values

        if damping_unit not in ['1', '%']:
            raise ValueError("`damping_unit` must be '1' or '%'.")

        if damping_unit == '1':
            self.damping *= 100  # unit: 1 --> %

        if check_values and np.any(self.damping > 100) or np.any(self.damping < 0):
            raise ValueError('The provided damping values must be between [0, 100].')

    #--------------------------------------------------------------------------
    def obtain_HH_x_param(self, population_size=800, n_gen=100,
                          lower_bound_power=-4, upper_bound_power=6, eta=0.1,
                          seed=0, show_fig=False, verbose=False):
        '''
        Obtain the HH_x parameters from the damping curve data, using the
        genetic algorithm provided in DEAP.

        Parameters
        ----------
        population_size : int
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
            Crowding degree of the mutation or crossover. A high eta will produce
            children resembling to their parents, while a small eta will produce
            solutions much more different.
        seed : int
            Seed value for the random number generator
        show_fig : bool
            Whether to show the curve fitting results as a figure
        verbose : bool
            Whether to display information (statistics of the loss in each
            generation) on the console

        Return
        ------
        HH_x_param : PySeismoSoil.class_parameters.HH_Param
            The best parameters found in the optimization
        '''

        from class_parameters import HH_Param

        damping_curve = np.column_stack((self.strain, self.damping))
        HH_x_param = hh.fit_HH_x_single_layer(damping_curve,
                                              population_size=population_size,
                                              n_gen=n_gen,
                                              lower_bound_power=lower_bound_power,
                                              upper_bound_power=upper_bound_power,
                                              eta=eta, seed=seed,
                                              show_fig=show_fig, verbose=verbose)

        self.HH_x_param = HH_Param(HH_x_param)
        return self.HH_x_param

#%%============================================================================
class Stress_Curve(Curve):
    '''
    Class implementation of a damping curve, as a function of shear strain.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array with 2 columns. Its 0th column contains the strain
        array, and the 1st column contains the G/Gmax values.
    strain_unit : {'1', '%'}
        The unit of the strain
    stress_unit : {'Pa', 'kPa', 'MPa', 'GPa'}
        The unit of the stress
    min_strain, max_strain : float
        Minimum and maximum strain value of the strain array. The raw `data` is
        internally interpolated at a strain array defined by `min_strain`,
        `max_strain`, and `n_pts`.
    n_pts : int
        Number of points of the desired strain array to do the interpolation
    log_scale : bool
        Whether the strain array for interpolation is in log scale (or linear
        scale)
    check_values : bool
        Whether to automatically check the validity of the G/Gmax values (i.e.,
        >= 0)

    Attributes
    ----------
    raw_data : numpy.ndarray
        The raw data that the user passed in
    strain : numpy.array
        The strain array at which interpolation happens, a 1D numpy array of
        shape (n_pts, ). The unit is percent (unit conversion happens internally
        if applicable).
    stress : numpy.array
        The interpolated damping values; same shape as `strain`. The unit is
        always 'kPa'.
    '''
    def __init__(self, data, strain_unit='1', stress_unit='kPa',
                 min_strain=0.0001, max_strain=10., n_pts=50, log_scale=True,
                 check_values=True):

        super(Stress_Curve, self).__init__(data, strain_unit=strain_unit,
                                           min_strain=min_strain,
                                           max_strain=max_strain,
                                           n_pts=n_pts, log_scale=log_scale,
                                           ensure_non_negative=check_values)
        self.stress = self.values
        del self.values

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

#%%============================================================================
class Multiple_Curves():
    '''
    Class implementation of multiple curves.

    Its behavior is similar to a list,
    but with a more stringent requirement: all elements are of the same data
    type, i.e., `element_class`.

    The list-like behaviors available are:
        - indexing: foo[3]
        - slicing: foo[:4]
        - setting values: foo[2] = ...
        - length: len(foo)
        - deleting iterm: del foo[2]
        - checking existance: bar in foo

    Parameters
    ----------
    list_of_curves : list<numpy.ndarray> or list<Curve>
        List of 2-column numpy arrays, which are in (strain [%], curve_value)
        format. Or list of a valid Curve-like type (such as GGmax_Curve).
    element_class : PySeismoSoil.class_curves.Curve or its subclass
        A class name. Each element of `list_of_curve` will be used to
        initialize an object of `element_class`.

    Attributes
    ----------
    curves : list<`element_class`>
        A list of curve objects whose type is specified by the user
    n_layer : int
        The number of soil layers (i.e., the length of the list)
    '''
    def __init__(self, list_of_curves, element_class=Curve):
        curves = []
        for curve in list_of_curves:
            if isinstance(curve, np.ndarray):
                curves.append(element_class(curve))
            elif isinstance(curve, element_class):
                curves.append(curve)
            else:
                raise TypeError('An element in `list_of_curves` has invalid '
                                'type.')

        self.element_class = element_class
        self.curves = curves
        self.n_layer = len(curves)

    def __repr__(self):
        return 'n_layers = %d, type: %s' % (self.n_layer, type(self.curves[0]))

    def __contains__(self, item): return item in self.curves
    def __len__(self): return self.n_layer
    def __setitem__(self, i, item): self.curves[i] = item
    def __getitem__(self, i):
        if isinstance(i, int):
            return self.curves[i]
        if isinstance(i, slice):  # return an object of the same class
            return self.__class__(self.curves[i])  # filled with the sliced data
        raise TypeError('Indices must be integers or slices, not %s' % type(i))
    def __delitem__(self, i):
        del self.curves[i]
        self.n_layer -= 1

#%%============================================================================
class Multiple_Damping_Curves(Multiple_Curves):
    '''
    Class implementation of multiple damping curves.

    Its behavior is similar to a list,
    but with a more stringent requirement: all elements are of the same data
    type, i.e., Damping_Curve.

    The list-like behaviors available are:
        - indexing: foo[3]
        - slicing: foo[:4]
        - setting values: foo[2] = ...
        - length: len(foo)
        - deleting iterm: del foo[2]
        - checking existance: bar in foo

    Parameters
    ----------
    filename_or_list_of_curves : str or list<numpy.ndarray>
        A file name of a validly formatted "curve file", or a list of 2-column
        numpy arrays, which are in (strain [%], damping [%]) format
    sep : str
        Delimiter of the file to be imported. If `filename_or_list_of_curves`
        is a list, `sep` has no effect.

    Attributes
    ----------
    curves : list<Damping_Curve>
        A list of Damping_Curve objects
    n_layer : int
        The number of soil layers (i.e., the length of the list)
    '''

    def __init__(self, filename_or_list_of_curves, sep='\t'):

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

        super(Multiple_Damping_Curves, self).__init__(list_of_damping_curves,
                                                      Damping_Curve)

    def obtain_HH_x_param(self, population_size=800, n_gen=100,
                          lower_bound_power=-4, upper_bound_power=6, eta=0.1,
                          seed=0, show_fig=False, verbose=False,
                          parallel=False, n_cores=None, save_file=False):
        '''
        Obtain the HH_x parameters from the damping curve data, using the
        genetic algorithm provided in DEAP.

        Parameters
        ----------
        population_size : int
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
            Crowding degree of the mutation or crossover. A high eta will produce
            children resembling to their parents, while a small eta will produce
            solutions much more different.
        seed : int
            Seed value for the random number generator
        show_fig : bool
            Whether to show the curve fitting results as a figure
        verbose : bool
            Whether to display information (statistics of the loss in each
            generation) on the console
        parallel : bool
            Whether to use parallel computing for each soil layer
        n_cores : int
            Number of CPU cores to use. If None, all cores are used. No effects
            if `parallel` is set to False.
        save_file : bool
            Whether to save the results as a "HH_X_STATION_NAME.txt" file

        Return
        ------
        HH_x_param : PySeismoSoil.class_parameters.HH_Param_Multi_Layer
            The best parameters for each soil layer found in the optimization
        '''
        from .class_parameters import HH_Param_Multi_Layer

        list_of_np_array = [_.raw_data for _ in self.curves]
        params = hh.fit_HH_x_multi_layers(list_of_np_array,
                                          population_size=population_size,
                                          n_gen=n_gen,
                                          lower_bound_power=lower_bound_power,
                                          upper_bound_power=upper_bound_power,
                                          eta=eta, seed=seed,
                                          show_fig=show_fig, verbose=verbose,
                                          parallel=parallel, n_cores=n_cores)

        if save_file:
            path_name, file_name = os.path.split(self._filename)
            file_name_, extension = os.path.splitext(file_name)
            if 'curve_' in file_name_:
                site_name = file_name_[6:]
            else:
                site_name = file_name_
            new_file_name = 'HH_x_%s.%s' % (site_name, extension)

            data_for_file = []
            for param in params:
                data_for_file.append(hh.serialize_params_to_array(param))

            data_for_file__ = np.column_stack(tuple(data_for_file))
            np.savetxt(os.path.join(path_name, new_file_name), data_for_file__,
                       fmt='%.6g', delimiter=self._sep)

        return HH_Param_Multi_Layer(params)

#%%============================================================================
class Multiple_GGmax_Curves(Multiple_Curves):
    '''
    Class implementation of multiple G/Gmax curves.

    Its behavior is similar to a list,
    but with a more stringent requirement: all elements are of the same data
    type, i.e., GGmax_Curve.

    The list-like behaviors available are:
        - indexing: foo[3]
        - slicing: foo[:4]
        - setting values: foo[2] = ...
        - length: len(foo)
        - deleting iterm: del foo[2]
        - checking existance: bar in foo

    Parameters
    ----------
    filename_or_list_of_curves : str or list<numpy.ndarray>
        A file name of a validly formatted "curve file", or a list of 2-column
        numpy arrays, which are in (strain [%], G/Gmax) format
    sep : str
        Delimiter of the file to be imported. If `filename_or_list_of_curves`
        is a list, `sep` has no effect.

    Attributes
    ----------
    curves : list<GGmax_Curve>
        A list of GGmax_Curve objects
    n_layer : int
        The number of soil layers (i.e., the length of the list)
    '''

    def __init__(self, filename_or_list_of_curves, sep='\t'):

        if isinstance(filename_or_list_of_curves, str):  # file name
            curves = np.genfromtxt(filename_or_list_of_curves, delimiter=sep)
            _, list_of_GGmax_curves = hlp.extract_from_curve_format(curves)
            self._filename = filename_or_list_of_curves
        elif isinstance(filename_or_list_of_curves, list):
            list_of_GGmax_curves = filename_or_list_of_curves
            self._filename = None
        else:
            raise TypeError('Unrecognized type for `filename_or_list_of_curves`.')

        self._sep = sep

        super(Multiple_GGmax_Curves, self).__init__(list_of_GGmax_curves,
                                                    GGmax_Curve)

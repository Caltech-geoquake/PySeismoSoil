# Author: Jian Shi

import collections
import numpy as np
import matplotlib.pyplot as plt

from . import helper_generic as hlp
from . import helper_hh_model as hh

#%%============================================================================
class HH_Param(collections.UserDict):
    '''
    Class implementation of the HH model parameters. After initialization, you
    can access/modify individual parameter values just like a dictionary.

    Parameters
    ----------
    param_dict : dict
        Values of the HH model parameters. Acceptable key names are:
            gamma_t, a, gamma_ref, beta, s, Gmax, nu, Tmax, d
    a, gamma_ref, beta, s, Gmax, mu, Tmax, d : float
        Alternatively (if you do not provide `param_dict`), use these keyword
        arguments to provide HH parameter values.

    Attributes
    ----------
    data : dict
        HH model parameters with the keys listed above
    '''
    def __init__(self, param_dict):
        self.allowable_keys = {'gamma_t', 'a', 'gamma_ref', 'beta', 's', 'Gmax',
                               'mu', 'Tmax', 'd'}
        if param_dict.keys() != self.allowable_keys:
            raise ValueError("Invalid keys exist in your input data. We only "
                             "allow {'gamma_t', 'a', 'gamma_ref', 'beta', 's', "
                             "'Gmax', 'mu', 'Tmax', 'd'}.")

        super(HH_Param, self).__init__(param_dict)

    def __repr__(self):
        return self.param

    def __setitem__(self, key, val):
        if key not in self.allowable_keys:
            raise KeyError("The HH model doesn't have a '%s' parameter." % key)
        self.data[key] = val
        return None

    def get_stress(self, strain_array=np.logspace(-2, 1)):
        '''
        Get the shear stress array inferred from the set of parameters

        Parameter
        ---------
        strain_array : numpy.ndarray
            Must be 1D numpy array. Unit: %

        Returns
        -------
        The shear stress array, with the same shape as the strain array. Its
        unit is identical to the unit of Gmax (one of the HH parameters).
        '''
        hlp.assert_1D_numpy_array(strain_array, name='`strain_array`')
        return hh.tau_HH(strain_array / 100., **self.data)

    def get_GGmax(self, strain_array=np.logspace(-2, 1)):
        '''
        Get the G/Gmax array inferred from the set of parameters

        Parameter
        ---------
        strain_array : numpy.ndarray
            Must be 1D numpy array. Unit: %

        Returns
        -------
        The G/Gmax array, with the same shape as the strain array
        '''
        T_HH = self.get_stress(strain_array=strain_array)
        Gmax = self.data['Gmax']
        GGmax = T_HH / Gmax
        return GGmax

    def get_damping(self, strain_array=np.logspace(-2, 1)):
        '''
        Get the damping array inferred from the set of parameters

        Parameter
        ---------
        strain_array : numpy.ndarray
            Must be 1D numpy array. Unit: %

        Returns
        -------
        The damping array, with the same shape as the strain array
        '''
        return hh.calc_damping_from_param(self.data, strain_array/100.) * 100

    def plot_curves(self, figsize=None, dpi=100, **kwargs_to_matplotlib):
        '''
        Plot G/Gmax and damping curves from the HH parameters

        Parameter
        ---------
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
        strain = np.logspace(-2, 1)  # unit: percent
        GGmax = self.get_GGmax(strain)
        damping = self.get_damping(strain)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = [None, None]
        ax[0] = plt.subplot(121)
        ax[0].semilogx(strain, GGmax)
        ax[0].grid(ls=':')
        ax[0].set_xlabel('Strain [%]')
        ax[0].set_ylabel('G/Gmax')

        ax[1] = plt.subplot(122)
        ax[1].semilogx(strain, damping)
        ax[1].grid(ls=':')
        ax[1].set_xlabel('Strain [%]')
        ax[1].set_ylabel('Damping [%]')

        return fig, ax

#%%============================================================================
class Param_Multi_Layer():
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
    list_of_param_data : list<dict> or list<Param>
        List of dict or a list of valid parameter class (such as HH_Param),
        which contain data for parameters of each layer
    element_class : PySeismoSoil.class_parameters.HH_Param_Single_Layer et al
        A class name. Each element of `list_of_param_dict` will be used to
        initialize an object of `element_class`.

    Attributes
    ----------
    param_list : list<`element_class`>
        A list of param objects whose type is specified by the user
    n_layer : int
        The number of soil layers (i.e., the length of the list)
    '''

    def __init__(self, list_of_param_data, element_class):
        param_list = []
        for param_data in list_of_param_data:
            if isinstance(param_data, dict):
                param_list.append(element_class(param_data))
            elif isinstance(param_data, element_class):
                param_list.append(param_data)
            else:
                raise TypeError('An element in `list_of_param_data` has '
                                'invalid type.')
        self.param_list = param_list
        self.n_layer = len(param_list)

    def __contains__(self, item): return item in self.param_list
    def __len__(self): return self.n_layer
    def __setitem__(self, i, item): self.param_list[i] = item
    def __getitem__(self, i):
        if isinstance(i, int):
            return self.param_list[i]
        if isinstance(i, slice):  # return an object of the same class
            return self.__class__(self.param_list[i])  # filled with the sliced data
        raise TypeError('Indices must be integers or slices, not %s' % type(i))
    def __delitem__(self, i):
        del self.param_list[i]
        self.n_layer -= 1

#%%============================================================================
class HH_Param_Multi_Layer(Param_Multi_Layer):
    '''
    Class implementation of multiple sets of HH parameters for multiple layers.

    Its behavior is similar to a list,
    but with a more stringent requirement: all elements are of the same data
    type, i.e., HH_Param.

    The list-like behaviors available are:
        - indexing: foo[3]
        - slicing: foo[:4]
        - setting values: foo[2] = ...
        - length: len(foo)
        - deleting iterm: del foo[2]
        - checking existance: bar in foo

    Parameters
    ----------
    filename_or_list : str or list<dict> or list<HH_Param>
        A file name of a validly formatted parameter file, or a list containing
        HH parameter data
    sep : str
        Delimiter of the file to be imported. If `filename_or_list_of_curves`
        is a list, `sep` has no effect.

    Attributes
    ----------
    param_list : list<HH_Param>
        A list of HH model parameters
    n_layer : int
        The number of soil layers (i.e., the length of the list)
    '''

    def __init__(self, filename_or_list, sep='\t'):
        if isinstance(filename_or_list, str):  # file name
            self._filename = filename_or_list
            params = np.genfromtxt(filename_or_list, delimiter=sep)
            list_of_param_array = hlp.extract_from_param_format(params)
            list_of_param = [hh.deserialize_array_to_params(_)
                             for _ in list_of_param_array]
        elif isinstance(filename_or_list, list):
            self._filename = None
            list_of_param = filename_or_list
        else:
            raise TypeError('Unrecognized type for `filename_or_list`.')

        self._sep = sep

        super(HH_Param_Multi_Layer, self).__init__(list_of_param, HH_Param)


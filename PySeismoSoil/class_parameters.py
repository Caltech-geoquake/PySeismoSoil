# Author: Jian Shi

import os
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
    def __init__(self, param_dict):#, gamma_t=None, a=None, gamma_ref=None,
                 #beta=None, s=None, Gmax=None, mu=None, Tmax=None, d=None):

#        param_list = [gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d]
#        is_not_None = [_ is not None for _ in param_list]
#        is_a_number = [isinstance(_, (np.number, float, int)) for _ in param_list]
#
#        if param_dict is not None and any(is_not_None):
#            raise ValueError('If you provide `param_dict`, please do not pass '
#                             'any value to the other keyword arguments.')
#        if param_dict is None and not all(is_a_number):
#            raise ValueError('You did not provide `param_dict`, therefore you '
#                             'need to provide values to all other keyword '
#                             'arguments.')

        self.allowable_keys = {'gamma_t', 'a', 'gamma_ref', 'beta', 's', 'Gmax',
                               'nu', 'Tmax', 'd'}
        if param_dict.keys() != self.allowable_keys:
            raise ValueError("Invalid keys exist in your input data. We only "
                             "allow {'gamma_t', 'a', 'gamma_ref', 'beta', 's', "
                             "'Gmax', 'nu', 'Tmax', 'd'}.")

        super(HH_Param, self).__init__(param_dict)


#        self.param = dict()
#        if param_dict is not None:
#            self.param['gamma_t'] = param_dict['gamma_t']
#            self.param['a'] = param_dict['a']
#            self.param['gamma_ref'] = param_dict['gamma_ref']
#            self.param['beta'] = param_dict['beta']
#            self.param['s'] = param_dict['s']
#            self.param['Gmax'] = param_dict['Gmax']
#            self.param['mu'] = param_dict['mu']
#            self.param['Tmax'] = param_dict['Tmax']
#            self.param['d'] = param_dict['d']
#        else:
#            self.param['gamma_t'] = gamma_t
#            self.param['a'] = a
#            self.param['gamma_ref'] = gamma_ref
#            self.param['beta'] = beta
#            self.param['s'] = s
#            self.param['Gmax'] = Gmax
#            self.param['mu'] = mu
#            self.param['Tmax'] = Tmax
#            self.param['d'] = d

    def __repr__(self):
        return self.param

    def __setitem__(self, key, val):
        if key not in self.allowable_keys:
            raise KeyError("The key '%s' is not permitted." % key)
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

    def __init__(self, list_of_param_data, *, element_class):
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

#%%============================================================================
class HH_Param_Multi_Layer(Param_Multi_Layer):
    '''
    Class implementation of multiple sets of HH parameters for multiple layers.

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


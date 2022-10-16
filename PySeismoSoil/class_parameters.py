import collections
import numpy as np
import matplotlib.pyplot as plt

from PySeismoSoil import helper_generic as hlp
from PySeismoSoil import helper_hh_model as hh
from PySeismoSoil import helper_mkz_model as mkz
from PySeismoSoil import helper_site_response as sr

from PySeismoSoil.class_curves import Multiple_GGmax_Damping_Curves

STRAIN_RANGE_PCT = np.logspace(-2, 1)


class Parameter(collections.UserDict):
    """
    Class implementation of parameters for different soil constitutive models.
    Its objects have dictionary-like behaviors. The keys in the dictionary are
    the parameter names (such as "gamma_ref", "Gmax").

    Inheriting from this base class provides you the ability to control the
    allowable keys that a potential user may provide, making it safer for
    subsequent computations.

    Parameters
    ----------
    param_dict : dict
        Name-value pairs of the parameters.
    allowable_keys : set<str>
        The allowable parameter names of the constitutive model.
    func_stress : Python function
        A function to calculate shear stress from the parameters.

    Attributes
    ----------
    data : dict
        The original data, stored as a regular dictionary.
    allowable_keys : set<str>
        Same as the input parameter.
    """

    def __init__(self, param_dict, *, allowable_keys=None, func_stress=None):
        if not isinstance(param_dict, dict):
            raise TypeError('`param_dict` must be a dictionary.')
        if not isinstance(allowable_keys, set) or any(
            type(_) != str for _ in allowable_keys
        ):
            raise TypeError('`allowable_keys` should be a set of str.')
        if param_dict.keys() != allowable_keys:
            raise KeyError(
                'Invalid keys exist in your input data. We only '
                'allow %s.' % allowable_keys,
            )
        self.allowable_keys = allowable_keys
        self.func_stress = func_stress
        super().__init__(param_dict)

    def __repr__(self):
        import json

        return json.dumps(self.data, indent=2).replace('"', '')

    def __setitem__(self, key, item):
        if key not in self.allowable_keys:
            raise KeyError("The model does not have a '%s' parameter." % key)
        self.data[key] = item
        return None

    def __delitem__(self, key):
        raise ValueError('Deleting items from the parameter set is not allowed.')

    def serialize(self):
        """
        Serialize the parameter values into an array of floats. The order of
        the parameters are arbitrary, so any subclass of this class is
        recommended to override this method.

        Returns
        -------
        result : numpy.ndarray
            Serialized parameters.
        """
        param_array = []
        for _, val in self.data.items():
            param_array.append(val)
        return np.array(param_array)

    def get_stress(self, strain_in_pct=STRAIN_RANGE_PCT):
        """
        Get the shear stress array inferred from the set of parameters

        Parameters
        ----------
        strain_in_pct : numpy.ndarray
            Strain array. Must be a 1D numpy array. Unit: %

        Returns
        -------
        result : numpy.ndarray
            The shear stress array, with the same shape as the strain array.
            Its unit is identical to the unit of Gmax (one of the HH parameters).
        """
        if self.func_stress is None:
            print('You did not provide a function to calculate shear stress.')
            return None
        hlp.assert_1D_numpy_array(strain_in_pct, name='`strain_in_pct`')
        return self.func_stress(strain_in_pct / 100.0, **self.data)

    def get_GGmax(self, strain_in_pct=STRAIN_RANGE_PCT):
        """
        Get the G/Gmax array inferred from the set of parameters

        Parameters
        ----------
        strain_in_pct : numpy.ndarray
            Strain array. Must be a 1D numpy array. Unit: %

        Returns
        -------
        result : numpy.ndarray
            The G/Gmax array, with the same shape as the strain array.
        """
        tau = self.get_stress(strain_in_pct=strain_in_pct)
        if tau is None:
            print('You did not provide a function to calculate shear stress.')
            return None
        Gmax = self.data['Gmax']
        strain_in_1 = strain_in_pct / 100.0
        GGmax = sr.calc_GGmax_from_stress_strain(strain_in_1, tau, Gmax=Gmax)
        return GGmax

    def get_damping(self, strain_in_pct=STRAIN_RANGE_PCT):
        """
        Get the damping array inferred from the set of parameters

        Parameters
        ----------
        strain_in_pct : numpy.ndarray
            Strain array. Must be a 1D numpy array. Unit: %

        Returns
        -------
        result : numpy.ndarray
            The damping array (unit: %), with the same shape as the strain array
        """
        if self.func_stress is None:
            print('You did not provide a function to calculate shear stress.')
            return None
        damping_in_1 = sr.calc_damping_from_param(
            self.data,
            strain_in_pct / 100.0,
            self.func_stress,
        )
        return damping_in_1 * 100

    def plot_curves(self, figsize=None, dpi=100, **kwargs_to_matplotlib):
        """
        Plot G/Gmax and damping curves from the model parameters

        Parameters
        ----------
        figsize: (float, float)
            Figure size in inches, as a tuple of two numbers. If ``None``, use
            (3, 6).
        dpi : float
            Figure resolution. If ``None``, use 100.
        **kwargs_to_matplotlib :
            Keyword arguments to be passed to ``matplotlib.pyplot.plot()``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : list<matplotlib.axes._subplots.AxesSubplot>
            A list of two axes objects.
        """
        strain = np.logspace(-4, 1)  # unit: percent
        GGmax = self.get_GGmax(strain)
        damping = self.get_damping(strain)

        if figsize is None:
            figsize = (6, 3)
        if dpi is None:
            dpi = 100

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

        fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.5)

        return fig, ax


class HH_Param(Parameter):
    """
    Class implementation of the HH model parameters. After initialization, you
    can access/modify individual parameter values just like a dictionary.

    Parameters
    ----------
    param_dict : dict
        Values of the HH model parameters. Acceptable key names are:
            gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d

    Attributes
    ----------
    data : dict
        HH model parameters with the keys listed above.
    allowable_keys : set<str>
        Valid parameter names of the HH model.
    """

    def __init__(self, param_dict):
        allowable_keys = {
            'gamma_t',
            'a',
            'gamma_ref',
            'beta',
            's',
            'Gmax',
            'mu',
            'Tmax',
            'd',
        }
        super().__init__(
            param_dict,
            func_stress=hh.tau_HH,
            allowable_keys=allowable_keys,
        )

    def serialize(self):
        """
        Return an array of parameter values in the order of:
        {'gamma_t', 'a', 'gamma_ref', 'beta', 's', 'Gmax', 'mu', 'Tmax', 'd'}
        """
        return hh.serialize_params_to_array(self.data)


class MKZ_Param(Parameter):
    """
    Class implementation of the MKZ model parameters. After initialization, you
    can access/modify individual parameter values just like a dictionary.

    Parameters
    ----------
    param_dict : dict
        Values of the HH model parameters. Acceptable key names are:
            gamma_ref, s, beta, Gmax

    Attributes
    ----------
    data : dict
        MKZ model parameters with the keys listed above.
    allowable_keys : set<str>
        Valid parameter names of the MKZ model.
    """

    def __init__(self, param_dict):
        allowable_keys = {'gamma_ref', 's', 'beta', 'Gmax'}
        super().__init__(
            param_dict,
            func_stress=mkz.tau_MKZ,
            allowable_keys=allowable_keys,
        )

    def serialize(self):
        """
        Return an array of parameter values in the order of:
        {'gamma_ref', 's', 'beta', 'Gmax'}
        """
        return mkz.serialize_params_to_array(self.data)


class Param_Multi_Layer:
    """
    Class implementation of multiple curves.

    Its behavior is similar to a list,
    but with a more stringent requirement: all elements are of the same data
    type, i.e., ``element_class``.

    The list-like behaviors available are:
        - indexing: foo[3]
        - slicing: foo[:4]
        - setting values: foo[2] = ...
        - length: len(foo)
        - deleting item: del foo[2]
        - checking existance: bar in foo

    Parameters
    ----------
    list_of_param_data : list<dict> or list<Param>
        List of dict or a list of valid parameter class (such as ``HH_Param``),
        which contain data for parameters of each layer.
    element_class : PySeismoSoil.class_parameters.HH_Param_Single_Layer et al
        A class name, such as ``HH_Param``. Each element of ``list_of_param_dict``
        will be used to initialize an object of ``element_class``.

    Attributes
    ----------
    param_list : list<``element_class``>
        A list of param objects whose type is specified by the user.
    n_layer : int
        The number of soil layers (i.e., the length of the list).
    """

    def __init__(self, list_of_param_data, *, element_class):
        param_list = []
        for param_data in list_of_param_data:
            if isinstance(param_data, dict):
                param_list.append(element_class(param_data))
            elif isinstance(param_data, element_class):
                param_list.append(param_data)
            else:
                raise TypeError(
                    'An element in ``list_of_param_data`` has invalid type.',
                )
        self.param_list = param_list
        self.n_layer = len(param_list)

    def __contains__(self, item):
        return item in self.param_list

    def __len__(self):
        return self.n_layer

    def __setitem__(self, i, item):
        self.param_list[i] = item

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.param_list[i]
        if isinstance(i, slice):  # return an object of the same class
            return self.__class__(self.param_list[i])  # filled with the sliced data
        raise TypeError('Indices must be integers or slices, not %s' % type(i))

    def __delitem__(self, i):
        del self.param_list[i]
        self.n_layer -= 1

    def construct_curves(self, strain_in_pct=STRAIN_RANGE_PCT):
        """
        Construct G/Gmax and damping curves from parameter values.

        Parameters
        ----------
        strain_in_pct : numpy.ndarray
            Strain array. Must be a 1D numpy array. Unit: %

        Returns
        -------
        mgc : PySeismoSoil.class_curves.Multiple_GGmax_Curves
            G/Gmax curves for each soil layer.
        mdc : PySeismoSoil.class_curves.Multiple_Damping_Curves
            Damping curves for each soil layer.
        """
        curves = None
        for param in self.param_list:
            GGmax = param.get_GGmax(strain_in_pct=strain_in_pct)
            damping = param.get_damping(strain_in_pct=strain_in_pct)
            if curves is None:
                curves = np.column_stack((strain_in_pct, GGmax, strain_in_pct, damping))
            else:
                curves = np.column_stack(
                    (curves, strain_in_pct, GGmax, strain_in_pct, damping),
                )

        mgdc = Multiple_GGmax_Damping_Curves(data=curves)
        mgc, mdc = mgdc.get_MGC_MDC_objects()
        return mgc, mdc

    def serialize_to_2D_array(self):
        """
        Serielizes the parameter data to a 2D numpy array.

        Returns
        -------
        param_2D_array : numpy.ndarray
            A 2D numpy array whose columns are parameters of each layer.
        """
        output = []
        for param_single_layer in self.param_list:
            param_array = param_single_layer.serialize()
            output.append(param_array)

        param_2D_array = np.array(output).T
        return param_2D_array

    def save_txt(self, filename, precision='%.5g', sep='\t', **kw_to_savetxt):
        """
        Save data as text file.

        Parameters
        ----------
        filename : str
            File name (including path) of the output file.
        precision : str
            Precision of the numbers to be saved.
        sep : str
            Delimiter identifier.
        **kw_to_savetxt :
            Additional keyword arguments to pass to ``numpy.savetxt()``.
        """
        param_2D_array = self.serialize_to_2D_array()
        np.savetxt(
            filename,
            param_2D_array,
            fmt=precision,
            delimiter=sep,
            **kw_to_savetxt,
        )


class HH_Param_Multi_Layer(Param_Multi_Layer):
    """
    Class implementation of multiple sets of HH parameters for multiple layers.

    Its behavior is similar to a list,
    but with a more stringent requirement: all elements are of the same data
    type, i.e., HH_Param.

    The list-like behaviors available are:
        - indexing: foo[3]
        - slicing: foo[:4]
        - setting values: foo[2] = ...
        - length: len(foo)
        - deleting item: del foo[2]
        - checking existance: bar in foo

    Parameters
    ----------
    filename_or_data : str, numpy.ndarray, list<dict>, or list<``HH_Param``>
        A file name of a validly formatted "parameter file", i.e., having the
        following format:
            +----------------+-----------------+-----------------+-----+
            |  param_layer_1 |  param_layer_2  |  param_layer_3  | ... |
            +================+=================+=================+=====+
            |      1.1       |      2.2        |      3.3        | ... |
            +----------------+-----------------+-----------------+-----+
            |      1.2       |      2.3        |      3.4        | ... |
            +----------------+-----------------+-----------------+-----+
            |      ...       |      ...        |      ...        | ... |
            +----------------+-----------------+-----------------+-----+

        or a 2D numpy array containing the data of the format above, or a
        list containing HH parameter data.
    sep : str
        Delimiter of the file to be imported. If ``filename_or_data`` is not
        a file name, ``sep`` has no effect.

    Attributes
    ----------
    param_list : list<``HH_Param``>
        A list of HH model parameters.
    n_layer : int
        The number of soil layers (i.e., the length of the list).
    """

    def __init__(self, filename_or_data, *, sep='\t'):
        if isinstance(filename_or_data, str):  # file name
            self._filename = filename_or_data
            params = np.genfromtxt(filename_or_data, delimiter=sep)
            list_of_param_array = hlp.extract_from_param_format(params)
            list_of_param = [
                hh.deserialize_array_to_params(_) for _ in list_of_param_array
            ]
        elif isinstance(filename_or_data, np.ndarray):
            hlp.assert_2D_numpy_array(filename_or_data, name='`filename_or_data`')
            list_of_param_array = hlp.extract_from_param_format(filename_or_data)
            list_of_param = [
                hh.deserialize_array_to_params(_) for _ in list_of_param_array
            ]
        elif isinstance(filename_or_data, list):
            self._filename = None
            list_of_param = filename_or_data
        else:
            raise TypeError('Unrecognized type for ``filename_or_data``.')

        self._sep = sep

        super().__init__(
            list_of_param,
            element_class=HH_Param,
        )


class MKZ_Param_Multi_Layer(Param_Multi_Layer):
    """
    Class implementation of multiple sets of MKZ parameters for multiple layers.

    Its behavior is similar to a list,
    but with a more stringent requirement: all elements are of the same data
    type, i.e., MKZ_Param.

    The list-like behaviors available are:
        - indexing: foo[3]
        - slicing: foo[:4]
        - setting values: foo[2] = ...
        - length: len(foo)
        - deleting item: del foo[2]
        - checking existance: bar in foo

    Parameters
    ----------
    filename_or_data : str, numpy.ndarray, list<dict>, or list<``MKZ_Param``>
        A file name of a validly formatted "parameter file", i.e., having the
        following format:
            +----------------+-----------------+-----------------+-----+
            |  param_layer_1 |  param_layer_2  |  param_layer_3  | ... |
            +================+=================+=================+=====+
            |      1.1       |      2.2        |      3.3        | ... |
            +----------------+-----------------+-----------------+-----+
            |      1.2       |      2.3        |      3.4        | ... |
            +----------------+-----------------+-----------------+-----+
            |      ...       |      ...        |      ...        | ... |
            +----------------+-----------------+-----------------+-----+

        or a 2D numpy array containing the data of the format above, or a
        list containing MKZ parameter data.
    sep : str
        Delimiter of the file to be imported. If ``filename_or_data`` is not
        a file name, ``sep`` has no effect.

    Attributes
    ----------
    param_list : list<``MKZ_Param``>
        A list of MKZ model parameters.
    n_layer : int
        The number of soil layers (i.e., the length of the list).
    """

    def __init__(self, filename_or_data, *, sep='\t'):
        if isinstance(filename_or_data, str):  # file name
            self._filename = filename_or_data
            params = np.genfromtxt(filename_or_data, delimiter=sep)
            list_of_param_array = hlp.extract_from_param_format(params)
            list_of_param = [
                mkz.deserialize_array_to_params(_) for _ in list_of_param_array
            ]
        elif isinstance(filename_or_data, np.ndarray):
            hlp.assert_2D_numpy_array(filename_or_data, name='`filename_or_data`')
            list_of_param_array = hlp.extract_from_param_format(filename_or_data)
            list_of_param = [
                mkz.deserialize_array_to_params(_) for _ in list_of_param_array
            ]
        elif isinstance(filename_or_data, list):
            self._filename = None
            list_of_param = filename_or_data
        else:
            raise TypeError('Unrecognized type for ``filename_or_data``.')

        self._sep = sep

        super().__init__(
            list_of_param,
            element_class=MKZ_Param,
        )

from __future__ import annotations

import collections
import json
from typing import TYPE_CHECKING, Any, Callable, Type

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from PySeismoSoil import helper_generic as hlp
from PySeismoSoil import helper_hh_model as hh
from PySeismoSoil import helper_mkz_model as mkz
from PySeismoSoil import helper_site_response as sr

if TYPE_CHECKING:  # to avoid circular imports
    from PySeismoSoil.class_curves import (
        Multiple_Damping_Curves,
        Multiple_GGmax_Curves,
    )

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
    param_dict : dict[str, float]
        Name-value pairs of the parameters.
    allowable_keys : set[str] | None
        The allowable parameter names of the constitutive model.
    func_stress : Callable[[dict[str, float], ...], np.ndarray] | None
        A function to calculate shear stress from the parameters.

    Attributes
    ----------
    data : dict[str, float]
        The original data, stored as a regular dictionary.
    allowable_keys : set[str] | None
        Same as the input parameter.

    Raises
    ------
    TypeError
        When input arguments have invalid types
    KeyError
        When keys outside ``allowable_keys`` exist in ``param_dict``
    """

    def __init__(
            self,
            param_dict: dict[str, float],
            *,
            allowable_keys: set[str] | None = None,
            func_stress: Callable[[dict[str, float], ...], np.ndarray]
            | None = None,
    ) -> None:
        if not isinstance(param_dict, dict):
            raise TypeError('`param_dict` must be a dictionary.')

        if not isinstance(allowable_keys, set) or any(
            not isinstance(_, str) for _ in allowable_keys
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

    def __repr__(self) -> str:
        return json.dumps(self.data, indent=2).replace('"', '')

    def __setitem__(self, key, item) -> None:
        if key not in self.allowable_keys:
            raise KeyError("The model does not have a '%s' parameter." % key)

        self.data[key] = item

    def __delitem__(self, key) -> None:
        raise ValueError(
            'Deleting items from the parameter set is not allowed.'
        )

    def serialize(self) -> np.ndarray:
        """
        Serialize the parameter values into an array of floats. The order of
        the parameters are arbitrary, so any subclass of this class is
        recommended to override this method.

        Returns
        -------
        result : np.ndarray
            Serialized parameters.
        """
        param_array = []
        for _, val in self.data.items():
            param_array.append(val)

        return np.array(param_array)

    def get_stress(
            self, strain_in_pct: np.ndarray = STRAIN_RANGE_PCT
    ) -> np.ndarray | None:
        """
        Get the shear stress array inferred from the set of parameters

        Parameters
        ----------
        strain_in_pct : np.ndarray
            Strain array. Must be a 1D numpy array. Unit: %

        Returns
        -------
        result : np.ndarray | None
            The shear stress array, with the same shape as the strain array.
            Its unit is identical to the unit of Gmax (one of the HH parameters).
        """
        if self.func_stress is None:
            print('You did not provide a function to calculate shear stress.')
            return None

        hlp.assert_1D_numpy_array(strain_in_pct, name='`strain_in_pct`')
        return self.func_stress(strain_in_pct / 100.0, **self.data)

    def get_GGmax(
            self, strain_in_pct: np.ndarray = STRAIN_RANGE_PCT
    ) -> np.ndarray:
        """
        Get the G/Gmax array inferred from the set of parameters

        Parameters
        ----------
        strain_in_pct : np.ndarray
            Strain array. Must be a 1D numpy array. Unit: %

        Returns
        -------
        result : np.ndarray
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

    def get_damping(
            self, strain_in_pct: np.ndarray = STRAIN_RANGE_PCT
    ) -> np.ndarray:
        """
        Get the damping array inferred from the set of parameters

        Parameters
        ----------
        strain_in_pct : np.ndarray
            Strain array. Must be a 1D numpy array. Unit: %

        Returns
        -------
        result : np.ndarray
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

    def plot_curves(
            self,
            figsize: tuple[float, float] = None,
            dpi: float = 100,
            **kwargs_to_matplotlib: dict[Any, Any],
    ) -> tuple[Figure, list[Axes]]:
        """
        Plot G/Gmax and damping curves from the model parameters

        Parameters
        ----------
        figsize: tuple[float, float]
            Figure size in inches, as a tuple of two numbers. If ``None``, use
            (3, 6).
        dpi : float
            Figure resolution. If ``None``, use 100.
        **kwargs_to_matplotlib : dict[Any, Any]
            Keyword arguments to be passed to ``matplotlib.pyplot.plot()``.

        Returns
        -------
        fig : Figure
            The figure object.
        ax : list[Axes]
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
    param_dict : dict[str, float]
        Values of the HH model parameters. Acceptable key names are:
            gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d

    Attributes
    ----------
    data : dict[str, float]
        HH model parameters with the keys listed above.
    allowable_keys : set[str]
        Valid parameter names of the HH model.
    """

    def __init__(self, param_dict: dict[str, float]) -> None:
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

    def serialize(self) -> np.ndarray:
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
    param_dict : dict[str, float]
        Values of the HH model parameters. Acceptable key names are:
            gamma_ref, s, beta, Gmax

    Attributes
    ----------
    data : dict[str, float]
        MKZ model parameters with the keys listed above.
    allowable_keys : set[str]
        Valid parameter names of the MKZ model.
    """

    def __init__(self, param_dict: dict[str, float]) -> None:
        allowable_keys = {'gamma_ref', 's', 'beta', 'Gmax'}
        super().__init__(
            param_dict,
            func_stress=mkz.tau_MKZ,
            allowable_keys=allowable_keys,
        )

    def serialize(self) -> np.ndarray:
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
    list_of_param_data : list[dict[str, float]] | list[Parameter]
        List of dict or a list of valid parameter class (such as ``HH_Param``),
        which contain data for parameters of each layer.
    element_class : Type[Parameter]
        A class name, such as ``HH_Param``. Each element of ``list_of_param_dict``
        will be used to initialize an object of ``element_class``.

    Attributes
    ----------
    param_list : list[Parameter]
        A list of param objects whose type is specified by the user.
    n_layer : int
        The number of soil layers (i.e., the length of the list).

    Raises
    ------
    TypeError
        When an element in ``list_of_param_data`` has invalid type
    """

    def __init__(
            self,
            list_of_param_data: list[dict[str, float]] | list[Parameter],
            *,
            element_class: Type[Parameter],
    ) -> None:
        param_list: list[Parameter] = []
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

    def __contains__(self, item: Any) -> bool:
        return item in self.param_list

    def __len__(self) -> int:
        return self.n_layer

    def __setitem__(self, i, item) -> None:
        self.param_list[i] = item

    def __getitem__(self, i) -> Parameter | Param_Multi_Layer:
        if isinstance(i, int):
            return self.param_list[i]

        if isinstance(i, slice):  # return an object of the same class
            return self.__class__(self.param_list[i])  # filled with the sliced data

        raise TypeError('Indices must be integers or slices, not %s' % type(i))

    def __delitem__(self, i) -> None:
        del self.param_list[i]
        self.n_layer -= 1

    def construct_curves(
            self,
            strain_in_pct: np.ndarray = STRAIN_RANGE_PCT,
            curve_type: str | None = None,
    ) -> tuple['Multiple_GGmax_Curves', 'Multiple_Damping_Curves']:
        """
        Construct G/Gmax and damping curves from parameter values.

        Parameters
        ----------
        strain_in_pct : np.ndarray
            Strain array. Must be a 1D numpy array. Unit: %
        curve_type : str
            Either "ggmax" or "xi" for option to calculate only one of them.
            "None" will be returned for the other curve object in this case.

        Returns
        -------
        mgc : Multiple_GGmax_Curves
            G/Gmax curves for each soil layer.
        mdc : Multiple_Damping_Curves
            Damping curves for each soil layer.
        """
        # Importing within the method to avoid circular imports
        from PySeismoSoil.class_curves import (
            Multiple_Damping_Curves,
            Multiple_GGmax_Curves,
            Multiple_GGmax_Damping_Curves,
        )

        curves = None
        for param in self.param_list:
            GGmax = param.get_GGmax(strain_in_pct=strain_in_pct)
            damping = param.get_damping(strain_in_pct=strain_in_pct)
            if curves is None:
                curves = np.column_stack(
                    (strain_in_pct, GGmax, strain_in_pct, damping)
                )
            else:
                curves = np.column_stack(
                    (curves, strain_in_pct, GGmax, strain_in_pct, damping),
                )

        if curve_type == 'ggmax':
            GGmax_curve_list, _ = hlp.extract_from_curve_format(
                curves,
                ensure_non_negative=False,
            )
            mgc = Multiple_GGmax_Curves(GGmax_curve_list)
            mdc = None
        elif curve_type == 'xi':
            _, damping_curves_list = hlp.extract_from_curve_format(
                curves,
                ensure_non_negative=False,
            )
            mgc = None
            mdc = Multiple_Damping_Curves(damping_curves_list)
        else:
            mgdc = Multiple_GGmax_Damping_Curves(data=curves)
            mgc, mdc = mgdc.get_MGC_MDC_objects()

        return mgc, mdc

    def serialize_to_2D_array(self) -> np.ndarray:
        """
        Serialize the parameter data to a 2D numpy array.

        Returns
        -------
        param_2D_array : np.ndarray
            A 2D numpy array whose columns are parameters of each layer.
        """
        output = []
        for param_single_layer in self.param_list:
            param_array = param_single_layer.serialize()
            output.append(param_array)

        param_2D_array = np.array(output).T
        return param_2D_array

    def save_txt(
            self,
            filename: str,
            precision: str = '%.5g',
            sep: str = '\t',
            **kw_to_savetxt: dict[Any, Any],
    ) -> None:
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
        **kw_to_savetxt : dict[Any, Any]
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
        - checking existence: bar in foo

    Parameters
    ----------
    filename_or_data : str | np.ndarray | list[dict[str, float]] | list[HH_Param]
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

    Raises
    ------
    TypeError
        When the type of ``filename_or_data`` is not valid
    """

    def __init__(
            self,
            filename_or_data: str
            | np.ndarray
            | list[dict[str, float]]
            | list[HH_Param],
            *,
            sep: str = '\t',
    ) -> None:
        if isinstance(filename_or_data, str):  # file name
            self._filename = filename_or_data
            params = np.genfromtxt(filename_or_data, delimiter=sep)
            list_of_param_array = hlp.extract_from_param_format(params)
            list_of_param = [
                hh.deserialize_array_to_params(_) for _ in list_of_param_array
            ]
        elif isinstance(filename_or_data, np.ndarray):
            hlp.assert_2D_numpy_array(
                filename_or_data, name='`filename_or_data`'
            )
            list_of_param_array = hlp.extract_from_param_format(
                filename_or_data
            )
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
    filename_or_data : str | np.ndarray | list[dict[str, float]] | list[HH_Param]
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

    Raises
    ------
    TypeError
        When then type of ``filename_or_data`` is not valid
    """

    def __init__(
            self,
            filename_or_data: str
            | np.ndarray
            | list[dict[str, float]]
            | list[HH_Param],
            *,
            sep: str = '\t',
    ) -> None:
        if isinstance(filename_or_data, str):  # file name
            self._filename = filename_or_data
            params = np.genfromtxt(filename_or_data, delimiter=sep)
            list_of_param_array = hlp.extract_from_param_format(params)
            list_of_param = [
                mkz.deserialize_array_to_params(_) for _ in list_of_param_array
            ]
        elif isinstance(filename_or_data, np.ndarray):
            hlp.assert_2D_numpy_array(
                filename_or_data, name='`filename_or_data`'
            )
            list_of_param_array = hlp.extract_from_param_format(
                filename_or_data
            )
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

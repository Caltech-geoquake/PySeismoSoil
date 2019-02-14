# Author: Jian Shi

import numpy as np

#%%----------------------------------------------------------------------------
def read_two_column_stuff(data, delta=None, sep='\t', **kwargs_to_genfromtxt):
    '''
    Internal helper function. Processes "data" into a two-columned "data_".

    data : str or numpy.ndarray
        If str: the full file name on the hard drive containing the data.
        If np.ndarray: the numpy array containing the data.

        The data can have one column (which contains the motion/spectrum) or two
        columns (1st column: time/freq; 2nd column: motion/spectrum). If only
        one column is supplied, another input parameter "d_" must also be
        supplied.
    delta : float
        The time or frequency interval. If data is a file name, this parameter
        is ignored.
    sep : str
        The file delimiter. If data is not a file name, this parameter is
        ignored.
    **kwargs_to_genfromtxt :
        Any extra keyword arguments will be passed to numpy.genfromtxt()
        function for loading the data from the hard drive.

    Returns
    -------
    data_ : numpy.ndarray
        A two column numpy array containing the "delta array" (such as the time
        array or the frequency array) and the data.
    delta : float
        The "delta value", such as dt or df
    '''

    if isinstance(data, str):  # "data" is a file name
        data_ = np.genfromtxt(data, delimiter=sep, **kwargs_to_genfromtxt)
    elif isinstance(data, np.ndarray):
        data_ = data
    else:
        raise TypeError('`data` must be a file name or a numpy array.')

    if data_.ndim == 1 or (data_.ndim == 2 and min(data_.shape) == 1):
        if delta == None:
            raise ValueError('`delta` (such as dt or df) is needed for '
                             'one-column `data`.')
        else:
            n = len(data_)
            col1 = np.linspace(delta, n * delta, num=n)
            assert(np.abs(col1[1] - col1[0] - delta) / delta <= 1e-8)
            data_ = np.column_stack((col1, data_))
    elif data_.ndim == 2 and data_.shape[1] == 2:  # two columns
        col1 = data_[:, 0]
        delta = col1[1] - col1[0]
    elif data_.shape[1] != 2:
        raise TypeError('The provided data should be a two-column 2D numpy '
                        'array, or a one-column array with a `delta` value.')
    else:
        raise TypeError('"data" must be a file name or a numpy array.')

    return data_, delta

#%%----------------------------------------------------------------------------
def assert_1D_numpy_array(something, name=None):
    '''
    Assert that `something` is a 1D numpy array

    Parameters
    ----------
    something :
        Any Python object
    name : str or None
        The name of `something` to be displayed in the potential error message
    '''
    if not isinstance(something, np.ndarray) or something.ndim != 1:
        name = '`something`' if name is None else name
        raise TypeError('%s must be a 1D numpy array.' % name)

#%%----------------------------------------------------------------------------
def check_two_column_format(something, name=None, ensure_non_negative=False):
    '''
    Check that `something` is a 2D numpy array with two columns. Raises an
    error if `something` is the wrong format.

    Parameters
    ----------
    something :
        Any Python object
    name : str or None
        The name of `something` to be displayed in the potential error message
    ensure_non_negative : bool
        Whether to ensure that all values in `something` >= 0
    '''
    if name is None:
        name = '`something`'

    if not isinstance(something, np.ndarray):
        raise TypeError('%s should be an numpy array.' % name)
    if something.ndim != 2:
        raise TypeError('%s should be a 2D numpy array.' % name)
    if something.shape[1] != 2:
        raise TypeError('%s should have two columns.' % name)
    if check_numbers_valid(something) == -1:
        raise ValueError("%s should only contain numeric elements." % name)
    if check_numbers_valid(something) == -2:
        raise ValueError("%s should contain no NaN values." % name)
    if ensure_non_negative and check_numbers_valid(something) == -3:
        raise ValueError('%s should have all non-negative values.' % name)

    return None

#%%----------------------------------------------------------------------------
def check_Vs_profile_format(data):
    '''
    Check that `data` is in a valid format as a Vs profile.

    Parameter
    ---------
    data :
        Any Python object
    '''

    if not isinstance(data, np.ndarray):
        raise TypeError("`data` should be a numpy array.")
    if check_numbers_valid(data) == -1:
        raise ValueError("`data` should only contain numeric elements.")
    if check_numbers_valid(data) == -2:
        raise ValueError("`data` should contain no NaN values.")
    if check_numbers_valid(data) == -3:
        raise ValueError("`data` should not contain negative values.")
    if data.ndim != 2:
        raise TypeError("`data` should be a 2D numpy array.")
    if data.shape[1] not in [2, 5]:
        raise ValueError("`data` should have either 2 or 5 columns.")

    return None

#%%----------------------------------------------------------------------------
def check_numbers_valid(array):
    '''
    Generic helper function to check the contents in `array` is valid.

    Parameter
    ---------
    array : numpy.ndarray
        The numpy array to be tested

    Returns
    -------
    error_flag : int
        Flag indicating type of errors
    '''

    assert(isinstance(array, np.ndarray))

    if not np.issubdtype(array.dtype, np.number):
        return -1
    if not np.isfinite(array).all():
        return -2
    if np.any(array < 0):
        return -3

    return 0

#%%----------------------------------------------------------------------------
def interpolate(x_query_min, x_query_max, n_pts, x_ref, y_ref, log_scale=True,
                **kwargs_to_interp):
    '''
    Interpolate data (x_ref and y_ref) at x query points defined by x_query_min,
    x_query_max, and n_pts.

    Parameters
    ----------
    x_query_min : float
        Minimum x value at which you want to query (inclusive)
    x_query_max : float
        Maximum x value at which you want to query (inclusive)
    n_pts : int
        An array of x values are constructed between `x_query_min` and
        `x_query_max`, at which we query the y values. `n_pts` controls the
        length of this array.
    x_ref : numpy.ndarray
        Reference x values for interpolation. Must be a 1D numpy array.
    y_ref : numpy.ndarray
        Reference y values for interpolation. Must be a 1D numpy array.
    log_scale : bool
        Whether to construct the query array in log or linear scale
    **kwargs_to_interp :
        Extra keyword arguments to be passed to numpy.interp()

    Returns
    -------
    x_query_array : numpy.array
        A 1D numpy array constructed from x_query_min, x_query_max, and n_pts
    y_query_array : numpy.array
        The interpolation result. Same shape as x_query_array
    '''
    if log_scale:
        x_query_array = np.logspace(np.log10(x_query_min),
                                    np.log10(x_query_max), n_pts)
    else:
        x_query_array = np.linspace(x_query_min, x_query_max, n_pts)

    assert_1D_numpy_array(x_ref, name='`x_ref`')
    assert_1D_numpy_array(y_ref, name='`y_ref`')
    y_query_array = np.interp(x_query_array, x_ref, y_ref, **kwargs_to_interp)

    return x_query_array, y_query_array

#%%----------------------------------------------------------------------------
def mean_absolute_error(y_true, y_pred):
    '''
    Calculate the mean squared error between ground truth and prediction.

    Parameters
    ----------
    y_true : numpy.array
        Ground truth. Needs to be 1D numpy array.
    y_pred : numpy.array
        Prediction. Needs to be 1D numpy array.

    Returns
    -------
    mse : float
        Mean squared error
    '''
    assert_1D_numpy_array(y_true, name='`y_true`')
    assert_1D_numpy_array(y_pred, name='`y_pred`')
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

#%%----------------------------------------------------------------------------
def extract_from_curve_format(curves):
    '''
    Extract G/Gmax and damping curves from a "curve formatted" 2D numpy array.
    All G/Gmax curves are organized into a list, and all damping curves are
    organized into another list.

    Parameter
    ---------
    curves : numpy.ndarray
        A 2D numpy array that follows the following format:

        strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ...
        -----------+--------+------------+-------------+-------------+--------+ ...
           ...     |  ...   |    ...     |    ...      |    ...      |  ...   |

        Such an array can be constructed by hand, or by directly imported from
        a "curve_STATION_NAME.txt" file.

    Returns
    -------
    GGmax_curves_list, damping_curves_list : list<numpy.ndarray>
        The parsed G/Gmax and damping curves. Each element in the list is a 2D
        numpy array with 2 columns (strain and G/Gmax, or strain and damping).
        The units are shown in the above format.
    '''

    if not isinstance(curves, np.ndarray):
        raise TypeError('`curves` needs to be a numpy array.')
    else:
        if curves.ndim != 2:
            raise TypeError('If `curves` is a numpy array, it needs to be 2D.')
        if curves.shape[1] % 4 != 0:
            raise ValueError('If `curves` is a numpy array, its number of '
                             'columns needs to be a multiple of 4.')
        n_layer = curves.shape[1] // 4

        GGmax_curves_list = []
        damping_curves_list = []
        for j in range(n_layer):
            GGmax = curves[:, j * 4 + 0 : j * 4 + 2]
            damping = curves[:, j * 4 + 2 : j * 4 + 4]
            check_two_column_format(GGmax,
                                    name='G/Gmax curve for layer #%d' % j,
                                    ensure_non_negative=True)
            check_two_column_format(damping,
                                    name='Damping curve for layer #%d' % j,
                                    ensure_non_negative=True)
            GGmax_curves_list.append(GGmax)
            damping_curves_list.append(damping)

    return GGmax_curves_list, damping_curves_list

#%%----------------------------------------------------------------------------
def extract_from_param_format(params):
    '''
    Extract soil constituve model parameters from a 2D numpy array, which
    follows the following format:

        param_layer_1  |  param_layer_2  |  param_layer_3  | ...
        ---------------+-----------------+-----------------+-----
             1.1       |      2.2        |      3.3        | ...
             1.2       |      2.3        |      3.4        | ...
             ...       |      ...        |      ...        | ...

    Parameter
    ---------
    params : numpy.ndarray
        A 2D numpy array containing soil constitutive model parameters for each
        soil layer. Such an array can be constructed by hand, or directly
        imported from a "HH_x_STATION_NAME.txt" file or something similar.

    Returns
    -------
    param_list : list<numpy.ndarray>
        The parsed parameters for each layer. Each element of `param_list` is
        a 1D numpy array with length N, where N is the number of parameters for
        the particular soil constitutive model.
    '''

    if not isinstance(params, np.ndarray) or params.ndim != 2:
        raise TypeError('`params` needs to be a 2D numpy array.')

    n_layer = params.shape[1]
    param_list = []
    for j in range(n_layer):
        param_list.append(params[:, j])

    return param_list


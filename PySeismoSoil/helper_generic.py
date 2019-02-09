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
def check_two_column_format(something, name=None):
    '''
    Check that `something` is a 2D numpy array with two columns. Raises an
    error if `something` is the wrong format.

    Parameters
    ----------
    something :
        Any Python object
    name : str or None
        The name of `something` to be displayed in the potential error message.
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

    return None

#%%----------------------------------------------------------------------------
def check_Vs_profile_format(data):
    '''
    Check that `data` is in a valid format as a Vs profile.
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
        Reference x values for interpolation. Must be an 1D numpy array.
    y_ref : numpy.ndarray
        Reference y values for interpolation. Must be an 1D numpy array.
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

    y_query_array = np.interp(x_query_array, x_ref, y_ref, **kwargs_to_interp)

    return x_query_array, y_query_array













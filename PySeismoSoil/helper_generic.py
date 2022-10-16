import platform

import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt


def detect_OS():
    """
    Check which operating system is currently running.

    Returns
    -------
    result : str
        One of 'Windows', 'Linux', or 'Darwin'.
    """
    return platform.system()


def get_current_time(for_filename=True):
    """
    Get current time as a string (e.g., 2001-01-01 23:59:59).

    Parameters
    ----------
    for_filename : bool
        Whether the returned string is for filenames or not. If so, colons
        are substituted with dashes, and the space is substituted with an
        underscore.
    """
    import datetime

    if for_filename:
        return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    else:
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def find_closest_index(array, value):
    """
    Find the index in ``array`` corresponding to the value closest to the
    given ``value``.

    Parameters
    ----------
    array : numpy.ndarray
        A 1D numpy array. It does not need to be sorted, because this function
        has an O(n) time complexity.
    value : float
        The value to search for.

    Returns
    -------
    index : int
        The index whose value is closest to the given ``value``.
    closest_value : float
        The value in ``array`` that is closest to the given ``value``.
    """
    assert_1D_numpy_array(array, name='`array`')
    if not isinstance(value, (int, float, np.number)):
        raise TypeError('`value` must be a single number (such as a float).')

    if len(array) == 0:
        index = None
        closest_value = None
    else:
        deviation = np.abs(array - value)
        index = np.argmin(deviation)
        closest_value = array[index]

    return index, closest_value


def _process_fig_ax_objects(fig, ax, figsize=None, dpi=None, ax_proj=None):
    """
    Process figure and axes objects. If ``fig`` and ``ax`` are None, creates
    new figure and new axes according to ``figsize``, ``dpi``, and ``ax_proj``.
    Otherwise, uses the passed-in ``fig`` and/or ``ax``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    figsize: (float, float)
        Figure size in inches, as a tuple of two numbers. The figure
        size of ``fig`` (if not ``None``) will override this parameter.
    dpi : float
        Figure resolution. The dpi of ``fig`` (if not ``None``) will override
        this parameter.
    ax_proj : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}
        The projection type of the axes. The default None results in a
        'rectilinear' projection.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    """
    if fig is None:  # if a figure handle is not provided, create new figure
        fig = pl.figure(figsize=figsize, dpi=dpi)
    else:  # if provided, plot to the specified figure
        pl.figure(fig.number)

    if ax is None:  # if ax is not provided
        ax = plt.axes(projection=ax_proj)  # create new axes and plot lines on it
    else:
        ax = ax  # plot lines on the provided axes handle

    return fig, ax


def read_two_column_stuff(data, delta=None, sep='\t', **kwargs_to_genfromtxt):
    """
    Process "data" into a two-columned "data_".

    Parameters
    ----------
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
        Any extra keyword arguments will be passed to ``numpy.genfromtxt()``
        function for loading the data from the hard drive.

    Returns
    -------
    data_ : numpy.ndarray
        A two column numpy array containing the "delta array" (such as the time
        array or the frequency array) and the data.
    delta : float
        The "delta value", such as dt or df.
    """
    if isinstance(data, str):  # "data" is a file name
        data_ = np.genfromtxt(data, delimiter=sep, **kwargs_to_genfromtxt)
    elif isinstance(data, np.ndarray):
        data_ = data
    else:
        raise TypeError('`data` must be a file name or a numpy array.')

    if data_.ndim == 1 or (data_.ndim == 2 and min(data_.shape) == 1):
        if delta is None:
            raise ValueError(
                '`delta` (such as dt or df) is needed for one-column `data`.',
            )
        else:
            n = len(data_)
            col1 = np.linspace(delta, n * delta, num=n)
            assert np.abs(col1[1] - col1[0] - delta) / delta <= 1e-8
            data_ = np.column_stack((col1, data_))
    elif data_.ndim == 2 and data_.shape[1] == 2:  # two columns
        col1 = data_[:, 0]
        delta = col1[1] - col1[0]
    elif data_.shape[1] != 2:
        raise TypeError(
            'The provided data should be a two-column 2D numpy '
            'array, or a one-column array with a `delta` value.',
        )
    else:
        raise TypeError('"data" must be a file name or a numpy array.')

    return data_, delta


def assert_1D_numpy_array(something, name=None):
    """
    Assert that ``something`` is a 1D numpy array

    Parameters
    ----------
    something :
        Any Python object.
    name : str or None
        The name of ``something`` to be displayed in the potential error message.
    """
    if not isinstance(something, np.ndarray) or something.ndim != 1:
        name = '`something`' if name is None else name
        raise TypeError('%s must be a 1D numpy array.' % name)


def assert_array_length(something, length, name='`something`'):
    """
    Assert that ``something`` is a 1D of length ``length``.

    Parameters
    ----------
    something :
        Any Python object
    length : int or ``None``
        The length that ``something`` must have.
    name : str
        The name of ``something`` for displaying the error message, if necessary.
    """
    assert_1D_numpy_array(something, name=name)
    if len(something) != length:
        raise ValueError(
            '%s must have length %d, but not %d.' % (name, length, len(something)),
        )


def extend_scalar(scalar, length):
    """
    "Extend" a scalar (float, int, or numpy.number type) into a 1D numpy array
    whose length is ``length`` and whose elements are all ``scalar``.

    Parameters
    ----------
    scalar : float, int, numpy.number
        A single number.
    length : int
        The length of the desired output.

    Returns
    -------
    array : numpy.ndarray
        A 1D numpy array with length ``length`` and elements of value ``scalar``.
    """
    if not isinstance(scalar, (float, int, np.number)):
        raise TypeError('`scalar` must be a float, int, or a numpy.number type.')

    array = scalar * np.ones(length)
    return array


def check_length_or_extend_to_array(something, length, name='`something`'):
    """
    Check that ``something`` is a 1D numpy array with length ``length``, or
    if ``something`` is a single value, extend it to a 1D numpy array whose
    length is ``length`` and elements are all ``something``.

    Parameters
    ----------
    something :
        Any Python object.
    length : int
        The desired length of array.
    name : str
        The name of ``something`` for displaying the error message, if necessary.

    Returns
    -------
    array : numpy.ndarray
        The array that ``something`` is extended to (if ``something`` is a
        single value). Or ``something`` itself.
    """
    if isinstance(something, (float, int, np.number)):
        array = extend_scalar(something, length)
    else:
        assert_array_length(something, length, name=name)
        array = something

    return array


def assert_2D_numpy_array(something, name=None):
    """
    Assert that ``something`` is a 2D numpy array.

    Parameters
    ----------
    something :
        Any Python object.
    name : str or None
        The name of ``something`` to be displayed in the potential error message.
    """
    if not isinstance(something, np.ndarray) or something.ndim != 2:
        name = '`something`' if name is None else name
        raise TypeError('%s must be a 2D numpy array.' % name)


def check_two_column_format(
        something,
        name=None,
        ensure_non_negative=False,
        at_least_two_columns=False,
):
    """
    Check that ``something`` is a 2D numpy array with two columns. Raises an
    error if ``something`` is the wrong format.

    Parameters
    ----------
    something :
        Any Python object.
    name : str or None
        The name of ``something`` to be displayed in the potential error message.
    ensure_non_negative : bool
        Whether to ensure that all values in ``something`` >= 0.
    at_least_two_columns : bool
        Whether to relax the constraints to from "exactly 2 columns" to "at
        least two columns".
    """
    if name is None:
        name = '`something`'

    if not isinstance(something, np.ndarray):
        raise TypeError('%s should be a numpy array.' % name)
    if something.ndim != 2:
        raise TypeError('%s should be a 2D numpy array.' % name)
    if not at_least_two_columns and something.shape[1] != 2:
        raise TypeError('%s should have two columns.' % name)
    if at_least_two_columns and something.shape[1] < 2:
        raise TypeError('%s should have >= 2 columns.' % name)

    check_status = check_numbers_valid(something)
    if check_status == -1:
        raise ValueError('%s should only contain numeric elements.' % name)
    if check_status == -2:
        raise ValueError('%s should contain no NaN values.' % name)
    if ensure_non_negative and check_status == -3:
        raise ValueError('%s should have all non-negative values.' % name)

    return None


def check_Vs_profile_format(data):
    """
    Check that ``data`` is in a valid format as a Vs profile (i.e., 2D numpy
    array, two or five columns, non-negative or positive values, etc.)

    Parameters
    ----------
    data :
        Any Python object.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('`data` should be a numpy array.')

    check_status = check_numbers_valid(data)
    if check_status == -1:
        raise ValueError('`data` should only contain numeric elements.')
    if check_status == -2:
        raise ValueError('`data` should contain no NaN values.')
    if data.ndim != 2:
        raise ValueError('`data` should be a 2D numpy array.')
    if data.shape[1] not in [2, 5]:
        raise ValueError('`data` should have either 2 or 5 columns.')

    thk = data[:, 0]
    Vs = data[:, 1]
    if np.any(thk[:-1] <= 0):
        raise ValueError(
            'The thickness column should be all positive, except for the last layer.',
        )
    if np.any(thk[-1] < 0):
        raise ValueError('The last layer thickness should be non-negative.')
    if np.any(Vs <= 0):
        raise ValueError('The Vs column should be all positive.')

    if data.shape[1] == 5:
        xi = data[:, 2]
        rho = data[:, 3]
        mat = data[:, 4]
        if np.any(xi <= 0) or np.any(rho <= 0):
            raise ValueError('The damping and density columns should be positive.')
        if not all(is_int(_) for _ in mat):
            raise ValueError('The "material number" column should be all integers.')
        if np.any(mat[:-1] <= 0):
            raise ValueError(
                'The "material number" column should be all '
                'positive, except for the last error.',
            )
        if np.any(mat[-1] < 0):
            raise ValueError(
                'The material number of the last layer should be non-negative.',
            )

    return None


def is_int(number):
    """
    Check that a ``number`` represents an integer value. (Its data type does
    not need to be int or numpy.integer).

    Parameters
    ----------
    number :
        Any Python object.
    """
    if not isinstance(number, (int, float, np.number)):
        return False
    if isinstance(number, (int, np.integer)):
        return True
    try:
        if number.is_integer():
            return True
    except AttributeError:
        return False


def check_numbers_valid(array):
    """
    Check the contents in ``array`` is valid (i.e., are numbers, are not
    infinite, are positive).

    Parameters
    ----------
    array : numpy.ndarray
        The numpy array to be tested.

    Returns
    -------
    error_flag : int
        Flag indicating type of errors.
    """
    assert isinstance(array, np.ndarray)

    if not np.issubdtype(array.dtype, np.number):
        return -1
    if not np.isfinite(array).all():
        return -2
    if np.any(array < 0):
        return -3

    return 0


def interpolate(
        x_query_min,
        x_query_max,
        n_pts,
        x_ref,
        y_ref,
        log_scale=True,
        **kwargs_to_interp,
):
    """
    Interpolate data (``x_ref`` and ``y_ref``) at x query points defined by
    ``x_query_min``, ``x_query_max``, and ``n_pts``.

    Parameters
    ----------
    x_query_min : float
        Minimum x value at which you want to query (inclusive).
    x_query_max : float
        Maximum x value at which you want to query (inclusive).
    n_pts : int
        An array of x values are constructed between `x_query_min` and
        `x_query_max`, at which we query the y values. `n_pts` controls the
        length of this array.
    x_ref : numpy.ndarray
        Reference x values for interpolation. Must be a 1D numpy array.
    y_ref : numpy.ndarray
        Reference y values for interpolation. Must be a 1D numpy array.
    log_scale : bool
        Whether to construct the query array in log or linear scale.
    **kwargs_to_interp :
        Extra keyword arguments to be passed to ``numpy.interp()``.

    Returns
    -------
    x_query_array : numpy.ndarray
        A 1D numpy array constructed from ``x_query_min``, ``x_query_max``,
        and ``n_pts``.
    y_query_array : numpy.ndarray
        The interpolation result. Same shape as ``x_query_array``.
    """
    if log_scale:
        x_query_array = np.logspace(
            np.log10(x_query_min),
            np.log10(x_query_max),
            n_pts,
        )
    else:
        x_query_array = np.linspace(x_query_min, x_query_max, n_pts)

    assert_1D_numpy_array(x_ref, name='`x_ref`')
    assert_1D_numpy_array(y_ref, name='`y_ref`')
    y_query_array = np.interp(x_query_array, x_ref, y_ref, **kwargs_to_interp)

    return x_query_array, y_query_array


def mean_absolute_error(y_true, y_pred):
    """
    Calculate the mean squared error between ground truth and prediction.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth. Needs to be 1D numpy array.
    y_pred : numpy.ndarray
        Prediction. Needs to be 1D numpy array.

    Returns
    -------
    mse : float
        Mean squared error.
    """
    assert_1D_numpy_array(y_true, name='`y_true`')
    assert_1D_numpy_array(y_pred, name='`y_pred`')
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def extract_from_curve_format(curves, ensure_non_negative=True):
    """
    Extract G/Gmax and damping curves from a "curve formatted" 2D numpy array.
    All G/Gmax curves are organized into a list, and all damping curves are
    organized into another list.

    Parameters
    ----------
    curves : numpy.ndarray
        A 2D numpy array that follows the following format:

            +------------+--------+------------+-------------+-------------+--------+-----+
            | strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ... |
            +============+========+============+=============+=============+========+=====+
            |    ...     |  ...   |    ...     |    ...      |    ...      |  ...   | ... |
            +------------+--------+------------+-------------+-------------+--------+-----+

        Such an array can be constructed by hand, or by directly imported from
        a "curve_STATION_NAME.txt" file.
    ensure_non_negative : bool
        If ``True``, raise an exception if there exists at least one negative
        G/Gmax value or damping value in the data in ``curves``.

    Returns
    -------
    GGmax_curves_list : list<numpy.ndarray>:
        The parsed G/Gmax curves. Each element in the list is a 2D numpy array
        with 2 columns (strain and G/Gmax). The units are shown in the above
        table.
    damping_curves_list : list<numpy.ndarray>
        The parsed damping curves. Each element in the list is a 2D numpy array
        with 2 columns (strain and damping). The units are shown in the above
        table.
    """
    if not isinstance(curves, np.ndarray):
        raise TypeError('`curves` needs to be a numpy array.')
    else:
        if curves.ndim != 2:
            raise TypeError('If `curves` is a numpy array, it needs to be 2D.')
        if curves.shape[1] % 4 != 0:
            raise ValueError(
                'If `curves` is a numpy array, its number of '
                'columns needs to be a multiple of 4.',
            )
        n_layer = curves.shape[1] // 4

        GGmax_curves_list = []
        damping_curves_list = []
        for j in range(n_layer):
            GGmax = curves[:, j * 4 + 0 : j * 4 + 2]
            damping = curves[:, j * 4 + 2 : j * 4 + 4]
            check_two_column_format(
                GGmax,
                name='G/Gmax curve for layer #%d' % j,
                ensure_non_negative=ensure_non_negative,
            )
            check_two_column_format(
                damping,
                name='Damping curve for layer #%d' % j,
                ensure_non_negative=ensure_non_negative,
            )
            GGmax_curves_list.append(GGmax)
            damping_curves_list.append(damping)

    return GGmax_curves_list, damping_curves_list


def extract_from_param_format(params):
    """
    Extract soil constitutive model parameters from a 2D numpy array.

    The 2D numpy array should follow the following format:
        +----------------+-----------------+-----------------+-----+
        |  param_layer_1 |  param_layer_2  |  param_layer_3  | ... |
        +================+=================+=================+=====+
        |      1.1       |      2.2        |      3.3        | ... |
        +----------------+-----------------+-----------------+-----+
        |      1.2       |      2.3        |      3.4        | ... |
        +----------------+-----------------+-----------------+-----+
        |      ...       |      ...        |      ...        | ... |
        +----------------+-----------------+-----------------+-----+

    Parameters
    ----------
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
    """
    if not isinstance(params, np.ndarray) or params.ndim != 2:
        raise TypeError('`params` needs to be a 2D numpy array.')

    n_layer = params.shape[1]
    param_list = []
    for j in range(n_layer):
        param_list.append(params[:, j])

    return param_list


def merge_curve_matrices(GGmax_matrix, xi_matrix):
    """
    Merge G/Gmax curves matrix and damping curves matrix. Both matrices need to
    have the following format:

        +------------+--------+------------+-------------+-------------+--------+-----+
        | strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ... |
        +============+========+============+=============+=============+========+=====+
        |    ...     |  ...   |    ...     |    ...      |    ...      |  ...   | ... |
        +------------+--------+------------+-------------+-------------+--------+-----+

    They need to have the same shape. This function will take the G/Gmax
    information from ``GGmax_matrix`` and the damping information from
    ``xi_matrix``, and produce a new matrix.

    Parameters
    ----------
    GGmax_matrix : numpy.ndarray
        A 2D numpy array of the format above that contains G/Gmax information.
    xi_matrix : numpy.ndarray
        A 2D numpy array of the format above that contains damping information.

    Returns
    -------
    merged : numpy.ndarray
        The merged 2D numpy array.
    """
    assert_2D_numpy_array(GGmax_matrix, name='`GGmax_matrix`')
    assert_2D_numpy_array(xi_matrix, name='`xi_matrix`')
    if GGmax_matrix.shape[1] % 4 != 0:
        raise ValueError(
            'The number of columns of `GGmax_matrix` needs '
            'to be a multiple of 4. However, your '
            '`GGmax_matrix` has %d columns.' % GGmax_matrix.shape[1],
        )
    if xi_matrix.shape[1] % 4 != 0:
        raise ValueError(
            'The number of columns of `xi_matrix` needs '
            'to be a multiple of 4. However, your '
            '`xi_matrix` has %d columns.' % xi_matrix.shape[1],
        )
    if GGmax_matrix.shape[1] != xi_matrix.shape[1]:
        raise ValueError(
            '`GGmax_matrix` and `xi_matrix` need to have the '
            'same number of columns. You can use trim one or both'
            'of them outside this function to make the shape '
            'identical. Sorry for the inconvenience.',
        )
    if GGmax_matrix.shape[0] != xi_matrix.shape[0]:
        raise ValueError(
            '`GGmax_matrix` and `xi_matrix` need to have the '
            'same number of rows. You can use interpolation '
            'outside of this function to make the lengths '
            'identical. Sorry for the inconvenience.',
        )
    n_layer = GGmax_matrix.shape[1] // 4
    merged = np.column_stack((GGmax_matrix[:, :2], xi_matrix[:, 2:4]))
    for k in range(1, n_layer):
        merged = np.column_stack(
            (
                merged,
                GGmax_matrix[:, k * 4 : k * 4 + 2],
                xi_matrix[:, k * 4 + 2 : k * 4 + 4],
            ),
        )
    # END FOR
    return merged

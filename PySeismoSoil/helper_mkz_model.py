import numpy as np
import matplotlib.pyplot as plt

from . import helper_generic as hlp
from . import helper_site_response as sr


def tau_MKZ(gamma, *, gamma_ref, beta, s, Gmax):
    """
    Calculate the MKZ shear stress. The MKZ model is proposed in Matasovic and
    Vucetic (1993), and has the following form::

                              Gmax * gamma
        T(gamma) = ---------------------------------------
                      1 + beta * (gamma / gamma_ref)^s

    where:
        + T         = shear stress
        + gamma     = shear strain
        + Gmax      = initial shear modulus
        + beta      = a shape parameter of the MKZ model
        + gamma_ref = reference strain, another shape parameter of the MKZ model
        + s         = another shape parameter of the MKZ model

    Parameters
    ----------
    gamma : numpy.ndarray
        The shear strain array. Must be a 1D array. Its unit should be '1',
        rather than '%'.
    gamma_ref : float
        Reference shear strain, a shape parameter of the MKZ model. Unit: 1.
    beta : float
        A shape parameter of the MKZ model.
    s : float
        A shape parameter of the MKZ model.
    Gmax : float
        Initial shear modulus. Its unit can be arbitrary, but we recommend Pa.

    Returns
    -------
    T_MKZ : numpy.ndarray
        The shear stress determined by the formula above. Same shape as ``x``,
        and same unit as ``Gmax``.
    """
    hlp.assert_1D_numpy_array(gamma, name='`gamma`')
    T_MKZ = Gmax * gamma / ( 1 + beta * (np.abs(gamma) / gamma_ref)**s )

    return T_MKZ


def fit_H4_x_single_layer(
        damping_data_in_pct,
        *,
        use_scipy=True,
        pop_size=800,
        n_gen=100,
        lower_bound_power=-4,
        upper_bound_power=6,
        eta=0.1,
        seed=0,
        show_fig=False,
        verbose=False,
        suppress_warnings=True,
        parallel=False,
        n_cores=None,
):
    """
    Perform H4_x curve fitting for one damping curve using the genetic
    algorithm.

    Parameters
    ----------
    damping_data_in_pct : numpy.ndarray
        Damping data. Needs to have 2 columns (strain and damping ratio). Both
        columns need to use % as unit.
    use_scipy : bool
        Whether to use the "differential_evolution" algorithm in scipy
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
        to perform the optimization. If ``False``, use the algorithm in the
        DEAP package.
    pop_size : int
        The number of individuals in a generation. A larger number leads to
        potentially better curve-fitting, but a longer computing time.
    n_gen : int
        Number of generations that the evolution lasts. A larger number leads
        to potentially better curve-fitting, but a longer computing time.
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
    supress_warnings : bool
        Whether to suppress warning messages. For this particular task,
        overflow warnings are likely to occur.
    parallel : bool
        Whether to use multiple processors in the calculation. All CPU cores
        will be used if set to ``True``.

    Return
    ------
    best_param : dict
        The best parameters found in the optimization.
    """
    hlp.check_two_column_format(damping_data_in_pct, ensure_non_negative=True)

    init_damping = damping_data_in_pct[0, 1]  # small-strain damping
    damping_data_in_pct[:, 1] -= init_damping  # offset all dampings
    damping_data_in_unit_1 = damping_data_in_pct / 100  # unit: percent --> 1

    n_param = 3  # number of MKZ model parameters; do not change this
    N = 122  # denser strain array for more accurate damping calculation
    strain_dense = np.logspace(-6, -1, N)
    damping_dense = np.interp(
        strain_dense,
        damping_data_in_unit_1[:, 0],
        damping_data_in_unit_1[:, 1],
    )

    damping_data_ = np.column_stack((strain_dense, damping_dense))

    crossover_prob = 0.8  # hard-coded, because not much useful to tune them
    mutation_prob = 0.8

    result = sr.ga_optimization(
        n_param,
        lower_bound_power,
        upper_bound_power,
        damping_misfit,
        damping_data_,
        use_scipy=use_scipy,
        pop_size=pop_size,
        n_gen=n_gen,
        eta=eta,
        seed=seed,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        suppress_warnings=suppress_warnings,
        verbose=verbose,
        parallel=parallel,
        n_cores=n_cores,
    )

    best_param = {}
    best_param['gamma_ref'] = 10 ** result[0]
    best_param['s'] = 10 ** result[1]
    best_param['beta'] = 10 ** result[2]
    best_param['Gmax'] = 1.0

    if show_fig:
        sr._plot_damping_curve_fit(damping_data_in_pct, best_param, tau_MKZ)

    return best_param


def damping_misfit(param_without_Gmax, damping_data):
    """
    Calculate the misfit given a set of MKZ parameters. Note that the values
    in `param` are actually the 10-based power of the actual MKZ parameters.
    Using the powers in the genetic algorithm searching turns out to work
    much better for this particular problem.

    Parameters
    ----------
    param_without_Gmax : tuple<float>
        MKZ model parameters, in the order specified below:
            gamma_ref, s, beta
    damping_data : numpy.ndarray
        2D numpy array with two columns (strain and damping value). Both
        columns need to use "1" as the unit, not percent.

    Returns
    -------
    error : float
        The mean absolute error between the true damping values and the
        predicted damping values at each strain level.
    """
    gamma_ref_, s_, beta_ = param_without_Gmax

    gamma_ref = 10 ** gamma_ref_
    beta = 10 ** beta_
    s = 10 ** s_
    Gmax = 1.0  # does not affect damping, because it gets cancels out

    strain = damping_data[:, 0]
    damping_true = damping_data[:, 1]

    Tau_MKZ = tau_MKZ(strain, gamma_ref=gamma_ref, beta=beta, s=s, Gmax=Gmax)
    damping_pred = sr.calc_damping_from_stress_strain(strain, Tau_MKZ, Gmax)
    error = hlp.mean_absolute_error(damping_true, damping_pred)

    return error


def serialize_params_to_array(param, to_files=False):
    """
    Convert the MKZ parameters from a dictionary to an array, according to this
    order:
        gamma_ref, s, beta, Gmax

    Parameters
    ----------
    param : dict
        A dictionary containing the parameters of the MKZ model.
    to_files : bool
        Whether the result is for writing to files. If so, the last parameter,
        Gmax, is removed, and a dummy parameter, b, which is always 0, is
        inserted between gamma_ref and s. This is for historical reasons: the
        text files recognizable by MATLAB and Fortran functions have the
        convention of "gamma_ref, 0.0, s, beta".

    Returns
    -------
    param_array : numpy.array
        A numpy array of shape (9,) containing the parameters of the MKZ model
        in the order specified above.
    """
    assert(len(param) == 4)
    order = ['gamma_ref', 's', 'beta', 'Gmax']
    param_array = []
    for key in order:
        param_array.append(param[key])

    if to_files:
        param_array = [param_array[0], 0.0, param_array[1], param_array[2]]

    return np.array(param_array)


def deserialize_array_to_params(array, from_files=False):
    """
    Reconstruct a MKZ model parameter dictionary from an array of values.

    The users needs to ensure the order of values in ``array`` are in this order:
        gamma_ref, s, beta, Gmax (if ``from_files`` is ``False``)
    or:
        gamma_ref, b, s, beta (if ``from_files`` is ``True``)
    (b is always 0, for historical reasons)

    Parameters
    ----------
    array : numpy.ndarray
        A 1D numpy array of MKZ parameter values in this order:
            gamma_ref, s, beta, Gmax
    from_files : bool
        Whether the array was directly imported from a "H4_x_SITE_NAME.txt"
        file. If so, the 1st (0-based indexing) element, "b", which is always
        0, is neglected, and a dummy Gmax value (1.0) is padded at the end. The
        presence of "b" is due to historical reasons.

    Returns
    -------
    param : dict
        The dictionary with parameter name as keys and values as values.
    """
    hlp.assert_1D_numpy_array(array)
    assert(len(array) == 4)

    if from_files:
        param = dict()
        param['gamma_ref'] = array[0]
        param['s'] = array[2]
        param['beta'] = array[3]
        param['Gmax'] = 1.0  # "H4_G_SITE_NAME.txt" files don't have Gmax info
    else:
        param = dict()
        param['gamma_ref'] = array[0]
        param['s'] = array[1]
        param['beta'] = array[2]
        param['Gmax'] = array[3]

    return param


def fit_MKZ(curve_data, show_fig=False, verbose=False):
    """
    Fit MKZ model to G/Gmax curves.

    Parameters
    ----------
    curve_data : numpy.ndarray
        A 2D numpy array that represents G/Gmax and damping curves of each
        layer, in the following format:
         +------------+--------+------------+-------------+-------------+--------+-----+
         | strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ... |
         +============+========+============+=============+=============+========+=====+
         |    ...     |  ...   |    ...     |    ...      |    ...      |  ...   | ... |
         +------------+--------+------------+-------------+-------------+--------+-----+

        The damping information is neglected in this function, so users can
        supply some dummy values.
    show_fig : bool
        Whether or not to show curve-fitting results.
    verbose : bool
        Whether to show messages about the calculation progress on the console.

    Returns
    -------
    param : numpy.ndarray
        The fitted MKZ parameters. Shape: (n_mat, 4), where ``n_mat`` is
        the number of materials implied in ``curve_data``.
    fitted_curves : numpy.ndarray
        The fitted curves. Shape: (nr, 4 * n_mat), where ``nr`` is the length
        of the strain array. Currently hard-coded as 109.
    """
    from scipy.optimize import curve_fit

    hlp.assert_2D_numpy_array(curve_data, name='`curve_data`')

    nr = 109
    fitted_curves = np.zeros((nr, curve_data.shape[1]))

    n_ma = int(curve_data.shape[1] / 4.0)  # number of materials
    length = curve_data.shape[0]  # length of strain array

    ref_strain = np.zeros(n_ma)
    s_value = np.zeros(n_ma)
    beta = np.zeros(n_ma)

    gamma = np.zeros((length, n_ma))
    GGmax = np.zeros((length, n_ma))
    for k in range(n_ma):
        gamma[:, k] = curve_data[:, k*4 + 0] / 100.0  # percent --> 1
        GGmax[:, k] = curve_data[:, k*4 + 1]

    gamma_ = np.geomspace(1e-6, 0.1, num=nr)  # unit: 1
    GGmax_ = np.zeros((nr, n_ma))
    damping_ = np.zeros((nr, n_ma))

    #-------------- Curve-fitting, layer by layer -----------------------------
    def func(x, beta, gamma_ref, s):
        return 1.0 / (1 + beta * (x / gamma_ref)**s)

    if verbose:
        print('Fitting MKZ model to G/Gmax data. Total: %d layers.' % n_ma)
    for j in range(n_ma):
        if verbose:
            print('  Layer #%d' % j)
        x_data = gamma[:, j]
        y_data = GGmax[:, j]
        popt, _ = curve_fit(
            func,
            x_data,
            y_data,
            p0=[1, 0.005, 0.8],
            bounds=([0.2, 0, 0.6], [1.8, 0.1, 0.999]),
        )
        beta[j] = popt[0]
        ref_strain[j] = popt[1]
        s_value[j] = popt[2]

    param = np.column_stack((ref_strain, np.zeros(n_ma), s_value, beta))

    #------------ Calculate the fitted curve ----------------------------------
    for k in range(n_ma):
        param_k = param[k, :]
        T_MKZ = tau_MKZ(
            gamma_,
            gamma_ref=param_k[0],
            s=param_k[2],
            beta=param_k[3],
            Gmax=1.0,
        )
        GGmax_k = sr.calc_GGmax_from_stress_strain(gamma_, T_MKZ, Gmax=1.0)
        GGmax_[:, k] = GGmax_k

    #------------ Plotting ----------------------------------------------------
    if show_fig:
        ncol = 4
        nrow = int(np.ceil(n_ma / ncol))
        plt.figure(figsize=(ncol * 3.5, nrow * 3))
        for k in range(n_ma):
            plt.subplot(nrow, ncol, k + 1)
            plt.semilogx(
                gamma[:, k] * 100,
                GGmax[:, k],
                ls='-',
                marker='o',
                lw=1.5,
                alpha=0.8,
                label='Data points',
            )
            plt.semilogx(gamma_ * 100, GGmax_[:, k], lw=1.5, label='Curve fit',)
            plt.xlabel('Shear strain [%]')
            plt.ylabel('G/Gmax')
            plt.legend(loc='lower left')
            plt.grid(ls=':', lw=0.5)
            plt.title(
                r'$\gamma_{\mathrm{ref}}$ = %.3g, s = %.3g, $\beta$ = %.3g' \
                % (ref_strain[k], s_value[k], beta[k])
            )
        # END FOR
        plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)

    #---------- Produce fitting curves ----------------------------------------
    for k in range(n_ma):
        fitted_curves[:, k * 4 + 0] = gamma_ * 100
        fitted_curves[:, k * 4 + 1] = GGmax_[:, k]
        fitted_curves[:, k * 4 + 2] = gamma_ * 100
        fitted_curves[:, k * 4 + 3] = damping_[:, k] * 100

    return param, fitted_curves

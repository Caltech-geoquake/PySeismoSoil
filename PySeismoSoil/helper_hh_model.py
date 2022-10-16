import numpy as np

from PySeismoSoil import helper_generic as hlp
from PySeismoSoil import helper_mkz_model as mkz
from PySeismoSoil import helper_site_response as sr


def tau_FKZ(gamma, *, Gmax, mu, d, Tmax):
    """
    Calculate the FKZ shear stress. The FKZ model is proposed in Shi & Asimaki
    (2017), in Equation (6), and has the following form::

                        gamma^d * mu
        T(gamma) = ------------------------
                      1       gamma^d * mu
                    ------ + --------------
                     Gmax        Tmax

    where:
        + T     = shear stress
        + gamma = shear strain
        + Gmax  = initial shear modulus
        + d     = shape parameter
        + mu    = shape parameter
        + Tmax  = shear strength of soil

    Parameters
    ----------
    gamma : numpy.ndarray
        The shear strain array. Must be a 1D array. Its unit should be '1',
        rather than '%'.
    Gmax : float
        Initial shear modulus. Its unit can be arbitrary, but we recommend Pa.
    mu : float
        Shape parameter of the FKZ model.
    d : float
        Shape parameter of the FKZ model.
    Tmax : float
        Shear strength of soil. Its unit should match that of ``Gmax``.

    Returns
    -------
    T_FKZ : numpy.ndarray
        The shear stress determined by the formula above. Same shape as ``x``,
        and same unit as ``Gmax``.
    """
    hlp.assert_1D_numpy_array(gamma, name='`gamma`')
    T_FKZ = mu * Gmax * gamma**d / (1 + Gmax / Tmax * mu * np.abs(gamma) ** d)

    return T_FKZ


def transition_function(gamma, *, a, gamma_t):
    """
    Calculate the transition function of the HH model, as defined
    in Equation (7) of Shi & Asimaki (2017).

    Parameters
    ----------
    gamma : numpy.ndarray
        The shear strain array. Must be a 1D array. Its unit should be '1',
        rather than '%'.
    a : float
        A shape parameter describing how fast the transition happens.
    gamma_t : float
        Transition strain: the x value at which the transition happens.

    Returns
    -------
    w : numpy.ndarray
        The transition function, ranging from 0 to 1. Same shape as ``x``.
    """
    hlp.assert_1D_numpy_array(gamma, name='`gamma`')
    assert gamma_t > 0
    w = 1 - 1. / (1 + np.power(10, -a * (
        np.log10(np.abs(gamma) / gamma_t) - 4.039 * a ** (-1.036))))

    return w


def tau_HH(gamma, *, gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d):
    """
    Calculate the HH shear stress, which is proposed in Shi & Asimaki (2017).

    Parameters
    ----------
    gamma : numpy.ndarray
        The shear strain array. Must be a 1D array. Its unit should be '1',
        rather than '%'.
    gamma_t : float
        Transition strain: the x value at which the transition happens.
    a : float
        A shape parameter describing how fast the transition happens.
    gamma_ref : float
        Reference shear strain, a shape parameter of the MKZ model.
    beta : float
        A shape parameter of the MKZ model.
    s : float
        A shape parameter of the MKZ model.
    Gmax : float
        Initial shear modulus. Its unit can be arbitrary, but we recommend Pa.
    mu : float
        Shape parameter of the FKZ model.
    Tmax : float
        Shear strength of soil. Its unit should match that of Gmax.
    d : float
        Shape parameter of the FKZ model.

    Returns
    -------
    T_FKZ : numpy.ndarray
        The shear stress determined by the HH model. Same shape as ``x``,
        and same unit as ``Gmax``.
    """
    w = transition_function(gamma, a=a, gamma_t=gamma_t)
    T_MKZ = mkz.tau_MKZ(gamma, gamma_ref=gamma_ref, beta=beta, s=s, Gmax=Gmax)
    T_FKZ = tau_FKZ(gamma, Gmax=Gmax, mu=mu, d=d, Tmax=Tmax)

    T_HH = w * T_MKZ + (1 - w) * T_FKZ

    return T_HH


def fit_HH_x_single_layer(
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
    Perform HH_x curve fitting for one damping curve using the genetic
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
    hlp.check_two_column_format(
        damping_data_in_pct,
        name='damping_data_in_pct',
        ensure_non_negative=True,
    )

    init_damping = damping_data_in_pct[0, 1]  # small-strain damping
    damping_data_in_pct[:, 1] -= init_damping  # offset all dampings
    damping_data_in_unit_1 = damping_data_in_pct / 100  # unit: percent --> 1

    n_param = 9  # number of HH model parameters; do not change this for HH model
    N = 122  # denser strain array for more accurate damping calculation
    strain_dense = np.logspace(-6, -1, N)  # unit: 1
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
        _damping_misfit,
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
    best_param['gamma_t'] = 10 ** result[0]
    best_param['a'] = 10 ** result[1]
    best_param['gamma_ref'] = 10 ** result[2]
    best_param['beta'] = 10 ** result[3]
    best_param['s'] = 10 ** result[4]
    best_param['Gmax'] = 10 ** result[5]
    best_param['mu'] = 10 ** result[6]
    best_param['Tmax'] = 10 ** result[7]
    best_param['d'] = 10 ** result[8]

    if show_fig:
        sr._plot_damping_curve_fit(damping_data_in_pct, best_param, tau_HH)

    return best_param


def _damping_misfit(param, damping_data):
    """
    Calculate the misfit given a set of HH parameters. Note that the values
    in `param` are actually the 10-based power of the actual HH parameters.
    Using the powers in the genetic algorithm searching turns out to work
    much better for this particular problem.

    Parameters
    ----------
    param : tuple<float>
        HH model parameters, in the order specified below:
            gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d
    damping_data : numpy.ndarray
        2D numpy array with two columns (strain and damping value). Both
        columns need to use "1" as the unit, not percent.

    Returns
    -------
    error : float
        The mean absolute error between the true damping values and the
        predicted damping values at each strain level.
    """
    gamma_t_, a_, gamma_ref_, beta_, s_, Gmax_, mu_, Tmax_, d_ = param

    gamma_t = 10 ** gamma_t_
    a = 10 ** a_
    gamma_ref = 10 ** gamma_ref_
    beta = 10 ** beta_
    s = 10 ** s_
    Gmax = 10 ** Gmax_
    mu = 10 ** mu_
    Tmax = 10 ** Tmax_
    d = 10 ** d_

    strain = damping_data[:, 0]
    damping_true = damping_data[:, 1]

    Tau_HH = tau_HH(
        strain,
        gamma_t=gamma_t,
        a=a,
        gamma_ref=gamma_ref,
        beta=beta,
        s=s,
        Gmax=Gmax,
        mu=mu,
        Tmax=Tmax,
        d=d,
    )
    damping_pred = sr.calc_damping_from_stress_strain(strain, Tau_HH, Gmax)
    error = hlp.mean_absolute_error(damping_true, damping_pred)

    return error


def serialize_params_to_array(param):
    """
    Convert the HH parameters from a dictionary to an array, according to this
    order:
        gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d

    Parameters
    ----------
    param : dict
        A dictionary containing the parameters of the HH model.

    Returns
    -------
    param_array : numpy.ndarray
        A numpy array of shape (9,) containing the parameters of the HH model
        in the order specified above.
    """
    assert len(param) == 9
    order = ['gamma_t', 'a', 'gamma_ref', 'beta', 's', 'Gmax', 'mu', 'Tmax', 'd']
    param_array = []
    for key in order:
        param_array.append(param[key])

    return np.array(param_array)


def deserialize_array_to_params(array):
    """
    Reconstruct a HH model parameter dictionary from an array of values.

    The users need to ensure the order of values in ``array`` are in this order:
        gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d

    Parameters
    ----------
    array : numpy.ndarray
        A 1D numpy array of HH parameter values in this order: gamma_t, a,
        gamma_ref, beta, s, Gmax, mu, Tmax, d

    Returns
    -------
    param : dict
        The dictionary with parameter name as keys and values as values.
    """
    hlp.assert_1D_numpy_array(array)
    assert len(array) == 9

    param = {}
    param['gamma_t'] = array[0]
    param['a'] = array[1]
    param['gamma_ref'] = array[2]
    param['beta'] = array[3]
    param['s'] = array[4]
    param['Gmax'] = array[5]
    param['mu'] = array[6]
    param['Tmax'] = array[7]
    param['d'] = array[8]

    return param

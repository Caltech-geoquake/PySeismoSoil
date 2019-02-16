# Author: Jian Shi

import numpy as np
import matplotlib.pyplot as plt

from . import helper_generic as hlp
from . import helper_mkz_model as mkz
from . import helper_site_response as sr

#%%----------------------------------------------------------------------------
def tau_FKZ(gamma, *, Gmax, mu, d, Tmax):
    '''
    Calculate the FKZ shear stress. The FKZ model is proposed in Shi & Asimaki
    (2017), in Equation (6), and has the following form:

                        gamma^d * mu
        T(gamma) = ------------------------
                      1        x^d * mu
                    ------ + ------------
                     Gmax        Tmax

    where T     = shear stress
          gamma = shear strain
          Gmax  = initial shear modulus
          d     = shape parameter
          mu    = shape parameter
          Tmax  = shear strength of soil

    Parmeters
    ---------
    gamma : numpy.ndarray
        The shear strain array. Must be a 1D array. Its unit should be '1',
        rather than '%'.
    Gmax : float
        Initial shear modulus. Its unit can be arbitrary, but we recommend Pa.
    mu : float
        Shape parameter of the FKZ model
    d : float
        Shape parameter of the FKZ model
    Tmax : float
        Shear strength of soil. Its unit should match that of Gmax.

    Returns
    -------
    T_FKZ : numpy.ndarray
        The shear stress determined by the formula above. Same shape as `x`,
        and same unit as `Gmax`.
    '''
    hlp.assert_1D_numpy_array(gamma, name='`gamma`')
    T_FKZ = mu * Gmax * gamma**d / ( 1 + Gmax / Tmax * mu * np.abs(gamma)**d )

    return T_FKZ

#%%----------------------------------------------------------------------------
def transition_function(gamma, *, a, gamma_t):
    '''
    The transition function of the HH model, as defined in Equation (7) of Shi
    & Asimaki (2017).

    Parameters
    ----------
    gamma : numpy.ndarray
        The shear strain array. Must be a 1D array. Its unit should be '1',
        rather than '%'.
    a : float
        A shape parameter describing how fast the transition happens
    gamma_t : float
        Transition strain: the x value at which the transition happens

    Returns
    -------
    w : numpy.array
        The transition function, ranging from 0 to 1. Same shape as `x`.
    '''
    hlp.assert_1D_numpy_array(gamma, name='`gamma`')
    assert(gamma_t > 0)
    w = 1 - 1. / (1 + np.power(10, -a * (np.log10(np.abs(gamma)/gamma_t) \
                                         - 4.039 * a**(-1.036)) ))

    return w

#%%----------------------------------------------------------------------------
def tau_HH(gamma, *, gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d):
    '''
    Calculate the HH shear stress, which is proposed in Shi & Asimaki (2017).

    Parameters
    ----------
    gamma : numpy.ndarray
        The shear strain array. Must be a 1D array. Its unit should be '1',
        rather than '%'.
    gamma_t : float
        Transition strain: the x value at which the transition happens
    a : float
        A shape parameter describing how fast the transition happens
    gamma_ref : float
        Reference shear strain, a shape parameter of the MKZ model
    beta : float
        A shape parameter of the MKZ model
    s : float
        A shape parameter of the MKZ model
    Gmax : float
        Initial shear modulus. Its unit can be arbitrary, but we recommend Pa.
    mu : float
        Shape parameter of the FKZ model
    Tmax : float
        Shear strength of soil. Its unit should match that of Gmax.
    d : float
        Shape parameter of the FKZ model

    Returns
    -------
    T_FKZ : numpy.ndarray
        The shear stress determined by the HH model. Same shape as `x`,
        and same unit as `Gmax`.
    '''
    w = transition_function(gamma, a=a, gamma_t=gamma_t)
    T_MKZ = mkz.tau_MKZ(gamma, gamma_ref=gamma_ref, beta=beta, s=s, Gmax=Gmax)
    T_FKZ = tau_FKZ(gamma, Gmax=Gmax, mu=mu, d=d, Tmax=Tmax)

    T_HH = w * T_MKZ + (1 - w) * T_FKZ

    return T_HH

#%%----------------------------------------------------------------------------
def fit_HH_x_single_layer(damping_data_in_pct, population_size=800,
                          n_gen=100, lower_bound_power=-4, upper_bound_power=6,
                          eta=0.1, seed=0, show_fig=False, verbose=False):
    '''
    Perform HH_x curve fitting for one damping curve using the genetic
    algorithm provided in DEAP.

    Parameters
    ----------
    damping_data_in_pct : numpy.ndarray
        Damping data. Needs to have 2 columns (strain and damping ratio). Both
        columns need to use % as unit.
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
    best_param : dict
        The best parameters found in the optimization
    '''

    import random

    import deap.creator
    import deap.base
    import deap.algorithms
    import deap.tools

    import warnings  # suppress overflow warning when trying some parameters
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    hlp.check_two_column_format(damping_data_in_pct, ensure_non_negative=True)

    init_damping = damping_data_in_pct[0, 1]  # small-strain damping
    damping_data_in_pct[:, 1] -= init_damping  # offset all dampings
    damping_data_in_unit_1 = damping_data_in_pct / 100  # unit: percent --> 1

    NDIM = 9  # number of HH model parameters; do not change this for HH model
    N = 122  # make a denser data set which can help parameter searching
    strain_dense = np.logspace(-6, -1, N)
    damping_dense = np.interp(strain_dense, damping_data_in_unit_1[:, 0],
                              damping_data_in_unit_1[:, 1])

    damping_data_ = np.column_stack((strain_dense, damping_dense))

    def damping_misfit(param):
        '''
        Calculate the misfit given a set of HH parameters. Note that the values
        in `param` are actually the 10-based power of the actual HH parameters.
        Using the powers in the genetic algorithm searching turns out to work
        much better for this particular problem.
        '''
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

        strain = damping_data_[:, 0]
        damping_true = damping_data_[:, 1]

        Tau_HH = tau_HH(strain, gamma_t=gamma_t, a=a, gamma_ref=gamma_ref,
                        beta=beta, s=s, Gmax=Gmax, mu=mu, Tmax=Tmax, d=d)
        damping_pred = sr.calc_damping_from_stress_strain(strain, Tau_HH, Gmax)
        error = hlp.mean_absolute_error(damping_true, damping_pred)

        return error,

    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low]*size, [up]*size)]

    BOUND_LOW = lower_bound_power
    BOUND_UP = upper_bound_power

    deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
    deap.creator.create("Individual", list, fitness=deap.creator.FitnessMin)

    toolbox = deap.base.Toolbox()

    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", deap.tools.initIterate, deap.creator.Individual,
                     toolbox.attr_float)
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", damping_misfit)
    toolbox.register("mate", deap.tools.cxSimulatedBinaryBounded,
                     low=BOUND_LOW, up=BOUND_UP, eta=eta)
    toolbox.register("mutate", deap.tools.mutPolynomialBounded,
                     low=BOUND_LOW, up=BOUND_UP, eta=eta, indpb=1.0/NDIM)
    toolbox.register("select", deap.tools.selTournament, tournsize=10)

    random.seed(seed)

    pop = toolbox.population(n=population_size)
    hof = deap.tools.HallOfFame(1)
    stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", np.mean)
    stats.register("Std", np.std)
    stats.register("Min", np.min)
    stats.register("Max", np.max)

    deap.algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.8, ngen=n_gen,
                             stats=stats, halloffame=hof, verbose=verbose)

    hof_top = list(hof[0])

    best_param = {}
    best_param['gamma_t'] = 10 ** hof_top[0]
    best_param['a'] = 10 ** hof_top[1]
    best_param['gamma_ref'] = 10 ** hof_top[2]
    best_param['beta'] = 10 ** hof_top[3]
    best_param['s'] = 10 ** hof_top[4]
    best_param['Gmax'] = 10 ** hof_top[5]
    best_param['mu'] = 10 ** hof_top[6]
    best_param['Tmax'] = 10 ** hof_top[7]
    best_param['d'] = 10 ** hof_top[8]

    if show_fig:
        _plot_damping_curve_fit(damping_data_in_pct, best_param)

    return best_param

#%%----------------------------------------------------------------------------
def fit_HH_x_multi_layers(curves, population_size=800, n_gen=100,
                          lower_bound_power=-4, upper_bound_power=6,
                          eta=0.1, seed=0, show_fig=False, verbose=False,
                          parallel=False, n_cores=None):
    '''
    Perform HH_x curve fitting for multiple damping curves using the genetic
    algorithm provided in DEAP.

    Parameters
    ----------
    curves : numpy.ndarray or list<numpy.array>
        Can either be a 2D array in the "curve" format, or a list of individual
        damping curves.
        The "curve" format is as follows:

        strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ...
        -----------+--------+------------+-------------+-------------+--------+ ...
           ...     |  ...   |    ...     |    ...      |    ...      |  ...   |

        The G/Gmax information is redundant for this function.

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

    Return
    ------
    params : list<dict>
        The best parameters for each layer found in the optimization
    '''

    if isinstance(curves, np.ndarray):
        _, curves_list = hlp.extract_from_curve_format(curves)
    elif isinstance(curves, list):
        if not all([isinstance(_, np.ndarray) for _ in curves]):
            raise TypeError('If `curves` is a list, all its elements needs to '
                            'be 2D numpy arrays.')
        for j, curve in enumerate(curves):
            hlp.check_two_column_format(curve,
                                        name='Damping curve for layer #%d' % j,
                                        ensure_non_negative=True)
        curves_list = curves
    else:
        raise TypeError('Input data type of `curves` not recognized. '
                        'Please check the documentation of this function.')

    other_params = [(population_size, n_gen, lower_bound_power,
                     upper_bound_power, eta, seed, show_fig, verbose)]

    if parallel:
        import itertools
        import multiprocessing
        p = multiprocessing.Pool(n_cores)
        params = p.map(_fit_HH_x_loop, itertools.product(curves_list, other_params))
        if show_fig:
            for j, curve in enumerate(curves_list):
                _plot_damping_curve_fit(curve, params[j])
    else:
        params = []
        for curve in curves_list:
            params.append(_fit_HH_x_loop((curve, other_params[0])))

    return params

#%%----------------------------------------------------------------------------
def _fit_HH_x_loop(param):
    '''
    Loop body to be passed to the parallel pool.
    '''
    damping_curve, other_params = param

    population_size, n_gen, lower_bound_power, upper_bound_power, eta, seed, \
    show_fig, verbose = other_params

    best_para = fit_HH_x_single_layer(damping_curve, n_gen=n_gen, eta=eta,
                                      population_size=population_size,
                                      lower_bound_power=lower_bound_power,
                                      upper_bound_power=upper_bound_power,
                                      seed=seed, show_fig=show_fig,
                                      verbose=verbose)

    return best_para

#%%----------------------------------------------------------------------------
def _plot_damping_curve_fit(damping_data_in_pct, param,
                            min_strain_in_pct=1e-4, max_strain_in_pct=5):
    '''
    Plot damping data and curve-fit results together.

    Parameters
    ----------
    damping_data_in_pct : numpy.ndarray
        Damping data. Needs to have 2 columns (strain and damping ratio). Both
        columns need to use % as unit.
    param : dict
        HH_x parameters
    min_strain_in_pct, max_strain_in_pct : float
        Strain limits of the curve-fit result
    '''

    fig = plt.figure()
    ax = plt.axes()
    init_damping = damping_data_in_pct[0, 1]
    ax.semilogx(damping_data_in_pct[:, 0], damping_data_in_pct[:, 1],
                marker='o', alpha=0.8, label='data')

    min_strain_in_1 = min_strain_in_pct / 100.0
    max_strain_in_1 = max_strain_in_pct / 100.0
    strain = np.logspace(np.log10(min_strain_in_1), np.log10(max_strain_in_1))
    damping_curve_fit = calc_damping_from_param(param, strain)

    ax.semilogx(strain * 100, damping_curve_fit * 100 + init_damping,
                label='curve fit', alpha=0.8)
    ax.legend(loc='best')
    ax.grid(ls=':')
    ax.set_xlabel('Strain [%]')
    ax.set_ylabel('Damping ratio [%]')

    return fig, ax

#%%----------------------------------------------------------------------------
def serialize_params_to_array(param):
    '''
    Convert the HH parameters from a dictionary to an array, according to this
    order:
        gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d

    Parameter
    ---------
    param : dict
        A dictionary containing the parameters of the HH model

    Returns
    -------
    param_array : numpy.array
        A numpy array of shape (9,) containing the parameters of the HH model
        in the order specified above
    '''

    order = ['gamma_t', 'a', 'gamma_ref', 'beta', 's', 'Gmax', 'mu', 'Tmax', 'd']
    param_array = []
    for key in order:
        param_array.append(param[key])

    return np.array(param_array)

#%%----------------------------------------------------------------------------
def deserialize_array_to_params(array):
    '''
    Reconstruct a HH model parameter dictionary from an array of values.

    The users needs to ensure the order of values in `array` are in this order:
        gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d

    Parameter
    ---------
    array : numpy.ndarray
        A 1D numpy array of HH parameter values in this order: gamma_t, a,
        gamma_ref, beta, s, Gmax, mu, Tmax, d

    Returns
    -------
    param : dict
        The dictionary with parameter name as keys and values as values
    '''

    hlp.assert_1D_numpy_array(array)
    assert(len(array) == 9)

    param = dict()
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

# Author: Jian Shi

import numpy as np
import scipy.fftpack

from . import helper_generic as hlp
from . import helper_site_response as sr

#%%----------------------------------------------------------------------------
def check_layer_count(vs_profile, *, GGmax_and_damping_curves=None,
                      G_param=None, xi_param=None):
    '''
    Check that ``G_param`` and ``xi_param`` have enough sets of parameters for
    ``vs_profile``, or ``GGmax_curves`` and ``xi_curves`` have enough sets of
    curves for ``vs_profile``.

    Parameters
    ----------
    vs_profile : class_Vs_profile.Vs_Profile
        Vs profile.
    GGmax_and_damping_curves : class_curves.Multiple_GGmax_Damping_Curves
        G/Gmax and damping curves.
    G_param : class_parameters.HH_Param_Multi_Layer or MKZ_Param_Multi_Layer
        HH or MKZ parameters for G/Gmax curves.
    xi_param : class_parameters.HH_Param_Multi_Layer or MKZ_Param_Multi_Layer
        HH or MKZ parameters for damping curves.
    '''
    max_mat_num = np.max(vs_profile._material_number)
    if G_param is not None and G_param.n_layer < max_mat_num:
        raise ValueError('Not enough sets of parameters in `G_param` for '
                         '`vs_profile`.')
    if xi_param is not None and xi_param.n_layer < max_mat_num:
        raise ValueError('Not enough sets of parameters in `xi_param` for '
                         '`vs_profile`.')
    if GGmax_and_damping_curves is not None \
    and GGmax_and_damping_curves.n_layer < max_mat_num:
        raise ValueError('Not enough sets of curves in '
                         '`GGmax_and_damping_curves` for `vs_profile`.')
    return None

#%%----------------------------------------------------------------------------
def equiv_linear(vs_profile, input_motion, curve_matrix, boundary='elastic',
                 tol=0.075, R_gamma=0.65, max_iter=10, verbose=True):
    '''
    Equivalent linear site response simulation.

    Parameters
    ----------
    vs_profile : numpy.ndarray
        Shear-wave velocity profile, as a 2D numpy array. It should have the
        following columns:

            +---------------+----------+---------+---------+-----------------+
            | Thickness (m) | Vs (m/s) | Damping | Density | Material Number |
            +===============+==========+=========+=========+=================+
            |      ...      |   ...    |   ...   |   ...   |      ...        |
            +---------------+----------+---------+---------+-----------------+

    input_motion : numpy.ndarray
        Input acceleration on rock outcrop (unit: m/s/s). It should have two
        columns (time and acceleration).
    curve_matrix : numpy.ndarray
        A 2D numpy array that represents G/Gmax and damping curves of each
        layer, in the following format:
         +------------+--------+------------+-------------+-------------+--------+-----+
         | strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ... |
         +============+========+============+=============+=============+========+=====+
         |    ...     |  ...   |    ...     |    ...      |    ...      |  ...   | ... |
         +------------+--------+------------+-------------+-------------+--------+-----+

    boundary : {'elastic', 'rigid'}
        Boundary condition. 'Elastic' means that the input motion is the
        "rock outcrop" motion, and 'rigid' means that the input motion is
        the recorded motion at the bottom of the Vs profile.
    tol : float
        Tolerance level for convergence checking.
    R_gamma : float
        A dimensionless ratio to determine the "effective shear strain" from
        the maximum shear strain in the strain time history. Do not change the
        default value unless you really know what you are doing.
    max_iter : int
        Maximum number of iteration to run.
    verbose : bool
        Whether or not t show the iteration progress on the console.

    Return
    ------
    new_profile : numpy.ndarray
        Re-discretized Vs profile.
    freq_array : numpy.ndarray
        Frequency array of the transfer function.
    tf : numpy.ndarray
        Transfer function (complex-valued).
    accel_on_surface : numpy.ndarray
        Simulated acceleration on the ground surface (two-columed).
    out_a : numpy.ndarray
        Simulated acceleration time history of every layer.
        Shape: ``(num_time_step, n_layer)``.
    out_v : numpy.ndarray
        Simulated velocity time history of every layer. Same shape as ``out_a``.
    out_d : numpy.ndarray
        Simulated displacement time history of every layer. Same shape as
        ``out_a``.
    out_gamma : numpy.ndarray
        Simulated shear strain time history of every layer.
        Shape: ``(num_time_step, n_layer - 1)``.
    out_tau : numpy.ndarray
        Simulated shear stress time history of every layer. Same shape as
        ``out_gamma``.
    max_avd : numpy.ndarray
        Maximum acceleration, velocity, and displacement during the shaking
        process, of each layer. Shape: ``(n_layer, )``.
    max_gt : numpy.ndarray
        Maximum shear strain and shear stress during the shaking process, of
        each layer. Shape: ``(n_layer - 1, )``.

    Notes
    -----
    Based on the MATLAB function written by Wei Li and Jian Shi.
    '''
    hlp.check_Vs_profile_format(vs_profile)
    hlp.assert_2D_numpy_array(input_motion, name='`input_motion`')
    hlp.assert_2D_numpy_array(curve_matrix, name='`curve_matrix`')

    #-------- Part 1.1: Data preparation -- input ground motion ---------------
    incident_motion = input_motion.copy()
    incident_motion[:, 1] /= 2

    t = incident_motion[:, 0]
    dt = t[1] - t[0]
    accel_in = incident_motion[:, 1]
    n = len(accel_in)

    if n % 2 == 0:  # n is even
        flag = 0
        t = np.append(t, t[-1] + dt)  # pad one time point at the end
        accel_in = np.append(accel_in, accel_in[-1])
        n += 1
    else:
        flag = 1
    # END IF

    ACCEL_IN = scipy.fftpack.fft(accel_in)
    N = len(ACCEL_IN)
    assert(N == n)
    assert(N % 2 == 1)

    f = np.arange(1, N+1, 1) / (N * dt)  # frequency

    #-------- Part 1.2: Data preparation -- soil profile ----------------------
    new_profile = sr.stratify(vs_profile)
    h = new_profile[:, 0]
    vs = new_profile[:, 1]
    D = new_profile[:, 2]
    rho = new_profile[:, 3]
    mat_nr = new_profile[:, 4].astype(int) - 1  # from 1-indexing to 0-indexing

    n_layer = len(h)  # includes the rock layer
    Gmax = rho * vs ** 2.0
    G = Gmax.copy()  # initial value of G; Gmax cannot change, so needs a hard copy

    layer_boundary_depth = sr.thk2dep(h, midpoint=False)
    layer_midpoint_depth = sr.thk2dep(h, midpoint=True)

    #-------- Part 1.3: Data preparation -- modulus/damping curves ------------
    n_obs = curve_matrix.shape[0]  # number of strain points in a curve
    strain_G = np.zeros((n_obs, n_layer - 1))
    G_vector = np.zeros((n_obs, n_layer - 1))
    strain_D = np.zeros((n_obs, n_layer - 1))
    D_vector = np.zeros((n_obs, n_layer - 1))
    for k in range(n_layer - 1):  # map original curves to new layers
        strain_G[:, k] = curve_matrix[:, mat_nr[k] * 4 + 0] / 100.0
        G_vector[:, k] = curve_matrix[:, mat_nr[k] * 4 + 1]
        strain_D[:, k] = curve_matrix[:, mat_nr[k] * 4 + 2] / 100.0
        D_vector[:, k] = curve_matrix[:, mat_nr[k] * 4 + 3] / 100.0

    for k in range(n_layer - 1):  # offset initial damping value
        D_vector[:, k] = D_vector[:, k] - D_vector[0, k] + D[k]

    #-------- Part 2: Start iteration -----------------------------------------
    G_matrix = np.zeros((n_layer - 1, max_iter + 1))  # to store G of all iterations
    D_matrix = np.zeros((n_layer - 1, max_iter + 1))
    G_matrix[:, 0] = G[:-1]  # initial values
    D_matrix[:, 0] = D[:-1]

    for i_iter in range(max_iter):
        if verbose:
            print('Iteration No.%d.' % (i_iter + 1), end='')

        #------- Part 2.1: Linear transfer function ---------------------------
        # Make a denser frequency array to eliminate aliasing
        df = f[1] - f[0]
        max_f = np.max(f)
        f_oversample_factor = 15
        df_ = df / f_oversample_factor
        N_original = N
        N = int(N * f_oversample_factor)
        f = np.linspace(df_, max_f, num=N)
        omega = 2 * np.pi * f  # angular frequency
        half_N = int(N / 2 + 0.5)

        # Complex impedance ratio between each layer
        alpha = np.zeros(n_layer - 1, dtype=np.complex_)
        for j in range(n_layer - 1):  # layer by layer
            alpha[j] = (rho[j] * np.sqrt(G[j] * (1 + 2 * 1j * D[j]) / rho[j])) \
                       / (rho[j+1] * np.sqrt(G[j+1] * (1 + 2 * 1j * D[j+1])/rho[j+1]))

        if boundary == 'rigid':
            alpha[-1] = 0  # disallow energy transmission past the boundary

        # Complex shear-wave velocities of each layer (Kramer's book, page 260)
        vs_star = np.sqrt(G * (1 + 2 * 1j * D) /rho)  # shape: (n_layer, )
        assert(vs_star.shape == (n_layer, ))

        # Complex wave number (Kramer's book, page 260)
        vs_star_recip = (1.0 / vs_star).reshape((1, n_layer))  # (1, n_layer)
        k_star = omega[:half_N].reshape(half_N, 1) * vs_star_recip  # (half_N, n_layer)
        assert(k_star.shape == (half_N, n_layer))

        # Compute A and B (Kramer's book, page 269)
        A = np.zeros((half_N, n_layer), dtype=np.complex_)
        B = np.zeros((half_N, n_layer), dtype=np.complex_)
        A[:, 0] = 1
        B[:, 0] = 1
        for k in range(n_layer - 1):  # layer by layer
            A[:,k+1] \
                = 0.5*A[:,k] * (1+alpha[k]) * np.exp(1j*k_star[:,k]*h[k]) + \
                  0.5*B[:,k] * (1-alpha[k]) * np.exp(-1j*k_star[:,k]*h[k])  # left half
            B[:,k+1] \
                = 0.5*A[:,k] * (1-alpha[k]) * np.exp(1j*k_star[:,k]*h[k]) + \
                  0.5*B[:,k] * (1+alpha[k]) * np.exp(-1j*k_star[:,k]*h[k])  # left half

        # Compute linear transfer function
        H_ss = np.zeros((half_N, n_layer), dtype=np.complex_)  # single-sided transfer function
        H_append = np.zeros((half_N - 1, n_layer), dtype=np.complex_)  # the other half
        for k in range(n_layer):
            H_ss[:, k] = (A[:, k] + B[:, k]) / A[:, -1]
            H_ss[0, k] = np.real(H_ss[0, k])  # see Note (1) below
            H_append[:, k] = np.conj(np.flipud(H_ss[1:, k]))
        # END FOR

        H = np.row_stack((H_ss, H_append))

        H = H[::f_oversample_factor, :]  # down-sample back to original resolution
        f = f[::f_oversample_factor]
        N = N_original

        '''
        Notes:
         (1) The value of transfer function when freq = 0 should be either 1
             or 2 (a real number). Because of discretization errors, tf_ss(1)
            is not a real number, but very close.

         (2) Two examples:
            [a] fft([1, 2, 3, 4, 5, 6, 7, 8]) =
                  36
                  -4 + 9.6569i
                  -4 + 4.0000i
                  -4 + 1.6569i
                  -4
                  -4 - 1.6569i
                  -4 - 4.0000i
                  -4 - 9.6569i

            [b] fft([1, 2, 3, 4, 5, 6, 7]') =
                  28.0
                  -3.5 + 7.2678i
                  -3.5 + 2.7912i
                  -3.5 + 0.7989i
                  -3.5 - 0.7989i
                  -3.5 - 2.7912i
                  -3.5 - 7.2678i
        '''

        #------- Part 2.2: Response motion of each layer ----------------------
        ACCEL_OUT = H * ACCEL_IN.reshape(-1, 1)  # amplify accel. of each layer
        accel_out = np.real(scipy.fftpack.ifft(ACCEL_OUT, axis=0))  # column-wise
        veloc = np.cumsum(accel_out, axis=0) * dt
        displ = np.cumsum(veloc, axis=0) * dt

        # simple baseline correction of displacement time history
        offset = np.matmul(np.arange(1, N + 1, 1).reshape(-1, 1) / (N + 1.0),
                           displ[-1, :].reshape(1, -1))
        displ -= offset

        #------- Part 2.3: Strain time history and effective strain -----------
        strain = np.zeros((N, n_layer - 1))
        eff_strain = np.zeros(n_layer - 1)  # Kramer's book, pages 271-272
        for k in range(n_layer - 1):  # layer by layer
            strain[:, k] = (displ[:, k] - displ[:, k+1]) / h[k]  # unit: 1
            eff_strain[k] = R_gamma * np.max(np.abs(strain[:, k]))  # unit: 1
        # END FOR

        #------- Part 2.4: New modulus and damping ----------------------------
        G_new = np.zeros(n_layer - 1)
        D_new = np.zeros(n_layer - 1)
        for k in range(n_layer - 1):  # layer by layer
            # set upper/lower bounds for interpolation
            strain_G_ = np.append(0, np.append(strain_G[:, k], 100))
            G_vector_ = np.append(1,np.append(G_vector[:, k],
                                              np.min([1e-4, G_vector[-1, k]])))
            strain_D_ = np.append(0, np.append(strain_D[:, k], 100))
            D_vector_ = np.append(D_vector[0, k],
                                  np.append(D_vector[:, k], D_vector[-1, k]))

            G_new[k] = Gmax[k] * np.interp(eff_strain[k], strain_G_, G_vector_)
                       # ^ Interpolation needs to start from Gmax[k], otherwise
                       # the shear modulus values would get smaller and smaller
                       # and eventually to 0.
            D_new[k] = np.interp(eff_strain[k], strain_D_, D_vector_)
        # END FOR
        G_relative_diff = np.abs(G[:-1] - G_new) / G_new
        D_relative_diff = np.abs(D[:-1] - D_new) / D_new
        G[:-1] = G_new
        D[:-1] = D_new
        G_matrix[:, i_iter + 1] = G_new
        D_matrix[:, i_iter + 1] = D_new
        if verbose:
            print('  G_diff = %7.2f%%, D_diff = %7.2f%%' \
                  % (np.max(G_relative_diff) * 100, np.max(D_relative_diff) * 100))

        #--------- Part 2.5: Check whether to end iteration or not ------------
        if np.max(G_relative_diff) < tol and np.max(D_relative_diff) < tol:
            print('---------- Convergence achieved ---------------')
            break
        # END IF
    # END FOR

    #--------- Part 3: Calculate stress from strain ---------------------------
    stress_fft = np.zeros((N, n_layer - 1), dtype=np.complex_)

    strain_fft = scipy.fftpack.fft(strain, axis=0)
    modulus = (G * (1 + 2 * 1j * D))[:-1]

    half_N = int(N / 2 + 0.5)
    modulus_repmat = np.tile(modulus, (half_N, 1))
    stress_fft[:half_N, :] = strain_fft[:half_N, :] * modulus_repmat
    stress_fft[half_N:, :] = np.flipud(np.conj(stress_fft[1:half_N, :]))
    stress_fft[0, :] = np.real(stress_fft[0, :])
    stress = np.real(scipy.fftpack.ifft(stress_fft, axis=0))

    #--------- Part 4: Odd-even check -----------------------------------------
    if flag == 0:  # originally the input motion length is even
        freq_array = f[:half_N-1]
        tf = H[:half_N-1, 0] / 2.0
        accel_out = accel_out[:-1, :]
        veloc = veloc[:-1, :]
        displ = displ[:-1, :]
        strain = strain[:-1, :]
        stress = stress[:-1, :]
        accel_on_surface = np.column_stack((t[:-1], accel_out[:, 0]))
    else:
        freq_array = f[:half_N]
        tf = H[:half_N, 0] / 2.0
        accel_on_surface = np.column_stack((t, accel_out[:, 0]))

    #--------- Part 6: Post-processing ----------------------------------------
    out_gamma = strain
    out_tau = stress
    out_a = accel_out
    out_v = veloc
    out_d = displ

    max_a = np.max(np.abs(out_a), axis=0)
    max_v = np.max(np.abs(out_v), axis=0)
    max_d = np.max(np.abs(out_d), axis=0)
    max_gamma = np.max(np.abs(out_gamma), axis=0)
    max_tau = np.max(np.abs(out_tau), axis=0)

    max_avd = np.column_stack((layer_boundary_depth, max_a, max_v, max_d))
    max_gt = np.column_stack((layer_midpoint_depth, max_gamma, max_tau))

    return (new_profile, freq_array, tf, accel_on_surface, out_a, out_v,
            out_d, out_gamma, out_tau, max_avd, max_gt)


"""Helper functions for site response simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.fftpack

from PySeismoSoil import helper_generic as hlp
from PySeismoSoil import helper_site_response as sr

# Weight of the newly calculated effective strains (vs. those of the previous
# iteration) in each iteration of the frequency-dependent equivalent linear
# simulation
FD_EQL_RELAXATION = 0.5

if TYPE_CHECKING:
    from PySeismoSoil.class_curves import Multiple_GGmax_Damping_Curves
    from PySeismoSoil.class_parameters import (
        HH_Param_Multi_Layer,
        MKZ_Param_Multi_Layer,
    )
    from PySeismoSoil.class_Vs_profile import Vs_Profile


def check_layer_count(
        vs_profile: Vs_Profile,
        *,
        GGmax_and_damping_curves: Multiple_GGmax_Damping_Curves = None,
        G_param: HH_Param_Multi_Layer | MKZ_Param_Multi_Layer = None,
        xi_param: HH_Param_Multi_Layer | MKZ_Param_Multi_Layer = None,
) -> None:
    """
    Check that there are enough sets of parameters/curves for ``vs_profile``.

    Either ``G_param`` and ``xi_param`` have enough sets of parameters for
    ``vs_profile``, or ``GGmax_curves`` and ``xi_curves`` have enough sets of
    curves for ``vs_profile``.

    Parameters
    ----------
    vs_profile : Vs_Profile
        Vs profile.
    GGmax_and_damping_curves : Multiple_GGmax_Damping_Curves, default=None
        G/Gmax and damping curves.
    G_param : HH_Param_Multi_Layer | MKZ_Param_Multi_Layer, default=None
        HH or MKZ parameters for G/Gmax curves.
    xi_param : HH_Param_Multi_Layer | MKZ_Param_Multi_Layer, default=None
        HH or MKZ parameters for damping curves.

    Raises
    ------
    ValueError
        When checks fail
    """
    max_mat_num = np.max(vs_profile._material_number)  # noqa: SLF001
    if G_param is not None and G_param.n_layer < max_mat_num:
        raise ValueError(
            'Not enough sets of parameters in `G_param` for `vs_profile`.'
        )

    if xi_param is not None and xi_param.n_layer < max_mat_num:
        raise ValueError(
            'Not enough sets of parameters in `xi_param` for `vs_profile`.',
        )

    if (
        GGmax_and_damping_curves is not None
        and GGmax_and_damping_curves.n_layer < max_mat_num
    ):
        raise ValueError(
            'Not enough sets of curves in `GGmax_and_damping_curves` for'
            ' `vs_profile`.',
        )


def linear(
        vs_profile: np.ndarray,
        input_motion: np.ndarray,
        boundary: Literal['elastic', 'rigid'] = 'elastic',
) -> tuple[np.ndarray, ...]:
    """
    Linear site response simulation.

    ``helper_site_response.linear_site_resp()`` also performs linear site
    response calculation. The difference between this function and the other
    is: this function can produce the time histories of acceleration, velocity,
    displacement, stress, and strain of every layer, while the other function
    only produces the ground motion time histories on the ground surface.

    If the user only wants the ground surface motion, then the other function
    (``linear_site_resp()``) is faster.

    Parameters
    ----------
    vs_profile : np.ndarray
        Shear-wave velocity profile, as a 2D numpy array. It should have the
        following columns:

         +---------------+----------+---------+------------------+--------------+
         | Thickness [m] | Vs [m/s] | Damping | Density [kg/m^3] | Material No. |
         +===============+==========+=========+==================+==============+
         |      ...      |   ...    |   ...   |       ...        |      ...     |
         +---------------+----------+---------+------------------+--------------+
        (Damping unit: 1)

    input_motion : np.ndarray
        Input acceleration on rock outcrop (unit: m/s/s). It should have two
        columns (time and acceleration). It should be the "rock outrcop" motion
        if ``boundary`` is set to ``"elastic"``, and it should be the recorded
        motion at the bottom of the Vs profile (i.e., the "borehole" motion) if
        ``boundary`` is set to ``"rigid"``.
    boundary : Literal['elastic', 'rigid'], default='elastic'
        Boundary condition. 'Elastic' means that the input motion is the "rock
        outcrop" motion, and 'rigid' means that the input motion is the
        recorded motion at the bottom of the Vs profile.

    Returns
    -------
    tuple[np.ndarray, ...]
        A tuple of 11 numpy arrays:

        - ``new_profile``: Re-discretized Vs profile.
        - ``freq_array``: "Single-sided" frequency. Shape: ``(half_N, )`` or
          ``(half_N - 1, )``.
        - ``tf``: Transfer function (complex-valued). Same shape as
          ``freq_array``.
        - ``accel_on_surface``: Simulated acceleration on the ground surface
          (two-columed).
        - ``out_a``: Simulated acceleration time history of every layer. Shape:
          ``(num_time_step, n_layer)``.
        - ``out_v``: Simulated velocity time history of every layer. Same shape
          as ``out_a``.
        - ``out_d``: Simulated displacement time history of every layer. Same
          shape as ``out_a``.
        - ``out_gamma``: Simulated shear strain time history of every layer.
          Shape: ``(num_time_step, n_layer - 1)``.
        - ``out_tau``: Simulated shear stress time history of every layer. Same
          shape as ``out_gamma``.
        - ``max_avd``: Maximum acceleration, velocity, and displacement during
          the shaking process, of each layer. Shape: ``(n_layer, )``.
        - ``max_gt``: Maximum shear strain and shear stress during the shaking
          process, of each layer. Shape: ``(n_layer - 1, )``.
    """  # noqa: E501
    hlp.check_Vs_profile_format(vs_profile)
    hlp.assert_2D_numpy_array(input_motion, name='`input_motion`')

    # -------- Part 1: Data preparation -- soil profile and input motion -----
    (
        flag,
        N,
        freq,
        new_profile,
        h,
        _vs,
        D,
        rho,
        _mat_nr,
        n_layer,
        _Gmax,
        G,
        t,
        dt,
        ACCEL_IN,
    ) = _prepare_inputs(vs_profile=vs_profile, input_motion=input_motion)

    # -------- Part 2: Start calculation -------------------------------------
    H, accel_out, veloc, displ, strain, _ = _lin_resp_every_layer(
        dt=dt,
        freq=freq,
        N=N,
        n_layer=n_layer,
        h=h,
        G=G,
        D=D,
        rho=rho,
        boundary=boundary,
        ACCEL_IN=ACCEL_IN,
    )

    # --------- Part 3: Calculate stress from strain -------------------------
    stress, half_N = _calc_stress(
        G=G, D=D, strain=strain, N=N, n_layer=n_layer
    )

    # --------- Part 4: Post-processing ---------------------------------------
    (
        freq_array,
        tf,
        accel_on_surface,
        out_a,
        out_v,
        out_d,
        out_gamma,
        out_tau,
        max_avd,
        max_gt,
    ) = _post_processing(
        flag=flag,
        freq=freq,
        half_N=half_N,
        H=H,
        t=t,
        accel_out=accel_out,
        veloc=veloc,
        displ=displ,
        strain=strain,
        stress=stress,
        h=h,
    )

    return (
        new_profile,
        freq_array,
        tf,
        accel_on_surface,
        out_a,
        out_v,
        out_d,
        out_gamma,
        out_tau,
        max_avd,
        max_gt,
    )


def equiv_linear(
        vs_profile: np.ndarray,
        input_motion: np.ndarray,
        curve_matrix: np.ndarray,
        boundary: Literal['elastic', 'rigid'] = 'elastic',
        tol: float = 0.075,
        R_gamma: float = 0.65,
        max_iter: int = 10,
        *,
        verbose: bool = True,
) -> tuple[np.ndarray, ...]:
    """
    Equivalent linear site response simulation.

    Parameters
    ----------
    vs_profile : np.ndarray
        Shear-wave velocity profile, as a 2D numpy array. It should have the
        following columns:

         +---------------+----------+---------+------------------+--------------+
         | Thickness [m] | Vs [m/s] | Damping | Density [kg/m^3] | Material No. |
         +===============+==========+=========+==================+==============+
         |      ...      |   ...    |   ...   |       ...        |      ...     |
         +---------------+----------+---------+------------------+--------------+
        (Damping unit: 1)

    input_motion : np.ndarray
        Input acceleration on rock outcrop (unit: m/s/s). It should have two
        columns (time and acceleration). It should be the "rock outrcop" motion
        if ``boundary`` is set to ``"elastic"``, and it should be the recorded
        motion at the bottom of the Vs profile (i.e., the "borehole" motion) if
        ``boundary`` is set to ``"rigid"``.
    curve_matrix : np.ndarray
        A 2D numpy array that represents G/Gmax and damping curves of each
        layer, in the following format:
         +------------+--------+------------+-------------+-------------+--------+-----+
         | strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ... |
         +============+========+============+=============+=============+========+=====+
         |    ...     |  ...   |    ...     |    ...      |    ...      |  ...   | ... |
         +------------+--------+------------+-------------+-------------+--------+-----+

    boundary : Literal['elastic', 'rigid'], default='elastic'
        Boundary condition. 'Elastic' means that the input motion is the "rock
        outcrop" motion, and 'rigid' means that the input motion is the
        recorded motion at the bottom of the Vs profile.
    tol : float, default=0.075
        Tolerance level for convergence checking.
    R_gamma : float, default=0.65
        A dimensionless ratio to determine the "effective shear strain" from
        the maximum shear strain in the strain time history. Do not change the
        default value unless you really know what you are doing.
    max_iter : int, default=10
        Maximum number of iteration to run.
    verbose : bool, default=True
        Whether t show the iteration progress on the console.

    Returns
    -------
    tuple[np.ndarray, ...]
        A tuple of 11 numpy arrays:

        - ``new_profile``: Re-discretized Vs profile.
        - ``freq_array``: "Single-sided" frequency. Shape: ``(half_N, )`` or
          ``(half_N - 1, )``.
        - ``tf``: Transfer function (complex-valued). Same shape as
          ``freq_array``.
        - ``accel_on_surface``: Simulated acceleration on the ground surface
          (two-columed).
        - ``out_a``: Simulated acceleration time history of every layer. Shape:
          ``(num_time_step, n_layer)``.
        - ``out_v``: Simulated velocity time history of every layer. Same shape
          as ``out_a``.
        - ``out_d``: Simulated displacement time history of every layer. Same
          shape as ``out_a``.
        - ``out_gamma``: Simulated shear strain time history of every layer.
          Shape: ``(num_time_step, n_layer - 1)``.
        - ``out_tau``: Simulated shear stress time history of every layer. Same
          shape as ``out_gamma``.
        - ``max_avd``: Maximum acceleration, velocity, and displacement during
          the shaking process, of each layer. Shape: ``(n_layer, )``.
        - ``max_gt``: Maximum shear strain and shear stress during the shaking
          process, of each layer. Shape: ``(n_layer - 1, )``.

    Notes
    -----
    Based on the MATLAB function written by Wei Li and Jian Shi.
    """  # noqa: E501
    hlp.check_Vs_profile_format(vs_profile)
    hlp.assert_2D_numpy_array(input_motion, name='`input_motion`')
    hlp.assert_2D_numpy_array(curve_matrix, name='`curve_matrix`')

    # -------- Part 1.1: Data preparation -- soil profile and input motion ----
    (
        flag,
        N,
        freq,
        new_profile,
        h,
        _vs,
        D,
        rho,
        mat_nr,
        n_layer,
        Gmax,
        G,
        t,
        dt,
        ACCEL_IN,
    ) = _prepare_inputs(vs_profile=vs_profile, input_motion=input_motion)

    # -------- Part 1.2: Data preparation -- modulus/damping curves -----------
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

    # -------- Part 2: Start iteration ----------------------------------------
    # to store G and D of all iterations
    G_matrix = np.zeros((n_layer - 1, max_iter + 1))
    D_matrix = np.zeros((n_layer - 1, max_iter + 1))
    G_matrix[:, 0] = G[:-1]  # initial values
    D_matrix[:, 0] = D[:-1]

    for i_iter in range(max_iter):
        if verbose:
            print(f'Iteration No.{i_iter + 1}.', end='')

        H, accel_out, veloc, displ, strain, eff_strain = _lin_resp_every_layer(
            dt=dt,
            freq=freq,
            N=N,
            n_layer=n_layer,
            h=h,
            G=G,
            D=D,
            rho=rho,
            boundary=boundary,
            ACCEL_IN=ACCEL_IN,
            R_gamma=R_gamma,
        )

        # ------- Update modulus and damping ----------------------------------
        G_new = np.zeros(n_layer - 1)
        D_new = np.zeros(n_layer - 1)
        for k in range(n_layer - 1):  # layer by layer
            # set upper/lower bounds for interpolation
            strain_G_ = np.append(0, np.append(strain_G[:, k], 100))
            G_vector_ = np.append(
                1,
                np.append(G_vector[:, k], np.min([1e-4, G_vector[-1, k]])),
            )
            strain_D_ = np.append(0, np.append(strain_D[:, k], 100))
            D_vector_ = np.append(
                D_vector[0, k],
                np.append(D_vector[:, k], D_vector[-1, k]),
            )

            G_new[k] = Gmax[k] * np.interp(eff_strain[k], strain_G_, G_vector_)
            #            ^ Interpolation needs to start from Gmax[k], otherwise
            #            the shear modulus values would get smaller and smaller
            #            and eventually to 0.

            D_new[k] = np.interp(eff_strain[k], strain_D_, D_vector_)

        G_relative_diff = np.abs(G[:-1] - G_new) / G_new
        D_relative_diff = np.abs(D[:-1] - D_new) / D_new
        G[:-1] = G_new
        D[:-1] = D_new
        G_matrix[:, i_iter + 1] = G_new
        D_matrix[:, i_iter + 1] = D_new
        if verbose:
            print(
                f'  G_diff = {np.max(G_relative_diff) * 100:7.2f}%,'
                f' D_diff = {np.max(D_relative_diff) * 100:7.2f}%',
            )

        # --------- Check convergence -----------------------------------------
        if (
            np.max(G_relative_diff) < tol
            and np.max(D_relative_diff) < tol
            and verbose
        ):
            print('---------- Convergence achieved ---------------')
            break

    # --------- Part 3: Calculate stress from strain --------------------------
    stress, half_N = _calc_stress(
        G=G, D=D, strain=strain, N=N, n_layer=n_layer
    )

    # --------- Part 4: Post-processing ---------------------------------------
    (
        freq_array,
        tf,
        accel_on_surface,
        out_a,
        out_v,
        out_d,
        out_gamma,
        out_tau,
        max_avd,
        max_gt,
    ) = _post_processing(
        flag=flag,
        freq=freq,
        half_N=half_N,
        H=H,
        t=t,
        accel_out=accel_out,
        veloc=veloc,
        displ=displ,
        strain=strain,
        stress=stress,
        h=h,
    )

    return (
        new_profile,
        freq_array,
        tf,
        accel_on_surface,
        out_a,
        out_v,
        out_d,
        out_gamma,
        out_tau,
        max_avd,
        max_gt,
    )


def equiv_linear_freq_dependent(
        vs_profile: np.ndarray,
        input_motion: np.ndarray,
        curve_matrix: np.ndarray,
        boundary: Literal['elastic', 'rigid'] = 'elastic',
        tol: float = 0.075,
        R_gamma: float = 0.65,
        max_iter: int = 10,
        *,
        verbose: bool = True,
) -> tuple[np.ndarray, ...]:
    """
    Frequency-dependent equivalent linear site response simulation.

    In each iteration, the shear modulus and the damping of each layer depend
    on frequency: they are interpolated from the G/Gmax and damping curves at a
    frequency-dependent effective strain, which follows the smoothed shape of
    the Fourier spectrum of the strain proposed by Kausel & Assimaki (2002).
    The shape is 1 below the "average frequency" of the strain spectrum and
    decays as ``exp(-a * f) / f**b`` above it. Its peak is ``R_gamma`` times
    the maximum strain.

    Parameters
    ----------
    vs_profile : np.ndarray
        Shear-wave velocity profile, as a 2D numpy array with 5 columns
        (thickness, Vs, damping, density, and material number). The damping
        unit is 1, and the density unit is kg/m^3.
    input_motion : np.ndarray
        Input acceleration (unit: m/s/s), with two columns (time and
        acceleration). It should be the "rock outcrop" motion if ``boundary``
        is ``"elastic"``, and the recorded motion at the bottom of the Vs
        profile (i.e., the "borehole" motion) if ``boundary`` is ``"rigid"``.
    curve_matrix : np.ndarray
        G/Gmax and damping curves of each material, in the "curve" format
        (strain [%], G/Gmax, strain [%], damping [%], strain [%], ...).
    boundary : Literal['elastic', 'rigid'], default='elastic'
        Boundary condition.
    tol : float, default=0.075
        Tolerance level for convergence checking: the largest relative change
        of the maximum strain of any layer between two iterations.
    R_gamma : float, default=0.65
        Ratio of the effective strain (at the low frequencies) to the maximum
        strain in the strain time history.
    max_iter : int, default=10
        Maximum number of iteration to run.
    verbose : bool, default=True
        Whether to show the iteration progress on the console.

    Returns
    -------
    tuple[np.ndarray, ...]
        A tuple of 11 numpy arrays, the same as those of ``equiv_linear()``.

    References
    ----------
    1. Kausel & Assimaki (2002) "Seismic simulation of inelastic soils via
       frequency-dependent moduli and damping." Journal of Engineering
       Mechanics, 128(1), 34-47.
    """
    hlp.check_Vs_profile_format(vs_profile)
    hlp.assert_2D_numpy_array(input_motion, name='`input_motion`')
    hlp.assert_2D_numpy_array(curve_matrix, name='`curve_matrix`')

    (
        flag,
        N,
        _freq,
        new_profile,
        h,
        _vs,
        D,
        rho,
        mat_nr,
        n_layer,
        Gmax,
        _G,
        t,
        dt,
        ACCEL_IN,
    ) = _prepare_inputs(vs_profile=vs_profile, input_motion=input_motion)

    half_N = (N + 1) // 2  # `N` is odd
    freq = np.arange(N) / (N * dt)  # frequencies of the FFT bins

    # Curves of every layer (strain and damping in unit of 1). The damping
    # curves start from the damping of the Vs profile.
    log_strain_G, G_vector, log_strain_D, D_vector = [], [], [], []
    for k in range(n_layer - 1):
        col = mat_nr[k] * 4
        log_strain_G.append(np.log10(curve_matrix[:, col] / 100.0))
        G_vector.append(curve_matrix[:, col + 1])
        log_strain_D.append(np.log10(curve_matrix[:, col + 2] / 100.0))
        damping = curve_matrix[:, col + 3] / 100.0
        D_vector.append(damping - damping[0] + D[k])

    # Modulus and damping: shape (half_N, n_layer)
    G_f = np.tile(Gmax, (half_N, 1))
    D_f = np.tile(D, (half_N, 1))
    eff_strain = np.zeros((half_N, n_layer - 1))
    max_strain_prev = np.zeros(n_layer - 1)

    for i_iter in range(max_iter):
        if verbose:
            print(f'Iteration No.{i_iter + 1}.', end='')

        _, _, _, _, strain = _freq_dependent_response(
            freq=freq,
            dt=dt,
            h=h,
            G_f=G_f,
            D_f=D_f,
            rho=rho,
            boundary=boundary,
            ACCEL_IN=ACCEL_IN,
        )
        max_strain = np.max(np.abs(strain), axis=0)
        strain_diff = np.max(np.abs(max_strain - max_strain_prev) / max_strain)
        max_strain_prev = max_strain

        for k in range(n_layer - 1):
            strain_fas = np.abs(scipy.fftpack.fft(strain[:, k]))[:half_N]
            eff_strain_new = (
                R_gamma
                * max_strain[k]
                * _ka02_strain_shape(freq[:half_N], strain_fas)
            )
            # Under-relaxation, to damp out the oscillation between two
            # states that this iteration is prone to
            eff_strain[:, k] = (
                eff_strain_new
                if i_iter == 0
                else FD_EQL_RELAXATION * eff_strain_new
                + (1 - FD_EQL_RELAXATION) * eff_strain[:, k]
            )
            log_eff_strain = np.log10(np.maximum(eff_strain[:, k], 1e-12))
            G_f[:, k] = Gmax[k] * np.interp(
                log_eff_strain, log_strain_G[k], G_vector[k]
            )
            D_f[:, k] = np.interp(log_eff_strain, log_strain_D[k], D_vector[k])

        if verbose:
            print(f'  max. strain diff = {strain_diff * 100:7.2f}%')

        if i_iter > 0 and strain_diff < tol:
            if verbose:
                print('---------- Convergence achieved ---------------')

            break

    # The response with the final modulus and damping
    H, accel_out, veloc, displ, strain = _freq_dependent_response(
        freq=freq,
        dt=dt,
        h=h,
        G_f=G_f,
        D_f=D_f,
        rho=rho,
        boundary=boundary,
        ACCEL_IN=ACCEL_IN,
    )

    # Stress: complex modulus (at each frequency) times strain
    modulus = (G_f * (1 + 2j * D_f))[:, :-1]
    strain_fft = scipy.fftpack.fft(strain, axis=0)
    stress_fft = np.zeros_like(strain_fft)
    stress_fft[:half_N] = strain_fft[:half_N] * modulus
    stress_fft[half_N:] = np.flipud(np.conj(stress_fft[1:half_N]))
    stress_fft[0] = np.real(stress_fft[0])
    stress = np.real(scipy.fftpack.ifft(stress_fft, axis=0))

    (
        freq_array,
        tf,
        accel_on_surface,
        out_a,
        out_v,
        out_d,
        out_gamma,
        out_tau,
        max_avd,
        max_gt,
    ) = _post_processing(
        flag=flag,
        freq=freq[1:],  # skip the DC component
        half_N=half_N - 1,
        H=H[1:],
        t=t,
        accel_out=accel_out,
        veloc=veloc,
        displ=displ,
        strain=strain,
        stress=stress,
        h=h,
    )

    return (
        new_profile,
        freq_array,
        tf,
        accel_on_surface,
        out_a,
        out_v,
        out_d,
        out_gamma,
        out_tau,
        max_avd,
        max_gt,
    )


def _ka02_strain_shape(freq: np.ndarray, strain_fas: np.ndarray) -> np.ndarray:
    """
    Smoothed shape of a strain Fourier spectrum (Kausel & Assimaki, 2002).

    Parameters
    ----------
    freq : np.ndarray
        Frequencies (single-sided, starting from 0).
    strain_fas : np.ndarray
        Fourier amplitude of the strain at ``freq``.

    Returns
    -------
    np.ndarray
        The shape at ``freq``, from 0 to 1: 1 at frequencies lower than the
        "average frequency" of the spectrum, and ``exp(-a * f) / f**b`` (with
        the frequency normalized by the average frequency) above it.
    """
    area = np.trapezoid(strain_fas, x=freq)
    if area <= 0:
        return np.ones_like(freq)

    freq_avg = np.trapezoid(freq * strain_fas, x=freq) / area
    below = freq < freq_avg
    strain_avg = np.trapezoid(strain_fas[below], x=freq[below]) / freq_avg
    f_norm = freq / freq_avg
    s_norm = strain_fas / strain_avg
    above = ~below & (s_norm > 0)
    if np.count_nonzero(above) < 2:  # noqa: PLR2004
        return np.ones_like(freq)

    A = np.column_stack((-f_norm[above], -np.log(f_norm[above])))
    a, b = np.linalg.lstsq(A, np.log(s_norm[above]), rcond=None)[0]
    with np.errstate(divide='ignore', over='ignore'):
        shape = np.exp(-a * f_norm) / np.maximum(
            np.finfo(float).eps, f_norm**b
        )

    return np.minimum(1.0, shape)


def _freq_dependent_response(
        *,
        freq: np.ndarray,
        dt: float,
        h: np.ndarray,
        G_f: np.ndarray,
        D_f: np.ndarray,
        rho: np.ndarray,
        boundary: Literal['elastic', 'rigid'],
        ACCEL_IN: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear response of every layer, with frequency-dependent properties.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies of all the FFT bins. Shape: ``(N, )`` (``N`` is odd).
    dt : float
        Recording time interval.
    h : np.ndarray
        Layer thickness. Shape: ``(n_layer, )``.
    G_f : np.ndarray
        Shear modulus of every layer at every single-sided frequency. Shape:
        ``(half_N, n_layer)``.
    D_f : np.ndarray
        Damping ratio (unit: 1), in the same shape as ``G_f``.
    rho : np.ndarray
        Mass density of every layer. Shape: ``(n_layer, )``.
    boundary : Literal['elastic', 'rigid']
        Boundary condition.
    ACCEL_IN : np.ndarray
        Fourier spectrum of the incident motion. Shape: ``(N, )``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The transfer functions (relative to the incident motion) of every
        layer, and the acceleration, velocity, displacement, and strain time
        histories of every layer.
    """
    N = len(freq)
    half_N = (N + 1) // 2
    n_layer = len(h)
    omega = 2 * np.pi * freq[:half_N]
    modulus = G_f * (1 + 2j * D_f)  # (half_N, n_layer)
    impedance = np.sqrt(modulus * rho)  # rho * vs_star
    alpha = impedance[:, :-1] / impedance[:, 1:]
    if boundary == 'rigid':
        alpha[:, -1] = 0  # no energy transmission past the boundary

    k_star = omega[:, None] / np.sqrt(modulus / rho)
    A = np.ones((half_N, n_layer), dtype=np.complex128)
    B = np.ones((half_N, n_layer), dtype=np.complex128)
    for k in range(n_layer - 1):
        e_pos = np.exp(1j * k_star[:, k] * h[k])
        e_neg = np.exp(-1j * k_star[:, k] * h[k])
        A[:, k + 1] = 0.5 * (
            A[:, k] * (1 + alpha[:, k]) * e_pos
            + B[:, k] * (1 - alpha[:, k]) * e_neg
        )
        B[:, k + 1] = 0.5 * (
            A[:, k] * (1 - alpha[:, k]) * e_pos
            + B[:, k] * (1 + alpha[:, k]) * e_neg
        )

    H_ss = (A + B) / A[:, -1:]
    H_ss[0] = np.real(H_ss[0])
    H = np.vstack((H_ss, np.conj(np.flipud(H_ss[1:]))))

    accel_out = np.real(scipy.fftpack.ifft(H * ACCEL_IN[:, None], axis=0))
    veloc = np.cumsum(accel_out, axis=0) * dt
    displ = np.cumsum(veloc, axis=0) * dt
    offset = np.outer(np.arange(1, N + 1) / (N + 1.0), displ[-1, :])
    displ -= offset
    strain = (displ[:, :-1] - displ[:, 1:]) / h[:-1]
    return H, accel_out, veloc, displ, strain


def _prepare_inputs(
        *, vs_profile: np.ndarray, input_motion: np.ndarray
) -> tuple[Any, ...]:
    """
    Prepare input variables from ``vs_profile`` and ``input_motion``.

    Parameters
    ----------
    vs_profile : np.ndarray
        Shear-wave velocity profile, as a 2D numpy array.
    input_motion : np.ndarray
        Input acceleration on rock outcrop (unit: m/s/s). It should have two
        columns (time and acceleration).

    Returns
    -------
    tuple[Any, ...]
        A tuple of 15 elements:

        - ``flag`` (Literal[0, 1]): 0 if the original input motion length is
          even; 1 if odd.
        - ``N`` (int): Length of frequency array, after odd-even adjustment.
        - ``freq`` (np.ndarray): Frequency array.
        - ``new_profile`` (np.ndarray): Re-discretized Vs profile.
        - ``h`` (np.ndarray): Layer thickness.
        - ``vs`` (np.ndarray): Shear-wave velocity of every layer.
        - ``D`` (np.ndarray): Damping ratio of every layer. (Unit: 1)
        - ``rho`` (np.ndarray): Mass density of every layer. (Unit: kg/m^3)
        - ``mat_nr`` (np.ndarray): Material index of every layer. Every index
          maps a layer to its corresponding material (i.e., G/Gmax and
          damping).
        - ``n_layer`` (int): Number of soil layers, including the bedrock at
          the bottom.
        - ``Gmax`` (np.ndarray): Initial shear modulus of every layer. (Unit:
          Pa)
        - ``G`` (np.ndarray): Initial shear modulus of every layer (identical
          to ``Gmax``). (Unit: Pa)
        - ``t`` (np.ndarray): Time array.
        - ``dt`` (float): Recording time interval
        - ``ACCEL_IN`` (np.ndarray): An array of Fourier spectra (complex
          values) of the input acceleration.
    """
    # ---------- Input motion -------------------------------------
    # On 05/26/2019, confirmed with MATLAB SeismoSoil that this is correct:
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

    ACCEL_IN = scipy.fftpack.fft(accel_in)
    N = len(ACCEL_IN)
    assert n == N
    assert N % 2 == 1

    freq = np.arange(1, N + 1, 1) / (N * dt)  # frequency

    # -------- Soil profile -----------------------------
    new_profile = sr.stratify(vs_profile)
    h = new_profile[:, 0]
    vs = new_profile[:, 1]
    D = new_profile[:, 2]
    rho = new_profile[:, 3]
    mat_nr = new_profile[:, 4].astype(int) - 1  # from 1-indexing to 0-indexing

    n_layer = len(h)  # includes the rock layer
    Gmax = rho * vs**2.0

    # initial value of G; Gmax cannot change, so needs a hard copy
    G = Gmax.copy()

    return (
        flag,
        N,
        freq,
        new_profile,
        h,
        vs,
        D,
        rho,
        mat_nr,
        n_layer,
        Gmax,
        G,
        t,
        dt,
        ACCEL_IN,
    )


def _lin_resp_every_layer(
        *,
        dt: float,
        freq: np.ndarray,
        N: int,
        n_layer: int,
        h: np.ndarray,
        G: np.ndarray,
        D: np.ndarray,
        rho: np.ndarray,
        boundary: Literal['elastic', 'rigid'],
        ACCEL_IN: np.ndarray,
        R_gamma: float = 0.65,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Propagate input motion to get the linear site response of every layer.

    Parameters
    ----------
    dt : float
        Recording time interval.
    freq : np.ndarray
        Frequency array.
    N : int
        Length of frequency array, after odd-even adjustment.
    n_layer : int
        Number of soil layers, including the bedrock at the bottom.
    h : np.ndarray
        Layer thickness.
    G : np.ndarray
        Shear modulus of every layer. (Unit: Pa)
    D : np.ndarray
        Damping ratio of every layer. (Unit: 1)
    rho : np.ndarray
        Mass density of every layer. (Unit: kg/m^3)
    boundary : Literal['elastic', 'rigid']
        Boundary condition.
    ACCEL_IN : np.ndarray
        An array of Fourier spectra (complex values) of the input acceleration.
        It should be the "rock outrcop" motion if ``boundary`` is set to
        ``"elastic"``, and it should be the recorded motion at the bottom of
        the Vs profile (i.e., the "borehole" motion) if ``boundary`` is set to
        ``"rigid"``.
    R_gamma : float, default=0.65
        A dimensionless ratio to determine the "effective shear strain" from
        the maximum shear strain in the strain time history. Do not change the
        default value unless you really know what you are doing.

    Returns
    -------
    H : np.ndarray
        A 2D numpy array of shape ``(N, n_layer)``. Each column of ``H`` is the
        transfer function between the corresponding layer to the bottom layer.
    accel_out : np.ndarray
        A 2D numpy array of shape ``(N, n_layer)``. Each column of
        ``accel_out`` is the acceleration time history of the corresponding
        layer.
    veloc : np.ndarray
        Velocity time history of every layer. Shape: ``(N, n_layer)``.
    displ : np.ndarray
        Displacement time history of every layer. Shape: ``(N, n_layer)``.
    strain : np.ndarray
        Strain time history of every layer. Shape: ``(N, n_layer)``.
    eff_strain : np.ndarray
        The "effective strain level" of every layer. Shape: ``(n_layer, )``.
    """
    # ----- 1: Linear transfer function ---------------------------
    # (1) Make a denser frequency array to eliminate aliasing
    df = freq[1] - freq[0]
    max_f = np.max(freq)
    freq_oversample_factor = 15
    df_ = df / freq_oversample_factor
    N_original = N
    N = int(N * freq_oversample_factor)
    freq = np.linspace(df_, max_f, num=N)
    omega = 2 * np.pi * freq  # angular frequency
    half_N = int(N / 2 + 0.5)

    # (2) Complex impedance ratio between each layer
    alpha = np.zeros(n_layer - 1, dtype=np.complex128)
    for j in range(n_layer - 1):  # layer by layer
        alpha[j] = (rho[j] * np.sqrt(G[j] * (1 + 2 * 1j * D[j]) / rho[j])) / (
            rho[j + 1]
            * np.sqrt(G[j + 1] * (1 + 2 * 1j * D[j + 1]) / rho[j + 1])
        )

    if boundary == 'rigid':
        alpha[-1] = 0  # disallow energy transmission past the boundary

    # (3) Complex shear-wave velocities of each layer (Kramer's book, page 260)
    vs_star = np.sqrt(G * (1 + 2 * 1j * D) / rho)  # shape: (n_layer, )
    assert vs_star.shape == (n_layer,)

    # (4) Complex wave number (Kramer's book, page 260)
    vs_star_recip = (1.0 / vs_star).reshape((1, n_layer))  # (1, n_layer)

    # shape of k_star: (half_N, n_layer)
    k_star = omega[:half_N].reshape(half_N, 1) * vs_star_recip
    assert k_star.shape == (half_N, n_layer)

    # (5) Compute A and B (Kramer's book, page 269)
    A = np.zeros((half_N, n_layer), dtype=np.complex128)
    B = np.zeros((half_N, n_layer), dtype=np.complex128)
    A[:, 0] = 1
    B[:, 0] = 1
    for k in range(n_layer - 1):  # layer by layer
        # left half:
        A[:, k + 1] = 0.5 * A[:, k] * (1 + alpha[k]) * np.exp(
            1j * k_star[:, k] * h[k]
        ) + 0.5 * B[:, k] * (1 - alpha[k]) * np.exp(-1j * k_star[:, k] * h[k])

        # left half:
        B[:, k + 1] = 0.5 * A[:, k] * (1 - alpha[k]) * np.exp(
            1j * k_star[:, k] * h[k]
        ) + 0.5 * B[:, k] * (1 + alpha[k]) * np.exp(-1j * k_star[:, k] * h[k])

    # (6) Compute linear transfer function
    # single-sided transfer function:
    H_ss = np.zeros((half_N, n_layer), dtype=np.complex128)

    # the other half:
    H_append = np.zeros((half_N - 1, n_layer), dtype=np.complex128)

    for k in range(n_layer):
        H_ss[:, k] = (A[:, k] + B[:, k]) / A[:, -1]
        H_ss[0, k] = np.real(H_ss[0, k])  # see Note (1) below
        H_append[:, k] = np.conj(np.flipud(H_ss[1:, k]))

    H = np.vstack((H_ss, H_append))

    # down-sample back to original resolution:
    H = H[::freq_oversample_factor, :]
    freq = freq[::freq_oversample_factor]
    N = N_original

    # Notes:
    #  (1) The value of transfer function when freq = 0 should be either 1
    #      or 2 (a real number). Because of discretization errors, tf_ss(1)
    #     is not a real number, but very close.
    #
    #  (2) Two examples:
    #     [a] fft([1, 2, 3, 4, 5, 6, 7, 8]) =
    #           36
    #           -4 + 9.6569i  ----------
    #           -4 + 4.0000i  ------   |
    #           -4 + 1.6569i  ---  |   |
    #           -4              |  |   |
    #           -4 - 1.6569i  ---  |   |
    #           -4 - 4.0000i  ------   |
    #           -4 - 9.6569i  ---------|
    #
    #     [b] fft([1, 2, 3, 4, 5, 6, 7]') =
    #           28.0
    #           -3.5 + 7.2678i  ----------|
    #           -3.5 + 2.7912i  ------|   |
    #           -3.5 + 0.7989i  ---|  |   |
    #           -3.5 - 0.7989i  ---|  |   |
    #           -3.5 - 2.7912i  ------|   |
    #           -3.5 - 7.2678i  ----------|

    # ----- 2: Response motion of each layer ----------------------------------
    ACCEL_OUT = H * ACCEL_IN.reshape(-1, 1)  # amplify accel. of each layer
    accel_out = np.real(scipy.fftpack.ifft(ACCEL_OUT, axis=0))  # column-wise
    veloc = np.cumsum(accel_out, axis=0) * dt
    displ = np.cumsum(veloc, axis=0) * dt

    # simple baseline correction of displacement time history
    offset = np.matmul(
        np.arange(1, N + 1, 1).reshape(-1, 1) / (N + 1.0),
        displ[-1, :].reshape(1, -1),
    )
    displ -= offset

    # ----- 3: Strain time history and effective strain -----------------------
    strain = np.zeros((N, n_layer - 1))
    eff_strain = np.zeros(n_layer - 1)  # Kramer's book, pages 271-272
    for k in range(n_layer - 1):  # layer by layer
        strain[:, k] = (displ[:, k] - displ[:, k + 1]) / h[k]  # unit: 1
        eff_strain[k] = R_gamma * np.max(np.abs(strain[:, k]))  # unit: 1

    return H, accel_out, veloc, displ, strain, eff_strain


def _calc_stress(
        *,
        G: np.ndarray,
        D: np.ndarray,
        N: int,
        n_layer: int,
        strain: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Calculate stress time history from strain time history.

    Parameters
    ----------
    G : np.ndarray
        Shear modulus of every layer. Unit: Pa. Shape: ``(n_layer, )``.
    D : np.ndarray
        Damping ratio of every layer. Unit: 1. Shape: ``(n_layer, )``.
    N : int
        Length of frequency array, after odd-even adjustment.
    n_layer : int
        Number of soil layers, including the bedrock at the bottom.
    strain : np.ndarray
        Strain time history of every layer. Shape: ``(N, n_layer)``.

    Returns
    -------
    stress : np.ndarray
        Stress time history of every layer. Shape: ``(N, n_layer)``.
    half_N : int
        The length of the "single-sided" frequency spectrum.
    """
    stress_fft = np.zeros((N, n_layer - 1), dtype=np.complex128)

    strain_fft = scipy.fftpack.fft(strain, axis=0)
    modulus = (G * (1 + 2 * 1j * D))[:-1]

    half_N = int(N / 2 + 0.5)
    modulus_repmat = np.tile(modulus, (half_N, 1))
    stress_fft[:half_N, :] = strain_fft[:half_N, :] * modulus_repmat
    stress_fft[half_N:, :] = np.flipud(np.conj(stress_fft[1:half_N, :]))
    stress_fft[0, :] = np.real(stress_fft[0, :])
    stress = np.real(scipy.fftpack.ifft(stress_fft, axis=0))

    return stress, half_N


def _post_processing(
        *,
        flag: Literal[0, 1],
        freq: np.ndarray,
        half_N: int,
        H: np.ndarray,
        t: np.ndarray,
        accel_out: np.ndarray,
        veloc: np.ndarray,
        displ: np.ndarray,
        strain: np.ndarray,
        stress: np.ndarray,
        h: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Post-process simulation results.

    Parameters
    ----------
    flag : Literal[0, 1]
        0 if the original input motion length is even; 1 if odd.
    freq : np.ndarray
        Frequency array.
    half_N : int
        The length of the "single-sided" frequency spectrum.
    H : np.ndarray
        A 2D numpy array of shape ``(N, n_layer)``. Each column of ``H`` is the
        transfer function between the corresponding layer to the bottom layer.
    t : np.ndarray
        Time array.
    accel_out : np.ndarray
        A 2D numpy array of shape ``(N, n_layer)``. Each column of
        ``accel_out`` is the acceleration time history of the corresponding
        layer.
    veloc : np.ndarray
        Velocity time history of every layer. Shape: ``(N, n_layer)``.
    displ : np.ndarray
        Displacement time history of every layer. Shape: ``(N, n_layer)``.
    strain : np.ndarray
        Strain time history of every layer. Shape: ``(N, n_layer)``.
    stress : np.ndarray
        Stress time history of every layer. Shape: ``(N, n_layer)``.
    h : np.ndarray
        Layer thickness. Shape: ``(n_layer, )``.

    Returns
    -------
    tuple[np.ndarray, ...]
        A tuple of 10 numpy arrays:

        - ``freq_array`` (np.ndarray): "Single-sided" frequency. Shape:
          ``(half_N, )`` or ``(half_N - 1, )``.
        - ``tf`` (np.ndarray): Transfer function (complex-valued). Same shape
          as ``freq_array``.
        - ``accel_on_surface`` (np.ndarray): Simulated acceleration on the
          ground surface (two-columed).
        - ``out_a`` (np.ndarray): Simulated acceleration time history of every
          layer. Shape: ``(num_time_step, n_layer)``.
        - ``out_v`` (np.ndarray): Simulated velocity time history of every
          layer. Same shape as ``out_a``.
        - ``out_d`` (np.ndarray): Simulated displacement time history of every
          layer. Same shape as ``out_a``.
        - ``out_gamma`` (np.ndarray): Simulated shear strain time history of
          every layer. Shape: ``(num_time_step, n_layer - 1)``.
        - ``out_tau`` (np.ndarray): Simulated shear stress time history of
          every layer. Same shape as ``out_gamma``.
        - ``max_avd`` (np.ndarray): Maximum acceleration, velocity, and
          displacement during the shaking process, of each layer. Shape:
          ``(n_layer, )``.
        - ``max_gt`` (np.ndarray): Maximum shear strain and shear stress during
          the shaking process, of each layer. Shape: ``(n_layer - 1, )``.
    """
    if flag == 0:  # originally the input motion length is even
        freq_array = freq[: half_N - 1]
        tf = H[: half_N - 1, 0] / 2.0
        accel_out = accel_out[:-1, :]
        veloc = veloc[:-1, :]
        displ = displ[:-1, :]
        strain = strain[:-1, :]
        stress = stress[:-1, :]
        accel_on_surface = np.column_stack((t[:-1], accel_out[:, 0]))
    else:
        freq_array = freq[:half_N]
        tf = H[:half_N, 0] / 2.0
        accel_on_surface = np.column_stack((t, accel_out[:, 0]))

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

    layer_boundary_depth = sr.thk2dep(h, midpoint=False)
    layer_midpoint_depth = sr.thk2dep(h, midpoint=True)

    max_avd = np.column_stack((layer_boundary_depth, max_a, max_v, max_d))
    max_gt = np.column_stack((layer_midpoint_depth, max_gamma, max_tau))

    return (
        freq_array,
        tf,
        accel_on_surface,
        out_a,
        out_v,
        out_d,
        out_gamma,
        out_tau,
        max_avd,
        max_gt,
    )

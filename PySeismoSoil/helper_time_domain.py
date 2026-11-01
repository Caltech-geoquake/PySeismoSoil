"""
Time-domain 1D site response kernels.

This module is a Python port of the time-domain solvers of MATLAB SeismoSoil
(``nlHH.m``, ``nlH4.m``, ``nlH2.m``, and ``nlEPP.m``, plus a linear
viscoelastic variant that corresponds to ``TDLinear``). All of them share the
same explicit finite-difference scheme (staggered in space, with ``n_dt``
sub-steps per time step of the input motion) and the same small-strain damping
model (the memory variables of Liu & Archuleta, 2006). They only differ in the
constitutive model that maps shear strain to shear stress:

+-----------+-----------------------------------------------------------------+
| Model     | Constitutive model                                              |
+===========+=================================================================+
| linear    | Linear elastic: ``tau = Gmax * gamma``                          |
+-----------+-----------------------------------------------------------------+
| EPP       | Elastic-perfectly-plastic, as an Iwan model of parallel springs |
+-----------+-----------------------------------------------------------------+
| H2        | MKZ backbone curve and Masing rules, as an Iwan model           |
+-----------+-----------------------------------------------------------------+
| H4        | MKZ backbone curve and non-Masing rules (separate curves for    |
|           | G/Gmax and damping)                                             |
+-----------+-----------------------------------------------------------------+
| HH        | Hybrid hyperbolic (HH) backbone curve and non-Masing rules      |
+-----------+-----------------------------------------------------------------+
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numba import njit

# Three coefficients (tau, alpha, beta) for modeling Q, from Table 1 of Liu
# and Archuleta (2006) BSSA
TABK = np.array(
    [
        [1.72333e-03, 1.66958e-02, 8.98758e-02],
        [1.80701e-03, 3.81644e-02, 6.84635e-02],
        [5.38887e-03, 9.84666e-03, 9.67052e-02],
        [1.99322e-02, -1.36803e-02, 1.20172e-01],
        [8.49833e-02, -2.85125e-02, 1.30728e-01],
        [4.09335e-01, -5.37309e-02, 1.38746e-01],
        [2.05951e00, -6.65035e-02, 1.40705e-01],
        [1.32629e01, -1.33696e-01, 2.14647e-01],
    ],
)

# The same default values as MATLAB SeismoSoil
DEFAULT_F_MAX_HZ = 30.0  # maximum frequency modeled
DEFAULT_PPW = 10  # points per wavelength
DEFAULT_N_DT = 30  # number of sub-steps in one time step
DEFAULT_N_SPR = 120  # number of Iwan springs

# Initial capacity of the reversal-point memory of the non-Masing models (H4
# and HH), per sub-layer (the same as MATLAB SeismoSoil). It grows as needed.
INITIAL_REVERSAL_CAPACITY = 5000

# Number of rows in the parameter files
NUM_MKZ_PARAMS = 4  # "H2_n", "H4_G", or "H4_x" files
NUM_HH_PARAMS = 9  # "HH_G" or "HH_x" files

# Integer codes of the models and the boundary conditions (the same boundary
# codes as the Fortran kernels of MATLAB SeismoSoil)
_MODEL_LINEAR = 0
_MODEL_EPP = 1
_MODEL_H2 = 2
_MODEL_H4 = 3
_MODEL_HH = 4
MODEL_CODES = {
    'linear': _MODEL_LINEAR,
    'EPP': _MODEL_EPP,
    'H2': _MODEL_H2,
    'H4': _MODEL_H4,
    'HH': _MODEL_HH,
}
_BOUNDARY_ELASTIC = 1
_BOUNDARY_RIGID = 2
_BOUNDARY_BOREHOLE = 3
BOUNDARY_CODES = {
    'elastic': _BOUNDARY_ELASTIC,
    'rigid': _BOUNDARY_RIGID,
    'borehole': _BOUNDARY_BOREHOLE,
}

# The exponent beyond which 10**x overflows (or underflows) in the transition
# function of the HH model
_MAX_SAFE_LOG10_EXPONENT = 305.0

# Strains (and strain increments) smaller than this are treated as zero
_TINY_STRAIN = 1e-30

# A distance that the reversal points never reach (see MATLAB ``hyster()``)
_FAR_AWAY = 1.0e30

_NUM_PROFILE_COLUMNS = 5
_NUM_MOTION_COLUMNS = 2
_MIN_NUM_ROWS = 2

# Index of each scalar in the ``scalars`` state array of ``_march()``
_IDX_V_IN = 0
_IDX_A_IN = 1
_IDX_N_OUT = 2

Model = Literal['linear', 'EPP', 'H2', 'H4', 'HH']
Boundary = Literal['elastic', 'rigid', 'borehole']
ProgressCallback = Callable[[float], None]


def time_domain_site_response(  # noqa: PLR0915
        vs_profile: np.ndarray,
        incident_motion: np.ndarray,
        *,
        model: Model,
        boundary: Boundary = 'elastic',
        G_param: np.ndarray | None = None,
        xi_param: np.ndarray | None = None,
        tau_max: np.ndarray | None = None,
        f_max: float = DEFAULT_F_MAX_HZ,
        ppw: int = DEFAULT_PPW,
        n_dt: int = DEFAULT_N_DT,
        N_spr: int = DEFAULT_N_SPR,
        progress_callback: ProgressCallback | None = None,
        n_progress_updates: int = 100,
) -> tuple[np.ndarray, ...]:
    """
    Run a time-domain 1D site response simulation.

    Parameters
    ----------
    vs_profile : np.ndarray
        Vs profile with 5 columns: thickness (m), Vs (m/s), damping ratio
        (unit: 1), mass density (kg/m^3), and material number (starting from
        1). The last row is the bedrock half space. It is usually already
        re-discretized with ``helper_site_response.stratify()``.
    incident_motion : np.ndarray
        Input acceleration (unit: m/s/s) with two columns (time and
        acceleration), and a constant time step. It is the incident (upgoing)
        motion when ``boundary`` is ``"elastic"`` or ``"rigid"``, and the total
        motion at the bottom of the profile when ``boundary`` is
        ``"borehole"``.
    model : Model
        The constitutive model: ``"linear"``, ``"EPP"``, ``"H2"``, ``"H4"``, or
        ``"HH"``.
    boundary : Boundary, default='elastic'
        Boundary condition at the bottom of the profile: ``"elastic"`` (the
        bedrock absorbs the downgoing waves), ``"rigid"`` (input motion is the
        incident motion, and all downgoing waves are reflected back), or
        ``"borehole"`` (input motion is the total motion recorded at the bottom
        of the profile).
    G_param : np.ndarray | None, default=None
        The parameters of the G/Gmax curves, as in the parameter files: one
        column per material, and 4 rows for the MKZ model (``"H2"`` or
        ``"H4"``) or 9 rows for the HH model (``"HH"``). Required by ``"H2"``,
        ``"H4"``, and ``"HH"``.
    xi_param : np.ndarray | None, default=None
        The parameters of the damping curves, in the same format as
        ``G_param``. Required by ``"H4"`` and ``"HH"``.
    tau_max : np.ndarray | None, default=None
        Shear strength (unit: Pa) of each material. Required by ``"EPP"``.
    f_max : float, default=DEFAULT_F_MAX_HZ
        Maximum frequency (Hz) that the spatial discretization resolves.
    ppw : int, default=DEFAULT_PPW
        Number of points per wavelength (of ``f_max``).
    n_dt : int, default=DEFAULT_N_DT
        Number of sub-steps in each time step of the input motion.
    N_spr : int, default=DEFAULT_N_SPR
        Number of Iwan springs (``"EPP"`` and ``"H2"`` models).
    progress_callback : ProgressCallback | None, default=None
        A function called (with the fraction of the calculation finished, from
        0 to 1) about ``n_progress_updates`` times. It can raise an exception
        to abort the simulation.
    n_progress_updates : int, default=100
        How many times to call ``progress_callback``.

    Returns
    -------
    tuple[np.ndarray, ...]
        A tuple of 7 numpy arrays:

        - ``layer_depth``: Depth of the mid-point of every soil layer.
          Shape: ``(n_layer, )``.
        - ``node_depth``: Depth of the top of every layer (including the
          bedrock). Shape: ``(n_layer + 1, )``.
        - ``out_a``: Acceleration time history (m/s/s) at the top of every
          layer. Shape: ``(n_time, n_layer + 1)``.
        - ``out_v``: Velocity time history (m/s), relative to the bottom.
          Same shape as ``out_a``.
        - ``out_d``: Displacement time history (m), relative to the bottom.
          Same shape as ``out_a``.
        - ``out_gamma``: Shear strain time history (unit: 1) at the
          mid-point of every soil layer. Shape: ``(n_time, n_layer)``.
        - ``out_tau``: Shear stress time history (Pa) at the mid-point of
          every soil layer. Same shape as ``out_gamma``.

    Raises
    ------
    ValueError
        When the inputs are invalid
    """
    if model not in MODEL_CODES:
        raise ValueError(f'`model` must be one of {list(MODEL_CODES)}.')

    if boundary not in BOUNDARY_CODES:
        raise ValueError(f'`boundary` must be one of {list(BOUNDARY_CODES)}.')

    profile = np.asarray(vs_profile, dtype=float)
    motion = np.asarray(incident_motion, dtype=float)
    _check_inputs(profile, motion, model, G_param, xi_param, tau_max)

    # ------- Input motion: linearly interpolate to the sub-steps ------------
    t = motion[:, 0]
    a = motion[:, 1]
    nt_out = len(t)
    dt = (t[1] - t[0]) / n_dt
    n_new = (nt_out - 1) * n_dt + 1
    a_in = np.zeros(n_new)
    xi_frac = np.arange(1, n_dt + 1) / n_dt
    for i in range(1, nt_out):
        a_in[(i - 1) * n_dt : i * n_dt] = (1.0 - xi_frac) * a[i - 1] + (
            xi_frac * a[i]
        )

    # ------- Soil profile ---------------------------------------------------
    n_layer = profile.shape[0] - 1
    h = profile[:n_layer, 0]
    vs = profile[:, 1]
    damping = profile[:n_layer, 2]
    rho_all = profile[:, 3]
    mat = profile[:n_layer, 4].astype(np.int64) - 1  # to 0-indexing
    vs_r = vs[n_layer]
    rho_r = rho_all[n_layer]
    G = rho_all[:n_layer] * vs[:n_layer] ** 2

    # ------- Memory variables for the small-strain damping ------------------
    t_k = TABK[:, 0]
    Q = 1.0 / 2.0 / damping
    chi = (3.071 + 1.433 * Q ** (-1.158) * np.log(Q / 5.0)) / (1.0 + 0.415 * Q)
    w_k = chi[:, None] * (chi[:, None] * TABK[None, :, 1] + TABK[None, :, 2])

    # ------- Sub-layers ------------------------------------------------------
    node_depth = np.concatenate(([0.0], np.cumsum(h)))
    layer_depth = (node_depth[1:] + node_depth[:-1]) / 2.0
    n_sublay = np.floor(h / (vs[:n_layer] / f_max / ppw)).astype(np.int64) + 1
    node_index = np.concatenate(([0], np.cumsum(n_sublay)))  # 0-indexed
    # (Same as `floor((node_index(i) + node_index(i+1)) / 2)` in MATLAB,
    # which is 1-indexed)
    layer_index = (node_index[:-1] + node_index[1:] + 2) // 2 - 1
    owner = np.repeat(np.arange(n_layer), n_sublay)  # layer of each sub-layer
    dz = (h / n_sublay)[owner]
    rho = rho_all[:n_layer][owner]
    w_k_sub = np.ascontiguousarray(w_k[owner, :])
    lay_max = len(owner)

    # ------- Constitutive model parameters for each sub-layer ---------------
    gmax_lin = G[owner].copy()
    n_spr_eff = N_spr if model in {'EPP', 'H2'} else 1
    ak = np.zeros((lay_max, n_spr_eff))
    fy = np.zeros((lay_max, n_spr_eff))
    gmax2 = np.zeros((2, lay_max))
    n_p = NUM_HH_PARAMS if model == 'HH' else NUM_MKZ_PARAMS
    p_g = np.zeros((lay_max, n_p))
    p_x = np.zeros((lay_max, n_p))
    if model == 'EPP':
        strength = np.asarray(tau_max, dtype=float).ravel()
        for j in range(lay_max):
            ak[j], fy[j] = _make_springs_EPP(
                gmax_lin[j], strength[mat[owner[j]]], N_spr
            )
    elif model == 'H2':
        for j in range(lay_max):
            ak[j], fy[j] = _make_springs_MKZ(
                gmax_lin[j], G_param[:, mat[owner[j]]], N_spr
            )
    elif model in {'H4', 'HH'}:
        p_g[:] = np.asarray(G_param, dtype=float)[:, mat[owner]].T
        p_x[:] = np.asarray(xi_param, dtype=float)[:, mat[owner]].T
        if model == 'H4':
            gmax2[0] = gmax_lin
            gmax2[1] = gmax_lin
        else:  # HH: the "Gmax" of the HH parameters (6th row) is used
            gmax2[0] = p_g[:, 5]
            gmax2[1] = p_x[:, 5]

    # ------- Initialization -------------------------------------------------
    N = lay_max + 1  # number of nodes
    state_nodes = np.zeros((6, N))  # d_1, v_est_1, v_est_2, d_2, v_nt, a_nt
    state_layers = np.zeros((5, lay_max))  # tau, gamma_0/1, d_gamma_0/1
    memory = np.zeros((lay_max, len(t_k)))
    ss = np.zeros((lay_max, n_spr_eff))
    rev_cap = INITIAL_REVERSAL_CAPACITY if model in {'H4', 'HH'} else 1
    bb = np.zeros(lay_max, dtype=np.int64)
    xtu = np.zeros((lay_max, rev_cap))
    ytu = np.zeros((lay_max, rev_cap))
    xtu[:, 0] = _FAR_AWAY
    ytu[:, 0] = _FAR_AWAY
    scalars = np.zeros(3)

    out_a = np.zeros((nt_out, n_layer + 1))
    out_v = np.zeros((nt_out, n_layer + 1))
    out_d = np.zeros((nt_out, n_layer + 1))
    out_gamma = np.zeros((nt_out, n_layer))
    out_tau = np.zeros((nt_out, n_layer))

    # ------- Time marching (in chunks, to be able to report progress) ------
    n_chunks = max(1, min(int(n_progress_updates), n_new))
    bounds = np.linspace(0, n_new, n_chunks + 1).astype(np.int64)
    for k in range(n_chunks):
        pos = bounds[k]
        while pos < bounds[k + 1]:
            pos = _march(
                pos,
                bounds[k + 1],
                MODEL_CODES[model],
                BOUNDARY_CODES[boundary],
                dt,
                n_dt,
                a_in,
                dz,
                rho,
                vs_r,
                rho_r,
                t_k,
                w_k_sub,
                node_index,
                layer_index,
                state_nodes,
                state_layers,
                memory,
                gmax_lin,
                ak,
                fy,
                ss,
                gmax2,
                p_g,
                p_x,
                bb,
                xtu,
                ytu,
                scalars,
                out_a,
                out_v,
                out_d,
                out_gamma,
                out_tau,
            )
            if pos < bounds[k + 1]:  # reversal-point memory is full
                xtu, ytu = _grow_reversal_memory(xtu, ytu)

        if progress_callback is not None:
            progress_callback(bounds[k + 1] / n_new)

    return layer_depth, node_depth, out_a, out_v, out_d, out_gamma, out_tau


def _grow_reversal_memory(
        xtu: np.ndarray, ytu: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Double the capacity of the reversal-point memory.

    Parameters
    ----------
    xtu : np.ndarray
        Strains at the reversal points of every sub-layer.
    ytu : np.ndarray
        Stresses at the reversal points of every sub-layer.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The same arrays, with twice as many columns.
    """
    return (
        np.concatenate((xtu, np.zeros_like(xtu)), axis=1),
        np.concatenate((ytu, np.zeros_like(ytu)), axis=1),
    )


def _check_inputs(
        profile: np.ndarray,
        motion: np.ndarray,
        model: Model,
        G_param: np.ndarray | None,
        xi_param: np.ndarray | None,
        tau_max: np.ndarray | None,
) -> None:
    """
    Check the inputs of ``time_domain_site_response()``.

    Parameters
    ----------
    profile : np.ndarray
        The Vs profile.
    motion : np.ndarray
        The input motion.
    model : Model
        The constitutive model.
    G_param : np.ndarray | None
        Parameters of the G/Gmax curves.
    xi_param : np.ndarray | None
        Parameters of the damping curves.
    tau_max : np.ndarray | None
        Shear strength of each material.

    Raises
    ------
    ValueError
        When the inputs are invalid
    """
    if profile.ndim != 2 or profile.shape[1] != _NUM_PROFILE_COLUMNS:  # noqa: PLR2004
        raise ValueError('`vs_profile` must be a 2D array with 5 columns.')

    if profile.shape[0] < _MIN_NUM_ROWS:
        raise ValueError(
            '`vs_profile` must have at least one soil layer and the bedrock.'
        )

    if np.any(profile[:-1, 0] <= 0) or np.any(profile[:, 1:4] <= 0):
        raise ValueError(
            'Thickness (except the bedrock), Vs, damping, and density in'
            ' `vs_profile` must be positive.',
        )

    if motion.ndim != 2 or motion.shape[1] != _NUM_MOTION_COLUMNS:  # noqa: PLR2004
        raise ValueError('`incident_motion` must have two columns.')

    if motion.shape[0] < _MIN_NUM_ROWS:
        raise ValueError('`incident_motion` must have at least 2 time steps.')

    _check_params(
        int(np.max(profile[:-1, 4])), model, G_param, xi_param, tau_max
    )


def _check_params(
        n_mat: int,
        model: Model,
        G_param: np.ndarray | None,
        xi_param: np.ndarray | None,
        tau_max: np.ndarray | None,
) -> None:
    """
    Check the model parameters of ``time_domain_site_response()``.

    Parameters
    ----------
    n_mat : int
        Number of materials in the Vs profile.
    model : Model
        The constitutive model.
    G_param : np.ndarray | None
        Parameters of the G/Gmax curves.
    xi_param : np.ndarray | None
        Parameters of the damping curves.
    tau_max : np.ndarray | None
        Shear strength of each material.

    Raises
    ------
    ValueError
        When the parameters are invalid
    """
    if model in {'H2', 'H4', 'HH'}:
        n_rows = NUM_HH_PARAMS if model == 'HH' else NUM_MKZ_PARAMS
        params = [('G_param', G_param)]
        if model != 'H2':
            params.append(('xi_param', xi_param))

        for name, param in params:
            if param is None:
                raise ValueError(f'`{name}` is required by the {model} model.')

            param_ = np.asarray(param)
            if param_.ndim != 2 or param_.shape[0] != n_rows:  # noqa: PLR2004
                raise ValueError(
                    f'`{name}` must be a 2D array with {n_rows} rows.'
                )

            if param_.shape[1] < n_mat:
                raise ValueError(
                    f'`{name}` has fewer columns than the number of'
                    ' materials in `vs_profile`.',
                )

    if model == 'EPP':
        if tau_max is None:
            raise ValueError('`tau_max` is required by the EPP model.')

        if np.asarray(tau_max).size < n_mat:
            raise ValueError(
                '`tau_max` has fewer values than the number of materials in'
                ' `vs_profile`.',
            )


def _spring_strains(N_spr: int) -> np.ndarray:
    """
    Strain levels where the Iwan springs yield.

    Parameters
    ----------
    N_spr : int
        Number of springs.

    Returns
    -------
    np.ndarray
        ``N_spr`` strains, from 1e-6 with a step of 0.05 in log10 scale.
    """
    return 1e-6 * 10.0 ** (0.05 * np.arange(N_spr))


def _springs_from_backbone(
        gamma: np.ndarray, tau: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stiffness and yield stress of Iwan springs, from a backbone curve.

    Parameters
    ----------
    gamma : np.ndarray
        Strains where the springs yield.
    tau : np.ndarray
        Backbone curve (stress) at ``gamma``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The stiffness ``ak`` and the yield stress ``fy`` of each spring.
    """
    G_tan = np.empty_like(tau)
    G_tan[0] = tau[0] / gamma[0]
    G_tan[1:] = np.diff(tau) / np.diff(gamma)
    ak = np.empty_like(tau)
    ak[:-1] = G_tan[:-1] - G_tan[1:]
    ak[-1] = G_tan[-1]
    return ak, ak * gamma


def _make_springs_MKZ(
        Gmax: float, param: np.ndarray, N_spr: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iwan springs for the MKZ model (``make_spring_HY()`` in MATLAB).

    Parameters
    ----------
    Gmax : float
        Initial shear modulus.
    param : np.ndarray
        The 4 MKZ parameters (gamma_ref, b, s, beta).
    N_spr : int
        Number of springs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The stiffness ``ak`` and the yield stress ``fy`` of each spring.
    """
    a, b, s, beta = param
    gamma = _spring_strains(N_spr)
    ratio = 1.0 / (1.0 + beta * (gamma / (a + gamma * b)) ** s)
    return _springs_from_backbone(gamma, Gmax * ratio * gamma)


def _make_springs_EPP(
        Gmax: float, tau_max: float, N_spr: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iwan springs for the elastic-perfectly-plastic model.

    Parameters
    ----------
    Gmax : float
        Initial shear modulus.
    tau_max : float
        Shear strength.
    N_spr : int
        Number of springs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The stiffness ``ak`` and the yield stress ``fy`` of each spring.
    """
    gamma = _spring_strains(N_spr)
    elastic = Gmax * gamma <= tau_max
    G_tan = np.where(elastic, Gmax, 0.0)
    ak = np.empty(N_spr)
    ak[:-1] = G_tan[:-1] - G_tan[1:]
    ak[-1] = G_tan[-1]
    return ak, ak * gamma


@njit(cache=True)
def _backbone(
        xx: float,
        which: int,
        m: int,
        model: int,
        p_g: np.ndarray,
        p_x: np.ndarray,
        gmax2: np.ndarray,
) -> float:  # pragma: no cover (compiled by numba)
    """
    Backbone curve (stress) of the H4 or HH model.

    Parameters
    ----------
    xx : float
        Strain.
    which : int
        0: the G/Gmax curve; 1: the curve for unloading and reloading (i.e.,
        the damping).
    m : int
        Index of the sub-layer.
    model : int
        Model code (3: H4; 4: HH).
    p_g : np.ndarray
        Parameters of the G/Gmax curve of every sub-layer.
    p_x : np.ndarray
        Parameters of the damping curve of every sub-layer.
    gmax2 : np.ndarray
        The Gmax values of both curves of every sub-layer.

    Returns
    -------
    float
        Stress.
    """
    p = p_g[m] if which == 0 else p_x[m]
    gmax = gmax2[which, m]
    ax = abs(xx)
    if model == _MODEL_H4:
        return gmax * xx / (1.0 + p[3] * (ax / p[0]) ** p[2])

    # HH model: (gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d)
    if ax < _TINY_STRAIN:
        return gmax * xx

    expo = -p[1] * (np.log10(ax / p[0]) - 4.039 * p[1] ** (-1.036))
    if expo > _MAX_SAFE_LOG10_EXPONENT:
        w = 1.0
    elif expo < -_MAX_SAFE_LOG10_EXPONENT:
        w = 0.0
    else:
        w = 1.0 - 1.0 / (1.0 + 10.0**expo)

    t_mkz = gmax * xx / (1.0 + p[3] * (ax / p[2]) ** p[4])
    axd = ax ** p[8]
    t_fkz = np.sign(xx) * p[6] * gmax * axd / (1.0 + gmax / p[7] * p[6] * axd)
    return w * t_mkz + (1.0 - w) * t_fkz


@njit(cache=True)
def _branch(
        xx: float,
        xt: float,
        yt: float,
        ui: int,
        m: int,
        model: int,
        p_g: np.ndarray,
        p_x: np.ndarray,
        gmax2: np.ndarray,
) -> float:  # pragma: no cover (compiled by numba)
    """
    Stress on the current branch (``yi_func()`` in MATLAB).

    Parameters
    ----------
    xx : float
        Strain.
    xt : float
        Strain at the reversal point where the branch starts.
    yt : float
        Stress at the reversal point where the branch starts.
    ui : int
        Number of reversals (0 means the backbone curve).
    m : int
        Index of the sub-layer.
    model : int
        Model code.
    p_g : np.ndarray
        Parameters of the G/Gmax curve of every sub-layer.
    p_x : np.ndarray
        Parameters of the damping curve of every sub-layer.
    gmax2 : np.ndarray
        The Gmax values of both curves of every sub-layer.

    Returns
    -------
    float
        Stress.
    """
    if ui == 0:
        return _backbone(xx, 0, m, model, p_g, p_x, gmax2)

    return yt + 2.0 * _backbone((xx - xt) / 2.0, 1, m, model, p_g, p_x, gmax2)


@njit(cache=True)
def _hysteresis(
        x1: float,
        xx1: float,
        x: float,
        xx: float,
        m: int,
        model: int,
        p_g: np.ndarray,
        p_x: np.ndarray,
        gmax2: np.ndarray,
        bb: np.ndarray,
        xtu: np.ndarray,
        ytu: np.ndarray,
) -> float:  # pragma: no cover (compiled by numba)
    """
    Non-Masing hysteresis (``hyster()`` in MATLAB).

    The caller must make sure that ``xtu`` and ``ytu`` have room for one more
    reversal point.

    Parameters
    ----------
    x1 : float
        Strain increment in the previous step.
    xx1 : float
        Strain increment in the current step.
    x : float
        Strain in the previous step.
    xx : float
        Strain in the current step.
    m : int
        Index of the sub-layer.
    model : int
        Model code.
    p_g : np.ndarray
        Parameters of the G/Gmax curve of every sub-layer.
    p_x : np.ndarray
        Parameters of the damping curve of every sub-layer.
    gmax2 : np.ndarray
        The Gmax values of both curves of every sub-layer.
    bb : np.ndarray
        Number of reversal points of every sub-layer.
    xtu : np.ndarray
        Strains at the reversal points.
    ytu : np.ndarray
        Stresses at the reversal points.

    Returns
    -------
    float
        The stress.
    """
    y = 0.0
    found = False
    if abs(xx1) < _TINY_STRAIN:
        xx1 = _TINY_STRAIN

    if x1 * xx1 < 0.0:  # reversal at this step
        found = True
        b = bb[m]
        xm = x
        ym = _branch(
            xm, xtu[m, b + 1], ytu[m, b + 1], b, m, model, p_g, p_x, gmax2
        )
        if b == 0:  # reversal on the backbone: new hysteretic curve
            axm = abs(xm)
            if axm > 0.0:
                ratio1 = _backbone(axm, 0, m, model, p_g, p_x, gmax2) / (
                    axm * gmax2[0, m]
                )
                ratio2 = _backbone(axm, 1, m, model, p_g, p_x, gmax2) / (
                    axm * gmax2[1, m]
                )
                gmax2[1, m] = gmax2[0, m] * ratio1 / ratio2
            else:  # both ratios are 1 at zero strain
                gmax2[1, m] = gmax2[0, m]

            xtu[m, 1] = -xm
            ytu[m, 1] = -ym

        y = _branch(xx, xm, ym, b + 1, m, model, p_g, p_x, gmax2)
        xtu[m, b + 2] = xm
        ytu[m, b + 2] = ym
        bb[m] = b + 1

    # Check whether previous reversal points are reached (possibly more than
    # one at the same step)
    while (x - xtu[m, bb[m]]) * (xx - xtu[m, bb[m]]) <= 0.0:
        found = True
        b = bb[m]
        if b == 1:
            y = _backbone(xx, 0, m, model, p_g, p_x, gmax2)
            bb[m] = 0
        else:
            y = _branch(
                xx,
                xtu[m, b - 1],
                ytu[m, b - 1],
                b - 2,
                m,
                model,
                p_g,
                p_x,
                gmax2,
            )
            bb[m] = b - 2

    if not found:  # no reversals, and no previous reversal points reached
        b = bb[m]
        y = _branch(
            xx, xtu[m, b + 1], ytu[m, b + 1], b, m, model, p_g, p_x, gmax2
        )

    return y


@njit(cache=True, nogil=True)  # releases the GIL: motions can run in threads
def _march(  # noqa: C901, PLR0915
        nt_start: int,
        nt_end: int,
        model: int,
        boundary: int,
        dt: float,
        n_dt: int,
        a_in: np.ndarray,
        dz: np.ndarray,
        rho: np.ndarray,
        vs_r: float,
        rho_r: float,
        t_k: np.ndarray,
        w_k_sub: np.ndarray,
        node_index: np.ndarray,
        layer_index: np.ndarray,
        state_nodes: np.ndarray,
        state_layers: np.ndarray,
        memory: np.ndarray,
        gmax_lin: np.ndarray,
        ak: np.ndarray,
        fy: np.ndarray,
        ss: np.ndarray,
        gmax2: np.ndarray,
        p_g: np.ndarray,
        p_x: np.ndarray,
        bb: np.ndarray,
        xtu: np.ndarray,
        ytu: np.ndarray,
        scalars: np.ndarray,
        out_a: np.ndarray,
        out_v: np.ndarray,
        out_d: np.ndarray,
        out_gamma: np.ndarray,
        out_tau: np.ndarray,
) -> int:  # pragma: no cover (compiled by numba)
    """
    March in time from sub-step ``nt_start`` to ``nt_end`` (exclusive).

    For the H4 and HH models, it stops early (before changing any state) if a
    sub-layer has no room left for a new reversal point.

    Parameters
    ----------
    nt_start : int
        First sub-step (0-indexed).
    nt_end : int
        The sub-step after the last one.
    model : int
        Model code.
    boundary : int
        Boundary condition code.
    dt : float
        Time step of the sub-steps.
    n_dt : int
        Number of sub-steps in each output time step.
    a_in : np.ndarray
        Input acceleration at every sub-step.
    dz : np.ndarray
        Thickness of every sub-layer.
    rho : np.ndarray
        Mass density of every sub-layer.
    vs_r : float
        Vs of the bedrock.
    rho_r : float
        Mass density of the bedrock.
    t_k : np.ndarray
        Relaxation times of the memory variables.
    w_k_sub : np.ndarray
        Weights of the memory variables of every sub-layer.
    node_index : np.ndarray
        Node index at the top of every layer.
    layer_index : np.ndarray
        Sub-layer index at the mid-point of every layer.
    state_nodes : np.ndarray
        State variables of the nodes.
    state_layers : np.ndarray
        State variables of the sub-layers.
    memory : np.ndarray
        Memory variables of every sub-layer.
    gmax_lin : np.ndarray
        Gmax of every sub-layer (linear model).
    ak : np.ndarray
        Stiffness of the Iwan springs.
    fy : np.ndarray
        Yield stress of the Iwan springs.
    ss : np.ndarray
        Stress of the Iwan springs.
    gmax2 : np.ndarray
        The Gmax values of both curves (H4 and HH).
    p_g : np.ndarray
        Parameters of the G/Gmax curve (H4 and HH).
    p_x : np.ndarray
        Parameters of the damping curve (H4 and HH).
    bb : np.ndarray
        Number of reversal points (H4 and HH).
    xtu : np.ndarray
        Strains at the reversal points (H4 and HH).
    ytu : np.ndarray
        Stresses at the reversal points (H4 and HH).
    scalars : np.ndarray
        Input velocity, input acceleration, and output counter.
    out_a : np.ndarray
        Output acceleration.
    out_v : np.ndarray
        Output velocity.
    out_d : np.ndarray
        Output displacement.
    out_gamma : np.ndarray
        Output strain.
    out_tau : np.ndarray
        Output stress.

    Returns
    -------
    int
        The sub-step after the last one done: ``nt_end``, or less if it stops
        early.
    """
    d_1 = state_nodes[0]
    v_est_1 = state_nodes[1]
    v_est_2 = state_nodes[2]
    d_2 = state_nodes[3]
    v_nt = state_nodes[4]
    a_nt = state_nodes[5]
    tau_1 = state_layers[0]
    gamma_0 = state_layers[1]
    gamma_1 = state_layers[2]
    d_gamma_0 = state_layers[3]
    d_gamma_1 = state_layers[4]
    N = d_1.shape[0]
    n_layer = layer_index.shape[0]
    n_k = t_k.shape[0]
    decay = np.exp(-dt / t_k)

    hysteretic = model in {_MODEL_H4, _MODEL_HH}
    for nt in range(nt_start, nt_end):
        # Every step adds at most one reversal point per sub-layer
        if hysteretic and np.max(bb) + 3 >= xtu.shape[1]:
            return nt

        # ---- Strain, strain increment, and stress of every sub-layer -------
        for i in range(N - 1):
            g1 = (d_1[i + 1] - d_1[i]) / dz[i]
            g_mid = g1 / 2.0 + gamma_0[i] / 2.0
            total = 0.0
            for k in range(n_k):
                memory[i, k] = (
                    decay[k] * memory[i, k]
                    + w_k_sub[i, k] * (1.0 - decay[k]) * g_mid
                )
                total += memory[i, k]

            g1 -= total
            gamma_1[i] = g1
            d_gamma_1[i] = g1 - gamma_0[i]

            if model == _MODEL_LINEAR:
                tau_1[i] = gmax_lin[i] * g1
            elif model in {_MODEL_EPP, _MODEL_H2}:
                force = 0.0
                for j in range(ss.shape[1]):
                    sj = ss[i, j] + ak[i, j] * d_gamma_1[i]
                    sj = min(max(sj, -fy[i, j]), fy[i, j])
                    ss[i, j] = sj
                    force += sj

                tau_1[i] = force
            else:
                tau_1[i] = _hysteresis(
                    d_gamma_0[i],
                    d_gamma_1[i],
                    gamma_0[i],
                    g1,
                    i,
                    model,
                    p_g,
                    p_x,
                    gmax2,
                    bb,
                    xtu,
                    ytu,
                )

            gamma_0[i] = g1
            d_gamma_0[i] = d_gamma_1[i]

        # ---- Input velocity from the prescribed acceleration ---------------
        a_in_1 = a_in[nt]
        v_in_1 = scalars[_IDX_V_IN] + 0.5 * (a_in_1 + scalars[_IDX_A_IN]) * dt
        scalars[_IDX_A_IN] = a_in_1
        scalars[_IDX_V_IN] = v_in_1

        # ---- Predicted velocity ---------------------------------------------
        if boundary == _BOUNDARY_ELASTIC:
            v_est_2[N - 1] = (
                v_est_1[N - 1] * (dz[N - 2] - vs_r * dt)
                + 4.0 * vs_r * v_in_1 * dt
                - 2.0 * tau_1[N - 2] * dt / rho_r
            ) / (dz[N - 2] + vs_r * dt)
        elif boundary == _BOUNDARY_RIGID:
            v_est_2[N - 1] = -v_est_1[N - 1] + 4.0 * v_in_1
        else:  # borehole
            v_est_2[N - 1] = -v_est_1[N - 1] + 2.0 * v_in_1

        for i in range(1, N - 1):
            v_est_2[i] = (
                v_est_1[i]
                + 2.0
                * (tau_1[i] - tau_1[i - 1])
                * dt
                / (dz[i] + dz[i - 1])
                / rho[i]
            )

        v_est_2[0] = v_est_1[0] + 2.0 * tau_1[0] * dt / dz[0] / rho[0]

        # ---- Displacement, velocity, and acceleration -----------------------
        for i in range(N):
            d_2[i] = d_1[i] + v_est_2[i] * dt
            v_nt[i] = 0.5 * (v_est_2[i] + v_est_1[i])
            a_nt[i] = (v_est_2[i] - v_est_1[i]) / dt
            d_1[i] = d_2[i]
            v_est_1[i] = v_est_2[i]

        # ---- Output --------------------------------------------------------
        if nt % n_dt == 0:
            n_o = int(scalars[_IDX_N_OUT])
            for i in range(n_layer + 1):
                out_a[n_o, i] = a_nt[node_index[i]]
                out_v[n_o, i] = v_nt[node_index[i]] - v_nt[N - 1]
                out_d[n_o, i] = d_2[node_index[i]] - d_2[N - 1]

            for i in range(n_layer):
                out_tau[n_o, i] = tau_1[layer_index[i]]
                out_gamma[n_o, i] = gamma_1[layer_index[i]]

            scalars[_IDX_N_OUT] = n_o + 1

    return nt_end

"""
The calculations behind the GUI windows (without any GUI code).

The site response analyses follow the ``run*FromGUI.m`` functions of MATLAB
SeismoSoil: the same input conventions, the same output files (names and
formats), and the same figures.
"""

from __future__ import annotations

import contextlib
import dataclasses
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.optimize import differential_evolution

from PySeismoSoil import helper_hh_calibration as hhc
from PySeismoSoil import helper_hh_model as hh
from PySeismoSoil import helper_mkz_model as mkz
from PySeismoSoil import helper_simulations as sim
from PySeismoSoil import helper_site_response as sr
from PySeismoSoil import helper_time_domain as td
from PySeismoSoil.gui import plots
from PySeismoSoil.gui.data_io import (
    NUM_CURVE_COLUMNS_PER_MATERIAL,
    PRECISION_MOTION,
    InputError,
    save_matrix,
    split_name,
)

if TYPE_CHECKING:
    import pathlib

    from matplotlib.figure import Figure

BedrockType = Literal['elastic', 'rigid']
MotionType = Literal['incident', 'borehole', 'outcrop']
Status = Literal['ok', 'error', 'caution']
ProgressCallback = Callable[[float], None]

BOREHOLE_ON_ELASTIC_ERROR = (
    "If the input motions are 'recorded at borehole', the bedrock should be "
    'rigid, rather than elastic. Please modify your choices.'
)
OUTCROP_CAUTION = (
    'Use With Caution\n\nThe actual motion at the soil-rock interface is '
    'slightly different from the outcrop motion, so it is recommended that '
    'the users deconvolve the rock outcrop motion to incident motion.\n\n'
    'Proceed anyway?'
)

# Konno & Ohmachi (1998) smoothing coefficient
KONNO_OHMACHI_B = 40.0

# Strain array (unit: %) of the Darendeli curves, as in MATLAB SeismoSoil
DARENDELI_STRAIN_PCT = np.geomspace(1e-4, 6, 50)

# Strain array (unit: 1) for fitting the H2 parameters (as ``gaH2.m``)
_H2_STRAIN = 1e-6 * 1.1 ** np.arange(109)


@dataclasses.dataclass(frozen=True)
class Method:
    """
    An analysis method of the GUI.

    Attributes
    ----------
    title : str
        Title of the analysis window.
    folder_suffix : str
        Suffix of the default output folder name (after the site code).
    time_domain : bool
        Whether the analysis is in the time domain.
    model : str
        The constitutive model (time-domain analyses), or the type of the
        frequency-domain analysis.
    tf_name : str
        The name of the transfer function in the output file names.
    """

    title: str
    folder_suffix: str
    time_domain: bool
    model: str
    tf_name: str


METHODS: dict[str, Method] = {
    'linear_fd': Method(
        'Linear Analysis - Frequency Domain',
        'Linear_Analysis_Results',
        time_domain=False,
        model='linear',
        tf_name='linear',
    ),
    'linear_td': Method(
        'Linear Analysis - Time Domain',
        'Linear_Analysis_(Time_Domain)_Results',
        time_domain=True,
        model='linear',
        tf_name='linear',
    ),
    'eql': Method(
        'Equivalent Linear Analysis',
        'Equivalent_Linear_Analysis_Results',
        time_domain=False,
        model='eql',
        tf_name='equivalent_linear',
    ),
    'eql_fd': Method(
        'Equivalent Linear (Frequency Dependent) Analysis',
        'Freq_Dependent_Equiv_Linear_Results',
        time_domain=False,
        model='eql_fd',
        tf_name='equivalent_linear_(FD)',
    ),
    'nl_epp': Method(
        'Nonlinear Analysis - Elasto-Perfectly-Plastic',
        'Nonlinear_Analysis_(EPP)_Results',
        time_domain=True,
        model='EPP',
        tf_name='nonlinear',
    ),
    'nl_h2': Method(
        'Nonlinear Analysis - H2',
        'Nonlinear_Analysis_(H2)_Results',
        time_domain=True,
        model='H2',
        tf_name='nonlinear',
    ),
    'nl_h4': Method(
        'Nonlinear Analysis - H4',
        'Nonlinear_Analysis_(H4)_Results',
        time_domain=True,
        model='H4',
        tf_name='nonlinear',
    ),
    'nl_hh': Method(
        'Nonlinear Analysis - Hybrid Hyperbolic',
        'Nonlinear_Analysis_(HH)_Results',
        time_domain=True,
        model='HH',
        tf_name='nonlinear',
    ),
}


@dataclasses.dataclass(frozen=True)
class BoundarySetting:
    """
    How the input motion enters an analysis.

    Attributes
    ----------
    boundary : str
        The boundary condition of the solver: "elastic" or "rigid" for the
        frequency-domain solvers; "elastic", "rigid", or "borehole" for the
        time-domain solvers.
    division : float
        The input acceleration is divided by this number before it enters the
        solver (to make it an incident motion, or a borehole motion).
    outcrop_factor : float
        The motion that enters the solver times this number is the "rock
        outcrop" motion (for the figures).
    incident_factor : float
        2 if the motion that enters the solver is an incident motion, otherwise
        1 (for calculating the transfer function).
    status : Status, default='ok'
        "ok", "error" (this combination is not allowed), or "caution" (the user
        should confirm before proceeding).
    """

    boundary: str
    division: float
    outcrop_factor: float
    incident_factor: float
    status: Status = 'ok'


def resolve_boundary(
        bedrock_type: BedrockType,
        motion_type: MotionType,
        *,
        time_domain: bool,
) -> BoundarySetting:
    """
    Determine how the input motions enter an analysis.

    Parameters
    ----------
    bedrock_type : BedrockType
        "elastic" or "rigid".
    motion_type : MotionType
        "incident", "borehole", or "outcrop".
    time_domain : bool
        Whether the analysis is a time-domain analysis.

    Returns
    -------
    BoundarySetting
        The boundary condition and the conversion factors.

    Raises
    ------
    ValueError
        When ``bedrock_type`` or ``motion_type`` is invalid
    """
    if bedrock_type not in {'elastic', 'rigid'}:
        raise ValueError(f'Invalid bedrock type: {bedrock_type}')

    if motion_type not in {'incident', 'borehole', 'outcrop'}:
        raise ValueError(f'Invalid motion type: {motion_type}')

    if bedrock_type == 'elastic':
        if motion_type == 'incident':
            return BoundarySetting('elastic', 1, 2, 2)

        status: Status = 'error' if motion_type == 'borehole' else 'caution'
        return BoundarySetting('elastic', 2, 2, 2, status=status)

    # Rigid bedrock
    if motion_type == 'incident':
        return BoundarySetting('rigid', 1, 2, 2)

    # For rigid bedrock, outcrop motion = borehole motion
    if time_domain:
        return BoundarySetting('borehole', 1, 1, 1)

    return BoundarySetting('rigid', 2, 2, 2)


@dataclasses.dataclass
class AnalysisJob:
    """
    All the inputs of a site response analysis (except the motions).

    Attributes
    ----------
    method : str
        Key of ``METHODS``.
    vs_profile : np.ndarray
        The Vs profile, as loaded from the file.
    setting : BoundarySetting
        How the input motions enter the analysis.
    output_dir : pathlib.Path
        The output directory.
    rho_factor : float, default=1.0
        Mass density in the profile is divided by this number to get kg/m^3.
    xi_factor : float, default=1.0
        Damping in the profile is divided by this number to get unit 1.
    accel_factor : float, default=1.0
        Input accelerations are divided by this number to get m/s/s.
    curve : np.ndarray | None, default=None
        The G/Gmax and damping curves.
    G_param : np.ndarray | None, default=None
        The parameters of the G/Gmax curves (H2, H4, or HH).
    xi_param : np.ndarray | None, default=None
        The parameters of the damping curves (H4 or HH).
    tau_max : np.ndarray | None, default=None
        Shear strength (Pa) of every soil layer (EPP).
    """

    method: str
    vs_profile: np.ndarray
    setting: BoundarySetting
    output_dir: pathlib.Path
    rho_factor: float = 1.0
    xi_factor: float = 1.0
    accel_factor: float = 1.0
    curve: np.ndarray | None = None
    G_param: np.ndarray | None = None
    xi_param: np.ndarray | None = None
    tau_max: np.ndarray | None = None

    def profile_in_SI(self) -> np.ndarray:
        """
        Return the Vs profile with damping in unit 1, and density in kg/m^3.

        Returns
        -------
        np.ndarray
            The converted Vs profile.
        """
        profile = np.array(self.vs_profile, dtype=float)
        profile[:, 2] /= self.xi_factor
        profile[:, 3] /= self.rho_factor
        return profile


def run_motion(
        job: AnalysisJob,
        motion: np.ndarray,
        motion_name: str,
        *,
        progress_callback: ProgressCallback | None = None,
        verbose: bool = True,
        figure_lock: AbstractContextManager[Any] | None = None,
) -> list[Figure]:
    """
    Run an analysis for one input motion, and save the results.

    Parameters
    ----------
    job : AnalysisJob
        The inputs of the analysis.
    motion : np.ndarray
        The input motion (two columns), in the unit that the user selected.
    motion_name : str
        File name of the motion (used for the output folder and files).
    progress_callback : ProgressCallback | None, default=None
        Called with the fraction (0 to 1) of the calculation finished
        (time-domain analyses only).
    verbose : bool, default=True
        Whether to print the progress of the iterations (equivalent linear
        analyses).
    figure_lock : AbstractContextManager[Any] | None, default=None
        A lock to hold while creating and saving the figures (matplotlib is not
        fully thread-safe), when several motions run in parallel threads.

    Returns
    -------
    list[Figure]
        The figures of the results (which have also been saved as PNG files).
    """
    method = METHODS[job.method]
    stem, ext = split_name(motion_name)
    ext = ext or '.txt'
    out_dir = job.output_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    accel_in = np.array(motion, dtype=float)
    accel_in[:, 1] = accel_in[:, 1] / job.accel_factor / job.setting.division

    if method.time_domain:
        plot_data = _run_time_domain(
            job, accel_in, stem, ext, out_dir, progress_callback
        )
    else:
        plot_data = _run_frequency_domain(
            job, accel_in, stem, ext, out_dir, verbose=verbose
        )

    with figure_lock or contextlib.nullcontext():
        figures = _result_figures(job, accel_in, stem, plot_data)
        for fig, suffix in zip(
            figures,
            (f'{method.tf_name}_TF', 'accelerations', 'max_a_v_d_gamma_tau'),
            strict=True,
        ):
            fig.savefig(out_dir / f'{stem}_{suffix}.png', dpi=100)

    return figures


@dataclasses.dataclass
class _PlotData:
    """The results of one motion that are plotted."""

    freq: np.ndarray
    tf: np.ndarray
    tf_smoothed: np.ndarray | None
    accel_surface: np.ndarray
    max_avd: np.ndarray
    max_gt: np.ndarray


def _result_figures(
        job: AnalysisJob, accel_in: np.ndarray, stem: str, data: _PlotData
) -> list[Figure]:
    """Create the figures of the results of one motion."""
    method = METHODS[job.method]
    if method.time_domain:
        name = 'Linear' if method.model == 'linear' else 'Nonlinear'
    else:
        name = method.title.split(' - ')[0].replace(' Analysis', '')

    return [
        plots.transfer_function_figure(
            data.freq,
            data.tf,
            title=f'{name} Amplification Factor\n{stem}',
            tf_smoothed=data.tf_smoothed,
        ),
        plots.accelerations_figure(
            t_in=accel_in[:, 0],
            accel_in=accel_in[:, 1] * job.setting.outcrop_factor,
            t_out=data.accel_surface[:, 0],
            accel_out=data.accel_surface[:, 1],
            input_label='Rock outcrop',
            title=f'Input and Output Accelerations\n{stem}',
        ),
        plots.max_profiles_figure(data.max_avd, data.max_gt, stem),
    ]


def _save_results(
        out_dir: pathlib.Path,
        stem: str,
        ext: str,
        arrays: dict[str, np.ndarray],
) -> None:
    """Save result arrays as ``{stem}_{key}{ext}`` in ``out_dir``."""
    for key, data in arrays.items():
        precision = PRECISION_MOTION if key == 'accel_on_surface' else 6
        save_matrix(out_dir / f'{stem}_{key}{ext}', data, precision=precision)


def _run_frequency_domain(
        job: AnalysisJob,
        accel_in: np.ndarray,
        stem: str,
        ext: str,
        out_dir: pathlib.Path,
        *,
        verbose: bool,
) -> _PlotData:
    """Run a linear or equivalent linear analysis in the frequency domain."""
    method = METHODS[job.method]
    profile = job.profile_in_SI()
    # PySeismoSoil's solvers take the "outcrop" (or "borehole") motion, i.e.,
    # 2 times the incident motion
    pyss_input = accel_in.copy()
    pyss_input[:, 1] *= 2
    boundary = 'rigid' if job.setting.boundary == 'rigid' else 'elastic'
    if method.model == 'linear':
        results = sim.linear(profile, pyss_input, boundary=boundary)
    elif method.model == 'eql':
        results = sim.equiv_linear(
            profile, pyss_input, job.curve, boundary=boundary, verbose=verbose
        )
    else:
        results = sim.equiv_linear_freq_dependent(
            profile, pyss_input, job.curve, boundary=boundary, verbose=verbose
        )

    (
        new_profile,
        freq,
        tf,
        accel_surface,
        out_a,
        out_v,
        out_d,
        out_gamma,
        out_tau,
        max_avd,
        max_gt,
    ) = results
    keep = freq > 0
    tf_2col = np.column_stack((freq[keep], np.abs(tf[keep])))
    _save_results(
        out_dir,
        stem,
        ext,
        {
            f'{method.tf_name}_TF': tf_2col,
            'accel_on_surface': accel_surface,
            're-discretized_profile': new_profile,
            'time_history_accel': out_a,
            'time_history_veloc': out_v,
            'time_history_displ': out_d,
            'time_history_strain': out_gamma,
            'time_history_stress': out_tau,
            'max_a_v_d': max_avd,
            'max_gamma_tau': max_gt,
        },
    )
    return _PlotData(
        tf_2col[:, 0], tf_2col[:, 1], None, accel_surface, max_avd, max_gt
    )


def _run_time_domain(
        job: AnalysisJob,
        accel_in: np.ndarray,
        stem: str,
        ext: str,
        out_dir: pathlib.Path,
        progress_callback: ProgressCallback | None,
) -> _PlotData:
    """Run a linear or nonlinear analysis in the time domain."""
    method = METHODS[job.method]
    profile = job.profile_in_SI()
    tau_max = None
    if method.model == 'EPP':
        # The shear strength is defined for every soil layer, so every layer
        # is a different "material"
        n_soil = profile.shape[0] - 1
        tau_max = np.asarray(job.tau_max, dtype=float).ravel()[:n_soil]
        profile[:n_soil, 4] = np.arange(1, n_soil + 1)

    new_profile = sr.stratify(profile)
    results = td.time_domain_site_response(
        new_profile,
        accel_in,
        model=method.model,
        boundary=job.setting.boundary,
        G_param=job.G_param,
        xi_param=job.xi_param,
        tau_max=tau_max,
        progress_callback=progress_callback,
    )
    layer_depth, node_depth, out_a, out_v, out_d, out_gamma, out_tau = results
    t = accel_in[:, 0]
    accel_surface = np.column_stack((t, out_a[:, 0]))
    max_avd = np.column_stack((
        node_depth,
        *(np.max(np.abs(x), axis=0) for x in (out_a, out_v, out_d)),
    ))
    max_gt = np.column_stack((
        layer_depth,
        *(np.max(np.abs(x), axis=0) for x in (out_gamma, out_tau)),
    ))

    # Transfer function (up to f_max), as MATLAB SeismoSoil
    freq, tf_raw = transfer_function(
        accel_in[:, 1],
        out_a[:, 0],
        dt=t[1] - t[0],
        incident_factor=job.setting.incident_factor,
    )
    arrays = {
        'accel_on_surface': accel_surface,
        're-discretized_profile': new_profile,
        'time_history_accel': out_a,
        'time_history_veloc': out_v,
        'time_history_displ': out_d,
        'time_history_strain': out_gamma,
        'time_history_stress': out_tau,
        'max_a_v_d': max_avd,
        'max_gamma_tau': max_gt,
        f'{method.tf_name}_TF_raw': np.column_stack((freq, tf_raw)),
    }
    tf_smoothed = None
    if method.model != 'linear':
        tf_smoothed = konno_ohmachi(
            np.abs(np.fft.fft(out_a[:, 0]))[1 : len(freq) + 1], freq
        ) / (
            job.setting.incident_factor
            * konno_ohmachi(
                np.abs(np.fft.fft(accel_in[:, 1]))[1 : len(freq) + 1], freq
            )
        )
        arrays[f'{method.tf_name}_TF_smoothed'] = np.column_stack((
            freq,
            tf_smoothed,
        ))

    _save_results(out_dir, stem, ext, arrays)
    return _PlotData(freq, tf_raw, tf_smoothed, accel_surface, max_avd, max_gt)


def transfer_function(
        accel_in: np.ndarray,
        accel_out: np.ndarray,
        *,
        dt: float,
        incident_factor: float = 1.0,
        f_max: float = td.DEFAULT_F_MAX_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the (unsmoothed) amplitude of a transfer function.

    Parameters
    ----------
    accel_in : np.ndarray
        Input acceleration (1D).
    accel_out : np.ndarray
        Output acceleration (1D), same length as ``accel_in``.
    dt : float
        Time step.
    incident_factor : float, default=1.0
        The input acceleration is multiplied by this number (e.g., 2 to convert
        an incident motion to an outcrop motion).
    f_max : float, default=td.DEFAULT_F_MAX_HZ
        Maximum frequency.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Frequencies (from the first nonzero one to ``f_max``) and the
        amplitudes of the transfer function.
    """
    n = len(accel_in)
    freq = np.arange(n) / (n * dt)
    keep = (freq > 0) & (freq <= f_max)
    amp_in = np.abs(np.fft.fft(accel_in))[keep]
    amp_out = np.abs(np.fft.fft(accel_out))[keep]
    with np.errstate(divide='ignore', invalid='ignore'):
        tf = amp_out / (incident_factor * amp_in)

    return freq[keep], tf


def konno_ohmachi(
        spectrum: np.ndarray,
        freq: np.ndarray,
        b: float = KONNO_OHMACHI_B,
) -> np.ndarray:
    """
    Smooth a spectrum with the Konno & Ohmachi (1998) window.

    Parameters
    ----------
    spectrum : np.ndarray
        The spectrum (1D).
    freq : np.ndarray
        The frequencies (all positive), same length as ``spectrum``.
    b : float, default=KONNO_OHMACHI_B
        The bandwidth coefficient.

    Returns
    -------
    np.ndarray
        The smoothed spectrum.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    out = np.empty_like(spectrum)
    chunk = max(1, 4_000_000 // max(len(freq), 1))
    log_f = np.log10(freq)
    for start in range(0, len(freq), chunk):
        fc = log_f[start : start + chunk, None]
        x = b * (log_f[None, :] - fc)
        with np.errstate(divide='ignore', invalid='ignore'):
            w = (np.sin(x) / x) ** 4

        w[x == 0] = 1.0
        out[start : start + chunk] = (w @ spectrum) / np.sum(w, axis=1)

    return out


# ----------------------------------------------------------------------------
# Input preparation
# ----------------------------------------------------------------------------
def shear_strength_from_profile(vs_profile: np.ndarray) -> np.ndarray:
    """
    Estimate the shear strength of every soil layer from the Vs profile.

    This is ``calcSuFromVsProfile()`` of MATLAB SeismoSoil: the undrained shear
    strength (Ladd, 1991), with OCR from Vs (Mayne et al., 1998), and the mass
    density from Vs (Mayne et al., 1999).

    Parameters
    ----------
    vs_profile : np.ndarray
        The Vs profile (the last row is the bedrock).

    Returns
    -------
    np.ndarray
        Shear strength (Pa) of every soil layer.
    """
    h = np.array(vs_profile[:-1, 0], dtype=float)
    vs = np.array(vs_profile[:-1, 1], dtype=float)
    rho = hhc._calc_rho(h, vs)  # noqa: SLF001
    sigma_v0 = hhc._calc_vertical_stress(h, rho)  # noqa: SLF001
    OCR = hhc._calc_OCR(vs, rho, sigma_v0)  # noqa: SLF001
    dyna_coeff = 1.2
    return dyna_coeff * 0.28 * OCR**0.8 * sigma_v0


def darendeli_curves(
        vs_profile: np.ndarray,
        PI: float,
        strain_pct: np.ndarray = DARENDELI_STRAIN_PCT,
) -> np.ndarray:
    """
    Calculate G/Gmax and damping curves of every layer (Darendeli, 2001).

    Parameters
    ----------
    vs_profile : np.ndarray
        The Vs profile (5 columns; density in kg/m^3).
    PI : float
        Plasticity index (%) of all the layers.
    strain_pct : np.ndarray, default=DARENDELI_STRAIN_PCT
        The strains (%) of the curves.

    Returns
    -------
    np.ndarray
        The curves of every soil layer, in the "curve" format.
    """
    h = np.array(vs_profile[:-1, 0], dtype=float)
    vs = np.array(vs_profile[:-1, 1], dtype=float)
    rho = np.array(vs_profile[:-1, 3], dtype=float)
    sigma_v0 = hhc._calc_vertical_stress(h, rho)  # noqa: SLF001
    OCR = hhc._calc_OCR(vs, rho, sigma_v0)  # noqa: SLF001
    GGmax, xi, _ = hhc.produce_Darendeli_curves(
        sigma_v0, PI=PI, OCR=OCR, K0=None, strain_in_pct=strain_pct
    )
    n_layer = len(h)
    curve = np.zeros((
        len(strain_pct),
        NUM_CURVE_COLUMNS_PER_MATERIAL * n_layer,
    ))
    for j in range(n_layer):
        curve[:, 4 * j] = strain_pct
        curve[:, 4 * j + 1] = GGmax[:, j]
        curve[:, 4 * j + 2] = strain_pct
        curve[:, 4 * j + 3] = xi[:, j] * 100

    return curve


def _split_curve(curve: np.ndarray, j: int) -> tuple[np.ndarray, ...]:
    """Strain (%), G/Gmax, strain (%), damping (%) of the j-th material."""
    col = NUM_CURVE_COLUMNS_PER_MATERIAL * j
    return tuple(curve[:, col + k] for k in range(4))


def masing_damping(strain: np.ndarray, GGmax: np.ndarray) -> np.ndarray:
    """
    Calculate the damping from a G/Gmax curve with the Masing rules.

    Parameters
    ----------
    strain : np.ndarray
        Strain (unit: 1), increasing.
    GGmax : np.ndarray
        G/Gmax at ``strain``.

    Returns
    -------
    np.ndarray
        Damping ratio (unit: 1), the same as ``gaH2.m``.
    """
    tau = GGmax * strain
    area = np.empty_like(strain)
    area[0] = 0.5 * strain[0] * tau[0]
    area[1:] = area[0] + np.cumsum(
        0.5 * (tau[1:] + tau[:-1]) * np.diff(strain)
    )
    return 2 / np.pi * (2 * area / GGmax / strain**2 - 1)


def _mkz_GGmax(
        strain: np.ndarray, gamma_ref: float, s: float, beta: float
) -> np.ndarray:
    """G/Gmax of the MKZ model."""
    return 1.0 / (1.0 + beta * (strain / gamma_ref) ** s)


def fit_H2(
        curve: np.ndarray,
        *,
        seed: int = 0,
        progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, list[Figure]]:
    """
    Fit the H2 (MKZ + Masing rules) parameters to G/Gmax and damping curves.

    This follows ``gaH2.m`` of MATLAB SeismoSoil: for every material, the MKZ
    parameters minimize the misfit of G/Gmax plus 2 times the misfit of the
    damping (from the Masing rules), with the damping curve shifted to start
    from 0.

    Parameters
    ----------
    curve : np.ndarray
        The curves, in the "curve" format.
    seed : int, default=0
        Seed of the random number generator of the optimization.
    progress_callback : ProgressCallback | None, default=None
        Called after every material, with the fraction finished.

    Returns
    -------
    tuple[np.ndarray, list[Figure]]
        The H2 parameters (4 rows: gamma_ref, 0, s, beta; one column per
        material), and a figure of the fitting result of every material.
    """
    n_mat = curve.shape[1] // NUM_CURVE_COLUMNS_PER_MATERIAL
    strain = _H2_STRAIN
    params = np.zeros((4, n_mat))
    figures = []
    bounds = [(0.0005, 0.005), (0.6, 0.999), (0.2, 1.8)]
    for j in range(n_mat):
        strain_G, GGmax, strain_x, xi = _split_curve(curve, j)
        G_data = np.interp(np.log(strain), np.log(strain_G / 100), GGmax)
        xi_data = np.interp(np.log(strain), np.log(strain_x / 100), xi / 100)
        xi_data -= xi_data[0]

        def misfit(x: np.ndarray, G_data=G_data, xi_data=xi_data) -> float:  # noqa: ANN001
            ratio = _mkz_GGmax(strain, x[0], x[1], x[2])
            damping = masing_damping(strain, ratio)
            return float(
                2 * np.sum(np.abs(xi_data - damping)[:-1])
                + np.sum(np.abs(G_data - ratio)[:-1])
            )

        result = differential_evolution(
            misfit, bounds, seed=seed + j, tol=1e-8
        )
        gamma_ref, s, beta = result.x
        params[:, j] = [gamma_ref, 0.0, s, beta]
        ratio = _mkz_GGmax(strain, gamma_ref, s, beta)
        figures.append(
            plots.curve_fit_figure(
                strain_pct=strain * 100,
                GGmax_data=G_data,
                GGmax_fit=ratio,
                damping_data_pct=xi_data * 100,
                damping_fit_pct=masing_damping(strain, ratio) * 100,
                title=f'Material #{j + 1}',
            ),
        )
        if progress_callback is not None:
            progress_callback((j + 1) / n_mat)

    return params, figures


def fit_H4_G(curve: np.ndarray) -> tuple[np.ndarray, list[Figure]]:
    """
    Fit the MKZ parameters to the G/Gmax curves ("H4_G" parameters).

    Parameters
    ----------
    curve : np.ndarray
        The curves, in the "curve" format.

    Returns
    -------
    tuple[np.ndarray, list[Figure]]
        The H4_G parameters (4 rows: gamma_ref, 0, s, beta; one column per
        material), and a figure of the fitting result of every material.
    """
    param, fitted = mkz.fit_MKZ(curve)
    figures = []
    for j in range(param.shape[0]):
        strain_G, GGmax, _, _ = _split_curve(curve, j)
        fig = plots.curve_fit_figure(
            strain_pct=fitted[:, 4 * j],
            GGmax_data=np.interp(fitted[:, 4 * j], strain_G, GGmax),
            GGmax_fit=fitted[:, 4 * j + 1],
            damping_data_pct=None,
            damping_fit_pct=None,
            title=f'Material #{j + 1}',
        )
        figures.append(fig)

    return param.T, figures


def fit_damping_params(
        curve: np.ndarray,
        model: Literal['H4', 'HH'],
        *,
        pop_size: int = 800,
        n_gen: int = 100,
        progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, list[Figure]]:
    """
    Fit the H4_x or HH_x parameters to the damping curves.

    Parameters
    ----------
    curve : np.ndarray
        The curves, in the "curve" format (only the damping is used).
    model : Literal['H4', 'HH']
        The model.
    pop_size : int, default=800
        Population size of the genetic algorithm.
    n_gen : int, default=100
        Number of generations of the genetic algorithm.
    progress_callback : ProgressCallback | None, default=None
        Called after every material, with the fraction finished.

    Returns
    -------
    tuple[np.ndarray, list[Figure]]
        The parameters (4 rows for H4, 9 rows for HH; one column per material),
        and a figure of the fitting result of every material.
    """
    n_mat = curve.shape[1] // NUM_CURVE_COLUMNS_PER_MATERIAL
    columns = []
    figures = []
    strain_dense = np.geomspace(1e-4, 10, 100)  # unit: %
    for j in range(n_mat):
        _, _, strain_x, xi = _split_curve(curve, j)
        data = np.column_stack((strain_x, xi))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if model == 'H4':
                param = mkz.fit_H4_x_single_layer(
                    data.copy(), pop_size=pop_size, n_gen=n_gen
                )
                columns.append(
                    mkz.serialize_params_to_array(param, to_files=True)
                )
                func = mkz.tau_MKZ
            else:
                param = hh.fit_HH_x_single_layer(
                    data.copy(), pop_size=pop_size, n_gen=n_gen
                )
                columns.append(hh.serialize_params_to_array(param))
                func = hh.tau_HH

            fit = sr.calc_damping_from_param(param, strain_dense / 100, func)

        figures.append(
            plots.curve_fit_figure(
                strain_pct=strain_dense,
                GGmax_data=None,
                GGmax_fit=None,
                damping_data_pct=np.interp(
                    np.log(strain_dense), np.log(strain_x), xi - xi[0]
                ),
                damping_fit_pct=fit * 100,
                title=f'Material #{j + 1}',
            ),
        )
        if progress_callback is not None:
            progress_callback((j + 1) / n_mat)

    return np.column_stack(columns), figures


def fit_HH_G(
        vs_profile: np.ndarray,
        *,
        curve: np.ndarray | None = None,
        tau_max: np.ndarray | None = None,
        fig_output_dir: str | None = None,
        profile_name: str | None = None,
) -> np.ndarray:
    """
    Calculate the HH_G parameters of every soil layer.

    Parameters
    ----------
    vs_profile : np.ndarray
        The Vs profile (5 columns).
    curve : np.ndarray | None, default=None
        G/Gmax curves of every layer (in the "curve" format). If ``None``, the
        curves are calculated from the Vs profile (Darendeli, 2001).
    tau_max : np.ndarray | None, default=None
        Shear strength (Pa) of every soil layer. If ``None``, it is calculated
        from the Vs profile (Ladd, 1991).
    fig_output_dir : str | None, default=None
        If not ``None``, save the figures of the optimization to this folder.
    profile_name : str | None, default=None
        Name of the profile, for the figure file names.

    Returns
    -------
    np.ndarray
        The HH_G parameters: 9 rows, and one column per soil layer.

    Raises
    ------
    InputError
        When the curves do not have enough layers
    """
    save_fig = fig_output_dir is not None
    kwargs = {
        'Tmax': None if tau_max is None else np.asarray(tau_max).ravel(),
        'show_fig': save_fig,
        'save_fig': save_fig,
        'fig_output_dir': fig_output_dir,
        'profile_name': profile_name,
        'verbose': True,
    }
    if curve is None:
        return hhc.hh_param_from_profile(vs_profile, **kwargs)

    n_layer = vs_profile.shape[0] - 1
    if curve.shape[1] < n_layer * NUM_CURVE_COLUMNS_PER_MATERIAL:
        raise InputError(
            'The "curve" file should have the curves of every soil layer.'
        )

    return hhc.hh_param_from_curves(vs_profile, curve, **kwargs)

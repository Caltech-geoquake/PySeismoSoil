"""
Figures of the GUI.

Every function here returns a ``matplotlib.figure.Figure`` without using
``pyplot``, so the figures can be created in background threads, and then
either shown in a window or saved to a file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

from PySeismoSoil import helper_site_response as sr
from PySeismoSoil.gui.data_io import G_ACCEL, NUM_CURVE_COLUMNS_PER_MATERIAL

if TYPE_CHECKING:
    from matplotlib.axes import Axes

BLUE = '#1f77b4'  # the colors used by MATLAB SeismoSoil for input motions
ORANGE = '#ff7f0e'  # ... and for output motions
GRAY = '#999999'
TITLE_SIZE = 11
STRAIN_TICKS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 3]
STRAIN_RANGE_PCT = (1e-4, 3)
MIN_FREQ_FOR_AXIS_LIMITS_HZ = 0.1
TF_YLIM_MARGIN = 1.35


def _figure(width: float, height: float) -> Figure:
    """Create a figure with a constrained layout (size in inches)."""
    return Figure(figsize=(width, height), layout='constrained')


def motion_figure(
        motion: np.ndarray,
        unit: str,
        *,
        unit_factor: float = 1.0,
        title: str = 'Accelerogram',
        show_rms: bool = False,
) -> Figure:
    """
    Plot an acceleration time history and the velocity and displacement.

    Parameters
    ----------
    motion : np.ndarray
        Acceleration time history (two columns).
    unit : str
        Unit of the acceleration (for the label).
    unit_factor : float, default=1.0
        The number that converts m/s/s to ``unit``.
    title : str, default='Accelerogram'
        Title of the figure.
    show_rms : bool, default=False
        Whether to show the RMS acceleration as dashed lines.

    Returns
    -------
    Figure
        The figure.
    """
    t = motion[:, 0]
    a = motion[:, 1]
    veloc, displ = sr.num_int(motion)
    i_pga = int(np.argmax(np.abs(a)))
    pga = abs(a[i_pga])

    fig = _figure(5, 7)
    ax1, ax2, ax3 = fig.subplots(3, 1)
    if show_rms:
        duration = t[-1] - t[0] if t[-1] > t[0] else t[-1]
        a_rms = np.sqrt(np.sum(a**2) * (t[1] - t[0]) / duration)
        ax1.plot(t, a_rms * np.ones_like(t), 'r--', lw=1)
        ax1.plot(t, -a_rms * np.ones_like(t), 'r--', lw=1, label='_')
        ax1.text(
            t[0], a_rms, f' RMS = {a_rms:.3g} {unit}', color='r', va='bottom'
        )

    ax1.plot(t, a, color='b', lw=0.8)
    ax1.plot(t[i_pga], a[i_pga], 'ro', mfc='none', mew=1.75)
    ax1.annotate(
        f'PGA = {pga:.2f} {unit}',
        xy=(t[i_pga], a[i_pga]),
        xytext=(10, 0),
        textcoords='offset points',
        va='center',
    )
    ax1.set_ylabel(f'Acceleration [{unit}]')
    ax1.set_title(title, fontsize=TITLE_SIZE)
    ax2.plot(t, veloc[:, 1] / unit_factor * 100, color='b', lw=0.8)
    ax2.set_ylabel('Velocity [cm/s]')
    ax3.plot(t, displ[:, 1] / unit_factor * 100, color='b', lw=0.8)
    ax3.set_ylabel('Displacement [cm]')
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(t.min(), t.max())
        ax.set_xlabel('Time [s]')
        ax.grid(visible=True, alpha=0.4)

    return fig


def arias_intensity(motion_in_SI: np.ndarray) -> np.ndarray:
    """
    Calculate the cumulative Arias intensity.

    Parameters
    ----------
    motion_in_SI : np.ndarray
        Acceleration (m/s/s) with two columns.

    Returns
    -------
    np.ndarray
        Two columns: time and Arias intensity (m/s).
    """
    t = motion_in_SI[:, 0]
    dt = t[1] - t[0]
    Ia = np.pi / (2 * G_ACCEL) * np.cumsum(motion_in_SI[:, 1] ** 2) * dt
    return np.column_stack((t, Ia))


def arias_figure(Ia: np.ndarray, title: str = '') -> Figure:
    """
    Plot the cumulative Arias intensity.

    Parameters
    ----------
    Ia : np.ndarray
        Two columns: time and Arias intensity (m/s).
    title : str, default=''
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    fig = _figure(5, 3)
    ax = fig.add_subplot()
    ax.plot(Ia[:, 0], Ia[:, 1], lw=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Arias Intensity (m/s)')
    ax.set_xlim(0, Ia[:, 0].max())
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.grid(visible=True, alpha=0.4)
    return fig


def pga_histogram_figure(pga: np.ndarray, unit: str) -> Figure:
    """
    Plot a histogram of PGA values.

    Parameters
    ----------
    pga : np.ndarray
        PGA of every motion.
    unit : str
        Unit of PGA.

    Returns
    -------
    Figure
        The figure.
    """
    n = len(pga)
    n_bins = 10 if n <= 20 else 20 if n <= 50 else 30 if n <= 100 else 40  # noqa: PLR2004
    fig = _figure(6, 4)
    ax = fig.add_subplot()
    ax.hist(pga, bins=n_bins, edgecolor='k')
    ax.set_xlabel(f'PGA [{unit}]')
    ax.set_ylabel('Number of ground motions')
    ax.set_title(f'Total: {n} motions', fontsize=TITLE_SIZE)
    ax.grid(visible=True, alpha=0.4)
    return fig


def _plot_profile(ax: Axes, profile: np.ndarray) -> None:
    """Plot a Vs profile as a staircase on ``ax``."""
    h = profile[:, 0]
    vs = profile[:, 1]
    depth = np.concatenate(([0.0], np.cumsum(h)))
    z_max = depth[-1] + (h[0] if h[-1] == 0 else 0.0)
    x = np.repeat(vs, 2)
    y = np.empty_like(x)
    y[0::2] = depth[:-1]
    y[1::2] = np.append(depth[1:-1], z_max)
    ax.plot(x, y, 'k', lw=1.5)
    ax.set_ylim(z_max, 0)
    ax.set_xlim(0, vs.max() * 1.2)
    ax.set_xlabel('Shear-wave velocity [m/s]')
    ax.set_ylabel('Depth [m]')
    ax.grid(visible=True, alpha=0.4)


def vs_profile_figure(profile: np.ndarray, title: str = '') -> Figure:
    """
    Plot a Vs profile.

    Parameters
    ----------
    profile : np.ndarray
        Vs profile (at least 2 columns: thickness and Vs).
    title : str, default=''
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    fig = _figure(3.6, 5)
    ax = fig.add_subplot()
    _plot_profile(ax, profile)
    ax.set_title(title, fontsize=TITLE_SIZE)
    return fig


def curves_figure(curve: np.ndarray, title: str = '') -> Figure:
    """
    Plot G/Gmax and damping curves (in the "curve" format).

    Parameters
    ----------
    curve : np.ndarray
        The curves: strain (%), G/Gmax, strain (%), damping (%), etc.
    title : str, default=''
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    n_mat = curve.shape[1] // NUM_CURVE_COLUMNS_PER_MATERIAL
    fig = _figure(7.5, 3.4)
    ax1, ax2 = fig.subplots(1, 2)
    for i in range(n_mat):
        col = i * NUM_CURVE_COLUMNS_PER_MATERIAL
        ax1.semilogx(curve[:, col], curve[:, col + 1], label=f'#{i + 1}')
        ax2.semilogx(curve[:, col + 2], curve[:, col + 3])

    ax1.set_ylabel('G/G$_{max}$')
    ax2.set_ylabel('Damping Ratio (%)')
    for ax in (ax1, ax2):
        ax.set_xlabel('Shear Strain (%)')
        ax.set_xlim(*STRAIN_RANGE_PCT)
        ax.set_xticks(STRAIN_TICKS, [str(x) for x in STRAIN_TICKS])
        ax.grid(visible=True, alpha=0.4)

    if n_mat <= 10:  # noqa: PLR2004
        ax1.legend(fontsize=8, title='Material', title_fontsize=8)

    fig.suptitle(title, fontsize=TITLE_SIZE)
    return fig


def response_spectra_figure(
        *,
        Tn: np.ndarray,
        SA: np.ndarray,
        SV: np.ndarray,
        SD: np.ndarray,
        motion: np.ndarray,
        x_scale: str,
        unit: str,
        title: str,
) -> Figure:
    """
    Plot response spectra.

    Parameters
    ----------
    Tn : np.ndarray
        Periods.
    SA : np.ndarray
        Spectral acceleration.
    SV : np.ndarray
        Spectral velocity.
    SD : np.ndarray
        Spectral displacement.
    motion : np.ndarray
        The acceleration time history.
    x_scale : str
        Scale of the period axis ("linear" or "log").
    unit : str
        Unit of the acceleration.
    title : str
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    sv_unit, sd_unit = {
        'm/s^2': ('m/s', 'm'),
        'gal': ('cm/s', 'cm'),
        'g': ('x 9.81 m/s', 'x 9.81 m'),
    }.get(unit, ('', ''))
    fig = _figure(9, 7)
    axes = fig.subplots(2, 2)
    axes[0, 0].plot(motion[:, 0], motion[:, 1], lw=0.8)
    axes[0, 0].set_xlabel('Time [sec]')
    axes[0, 0].set_ylabel(f'Accel. [{unit}]')
    axes[0, 0].set_xlim(motion[:, 0].min(), motion[:, 0].max())
    axes[0, 0].set_title(title, fontsize=TITLE_SIZE)
    for ax, y, label in (
        (axes[0, 1], SA, f'Spectral Acceleration [{unit}]'),
        (axes[1, 0], SV, f'Spectral Velocity [{sv_unit}]'),
        (axes[1, 1], SD, f'Spectral Displacement [{sd_unit}]'),
    ):
        ax.plot(Tn, y, lw=1.5)
        ax.set_xscale(x_scale)
        ax.set_xlim(Tn.min(), Tn.max())
        ax.set_xlabel('Period [sec]')
        ax.set_ylabel(label)

    for ax in axes.flat:
        ax.grid(visible=True, alpha=0.4)

    return fig


def fourier_figure(
        *,
        freq: np.ndarray,
        amplitude: np.ndarray,
        smoothed: np.ndarray | None,
        x_scale: str,
        y_scale: str,
        grid: bool,
        title: str,
) -> Figure:
    """
    Plot a Fourier amplitude spectrum.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies.
    amplitude : np.ndarray
        Fourier amplitudes.
    smoothed : np.ndarray | None
        Smoothed Fourier amplitudes (or ``None``).
    x_scale : str
        Scale of the X axis.
    y_scale : str
        Scale of the Y axis.
    grid : bool
        Whether to show the grid.
    title : str
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    fig = _figure(6, 4.5)
    ax = fig.add_subplot()
    if smoothed is None:
        ax.plot(freq, amplitude, 'k', lw=0.8)
    else:
        ax.plot(freq, amplitude, color='0.3', lw=0.8, label='Original')
        ax.plot(freq, smoothed, 'r', lw=1.75, label='Smoothed')
        ax.legend(loc='upper left')

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    positive = freq[freq > 0]
    lower = (
        positive.min() if x_scale == 'log' and positive.size else freq.min()
    )
    ax.set_xlim(lower, freq.max())
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title, fontsize=TITLE_SIZE)
    if grid:
        ax.grid(visible=True, which='both', alpha=0.4)

    return fig


def _amplitude_spectrum(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided Fourier amplitude spectrum of a two-column signal."""
    dt = motion[1, 0] - motion[0, 0]
    freq = np.fft.rfftfreq(len(motion), dt)
    return freq[1:], np.abs(np.fft.rfft(motion[:, 1]))[1:] * dt


def filter_figure(
        original: np.ndarray,
        filtered: np.ndarray,
        *,
        x_scale: str,
        title: str,
) -> Figure:
    """
    Compare a signal before and after filtering.

    Parameters
    ----------
    original : np.ndarray
        Original signal (two columns).
    filtered : np.ndarray
        Filtered signal (two columns).
    x_scale : str
        Scale of the frequency axis.
    title : str
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    fig = _figure(6, 6)
    ax1, ax2 = fig.subplots(2, 1)
    ax1.plot(
        original[:, 0], original[:, 1], color=GRAY, lw=0.8, label='Original'
    )
    ax1.plot(filtered[:, 0], filtered[:, 1], 'r', lw=0.8, label='Filtered')
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Signal')
    ax1.set_xlim(original[:, 0].min(), original[:, 0].max())
    ax1.legend(loc='upper right')
    ax1.set_title(title, fontsize=TITLE_SIZE)
    for signal, color, label in (
        (original, GRAY, 'Original'),
        (filtered, 'r', 'Filtered'),
    ):
        freq, amp = _amplitude_spectrum(signal)
        ax2.plot(freq, amp, color=color, lw=0.8, label=label)

    ax2.set_xscale(x_scale)
    ax2.set_yscale('log')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Fourier amplitude')
    for ax in (ax1, ax2):
        ax.grid(visible=True, alpha=0.4)

    return fig


def baseline_figure(
        original: np.ndarray,
        corrected: np.ndarray,
        *,
        unit: str,
        title: str,
) -> Figure:
    """
    Compare acceleration, velocity, displacement before and after correction.

    Parameters
    ----------
    original : np.ndarray
        Original acceleration (two columns).
    corrected : np.ndarray
        Baseline-corrected acceleration (two columns).
    unit : str
        Unit of the acceleration.
    title : str
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    fig = _figure(6, 7)
    axes = fig.subplots(3, 1)
    labels = (f'Accel. [{unit}]', f'Veloc. [{unit}*s]', f'Displ. [{unit}*s^2]')
    for motion, color, label in (
        (original, GRAY, 'Original'),
        (corrected, 'r', 'Corrected'),
    ):
        veloc, displ = sr.num_int(motion)
        y_data = (motion[:, 1], veloc[:, 1], displ[:, 1])
        for ax, y in zip(axes, y_data, strict=True):
            ax.plot(motion[:, 0], y, color=color, lw=0.9, label=label)

    for ax, label in zip(axes, labels, strict=False):
        ax.set_ylabel(label)
        ax.set_xlabel('Time [sec]')
        ax.set_xlim(original[:, 0].min(), original[:, 0].max())
        ax.grid(visible=True, alpha=0.4)

    axes[0].legend(loc='upper right')
    axes[0].set_title(title, fontsize=TITLE_SIZE)
    return fig


def transfer_function_figure(
        freq: np.ndarray,
        tf: np.ndarray,
        *,
        title: str,
        tf_smoothed: np.ndarray | None = None,
) -> Figure:
    """
    Plot an amplification function, in log and linear frequency scales.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies.
    tf : np.ndarray
        Amplitude of the transfer function.
    title : str
        Title of the figure.
    tf_smoothed : np.ndarray | None, default=None
        Smoothed transfer function. If provided, ``tf`` is plotted in gray, and
        ``tf_smoothed`` in red.

    Returns
    -------
    Figure
        The figure.
    """
    fig = _figure(5, 7)
    ax1, ax2 = fig.subplots(2, 1)
    for ax in (ax1, ax2):
        if tf_smoothed is None:
            ax.plot(freq, tf, 'k', lw=1.5)
            ref = tf
        else:
            ax.plot(freq, tf, color=GRAY, lw=0.8)
            ax.plot(freq, tf_smoothed, 'r', lw=1.75)
            ref = tf_smoothed

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplification Factor')
        ax.grid(visible=True, which='both', alpha=0.4)
        positive = freq > 0
        low = max(freq[positive].min(), MIN_FREQ_FOR_AXIS_LIMITS_HZ)
        ax.set_xlim(low, freq.max())
        in_range = ref[freq >= low]
        if in_range.size and np.isfinite(in_range).all():
            ax.set_ylim(0, in_range.max() * TF_YLIM_MARGIN)

    ax1.set_xscale('log')
    ax1.set_title(title, fontsize=TITLE_SIZE)
    return fig


def accelerations_figure(
        *,
        t_in: np.ndarray,
        accel_in: np.ndarray,
        t_out: np.ndarray,
        accel_out: np.ndarray,
        input_label: str,
        title: str,
) -> Figure:
    """
    Plot the input and output accelerations.

    Parameters
    ----------
    t_in : np.ndarray
        Time of the input motion.
    accel_in : np.ndarray
        Input acceleration (m/s/s).
    t_out : np.ndarray
        Time of the output motion.
    accel_out : np.ndarray
        Output acceleration (m/s/s).
    input_label : str
        Legend label of the input motion.
    title : str
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    fig = _figure(6, 4)
    ax = fig.add_subplot()
    # Plot the larger one first, so that the smaller one is visible
    lines = [
        (t_in, accel_in, BLUE, input_label),
        (t_out, accel_out, ORANGE, 'Output'),
    ]
    if np.max(np.abs(accel_out)) > np.max(np.abs(accel_in)):
        lines.reverse()

    for t, a, color, label in lines:
        ax.plot(t, a, color=color, lw=1.2, label=label)

    ax.set_ylabel('Acceleration (m/s$^2$)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(0, max(t_in.max(), t_out.max()))
    ax.legend(loc='upper right')
    ax.grid(visible=True, alpha=0.4)
    ax.set_title(title, fontsize=TITLE_SIZE)
    return fig


def max_profiles_figure(
        max_avd: np.ndarray, max_gt: np.ndarray, title: str
) -> Figure:
    """
    Plot the profiles of maximum accel., veloc., displ., strain, and stress.

    Parameters
    ----------
    max_avd : np.ndarray
        Depth, max. acceleration (m/s/s), velocity (m/s), and displacement (m)
        of the layer boundaries.
    max_gt : np.ndarray
        Depth, max. strain (unit: 1), and stress (Pa) of the layer mid-points.
    title : str
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    fig = _figure(12, 6)
    axes = fig.subplots(1, 5, sharey=True)
    z_max = max_avd[:, 0].max()
    data = [
        (max_avd[:, 1], max_avd[:, 0], 'Max. accel. (m/s$^2$)'),
        (max_avd[:, 2] * 100, max_avd[:, 0], 'Max. veloc. (cm/s)'),
        (max_avd[:, 3] * 100, max_avd[:, 0], 'Max. displ. (cm)'),
        (max_gt[:, 1] * 100, max_gt[:, 0], r'$\gamma_{max}$ (%)'),
        (max_gt[:, 2] / 1000, max_gt[:, 0], r'$\tau_{max}$ (kPa)'),
    ]
    for ax, (x, z, label) in zip(axes, data, strict=False):
        ax.plot(x, z, 'b.-')
        ax.set_xlabel(label)
        ax.grid(visible=True, alpha=0.4)

    axes[0].set_ylim(z_max, 0)
    axes[0].set_ylabel('Depth (m)')
    axes[2].set_title(title, fontsize=TITLE_SIZE)
    return fig


def subplot_grid(n: int) -> tuple[int, int]:
    """
    Arrange ``n`` subplots in a grid (as MATLAB SeismoSoil does).

    Parameters
    ----------
    n : int
        Number of subplots.

    Returns
    -------
    tuple[int, int]
        Number of rows and columns.
    """
    n_row = int(np.floor(np.sqrt(n)))
    while n_row > 1 and n % n_row != 0:
        n_row -= 1

    return n_row, n // n_row


def stress_strain_loop_figures(
        strain: np.ndarray,
        stress: np.ndarray,
        depth: np.ndarray,
        n_per_figure: int,
) -> list[Figure]:
    """
    Plot the stress-strain loops of every layer.

    Parameters
    ----------
    strain : np.ndarray
        Strain time histories (unit: 1) of every layer (as columns).
    stress : np.ndarray
        Stress time histories (Pa) of every layer.
    depth : np.ndarray
        Depth of every layer (m).
    n_per_figure : int
        Number of subplots in each figure.

    Returns
    -------
    list[Figure]
        One figure per ``n_per_figure`` layers.
    """
    n_row, n_col = subplot_grid(n_per_figure)
    figures = []
    n_layer = strain.shape[1]
    for start in range(0, n_layer, n_per_figure):
        fig = _figure(n_col * 2.4, n_row * 2.2)
        axes = np.atleast_1d(fig.subplots(n_row, n_col, squeeze=False)).ravel()
        for k, ax in enumerate(axes):
            j = start + k
            if j >= n_layer:
                ax.set_visible(False)
                continue

            ax.plot(strain[:, j] * 100, stress[:, j] / 1000, lw=0.8)
            ax.set_xlabel('Strain [%]', fontsize=8)
            ax.set_ylabel('Stress [kPa]', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_title(
                f'Layer #{j + 1}\nDepth = {depth[j]:.2f} m', fontsize=8
            )
            ax.grid(visible=True, alpha=0.4)

        figures.append(fig)

    return figures


def linear_amplification_figure(
        profile: np.ndarray,
        tf_results: tuple[np.ndarray, ...],
        title: str,
) -> Figure:
    """
    Plot a Vs profile and its linear amplification factors.

    Parameters
    ----------
    profile : np.ndarray
        The Vs profile.
    tf_results : tuple[np.ndarray, ...]
        The results of ``helper_site_response.linear_tf()``.
    title : str
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    freq, af_ro, tf_ro, f0_ro, _, _, af_bh, tf_bh, f0_bh = tf_results
    fig = _figure(10.25, 5.5)
    grid = fig.add_gridspec(2, 5)
    ax_p = fig.add_subplot(grid[:, 0])
    _plot_profile(ax_p, profile)
    ax_p.set_xlabel('Vs [m/s]')
    ax_p.set_title(title, fontsize=TITLE_SIZE)
    for row, (af, tf, f0, name) in enumerate((
        (af_ro, tf_ro, f0_ro, 'Rock outcrop'),
        (af_bh, tf_bh, f0_bh, 'Borehole'),
    )):
        ax_a = fig.add_subplot(grid[row, 1:3])
        ax_a.semilogx(freq, af, 'k', lw=1.5)
        i0 = int(np.argmin(np.abs(freq - f0)))
        ax_a.plot(freq[i0], af[i0], 'ro', mfc='none', mew=1.75)
        ax_a.text(
            0.03,
            0.9,
            f'$f_0$ = {f0:.2f} Hz',
            transform=ax_a.transAxes,
            bbox={'facecolor': 'w', 'edgecolor': '0.5'},
        )
        ax_a.set_ylim(0, af.max() * 1.15)
        ax_a.set_ylabel('Amplification Factor')
        ax_a.set_title(name)
        ax_ph = fig.add_subplot(grid[row, 3:5])
        ax_ph.semilogx(freq, np.unwrap(np.angle(tf)), 'k', lw=1.5)
        ax_ph.set_ylabel('Phase shift (rad)')
        ax_ph.set_title(name)
        for ax in (ax_a, ax_ph):
            ax.set_xlim(freq.min(), freq.max())
            ax.grid(visible=True, which='both', alpha=0.4)
            if row == 1:
                ax.set_xlabel('Frequency [Hz]')

    return fig


def amplification_figure(
        freq: np.ndarray, af: np.ndarray, f0: float, title: str
) -> Figure:
    """
    Plot a linear amplification factor and its fundamental frequency.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies.
    af : np.ndarray
        Amplification factor.
    f0 : float
        Fundamental frequency.
    title : str
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    fig = _figure(6, 4.5)
    ax = fig.add_subplot()
    ax.semilogx(freq, af, 'k', lw=1.5)
    i0 = int(np.argmin(np.abs(freq - f0)))
    ax.plot(freq[i0], af[i0], 'ro', mfc='none', mew=1.75)
    ax.text(
        0.05,
        0.85,
        f'$f_0$ = {f0:.2f} Hz',
        transform=ax.transAxes,
        fontsize=12,
        bbox={'facecolor': 'w', 'edgecolor': '0.5'},
    )
    ax.set_xlim(freq.min(), freq.max())
    ax.set_ylim(0, af.max() * 1.15)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplification Factor')
    ax.grid(visible=True, which='both', alpha=0.4)
    ax.set_title(title, fontsize=TITLE_SIZE)
    return fig


def deconvolution_figure(
        surface: np.ndarray, deconvolved: np.ndarray, title: str
) -> Figure:
    """
    Plot a surface motion and its deconvolved motion.

    Parameters
    ----------
    surface : np.ndarray
        The free surface motion.
    deconvolved : np.ndarray
        The deconvolved motion.
    title : str
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    fig = _figure(6, 3.6)
    ax = fig.add_subplot()
    ax.plot(
        surface[:, 0], surface[:, 1], 'b', lw=0.8, label='Free surface motion'
    )
    ax.plot(
        deconvolved[:, 0],
        deconvolved[:, 1],
        'r',
        lw=0.8,
        label='Deconvolved motion',
    )
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Acceleration')
    ax.legend(loc='best')
    ax.grid(visible=True, alpha=0.4)
    ax.set_title(title, fontsize=TITLE_SIZE)
    return fig


def curve_fit_figure(
        *,
        strain_pct: np.ndarray,
        GGmax_data: np.ndarray | None,
        GGmax_fit: np.ndarray | None,
        damping_data_pct: np.ndarray | None,
        damping_fit_pct: np.ndarray | None,
        title: str,
) -> Figure:
    """
    Compare G/Gmax and/or damping data with the fitted curves.

    Parameters
    ----------
    strain_pct : np.ndarray
        Strain (%) of all the curves.
    GGmax_data : np.ndarray | None
        G/Gmax data (or ``None``).
    GGmax_fit : np.ndarray | None
        Fitted G/Gmax curve (or ``None``).
    damping_data_pct : np.ndarray | None
        Damping data in % (or ``None``).
    damping_fit_pct : np.ndarray | None
        Fitted damping curve in % (or ``None``).
    title : str
        Title of the figure.

    Returns
    -------
    Figure
        The figure.
    """
    panels = []
    if GGmax_data is not None or GGmax_fit is not None:
        panels.append((GGmax_data, GGmax_fit, 'G/G$_{max}$', 'lower left'))

    if damping_data_pct is not None or damping_fit_pct is not None:
        panels.append((
            damping_data_pct,
            damping_fit_pct,
            'Damping Ratio (%)',
            'upper left',
        ))

    fig = _figure(4.5, 3.2 * len(panels))
    axes = np.atleast_1d(fig.subplots(len(panels), 1, squeeze=False)).ravel()
    for ax, (data, fit, label, loc) in zip(axes, panels, strict=False):
        if data is not None:
            ax.semilogx(strain_pct, data, 'b', lw=1.5, label='Data')

        if fit is not None:
            ax.semilogx(strain_pct, fit, 'r--', lw=1.5, label='Fitted curve')

        ax.set_xlabel('Shear Strain (%)')
        ax.set_ylabel(label)
        ax.legend(loc=loc)
        ax.grid(visible=True, which='both', alpha=0.4)

    axes[0].set_title(title, fontsize=TITLE_SIZE)
    return fig


def movie_figure(
        *,
        displ: np.ndarray,
        depth: np.ndarray,
        surface_accel: np.ndarray,
) -> tuple[Figure, object]:
    """
    Create the figure of the ground displacement animation.

    Parameters
    ----------
    displ : np.ndarray
        Displacement time history (m) of every layer boundary.
    depth : np.ndarray
        Depth of every layer boundary (m).
    surface_accel : np.ndarray
        Acceleration on the ground surface (two columns).

    Returns
    -------
    tuple[Figure, object]
        The figure, and a function ``update(k: int, speed: int)`` that shows
        the k-th time step.
    """
    fig = _figure(5.5, 7.5)
    grid = fig.add_gridspec(10, 1)
    ax1 = fig.add_subplot(grid[:6, 0])
    ax2 = fig.add_subplot(grid[7:, 0])
    max_d = max(np.max(np.abs(displ)) * 100, 1e-12)
    (profile_line,) = ax1.plot(displ[0] * 100, depth, 'b-.o', ms=3)
    ax1.set_ylim(depth.max(), 0)
    ax1.set_xlim(-max_d, max_d)
    ax1.set_xlabel('Horizontal ground displacement [cm]')
    ax1.set_ylabel('Depth [m]')
    ax1.grid(visible=True, alpha=0.4)
    t = surface_accel[:, 0]
    ax2.plot(t, surface_accel[:, 1], color='0.4', lw=1.0)
    (past_line,) = ax2.plot(t[:1], surface_accel[:1, 1], 'b', lw=1.5)
    cursor = ax2.axvline(t[0], color='b', lw=1.0)
    ax2.set_xlim(t.min(), t.max())
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Acceleration [m/s/s]')
    ax2.grid(visible=True, alpha=0.4)
    title = ax1.set_title('')

    def update(k: int, speed: int) -> None:
        profile_line.set_xdata(displ[k] * 100)
        past_line.set_data(t[: k + 1], surface_accel[: k + 1, 1])
        cursor.set_xdata([t[k], t[k]])
        title.set_text(f'Time = {t[k]:.2f} sec ({speed} times faster)')

    update(0, 1)
    return fig, update

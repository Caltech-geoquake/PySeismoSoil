"""The "Tools" windows."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from PySeismoSoil import helper_signal_processing as sig
from PySeismoSoil import helper_site_response as sr
from PySeismoSoil.class_gof_scores import GOF_Scores
from PySeismoSoil.gui import analyses as an
from PySeismoSoil.gui import plots
from PySeismoSoil.gui.data_io import (
    ACCEL_UNITS,
    G_ACCEL,
    PRECISION_MOTION,
    InputError,
    check_data,
    load_matrix,
    read_peer_at2,
    read_smc,
    save_matrix,
    split_name,
)
from PySeismoSoil.gui.navigation import Panel, register
from PySeismoSoil.gui.widgets import (
    ALL_FILES,
    Console,
    FileList,
    RadioGroup,
    ask,
    button,
    get_open_file,
    get_open_files,
    group_box,
    info,
    open_folder,
    show_figure,
    warn,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySeismoSoil.gui.data_io import DataKind

UNIT_OPTIONS = [
    ('m/s^2', 'm/s^2'),
    ('gal ( = cm/s^2)', 'gal'),
    ('g ( = 9.8 m/s^2)', 'g'),
]
UNIT_FILE_TAGS = {'m/s^2': 'SI', 'gal': 'gal', 'g': 'g'}
ARRAY_DENSITY_HELP = (
    'This field defines how many uniformly-distributed points there will be '
    'in an order of magnitude (i.e., from 1 to 10, from 100 to 1000, etc.).'
)
GOF_NOTE = (
    'This tool can compute the goodness-of-fit score for comparing two '
    'earthquake time histories as used in Shi & Asimaki (2017).'
)
MIN_PERIOD_SEC = 0.01
DEFAULT_FMIN_HZ = 0.1


@register('tools')
class ToolsPanel(Panel):
    """The "Tools" panel (``SeismoSoil_Tools.m``)."""

    def __init__(self) -> None:
        super().__init__(
            'Tools',
            'SeismoSoil - Tools',
            return_to='main',
            return_text='Return to main panel',
        )
        sets = [
            (
                'Tool Set A: Plotting',
                [
                    ('Time history', 'tool_motion_plotter'),
                    ('Vs profile', 'tool_vs_profile_plotter'),
                    ('Linear ampl. factor', 'tool_linear_ampl_factor'),
                ],
            ),
            (
                'Tool Set B: Signal processing',
                [
                    ('Fourier transform', 'tool_fourier_transform'),
                    ('Elastic response spectra', 'tool_response_spectra'),
                    ('Signal filter', 'tool_signal_filters'),
                    ('Baseline correction', 'tool_baseline_correction'),
                ],
            ),
            (
                'Tool Set C: Ground motion format conversion',
                [
                    ('Motion unit converter', 'tool_unit_converter'),
                    ('PEER to 2-column converter', 'tool_peer_to_2col'),
                    ('SMC to 2-column converter', 'tool_smc_to_2col'),
                ],
            ),
            (
                'Tool Set D: Miscellaneous',
                [
                    ('Surface motion deconvolution', 'tool_deconvolution'),
                    ('Goodness-of-fit score', 'tool_gof_score'),
                ],
            ),
        ]
        for title, tools in sets:
            box = group_box(title)
            grid = QGridLayout(box)
            for k, (text, name) in enumerate(tools):
                btn = button(text, lambda n=name: self.open_sub_panel(n))
                btn.setMinimumHeight(34)
                grid.addWidget(btn, k // 2, k % 2)

            self.main_layout.addWidget(box)


class _FileListTool(Panel):
    """
    A tool that loads a list of files and processes all or some of them.

    Subclasses define ``FILE_KIND`` (for checking the files), and the
    ``process()`` method.
    """

    FILE_KIND: DataKind | None = 'motion'
    SELECT_TEXT = 'Select motion(s)'
    LIST_PLACEHOLDER = 'Motion list...'
    FILE_FILTER = 'Text files (*.dat *.txt);;All Files (*)'

    def __init__(self, title: str) -> None:
        super().__init__(title, f'SeismoSoil - {title}', return_to='tools')
        self.paths: list[pathlib.Path] = []
        self.data: list[np.ndarray] = []
        self.file_list = FileList(self.LIST_PLACEHOLDER)
        self.body = QHBoxLayout()
        self.controls = QVBoxLayout()
        self.controls.addWidget(button(self.SELECT_TEXT, self._select))
        self.body.addLayout(self.controls)
        self.body.addWidget(self.file_list)
        self.main_layout.addLayout(self.body)
        self.bottom = QHBoxLayout()
        self.bottom.addStretch()
        self.bottom.addWidget(
            button('Close all figures', self.close_all_figures)
        )
        self.main_layout.addLayout(self.bottom)

    def read_file(self, path: pathlib.Path) -> np.ndarray:
        """
        Read one file (subclasses may override this).

        Parameters
        ----------
        path : pathlib.Path
            Path of the file.

        Returns
        -------
        np.ndarray
            The data.
        """
        data = load_matrix(path)
        if self.FILE_KIND is not None:
            check_data(data, self.FILE_KIND, path.name)

        return data

    def load_files(self, paths: list[pathlib.Path]) -> int:
        """
        Load files into the list.

        Parameters
        ----------
        paths : list[pathlib.Path]
            Paths of the files.

        Returns
        -------
        int
            Number of files loaded.
        """
        loaded_paths, loaded_data = [], []
        for path in paths:
            try:
                loaded_data.append(self.read_file(path))
            except (OSError, InputError, ValueError) as err:
                warn(self, f'{path.name}: {err}')
                continue

            loaded_paths.append(path)

        if loaded_paths:
            self.paths = loaded_paths
            self.data = loaded_data
            self.file_list.set_names([p.name for p in loaded_paths])

        return len(loaded_paths)

    def _select(self) -> None:
        paths = get_open_files(
            self, self.SELECT_TEXT, file_filter=self.FILE_FILTER
        )
        if paths:
            self.load_files(paths)

    def indices(self, *, selected: bool) -> list[int] | None:
        """
        Return the indices of the files to process.

        Parameters
        ----------
        selected : bool
            Only the selected files (otherwise all of them).

        Returns
        -------
        list[int] | None
            The indices (``None`` if no files have been loaded).
        """
        if not self.data:
            warn(self, "You haven't selected any files yet.")
            return None

        if not selected or len(self.data) == 1:
            return list(range(len(self.data)))

        return self.file_list.selected_indices()

    def run(self, *, selected: bool, save: bool = False) -> None:
        """
        Process all (or the selected) files.

        Parameters
        ----------
        selected : bool
            Only the selected files.
        save : bool, default=False
            Whether to save the results.
        """
        indices = self.indices(selected=selected)
        if indices is None:
            return

        try:
            for i in indices:
                self.process(i, save=save)
        except (InputError, ValueError) as err:
            warn(self, str(err))
            return

        if save and indices:
            open_folder(self.paths[indices[0]].parent)

    def process(self, i: int, *, save: bool) -> None:
        """
        Process one file.

        Parameters
        ----------
        i : int
            Index of the file.
        save : bool
            Whether to save the results.

        Raises
        ------
        NotImplementedError
            In the base class
        """
        raise NotImplementedError

    def add_run_buttons(
            self, verb: str, *, view_only: bool = True, save: bool = True
    ) -> None:
        """
        Add the buttons that process all or the selected files.

        Parameters
        ----------
        verb : str
            The verb on the buttons (e.g., "Filter").
        view_only : bool, default=True
            Whether to add the "(view only)" buttons.
        save : bool, default=True
            Whether to add the "& save" buttons.
        """
        if view_only:
            suffix = ' (view only)' if save else ''
            self.controls.addWidget(
                button(f'{verb} all{suffix}', lambda: self.run(selected=False))
            )
            self.controls.addWidget(
                button(
                    f'{verb} selected{suffix}', lambda: self.run(selected=True)
                ),
            )

        if save:
            self.controls.addWidget(
                button(
                    f'{verb} all && save',
                    lambda: self.run(selected=False, save=True),
                ),
            )
            self.controls.addWidget(
                button(
                    f'{verb} selected && save',
                    lambda: self.run(selected=True, save=True),
                ),
            )

    def output_path(
            self, i: int, suffix: str, ext: str | None = None
    ) -> pathlib.Path:
        """
        Return the path of an output file, next to the i-th input file.

        Parameters
        ----------
        i : int
            Index of the input file.
        suffix : str
            Suffix of the file name (after the input file name).
        ext : str | None, default=None
            File extension (if ``None``, the same as the input file).

        Returns
        -------
        pathlib.Path
            The path.
        """
        stem, in_ext = split_name(self.paths[i].name)
        return (
            self.paths[i].parent / f'{stem}{suffix}{ext or in_ext or ".txt"}'
        )


# ----------------------------------------------------------------------------
# Tool Set A: plotting
# ----------------------------------------------------------------------------
@register('tool_motion_plotter')
class MotionPlotterPanel(_FileListTool):
    """Ground motion plotter (``SeismoSoil_Tools_Motion_Plotter.m``)."""

    def __init__(self) -> None:
        super().__init__('Ground Motion Plotter')
        self.unit = RadioGroup('Unit of acceleration', UNIT_OPTIONS)
        self.show_rms = QCheckBox('Show RMS acceleration')
        self.arias = QCheckBox('Also plot Arias intensity')
        for widget in (self.unit, self.show_rms, self.arias):
            self.controls.addWidget(widget)

        self.controls.addWidget(
            button('Plot all motions', lambda: self.run(selected=False))
        )
        self.controls.addWidget(
            button('Plot selected', lambda: self.run(selected=True))
        )
        self.controls.addWidget(
            button('Plot PGA histogram', self.plot_histogram)
        )
        self.controls.addStretch()

    def process(self, i: int, *, save: bool) -> None:  # noqa: ARG002
        """
        Plot one motion.

        Parameters
        ----------
        i : int
            Index of the motion.
        save : bool
            Not used.
        """
        unit = self.unit.value()
        name = self.paths[i].name
        show_figure(
            plots.motion_figure(
                self.data[i],
                unit,
                unit_factor=ACCEL_UNITS[unit],
                title=name,
                show_rms=self.show_rms.isChecked(),
            ),
            name,
        )
        if self.arias.isChecked():
            motion = self.data[i].copy()
            motion[:, 1] /= ACCEL_UNITS[unit]
            show_figure(
                plots.arias_figure(plots.arias_intensity(motion), name),
                f'{name} - Arias intensity',
            )

    def plot_histogram(self) -> None:
        """Plot the histogram of the PGA of all motions."""
        if not self.data:
            warn(self, "You haven't selected any motions yet.")
            return

        pga = np.array([np.max(np.abs(m[:, 1])) for m in self.data])
        show_figure(
            plots.pga_histogram_figure(pga, self.unit.value()), 'PGA histogram'
        )


@register('tool_vs_profile_plotter')
class VsProfilePlotterPanel(_FileListTool):
    """Vs profile plotter (``SeismoSoil_Tools_Vs_Profile_Plotter.m``)."""

    FILE_KIND = 'vs_profile_2_col'
    SELECT_TEXT = 'Select Profile(s)'
    LIST_PLACEHOLDER = 'Profile list...'

    def __init__(self) -> None:
        super().__init__('Vs Profile Plotter')
        self.add_run_buttons('Plot', save=False)
        self.controls.addStretch()

    def process(self, i: int, *, save: bool) -> None:  # noqa: ARG002
        """
        Plot one Vs profile.

        Parameters
        ----------
        i : int
            Index of the profile.
        save : bool
            Not used.
        """
        name = self.paths[i].name
        show_figure(plots.vs_profile_figure(self.data[i], name), name)


@register('tool_linear_ampl_factor')
class LinearAmplificationPanel(_FileListTool):
    """Linear amplification factor (``SeismoSoil_Tools_Linear_Ampl_...m``)."""

    FILE_KIND = 'vs_profile_2_col'
    SELECT_TEXT = 'Select profile(s)'
    LIST_PLACEHOLDER = 'Profile list...'

    def __init__(self) -> None:
        super().__init__('Plot Linear Amplification Factor')
        self.add_run_buttons('Plot', save=False)
        self.controls.addStretch()

    def process(self, i: int, *, save: bool) -> None:  # noqa: ARG002
        """
        Plot the linear amplification factors of one Vs profile.

        Parameters
        ----------
        i : int
            Index of the profile.
        save : bool
            Not used.
        """
        profile = self.data[i]
        name = self.paths[i].name
        results = sr.linear_tf(profile, show_fig=False)
        show_figure(
            plots.linear_amplification_figure(profile, results, name), name
        )


# ----------------------------------------------------------------------------
# Tool Set B: signal processing
# ----------------------------------------------------------------------------
@register('tool_signal_filters')
class SignalFiltersPanel(_FileListTool):
    """Digital signal filters (``SeismoSoil_Tools_Signal_Filters.m``)."""

    SELECT_TEXT = 'Select signal(s)'

    def __init__(self) -> None:
        super().__init__('Digital Signal Filter Tool')
        self.filter_type = RadioGroup(
            'Choose filter type',
            [
                ('High pass', 'highpass'),
                ('Low pass', 'lowpass'),
                ('Band pass', 'bandpass'),
                ('Band stop', 'bandstop'),
            ],
        )
        self.filter_type.changed.connect(self._update_enabled)
        self.fc_low = QLineEdit('0.2')
        self.fc_high = QLineEdit('30')
        self.x_scale = RadioGroup(
            'Scale of X axis',
            [('Linear', 'linear'), ('Log10', 'log')],
            default=1,
            horizontal=True,
        )
        self.controls.addWidget(self.filter_type)
        self.controls.addWidget(
            _with_unit('Cutoff frequency - low', self.fc_low, 'Hz')
        )
        self.controls.addWidget(
            _with_unit('Cutoff frequency - high', self.fc_high, 'Hz')
        )
        self.controls.addWidget(self.x_scale)
        self.add_run_buttons('Filter')
        self.controls.addStretch()
        self._update_enabled()

    def _update_enabled(self) -> None:
        kind = self.filter_type.value()
        self.fc_low.setEnabled(kind != 'lowpass')
        self.fc_high.setEnabled(kind != 'highpass')

    def filtered(self, i: int) -> np.ndarray:
        """
        Filter one signal.

        Parameters
        ----------
        i : int
            Index of the signal.

        Returns
        -------
        np.ndarray
            The filtered signal.

        Raises
        ------
        InputError
            When the cutoff frequencies are invalid
        """
        kind = self.filter_type.value()
        try:
            f_low = float(self.fc_low.text())
            f_high = float(self.fc_high.text())
        except ValueError as err:
            raise InputError('Cutoff frequencies should be numbers.') from err

        signal = self.data[i]
        if kind == 'highpass':
            return sig.highpass(signal, f_low, filter_order=4)

        if kind == 'lowpass':
            return sig.lowpass(signal, f_high, filter_order=4)

        if f_low >= f_high:
            raise InputError(
                'The low cutoff frequency should be lower than the high one.'
            )

        func = sig.bandpass if kind == 'bandpass' else sig.bandstop
        return func(signal, (f_low, f_high), filter_order=4)

    def process(self, i: int, *, save: bool) -> None:
        """
        Filter one signal, plot (and save) the result.

        Parameters
        ----------
        i : int
            Index of the signal.
        save : bool
            Whether to save the filtered signal.
        """
        result = self.filtered(i)
        name = self.paths[i].name
        show_figure(
            plots.filter_figure(
                self.data[i],
                result,
                x_scale=self.x_scale.value(),
                title=f'{name}\nComparison',
            ),
            name,
        )
        if save:
            path = self.output_path(i, f'_{self.filter_type.value()}_filtered')
            save_matrix(path, result)
            print(f'Saved {path}')


@register('tool_baseline_correction')
class BaselineCorrectionPanel(_FileListTool):
    """Baseline correction (``SeismoSoil_Tools_Baseline_Correction.m``)."""

    def __init__(self) -> None:
        super().__init__('Baseline Correction Tool')
        self.cutoff = QLineEdit('0.2')
        self.controls.addWidget(
            _with_unit('Cutoff frequency', self.cutoff, 'Hz')
        )
        self.controls.addWidget(QLabel('(Suggested value: 0.2 Hz)'))
        self.add_run_buttons('Calculate')
        note = QLabel(
            'Note: The units of acceleration, velocity, and displacement are '
            'consistent with the unit of the motion(s) imported.'
        )
        note.setWordWrap(True)
        self.controls.addWidget(note)
        self.controls.addStretch()

    def process(self, i: int, *, save: bool) -> None:
        """
        Baseline-correct one motion, plot (and save) the result.

        Parameters
        ----------
        i : int
            Index of the motion.
        save : bool
            Whether to save the corrected motion.

        Raises
        ------
        InputError
            When the cutoff frequency is invalid
        """
        try:
            cutoff = float(self.cutoff.text())
        except ValueError as err:
            raise InputError(
                'The cutoff frequency should be a number.'
            ) from err

        result = sig.baseline(self.data[i], cutoff_freq=cutoff)
        name = self.paths[i].name
        show_figure(
            plots.baseline_figure(
                self.data[i],
                result,
                unit='unit',
                title=f'"{name}"\nBaseline corrected',
            ),
            name,
        )
        if save:
            path = self.output_path(i, '_baseline_corrected')
            save_matrix(path, result, precision=PRECISION_MOTION)
            print(f'Saved {path}')


@register('tool_fourier_transform')
class FourierTransformPanel(_FileListTool):
    """Fourier transform (``SeismoSoil_Tools_Fourier_Transform.m``)."""

    SELECT_TEXT = 'Select signal(s)'
    LIST_PLACEHOLDER = 'Signal list...'

    def __init__(self) -> None:
        super().__init__('Fourier Transform')
        scales = [('Linear', 'linear'), ('Log10', 'log')]
        self.x_scale = RadioGroup('Scale of X axis', scales, default=1)
        self.y_scale = RadioGroup('Scale of Y axis', scales, default=0)
        self.sides = RadioGroup(
            'View spectra as',
            [('Single sided', False), ('Double sided', True)],
        )
        self.smoothing = RadioGroup(
            'Smoothing',
            [
                ('None', 'none'),
                ('Regular (faster)', 'regular'),
                ('Konno-Ohmachi (slower)', 'konno_ohmachi'),
            ],
        )
        self.grid_on = QCheckBox('Grid on when plotting?')
        row = QHBoxLayout()
        row.addWidget(self.x_scale)
        row.addWidget(self.y_scale)
        self.controls.addLayout(row)
        for widget in (self.sides, self.smoothing, self.grid_on):
            self.controls.addWidget(widget)

        self.add_run_buttons('Transform')
        self.controls.addWidget(
            QLabel(
                'Note: The unit of Fourier amplitude spectra equals the\n'
                'product of the unit of the signal and the unit of time.'
            ),
        )
        self.controls.addStretch()

    def spectrum(
            self, i: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Calculate the Fourier spectrum of one signal.

        Parameters
        ----------
        i : int
            Index of the signal.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray | None]
            The frequencies, the complex spectrum, and the smoothed amplitude
            (``None`` if no smoothing).
        """
        spectrum = sig.fourier_transform(
            self.data[i], real_val=False, double_sided=self.sides.value()
        )
        freq = np.real(spectrum[:, 0])
        values = spectrum[:, 1]
        smoothing = self.smoothing.value()
        smoothed = None
        if smoothing == 'regular':
            smoothed = sig.sine_smooth(
                np.column_stack((freq, np.abs(values))), window_span=0.3
            )
        elif smoothing == 'konno_ohmachi':
            smoothed = an.konno_ohmachi(np.abs(values), freq)

        return freq, values, smoothed

    def process(self, i: int, *, save: bool) -> None:
        """
        Fourier-transform one signal, plot (and save) the spectrum.

        Parameters
        ----------
        i : int
            Index of the signal.
        save : bool
            Whether to save the spectra.
        """
        freq, values, smoothed = self.spectrum(i)
        name = self.paths[i].name
        show_figure(
            plots.fourier_figure(
                freq=freq,
                amplitude=np.abs(values),
                smoothed=smoothed,
                x_scale=self.x_scale.value(),
                y_scale=self.y_scale.value(),
                grid=self.grid_on.isChecked(),
                title=name,
            ),
            name,
        )
        if save:
            amplitude = np.abs(values) if smoothed is None else smoothed
            save_matrix(
                self.output_path(i, '_Fourier_spectrum_absolute'),
                np.column_stack((freq, amplitude)),
            )
            save_matrix(
                self.output_path(i, '_Fourier_spectrum_complex'),
                np.column_stack((freq, values.real, values.imag)),
            )


@register('tool_response_spectra')
class ResponseSpectraPanel(_FileListTool):
    """Response spectra (``SeismoSoil_Tools_Response_Spectra.m``)."""

    def __init__(self) -> None:
        super().__init__('Response Spectra Calculator')
        self.unit = RadioGroup('Unit of input motions', UNIT_OPTIONS)
        self.x_scale = RadioGroup(
            'Scale of X axis',
            [('Linear', 'linear'), ('Log10', 'log')],
            default=1,
            horizontal=True,
        )
        self.damping = QLineEdit('5')
        self.max_period = QLineEdit('10')
        self.density = QLineEdit('100')
        for widget in (
            self.unit,
            self.x_scale,
            _with_unit('Damping Ratio', self.damping, '%'),
            _with_unit('Maximum Period', self.max_period, 'sec'),
        ):
            self.controls.addWidget(widget)

        box = QGroupBox('Array Density')
        row = QHBoxLayout(box)
        row.addWidget(self.density)
        row.addWidget(
            button(
                "What's this?", lambda: info(self, 'Help', ARRAY_DENSITY_HELP)
            )
        )
        self.controls.addWidget(box)
        self.add_run_buttons('Calculate')
        self.controls.addStretch()

    def spectra(self, i: int) -> tuple[np.ndarray, ...]:
        """
        Calculate the response spectra of one motion.

        Parameters
        ----------
        i : int
            Index of the motion.

        Returns
        -------
        tuple[np.ndarray, ...]
            Tn, SA, PSA, SV, PSV, SD (in the unit of the input motion).

        Raises
        ------
        InputError
            When the parameters are invalid
        """
        try:
            damping = float(self.damping.text()) / 100
            T_max = float(self.max_period.text())
            density = float(self.density.text())
        except ValueError as err:
            raise InputError('The parameters should be numbers.') from err

        if T_max <= MIN_PERIOD_SEC or density <= 0 or damping < 0:
            raise InputError('The parameters are out of range.')

        n_pts = max(2, round(density * np.log10(T_max / MIN_PERIOD_SEC)))
        Tn, SA, PSA, SV, PSV, SD, _ = sr.response_spectra(
            self.data[i],
            T_min=MIN_PERIOD_SEC,
            T_max=T_max,
            n_pts=n_pts,
            damping=damping,
        )
        return Tn, SA, PSA, SV, PSV, SD

    def process(self, i: int, *, save: bool) -> None:
        """
        Calculate, plot (and save) the response spectra of one motion.

        Parameters
        ----------
        i : int
            Index of the motion.
        save : bool
            Whether to save the spectra.
        """
        Tn, SA, PSA, SV, PSV, SD = self.spectra(i)
        name = self.paths[i].name
        show_figure(
            plots.response_spectra_figure(
                Tn=Tn,
                SA=SA,
                SV=SV,
                SD=SD,
                motion=self.data[i],
                x_scale=self.x_scale.value(),
                unit=self.unit.value(),
                title=name,
            ),
            name,
        )
        if save:
            save_matrix(
                self.output_path(i, '_response_spectra'),
                np.column_stack((Tn, SA, PSA, SV, PSV, SD)),
            )


def _with_unit(title: str, edit: QLineEdit, unit: str) -> QGroupBox:
    """Create a group box with a text field and a unit label."""
    box = QGroupBox(title)
    row = QHBoxLayout(box)
    edit.setMaximumWidth(90)
    row.addWidget(edit)
    row.addWidget(QLabel(unit))
    row.addStretch()
    return box


# ----------------------------------------------------------------------------
# Tool Set C: ground motion format conversion
# ----------------------------------------------------------------------------
class _ConverterTool(_FileListTool):
    """Shared parts of the unit and format converters."""

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.desired_unit = RadioGroup('Desired output unit', UNIT_OPTIONS)

    def run(self, *, selected: bool, save: bool = True) -> None:  # noqa: ARG002
        """
        Convert all (or the selected) files.

        Parameters
        ----------
        selected : bool
            Only the selected files.
        save : bool, default=True
            Not used (the results are always saved).
        """
        indices = self.indices(selected=selected)
        if indices is None:
            return

        for k, i in enumerate(indices):
            print(
                f'Converting {k + 1} of {len(indices)}: {self.paths[i].name}'
            )
            self.process(i, save=True)

        if indices and ask(
            self,
            'Finished. Open containing folder?',
            title='Finished',
            default_yes=False,
        ):
            open_folder(self.paths[indices[0]].parent)

    def add_convert_buttons(self) -> None:
        """Add the "Convert all" and "Convert selected" buttons."""
        self.controls.addWidget(
            button('Convert selected', lambda: self.run(selected=True))
        )
        self.controls.addWidget(
            button('Convert all', lambda: self.run(selected=False))
        )


@register('tool_unit_converter')
class UnitConverterPanel(_ConverterTool):
    """Motion unit converter (``SeismoSoil_Tools_Motion_Unit_Converter.m``)."""

    def __init__(self) -> None:
        super().__init__('Ground Motion Unit Converter')
        self.initial_unit = RadioGroup('Initial unit', UNIT_OPTIONS)
        self.desired_unit.setTitle('Desired unit')
        self.controls.addWidget(self.initial_unit)
        self.controls.addWidget(self.desired_unit)
        self.add_convert_buttons()
        self.controls.addStretch()

    def process(self, i: int, *, save: bool) -> None:  # noqa: ARG002
        """
        Convert the unit of one motion, and save it.

        Parameters
        ----------
        i : int
            Index of the motion.
        save : bool
            Not used (always saved).
        """
        to_unit = self.desired_unit.value()
        factor = ACCEL_UNITS[to_unit] / ACCEL_UNITS[self.initial_unit.value()]
        motion = self.data[i].copy()
        motion[:, 1] *= factor
        path = self.output_path(i, f'_in_{UNIT_FILE_TAGS[to_unit]}')
        save_matrix(path, motion)
        print(f'Saved {path}')


class _RawFormatConverter(_ConverterTool):
    """Shared parts of the PEER and SMC converters."""

    FILE_KIND = None
    SELECT_TEXT = 'Select data file(s)'
    LIST_PLACEHOLDER = 'File list...'
    FILE_FILTER = ALL_FILES
    TO_SI = 1.0  # the number that converts the raw unit to m/s/s
    reader: Callable[[pathlib.Path], tuple[float, np.ndarray]]

    def __init__(self, title: str, doc_url: str) -> None:
        super().__init__(title)
        self.main_layout.insertWidget(
            1, QLabel(f'(Format documentation: {doc_url})')
        )
        self.controls.addWidget(self.desired_unit)
        self.show_waveforms = QCheckBox('Show waveforms after conversion')
        self.show_waveforms.setChecked(True)
        self.controls.addWidget(self.show_waveforms)
        self.add_convert_buttons()
        self.controls.addStretch()

    def read_file(self, path: pathlib.Path) -> np.ndarray:
        """
        Read a raw data file, as a two-column motion in its original unit.

        Parameters
        ----------
        path : pathlib.Path
            Path of the file.

        Returns
        -------
        np.ndarray
            Time and acceleration.
        """
        dt, accel = type(self).reader(path)
        time = dt * np.arange(1, len(accel) + 1)
        return np.column_stack((time, accel))

    def process(self, i: int, *, save: bool) -> None:  # noqa: ARG002
        """
        Convert one file to the 2-column format, and save it.

        Parameters
        ----------
        i : int
            Index of the file.
        save : bool
            Not used (always saved).
        """
        unit = self.desired_unit.value()
        motion = self.data[i].copy()
        motion[:, 1] *= self.TO_SI * ACCEL_UNITS[unit]
        stem, _ = split_name(self.paths[i].name)
        path = (
            self.paths[i].parent / f'{stem}_(unit={UNIT_FILE_TAGS[unit]}).txt'
        )
        save_matrix(path, motion)
        print(f'Saved {path}')
        if self.show_waveforms.isChecked():
            label = 'm/s^2' if unit == 'm/s^2' else unit
            show_figure(
                plots.motion_figure(
                    motion, label, unit_factor=ACCEL_UNITS[unit], title=stem
                ),
                stem,
            )


@register('tool_peer_to_2col')
class PeerConverterPanel(_RawFormatConverter):
    """PEER to 2-column converter (``SeismoSoil_Tools_PEER_To_2COL.m``)."""

    SELECT_TEXT = 'Select PEER data files(s)'
    TO_SI = G_ACCEL  # PEER files have units of g
    reader = staticmethod(read_peer_at2)

    def __init__(self) -> None:
        super().__init__(
            'PEER to 2-Column Acceleration Format Converter',
            'http://scec.usc.edu/scecpedia/PEER_Data_Format',
        )


@register('tool_smc_to_2col')
class SmcConverterPanel(_RawFormatConverter):
    """SMC to 2-column converter (``SeismoSoil_Tools_SMC_To_2COL.m``)."""

    SELECT_TEXT = 'Select SMC data files(s)'
    TO_SI = 0.01  # SMC files have units of cm/s/s
    reader = staticmethod(read_smc)

    def __init__(self) -> None:
        super().__init__(
            'SMC (V1) to 2-Column Acceleration Format Converter',
            'http://escweb.wr.usgs.gov/nsmp-data/smcfmt.html',
        )


# ----------------------------------------------------------------------------
# Tool Set D: miscellaneous
# ----------------------------------------------------------------------------
@register('tool_deconvolution')
class DeconvolutionPanel(Panel):
    """Surface motion deconvolution (``SeismoSoil_Tools_Deconvolution.m``)."""

    def __init__(self) -> None:
        super().__init__(
            'Deconvolution Tool',
            'SeismoSoil - Deconvolution Tool',
            return_to='tools',
        )
        self.profile: np.ndarray | None = None
        self.motion_paths: list[pathlib.Path] = []
        self.profile_type = RadioGroup(
            'Profile type',
            [
                ('One uniform rock layer', 'uniform'),
                ('Multiple layers', 'layers'),
            ],
        )
        self.profile_type.changed.connect(self._update_enabled)
        self.main_layout.addWidget(self.profile_type)
        self.main_layout.addWidget(
            QLabel(
                'Based on profile type, either enter rock properties or load '
                'a profile:'
            ),
        )
        self.load_button = button(
            'Load profile of multiple layers', self._load_profile
        )
        self.main_layout.addWidget(self.load_button)
        self.main_layout.addWidget(self._rock_panel())
        self.main_layout.addWidget(self._motion_panel())
        self.desired = RadioGroup(
            'Type of motion desired',
            [
                ('Incident', 'incident'),
                ('Rock-outcrop', 'outcrop'),
                ('Total (borehole)', 'total'),
            ],
            horizontal=True,
        )
        row = QHBoxLayout()
        row.addWidget(self.desired)
        calc = QVBoxLayout()
        calc.addWidget(button('Calculate', self.calculate))
        calc.addWidget(button('Close all figures', self.close_all_figures))
        row.addLayout(calc)
        self.main_layout.addLayout(row)
        self._update_enabled()

    def _rock_panel(self) -> QGroupBox:
        box = QGroupBox('Rock properties')
        grid = QGridLayout(box)
        self.thickness = QLineEdit()
        self.vs = QLineEdit()
        self.density = QLineEdit()
        self.damping = QLineEdit()
        for k, (label, edit, unit) in enumerate(
            (
                ('Thickness =', self.thickness, 'm'),
                ('Vs =', self.vs, 'm/s'),
                ('Density =', self.density, 'kg/m^3'),
                ('Damping =', self.damping, '%'),
            ),
        ):
            row, col = k % 2, (k // 2) * 3
            grid.addWidget(QLabel(label), row, col)
            grid.addWidget(edit, row, col + 1)
            grid.addWidget(QLabel(unit), row, col + 2)

        self.rock_box = box
        return box

    def _motion_panel(self) -> QGroupBox:
        box = QGroupBox('Free surface motion')
        layout = QGridLayout(box)
        layout.addWidget(
            QLabel(
                'Please select free surface motion (i.e., rock outcrop '
                'motion)\n(The unit of acceleration does not matter. The '
                'output acceleration will have the same unit as input.)'
            ),
            0,
            0,
        )
        layout.addWidget(button('Select', self._select_motions), 0, 1)
        return box

    def _update_enabled(self) -> None:
        uniform = self.profile_type.value() == 'uniform'
        self.rock_box.setEnabled(uniform)
        self.load_button.setEnabled(not uniform)

    def load_profile(self, path: pathlib.Path) -> bool:
        """
        Load a Vs profile of multiple layers.

        Parameters
        ----------
        path : pathlib.Path
            Path of the file.

        Returns
        -------
        bool
            Whether the file was loaded.
        """
        try:
            profile = load_matrix(path)
            check_data(profile, 'vs_profile', path.name)
        except (OSError, InputError) as err:
            warn(self, str(err))
            return False

        self.profile = profile
        show_figure(plots.vs_profile_figure(profile, path.name), path.name)
        return True

    def _load_profile(self) -> None:
        path = get_open_file(self, 'Select Vs profile data file...')
        if path is not None:
            self.load_profile(path)

    def set_motions(self, paths: list[pathlib.Path]) -> None:
        """
        Set the free surface motions to deconvolve.

        Parameters
        ----------
        paths : list[pathlib.Path]
            Paths of the motion files.
        """
        self.motion_paths = list(paths)
        for path in paths:
            try:
                motion = load_matrix(path)
            except (OSError, InputError) as err:
                warn(self, str(err))
                continue

            show_figure(
                plots.motion_figure(motion, 'unknown', title=path.name),
                path.name,
            )

    def _select_motions(self) -> None:
        paths = get_open_files(self, 'Select free surface motion(s)...')
        if paths:
            self.set_motions(paths)

    def _uniform_profile(self) -> np.ndarray | None:
        try:
            h = float(self.thickness.text())
            vs = float(self.vs.text())
            rho = float(self.density.text())
            xi = float(self.damping.text()) / 100
        except ValueError:
            warn(self, 'Please enter all the rock properties.', 'Error...')
            return None

        return np.array([[h, vs, xi, rho, 1], [0, vs, xi, rho, 1]])

    def calculate(self) -> list[pathlib.Path]:
        """
        Deconvolve the free surface motions, and save the results.

        Returns
        -------
        list[pathlib.Path]
            The files of the deconvolved motions.
        """
        if self.profile_type.value() == 'uniform':
            profile = self._uniform_profile()
        else:
            profile = self.profile
            if profile is None:
                warn(self, 'Please select a Vs profile.', 'Error...')

        if profile is None:
            return []

        if not self.motion_paths:
            warn(self, 'Please select a free surface motion.', 'Error...')
            return []

        desired = self.desired.value()
        boundary = 'rigid' if desired == 'total' else 'elastic'
        print('Deconvolving... ', end='')
        outputs = []
        for path in self.motion_paths:
            motion = load_matrix(path)
            response, _ = sr.linear_site_resp(
                profile, motion, boundary=boundary, deconv=True
            )
            if desired == 'incident':
                response[:, 1] /= 2

            stem, ext = split_name(path.name)
            out = path.parent / f'{stem}_deconvolved{ext or ".txt"}'
            save_matrix(out, response)
            outputs.append(out)
            show_figure(
                plots.deconvolution_figure(motion, response, path.name),
                path.name,
            )

        print('done.')
        open_folder(self.motion_paths[0].parent)
        return outputs


@register('tool_gof_score')
class GoodnessOfFitPanel(Panel):
    """Goodness-of-fit scores (``SeismoSoil_Tools_GoF_score.m``)."""

    def __init__(self) -> None:
        super().__init__(
            'Calculate goodness-of-fit scores',
            'SeismoSoil - Goodness-of-fit score',
            return_to='tools',
        )
        grid = QGridLayout()
        self.meas_path = QLineEdit()
        self.simu_path = QLineEdit()
        for row, (label, edit) in enumerate((
            ('Measurement', self.meas_path),
            ('Simulation', self.simu_path),
        )):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(
                button('Select', lambda e=edit: self._select(e)), row, 2
            )

        self.main_layout.addLayout(grid)
        row = QHBoxLayout()
        self.meas_unit = RadioGroup(
            'Unit of "measurement"',
            [('m/s/s', 'm/s^2'), ('gal', 'gal'), ('g', 'g')],
        )
        self.simu_unit = RadioGroup(
            'Unit of "simulation"',
            [('m/s/s', 'm/s^2'), ('gal', 'gal'), ('g', 'g')],
        )
        row.addWidget(self.meas_unit)
        row.addWidget(self.simu_unit)
        freq = QGridLayout()
        self.fmin = QLineEdit()
        self.fmax = QLineEdit()
        freq.addWidget(QLabel('F_min = '), 0, 0)
        freq.addWidget(self.fmin, 0, 1)
        freq.addWidget(QLabel('Hz'), 0, 2)
        freq.addWidget(QLabel('F_max = '), 1, 0)
        freq.addWidget(self.fmax, 1, 1)
        freq.addWidget(QLabel('Hz'), 1, 2)
        row.addLayout(freq)
        self.main_layout.addLayout(row)
        bottom = QHBoxLayout()
        note = QLabel(GOF_NOTE)
        note.setWordWrap(True)
        bottom.addWidget(note)
        right = QVBoxLayout()
        calc: QPushButton = button('Calculate Scores', self.calculate)
        right.addWidget(calc)
        right.addWidget(button('Close all figures', self.close_all_figures))
        bottom.addLayout(right)
        self.main_layout.addLayout(bottom)

    def _select(self, edit: QLineEdit) -> None:
        path = get_open_file(self, 'Select a motion...')
        if path is not None:
            edit.setText(str(path))

    @staticmethod
    def _load(edit: QLineEdit, unit: RadioGroup) -> np.ndarray:
        motion = load_matrix(pathlib.Path(edit.text().strip()))
        check_data(motion, 'motion', pathlib.Path(edit.text()).name)
        motion[:, 1] /= ACCEL_UNITS[unit.value()]
        return motion

    def calculate(self) -> np.ndarray | None:
        """
        Calculate the goodness-of-fit scores.

        Returns
        -------
        np.ndarray | None
            The scores (``None`` if the inputs are invalid).
        """
        try:
            meas = GoodnessOfFitPanel._load(self.meas_path, self.meas_unit)
            simu = GoodnessOfFitPanel._load(self.simu_path, self.simu_unit)
            fmin = float(self.fmin.text()) if self.fmin.text().strip() else 0.0
            fmax = (
                float(self.fmax.text()) if self.fmax.text().strip() else None
            )
        except (OSError, InputError, ValueError) as err:
            warn(self, str(err))
            return None

        if fmin <= 0:
            print(
                '****  Warning: f_min value is invalid; using 0.1 Hz instead.'
                '  ****'
            )
            fmin = DEFAULT_FMIN_HZ

        meas, simu = _same_time_step(meas, simu)
        print('Calculating goodness-of-fit scores...\n')
        gof = GOF_Scores(meas, simu)
        scores = gof.calc_scores(fmin=fmin, fmax=fmax)
        text = str(gof)
        print(text)
        Console.instance().show_console()
        info(self, 'Goodness-of-fit scores', text)
        return scores


def _same_time_step(
        meas: np.ndarray, simu: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resample the motion with the smaller time step to the larger one."""
    dt_meas = meas[1, 0] - meas[0, 0]
    dt_simu = simu[1, 0] - simu[0, 0]
    if np.isclose(dt_meas, dt_simu):
        n = min(len(meas), len(simu))
        return meas[:n], simu[:n]

    coarse, fine = (meas, simu) if dt_meas > dt_simu else (simu, meas)
    t_end = min(coarse[-1, 0], fine[-1, 0])
    t = coarse[coarse[:, 0] <= t_end, 0]
    resampled = np.column_stack((t, np.interp(t, fine[:, 0], fine[:, 1])))
    trimmed = coarse[: len(t)]
    return (trimmed, resampled) if dt_meas > dt_simu else (resampled, trimmed)

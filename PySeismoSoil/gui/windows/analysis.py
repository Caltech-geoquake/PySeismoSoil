"""
The analysis windows: linear, equivalent linear, and nonlinear.

One class (``AnalysisPanel``) implements the 8 analysis windows of MATLAB
SeismoSoil (``SeismoSoil_Linear_Analysis.m``, etc.), since they share most of
their steps.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import os
import pathlib
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from PySeismoSoil import helper_signal_processing as sig
from PySeismoSoil import helper_site_response as sr
from PySeismoSoil.gui import analyses as an
from PySeismoSoil.gui import plots
from PySeismoSoil.gui.data_io import (
    ACCEL_UNITS,
    NUM_PROFILE_COLUMNS,
    InputError,
    check_consistency,
    check_data,
    format_duration,
    load_matrix,
    save_matrix,
    site_code_from_filename,
)
from PySeismoSoil.gui.navigation import Panel, register
from PySeismoSoil.gui.widgets import (
    FITTING_MESSAGE,
    BackgroundTask,
    Console,
    FileList,
    RadioGroup,
    TaskContext,
    ask,
    button,
    get_directory,
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

    from matplotlib.figure import Figure

PROFILE_NAME_HELP = (
    "If you don't specify a Vs profile name, a site code modified from the "
    'file name is used. For example, the file name is usually '
    '"profile_CE.11023.dat", then the Vs profile name will be "CE.11023".'
)
BASELINE_CUTOFF_HZ = 0.2  # as in MATLAB SeismoSoil
UNIT_OPTIONS = [
    ('m/s^2', 'm/s^2'),
    ('gal ( = cm/s^2)', 'gal'),
    ('g ( = 9.8 m/s^2)', 'g'),
]
_FIGURE_LOCK = threading.Lock()  # matplotlib is not fully thread-safe


@dataclasses.dataclass(frozen=True)
class AnalysisSpec:
    """
    What an analysis window contains.

    Attributes
    ----------
    method : str
        Key of ``analyses.METHODS``.
    curves : bool, default=False
        Whether the analysis needs a "curve" file.
    params : str | None, default=None
        The model parameters needed: "H2", "H4", "HH", or ``None``.
    strength : bool, default=False
        Whether the analysis needs the shear strength (EPP).
    linear_tf_buttons : bool, default=False
        Whether to show the buttons of the linear amplification function.
    parallel_option : bool, default=True
        Whether to show the "Use multiple CPU cores" option.
    """

    method: str
    curves: bool = False
    params: str | None = None
    strength: bool = False
    linear_tf_buttons: bool = False
    parallel_option: bool = True


@dataclasses.dataclass
class _Step:
    """A step of the workflow, and how to know whether it is complete."""

    title: str
    is_complete: Callable[[], bool]


class AnalysisPanel(Panel):
    """
    An analysis window (the base class of the 8 analysis windows).

    Subclasses set ``SPEC``.
    """

    SPEC: ClassVar[AnalysisSpec]

    def __init__(self) -> None:
        spec = self.SPEC
        self.method = an.METHODS[spec.method]
        super().__init__(
            self.method.title,
            f'SeismoSoil - {self.method.title}',
            return_to='main',
            return_text='Return to main panel',
        )
        self.profile: np.ndarray | None = None
        self.profile_path: pathlib.Path | None = None
        self.site_code = ''
        self.curve: np.ndarray | None = None
        self.curve_name = ''
        self.G_param: np.ndarray | None = None
        self.xi_param: np.ndarray | None = None
        self.tau_max: np.ndarray | None = None
        self.motions: list[np.ndarray] = []
        self.motion_names: list[str] = []
        self.corrected: list[bool] = []
        self.output_dir: pathlib.Path | None = None
        self.last_results: list[Figure] = []
        self.steps: list[_Step] = []
        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _next_step_title(self, title: str, check: Callable[[], bool]) -> str:
        self.steps.append(_Step(title, check))
        return f'Step {len(self.steps)} - {title}'

    def _build(self) -> None:
        grid = QGridLayout()
        left = QVBoxLayout()
        left.addWidget(self._profile_panel())
        if self.SPEC.curves:
            left.addWidget(self._curves_panel())

        if self.SPEC.strength:
            left.addWidget(self._strength_panel())

        left.addStretch()
        right = QVBoxLayout()
        params = self._params_panels()
        for panel in params:
            right.addWidget(panel)

        right.addWidget(self._motions_panel())
        right.addStretch()
        grid.addLayout(left, 0, 0)
        grid.addLayout(right, 0, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(self._output_panel())
        bottom.addWidget(self._run_panel())
        grid.addLayout(bottom, 1, 0, 1, 2)
        self.main_layout.addLayout(grid)

    def _profile_panel(self) -> QGroupBox:
        box = group_box(
            self._next_step_title(
                'Soil Vs Profile', lambda: self.profile is not None
            )
        )
        layout = QGridLayout(box)
        layout.addWidget(QLabel('1. Select a soil profile'), 0, 0, 1, 2)
        layout.addWidget(
            button('Select Vs profile', self._select_profile), 1, 0
        )
        layout.addWidget(button('Plot', self._plot_profile), 1, 1)
        layout.addWidget(QLabel('2. Choose bedrock type'), 2, 0, 1, 2)
        self.bedrock = QComboBox()
        self.bedrock.addItems(['Elastic', 'Rigid'])
        layout.addWidget(self.bedrock, 3, 0)
        layout.addWidget(QLabel('3. Rename profile (optional)'), 4, 0, 1, 2)
        self.profile_name_edit = QLineEdit()
        self.profile_name_edit.editingFinished.connect(self._rename_profile)
        layout.addWidget(self.profile_name_edit, 5, 0)
        layout.addWidget(
            button(
                "What's this?", lambda: info(self, 'Help', PROFILE_NAME_HELP)
            ),
            5,
            1,
        )
        layout.addWidget(QLabel('4. Specify units'), 6, 0, 1, 2)
        self.rho_unit = RadioGroup(
            'Mass density', [('kg/m^3', 1.0), ('g/cm^3', 1 / 1000)]
        )
        self.xi_unit = RadioGroup(
            'Low-strain damping', [('1', 1.0), ('%', 100.0)]
        )
        layout.addWidget(self.rho_unit, 7, 0)
        layout.addWidget(self.xi_unit, 7, 1)
        if self.SPEC.linear_tf_buttons:
            layout.addWidget(
                button('Plot amplification function', self._plot_linear_tf),
                8,
                0,
                1,
                2,
            )
            layout.addWidget(
                button(
                    'Save amplification function to disk', self._save_linear_tf
                ),
                9,
                0,
                1,
                2,
            )

        return box

    def _curves_panel(self) -> QGroupBox:
        box = group_box(
            self._next_step_title(
                'Dynamic Soil Properties', lambda: self.curve is not None
            )
        )
        layout = QVBoxLayout(box)
        layout.addWidget(button('Select curve file', self._select_curve))
        layout.addWidget(
            button('Plot G/Gmax and damping curves', self._plot_curves)
        )
        return box

    def _strength_panel(self) -> QGroupBox:
        box = group_box(
            self._next_step_title(
                'Shear strength', lambda: self.tau_max is not None
            )
        )
        layout = QHBoxLayout(box)
        layout.addWidget(button('Load Tau_max file', self._load_tau_max))
        layout.addWidget(QLabel('OR'))
        layout.addWidget(
            button('Calculate from empirical formulas', self._calc_tau_max)
        )
        return box

    def _params_panels(self) -> list[QGroupBox]:
        model = self.SPEC.params
        if model is None:
            return []

        if model == 'H2':
            title = self._next_step_title(
                'Curve fitting parameters', lambda: self.G_param is not None
            )
            return [self._param_box(title, 'H2_n', 'G')]

        n = len(self.steps) + 1
        self.steps.append(
            _Step(
                'Parameters',
                lambda: self.G_param is not None and self.xi_param is not None,
            ),
        )
        return [
            self._param_box(
                f'Step {n}A - G/Gmax parameters', f'{model}_G', 'G'
            ),
            self._param_box(
                f'Step {n}B - Damping parameters', f'{model}_x', 'x'
            ),
        ]

    def _param_box(self, title: str, file_kind: str, which: str) -> QGroupBox:
        box = group_box(title)
        layout = QHBoxLayout(box)
        layout.addWidget(
            button(
                f'Select {file_kind} file',
                lambda: self._select_param(file_kind, which),
            ),
        )
        if self.SPEC.params in {'H2', 'H4'}:
            layout.addWidget(QLabel('OR'))
            layout.addWidget(
                button('Do curve fitting now', lambda: self._fit_params(which))
            )

        return box

    def _motions_panel(self) -> QGroupBox:
        box = group_box(
            self._next_step_title('Input Motions', lambda: bool(self.motions))
        )
        layout = QGridLayout(box)
        self.accel_unit = RadioGroup('1. Unit of acceleration', UNIT_OPTIONS)
        self.motion_type = RadioGroup(
            '2. Select motion type (the input motions are)',
            [
                ('incident motions', 'incident'),
                ('recorded at borehole', 'borehole'),
                ('recorded on rock outcrop', 'outcrop'),
            ],
        )
        layout.addWidget(self.accel_unit, 0, 0)
        layout.addWidget(self.motion_type, 1, 0)
        buttons = QVBoxLayout()
        buttons.addWidget(QLabel('3. Select and plot motions'))
        buttons.addWidget(button('Select motions', self._select_motions))
        buttons.addWidget(button('Plot all motions', self._plot_all_motions))
        buttons.addWidget(button('Plot selected', self._plot_selected_motions))
        buttons.addWidget(QLabel('4. Baseline correction (optional)'))
        buttons.addWidget(
            button('To all motions', lambda: self._baseline_correct(all_=True))
        )
        buttons.addWidget(
            button(
                'To selected motions',
                lambda: self._baseline_correct(all_=False),
            ),
        )
        buttons.addStretch()
        layout.addLayout(buttons, 0, 1, 2, 1)
        self.motion_list = FileList('Motion list...')
        layout.addWidget(self.motion_list, 0, 2, 2, 1)
        return box

    def _output_panel(self) -> QGroupBox:
        box = group_box(f'Step {len(self.steps) + 1} - Output directory')
        layout = QVBoxLayout(box)
        layout.addWidget(
            QLabel(
                '(Optional) If not designated, the output folder will be in '
                'the current directory.'
            ),
        )
        self.output_label = QLabel('')
        self.output_label.setWordWrap(True)
        layout.addWidget(self.output_label)
        layout.addWidget(button('Select folder', self._select_output_dir))
        return box

    def _run_panel(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        checks = QVBoxLayout()
        self.view_popup = QCheckBox('View results as pop-up figures')
        self.open_when_done = QCheckBox('Open result folder when done')
        self.open_when_done.setChecked(True)
        self.use_parallel = QCheckBox('Use multiple CPU cores')
        checks.addWidget(self.view_popup)
        checks.addWidget(self.open_when_done)
        if self.SPEC.parallel_option:
            checks.addWidget(self.use_parallel)

        layout.addLayout(checks, 0, 0)
        start = QPushButton('Start')
        start.setMinimumSize(140, 60)
        font = start.font()
        font.setPointSize(18)
        start.setFont(font)
        start.clicked.connect(lambda *_: self.start())
        self.start_button = start
        layout.addWidget(start, 1, 0)
        others = QVBoxLayout()
        others.addWidget(
            button('Open result folder', self._open_result_folder)
        )
        others.addWidget(
            button(
                'Post process results',
                lambda: self.open_sub_panel('post_processing', keep_open=True),
            ),
        )
        others.addWidget(button('Clear console', self.clear_console))
        others.addWidget(button('Close all figures', self.close_all_figures))
        layout.addLayout(others, 0, 1, 2, 1)
        return panel

    # ------------------------------------------------------------------
    # Step 1: Vs profile
    # ------------------------------------------------------------------
    def load_profile(self, path: pathlib.Path) -> bool:
        """
        Load a Vs profile file.

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
        except (OSError, InputError) as err:
            warn(self, str(err))
            return False

        try:
            check_data(profile, 'vs_profile')
        except InputError as err:
            # As MATLAB SeismoSoil, only warn (e.g., the damping can be in %)
            warn(self, str(err))
            if profile.ndim != 2 or profile.shape[1] != NUM_PROFILE_COLUMNS:  # noqa: PLR2004
                return False

        self.profile = profile
        self.profile_path = path
        self.site_code = site_code_from_filename(path.name)
        self.profile_name_edit.setText(self.site_code)
        self._set_default_output_dir()
        return True

    def _select_profile(self) -> None:
        path = get_open_file(self, 'Select Vs profile data file...')
        if path is not None:
            self.load_profile(path)

    def _plot_profile(self) -> None:
        if self.profile is None:
            warn(self, 'Please select a Vs profile first.')
            return

        name = self.profile_path.name if self.profile_path else ''
        show_figure(plots.vs_profile_figure(self.profile, name), name)

    def _rename_profile(self) -> None:
        self.site_code = self.profile_name_edit.text().strip()
        self._set_default_output_dir()

    def _set_default_output_dir(self) -> None:
        folder = f'{self.site_code}_{self.method.folder_suffix}'
        self.output_dir = pathlib.Path.cwd() / folder
        self.output_label.setText(f'Output folder: {self.output_dir}')

    def _profile_in_SI(self) -> np.ndarray:
        profile = np.array(self.profile, dtype=float)
        profile[:, 2] /= self.xi_unit.value()
        profile[:, 3] /= self.rho_unit.value()
        return profile

    def _linear_tf(self) -> tuple[np.ndarray, ...] | None:
        if self.profile is None:
            warn(self, 'Please select a Vs profile first.')
            return None

        return sr.linear_tf(self._profile_in_SI(), show_fig=False)

    def _plot_linear_tf(self) -> None:
        results = self._linear_tf()
        if results is None:
            return

        freq, af_ro, _, f0_ro, _, _, af_bh, _, f0_bh = results
        rigid = self.bedrock.currentText() == 'Rigid'
        af, f0 = (af_bh, f0_bh) if rigid else (af_ro, f0_ro)
        show_figure(
            plots.amplification_figure(
                freq, af, f0, f'{self.site_code}\nLinear Amplification Factor'
            ),
            'Linear amplification factor',
        )

    def _save_linear_tf(self) -> None:
        results = self._linear_tf()
        if results is None:
            return

        out_dir = get_directory(
            self, 'Select directory for exporting results...'
        )
        if out_dir is None:
            return

        freq, af_ro, tf_ro, _, _, _, af_bh, tf_bh, _ = results
        rigid = self.bedrock.currentText() == 'Rigid'
        af, tf = (af_bh, tf_bh) if rigid else (af_ro, tf_ro)
        save_matrix(
            out_dir / f'{self.site_code}_linear_amplification_factor_real.dat',
            np.column_stack((freq, af)),
        )
        save_matrix(
            out_dir
            / f'{self.site_code}_linear_amplification_factor_complex.dat',
            np.column_stack((freq, tf.real, tf.imag)),
        )
        open_folder(out_dir)

    # ------------------------------------------------------------------
    # Step 2/3: curves and model parameters
    # ------------------------------------------------------------------
    def load_curve(self, path: pathlib.Path) -> bool:
        """
        Load a "curve" file (G/Gmax and damping curves).

        Parameters
        ----------
        path : pathlib.Path
            Path of the file.

        Returns
        -------
        bool
            Whether the file was loaded.
        """
        return self._load_into('curve', path, 'curve')

    def load_param(self, path: pathlib.Path, which: str) -> bool:
        """
        Load a file of model parameters.

        Parameters
        ----------
        path : pathlib.Path
            Path of the file.
        which : str
            "G" (G/Gmax curve, or the H2 parameters) or "x" (damping curve).

        Returns
        -------
        bool
            Whether the file was loaded.
        """
        kind = 'hh_param' if self.SPEC.params == 'HH' else 'mkz_param'
        attr = 'G_param' if which == 'G' else 'xi_param'
        return self._load_into(attr, path, kind)

    def load_tau_max(self, path: pathlib.Path) -> bool:
        """
        Load a "Tau_max" (shear strength) file.

        Parameters
        ----------
        path : pathlib.Path
            Path of the file.

        Returns
        -------
        bool
            Whether the file was loaded.
        """
        return self._load_into('tau_max', path, 'tau_max')

    def _load_into(self, attr: str, path: pathlib.Path, kind: Any) -> bool:
        try:
            data = load_matrix(path)
            check_data(data, kind, path.name)
        except (OSError, InputError) as err:
            warn(self, str(err))
            return False

        setattr(self, attr, data)
        if attr == 'curve':
            self.curve_name = path.name

        return True

    def _select_curve(self) -> None:
        path = get_open_file(self, 'Select "curve" data file...')
        if path is not None:
            self.load_curve(path)

    def _plot_curves(self) -> None:
        if self.curve is None:
            warn(self, 'Please select a "curve" file first.')
            return

        show_figure(plots.curves_figure(self.curve, self.curve_name), 'Curves')

    def _select_param(self, file_kind: str, which: str) -> None:
        current = self.G_param if which == 'G' else self.xi_param
        if current is not None and not ask(
            self,
            f"You've already imported {file_kind} data. Importing again will "
            'overwrite it. Do it anyway?',
            default_yes=False,
        ):
            return

        path = get_open_file(self, f'Select "{file_kind}" data file...')
        if path is not None:
            self.load_param(path, which)

    def _fit_params(self, which: str) -> None:
        if self.curve is None:
            warn(self, 'Please select a "curve" file first (Step 2).')
            return

        current = self.G_param if which == 'G' else self.xi_param
        if current is not None and not ask(
            self,
            'The parameters have already been imported. This calculation '
            'will overwrite the imported data. Do it anyway?',
            default_yes=False,
        ):
            return

        curve = self.curve.copy()
        model = self.SPEC.params

        def task(ctx: TaskContext) -> tuple[np.ndarray, list[Figure]]:
            ctx.progress(0, 'Curve fitting...')
            if model == 'H2':
                return an.fit_H2(curve, progress_callback=ctx.progress)

            if which == 'G':
                return an.fit_H4_G(curve)

            return an.fit_damping_params(
                curve, 'H4', progress_callback=ctx.progress
            )

        def done(result: tuple[np.ndarray, list[Figure]]) -> None:
            params, figures = result
            setattr(self, 'G_param' if which == 'G' else 'xi_param', params)
            print('Fitted parameters:')
            print(params)
            for fig in figures:
                show_figure(fig, 'Curve fitting')

        BackgroundTask(
            self,
            task,
            on_done=done,
            message=FITTING_MESSAGE,
        ).start()

    def _load_tau_max(self) -> None:
        path = get_open_file(self, 'Select "Tau_max" data file...')
        if path is not None:
            self.load_tau_max(path)

    def _calc_tau_max(self) -> None:
        if self.profile is None:
            warn(self, 'Please select a Vs profile first (Step 1).')
            return

        if self.tau_max is not None and not ask(
            self,
            "You've already imported shear strength. This calculation will "
            'overwrite the imported data. Do it anyway?',
            default_yes=False,
        ):
            return

        su = an.shear_strength_from_profile(self._profile_in_SI())
        self.tau_max = su
        profile = np.column_stack((self.profile[:-1, 0], su / 1000))
        fig = plots.vs_profile_figure(profile, 'Shear strength')
        fig.axes[0].set_xlabel('Shear strength [kPa]')
        show_figure(fig, 'Shear strength')

    # ------------------------------------------------------------------
    # Input motions
    # ------------------------------------------------------------------
    def load_motions(self, paths: list[pathlib.Path]) -> int:
        """
        Load input motion files (two columns: time and acceleration).

        Parameters
        ----------
        paths : list[pathlib.Path]
            Paths of the files.

        Returns
        -------
        int
            The number of motions loaded.
        """
        motions, names = [], []
        for i, path in enumerate(paths):
            try:
                motion = load_matrix(path)
                check_data(motion, 'motion', f'Motion #{i + 1}')
            except (OSError, InputError) as err:
                warn(self, str(err))
                continue

            motions.append(motion)
            names.append(path.name)

        if motions:
            self.motions = motions
            self.motion_names = names
            self.corrected = [False] * len(motions)
            self.motion_list.set_names(names)

        return len(motions)

    def _select_motions(self) -> None:
        paths = get_open_files(self, 'Select input motion(s)...')
        if paths:
            self.load_motions(paths)

    def _check_motions(self) -> bool:
        if not self.motions:
            warn(self, "You haven't selected any motions yet.")
            return False

        return True

    def _selected(self) -> list[int]:
        if len(self.motions) == 1:
            return [0]

        return self.motion_list.selected_indices()

    def _plot_motions(self, indices: list[int]) -> None:
        unit = self.accel_unit.value()
        for i in indices:
            fig = plots.motion_figure(
                self.motions[i],
                unit,
                unit_factor=ACCEL_UNITS[unit],
                title=self.motion_names[i],
            )
            show_figure(fig, self.motion_names[i])

    def _plot_all_motions(self) -> None:
        if self._check_motions():
            self._plot_motions(list(range(len(self.motions))))

    def _plot_selected_motions(self) -> None:
        if self._check_motions():
            self._plot_motions(self._selected())

    def _baseline_correct(self, *, all_: bool) -> None:
        if not self._check_motions():
            return

        indices = list(range(len(self.motions))) if all_ else self._selected()
        done = [i for i in indices if self.corrected[i]]
        if done and len(done) == len(indices):
            info(
                self,
                'Information',
                'All of the selected motions have been corrected. They will '
                'not be processed again.',
            )
            return

        if done:
            info(
                self,
                'Information',
                'Some of the selected motions have been corrected. They will '
                'not be processed again.',
            )

        for i in indices:
            if not self.corrected[i]:
                self.motions[i] = sig.baseline(
                    self.motions[i], cutoff_freq=BASELINE_CUTOFF_HZ
                )
                self.corrected[i] = True

        info(self, 'Finished', 'All baseline correction finished.')

    # ------------------------------------------------------------------
    # Output, and running the analysis
    # ------------------------------------------------------------------
    def _select_output_dir(self) -> None:
        path = get_directory(self, 'Select directory for exporting results...')
        if path is not None:
            self.output_dir = path
            self.output_label.setText(f'Output folder: {path}')
            print(f'Output directory: {path}')

    def _open_result_folder(self) -> None:
        if self.output_dir is None or not self.output_dir.exists():
            warn(self, 'The result folder does not exist yet.')
            return

        open_folder(self.output_dir)

    def _unfinished_steps(self) -> list[int]:
        return [
            i + 1
            for i, step in enumerate(self.steps)
            if not step.is_complete()
        ]

    def make_job(self) -> an.AnalysisJob | None:
        """
        Collect the inputs of the analysis, after checking them.

        Returns
        -------
        an.AnalysisJob | None
            The inputs (``None`` if they are incomplete or invalid, in which
            case a message has been shown to the user).
        """
        unfinished = self._unfinished_steps()
        if unfinished:
            warn(
                self,
                f"You haven't finished step(s) "
                f'{", ".join(str(i) for i in unfinished)}.\nYou need to '
                'finish it/them before starting the analysis.',
            )
            return None

        setting = an.resolve_boundary(
            self.bedrock.currentText().lower(),
            self.motion_type.value(),
            time_domain=self.method.time_domain,
        )
        job = an.AnalysisJob(
            method=self.SPEC.method,
            vs_profile=self.profile,
            setting=setting,
            output_dir=self.output_dir,
            rho_factor=self.rho_unit.value(),
            xi_factor=self.xi_unit.value(),
            accel_factor=ACCEL_UNITS[self.accel_unit.value()],
            curve=self.curve,
            G_param=self.G_param,
            xi_param=self.xi_param,
            tau_max=self.tau_max,
        )
        if not self._job_is_valid(job):
            return None

        return job

    def _check_inputs(self, job: an.AnalysisJob) -> None:
        check_data(job.profile_in_SI(), 'vs_profile', 'Vs profile')
        if self.SPEC.curves or self.SPEC.params:
            check_consistency(
                job.profile_in_SI(),
                curve=self.curve,
                G_param=self.G_param,
                xi_param=self.xi_param,
                param_name=self.SPEC.params or 'H2',
            )

        n_soil = len(self.profile) - 1
        if self.SPEC.strength and np.size(self.tau_max) < n_soil:
            raise InputError(
                'Number of layers in tau_max and in vs_profile should match.'
            )

    def _job_is_valid(self, job: an.AnalysisJob) -> bool:
        try:
            self._check_inputs(job)
        except InputError as err:
            warn(self, str(err))
            return False

        if job.setting.status == 'error':
            warn(self, an.BOREHOLE_ON_ELASTIC_ERROR, 'Error')
            return False

        return job.setting.status != 'caution' or ask(self, an.OUTCROP_CAUTION)

    def start(self) -> BackgroundTask | None:
        """
        Start the analysis (the "Start" button).

        Returns
        -------
        BackgroundTask | None
            The background task that runs the analysis (``None`` if the inputs
            are incomplete or invalid).
        """
        job = self.make_job()
        if job is None:
            return None

        job.output_dir.mkdir(parents=True, exist_ok=True)
        motions = list(self.motions)
        names = list(self.motion_names)
        parallel = self.SPEC.parallel_option and self.use_parallel.isChecked()
        t_start = time.perf_counter()
        Console.instance().show_console()

        def task(ctx: TaskContext) -> list[Figure]:
            return _run_all_motions(
                job, motions, names, ctx, parallel=parallel
            )

        def done(figures: list[Figure]) -> None:
            elapsed = format_duration(time.perf_counter() - t_start)
            print(f'Simulation finished in {elapsed}.')
            self.last_results = figures
            if self.view_popup.isChecked():
                for fig in figures:
                    show_figure(fig, self.method.title)

            info(self, 'Done', f'Simulation finished in {elapsed}.')
            if self.open_when_done.isChecked():
                open_folder(job.output_dir)

        return BackgroundTask(self, task, on_done=done).start()


def _run_all_motions(
        job: an.AnalysisJob,
        motions: list[np.ndarray],
        names: list[str],
        ctx: TaskContext,
        *,
        parallel: bool,
) -> list[Figure]:
    """Run the analysis of every motion (in a background thread)."""
    n = len(motions)
    fractions = [0.0] * n
    lock = threading.Lock()

    def report(i: int, fraction: float) -> None:
        with lock:
            fractions[i] = fraction
            total = sum(fractions) / n

        ctx.progress(total, f'Ground motion No. {i + 1} of {n}: {names[i]}')

    def run_one(i: int) -> list[Figure]:
        print(f'Ground motion No. {i + 1}: {names[i]}')
        report(i, 0.0)
        figures = an.run_motion(
            job,
            motions[i],
            names[i],
            progress_callback=lambda f: report(i, f),
            verbose=True,
            figure_lock=_FIGURE_LOCK if parallel else None,
        )
        report(i, 1.0)
        return figures

    if not parallel or n == 1:
        return [fig for i in range(n) for fig in run_one(i)]

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(n, os.cpu_count() or 1)
    ) as pool:
        results = list(pool.map(run_one, range(n)))

    return [fig for figs in results for fig in figs]


@register('analysis_linear_fd')
class LinearFrequencyDomainPanel(AnalysisPanel):
    """Linear analysis in the frequency domain."""

    SPEC = AnalysisSpec(
        'linear_fd', linear_tf_buttons=True, parallel_option=False
    )


@register('analysis_linear_td')
class LinearTimeDomainPanel(AnalysisPanel):
    """Linear analysis in the time domain."""

    SPEC = AnalysisSpec('linear_td')


@register('analysis_eql')
class EquivalentLinearPanel(AnalysisPanel):
    """Equivalent linear analysis (Seed & Idriss, 1970)."""

    SPEC = AnalysisSpec('eql', curves=True)


@register('analysis_eql_fd')
class FreqDependentEquivalentLinearPanel(AnalysisPanel):
    """Frequency-dependent equivalent linear analysis."""

    SPEC = AnalysisSpec('eql_fd', curves=True)


@register('analysis_nl_epp')
class NonlinearEPPPanel(AnalysisPanel):
    """Nonlinear analysis with the elastic-perfectly-plastic model."""

    SPEC = AnalysisSpec('nl_epp', strength=True)


@register('analysis_nl_h2')
class NonlinearH2Panel(AnalysisPanel):
    """Nonlinear analysis with the H2 model (MKZ + Masing rules)."""

    SPEC = AnalysisSpec('nl_h2', curves=True, params='H2')


@register('analysis_nl_h4')
class NonlinearH4Panel(AnalysisPanel):
    """Nonlinear analysis with the H4 model (MKZ + non-Masing rules)."""

    SPEC = AnalysisSpec('nl_h4', curves=True, params='H4')


@register('analysis_nl_hh')
class NonlinearHHPanel(AnalysisPanel):
    """Nonlinear analysis with the hybrid hyperbolic (HH) model."""

    SPEC = AnalysisSpec('nl_hh', curves=True, params='HH')

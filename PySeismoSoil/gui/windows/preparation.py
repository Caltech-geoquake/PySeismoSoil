"""The "Input Files Preparation" windows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from PySeismoSoil import helper_site_response as sr
from PySeismoSoil.gui import analyses as an
from PySeismoSoil.gui import plots
from PySeismoSoil.gui.data_io import (
    G_ACCEL,
    InputError,
    check_data,
    load_matrix,
    save_matrix,
    site_code_from_filename,
    unique_path,
)
from PySeismoSoil.gui.navigation import Panel, register
from PySeismoSoil.gui.widgets import (
    FITTING_MESSAGE,
    BackgroundTask,
    RadioGroup,
    TaskContext,
    button,
    get_open_file,
    get_save_file,
    group_box,
    info,
    open_folder,
    show_figure,
    show_pyplot_figures,
    warn,
)

if TYPE_CHECKING:
    import pathlib

    from matplotlib.figure import Figure

PATIENCE_NOTE = '(Please be patient if the number of soil layers is large.)'
HH_X_NOTE = (
    'Note: The "curve" file should contain G/Gmax curves and damping curves, '
    'with the format described in the manual. However, this fitting routine '
    'only uses the damping information in the "curve" file. Without '
    'lab-measured damping info, the user is recommended to use the empirical '
    'formulas in Darendeli (2001) to calculate damping curves.'
)
PROFILE_COLUMNS = [
    'Thickness (m)',
    'Vs (m/s)',
    'Damping',
    'Density',
    'Material No. *',
]
DEFAULT_PI = 20.0


@register('preparation')
class PreparationPanel(Panel):
    """The "Input Files Preparation" panel (``SeismoSoil_Preparation.m``)."""

    def __init__(self) -> None:
        super().__init__(
            'Input Files Preparation',
            'SeismoSoil - Preparation',
            return_to='main',
            return_text='Return to main panel',
        )
        self.main_layout.addWidget(
            _big(
                'Manually enter Vs profile',
                lambda: self.open_sub_panel('vs_profile_generator'),
            ),
        )
        self.main_layout.addWidget(
            _big(
                'Modulus and damping (Darendeli, 2001)',
                lambda: self.open_sub_panel('modulus_damping'),
            ),
        )
        box = group_box('Nonlinear model parameter fitting')
        layout = QGridLayout(box)
        layout.addWidget(QLabel('MKZ model:'), 0, 0)
        layout.addWidget(
            _big(
                'MKZ ("H2" or "H4") curve fitting',
                lambda: self.open_sub_panel('mkz_curve_fitting'),
            ),
            1,
            0,
            1,
            2,
        )
        layout.addWidget(QLabel('Hybrid hyperbolic (HH) model:'), 2, 0)
        layout.addWidget(
            _big('HH_G', lambda: self.open_sub_panel('hh_g_fit')), 3, 0
        )
        layout.addWidget(
            _big('HH_x', lambda: self.open_sub_panel('hh_x_fit')), 3, 1
        )
        self.main_layout.addWidget(box)


def _big(text: str, slot: object) -> QPushButton:
    """Create a large push button."""
    btn = button(text, slot)
    btn.setMinimumHeight(36)
    font = btn.font()
    font.setPointSize(12)
    btn.setFont(font)
    return btn


def _bottom_row(panel: Panel) -> QHBoxLayout:
    """Create a row of the "Clear console" and "Close all figures" buttons."""
    row = QHBoxLayout()
    row.addStretch()
    row.addWidget(button('Clear console', panel.clear_console))
    row.addWidget(button('Close all figures', panel.close_all_figures))
    return row


# ----------------------------------------------------------------------------
# Vs profile generator
# ----------------------------------------------------------------------------
@register('vs_profile_generator')
class VsProfileGeneratorPanel(Panel):
    """The Vs profile generator (``SeismoSoil_Preparation_Vs_Profile.m``)."""

    def __init__(self) -> None:
        super().__init__(
            'Vs Profile Generator',
            'SeismoSoil - Vs Profile Generator',
            return_to='preparation',
        )
        grid = QGridLayout()
        grid.addWidget(_bold_label('Step 1. Define soil layers'), 0, 0)
        row = QHBoxLayout()
        row.addWidget(
            QLabel('Number of soil layers (excluding rock halfspace) = ')
        )
        self.n_layers = QSpinBox()
        self.n_layers.setRange(1, 10000)
        self.n_layers.setValue(5)
        self.n_layers.valueChanged.connect(self._resize_table)
        row.addWidget(self.n_layers)
        row.addStretch()
        grid.addLayout(row, 1, 0)
        self.soil_table = _table(5, PROFILE_COLUMNS)
        grid.addWidget(self.soil_table, 2, 0)
        grid.addWidget(
            QLabel('*) Material number is optional for linear analyses.'), 3, 0
        )
        grid.addWidget(_bold_label('Step 2. Define rock halfspace'), 4, 0)
        self.rock_table = _table(1, PROFILE_COLUMNS)
        self.rock_table.setMaximumHeight(70)
        for col, text in ((0, 'Infinity'), (4, 'N/A')):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.rock_table.setItem(0, col, item)

        grid.addWidget(self.rock_table, 5, 0)
        grid.addLayout(self._side_panel(), 0, 1, 6, 1)
        self.main_layout.addLayout(grid)

    def _side_panel(self) -> QVBoxLayout:
        side = QVBoxLayout()
        self.fourth_column = RadioGroup(
            '4th Column',
            [
                ('Density (kg/m^3)', 'density'),
                ('Unit weight (kN/m^3)', 'weight'),
            ],
        )
        self.damping_unit = RadioGroup(
            'Unit of damping ratio',
            [('1 (unity)', 1.0), ('% (percent)', 100.0)],
            default=1,
        )
        side.addWidget(self.fourth_column)
        side.addWidget(self.damping_unit)
        side.addWidget(
            button('Auto calculate damping/density', self.auto_calculate)
        )
        side.addWidget(button('Plot Vs profile', self._plot))
        side.addWidget(button('Save as text file', self._save))
        side.addStretch()
        return side

    def _resize_table(self) -> None:
        self.soil_table.setRowCount(self.n_layers.value())

    @staticmethod
    def _cell(table: QTableWidget, row: int, col: int) -> float:
        item = table.item(row, col)
        text = item.text().strip() if item is not None else ''
        if not text:
            return 0.0

        return float(text)

    def set_cell(
            self, row: int, col: int, value: float, *, rock: bool = False
    ) -> None:
        """
        Set a value in the table of soil layers (or of the rock).

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.
        value : float
            The value.
        rock : bool, default=False
            Whether to set the value in the table of the rock half space.
        """
        table = self.rock_table if rock else self.soil_table
        table.setItem(row, col, QTableWidgetItem(f'{value:.6g}'))

    def profile(self) -> np.ndarray:
        """
        Return the Vs profile (damping in 1, density in kg/m^3) in the tables.

        Returns
        -------
        np.ndarray
            The Vs profile (5 columns; the last row is the bedrock).

        Raises
        ------
        InputError
            When the tables have invalid values
        """
        n = self.soil_table.rowCount()
        try:
            layers = np.array([
                [self._cell(self.soil_table, i, j) for j in range(5)]
                for i in range(n)
            ])
            rock = [
                0.0,
                *(self._cell(self.rock_table, 0, j) for j in (1, 2, 3)),
                0.0,
            ]
        except ValueError as err:
            raise InputError(
                "You haven't finished entering all the properties correctly."
            ) from err

        profile = np.vstack((layers, rock))
        profile[:, 2] /= self.damping_unit.value()
        if self.fourth_column.value() == 'weight':
            profile[:, 3] *= 1000 / G_ACCEL

        return profile

    def auto_calculate(self) -> None:
        """Calculate the damping and density of all layers from Vs."""
        try:
            vs = np.array([
                self._cell(self.soil_table, i, 1)
                for i in range(self.soil_table.rowCount())
            ])
            rock_vs = self._cell(self.rock_table, 0, 1)
        except ValueError:
            warn(self, 'You need to enter both soil and rock Vs info.')
            return

        if np.any(vs <= 0) or rock_vs <= 0:
            warn(self, 'You need to enter both soil and rock Vs info.')
            return

        factor = self.damping_unit.value()
        xi, rho = sr.get_xi_rho(np.append(vs, rock_vs), formula_type=3)
        density_factor = (
            1.0 if self.fourth_column.value() == 'density' else G_ACCEL / 1000
        )
        for i in range(len(vs)):
            self.set_cell(i, 2, xi[i] * factor)
            self.set_cell(i, 3, rho[i] * density_factor)
            self.set_cell(i, 4, i + 1)

        self.set_cell(0, 2, xi[-1] * factor, rock=True)
        self.set_cell(0, 3, rho[-1] * density_factor, rock=True)

    def _plot(self) -> None:
        try:
            profile = self.profile()
        except InputError as err:
            warn(self, str(err), 'Error...')
            return

        show_figure(plots.vs_profile_figure(profile), 'Vs profile')

    def _save(self) -> None:
        try:
            profile = self.profile()
        except InputError as err:
            warn(self, str(err), 'Error...')
            return

        path = get_save_file(self, 'Save Vs profile as...', 'profile.txt')
        if path is not None:
            save_matrix(path, profile)
            print(f'Vs profile saved as {path}')


def _bold_label(text: str) -> QLabel:
    """Create a bold label."""
    label = QLabel(text)
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    return label


def _table(n_rows: int, headers: list[str]) -> QTableWidget:
    """Create an editable table."""
    table = QTableWidget(n_rows, len(headers))
    table.setHorizontalHeaderLabels(headers)
    header = table.horizontalHeader()
    header.setMinimumSectionSize(110)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setMinimumWidth(110 * len(headers) + 40)
    return table


# ----------------------------------------------------------------------------
# Modulus and damping curves (Darendeli, 2001), and MKZ curve fitting
# ----------------------------------------------------------------------------
class _FittingPanel(Panel):
    """Shared parts of the windows that fit the H2 and H4 parameters."""

    INPUT_KIND = 'curve'

    def __init__(self, title: str, window_title: str) -> None:
        super().__init__(title, window_title, return_to='preparation')
        self.input_dir: pathlib.Path | None = None
        self.site_code = ''
        self.curve: np.ndarray | None = None

    def _file_panel(self, prompt: str, slot: object) -> QGroupBox:
        box = QGroupBox()
        layout = QGridLayout(box)
        layout.addWidget(QLabel(prompt), 0, 0)
        layout.addWidget(button('Select', slot), 0, 1)
        layout.addWidget(QLabel('Edit name of profile (optional):'), 1, 0)
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(
            lambda: setattr(self, 'site_code', self.name_edit.text().strip())
        )
        layout.addWidget(self.name_edit, 1, 1)
        return box

    def _fitting_panel(self) -> QGroupBox:
        box = QGroupBox()
        layout = QGridLayout(box)
        layout.addWidget(QLabel('Do H2 or H4 parameter fitting'), 0, 0, 1, 2)
        layout.addWidget(
            button('H2 fitting && save', lambda: self.fit('H2')), 1, 0
        )
        layout.addWidget(
            button('H4 fitting && save', lambda: self.fit('H4')), 1, 1
        )
        layout.addWidget(QLabel(PATIENCE_NOTE), 2, 0, 1, 2)
        self.show_fits = QCheckBox(
            'Show figures of fit results for each layer'
        )
        self.show_fits.setChecked(True)
        layout.addWidget(self.show_fits, 3, 0, 1, 2)
        return box

    def _set_input(self, path: pathlib.Path, prefix: str) -> None:
        self.input_dir = path.parent
        stem = path.stem
        self.site_code = (
            stem.split(prefix, 1)[1] if stem.startswith(prefix) else stem
        )
        self.name_edit.setText(self.site_code)

    def fit(self, model: str) -> BackgroundTask | None:
        """
        Fit the H2 or H4 parameters, and save them to files.

        Parameters
        ----------
        model : str
            "H2" or "H4".

        Returns
        -------
        BackgroundTask | None
            The background task of the fitting (``None`` if the curves are not
            ready).
        """
        if self.curve is None or self.input_dir is None:
            warn(
                self,
                'You need to calculate (or select) G/Gmax and damping curves '
                'first.',
                'Error...',
            )
            return None

        curve = self.curve.copy()
        out_dir = self.input_dir
        code = self.site_code

        def task(ctx: TaskContext) -> list[Figure]:
            ctx.progress(0, f'{model} curve fitting...')
            if model == 'H2':
                params, figures = an.fit_H2(
                    curve, progress_callback=ctx.progress
                )
                path = unique_path(out_dir / f'H2_n_{code}.txt')
                save_matrix(path, params)
                print(f'H2_n file saved as {path}')
                return figures

            G_param, figures = an.fit_H4_G(curve)
            xi_param, figures_x = an.fit_damping_params(
                curve, 'H4', progress_callback=ctx.progress
            )
            for name, params in (('H4_G', G_param), ('H4_x', xi_param)):
                path = unique_path(out_dir / f'{name}_{code}.txt')
                save_matrix(path, params)
                print(f'{name} file saved as {path}')

            return figures + figures_x

        def done(figures: list[Figure]) -> None:
            if self.show_fits.isChecked():
                for fig in figures:
                    show_figure(fig, f'{model} curve fitting')

            info(self, 'Finished', f'{model} curve fitting complete!')
            open_folder(out_dir)

        return BackgroundTask(
            self,
            task,
            on_done=done,
            message=FITTING_MESSAGE,
        ).start()


@register('modulus_damping')
class ModulusDampingPanel(_FittingPanel):
    """Darendeli curves (``SeismoSoil_Preparation_Modulus_Damping.m``)."""

    def __init__(self) -> None:
        super().__init__(
            'Modulus and damping curves (Darendeli, 2001)',
            'SeismoSoil - Modulus and Damping',
        )
        self.profile: np.ndarray | None = None
        self.main_layout.addWidget(
            self._file_panel(
                'Please select soil profile file:', self._select_profile
            )
        )
        box = QGroupBox()
        layout = QGridLayout(box)
        layout.addWidget(
            QLabel('Click to calculate G/Gmax and damping ratio curves'),
            0,
            0,
            1,
            3,
        )
        layout.addWidget(QLabel('Plasticity Index ='), 1, 0)
        self.PI_edit = QLineEdit(f'{DEFAULT_PI:g}')
        self.PI_edit.setMaximumWidth(80)
        layout.addWidget(self.PI_edit, 1, 1)
        layout.addWidget(QLabel('%'), 1, 2)
        layout.addWidget(
            button('Calculate && save', self.calculate_curves), 1, 3
        )
        self.main_layout.addWidget(box)
        self.main_layout.addWidget(self._fitting_panel())
        self.main_layout.addLayout(_bottom_row(self))

    def load_profile(self, path: pathlib.Path) -> bool:
        """
        Load a Vs profile.

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
        self.curve = None
        self._set_input(path, 'profile_')
        show_figure(plots.vs_profile_figure(profile, path.name), path.name)
        return True

    def _select_profile(self) -> None:
        path = get_open_file(self, 'Select soil profile file...')
        if path is not None:
            self.load_profile(path)

    def calculate_curves(self) -> pathlib.Path | None:
        """
        Calculate the curves of every layer, and save them as a "curve" file.

        Returns
        -------
        pathlib.Path | None
            The "curve" file (``None`` if the inputs are not ready).
        """
        if self.profile is None or self.input_dir is None:
            warn(
                self, 'You need to select a Vs profile file first.', 'Error...'
            )
            return None

        try:
            PI = float(self.PI_edit.text())
        except ValueError:
            warn(self, 'Plasticity index should be a number.', 'Error...')
            return None

        self.curve = an.darendeli_curves(self.profile, PI)
        path = unique_path(self.input_dir / f'curve_{self.site_code}.txt')
        save_matrix(path, self.curve)
        print(f'"curve" file saved as {path}')
        show_figure(plots.curves_figure(self.curve, path.name), path.name)
        open_folder(self.input_dir)
        return path


@register('mkz_curve_fitting')
class MKZCurveFittingPanel(_FittingPanel):
    """H2 and H4 curve fitting (``SeismoSoil_Preparation_H2_H4_...m``)."""

    def __init__(self) -> None:
        super().__init__(
            'H2 and H4 Curve Fitting', 'SeismoSoil - MKZ Curve Fitting'
        )
        self.main_layout.addWidget(
            self._file_panel('Please select "curve" file:', self._select_curve)
        )
        self.main_layout.addWidget(self._fitting_panel())
        self.main_layout.addLayout(_bottom_row(self))

    def load_curve(self, path: pathlib.Path) -> bool:
        """
        Load a "curve" file.

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
            curve = load_matrix(path)
            check_data(curve, 'curve', path.name)
        except (OSError, InputError) as err:
            warn(self, str(err))
            return False

        self.curve = curve
        self._set_input(path, 'curve_')
        return True

    def _select_curve(self) -> None:
        path = get_open_file(self, 'Select "curve" data file...')
        if path is not None:
            self.load_curve(path)


# ----------------------------------------------------------------------------
# HH model parameters
# ----------------------------------------------------------------------------
@register('hh_g_fit')
class HHGFitPanel(Panel):
    """HH_G parameters (``SeismoSoil_Preparation_HH_G_fit.m``)."""

    def __init__(self) -> None:
        super().__init__(
            'Hybrid hyperbolic (HH) model: HH_G parameters',
            'SeismoSoil - HH_G parameters',
            return_to='preparation',
        )
        self.profile: np.ndarray | None = None
        self.profile_path: pathlib.Path | None = None
        self.curve: np.ndarray | None = None
        self.tau_max: np.ndarray | None = None
        row = QHBoxLayout()
        row.addWidget(self._step1())
        right = QVBoxLayout()
        right.addWidget(self._step2())
        self.show_fig = QCheckBox('Show optimization figures')
        self.save_fig = QCheckBox('Save optimization figures')
        self.save_fig.setChecked(True)
        right.addWidget(self.show_fig)
        right.addWidget(self.save_fig)
        right.addWidget(_big('Calculate HH_G parameters', self.calculate))
        right.addLayout(_bottom_row(self))
        row.addLayout(right)
        self.main_layout.addLayout(row)
        self._update_enabled()

    def _step1(self) -> QGroupBox:
        box = group_box('Step 1: Small-strain G/Gmax')
        layout = QVBoxLayout(box)
        self.source = RadioGroup(
            'Small-strain G/Gmax data source',
            [
                (
                    (
                        '(A) Calculate from Vs profile using Darendeli (2001) '
                        'empirical formulas'
                    ),
                    'profile',
                ),
                ('(B) Provided from external "curve" file', 'curve'),
            ],
        )
        self.source.changed.connect(self._update_enabled)
        layout.addWidget(self.source)
        self.box_a = QGroupBox('Step 1(A) Calculate from Vs profile')
        a = QVBoxLayout(self.box_a)
        a.addWidget(button('Choose Vs profile', self._choose_profile))
        a.addWidget(button('Plot profile', self._plot_profile))
        self.box_b = QGroupBox('Step 1(B) External "curve" file')
        b = QVBoxLayout(self.box_b)
        b.addWidget(button('Choose a "curve" file', self._choose_curve))
        b.addWidget(button('Choose a Vs profile', self._choose_profile))
        layout.addWidget(self.box_a)
        layout.addWidget(self.box_b)
        return box

    def _step2(self) -> QGroupBox:
        box = group_box('Step 2: Shear strength data')
        layout = QVBoxLayout(box)
        self.strength_source = RadioGroup(
            'Shear strength data source',
            [
                ('Calculate from Ladd (1991)', 'ladd'),
                ('Import from file', 'file'),
            ],
        )
        self.strength_source.changed.connect(self._update_enabled)
        layout.addWidget(self.strength_source)
        self.box_tau = QGroupBox('External Tau_max data file')
        QVBoxLayout(self.box_tau).addWidget(
            button('Choose a "Tau_max" file', self._choose_tau_max)
        )
        layout.addWidget(self.box_tau)
        return box

    def _update_enabled(self) -> None:
        from_profile = self.source.value() == 'profile'
        self.box_a.setEnabled(from_profile)
        self.box_b.setEnabled(not from_profile)
        self.box_tau.setEnabled(self.strength_source.value() == 'file')

    def load_profile(self, path: pathlib.Path) -> bool:
        """
        Load a Vs profile.

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
        self.profile_path = path
        return True

    def _choose_profile(self) -> None:
        path = get_open_file(self, 'Select Vs profile data file...')
        if path is not None:
            self.load_profile(path)

    def _plot_profile(self) -> None:
        if self.profile is None:
            warn(self, 'Please choose a Vs profile first.')
            return

        name = self.profile_path.name if self.profile_path else ''
        show_figure(plots.vs_profile_figure(self.profile, name), name)

    def _choose_curve(self) -> None:
        path = get_open_file(self, 'Select "curve" data file...')
        if path is None:
            return

        try:
            curve = load_matrix(path)
            check_data(curve, 'curve', path.name)
        except (OSError, InputError) as err:
            warn(self, str(err))
            return

        self.curve = curve

    def _choose_tau_max(self) -> None:
        path = get_open_file(self, 'Select "Tau_max" data file...')
        if path is None:
            return

        try:
            tau = load_matrix(path)
            check_data(tau, 'tau_max', path.name)
        except (OSError, InputError) as err:
            warn(self, str(err))
            return

        self.tau_max = tau.ravel()

    def calculate(self) -> BackgroundTask | None:
        """
        Calculate the HH_G parameters, and save them to a file.

        Returns
        -------
        BackgroundTask | None
            The background task (``None`` if the inputs are not ready).
        """
        use_curve = self.source.value() == 'curve'
        use_file = self.strength_source.value() == 'file'
        if self.profile is None or self.profile_path is None:
            warn(self, 'Please choose a Vs profile.')
            return None

        if use_curve and self.curve is None:
            warn(self, 'Please choose a "curve" file.')
            return None

        if use_file and self.tau_max is None:
            warn(self, 'Please choose a "Tau_max" file.')
            return None

        profile = self.profile.copy()
        curve = self.curve.copy() if use_curve else None
        tau_max = self.tau_max if use_file else None
        out_dir = self.profile_path.parent
        code = site_code_from_filename(self.profile_path.name)
        fig_dir = (
            out_dir / f'HH_G_fit_figures_{code}'
            if self.save_fig.isChecked()
            else None
        )

        def task(ctx: TaskContext) -> pathlib.Path:
            ctx.progress(0, 'Calculating HH_G parameters...')
            if fig_dir is not None:
                fig_dir.mkdir(parents=True, exist_ok=True)

            params = an.fit_HH_G(
                profile,
                curve=curve,
                tau_max=tau_max,
                fig_output_dir=None if fig_dir is None else str(fig_dir),
                profile_name=code,
            )
            path = unique_path(out_dir / f'HH_G_{code}.txt')
            save_matrix(path, params)
            print(f'HH_G file saved as: {path}')
            return path

        def done(path: pathlib.Path) -> None:
            show_pyplot_figures(show=self.show_fig.isChecked())
            info(self, 'Finished', f'HH_G parameters saved as:\n{path}')
            open_folder(path.parent)

        return BackgroundTask(self, task, on_done=done).start()


@register('hh_x_fit')
class HHxFitPanel(Panel):
    """HH_x parameters (``SeismoSoil_Preparation_HH_x_fit.m``)."""

    def __init__(self) -> None:
        super().__init__(
            'Hybrid hyperbolic (HH) model: HH_x parameters',
            'SeismoSoil - HH_x parameters',
            return_to='preparation',
        )
        self.curve: np.ndarray | None = None
        self.curve_path: pathlib.Path | None = None
        row = QHBoxLayout()
        left = QVBoxLayout()
        box = QGroupBox('Import damping data')
        QVBoxLayout(box).addWidget(
            button('Choose a "curve" file', self._choose_curve)
        )
        left.addWidget(box)
        note = QLabel(HH_X_NOTE)
        note.setWordWrap(True)
        left.addWidget(note)
        right = QVBoxLayout()
        self.show_fig = QCheckBox('Show optimization figures')
        self.save_fig = QCheckBox('Save figures')
        self.save_fig.setChecked(True)
        right.addWidget(self.show_fig)
        right.addWidget(self.save_fig)
        self.calc_button = _big('HH_x parameters', self.calculate)
        self.calc_button.setEnabled(False)
        right.addWidget(self.calc_button)
        right.addLayout(_bottom_row(self))
        row.addLayout(left)
        row.addLayout(right)
        self.main_layout.addLayout(row)

    def load_curve(self, path: pathlib.Path) -> bool:
        """
        Load a "curve" file.

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
            curve = load_matrix(path)
            check_data(curve, 'curve', path.name)
        except (OSError, InputError) as err:
            warn(self, str(err))
            return False

        self.curve = curve
        self.curve_path = path
        self.calc_button.setEnabled(True)
        return True

    def _choose_curve(self) -> None:
        path = get_open_file(self, 'Select "curve" data file...')
        if path is not None:
            self.load_curve(path)

    def calculate(
            self, *, pop_size: int = 800, n_gen: int = 100
    ) -> BackgroundTask | None:
        """
        Fit the HH_x parameters, and save them to a file.

        Parameters
        ----------
        pop_size : int, default=800
            Population size of the genetic algorithm.
        n_gen : int, default=100
            Number of generations of the genetic algorithm.

        Returns
        -------
        BackgroundTask | None
            The background task (``None`` if the inputs are not ready).
        """
        if self.curve is None or self.curve_path is None:
            warn(self, 'Please choose a "curve" file first.')
            return None

        curve = self.curve.copy()
        out_dir = self.curve_path.parent
        stem = self.curve_path.stem
        code = (
            stem.split('curve_', 1)[1] if stem.startswith('curve_') else stem
        )
        save_fig = self.save_fig.isChecked()

        def task(ctx: TaskContext) -> list[Figure]:
            ctx.progress(0, 'Fitting HH_x parameters...')
            params, figures = an.fit_damping_params(
                curve,
                'HH',
                pop_size=pop_size,
                n_gen=n_gen,
                progress_callback=ctx.progress,
            )
            path = unique_path(out_dir / f'HH_X_{code}.txt')
            save_matrix(path, params)
            print(f'HH_x file saved as: {path}')
            if save_fig:
                for k, fig in enumerate(figures):
                    fig.savefig(
                        out_dir / f'HH_x_fit_{code}_material_{k + 1}.png'
                    )

            return figures

        def done(figures: list[Figure]) -> None:
            if self.show_fig.isChecked():
                for fig in figures:
                    show_figure(fig, 'HH_x curve fitting')

            info(self, 'Finished', 'HH_x curve fitting complete!')
            open_folder(out_dir)

        return BackgroundTask(
            self,
            task,
            on_done=done,
            message=FITTING_MESSAGE,
        ).start()

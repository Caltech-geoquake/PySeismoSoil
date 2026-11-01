import functools
import io as std_io
import os
import pathlib
import shutil
import sys

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
# Skip these tests if PySide6 (or a system library that it needs) is missing
pytest.importorskip('PySide6.QtWidgets')
pytest.importorskip('pytestqt')

from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from pytestqt.qtbot import QtBot

from PySeismoSoil.gui import analyses as an
from PySeismoSoil.gui import app as gui_app
from PySeismoSoil.gui import navigation
from PySeismoSoil.gui.data_io import load_matrix, save_matrix
from PySeismoSoil.gui.widgets import (
    BackgroundTask,
    Console,
    FigureWindow,
)
from PySeismoSoil.gui.windows.analysis import AnalysisPanel

f_dir = pathlib.Path(__file__).parent / 'files'
TIMEOUT_MS = 300_000


class Dialogs:
    """Record the messages that the GUI shows, instead of showing them."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.opened: list[str] = []
        self.answer = QMessageBox.StandardButton.Yes
        self.directory = ''
        self.save_name = ''

    def text(self) -> str:
        return '\n'.join(self.messages)


@pytest.fixture
def gui(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> Dialogs:
    assert gui_app.get_app() is qapp
    dialogs = Dialogs()

    def record(*args: object, **_: object) -> QMessageBox.StandardButton:
        dialogs.messages.append(str(args[2]))
        return QMessageBox.StandardButton.Ok

    def question(*args: object, **_: object) -> QMessageBox.StandardButton:
        dialogs.messages.append(str(args[2]))
        return dialogs.answer

    monkeypatch.setattr(QMessageBox, 'warning', staticmethod(record))
    monkeypatch.setattr(QMessageBox, 'information', staticmethod(record))
    monkeypatch.setattr(QMessageBox, 'question', staticmethod(question))
    monkeypatch.setattr(
        QMessageBox, 'open', lambda box: record(0, 0, box.text())
    )
    monkeypatch.setattr(
        QDesktopServices,
        'openUrl',
        staticmethod(lambda url: dialogs.opened.append(url.toLocalFile())),
    )
    monkeypatch.setattr(
        QFileDialog,
        'getExistingDirectory',
        staticmethod(lambda *_: dialogs.directory),
    )
    monkeypatch.setattr(
        QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *_: (dialogs.save_name, '')),
    )
    # Faster genetic algorithms
    monkeypatch.setattr(
        an,
        'fit_damping_params',
        functools.partial(an.fit_damping_params, pop_size=20, n_gen=5),
    )
    yield dialogs
    for window in navigation.open_windows():
        window.close()

    FigureWindow.close_all()


def _wait(qtbot: QtBot, task: BackgroundTask | None) -> None:
    assert task is not None
    qtbot.waitUntil(lambda: task.done, timeout=TIMEOUT_MS)
    assert task.error is None


def _wait_all(qtbot: QtBot) -> None:
    tasks = BackgroundTask.running()
    assert tasks
    for task in tasks:
        _wait(qtbot, task)


def _short_motion(
        tmp_path: pathlib.Path, name: str, n_pts: int = 1200
) -> pathlib.Path:
    data = load_matrix(f_dir / 'Accel_02_weak_motion_(unit_gal).txt')[:n_pts]
    path = tmp_path / name
    save_matrix(path, data, precision=8)
    return path


def _copy(
        name: str, tmp_path: pathlib.Path, new_name: str | None = None
) -> pathlib.Path:
    return pathlib.Path(
        shutil.copy(f_dir / name, tmp_path / (new_name or name))
    )


# ----------------------------------------------------------------------------
# All windows, navigation, and the console
# ----------------------------------------------------------------------------
def test_all_windows_open(gui: Dialogs) -> None:
    names = navigation.registered_names()
    assert len(names) == 29
    for name in names:
        window = navigation.open_window(name)
        assert window.isVisible()
        assert window.windowTitle()
        window.close()

    assert not gui.messages


@pytest.mark.usefixtures('gui')
def test_navigation() -> None:
    main = navigation.open_window('main')
    tools = main.open_sub_panel('tools')
    assert tools in navigation.open_windows()
    assert main not in navigation.open_windows()
    filters = tools.open_sub_panel('tool_signal_filters')
    filters.go_back()
    assert type(navigation.open_windows()[-1]).__name__ == 'ToolsPanel'
    navigation.open_windows()[-1].go_back()
    assert type(navigation.open_windows()[-1]).__name__ == 'MainPanel'


@pytest.mark.usefixtures('gui')
def test_console(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'stdout', std_io.StringIO())
    console = Console.instance()
    console.redirect_std_streams()
    console.clear()
    print('hello console')
    qtbot.waitUntil(lambda: 'hello console' in console.text())
    console.clear()
    assert console.text() == ''


# ----------------------------------------------------------------------------
# Analysis windows
# ----------------------------------------------------------------------------
def _setup_hh(
        window: AnalysisPanel, tmp_path: pathlib.Path, n_motions: int = 1
) -> None:
    assert window.load_profile(f_dir / 'profile_FKSH14.txt')
    assert window.site_code == 'FKSH14'
    assert window.load_curve(f_dir / 'curve_FKSH14.txt')
    assert window.load_param(f_dir / 'HH_G_FKSH14.txt', 'G')
    assert window.load_param(f_dir / 'HH_X_FKSH14.txt', 'x')
    paths = [
        _short_motion(tmp_path, f'motion_{k}.txt') for k in range(n_motions)
    ]
    assert window.load_motions(paths) == n_motions
    window.accel_unit.set_index(1)  # gal
    window.output_dir = tmp_path / 'results'


def test_analysis_hh(
        gui: Dialogs, qtbot: QtBot, tmp_path: pathlib.Path
) -> None:
    window = navigation.open_window('analysis_nl_hh')
    _setup_hh(window, tmp_path)
    window.view_popup.setChecked(True)
    n_figs = len(FigureWindow.open_windows())
    _wait(qtbot, window.start())
    out = tmp_path / 'results' / 'motion_0'
    assert (out / 'motion_0_accel_on_surface.txt').exists()
    assert (out / 'motion_0_nonlinear_TF_smoothed.txt').exists()
    assert (out / 'motion_0_max_a_v_d_gamma_tau.png').exists()
    assert 'Simulation finished' in gui.text()
    assert len(FigureWindow.open_windows()) == n_figs + 3
    assert str((tmp_path / 'results').resolve()) in gui.opened


@pytest.mark.usefixtures('gui')
def test_analysis_parallel_runs(qtbot: QtBot, tmp_path: pathlib.Path) -> None:
    window = navigation.open_window('analysis_nl_hh')
    _setup_hh(window, tmp_path, n_motions=3)
    window.use_parallel.setChecked(True)
    window.open_when_done.setChecked(False)
    _wait(qtbot, window.start())
    results = [
        np.genfromtxt(
            tmp_path
            / 'results'
            / f'motion_{k}'
            / f'motion_{k}_accel_on_surface.txt'
        )
        for k in range(3)
    ]
    # The same motion gives the same results in every thread
    assert np.allclose(results[0], results[1])
    assert np.allclose(results[0], results[2])


def test_analysis_incomplete_steps(gui: Dialogs) -> None:
    window = navigation.open_window('analysis_nl_h4')
    assert window.start() is None
    assert "haven't finished step(s) 1, 2, 3, 4" in gui.text()


def test_analysis_boundary_checks(
        gui: Dialogs, tmp_path: pathlib.Path
) -> None:
    window = navigation.open_window('analysis_eql')
    window.load_profile(f_dir / 'profile_FKSH14.txt')
    window.load_curve(f_dir / 'curve_FKSH14.txt')
    window.load_motions([_short_motion(tmp_path, 'm.txt')])
    window.output_dir = tmp_path / 'out'
    window.motion_type.set_index(1)  # borehole, on elastic bedrock
    assert window.start() is None
    assert an.BOREHOLE_ON_ELASTIC_ERROR in gui.text()
    window.motion_type.set_index(2)  # outcrop: ask the user
    gui.answer = QMessageBox.StandardButton.No
    assert window.start() is None
    assert 'Use With Caution' in gui.text()


def test_analysis_invalid_inputs(gui: Dialogs, tmp_path: pathlib.Path) -> None:
    window = navigation.open_window('analysis_nl_hh')
    _setup_hh(window, tmp_path)
    window.xi_param = window.xi_param[:, :3]
    assert window.start() is None
    assert 'dimensions do not match' in gui.text()
    bad = tmp_path / 'bad_motion.txt'
    bad.write_text('0 1\n0 2\n')
    assert window.load_motions([bad]) == 0
    assert 'monotonically' in gui.text()


def test_analysis_damping_in_percent(
        gui: Dialogs, qtbot: QtBot, tmp_path: pathlib.Path
) -> None:
    profile = load_matrix(f_dir / 'profile_FKSH14.txt')
    profile[:, 2] *= 100
    profile[:, 3] /= 1000
    path = tmp_path / 'profile_PCT.txt'
    save_matrix(path, profile)
    window = navigation.open_window('analysis_linear_fd')
    window.load_profile(path)  # warns about the unit of damping
    assert 'Unit of damping' in gui.text()
    window.xi_unit.set_index(1)  # %
    window.rho_unit.set_index(1)  # g/cm^3
    window.load_motions([_short_motion(tmp_path, 'm.txt')])
    window.output_dir = tmp_path / 'out'
    _wait(qtbot, window.start())
    assert (tmp_path / 'out' / 'm' / 'm_linear_TF.txt').exists()


def test_linear_tf_buttons(gui: Dialogs, tmp_path: pathlib.Path) -> None:
    window = navigation.open_window('analysis_linear_fd')
    window.load_profile(f_dir / 'profile_FKSH14.txt')
    n_figs = len(FigureWindow.open_windows())
    window._plot_linear_tf()
    assert len(FigureWindow.open_windows()) == n_figs + 1
    gui.directory = str(tmp_path)
    window.bedrock.setCurrentIndex(1)  # rigid
    window._save_linear_tf()
    assert (tmp_path / 'FKSH14_linear_amplification_factor_real.dat').exists()
    assert (
        tmp_path / 'FKSH14_linear_amplification_factor_complex.dat'
    ).exists()


def test_plot_and_baseline_buttons(
        gui: Dialogs, tmp_path: pathlib.Path
) -> None:
    window = navigation.open_window('analysis_eql')
    window._plot_all_motions()
    assert "haven't selected any motions" in gui.text()
    window.load_profile(f_dir / 'profile_FKSH14.txt')
    window.load_curve(f_dir / 'curve_FKSH14.txt')
    window.load_motions([
        _short_motion(tmp_path, f'm{k}.txt') for k in range(2)
    ])
    n_figs = len(FigureWindow.open_windows())
    window._plot_profile()
    window._plot_curves()
    window._plot_all_motions()
    window.motion_list.setCurrentRow(1)
    window._plot_selected_motions()
    assert len(FigureWindow.open_windows()) == n_figs + 5
    original = window.motions[1].copy()
    window._baseline_correct(all_=False)
    assert window.corrected == [False, True]
    assert not np.allclose(window.motions[1], original)
    window._baseline_correct(all_=True)
    assert window.corrected == [True, True]
    window._baseline_correct(all_=True)
    assert 'will not be processed again' in gui.text()


def test_fit_params_in_analysis_window(
        gui: Dialogs, qtbot: QtBot, tmp_path: pathlib.Path
) -> None:
    window = navigation.open_window('analysis_nl_h4')
    window.load_profile(f_dir / 'profile_FKSH14.txt')
    window._fit_params('G')
    assert 'select a "curve" file first' in gui.text()
    window.load_curve(f_dir / 'curve_FKSH14.txt')
    window._fit_params('G')
    _wait_all(qtbot)
    window._fit_params('x')
    _wait_all(qtbot)
    assert window.G_param.shape == (4, 5)
    assert window.xi_param.shape == (4, 5)
    assert np.all(window.G_param[1] == 0)
    window.load_motions([_short_motion(tmp_path, 'm.txt')])
    window.output_dir = tmp_path / 'out'
    _wait(qtbot, window.start())


def test_epp_shear_strength(
        gui: Dialogs, qtbot: QtBot, tmp_path: pathlib.Path
) -> None:
    window = navigation.open_window('analysis_nl_epp')
    window._calc_tau_max()
    assert 'select a Vs profile first' in gui.text()
    window.load_profile(f_dir / 'profile_FKSH14.txt')
    window._calc_tau_max()
    assert window.tau_max.shape == (5,)
    tau_file = tmp_path / 'Tau_max.txt'
    save_matrix(tau_file, window.tau_max[:, None] * 2)
    assert window.load_tau_max(tau_file)
    window.load_motions([_short_motion(tmp_path, 'm.txt')])
    window.output_dir = tmp_path / 'out'
    _wait(qtbot, window.start())
    assert (tmp_path / 'out' / 'm' / 'm_nonlinear_TF_raw.txt').exists()


# ----------------------------------------------------------------------------
# Preparation windows
# ----------------------------------------------------------------------------
def test_vs_profile_generator(gui: Dialogs, tmp_path: pathlib.Path) -> None:
    window = navigation.open_window('vs_profile_generator')
    window.n_layers.setValue(2)
    window.auto_calculate()
    assert 'enter both soil and rock Vs' in gui.text()
    for row, (h, vs) in enumerate([(5, 150), (10, 400)]):
        window.set_cell(row, 0, h)
        window.set_cell(row, 1, vs)

    window.set_cell(0, 1, 1200, rock=True)
    window.auto_calculate()
    profile = window.profile()
    assert profile.shape == (3, 5)
    assert np.allclose(profile[:, 1], [150, 400, 1200])
    assert np.all(profile[:, 2] < 1)  # converted from %
    assert np.allclose(profile[:-1, 4], [1, 2])
    gui.save_name = str(tmp_path / 'profile_new.txt')
    window._save()
    assert np.allclose(np.genfromtxt(tmp_path / 'profile_new.txt'), profile)
    window.fourth_column.set_index(1)  # unit weight
    window.set_cell(0, 3, 18.0)
    assert np.isclose(window.profile()[0, 3], 18.0 * 1000 / 9.81)


def test_modulus_damping_and_fitting(
        gui: Dialogs, qtbot: QtBot, tmp_path: pathlib.Path
) -> None:
    window = navigation.open_window('modulus_damping')
    assert window.calculate_curves() is None
    path = _copy('profile_FKSH14.txt', tmp_path)
    assert window.load_profile(path)
    assert window.site_code == 'FKSH14'
    curve_path = window.calculate_curves()
    assert curve_path == tmp_path / 'curve_FKSH14.txt'
    assert load_matrix(curve_path).shape == (50, 20)
    window.curve = window.curve[:, :8]  # only 2 layers, to be faster
    _wait(qtbot, window.fit('H2'))
    _wait(qtbot, window.fit('H4'))
    for name in ('H2_n', 'H4_G', 'H4_x'):
        assert load_matrix(tmp_path / f'{name}_FKSH14.txt').shape == (4, 2)

    assert 'H4 curve fitting complete' in gui.text()


@pytest.mark.usefixtures('gui')
def test_mkz_curve_fitting(qtbot: QtBot, tmp_path: pathlib.Path) -> None:
    window = navigation.open_window('mkz_curve_fitting')
    curve = load_matrix(f_dir / 'curve_FKSH14.txt')[:, :4]
    path = tmp_path / 'curve_SITE.txt'
    save_matrix(path, curve)
    assert window.load_curve(path)
    window.show_fits.setChecked(False)
    _wait(qtbot, window.fit('H2'))
    assert load_matrix(tmp_path / 'H2_n_SITE.txt').shape == (4, 1)


def test_hh_g_fit(gui: Dialogs, qtbot: QtBot, tmp_path: pathlib.Path) -> None:
    window = navigation.open_window('hh_g_fit')
    assert window.calculate() is None
    path = _copy('profile_FKSH14.txt', tmp_path)
    window.load_profile(path)
    window.show_fig.setChecked(True)
    n_figs = len(FigureWindow.open_windows())
    _wait(qtbot, window.calculate())
    params = load_matrix(tmp_path / 'HH_G_FKSH14.txt')
    assert params.shape == (9, 5)
    assert list((tmp_path / 'HH_G_fit_figures_FKSH14').glob('*.png'))
    assert len(FigureWindow.open_windows()) > n_figs
    window.source.set_index(1)  # from a "curve" file
    assert window.calculate() is None
    assert 'choose a "curve" file' in gui.text()


@pytest.mark.usefixtures('gui')
def test_hh_x_fit(qtbot: QtBot, tmp_path: pathlib.Path) -> None:
    window = navigation.open_window('hh_x_fit')
    assert not window.calc_button.isEnabled()
    curve = load_matrix(f_dir / 'curve_FKSH14.txt')[:, :8]
    path = tmp_path / 'curve_SITE.txt'
    save_matrix(path, curve)
    window.load_curve(path)
    assert window.calc_button.isEnabled()
    _wait(qtbot, window.calculate(pop_size=20, n_gen=5))
    assert load_matrix(tmp_path / 'HH_X_SITE.txt').shape == (9, 2)
    assert (tmp_path / 'HH_x_fit_SITE_material_2.png').exists()


# ----------------------------------------------------------------------------
# Post processing
# ----------------------------------------------------------------------------
def test_post_processing(
        gui: Dialogs, qtbot: QtBot, tmp_path: pathlib.Path
) -> None:
    job = an.AnalysisJob(
        method='linear_td',
        vs_profile=load_matrix(f_dir / 'profile_FKSH14.txt'),
        setting=an.resolve_boundary('elastic', 'incident', time_domain=True),
        output_dir=tmp_path,
        accel_factor=100,
    )
    motion = load_matrix(f_dir / 'Accel_02_weak_motion_(unit_gal).txt')[:300]
    an.run_motion(job, motion, 'quake.txt')
    window = navigation.open_window('post_processing')
    window.plot_loops(save=False)
    assert 'not selected a folder' in gui.text()
    assert not window.set_results_dir(tmp_path)
    assert window.set_results_dir(tmp_path / 'quake')
    assert window.event_name == 'quake'
    window.n_subplots.setText('25')
    figures = window.plot_loops(save=True)
    n_layers = load_matrix(
        tmp_path / 'quake' / 'quake_time_history_strain.txt'
    ).shape[1]
    assert len(figures) == int(np.ceil(n_layers / 25))
    assert list((tmp_path / 'quake' / 'Stress strain loops').glob('*.png'))
    window.speed.setText('10')
    movie = window.view_movie(save=True)
    assert movie is not None
    _wait_all(qtbot)
    assert (tmp_path / 'quake' / 'quake.gif').exists()
    window.speed.setText('-1')
    assert window.view_movie(save=False) is None


# ----------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------
def _figure_count() -> int:
    return len(FigureWindow.open_windows())


def test_motion_plotter(gui: Dialogs, tmp_path: pathlib.Path) -> None:
    window = navigation.open_window('tool_motion_plotter')
    window.run(selected=False)
    assert "haven't selected any files" in gui.text()
    paths = [_short_motion(tmp_path, f'm{k}.txt') for k in range(3)]
    assert window.load_files(paths) == 3
    window.show_rms.setChecked(True)
    window.arias.setChecked(True)
    n_figs = _figure_count()
    window.run(selected=False)
    window.plot_histogram()
    assert _figure_count() == n_figs + 3 * 2 + 1


@pytest.mark.usefixtures('gui')
def test_profile_plotters() -> None:
    for name in ('tool_vs_profile_plotter', 'tool_linear_ampl_factor'):
        window = navigation.open_window(name)
        window.load_files([
            f_dir / 'profile_FKSH14.txt',
            f_dir / 'profile_P001.txt',
        ])
        n_figs = _figure_count()
        window.file_list.setCurrentRow(1)
        window.run(selected=True)
        assert _figure_count() == n_figs + 1


@pytest.mark.usefixtures('gui')
@pytest.mark.parametrize('filter_index', [0, 1, 2, 3])
def test_signal_filters(tmp_path: pathlib.Path, filter_index: int) -> None:
    window = navigation.open_window('tool_signal_filters')
    path = _short_motion(tmp_path, 'm.txt')
    window.load_files([path])
    window.filter_type.set_index(filter_index)
    window.fc_low.setText('0.5')
    window.fc_high.setText('10')
    window.run(selected=False, save=True)
    kind = window.filter_type.value()
    result = load_matrix(tmp_path / f'm_{kind}_filtered.txt')
    assert result.shape == load_matrix(path).shape


def test_signal_filters__invalid_cutoffs(
        gui: Dialogs, tmp_path: pathlib.Path
) -> None:
    window = navigation.open_window('tool_signal_filters')
    window.load_files([_short_motion(tmp_path, 'm.txt')])
    window.filter_type.set_index(2)
    window.fc_low.setText('10')
    window.fc_high.setText('1')
    window.run(selected=False)
    assert 'should be lower' in gui.text()


def test_baseline_correction_tool(
        gui: Dialogs, tmp_path: pathlib.Path
) -> None:
    window = navigation.open_window('tool_baseline_correction')
    window.load_files([_short_motion(tmp_path, 'm.txt')])
    window.run(selected=True, save=True)
    assert (tmp_path / 'm_baseline_corrected.txt').exists()
    assert str(tmp_path) in gui.opened


@pytest.mark.usefixtures('gui')
@pytest.mark.parametrize('smoothing', [0, 1, 2])
def test_fourier_transform_tool(
        tmp_path: pathlib.Path, smoothing: int
) -> None:
    window = navigation.open_window('tool_fourier_transform')
    window.load_files([_short_motion(tmp_path, 'm.txt')])
    window.smoothing.set_index(smoothing)
    window.y_scale.set_index(1)
    window.run(selected=False, save=True)
    absolute = load_matrix(tmp_path / 'm_Fourier_spectrum_absolute.txt')
    complex_ = load_matrix(tmp_path / 'm_Fourier_spectrum_complex.txt')
    assert absolute.shape[1] == 2
    assert complex_.shape[1] == 3
    window.sides.set_index(1)  # double sided
    window.run(selected=False, save=True)
    assert (
        len(load_matrix(tmp_path / 'm_Fourier_spectrum_absolute.txt')) == 1200
    )


def test_response_spectra_tool(gui: Dialogs, tmp_path: pathlib.Path) -> None:
    window = navigation.open_window('tool_response_spectra')
    window.load_files([_short_motion(tmp_path, 'm.txt')])
    window.density.setText('20')
    window.max_period.setText('5')
    window.run(selected=False, save=True)
    spectra = load_matrix(tmp_path / 'm_response_spectra.txt')
    assert spectra.shape == (round(20 * np.log10(5 / 0.01)), 6)
    window.max_period.setText('abc')
    window.run(selected=False)
    assert 'should be numbers' in gui.text()


def test_unit_converter(gui: Dialogs, tmp_path: pathlib.Path) -> None:
    window = navigation.open_window('tool_unit_converter')
    path = _short_motion(tmp_path, 'm.txt')
    window.load_files([path])
    window.initial_unit.set_index(1)  # gal
    window.desired_unit.set_index(2)  # g
    gui.answer = QMessageBox.StandardButton.No
    window.run(selected=False)
    converted = load_matrix(tmp_path / 'm_in_g.txt')
    assert np.allclose(converted[:, 1], load_matrix(path)[:, 1] / 100 / 9.81)


@pytest.mark.usefixtures('gui')
def test_peer_and_smc_converters(tmp_path: pathlib.Path) -> None:
    from test_gui_data_io import PEER_NEW_FORMAT, _smc_text  # noqa: PLC0415

    peer = tmp_path / 'record.AT2'
    peer.write_text(PEER_NEW_FORMAT)
    window = navigation.open_window('tool_peer_to_2col')
    window.load_files([peer])
    window.desired_unit.set_index(1)  # gal
    window.run(selected=False)
    motion = load_matrix(tmp_path / 'record_(unit=gal).txt')
    assert np.allclose(motion[:, 0], 0.005 * np.arange(1, 8))
    assert np.isclose(motion[0, 1], -0.2537828e-5 * 9.81 * 100)

    smc = tmp_path / 'record.smc'
    smc.write_text(_smc_text([1.0, -2.0, 3.0], sampling_rate=100))
    window = navigation.open_window('tool_smc_to_2col')
    window.load_files([smc])
    window.run(selected=False)
    motion = load_matrix(tmp_path / 'record_(unit=SI).txt')
    assert np.allclose(motion, [[0.01, 0.01], [0.02, -0.02], [0.03, 0.03]])


def test_deconvolution_tool(gui: Dialogs, tmp_path: pathlib.Path) -> None:
    window = navigation.open_window('tool_deconvolution')
    assert window.calculate() == []
    path = _short_motion(tmp_path, 'surface.txt')
    window.set_motions([path])
    window.thickness.setText('30')
    window.vs.setText('500')
    window.density.setText('2000')
    window.damping.setText('2')
    for desired in range(3):
        window.desired.set_index(desired)
        outputs = window.calculate()
        assert outputs == [tmp_path / 'surface_deconvolved.txt']

    window.profile_type.set_index(1)
    assert window.calculate() == []
    assert 'select a Vs profile' in gui.text()
    window.load_profile(f_dir / 'profile_FKSH14.txt')
    assert len(window.calculate()) == 1


def test_gof_tool(gui: Dialogs, tmp_path: pathlib.Path) -> None:
    window = navigation.open_window('tool_gof_score')
    meas = _short_motion(tmp_path, 'meas.txt', n_pts=2000)
    simu = tmp_path / 'simu.txt'
    data = load_matrix(meas)
    data[:, 1] *= 0.8 / 100  # in m/s/s
    save_matrix(simu, data[::2], precision=8)  # a different time step
    window.meas_path.setText(str(meas))
    window.simu_path.setText(str(simu))
    window.meas_unit.set_index(1)  # gal
    window.fmax.setText('10')
    scores = window.calculate()
    assert scores is not None
    assert np.all(np.isfinite(scores))
    assert 'Warning: f_min value is invalid' not in gui.text()
    window.simu_path.setText(str(tmp_path / 'missing.txt'))
    assert window.calculate() is None

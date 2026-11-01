"""The "Post Process Results" window."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
)

from PySeismoSoil import helper_site_response as sr
from PySeismoSoil.gui import plots
from PySeismoSoil.gui.data_io import InputError, load_matrix
from PySeismoSoil.gui.navigation import Panel, register
from PySeismoSoil.gui.widgets import (
    BackgroundTask,
    TaskContext,
    button,
    get_directory,
    info,
    show_figure,
    warn,
)

if TYPE_CHECKING:
    import pathlib

    import numpy as np

    from PySeismoSoil.gui.widgets import FigureWindow

ACCEL_FIGURE_SUFFIX = '_accelerations.png'
LOOPS_FOLDER = 'Stress strain loops'
DEFAULT_SPEED = 20
DEFAULT_SUBPLOTS = 20
MIN_FRAME_INTERVAL_MS = 1


@register('post_processing')
class PostProcessingPanel(Panel):
    """The "Post Process Results" panel (``SeismoSoil_PostProcessing.m``)."""

    def __init__(self) -> None:
        super().__init__(
            'Post Process Results', 'SeismoSoil - Post Processing'
        )
        self.results_dir: pathlib.Path | None = None
        self.event_name = ''
        self._timer: QTimer | None = None
        self.main_layout.addWidget(
            button('Select output results folder', self._select_dir)
        )
        self.folder_label = QLabel('')
        self.folder_label.setWordWrap(True)
        self.main_layout.addWidget(self.folder_label)

        loops = QGroupBox('Stress-strain loops of each layer')
        grid = QGridLayout(loops)
        grid.addWidget(
            QLabel('Number of subplots to show on each plot:'), 0, 0
        )
        self.n_subplots = QLineEdit(str(DEFAULT_SUBPLOTS))
        self.n_subplots.setMaximumWidth(70)
        grid.addWidget(self.n_subplots, 0, 1)
        grid.addWidget(
            button('Plot loops', lambda: self.plot_loops(save=False)), 1, 0
        )
        grid.addWidget(
            button('Plot && save loops', lambda: self.plot_loops(save=True)),
            1,
            1,
        )
        self.main_layout.addWidget(loops)

        movie = QGroupBox('Ground displacement animation')
        grid = QGridLayout(movie)
        grid.addWidget(
            button('View movie', lambda: self.view_movie(save=False)), 0, 0
        )
        grid.addWidget(
            button('View && save movie', lambda: self.view_movie(save=True)),
            0,
            1,
        )
        row = QHBoxLayout()
        row.addWidget(QLabel('Show movie as'))
        self.speed = QLineEdit(str(DEFAULT_SPEED))
        self.speed.setMaximumWidth(70)
        row.addWidget(self.speed)
        row.addWidget(QLabel('times faster than reality'))
        row.addStretch()
        grid.addLayout(row, 1, 0, 1, 2)
        self.main_layout.addWidget(movie)
        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(button('Close all figures', self.close_all_figures))
        self.main_layout.addLayout(bottom)

    def set_results_dir(self, path: pathlib.Path) -> bool:
        """
        Select the folder of the results of one motion.

        Parameters
        ----------
        path : pathlib.Path
            The folder (which contains "*_accelerations.png").

        Returns
        -------
        bool
            Whether the folder contains valid results.
        """
        figures = sorted(path.glob(f'*{ACCEL_FIGURE_SUFFIX}'))
        if not figures:
            warn(
                self,
                'The folder you selected does not contain valid simulation '
                'results.',
            )
            return False

        self.results_dir = path
        self.event_name = figures[0].name[: -len(ACCEL_FIGURE_SUFFIX)]
        self.folder_label.setText(f'{path}\n(Motion: {self.event_name})')
        return True

    def _select_dir(self) -> None:
        path = get_directory(self, 'Select folder where results are stored...')
        if path is not None:
            self.set_results_dir(path)

    def _result(self, key: str) -> np.ndarray:
        for ext in ('.txt', '.dat'):
            path = self.results_dir / f'{self.event_name}_{key}{ext}'
            if path.exists():
                return load_matrix(path)

        raise InputError(f'Cannot find the "{key}" file of the results.')

    def _positive_int(self, edit: QLineEdit, what: str) -> int | None:
        try:
            value = int(float(edit.text()))
        except ValueError:
            value = 0

        if value <= 0:
            warn(
                self, f'Please specify a positive integer for {what}.', 'Error'
            )
            return None

        return value

    def plot_loops(self, *, save: bool) -> list[FigureWindow]:
        """
        Plot the stress-strain loops of every layer.

        Parameters
        ----------
        save : bool
            Whether to save the figures to the results folder.

        Returns
        -------
        list[FigureWindow]
            The figure windows.
        """
        if self.results_dir is None:
            warn(self, 'You have not selected a folder yet.')
            return []

        n_subplots = self._positive_int(
            self.n_subplots, 'the number of subplots'
        )
        if n_subplots is None:
            return []

        try:
            strain = self._result('time_history_strain')
            stress = self._result('time_history_stress')
            profile = self._result('re-discretized_profile')
        except InputError as err:
            warn(self, str(err))
            return []

        depth = sr.thk2dep(profile[:, 0])
        figures = plots.stress_strain_loop_figures(
            strain, stress, depth, n_subplots
        )
        if save:
            out = self.results_dir / LOOPS_FOLDER
            out.mkdir(exist_ok=True)
            for k, fig in enumerate(figures):
                last = min((k + 1) * n_subplots, strain.shape[1])
                fig.savefig(out / f'Stress_strain_loops_Layer_{last}.png')

            info(self, 'Message', f'Figures saved to {out}.')

        return [show_figure(fig, 'Stress-strain loops') for fig in figures]

    def _movie_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self.results_dir is None:
            warn(self, 'You have not selected a folder yet.')
            return None

        try:
            displ = self._result('time_history_displ')
            surface = self._result('accel_on_surface')
            profile = self._result('re-discretized_profile')
        except InputError as err:
            warn(self, str(err))
            return None

        depth = sr.thk2dep(profile[:, 0])[: displ.shape[1]]
        n = min(len(displ), len(surface))
        return displ[:n], depth, surface[:n]

    def view_movie(self, *, save: bool) -> FigureWindow | None:
        """
        Show the animation of the ground displacement.

        Parameters
        ----------
        save : bool
            Whether to also save the animation (as a GIF file).

        Returns
        -------
        FigureWindow | None
            The window of the animation.
        """
        data = self._movie_data()
        speed = self._positive_int(self.speed, 'the speed')
        if data is None or speed is None:
            return None

        displ, depth, surface = data
        fig, update = plots.movie_figure(
            displ=displ, depth=depth, surface_accel=surface
        )
        window = show_figure(fig, 'Ground displacement animation')
        dt = surface[1, 0] - surface[0, 0]
        frames = iter(range(0, len(displ), speed))
        timer = QTimer(window)

        def next_frame() -> None:
            k = next(frames, None)
            if k is None:
                timer.stop()
                return

            update(k, speed)
            window.canvas.draw_idle()

        timer.timeout.connect(next_frame)
        timer.start(max(MIN_FRAME_INTERVAL_MS, round(dt * 1000)))
        self._timer = timer
        if save:
            self._save_movie(displ, depth, surface, speed)

        return window

    def _save_movie(
            self,
            displ: np.ndarray,
            depth: np.ndarray,
            surface: np.ndarray,
            speed: int,
    ) -> BackgroundTask:
        path = self.results_dir / f'{self.event_name}.gif'
        dt = surface[1, 0] - surface[0, 0]

        def task(ctx: TaskContext) -> pathlib.Path:
            fig, update = plots.movie_figure(
                displ=displ, depth=depth, surface_accel=surface
            )
            FigureCanvasAgg(fig)
            frames = list(range(0, len(displ), speed))
            animation = FuncAnimation(
                fig, lambda k: update(k, speed), frames=frames, blit=False
            )
            fps = max(1, round(1 / dt))
            animation.save(
                str(path),
                writer=PillowWriter(fps=fps),
                progress_callback=lambda i, n: ctx.progress(i / max(n, 1)),
            )
            return path

        return BackgroundTask(
            self,
            task,
            on_done=lambda p: info(self, 'Message', f'Movie saved to {p}.'),
            message='Saving the movie...',
        ).start()

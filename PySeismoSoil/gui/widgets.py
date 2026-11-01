"""Widgets and helpers shared by the windows of the GUI."""

from __future__ import annotations

import logging
import pathlib
import sys
import threading
import traceback
from typing import TYPE_CHECKING, Any, ClassVar, TextIO

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.figure import Figure
    from PySide6.QtGui import QCloseEvent

_LOGGER = logging.getLogger(__name__)

TEXT_FILES = 'Text files (*.dat *.txt);;All Files (*)'
ALL_FILES = 'All Files (*)'
RUNNING_MESSAGE = 'Analysis in progress. Please do not click other buttons.'
FITTING_MESSAGE = (
    'Curve-fitting in progress. Please do not click other buttons.'
)


class Session:
    """State shared by all the windows (like the globals of MATLAB)."""

    start_dir: ClassVar[pathlib.Path] = pathlib.Path.cwd()


def title_label(text: str, size: int = 20) -> QLabel:
    """
    Create a bold title label.

    Parameters
    ----------
    text : str
        The title.
    size : int, default=20
        Font size (pt).

    Returns
    -------
    QLabel
        The label.
    """
    label = QLabel(text)
    font = QFont()
    font.setPointSize(size)
    font.setBold(True)
    label.setFont(font)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


def group_box(title: str, *, bold: bool = True) -> QGroupBox:
    """
    Create a group box (the "uipanel" of MATLAB).

    Parameters
    ----------
    title : str
        The title.
    bold : bool, default=True
        Whether the title is bold.

    Returns
    -------
    QGroupBox
        The group box.
    """
    box = QGroupBox(title)
    if bold:
        box.setStyleSheet('QGroupBox { font-weight: bold; }')

    return box


def button(
        text: str, slot: Callable[[], Any], tooltip: str = ''
) -> QPushButton:
    """
    Create a push button that calls ``slot`` (without arguments) on click.

    Parameters
    ----------
    text : str
        Text of the button.
    slot : Callable[[], Any]
        The function to call.
    tooltip : str, default=''
        Tooltip of the button.

    Returns
    -------
    QPushButton
        The button.
    """
    btn = QPushButton(text)
    btn.clicked.connect(lambda *_: slot())
    if tooltip:
        btn.setToolTip(tooltip)

    return btn


class RadioGroup(QGroupBox):
    """
    A group of radio buttons (the "uibuttongroup" of MATLAB).

    Parameters
    ----------
    title : str
        Title of the group.
    options : Sequence[tuple[str, Any]]
        The (label, value) of every radio button.
    default : int, default=0
        Index of the option checked by default.
    horizontal : bool, default=False
        Whether to arrange the buttons horizontally.

    Attributes
    ----------
    changed : Signal, default=Signal()
        Emitted when the user checks another option.
    """

    changed: Signal = Signal()

    def __init__(
            self,
            title: str,
            options: Sequence[tuple[str, Any]],
            *,
            default: int = 0,
            horizontal: bool = False,
    ) -> None:
        super().__init__(title)
        self._values = [value for _, value in options]
        self._group = QButtonGroup(self)
        layout = QHBoxLayout(self) if horizontal else QVBoxLayout(self)
        for i, (label, _) in enumerate(options):
            radio = QRadioButton(label)
            self._group.addButton(radio, i)
            layout.addWidget(radio)

        self._group.button(default).setChecked(True)
        self._group.idToggled.connect(self._on_toggled)

    def _on_toggled(self, _id: int, checked: object) -> None:
        if checked:
            self.changed.emit()

    def value(self) -> Any:
        """
        Return the value of the checked option.

        Returns
        -------
        Any
            The value.
        """
        return self._values[self._group.checkedId()]

    def set_index(self, index: int) -> None:
        """
        Check an option.

        Parameters
        ----------
        index : int
            Index of the option.
        """
        self._group.button(index).setChecked(True)


class FileList(QListWidget):
    """A list of file names, with multiple selection (the MATLAB "listbox")."""

    def __init__(self, placeholder: str = 'Motion list...') -> None:
        super().__init__()
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.addItem(placeholder)
        self.setMinimumWidth(220)

    def set_names(self, names: Sequence[str]) -> None:
        """
        Show a list of names, and select the first one.

        Parameters
        ----------
        names : Sequence[str]
            The names.
        """
        self.clear()
        self.addItems(list(names))
        if names:
            self.setCurrentRow(0)

    def selected_indices(self) -> list[int]:
        """
        Return the indices of the selected items (in order).

        Returns
        -------
        list[int]
            The indices.
        """
        return sorted(self.row(item) for item in self.selectedItems())


# ----------------------------------------------------------------------------
# Message boxes and file dialogs
# ----------------------------------------------------------------------------
def info(parent: QWidget | None, title: str, text: str) -> None:
    """
    Show an information message (the "msgbox" of MATLAB).

    Parameters
    ----------
    parent : QWidget | None
        The parent window.
    title : str
        Title of the message box.
    text : str
        The message.
    """
    QMessageBox.information(parent, title, text)


def warn(parent: QWidget | None, text: str, title: str = 'Warning') -> None:
    """
    Show a warning message.

    Parameters
    ----------
    parent : QWidget | None
        The parent window.
    text : str
        The message.
    title : str, default='Warning'
        Title of the message box.
    """
    QMessageBox.warning(parent, title, text)


def ask(
        parent: QWidget | None,
        text: str,
        *,
        title: str = 'Caution',
        default_yes: bool = True,
) -> bool:
    """
    Ask a yes/no question (the "questdlg" of MATLAB).

    Parameters
    ----------
    parent : QWidget | None
        The parent window.
    text : str
        The question.
    title : str, default='Caution'
        Title of the dialog.
    default_yes : bool, default=True
        Whether "Yes" is the default button.

    Returns
    -------
    bool
        Whether the user answered "Yes".
    """
    yes = QMessageBox.StandardButton.Yes
    no = QMessageBox.StandardButton.No
    answer = QMessageBox.question(
        parent, title, text, yes | no, yes if default_yes else no
    )
    return answer == yes


def get_open_files(
        parent: QWidget | None,
        title: str,
        *,
        file_filter: str = TEXT_FILES,
) -> list[pathlib.Path]:
    """
    Ask the user to select one or more files.

    Parameters
    ----------
    parent : QWidget | None
        The parent window.
    title : str
        Title of the dialog.
    file_filter : str, default=TEXT_FILES
        File filter of the dialog.

    Returns
    -------
    list[pathlib.Path]
        The selected files (empty if the user cancels).
    """
    names, _ = QFileDialog.getOpenFileNames(
        parent, title, str(Session.start_dir), file_filter
    )
    paths = [pathlib.Path(name) for name in names]
    if paths:
        Session.start_dir = paths[0].parent
        print(f'User selected {len(paths)} file(s) in {paths[0].parent}')
    else:
        print('User did not select any files.')

    return paths


def get_open_file(
        parent: QWidget | None,
        title: str,
        *,
        file_filter: str = TEXT_FILES,
) -> pathlib.Path | None:
    """
    Ask the user to select one file.

    Parameters
    ----------
    parent : QWidget | None
        The parent window.
    title : str
        Title of the dialog.
    file_filter : str, default=TEXT_FILES
        File filter of the dialog.

    Returns
    -------
    pathlib.Path | None
        The selected file (``None`` if the user cancels).
    """
    name, _ = QFileDialog.getOpenFileName(
        parent, title, str(Session.start_dir), file_filter
    )
    if not name:
        print('User did not select any files.')
        return None

    path = pathlib.Path(name)
    Session.start_dir = path.parent
    print(f'User selected {path}')
    return path


def get_directory(parent: QWidget | None, title: str) -> pathlib.Path | None:
    """
    Ask the user to select a folder.

    Parameters
    ----------
    parent : QWidget | None
        The parent window.
    title : str
        Title of the dialog.

    Returns
    -------
    pathlib.Path | None
        The selected folder (``None`` if the user cancels).
    """
    name = QFileDialog.getExistingDirectory(
        parent, title, str(Session.start_dir)
    )
    if not name:
        return None

    Session.start_dir = pathlib.Path(name)
    return pathlib.Path(name)


def get_save_file(
        parent: QWidget | None, title: str, default_name: str
) -> pathlib.Path | None:
    """
    Ask the user for the name of a file to save.

    Parameters
    ----------
    parent : QWidget | None
        The parent window.
    title : str
        Title of the dialog.
    default_name : str
        The default file name.

    Returns
    -------
    pathlib.Path | None
        The file (``None`` if the user cancels).
    """
    name, _ = QFileDialog.getSaveFileName(
        parent, title, str(Session.start_dir / default_name), TEXT_FILES
    )
    return pathlib.Path(name) if name else None


def open_folder(path: str | pathlib.Path) -> None:
    """
    Open a folder in the file manager of the operating system.

    Parameters
    ----------
    path : str | pathlib.Path
        The folder.
    """
    folder = pathlib.Path(path).resolve()
    print(f'Opening {folder}')
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


# ----------------------------------------------------------------------------
# Console
# ----------------------------------------------------------------------------
class _Emitter(QObject):
    """Emit text written to the console from any thread."""

    text_written = Signal(str)


class _TeeStream:
    """A text stream that writes to the original stream and the console."""

    def __init__(self, original: TextIO | None, emitter: _Emitter) -> None:
        self._original = original
        self._emitter = emitter

    def write(self, text: str) -> int:
        if self._original is not None:
            self._original.write(text)

        if text:
            self._emitter.text_written.emit(text)

        return len(text)

    def flush(self) -> None:
        if self._original is not None:
            self._original.flush()

    @staticmethod
    def isatty() -> bool:
        return False


class Console(QWidget):
    """
    A window that shows the printed messages (the MATLAB command window).

    Use ``Console.instance()`` to get the one shared by all windows.
    """

    _instance: ClassVar[Console | None] = None

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('SeismoSoil console')
        self.resize(720, 360)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont('Monospace', 9))
        self._text.setMaximumBlockCount(20000)
        layout = QVBoxLayout(self)
        layout.addWidget(self._text)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(button('Clear console', self.clear))
        row.addWidget(button('Hide', self.hide))
        layout.addLayout(row)
        self._emitter = _Emitter()
        self._emitter.text_written.connect(self._append)

    @classmethod
    def instance(cls) -> Console:
        """
        Return the console shared by all windows.

        Returns
        -------
        Console
            The console.
        """
        if cls._instance is None:
            cls._instance = Console()

        return cls._instance

    def redirect_std_streams(self) -> None:
        """Copy everything printed to stdout and stderr to this console."""
        if not isinstance(sys.stdout, _TeeStream):
            sys.stdout = _TeeStream(sys.stdout, self._emitter)

        if not isinstance(sys.stderr, _TeeStream):
            sys.stderr = _TeeStream(sys.stderr, self._emitter)

    def _append(self, text: str) -> None:
        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self._text.setTextCursor(cursor)
        self._text.ensureCursorVisible()

    def clear(self) -> None:
        """Clear the console (the "clc" of MATLAB)."""
        self._text.clear()

    def text(self) -> str:
        """
        Return all the text in the console.

        Returns
        -------
        str
            The text.
        """
        return self._text.toPlainText()

    def show_console(self) -> None:
        """Show the console window."""
        self.show()
        self.raise_()


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
class FigureWindow(QWidget):
    """
    A window that shows a matplotlib figure, with the navigation toolbar.

    Parameters
    ----------
    fig : Figure
        The figure.
    title : str, default='Figure'
        Title of the window.
    """

    _open_windows: ClassVar[list[FigureWindow]] = []

    def __init__(self, fig: Figure, title: str = 'Figure') -> None:
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.figure = fig
        self.canvas = FigureCanvasQTAgg(fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)
        width, height = fig.get_size_inches()
        dpi = fig.get_dpi()
        self.resize(int(width * dpi) + 10, int(height * dpi) + 50)
        FigureWindow._open_windows.append(self)

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Forget this window when it is closed.

        Parameters
        ----------
        event : QCloseEvent
            The close event.
        """
        if self in FigureWindow._open_windows:
            FigureWindow._open_windows.remove(self)

        super().closeEvent(event)

    @classmethod
    def close_all(cls) -> None:
        """Close all the figure windows (the "close all" of MATLAB)."""
        for window in list(cls._open_windows):
            window.close()

    @classmethod
    def open_windows(cls) -> list[FigureWindow]:
        """
        Return the open figure windows.

        Returns
        -------
        list[FigureWindow]
            The windows.
        """
        return list(cls._open_windows)


def show_figure(fig: Figure, title: str = 'Figure') -> FigureWindow:
    """
    Show a matplotlib figure in a new window.

    Parameters
    ----------
    fig : Figure
        The figure.
    title : str, default='Figure'
        Title of the window.

    Returns
    -------
    FigureWindow
        The window.
    """
    window = FigureWindow(fig, title)
    window.show()
    return window


def show_pyplot_figures(*, show: bool) -> None:
    """
    Show (or discard) the figures that were created with pyplot.

    Some functions of PySeismoSoil create figures with pyplot (whose backend is
    "Agg" in the GUI). This moves them into figure windows.

    Parameters
    ----------
    show : bool
        Whether to show the figures. If ``False``, they are closed.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    for num in plt.get_fignums():
        fig = plt.figure(num)
        plt.close(fig)
        if show:
            show_figure(fig, f'Figure {num}')


# ----------------------------------------------------------------------------
# Background tasks
# ----------------------------------------------------------------------------
class TaskCancelledError(Exception):
    """Raised (in the task) when the user cancels a background task."""


class TaskContext:
    """
    Passed to a background task, to report progress and to check cancellation.

    Parameters
    ----------
    emit_progress : Callable[[float, str], None]
        Function to report the progress (0 to 1) and a message.
    """

    def __init__(self, emit_progress: Callable[[float, str], None]) -> None:
        self._emit = emit_progress
        self._cancelled = threading.Event()
        self._message = ''

    def progress(self, fraction: float, message: str | None = None) -> None:
        """
        Report the progress, and raise if the user cancelled the task.

        Parameters
        ----------
        fraction : float
            Fraction of the task finished (0 to 1).
        message : str | None, default=None
            A message to show (if ``None``, keep the previous one).

        Raises
        ------
        TaskCancelledError
            When the user cancelled the task
        """
        if message is not None:
            self._message = message

        self._emit(float(fraction), self._message)
        if self._cancelled.is_set():
            raise TaskCancelledError

    def cancel(self) -> None:
        """Ask the task to stop (at its next progress report)."""
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        """
        Whether the user cancelled the task.

        Returns
        -------
        bool
            Whether the task was cancelled.
        """
        return self._cancelled.is_set()


class _Worker(QObject):
    """Run a function in a background thread."""

    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, func: Callable[[TaskContext], Any]) -> None:
        super().__init__()
        self._func = func
        self.context = TaskContext(self.progress.emit)

    def run(self) -> None:
        try:
            result = self._func(self.context)
        except TaskCancelledError:
            self.cancelled.emit()
        except Exception as err:
            _LOGGER.exception('Error in a background task')
            self.failed.emit(
                f'{type(err).__name__}: {err}', traceback.format_exc()
            )
        else:
            self.finished.emit(result)


class BackgroundTask(QObject):
    """
    Run a function in a background thread, with a progress dialog.

    The GUI stays responsive while the function runs. The function receives a
    ``TaskContext``. ``on_done`` (if not ``None``) is called in the GUI thread
    with the return value of the function.

    Parameters
    ----------
    parent : QWidget
        The window that starts the task.
    func : Callable[[TaskContext], Any]
        The function.
    on_done : Callable[[Any], None] | None, default=None
        Called when the function returns.
    title : str, default='Calculating'
        Title of the progress dialog.
    message : str, default=RUNNING_MESSAGE
        Message of the progress dialog.
    """

    _running: ClassVar[list[BackgroundTask]] = []

    def __init__(
            self,
            parent: QWidget,
            func: Callable[[TaskContext], Any],
            *,
            on_done: Callable[[Any], None] | None = None,
            title: str = 'Calculating',
            message: str = RUNNING_MESSAGE,
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._on_done = on_done
        self._message = message
        self._thread = QThread()
        self._worker = _Worker(func)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self.dialog = QProgressDialog(message, 'Cancel', 0, 1000, parent)
        self.dialog.setWindowTitle(title)
        self.dialog.setMinimumDuration(0)
        self.dialog.setAutoClose(False)
        self.dialog.setAutoReset(False)
        self.dialog.canceled.connect(self._worker.context.cancel)
        self.result: Any = None
        self.error: str | None = None
        self.done = False

    def start(self) -> BackgroundTask:
        """
        Start the task.

        Returns
        -------
        BackgroundTask
            This task.
        """
        BackgroundTask._running.append(self)
        self.dialog.setValue(0)
        self.dialog.show()
        self._thread.start()
        return self

    def _on_progress(self, fraction: float, message: str) -> None:
        self.dialog.setValue(round(fraction * 1000))
        text = f'{self._message}\n\n{message}' if message else self._message
        self.dialog.setLabelText(text)

    def _cleanup(self) -> None:
        self._thread.quit()
        self._thread.wait()
        self.dialog.close()
        self.done = True
        if self in BackgroundTask._running:
            BackgroundTask._running.remove(self)

    def _on_finished(self, result: object) -> None:
        self.result = result
        self._cleanup()
        if self._on_done is not None:
            self._on_done(result)

    def _on_failed(self, message: str, details: str) -> None:
        self.error = message
        self._cleanup()
        print(details, file=sys.stderr)
        box = QMessageBox(self._parent)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle('Error')
        box.setText(message)
        box.setDetailedText(details)
        box.open()

    def _on_cancelled(self) -> None:
        self._cleanup()
        print('Cancelled by the user.')

    @classmethod
    def running(cls) -> list[BackgroundTask]:
        """
        Return the tasks that are running.

        Returns
        -------
        list[BackgroundTask]
            The tasks.
        """
        return list(cls._running)

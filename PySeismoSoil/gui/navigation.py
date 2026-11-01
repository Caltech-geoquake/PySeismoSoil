"""
Navigation between the windows of the GUI.

As in MATLAB SeismoSoil, opening a sub-panel (e.g., "Tools") closes the current
panel, and "Return to main panel" (or "Return to upper level") closes the
sub-panel and opens the main panel again. The windows register their classes
here by name, so that they can open each other without importing each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from PySeismoSoil.gui.widgets import (
    Console,
    FigureWindow,
    button,
    title_label,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtGui import QCloseEvent

LONG_TITLE_LENGTH = 32  # titles longer than this use a smaller font
LONG_TITLE_FONT_SIZE = 16

_REGISTRY: dict[str, type[QWidget]] = {}
_OPEN_WINDOWS: list[QWidget] = []

WindowType = TypeVar('WindowType', bound=type[QWidget])


def register(name: str) -> Callable[[WindowType], WindowType]:
    """
    Register a window class under a name (a class decorator).

    Parameters
    ----------
    name : str
        The name of the window.

    Returns
    -------
    Callable[[WindowType], WindowType]
        The decorator.
    """

    def decorator(cls: WindowType) -> WindowType:
        _REGISTRY[name] = cls
        return cls

    return decorator


def open_window(name: str, *, replace: QWidget | None = None) -> QWidget:
    """
    Open a window by its name.

    Parameters
    ----------
    name : str
        The name of the window.
    replace : QWidget | None, default=None
        A window to close (i.e., the window to be replaced by the new one).

    Returns
    -------
    QWidget
        The new window.
    """
    window = _REGISTRY[name]()
    _OPEN_WINDOWS.append(window)
    window.show()
    window.raise_()
    if replace is not None:
        replace.close()

    return window


def registered_names() -> list[str]:
    """
    Return the names of all the registered windows.

    Returns
    -------
    list[str]
        The names.
    """
    return list(_REGISTRY)


def open_windows() -> list[QWidget]:
    """
    Return the windows (panels) that are open.

    Returns
    -------
    list[QWidget]
        The windows.
    """
    return list(_OPEN_WINDOWS)


class Panel(QWidget):
    """
    Base class of the panels (windows) of the GUI.

    Parameters
    ----------
    title : str
        The title shown at the top of the panel.
    window_title : str
        The title of the window.
    return_to : str | None, default=None
        Name of the window that the "Return" button opens (``None`` for no
        "Return" button).
    return_text : str, default='Return to upper level'
        Text of the "Return" button.
    title_size : int, default=20
        Font size of the title.
    """

    def __init__(
            self,
            title: str,
            window_title: str,
            *,
            return_to: str | None = None,
            return_text: str = 'Return to upper level',
            title_size: int = 20,
    ) -> None:
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(window_title)
        self._return_to = return_to
        self.main_layout = QVBoxLayout(self)
        header = QHBoxLayout()
        if len(title) > LONG_TITLE_LENGTH:
            title_size = min(title_size, LONG_TITLE_FONT_SIZE)

        self.title = title_label(title, title_size)
        self.title.setWordWrap(True)
        header.addWidget(self.title, stretch=1)
        if return_to is not None:
            header.addWidget(
                button(return_text, self.go_back),
                alignment=Qt.AlignmentFlag.AlignTop,
            )

        self.main_layout.addLayout(header)

    def go_back(self) -> None:
        """Close this panel, and open the upper-level panel."""
        if self._return_to is not None:
            open_window(self._return_to, replace=self)

    def open_sub_panel(self, name: str, *, keep_open: bool = False) -> QWidget:
        """
        Open another panel.

        Parameters
        ----------
        name : str
            Name of the panel.
        keep_open : bool, default=False
            Whether to keep this panel open.

        Returns
        -------
        QWidget
            The new panel.
        """
        return open_window(name, replace=None if keep_open else self)

    @staticmethod
    def close_all_figures() -> None:
        """Close all the figure windows."""
        FigureWindow.close_all()

    @staticmethod
    def clear_console() -> None:
        """Clear the console."""
        Console.instance().clear()

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Forget this panel when it is closed.

        Parameters
        ----------
        event : QCloseEvent
            The close event.
        """
        if self in _OPEN_WINDOWS:
            _OPEN_WINDOWS.remove(self)

        super().closeEvent(event)

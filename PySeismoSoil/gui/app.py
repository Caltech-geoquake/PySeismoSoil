"""The Qt application of the GUI."""

from __future__ import annotations

import sys

import matplotlib as mpl
from PySide6.QtWidgets import QApplication

from PySeismoSoil.gui.navigation import open_window
from PySeismoSoil.gui.widgets import Console

# Importing the windows registers them in ``navigation``
from PySeismoSoil.gui.windows import (  # noqa: F401
    analysis,
    main_window,
    post_processing,
    preparation,
    tools,
)


def get_app(argv: list[str] | None = None) -> QApplication:
    """
    Get the Qt application (and create it if it does not exist yet).

    Parameters
    ----------
    argv : list[str] | None, default=None
        Command line arguments for Qt. If ``None``, use ``sys.argv``.

    Returns
    -------
    QApplication
        The application.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if argv is None else argv)
        app.setApplicationName('SeismoSoil')

    # The library functions that make figures with pyplot (e.g., the HH
    # calibration) can run in background threads. Their figures are shown
    # in the GUI's own figure windows instead.
    mpl.use('Agg')
    return app


def run(argv: list[str] | None = None) -> int:
    """
    Start the GUI, and return when the user closes all its windows.

    Parameters
    ----------
    argv : list[str] | None, default=None
        Command line arguments for Qt. If ``None``, use ``sys.argv``.

    Returns
    -------
    int
        The exit code of the Qt application.
    """
    app = get_app(argv)
    Console.instance().redirect_std_streams()
    open_window('main')
    return app.exec()

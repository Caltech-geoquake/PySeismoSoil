"""Start the GUI (without importing Qt until it is needed)."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """
    Start the GUI.

    Parameters
    ----------
    argv : list[str] | None, default=None
        Command line arguments for Qt. If ``None``, use ``sys.argv``.

    Returns
    -------
    int
        The exit code of the Qt application.

    Raises
    ------
    ImportError
        When PySide6 is not installed
    """
    try:
        from PySeismoSoil.gui import app  # noqa: PLC0415
    except ImportError as err:
        if 'PySide6' not in str(err):
            raise

        raise ImportError(
            'The PySeismoSoil GUI needs PySide6. Please install it with '
            '`pip install PySeismoSoil[gui]` (or `pip install PySide6`).',
        ) from err

    return app.run(argv)

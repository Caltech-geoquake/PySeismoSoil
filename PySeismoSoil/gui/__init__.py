"""
A graphical user interface (GUI) for PySeismoSoil.

The GUI replicates the windows and workflows of MATLAB SeismoSoil
(https://github.com/Caltech-geoquake/SeismoSoil): preparing input files,
running linear, equivalent linear, and nonlinear site response analyses,
post-processing the results, and a set of tools for ground motions and Vs
profiles. It needs PySide6 (``pip install PySeismoSoil[gui]``).

Start it with the ``seismosoil-gui`` command, ``python -m PySeismoSoil.gui``,
or ``PySeismoSoil.gui.main()``.
"""

from PySeismoSoil.gui.launcher import main

__all__ = ['main']

"""The main panel of the GUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

import PySeismoSoil
from PySeismoSoil.gui.navigation import Panel, register
from PySeismoSoil.gui.widgets import (
    Console,
    button,
    group_box,
    info,
    title_label,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ABOUT_TEXT = (
    'PySeismoSoil GUI: a Python (PySide6) replica of MATLAB SeismoSoil.\n'
    'PySeismoSoil version: {version}\n\n'
    'MATLAB SeismoSoil: v1.3.4.2, 3/6/2019 (first draft: 6/30/2013)\n'
    'Authors: Domniki Asimaki, Jian Shi, Wei Li\n'
    'GUI Design: Jian Shi\n\n'
    'GeoQuake Research Group, California Institute of Technology'
)


@register('main')
class MainPanel(Panel):
    """The main panel (``SeismoSoil.m``)."""

    def __init__(self) -> None:
        super().__init__('SeismoSoil', 'SeismoSoil', title_size=32)
        self.main_layout.addWidget(
            _subtitle('1D Site-Specific Response Analysis')
        )

        top = QHBoxLayout()
        prep = group_box('Preparation')
        QHBoxLayout(prep).addWidget(
            self._big_button(
                'Input files preparation',
                lambda: self.open_sub_panel('preparation'),
            ),
        )
        tools = group_box('Tools')
        QHBoxLayout(tools).addWidget(
            self._big_button('Tools', lambda: self.open_sub_panel('tools'))
        )
        top.addWidget(prep)
        top.addWidget(tools)
        self.main_layout.addLayout(top)
        self.main_layout.addWidget(self._analyses_panel())

        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(
            button('Show console', Console.instance().show_console)
        )
        bottom.addWidget(button('About SeismoSoil', self._about))
        self.main_layout.addLayout(bottom)

    @staticmethod
    def _big_button(text: str, slot: Callable[[], object]) -> QPushButton:
        btn = button(text, slot)
        btn.setMinimumHeight(40)
        font = btn.font()
        font.setPointSize(13)
        btn.setFont(font)
        return btn

    def _analyses_panel(self) -> QGroupBox:
        panel = group_box('Analyses')
        grid = QGridLayout(panel)
        rows = [
            (
                'Linear',
                [
                    ('Frequency Domain', 'analysis_linear_fd'),
                    ('Time Domain', 'analysis_linear_td'),
                ],
            ),
            (
                'Equivalent Linear',
                [
                    ('Seed && Idriss (1970)', 'analysis_eql'),
                    ('Frequency Dependent', 'analysis_eql_fd'),
                ],
            ),
            (
                'Nonlinear',
                [
                    ('EPP', 'analysis_nl_epp'),
                    ('H2', 'analysis_nl_h2'),
                    ('H4', 'analysis_nl_h4'),
                    ('HH', 'analysis_nl_hh'),
                ],
            ),
        ]
        for i, (name, buttons) in enumerate(rows):
            box = group_box(name, bold=False)
            row = QHBoxLayout(box)
            for text, window in buttons:
                row.addWidget(
                    self._big_button(
                        text,  # "&&" shows "&" on a Qt button
                        lambda w=window: self.open_sub_panel(w),
                    ),
                )

            grid.addWidget(box, i, 0)

        return panel

    def _about(self) -> None:
        info(
            self,
            'About',
            ABOUT_TEXT.format(version=PySeismoSoil.__version__),
        )


def _subtitle(text: str) -> QLabel:
    """Create a subtitle label."""
    label = title_label(text, 16)
    font = label.font()
    font.setBold(False)
    label.setFont(font)
    return label

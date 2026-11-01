import os
from pathlib import Path

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import PySeismoSoil.helper_generic as hlp
from PySeismoSoil.class_frequency_spectrum import Frequency_Spectrum as FS

f_dir = Path(__file__).resolve().parent / 'files'


def test_load_data() -> None:
    txt_filename = str(f_dir / 'two_column_data_example.txt')

    fs_bench, df_bench = hlp.read_two_column_stuff(txt_filename)
    fs = FS(txt_filename, fmin=0.1, fmax=2.5, n_pts=20, log_scale=False)

    assert fs.raw_df == pytest.approx(df_bench, abs=1e-7)
    assert np.allclose(fs.raw_data, fs_bench)
    assert fs.spectrum[0] == pytest.approx(1, abs=1e-7)
    assert fs.spectrum[-1] == pytest.approx(7, abs=1e-7)


def test_plot() -> None:
    txt_filename = str(f_dir / 'two_column_data_example.txt')
    fs = FS(txt_filename, fmin=0.1, fmax=2.5, n_pts=20, log_scale=False)
    _, ax = fs.plot()
    assert ax.title.get_text() == os.path.split(txt_filename)[1]


def test_get_f0__peak_at_the_second_to_last_frequency() -> None:
    # The end of an amplification function that rises to its first peak at
    # 29.9 Hz, the second-to-last frequency
    freq = np.array([29.8, 29.85, 29.9, 29.95])
    ampl = np.array([4.8853, 4.8856, 4.8860, 4.8856])
    fs = FS(np.column_stack((freq, ampl)))
    assert fs.get_f0() == pytest.approx(29.9, abs=1e-9)


@pytest.mark.parametrize(
    'show_fig',
    [
        pytest.param(False, id='without_figure'),
        pytest.param(True, id='with_figure'),
    ],
)
def test_get_smoothed(*, show_fig: bool) -> None:
    freq = np.logspace(-1, 1.5, num=200)  # log-spaced (`log_scale=True`)
    fs = FS(np.column_stack((freq, 1.0 + freq)))
    smoothed, fig, ax = fs.get_smoothed(show_fig=show_fig)

    assert smoothed.shape == fs.spectrum.shape
    # `log_scale` is `True` by default
    assert np.array_equal(smoothed, fs.get_smoothed(log_scale=True)[0])
    if show_fig:
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
    else:
        assert fig is None
        assert ax is None

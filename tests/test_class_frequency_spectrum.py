import os
from pathlib import Path

import numpy as np
import pytest

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

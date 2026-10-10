from pathlib import Path

import numpy as np
import pytest

from PySeismoSoil.class_ground_motion import Ground_Motion
from PySeismoSoil.class_site_effect_adjustment import Site_Effect_Adjustment

f_dir = Path(__file__).resolve().parent / 'files'


def test_init() -> None:
    gm = Ground_Motion(str(f_dir / 'sample_accel.txt'), unit='gal')
    vs30 = 250
    z1 = 150
    Site_Effect_Adjustment(gm, vs30, z1)


def test_run__normal_case() -> None:
    gm_in = Ground_Motion(str(f_dir / 'sample_accel.txt'), unit='gal')
    vs30 = 207
    z1 = 892
    sea = Site_Effect_Adjustment(gm_in, vs30, z1)
    gm_out = sea.run(show_fig=True, dpi=150)[0]
    assert isinstance(gm_out, Ground_Motion)


@pytest.mark.parametrize(
    ('out_of_bound_values', 'values_at_the_bound'),
    [
        pytest.param((170, 75), (175, 75), id='out_of_bound_Vs30'),
        pytest.param((360, 927), (360, 900), id='out_of_bound_z1'),
    ],
)
def test_run__lenient_case(
        out_of_bound_values: tuple[float, float],
        values_at_the_bound: tuple[float, float],
) -> None:
    gm_in = Ground_Motion(str(f_dir / 'sample_accel.txt'), unit='gal')
    sea1 = Site_Effect_Adjustment(gm_in, *out_of_bound_values, lenient=True)
    sea2 = Site_Effect_Adjustment(gm_in, *values_at_the_bound)
    motion_out1 = sea1.run()[0]
    motion_out2 = sea2.run()[0]
    assert np.allclose(motion_out1.accel, motion_out2.accel)

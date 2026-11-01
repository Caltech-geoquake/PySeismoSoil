from pathlib import Path

import numpy as np
import pytest

import PySeismoSoil.helper_site_response as sr
from PySeismoSoil.class_ground_motion import Ground_Motion
from PySeismoSoil.class_simulation_results import Simulation_Results
from PySeismoSoil.class_Vs_profile import Vs_Profile

f_dir = Path(__file__).resolve().parent / 'files'


def test_plot() -> None:
    # Test that the desired data are correctly imported to the object
    accel_in = Ground_Motion(str(f_dir / 'sample_accel.txt'), unit='m/s/s')
    accel_tmp = accel_in.accel.copy()
    accel_tmp[:, 1] *= 5.0
    accel_out = Ground_Motion(accel_tmp, unit='m/s/s')
    vs_profile = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    thk = vs_profile._thk
    depth_bound = sr.thk2dep(thk, midpoint=False)
    depth_midpoint = sr.thk2dep(thk, midpoint=True)
    max_a_v_d = np.column_stack(
        (depth_bound, depth_bound * 1, depth_bound * 2, depth_bound * 3),
    )
    max_gamma_tau = np.column_stack(
        (depth_midpoint, depth_midpoint * 1, depth_midpoint * 2),
    )
    tf_RO, _, _ = vs_profile.get_transfer_function()
    sim_results = Simulation_Results(
        accel_in,
        accel_out,
        vs_profile,
        max_a_v_d=max_a_v_d,
        max_strain_stress=max_gamma_tau,
        trans_func=tf_RO,
    )
    sim_results.plot(save_fig=False)

    # Test that it can produce a plot without max profiles and trans. func.
    sim_results_ = Simulation_Results(accel_in, accel_out, vs_profile)
    sim_results_.plot(save_fig=False)


@pytest.mark.parametrize(
    'output_dir',
    [
        pytest.param(None, id='output_dir_of_the_object'),
        pytest.param('figs', id='output_dir_argument'),
    ],
)
def test_plot__save_fig_saves_both_figures_to_output_dir(
        tmp_path: Path,
        output_dir: str | None,
) -> None:
    accel_in = Ground_Motion(str(f_dir / 'sample_accel.txt'), unit='m/s/s')
    vs_profile = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    depth_bound = np.arange(vs_profile.n_layer + 1.0)
    depth_midpoint = depth_bound[:-1] + 0.5
    sim_results = Simulation_Results(
        accel_in,
        accel_in,
        vs_profile,
        max_a_v_d=np.column_stack([depth_bound] * 4),
        max_strain_stress=np.column_stack([depth_midpoint] * 3),
        motion_name='test_motion',
        output_dir=str(tmp_path / 'sim_results'),
    )
    if output_dir is None:  # the `output_dir` of the object
        fig_dir = tmp_path / 'sim_results'
    else:
        fig_dir = tmp_path / output_dir
        output_dir = str(fig_dir)

    figs, _ = sim_results.plot(save_fig=True, output_dir=output_dir)

    # Both figures (the motions and the max. profiles) are in `fig_dir`, and
    # there are no other files
    assert figs[1] is not None
    assert sorted(tmp_path.rglob('*')) == [
        fig_dir,
        fig_dir / 'test_motion_ground_motions.png',
        fig_dir / 'test_motion_max_profiles.png',
    ]

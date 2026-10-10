import re
from pathlib import Path

import numpy as np
import pytest

from PySeismoSoil.class_batch_simulation import Batch_Simulation
from PySeismoSoil.class_curves import Multiple_GGmax_Damping_Curves
from PySeismoSoil.class_ground_motion import Ground_Motion
from PySeismoSoil.class_parameters import HH_Param_Multi_Layer
from PySeismoSoil.class_simulation import (
    Equiv_Linear_Simulation,
    Linear_Simulation,
    Nonlinear_Simulation,
)
from PySeismoSoil.class_Vs_profile import Vs_Profile

f_dir = Path(__file__).resolve().parent / 'files'


def test_init__case_1_not_a_list() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape('`list_of_simulations` should be a list.'),
    ):
        Batch_Simulation(1.4)


def test_init__case_2_a_list_of_0_length() -> None:
    with pytest.raises(ValueError, match='should have at least one element'):
        Batch_Simulation([])


def test_init__case_3_wrong_type() -> None:
    msg = 'Elements of `list_of_simulations` should be of type'
    with pytest.raises(TypeError, match=msg):
        Batch_Simulation([1, 2, 3])


def test_init__case_4_inhomogeneous_element_type() -> None:
    gm = Ground_Motion(str(f_dir / 'sample_accel.txt'), unit='gal')
    prof = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    mgdc = Multiple_GGmax_Damping_Curves(data=str(f_dir / 'curve_FKSH14.txt'))
    lin_sim = Linear_Simulation(prof, gm)
    equiv_sim = Equiv_Linear_Simulation(prof, gm, mgdc)
    with pytest.raises(TypeError, match='should be of the same type'):
        Batch_Simulation([lin_sim, equiv_sim])


def test_linear() -> None:
    gm = Ground_Motion(str(f_dir / 'sample_accel.txt'), unit='gal')
    prof_1 = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    prof_2 = Vs_Profile(str(f_dir / 'profile_P001.txt'))

    sim_1 = Linear_Simulation(prof_1, gm, boundary='elastic')
    sim_2 = Linear_Simulation(prof_2, gm, boundary='elastic')
    sim_list = [sim_1, sim_2]

    batch_sim = Batch_Simulation(sim_list)
    options = {'show_fig': False, 'save_txt': False, 'verbose': True}
    non_par_results = batch_sim.run(parallel=False, n_cores=2, options=options)
    par_results = batch_sim.run(parallel=True, options=options)

    accel_out_1_non_par = non_par_results[1].accel_on_surface.accel
    accel_out_1_par = par_results[1].accel_on_surface.accel

    assert np.allclose(
        accel_out_1_non_par, accel_out_1_par, atol=0.0, rtol=1e-3
    )


def test_equiv_linear() -> None:
    gm_raw = Ground_Motion(str(f_dir / 'sample_accel.txt'), unit='gal')

    # Make a very weak motion to speed up equivalent linear calculation
    gm = gm_raw.scale_motion(target_PGA_in_g=0.001)

    prof_2 = Vs_Profile(str(f_dir / 'profile_P001.txt'))
    mgdc_2 = Multiple_GGmax_Damping_Curves(data=str(f_dir / 'curve_P001.txt'))

    sim_2 = Equiv_Linear_Simulation(prof_2, gm, mgdc_2, boundary='elastic')
    sim_list = [sim_2]

    batch_sim = Batch_Simulation(sim_list)
    options = {'show_fig': False, 'save_txt': False, 'verbose': True}
    non_par_results = batch_sim.run(parallel=False, options=options)
    par_results = batch_sim.run(parallel=True, n_cores=2, options=options)

    accel_out_0_non_par = non_par_results[0].accel_on_surface.accel
    accel_out_0_par = par_results[0].accel_on_surface.accel

    assert np.allclose(
        accel_out_0_non_par, accel_out_0_par, atol=0.0, rtol=1e-3
    )


def test_nonlinear() -> None:
    accel_data = np.genfromtxt(f_dir / 'sample_accel.txt')
    accel_downsample = accel_data[::50]  # for faster testing speed
    gm = Ground_Motion(accel_downsample, unit='gal')
    prof = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    hh_g = HH_Param_Multi_Layer(str(f_dir / 'HH_G_FKSH14.txt'))
    hh_x = HH_Param_Multi_Layer(str(f_dir / 'HH_X_FKSH14.txt'))
    sim = Nonlinear_Simulation(prof, gm, G_param=hh_g, xi_param=hh_x)

    batch_sim = Batch_Simulation([sim])
    options = {
        'show_fig': False,
        'save_txt': False,
        'remove_sim_dir': True,
    }

    non_par_results = batch_sim.run(parallel=False, options=options)
    par_results = batch_sim.run(parallel=True, n_cores=2, options=options)

    accel_out_0_non_par = non_par_results[0].accel_on_surface.accel
    accel_out_0_par = par_results[0].accel_on_surface.accel

    assert np.allclose(
        accel_out_0_non_par, accel_out_0_par, atol=0.0, rtol=1e-3
    )

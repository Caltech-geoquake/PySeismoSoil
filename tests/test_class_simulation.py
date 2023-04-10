import unittest
import numpy as np

from PySeismoSoil.class_simulation import (
    Linear_Simulation,
    Equiv_Linear_Simulation,
    Nonlinear_Simulation,
)
from PySeismoSoil.class_ground_motion import Ground_Motion
from PySeismoSoil.class_Vs_profile import Vs_Profile
from PySeismoSoil.class_parameters import HH_Param_Multi_Layer
from PySeismoSoil.class_curves import Multiple_GGmax_Damping_Curves

from test_class_ground_motion import Test_Class_Ground_Motion

import os
from os.path import join as _join


f_dir = _join(os.path.dirname(os.path.realpath(__file__)), 'files')


class Test_Class_Simulation(unittest.TestCase):
    def test_linear(self):
        input_motion = Ground_Motion(_join(f_dir, 'sample_accel.txt'), unit='m')
        soil_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        ls = Linear_Simulation(soil_profile, input_motion)
        sim_result = ls.run(every_layer=True, show_fig=True)
        output = sim_result.accel_on_surface

        self.assertEqual(output.accel.shape, input_motion.accel.shape)
        self.assertEqual(output.dt, input_motion.dt)
        self.assertEqual(output.npts, input_motion.npts)

        sim_result_ = ls.run(every_layer=False, show_fig=False)
        output_ = sim_result_.accel_on_surface

        # Check that two algorithms produce nearly identical results
        nearly_identical = Test_Class_Ground_Motion.nearly_identical
        self.assertTrue(nearly_identical(output.accel, output_.accel, thres=0.99))

    def test_equiv_linear(self):
        soil_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        input_motion = Ground_Motion(_join(f_dir, 'sample_accel.txt'), unit='gal')
        curves = Multiple_GGmax_Damping_Curves(data=_join(f_dir, 'curve_FKSH14.txt'))
        equiv_lin_sim = Equiv_Linear_Simulation(
            soil_profile, input_motion, curves, boundary='elastic',
        )
        output = equiv_lin_sim.run(show_fig=True)
        max_v = output.max_a_v_d[:, 2]
        max_v_benchmark = [
            0.404085, 0.403931, 0.402998, 0.399842, 0.390005,
            0.389336, 0.388176, 0.386563, 0.384449, 0.381578,
            0.377044, 0.371955, 0.366454, 0.36505, 0.363536,
            0.361912, 0.360214, 0.358405, 0.356464, 0.354368,
            0.352104, 0.349671, 0.347186, 0.344585, 0.342999,
            0.344762, 0.345623, 0.346219, 0.34607, 0.344739,
            0.342724, 0.339295, 0.334776, 0.32954, 0.324481,
            0.319054, 0.317311, 0.316842, 0.318159, 0.319668,
            0.321421, 0.323498, 0.325626, 0.326861, 0.328234,
            0.328466, 0.327704, 0.326466, 0.323216, 0.322324,
            0.320209, 0.316914, 0.312529, 0.308906, 0.304708,
            0.300202, 0.295505, 0.29226, 0.289536, 0.287653,
            0.287429, 0.290265, 0.292502,  # from MATLAB SeismoSoil
        ]
        tol = 0.01  # FFT of scipy and MATLAB are different, hence a lenient tolerance
        self.assertTrue(np.allclose(max_v, max_v_benchmark, rtol=tol, atol=0.0))

    def test_nonlinear_init(self):
        input_motion = Ground_Motion(_join(f_dir, 'sample_accel.txt'), unit='m')
        soil_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        HH_G = HH_Param_Multi_Layer(_join(f_dir, 'HH_G_FKSH14.txt'))
        HH_x = HH_Param_Multi_Layer(_join(f_dir, 'HH_X_FKSH14.txt'))

        # this should succeed
        Nonlinear_Simulation(
            soil_profile,
            input_motion,
            G_param=HH_G,
            xi_param=HH_x,
            boundary='elastic',
        )

        # this should fail with ValueError
        HH_G_data = HH_G.param_list
        HH_x_data = HH_x.param_list
        HH_G_ = HH_Param_Multi_Layer(HH_G_data[:-1])  # exclude one layer
        HH_x_ = HH_Param_Multi_Layer(HH_x_data[:-1])  # exclude one layer
        with self.assertRaisesRegex(ValueError, 'Not enough sets of parameters'):
            Nonlinear_Simulation(
                soil_profile,
                input_motion,
                G_param=HH_G_,
                xi_param=HH_x,
            )

        with self.assertRaisesRegex(ValueError, 'Not enough sets of parameters'):
            Nonlinear_Simulation(
                soil_profile,
                input_motion,
                G_param=HH_G,
                xi_param=HH_x_,
            )


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Simulation)
    unittest.TextTestRunner(verbosity=2).run(SUITE)

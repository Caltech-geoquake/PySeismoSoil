# Author: Jian Shi

import unittest

from PySeismoSoil.class_simulation import Linear_Simulation, Nonlinear_Simulation
from PySeismoSoil.class_ground_motion import Ground_Motion
from PySeismoSoil.class_Vs_profile import Vs_Profile
from PySeismoSoil.class_parameters import HH_Param_Multi_Layer

class Test_Class_Simulation(unittest.TestCase):
    def test_linear(self):
        input_motion = Ground_Motion('./files/sample_accel.txt', unit='m')
        soil_profile = Vs_Profile('./files/profile_FKSH14.txt')
        ls = Linear_Simulation(soil_profile, input_motion)
        output = ls.run(show_fig=True)

        self.assertEqual(output.accel.shape, input_motion.accel.shape)
        self.assertEqual(output.dt, input_motion.dt)
        self.assertEqual(output.npts, input_motion.npts)

    def test_nonlinear_init(self):
        input_motion = Ground_Motion('./files/sample_accel.txt', unit='m')
        soil_profile = Vs_Profile('./files/profile_FKSH14.txt')
        HH_G = HH_Param_Multi_Layer('./files/HH_G_FKSH14.txt')
        HH_x = HH_Param_Multi_Layer('./files/HH_X_FKSH14.txt')

        # this should succeed
        Nonlinear_Simulation(soil_profile, input_motion, G_param=HH_G,
                             xi_param=HH_x, boundary='elastic')

        # this should fail with ValueError
        HH_G_data = HH_G.param_list
        HH_x_data = HH_x.param_list
        HH_G_ = HH_Param_Multi_Layer(HH_G_data[:-1])  # exclude one layer
        HH_x_ = HH_Param_Multi_Layer(HH_x_data[:-1])  # exclude one layer
        with self.assertRaisesRegex(ValueError, 'Not enough sets of parameters'):
            Nonlinear_Simulation(soil_profile, input_motion, G_param=HH_G_,
                                 xi_param=HH_x)
        with self.assertRaisesRegex(ValueError, 'Not enough sets of parameters'):
            Nonlinear_Simulation(soil_profile, input_motion, G_param=HH_G,
                                 xi_param=HH_x_)

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Simulation)
    unittest.TextTestRunner(verbosity=2).run(SUITE)

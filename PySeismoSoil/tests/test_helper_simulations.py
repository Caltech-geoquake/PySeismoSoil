# Author: Jian Shi

import unittest
import numpy as np

import PySeismoSoil.helper_simulations as sim

from PySeismoSoil.class_Vs_profile import Vs_Profile
from PySeismoSoil.class_parameters import HH_Param_Multi_Layer
from PySeismoSoil.class_curves import Multiple_GGmax_Damping_Curves

class Test_Helper_Simulations(unittest.TestCase):
    def test_check_layer_count(self):
        # Case 1(a): normal case, with parameters
        vs_profile = Vs_Profile('./files/profile_FKSH14.txt')
        HH_G = HH_Param_Multi_Layer('./files/HH_G_FKSH14.txt')
        HH_x = HH_Param_Multi_Layer('./files/HH_X_FKSH14.txt')
        sim.check_layer_count(vs_profile, G_param=HH_G, xi_param=HH_x)

        # Case 1(b): normal case, with curves
        curves_data = np.genfromtxt('./files/curve_FKSH14.txt')
        mgdc = Multiple_GGmax_Damping_Curves(data=curves_data)
        sim.check_layer_count(vs_profile, GGmax_and_damping_curves=mgdc)

        # Case 2(a): abnormal case, with parameters
        del HH_G[-1]
        with self.assertRaisesRegex(ValueError, 'Not enough sets of parameters'):
            sim.check_layer_count(vs_profile, G_param=HH_G)

        # Case 2(b): abnormal case, with curves
        curves_data_ = curves_data[:, :-4]
        with self.assertRaisesRegex(ValueError, 'Not enough sets of curves'):
            mgdc_ = Multiple_GGmax_Damping_Curves(data=curves_data_)
            sim.check_layer_count(vs_profile, GGmax_and_damping_curves=mgdc_)

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Helper_Simulations)
    unittest.TextTestRunner(verbosity=2).run(SUITE)

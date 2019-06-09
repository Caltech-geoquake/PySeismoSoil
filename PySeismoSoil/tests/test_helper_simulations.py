# Author: Jian Shi

import unittest
import numpy as np

import PySeismoSoil.helper_simulations as sim
import PySeismoSoil.helper_site_response as sr

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

    def test_linear(self):
        '''
        Test that ``helper_simulations.linear()`` produces identical results
        to ``helper_site_response.linear_site_resp()``.
        '''
        vs_profile = np.genfromtxt('./files/profile_FKSH14.txt')
        accel_in = np.genfromtxt('./files/sample_accel.txt')

        #---------- Elastic boundary ------------------------------------------
        result_1 = sim.linear(vs_profile, accel_in, boundary='elastic')[3]
        result_1_ = sr.linear_site_resp(vs_profile, accel_in, boundary='elastic')

        # Time arrays need to match well
        self.assertTrue(np.allclose(result_1[:, 0], result_1_[:, 0], rtol=0.0001,
		                            atol=0.0))

        # Only check correlation (more lenient). Because `sim.linear()`
        # re-discretizes soil profiles into finer layers, so numerical errors
        # may accumulate over the additional layers.
        r_1 = np.corrcoef(result_1[:, 1], result_1_[:, 1])
        self.assertTrue(r_1[0, 1] >= 0.99)

        #---------- Rigid boundary --------------------------------------------
        result_2 = sim.linear(vs_profile, accel_in, boundary='rigid')[3]
        result_2_ = sr.linear_site_resp(vs_profile, accel_in, boundary='rigid')
        self.assertTrue(np.allclose(result_2[:, 0], result_2_[:, 0], rtol=0.0001,
		                            atol=0.0))
        r_2 = np.corrcoef(result_2[:, 1], result_2_[:, 1])
        self.assertTrue(r_2[0, 1] >= 0.97)  # rigid cases can lead to higher errors

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(result_1[:, 0], result_1[:, 1], label='every layer', alpha=0.6)
        plt.plot(result_1_[:, 0], result_1_[:, 1], label='surface only', alpha=0.6)
        plt.legend()
        plt.xlabel('Time [sec]')
        plt.ylabel('Acceleration')
        plt.grid(ls=':', lw=0.5)
        plt.title('Elastic boundary')

        plt.subplot(122)
        plt.plot(result_2[:, 0], result_2[:, 1], label='every layer', alpha=0.6)
        plt.plot(result_2_[:, 0], result_2_[:, 1], label='surface only', alpha=0.6)
        plt.legend()
        plt.xlabel('Time [sec]')
        plt.grid(ls=':', lw=0.5)
        plt.title('Rigid boundary')

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Helper_Simulations)
    unittest.TextTestRunner(verbosity=2).run(SUITE)

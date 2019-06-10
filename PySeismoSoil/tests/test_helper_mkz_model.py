# Author: Jian Shi

import unittest
import numpy as np
from scipy import stats

import PySeismoSoil.helper_mkz_model as mkz
import PySeismoSoil.helper_site_response as sr

class Test_Helper_MKZ_Model(unittest.TestCase):
    '''
    Unit tests for helper functions in helper_mkz_model.py
    '''

    def __init__(self, methodName='runTest'):
        self.strain = np.logspace(-2, 1, num=12)
        self.atol = 1e-4
        self.param = {'gamma_ref': 0.1, 's': 0.2, 'beta': 0.3, 'Gmax': 0.4}
        self.array = np.array([1, 2, 3, 4]) / 10.0
        super(Test_Helper_MKZ_Model, self).__init__(methodName=methodName)

    def test_tau_MKZ(self):
        T = mkz.tau_MKZ(self.strain, gamma_ref=1, beta=2, s=3, Gmax=4)
        # note: benchmark results come from comparable functions in MATLAB
        self.assertTrue(np.allclose(T, [0.0400, 0.0750, 0.1404, 0.2630, 0.4913,
                                        0.9018, 1.4898, 1.5694, 0.7578, 0.2413,
                                        0.0700, 0.0200], atol=self.atol, rtol=0.0))

    def test_calc_damping_from_param(self):
        xi = sr.calc_damping_from_param(self.param, self.strain, mkz.tau_MKZ)
        self.assertTrue(np.allclose(xi, [0, 0.0072, 0.0101, 0.0119, 0.0133,
                                         0.0147, 0.0163, 0.0178, 0.0195, 0.0213,
                                         0.0232, 0.0251], atol=self.atol, rtol=0.0))

    def test_serialize_params_to_array(self):
        array = mkz.serialize_params_to_array(self.param)
        self.assertTrue(np.allclose(array, self.array))

        # Test error with incorrect number of dict items
        with self.assertRaises(AssertionError, msg=''):
            mkz.serialize_params_to_array({'test': 2})

        # Test error with incorrect dict keys: only one key name is wrong
        with self.assertRaises(KeyError, msg=''):
            mkz.serialize_params_to_array({'gamma_ref': 1,
                                           's': 1,
                                           'beta': 1,
                                           'Gmax__': 1})

    def test_deserialize_array_to_params(self):
        param = mkz.deserialize_array_to_params(self.array)
        self.assertEqual(param, self.param)

        # Test error with input data type
        with self.assertRaises(TypeError, msg='must be a 1D numpy array'):
            mkz.deserialize_array_to_params([1,2,3,4])

        # Test error with incorrecto number of parameters (should be 4)
        with self.assertRaises(AssertionError, msg=''):
            mkz.deserialize_array_to_params(np.array([1,2,3,4,5]))

    def test_fit_MKZ(self):
        strain_in_1 = np.geomspace(1e-6, 0.1, num=50)  # unit: 1
        strain_in_pct = strain_in_1 * 100

        param_1 = dict(gamma_ref=0.0035, beta=0.85, s=1.0, Gmax=1e6)
        T_MKZ_1 = mkz.tau_MKZ(strain_in_1, **param_1)
        GGmax_1 = sr.calc_GGmax_from_stress_strain(strain_in_1, T_MKZ_1)

        param_2 = dict(gamma_ref=0.02, beta=1.4, s=0.7, Gmax=2e7)
        T_MKZ_2 = mkz.tau_MKZ(strain_in_1, **param_2)
        GGmax_2 = sr.calc_GGmax_from_stress_strain(strain_in_1, T_MKZ_2)

        damping = np.ones_like(strain_in_pct)  # dummy values
        curve_data \
            = np.column_stack((strain_in_pct, GGmax_1, strain_in_pct, damping,
                               strain_in_pct, GGmax_2, strain_in_pct, damping))
        param, fitted_curve = mkz.fit_MKZ(curve_data, show_fig=True)

        # Make sure that the R^2 score between data and fit >= 0.99
        GGmax_fitted_1 = np.interp(strain_in_pct, fitted_curve[:, 0],
                                   fitted_curve[:, 1])
        GGmax_fitted_2 = np.interp(strain_in_pct, fitted_curve[:, 4],
                                   fitted_curve[:, 5])
        r2_1 = stats.linregress(GGmax_1, GGmax_fitted_1)[2]
        r2_2 = stats.linregress(GGmax_2, GGmax_fitted_2)[2]
        self.assertGreaterEqual(r2_1, 0.99)
        self.assertGreaterEqual(r2_2, 0.99)

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Helper_MKZ_Model)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
